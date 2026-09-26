"""外部服务探活：回归套件里【唯一】一处判断「这条用例现在能不能跑」的地方。

为什么需要它：一部分回归用例打的是真实服务（PostgreSQL / Milvus / 两个 API Key /
MCP 子进程）。这些服务不在时只有两种处理方式，而两种都有代价：

    直接失败 —— 「今天没开 Docker」看起来像「代码坏了」，
                于是红几次之后所有人都不再看这个套件；
    静默跳过 —— 「一条都没测」和「全部通过」在输出上一模一样，更危险。

这里的做法是：**默认跳过，但跳过必须看得见，并且可以关掉。**
跳过的原因会写进 skip 理由（pytest.ini 里的 `-rs` 保证它被打印出来），
而 `REGRESSION_REQUIRE_SERVICES=1` 会把跳过升级成失败，给 CI 用 ——
CI 上「服务没起」本来就是个必须修的问题，不该被跳过盖过去。
"""

import http.client
import logging
import os
import socket
from urllib.parse import urlparse

import pytest

from app.core.config import settings

logger = logging.getLogger(__name__)

# 「要求外部服务必须可用」的开关。
#
# 为什么用环境变量而不是加进 app.core.config.settings：那是【应用的】配置，
# 生产代码里多出一个只服务于测试运行方式的字段，是让测试的关切渗进业务配置。
# 环境变量本来就是「运行这个进程时的选择」，正是这个语义。
REQUIRE_SERVICES_ENV = "REGRESSION_REQUIRE_SERVICES"

# 本机服务（PostgreSQL / Milvus）的探活超时。取 1 秒：本机服务要么立刻连上，
# 要么就是没起，等更久只会让「跑测试之前先探活」这件事本身变慢。
PROBE_TIMEOUT_SECONDS = 1.0

# 外部 HTTPS 接口的探活超时。给得比本机松：要过 DNS、TCP、TLS 三关，
# 网络正常时也就几百毫秒，而卡住的时候必须能自己放弃。
API_PROBE_TIMEOUT_SECONDS = 5.0


class ServiceStatus:
    """本轮测试内各外部依赖的可用性快照。

    探活在实例化时一次性做完（由 tests/regression/conftest.py 里的
    session 级 fixture 持有），之后所有用例复用 —— 一条连接一探会让
    「服务在不在」这件事本身的耗时超过用例本身。
    """

    def __init__(self) -> None:
        self.postgres: str | None = _probe_tcp(
            settings.POSTGRES_HOST, settings.POSTGRES_PORT
        )
        self.milvus: str | None = _probe_tcp(settings.MILVUS_HOST, settings.MILVUS_PORT)
        # 两个 Key 分开记：RAG 检索只需要 embedding（SiliconFlow），
        # Agent 才需要 DeepSeek。合成一个「LLM 可用」会让
        # 「只缺 DeepSeek」的机器把检索用例也一起跳掉。
        self.siliconflow_key: str | None = _probe_key(
            "SILICONFLOW_API_KEY", settings.SILICONFLOW_API_KEY
        )
        self.deepseek_key: str | None = _probe_key(
            "DEEPSEEK_API_KEY", settings.DEEPSEEK_API_KEY
        )
        # 除了「Key 配了没」，还要探「接口连不连得上」。
        #
        # 这两件事会各自单独发生：Key 配了但网络断了（VPN 掉线、公司网络拦了），
        # 或者网络通了但 Key 没配。只探其中一样，另一种情况下用例照样会红，
        # 而红出来的是一串 ConnectError —— 看到的人只会以为是代码坏了。
        #
        # 代价是每次回归多两个 HTTPS 请求（一次会话只探一轮），
        # 换来的是「网络不通」和「代码坏了」在输出上能被区分开。
        self.siliconflow_api: str | None = _probe_https(settings.SILICONFLOW_BASE_URL)
        self.deepseek_api: str | None = _probe_https(settings.DEEPSEEK_BASE_URL)

    # ---- 各用例按需声明自己依赖什么 ----
    # 每个 require_* 只做一件事：依赖不可用就跳过（严格模式下失败）。

    def require_postgres(self) -> None:
        _require("PostgreSQL", self.postgres)

    def require_milvus(self) -> None:
        _require("Milvus", self.milvus)

    def require_embedding(self) -> None:
        # Key 没配和接口连不上是两种不同的原因，谁先出问题就报谁。
        _require("SiliconFlow Embedding", _first(self.siliconflow_key, self.siliconflow_api))

    def require_llm(self) -> None:
        _require("DeepSeek", _first(self.deepseek_key, self.deepseek_api))


def _probe_tcp(host: str, port: int) -> str | None:
    """能建立 TCP 连接就算「服务在」。返回 None 表示可用，否则返回不可用原因。

    刻意只探端口、不做协议层握手：这一步的目的是把「容器没起」和
    「代码坏了」分开。真去调一次业务函数（比如 check_connection）反而更糟 ——
    那会把「服务在但认证配错了」也算成「服务不在」，
    于是真正的配置问题被跳过盖住，永远没人发现。
    """
    try:
        with socket.create_connection((host, port), timeout=PROBE_TIMEOUT_SECONDS):
            return None
    except OSError as exc:
        return f"{host}:{port} 连不上（{type(exc).__name__}）"


def _probe_key(name: str, value: object) -> str | None:
    """只判断 Key 配没配，绝不读取或打印它的内容。"""
    # SecretStr 是唯一需要取明文的地方，但这里连明文都不需要 ——
    # 长度是否为 0 就够了，也就不存在「取出来顺手打一下」的机会。
    if not getattr(value, "get_secret_value")():
        return f"{name} 未配置（见仓库根目录 .env.example，需写入 .env）"
    return None


def _probe_https(base_url: str) -> str | None:
    """能不能和这个 HTTPS 接口完成一次请求。返回 None 表示通。

    刻意只探「连得上」，不探「密钥有效」：带着假 Key 收到 401 也算通 ——
    这条探测要回答的是「网络和 TLS 这一层通不通」，
    而 401 是业务层的事，由真调用去发现（那时红得明明白白）。

    用标准库的 http.client 而不是 httpx：探活是套件自己的基础设施，
    不该让它再依赖一个 HTTP 客户端库，哪怕那个库本来就在装依赖里。
    """
    parsed = urlparse(base_url)
    host = parsed.netloc
    path = parsed.path or "/"

    # 用 http.client 而不是 urllib：少一层重定向与 opener 的默认行为，
    # 探测这件事越直白越好。
    #
    # 注意它【不是上下文管理器】（写 `with conn:` 会得到
    # TypeError: does not support the context manager protocol），
    # 所以这里显式 close，而不是靠 with 收尾。
    conn: http.client.HTTPConnection = (
        http.client.HTTPSConnection(host, timeout=API_PROBE_TIMEOUT_SECONDS)
        if parsed.scheme == "https"
        else http.client.HTTPConnection(host, timeout=API_PROBE_TIMEOUT_SECONDS)
    )

    try:
        conn.request("GET", path)
        # 必须把响应读出来：不读的话连接不会正常结束，
        # 「探活成功」这件事就没被真正验证过。
        response = conn.getresponse()
        response.read()
        logger.debug("API 探活：%s 返回 HTTP %s", host, response.status)
        return None
    except Exception as exc:
        # 这里必须捕获 Exception 而不是 OSError：TLS 握手失败抛的是
        # ssl.SSLError，它不属于 OSError 家族。漏掉它会让探活自己抛异常，
        # 那就从「跳过并说明原因」变成了「所有用例都 ERROR」。
        return f"{host} 连不上（{type(exc).__name__}）"
    finally:
        conn.close()


def _first(*reasons: str | None) -> str | None:
    """返回第一个「不可用」的原因；全都可用则返回 None。"""
    for reason in reasons:
        if reason is not None:
            return reason
    return None


def _require(service: str, reason: str | None) -> None:
    """依赖可用就返回；不可用则跳过（或按严格模式失败）。"""
    if reason is None:
        return

    if os.environ.get(REQUIRE_SERVICES_ENV) == "1":
        pytest.fail(
            f"{service} 不可用：{reason}。"
            f"当前设置了 {REQUIRE_SERVICES_ENV}=1，要求外部服务必须可用，故记为失败。"
        )

    pytest.skip(
        f"{service} 不可用：{reason} —— 本条集成用例未执行"
        f"（要让这种情况失败而不是跳过，设置 {REQUIRE_SERVICES_ENV}=1）"
    )
