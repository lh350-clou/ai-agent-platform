"""FastAPI 应用本体与 /health 契约的回归。

`/health` 是部署侧唯一的探活入口（docker-compose 的 healthcheck、
负载均衡、k8s 探针都打它），所以它有两条必须一直成立的契约：

    1. 字段形态稳定：status / app_name / version / database 一个都不能少；
    2. **数据库连不上时依然返回 200**，只把 database 标成 error ——
       这样探活方能区分「进程没起来」和「进程起来了但数据库连不上」。
       一旦有人为了「让探活更严格」把它改成 500，容器会被反复重启，
       而根因看起来完全在别处。

第 2 条正是「必须有一条不依赖数据库的用例来守」的典型：数据库挂了的时候，
恰恰是这条契约最需要被验证的时候，而此时任何要连库的用例都已经跑不了了。
"""

import logging

import httpx
import pytest

from app.api import health as health_api
from app.core.config import settings
from app.main import app
from tests.regression.availability import ServiceStatus

pytestmark = [pytest.mark.regression]


def _client() -> httpx.AsyncClient:
    """把客户端直接挂在 ASGI 应用上：不起 uvicorn、不占端口、不依赖启动脚本。

    刻意不走 lifespan：ASGI 传输层不支持 lifespan 事件，而它只做一次
    「探一次数据库并记日志」，不是本文件要验的东西 —— 数据库的真实状态
    由下面几条用例分别断言。
    """
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://regression.local",
    )


@pytest.mark.unit
async def test_health_contract_fields() -> None:
    """无论数据库什么状态，字段形态都必须是稳定的。

    这里【只断言 database 的取值范围】，不断言它的值：这条用例要守的是
    「接口形态」，数据库真可用时返回 ok 由下一条用例负责。
    把两种断言混在一起写，数据库一挂这条用例就会红，
    而它本来正是那个「数据库挂了也要能跑」的用例。
    """
    async with _client() as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_name"] == settings.APP_NAME
    assert body["version"] == settings.APP_VERSION
    assert body["database"] in {"ok", "error"}


@pytest.mark.unit
async def test_health_returns_200_when_database_is_down(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """数据库连不上：仍然 200 + database=error，且错误细节不进响应体。

    把 check_connection 换成会抛异常的假实现，而不是真去关掉数据库：
    要验的是「接口在数据库出错时怎么表现」，用假故障就能验，
    而真去停容器会让这条用例依赖一个破坏性的手工步骤，跑不了第二遍。
    """
    # 这条路径会走 logger.exception 打一整段堆栈，测试输出会很吵。
    caplog.set_level(logging.CRITICAL)

    async def failing_check() -> None:
        raise RuntimeError("模拟数据库不可用：连接被拒绝")

    # 补丁打在 api/health.py 这个模块上，而不是 core/database.py ——
    # 它用的是 `from ... import check_connection`，名字已经绑定在那边了。
    monkeypatch.setattr(health_api, "check_connection", failing_check)

    async with _client() as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["database"] == "error"
    assert body["status"] == "ok"
    # 异常原文（可能带主机名、端口、认证细节）绝不能出现在响应里。
    assert "模拟数据库不可用" not in response.text


@pytest.mark.integration
async def test_health_reports_database_ok(services: ServiceStatus) -> None:
    """数据库真的可用时，database 必须是 ok。

    这条是上面那条的对照：没有它，「永远返回 error」的实现也能让
    test_health_returns_200_when_database_is_down 通过。
    """
    services.require_postgres()

    async with _client() as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["database"] == "ok"
