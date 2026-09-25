"""MCP 客户端：连接 MCP Server、发现工具、调用工具。

Agent 只跟这个模块打交道，不 import mcp_server 里的任何东西 ——
那是「绕过 MCP 直接调函数」，等于把 MCP 这层协议白加了：
既享受不到进程隔离，也失去了「换一个 MCP Server 就能换一套工具」的意义。

关于传输方式：用 stdio，也就是由客户端负责把 Server 作为一个子进程拉起来，
通过标准输入输出通信。好处是不需要事先启动服务、不占端口、不受网络配置影响；
代价是每次会话都要付一次进程启动的开销 —— 所以 Agent 在一次请求里
只开一个会话，而不是每次工具调用都开一个。
"""

import json
import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from openai.types.chat import ChatCompletionToolParam

from app.core.config import BASE_DIR, settings

logger = logging.getLogger(__name__)

# 暴露给模型的 MCP 工具名前缀。
#
# 加前缀是为了让「工具来自哪里」在名字上就一目了然，也为了避免和内部工具
# 撞名：现在只有一个 get_current_time，但接第二个 MCP Server 时，
# 撞名几乎是必然的。等到那时候再改，就要动模型见过的名字，
# 而现在加前缀几乎是零成本。
MCP_TOOL_PREFIX = "mcp_"

# MCP Server 的工作目录。
# 用 `python -m app.mcp_server.server` 启动时，Python 需要能 import 到 app 包，
# 所以工作目录必须是 backend/（app 包的父目录）。
BACKEND_DIR = BASE_DIR / "backend"


def build_server_params() -> StdioServerParameters:
    """构造启动 MCP Server 所需的参数。

    解释器固定用 sys.executable：MCP Server 跑在哪个 Python 上，
    必须和当前进程是同一个，否则会出现「命令能跑但第三方包不在」这种
    极难排查的问题（尤其是机器上有多个 Python 环境时）。
    所以配置里只暴露「跑哪个模块」，解释器交给代码补全。
    """
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", settings.MCP_SERVER_MODULE],
        cwd=str(BACKEND_DIR),
    )


@asynccontextmanager
async def open_session() -> AsyncGenerator[Client, None]:
    """打开一个到 MCP Server 的会话。

    用法是「一次 Agent 请求开一个会话」，而不是每次工具调用都开：

        async with mcp_client.open_session() as session:
            tools = await mcp_client.list_tools(session)
            ...
            await mcp_client.call_tool(session, name, args)

    每次调用都开关一次的话，每次都要重新拉起一个 Python 进程（几百毫秒起步），
    一次 Agent 请求下来光启动开销就够呛。

    连不上时【直接抛异常】，由调用方决定怎么办 ——
    这里不做「悄悄降级成没有工具」的决定，那是策略，属于上层。
    """
    async with Client(build_server_params()) as client:
        logger.info("已连接 MCP Server：%s", settings.MCP_SERVER_MODULE)
        yield client


async def list_tools(client: Client) -> list[dict[str, Any]]:
    """发现 MCP Server 暴露的工具，返回中立的描述结构。

    刻意不把 SDK 的 Tool 对象直接往上抛：让 Agent 依赖 MCP SDK 的类型，
    等于把「我们用的是哪个 MCP 实现」焊进了 Agent。转成普通 dict 之后，
    将来换 SDK 版本或换传输方式，Agent 那侧不用动。

    返回的每一项：
        {"name": "get_current_time", "description": "...", "input_schema": {...}}
        name 是【原始名】（不带前缀），前缀在暴露给模型时才加。
    """
    result = await client.list_tools()
    tools: list[dict[str, Any]] = []
    for tool in result.tools:
        tools.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema,
            }
        )
    logger.info("MCP Server 暴露了 %d 个工具：%s", len(tools), [t["name"] for t in tools])
    return tools


def to_openai_tool(tool: dict[str, Any]) -> ChatCompletionToolParam:
    """把 MCP 工具转成 OpenAI 的工具定义，供模型选择。

    参数结构直接沿用 MCP 给的 input_schema —— 它本来就是 JSON Schema，
    而 OpenAI 的 parameters 也是 JSON Schema，两者是同一套东西，不需要翻译。
    手写一份映射反而会引入「两边定义不一致」的风险
    （工具描述改了但映射忘了改，模型就会按旧的参数格式调用）。
    """
    return {
        "type": "function",
        "function": {
            "name": f"{MCP_TOOL_PREFIX}{tool['name']}",
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }


def strip_prefix(model_facing_name: str) -> str:
    """把模型看到的带前缀名字还原成 MCP Server 里的原始工具名。"""
    return model_facing_name.removeprefix(MCP_TOOL_PREFIX)


async def call_tool(
    client: Client, name: str, arguments: dict[str, Any]
) -> tuple[str, str | None]:
    """调用一个 MCP 工具，返回 (要交给模型的文本, 失败原因)。

    **这个函数不抛异常**：任何失败都会变成一段 JSON 错误说明返回给模型。
    理由和内部工具一致 —— 让模型看到「这次没成功」，它还有机会换个方式重试
    或基于已有信息作答；而让异常穿出去，整个 Agent 请求就废了。

    失败原因单独作为第二个返回值给出，而不是让调用方去解析那段 JSON：
    错误文本是【给模型看的】，格式服务于模型的可读性；上层（Agent 的 Trace）
    要知道的是「这次成没成功」，两件事不该挤在同一个字符串里。
    反解析还有个实际风险 —— Server 正常返回的内容里也可能出现 error 字段，
    那样会把一次成功的调用误判成失败。

    参数：
        name:      MCP Server 里的【原始】工具名（不带前缀）。
        arguments: 工具参数。

    返回：
        (模型可读的文本, 失败原因)。成功时第二个元素是 None；
        失败时是形如 {"error": "..."} 的 JSON 加一句简短的原因说明。
    """
    try:
        result = await client.call_tool(
            name,
            arguments,
            # 没有超时的话，一个卡死的 Server 会把整个请求吊住。
            read_timeout_seconds=settings.MCP_TOOL_TIMEOUT_SECONDS,
        )
    except Exception:
        # 连接断了、Server 崩了、超时了都会走到这里。
        # 详细堆栈只进日志 —— 它可能带着本机路径、解释器位置这些内部信息，
        # 而这段文本是要进模型上下文、有可能被复述给用户的。
        logger.exception("调用 MCP 工具失败：%s", name)
        return json.dumps({"error": "mcp tool call failed"}, ensure_ascii=False), "mcp tool call failed"

    if result.is_error:
        # Server 自己报了错。刻意【原样透传 Server 的错误文本】之前的那层判断：
        # MCP 的错误信息虽然通常不含堆栈，但那是 Server 的实现细节，
        # 我们不为它背书。统一换成我们自己的措辞，细节留在日志里。
        logger.warning("MCP 工具返回错误：%s", name)
        return (
            json.dumps({"error": "mcp tool execution failed"}, ensure_ascii=False),
            "mcp tool execution failed",
        )

    # 提取文本内容。MCP 的内容块可以是文本、图片、资源引用等多种类型，
    # 这里只取文本 —— 本项目的工具都是返回 JSON 文本的，
    # 遇到别的类型说明工具用法超出了预期，如实报告而不是硬凑。
    texts = [
        block.text
        for block in result.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    if not texts:
        logger.warning("MCP 工具没有返回文本内容：%s", name)
        return (
            json.dumps({"error": "mcp tool returned no text content"}, ensure_ascii=False),
            "mcp tool returned no text content",
        )

    return "\n".join(texts), None
