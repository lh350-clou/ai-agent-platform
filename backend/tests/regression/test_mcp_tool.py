"""MCP 客户端的回归：真拉起 Server 子进程，真发现工具、真调用。

为什么不能用假的：MCP 这条路的价值全在「跨进程 + 按协议通信」上，
而协议版本、传输层、参数 schema 恰好是最容易在升级里悄悄坏掉的地方
（requirements.txt 里已经写明 mcp 是 2.x，1.x 的示例代码在它上面跑不起来）。
把子进程换成假实现，等于把要验的东西本身换掉了。

这里不经过模型 —— Agent 那侧怎么用这些工具由 test_agent_loop.py 覆盖。
本文件只关心 Client 与 Server 之间的契约：工具叫什么、参数长什么样、
出错时返回的是「可读的错误」还是「协议层失败」。
"""

import json
from datetime import datetime

import pytest

from app.services import mcp_client

# 只有 regression 是模块级的：本文件里既有要起子进程的用例（integration），
# 也有纯函数的用例（unit），标在每条用例上更准确 ——
# 一条用例同时挂两个「怎么跑」的标记，会让 -m unit / -m integration 两个
# 筛选结果加起来比总数多，看到的人只会以为标记写错了。
pytestmark = [pytest.mark.regression]

# Server 里定义的工具名与允许的时区，写死在这里而不是 import
# app.mcp_server.server 的常量：那等于把 Server 的实现细节当成本文件的依据，
# 而「Server 换了实现」恰恰是 MCP 这层要能扛住的事。
TIME_TOOL = "get_current_time"
ALLOWED_TIMEZONE = "Asia/Shanghai"

# 每条用例自己开一个会话，【不用 fixture 持有】。
#
# 试过把它做成 async fixture，结果是每条用例都在 teardown 报
# 「Attempted to exit cancel scope in a different task than it was entered in」——
# MCP Client 内部用 anyio 的 cancel scope，而它要求在【同一个 task】里进入和退出，
# pytest 的 fixture 建立与销毁并不保证这一点。
# 写在用例体内则是同一个协程的 async with，进出天然同 task：
# 顺带也更贴近生产用法 —— run_agent 每次请求也是自己开一个会话。


@pytest.mark.integration
async def test_server_exposes_time_tool_with_schema() -> None:
    """工具名和参数 schema 必须还在。

    工具名是模型和白名单共同依赖的契约，改名字的后果不是报错，
    而是「模型请求了工具但执行时匹配不到」，表现为 Agent 从来不调工具。
    """
    async with mcp_client.open_session() as session:
        tools = await mcp_client.list_tools(session)
    by_name = {tool["name"]: tool for tool in tools}

    assert TIME_TOOL in by_name, f"MCP Server 暴露的工具变了：{sorted(by_name)}"

    tool = by_name[TIME_TOOL]
    assert tool["description"], "工具描述为空会让模型无从判断该不该调用它"
    # input_schema 会被原样当作 OpenAI 的 parameters 用（见 to_openai_tool），
    # 所以它必须是一份带 properties 的 JSON Schema，不能是别的东西。
    assert "timezone" in tool["input_schema"].get("properties", {})


@pytest.mark.integration
async def test_call_returns_time_for_allowed_timezone() -> None:
    """正常调用：拿回一段能被 json.loads 解析的结果。"""
    async with mcp_client.open_session() as session:
        content, error = await mcp_client.call_tool(
            session, TIME_TOOL, {"timezone": ALLOWED_TIMEZONE}
        )

    assert error is None
    payload = json.loads(content)
    assert payload["timezone"] == ALLOWED_TIMEZONE
    assert payload["utc_offset"] == "+0800"
    # 时间必须真的能被解析成 datetime —— 只说「非空字符串」的话，
    # 返回一个 "current time" 之类的占位文本也能通过。
    datetime.fromisoformat(payload["datetime"])


@pytest.mark.integration
async def test_invalid_timezone_returns_readable_error() -> None:
    """非法参数：工具自己返回可读错误，而不是让协议层失败。

    两种处理的差别很大：
        返回错误文本 —— 模型看得到「不支持的时区，可选值是这些」，能自我纠正；
        抛异常      —— MCP 层只会回一句 "Error executing tool"，
                       模型既不知道错在哪，也不知道该改成什么。
    顺带守住「路径类输入不会出事」：时区名是外部输入，历史上 ZoneInfo
    被证明可以借它读文件，所以 Server 用白名单挡着，这里拿一个路径样式的
    输入确认它被挡在白名单外。
    """
    async with mcp_client.open_session() as session:
        content, error = await mcp_client.call_tool(
            session, TIME_TOOL, {"timezone": "../../etc/passwd"}
        )

    assert error is None, "工具级的参数错误不该被记成调用失败"
    payload = json.loads(content)
    assert payload["error"] == "unsupported timezone"
    # 错误文本里要给出可选值，模型才有自我纠正的依据。
    assert ALLOWED_TIMEZONE in payload["supported"]
    # 不回显用户传进来的那个字符串。
    assert "../../etc/passwd" not in content


@pytest.mark.unit
def test_tool_definition_uses_model_facing_prefix() -> None:
    """暴露给模型的名字带 mcp_ 前缀，回给 Server 的名字不带。

    两次转换必须是互逆的。转错的后果是「Agent 调用了一个 Server 不认识的
    工具名」，报错信息是 Client 侧的，看不出是名字转换出了问题。
    """
    tool = mcp_client.to_openai_tool(
        {
            "name": TIME_TOOL,
            "description": "查询时间",
            "input_schema": {"type": "object", "properties": {"timezone": {"type": "string"}}},
        }
    )

    assert tool["type"] == "function"
    assert tool["function"]["name"] == f"{mcp_client.MCP_TOOL_PREFIX}{TIME_TOOL}"
    # parameters 直接沿用 Server 给的 schema，不做翻译。
    assert tool["function"]["parameters"]["properties"]["timezone"] == {"type": "string"}
    assert (
        mcp_client.strip_prefix(tool["function"]["name"]) == TIME_TOOL
    ), "前缀加上去之后必须能原样还原回 Server 里的工具名"
