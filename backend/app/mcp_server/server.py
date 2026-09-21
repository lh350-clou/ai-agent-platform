"""MCP Server 本体：定义并暴露工具。

启动方式（由 MCP Client 通过 stdio 自动拉起，也可以手动跑）：

    cd backend
    python -m app.mcp_server.server

本模块只依赖标准库 + MCP SDK。刻意不 import 应用的任何东西 ——
它要被 MCP Client 当成一个独立的工具服务来用，
而不是「应用的一块内部代码」。
"""

import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from mcp.server.mcpserver import MCPServer

logger = logging.getLogger(__name__)

SERVER_NAME = "ai-agent-platform-tools"

# 允许查询的时区白名单。
#
# 为什么是白名单而不是「任意合法的 IANA 时区」：zoneinfo.ZoneInfo 接受的是
# 一个键，而这个键在历史上被证明可以被用来读取任意文件（通过绝对路径或
# ../ 之类的构造）。即便当前版本的 Python 已经收紧了这块，把「用户可控的
# 字符串直接喂给时区加载器」这件事本身也是不必要的暴露面。
#
# 需要支持新时区时，往这个集合里加一个名字即可 —— 加的是常量，
# 不是让外部输入去影响加载行为。
ALLOWED_TIMEZONES: frozenset[str] = frozenset({
    "Asia/Shanghai",
    "Asia/Tokyo",
    "UTC",
})

server = MCPServer(name=SERVER_NAME)


@server.tool(
    name="get_current_time",
    description=(
        "查询指定时区的当前时间。"
        "支持 Asia/Shanghai、Asia/Tokyo、UTC。"
    ),
)
def get_current_time(timezone: str) -> str:
    """返回指定时区的当前时间。

    返回值是一段 JSON 文本（MCP 工具的返回值最终会以文本形式传给模型，
    用 JSON 是为了让模型能稳定地解析出字段，而不是去猜一段自然语言里的数字）。

    非法时区【不抛异常】，而是返回一段说明错误和可选值的结果 ——
    抛出去的话，MCP 层只会回给调用方一句 "Error executing tool"，
    模型既不知道错在哪，也不知道该改成什么，只能原样重试一次。
    返回可读的错误反而能让它自我纠正。
    """
    if timezone not in ALLOWED_TIMEZONES:
        # 这里只回显了「我支持哪些」，没有回显用户传进来的那个字符串 ——
        # 少一次原样回显就少一条「把外部输入写进日志/响应」的路径。
        return json.dumps(
            {
                "error": "unsupported timezone",
                "supported": sorted(ALLOWED_TIMEZONES),
            },
            ensure_ascii=False,
        )

    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception:
        # 白名单已经挡在前面，理论上到不了这里。留着是为了兜底：
        # 万一时区数据库损坏或缺失，也只回一句笼统的错误，
        # 绝不把异常的堆栈、文件路径带出去。
        logger.exception("读取时区失败：%s", timezone)
        return json.dumps({"error": "failed to read timezone"}, ensure_ascii=False)

    return json.dumps(
        {
            "timezone": timezone,
            "datetime": now.isoformat(timespec="seconds"),
            # UTC 偏移单独给一份：模型从 ISO 字符串里抠偏移量容易出错，
            # 直接给出 +08:00 / +09:00 这样的形式更省事。
            "utc_offset": now.strftime("%z"),
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    # stdio 传输：Server 从标准输入读请求、往标准输出写响应。
    # 注意这意味着【任何普通的 print 都会破坏协议】——
    # 本模块里一处 print 都没有，日志也只走 logging（默认输出到 stderr）。
    server.run(transport="stdio")
