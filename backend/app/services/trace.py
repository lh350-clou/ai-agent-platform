"""轻量级 Agent Trace：记录一次 run_agent 里发生了什么、每一步各花了多久。

为什么不用 OpenTelemetry / Jaeger：
那套方案的重量在「基础设施」—— 要起 Collector、配 exporter、接一个存储后端，
才能看到任何东西。而本项目现在要回答的问题很具体：这一轮回答经过了几次模型调用、
几次工具调用、每一步各花了多久、哪一步失败了。
一个计时器加几个 Pydantic 模型就能覆盖，而且没有任何额外进程要运维。
真到了需要跨服务串联链路的那天再换 OTel 也不亏：调用方拿到的始终是「一条 Trace」，
替换成本被关在本模块内部。

职责边界：本模块只负责「怎么记」，不判断「什么时候该记」，
也不 import agent / llm 里的任何东西 —— 谁调用它，它就把谁记下来。

第一版 Trace 【只在当前请求的生命周期里存在】：随接口响应返回、或写进日志，
不落库。要做持久化的话是在本模块加一个「导出」出口，
而不是让每个记录点自己去找数据库。

安全约定：Trace 会进日志、也可能随接口返回，所以这里【绝不记录 API Key、
完整 prompt / messages】。工具参数只保留模型自己给出的那几个业务参数。
"""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    """当前时间。

    用带时区的 UTC 而不是 datetime.now() 的本地时间：Trace 要跨机器、跨日志
    比对，一个不带时区的时间戳没法判断它属于哪个时区。
    """
    return datetime.now(timezone.utc)


def _elapsed_ms(started_perf: float) -> float:
    """从计时起点到现在的毫秒数。

    用 time.perf_counter() 而不是 time.time()：后者会被系统对时（NTP）往回拨，
    于是算出负的耗时；perf_counter 是单调时钟，只增不减。
    保留两位小数 —— 毫秒下再多的小数位没有意义，反而让日志变长。
    """
    return round((time.perf_counter() - started_perf) * 1000, 2)


def format_error(exc: BaseException) -> str:
    """把异常压成一行可读文本，供 Trace 记录。

    只取「异常类型 + 消息」，【不带 traceback】：堆栈里的本机路径、
    变量内容都属于内部信息，而 Trace 会进日志、也可能随接口返回。
    消息本身是安全的 —— llm / mcp_client 抛出的文案里刻意不含密钥和连接串
    （见它们的 docstring）。

    全项目只在这里格式化一次错误文本，是为了让「不泄漏内部细节」这条规则
    只有一个实现处，不会因为某处顺手多写了个字段而破功。
    """
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


class LLMCallTrace(BaseModel):
    """一次 LLM 调用。

    只记「这次调用的客观事实」，不记 prompt、messages、密钥 ——
    排查慢和排查错都不需要看到提问原文，而记下来就等于把用户数据抄了一份。
    """

    model: str = Field(description="模型名，例如 deepseek-chat")
    duration_ms: float = Field(description="本次调用耗时（毫秒）")
    success: bool = Field(default=True, description="是否成功")
    error: str | None = Field(default=None, description="失败原因；成功时为 null")


class ToolCallTrace(BaseModel):
    """一次工具调用。

    arguments 只保留模型给出的业务参数（如 query / top_k），
    不含服务端注入的内容（如知识库 ID）—— 那是调用方的上下文，
    不是模型能影响的东西，混在一起会让「模型到底传了什么」变得看不清楚。
    """

    tool: str = Field(description="工具名，模型看到的就是这个名字")
    arguments: dict[str, Any] = Field(default_factory=dict, description="本次调用使用的参数")
    duration_ms: float = Field(description="本次调用耗时（毫秒）")
    success: bool = Field(default=True, description="是否成功")
    error: str | None = Field(default=None, description="失败原因；成功时为 null")

    def mark_failed(self, error: str) -> None:
        """把本次调用标记为失败。

        只改「结果」，不碰耗时：耗时由 tool_call() 在退出时统一写入，
        于是判定失败的代码不必关心计时，两件事各归各的。
        """
        self.success = False
        self.error = error


class Trace(BaseModel):
    """一次 Agent Run 的完整记录。"""

    trace_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="本次运行的唯一标识，日志里靠它把一轮请求串起来",
    )
    started_at: datetime = Field(default_factory=_now, description="开始时间（UTC）")
    finished_at: datetime | None = Field(
        default=None, description="结束时间（UTC）；尚未结束时为 null"
    )
    # 用 0 而不是 None 表示「还没结束」：调用方（接口层）拿到的 Trace 一定是
    # 已经 finish 过的，不必为「理论上可能为空」多写一次判断。
    total_duration_ms: float = Field(default=0.0, description="整个 Run 的总耗时（毫秒）")
    iterations: int = Field(default=0, description="「调模型 → 执行工具」循环了几轮")
    llm_calls: list[LLMCallTrace] = Field(
        default_factory=list, description="每一次 LLM 调用，按发生顺序排列"
    )
    tool_calls: list[ToolCallTrace] = Field(
        default_factory=list, description="每一次工具调用，按发生顺序排列"
    )
    error: str | None = Field(default=None, description="Run 失败的原因；成功时为 null")


class TraceContext:
    """一次 run_agent 调用期间的收集器。

    生命周期就是「一次请求」：run_agent 开头建一个，结尾调用 finish()，
    中途所有要记的事情都往这里塞。

    为什么要有这么个对象，而不是散着传一个 Trace：Trace 是【结果」，
    记录过程中还需要计时起点这类「过程状态」。把这些挡在 TraceContext 里，
    Trace 就始终是干净的、可以直接序列化返回的数据。

    用法：

        trace = TraceContext()
        async with trace.llm_call(model="deepseek-chat"):
            message = await llm.chat_with_tools(...)
        ...
        trace.finish()
    """

    def __init__(self) -> None:
        self._trace = Trace()
        # 单独存一个 perf_counter 起点：total_duration_ms 要从「建 Trace」那一刻
        # 算起，而 started_at 是人类可读的时间戳（微秒级、还可能被对时调整），
        # 不适合拿来算耗时。
        self._started_perf = time.perf_counter()

    @property
    def trace(self) -> Trace:
        """当前这条 Trace。

        自始至终是【同一个对象】：记录点直接往里追加，finish() 也是就地补上
        结束信息。所以 finish() 之后再读到的，就是一条完整记录。
        """
        return self._trace

    def count_iteration(self) -> None:
        """记一轮 Agent 循环（一次「调模型 + 执行它要求的工具」）。"""
        self._trace.iterations += 1

    @asynccontextmanager
    async def llm_call(self, model: str) -> AsyncIterator[LLMCallTrace]:
        """记录一次 LLM 调用的耗时与成败。

        【不吞异常】：失败先记进 Trace，再把原异常抛出去 ——
        调用方看到的错误和没有 Trace 时完全一样。
        """
        record = LLMCallTrace(model=model, duration_ms=0.0)
        started = time.perf_counter()
        try:
            yield record
        except Exception as exc:
            record.success = False
            record.error = format_error(exc)
            raise
        finally:
            # 放在 finally 里：成功和失败都要记时长 —— 失败的调用同样耗时，
            # 而且「失败前等了多久」往往正是排查超时最需要的那条信息。
            record.duration_ms = _elapsed_ms(started)
            self._trace.llm_calls.append(record)

    @asynccontextmanager
    async def tool_call(
        self, tool: str, arguments: dict[str, Any] | None = None
    ) -> AsyncIterator[ToolCallTrace]:
        """记录一次工具调用的耗时与成败。

        参数可以先不传：有的工具要等参数校验完才知道最终用了什么值
        （比如 top_k 被收敛过、多传的键被丢弃），那时再给 record.arguments
        赋值即可 —— 记录是就地改的，退出时记下的就是最终值。
        """
        record = ToolCallTrace(tool=tool, arguments=dict(arguments or {}), duration_ms=0.0)
        started = time.perf_counter()
        try:
            yield record
        except Exception as exc:
            record.success = False
            record.error = format_error(exc)
            raise
        finally:
            record.duration_ms = _elapsed_ms(started)
            self._trace.tool_calls.append(record)

    def finish(self, error: str | None = None) -> Trace:
        """收尾：补上结束时间、总耗时，以及（如果有）错误。返回这条 Trace。

        可以安全地重复调用：结束信息只在第一次写入。这样「正常路径记一次、
        异常路径再记一次」的重叠写法不会把总耗时算成两遍。
        """
        if self._trace.finished_at is None:
            self._trace.finished_at = _now()
            self._trace.total_duration_ms = _elapsed_ms(self._started_perf)

        if error is not None:
            self._trace.error = error

        return self._trace
