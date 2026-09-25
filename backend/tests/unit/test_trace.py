"""Trace 数据结构本身的单元测试。

这里只测 services/trace.py 自己：能不能生成 id、时长是不是真的在计、
失败会不会被记下来、以及最要紧的一条 —— 【它绝不能吞异常】。
Agent 那一侧怎么用它，在 test_agent_trace.py 里测。
"""

import asyncio
import pytest

from app.services.trace import LLMCallTrace, ToolCallTrace, Trace, TraceContext

# 让异步上下文管理器里真的过掉一段时间，再断言记到的时长对得上。
# 取 20ms 是因为 Windows 的定时器精度只有毫秒级偏粗，睡 10ms 实际可能到 15ms 以上，
# 而 asyncio.sleep 【不会提前返回】，所以「>= 20」是个稳的下界，
# 同时又能挡住「时长恒为 0」这种没接上计时的实现。
SLEEP_SECONDS = 0.02
SLEEP_MS = 20.0


def test_trace_generates_trace_id() -> None:
    """每条 Trace 自带一个 uuid4 格式的 id，且互不相同。"""
    first = Trace()
    second = Trace()

    assert first.trace_id
    assert first.trace_id != second.trace_id

    # 形状校验：uuid4 的标准写法是 8-4-4-4-12、第三段以 4 开头。
    # 只看「非空」的话，改成递增序号也能通过，但那不是我们要的 id。
    parts = first.trace_id.split("-")
    assert [len(part) for part in parts] == [8, 4, 4, 4, 12]
    assert parts[2].startswith("4")


async def test_llm_call_records_duration() -> None:
    """llm_call 记下真实耗时，并默认算成功。"""
    trace = TraceContext()

    async with trace.llm_call(model="deepseek-chat"):
        await asyncio.sleep(SLEEP_SECONDS)

    assert len(trace.trace.llm_calls) == 1
    call = trace.trace.llm_calls[0]
    assert call.model == "deepseek-chat"
    assert call.duration_ms >= SLEEP_MS
    assert call.success is True
    assert call.error is None


async def test_llm_call_records_error_and_reraises() -> None:
    """LLM 调用失败：错误记进 Trace，异常照旧往外抛。"""
    trace = TraceContext()

    with pytest.raises(RuntimeError, match="DeepSeek"):
        async with trace.llm_call(model="deepseek-chat"):
            raise RuntimeError("DeepSeek 调用失败")

    # 异常被原样抛出（pytest.raises 已经验过），同时记录留下来了 ——
    # 这两件事必须【同时】成立：Trace 是旁路，不能因为记录而改变错误语义。
    call = trace.trace.llm_calls[0]
    assert call.success is False
    assert call.error == "RuntimeError: DeepSeek 调用失败"
    # 失败的调用同样耗时，所以时长照样要记。
    assert call.duration_ms >= 0


async def test_tool_call_records_duration_and_arguments() -> None:
    """tool_call 记下工具名、参数和真实耗时。"""
    trace = TraceContext()

    async with trace.tool_call("search_knowledge_base") as call:
        # 参数可以在拿到手之后再补：实际参数往往要等校验完才知道。
        call.arguments = {"query": "Milvus 是什么", "top_k": 3}
        await asyncio.sleep(SLEEP_SECONDS)

    assert len(trace.trace.tool_calls) == 1
    recorded = trace.trace.tool_calls[0]
    assert recorded.tool == "search_knowledge_base"
    assert recorded.arguments == {"query": "Milvus 是什么", "top_k": 3}
    assert recorded.duration_ms >= SLEEP_MS
    assert recorded.success is True


async def test_tool_call_can_be_marked_failed() -> None:
    """工具失败有两种情形：抛异常，和自己报错（mark_failed）。

    后者必须支持，是因为本项目的工具执行【不抛异常】—— 失败是作为结果返回的，
    没有异常可记，只能由执行方显式标记。
    """
    trace = TraceContext()

    async with trace.tool_call("search_knowledge_base") as call:
        call.mark_failed("knowledge base search failed")

    recorded = trace.trace.tool_calls[0]
    assert recorded.success is False
    assert recorded.error == "knowledge base search failed"


def test_tool_call_trace_does_not_share_arguments_dict() -> None:
    """传进去的参数字典被复制一份，外部改动不会污染记录。

    不是洁癖：调用方手里的那个 dict 后面还可能被改写（比如收敛 top_k），
    共享同一个对象就会让「记录的参数」跟着变，事后对不上账。
    """
    trace = TraceContext()
    arguments = {"query": "原始查询"}

    async def scenario() -> None:
        async with trace.tool_call("search_knowledge_base", arguments):
            pass

    asyncio.run(scenario())
    arguments["query"] = "改过的查询"

    assert trace.trace.tool_calls[0].arguments == {"query": "原始查询"}


def test_finish_sets_total_duration_and_error() -> None:
    """finish 补上结束时间与总耗时；重复调用不会把耗时算两遍。"""
    trace = TraceContext()

    finished = trace.finish()
    assert finished.finished_at is not None
    assert finished.total_duration_ms >= 0
    assert finished.error is None

    first_total = finished.total_duration_ms
    trace.finish()
    assert trace.trace.total_duration_ms == first_total


def test_finish_records_error() -> None:
    """finish(error=...) 把失败原因留在 Trace 上。"""
    trace = TraceContext()

    trace.finish(error="RuntimeError: boom")

    assert trace.trace.error == "RuntimeError: boom"


def test_trace_models_default_to_success() -> None:
    """两个调用记录默认是成功的，只有显式标记才会变成失败。"""
    assert LLMCallTrace(model="m", duration_ms=1.0).success is True
    assert ToolCallTrace(tool="t", duration_ms=1.0).success is True
