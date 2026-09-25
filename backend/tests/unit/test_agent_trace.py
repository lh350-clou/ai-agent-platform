"""Agent 执行链路上的 Trace 测试。

关注的不是「Trace 这个数据结构对不对」（那是 test_trace.py 的事），
而是「run_agent 有没有在正确的地方记下正确的数」：
模型调用和工具调用的耗时、轮数、失败，以及原有的行为有没有被 Trace 改掉。

测试手法是【把外部依赖换掉】—— LLM、MCP、embedding、向量库全是假的，
不联网、不连库、不起子进程，于是这些用例在任何机器上都能跑。
被换掉的只是「外部世界」，被验证的仍然是 run_agent 自己那段编排逻辑。
"""

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from app.api.agent import _to_agent_llm_call, _to_agent_tool_call
from app.schemas.agent import AgentResponse
from app.services import agent
from app.services.agent import MAX_TOOL_ITERATIONS, SEARCH_TOOL_NAME, run_agent
from app.services.trace import LLMCallTrace, ToolCallTrace, Trace, TraceContext

KB_ID = uuid4()

# 假的 LLM 调用要真的等一会儿，才能验出「耗时是真记的」而不是恒为 0。
# 20ms 的依据见 test_trace.py 里的 SLEEP_MS。
LLM_SLEEP_SECONDS = 0.02
LLM_SLEEP_MS = 20.0
TOOL_SLEEP_SECONDS = 0.02
TOOL_SLEEP_MS = 20.0


def _final_message(text: str) -> Any:
    """模型直接作答的那条消息：没有工具调用。"""
    return SimpleNamespace(content=text, tool_calls=None)


def _tool_call_message(call_id: str, name: str, arguments: dict[str, Any]) -> Any:
    """模型要求调用工具的那条消息。

    用 SimpleNamespace 而不是构造真的 ChatCompletionMessage：
    agent.py 只读 content / tool_calls / id / function.name / function.arguments
    这几个属性，手搓一个「只有这几个属性的对象」比依赖 SDK 的构造函数
    更直白，也不会因为 SDK 升级改了字段名而莫名其妙地失败。
    """
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
            )
        ],
    )


def _install_llm(monkeypatch: pytest.MonkeyPatch, replies: list[Any]) -> None:
    """把 LLM 换成一份「按顺序回放」的假实现。"""

    async def fake_chat_with_tools(messages: Any, tools: Any) -> Any:
        await asyncio.sleep(LLM_SLEEP_SECONDS)
        return replies.pop(0)

    monkeypatch.setattr(agent.llm, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(agent.llm, "model_name", lambda: "deepseek-chat")


def _install_no_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """让本次 Run 看到「没有 MCP 可用」。

    真去连 MCP 会拉起一个 Python 子进程（几百毫秒起步，还可能因为环境问题挂掉），
    而这里要测的东西和 MCP 无关。置空之后 Agent 照常降级成只用内置工具。
    """

    async def fake_open_mcp(stack: Any) -> tuple[Any, list[Any], frozenset[str]]:
        return None, [], frozenset()

    monkeypatch.setattr(agent, "_open_mcp", fake_open_mcp)


def _install_search_tool(
    monkeypatch: pytest.MonkeyPatch, hits: list[dict[str, Any]] | None = None
) -> None:
    """把内置检索工具依赖的向量化与检索换成假的。"""

    async def fake_embed_text(text: str) -> list[float]:
        return [0.0] * 8

    async def fake_search(**kwargs: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(TOOL_SLEEP_SECONDS)
        return hits if hits is not None else [{"chunk_id": "c1", "content": "Milvus 是向量数据库"}]

    monkeypatch.setattr(agent, "embed_text", fake_embed_text)
    monkeypatch.setattr(agent.vector_store, "search", fake_search)


async def test_agent_result_carries_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """正常跑完一轮：AgentResult 带着一条完整的 Trace。"""
    _install_llm(monkeypatch, [_final_message("你好，有什么可以帮你？")])
    _install_no_mcp(monkeypatch)

    result = await run_agent(question="你好", knowledge_base_id=KB_ID)

    assert result.answer == "你好，有什么可以帮你？"
    # 模型一次就作答，没有检索 —— 但 Trace 不该是空的。
    assert result.trace.tool_calls == []
    assert result.trace.iterations == 1
    assert result.trace.error is None
    assert result.trace.finished_at is not None
    assert result.trace.total_duration_ms > 0
    assert result.trace.trace_id


async def test_llm_duration_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """每次模型调用的耗时被记下来，失败为假、模型名来自 LLM 层。"""
    _install_llm(monkeypatch, [_final_message("答")])
    _install_no_mcp(monkeypatch)

    result = await run_agent(question="问题", knowledge_base_id=KB_ID)

    assert len(result.trace.llm_calls) == 1
    call = result.trace.llm_calls[0]
    assert call.model == "deepseek-chat"
    assert call.duration_ms >= LLM_SLEEP_MS
    assert call.success is True
    assert call.error is None


async def test_tool_call_duration_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型要求检索时：工具名、参数、耗时都记进 Trace。"""
    _install_llm(
        monkeypatch,
        [
            _tool_call_message(
                "call-1", SEARCH_TOOL_NAME, {"query": "Milvus 是什么", "top_k": 3}
            ),
            _final_message("Milvus 是一个向量数据库。"),
        ],
    )
    _install_no_mcp(monkeypatch)
    _install_search_tool(monkeypatch)

    result = await run_agent(question="Milvus 是什么？", knowledge_base_id=KB_ID)

    # 检索一轮 + 作答一轮 = 两轮；两轮各调了一次模型。
    assert result.trace.iterations == 2
    assert len(result.trace.llm_calls) == 2

    assert len(result.trace.tool_calls) == 1
    call = result.trace.tool_calls[0]
    assert call.tool == SEARCH_TOOL_NAME
    assert call.arguments == {"query": "Milvus 是什么", "top_k": 3}
    assert call.duration_ms >= TOOL_SLEEP_MS
    assert call.success is True

    # Trace 是新增的旁路，原有的 tool_calls 行为一点没变：
    # 仍然只有真正执行过的工具，仍然带 arguments。
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool == SEARCH_TOOL_NAME
    assert result.tool_calls[0].arguments == {"query": "Milvus 是什么", "top_k": 3}


async def test_tool_failure_is_recorded_without_crashing_agent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """检索失败：Trace 里记成失败，但 Agent 照常给出回答。

    这两件事必须同时成立 —— 「工具错误不让整个 Agent 崩溃」是原有的行为保证，
    不能因为要记 Trace 而丢掉；而失败也不能在记录里显示成成功。
    """
    # 工具失败会走 logger.exception 打一整段堆栈，测试输出会很吵。
    # 抬高日志级别把这段噪音关掉，不影响测试本身要验的东西。
    caplog.set_level(logging.CRITICAL)

    _install_llm(
        monkeypatch,
        [
            _tool_call_message("call-1", SEARCH_TOOL_NAME, {"query": "随便问问"}),
            _final_message("根据当前知识库无法确定。"),
        ],
    )
    _install_no_mcp(monkeypatch)
    _install_search_tool(monkeypatch)

    async def failing_search(**kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("Milvus 连不上（假装）")

    monkeypatch.setattr(agent.vector_store, "search", failing_search)

    result = await run_agent(question="知识库里有什么？", knowledge_base_id=KB_ID)

    # 没有抛异常，模型拿到了错误提示后照常作答。
    assert result.answer == "根据当前知识库无法确定。"
    assert result.trace.error is None

    call = result.trace.tool_calls[0]
    assert call.success is False
    # 记的是给模型的那句笼统说明，不是带堆栈的异常原文。
    assert call.error == "knowledge base search failed"


async def test_rejected_tool_call_is_recorded_as_failed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """模型请求了不存在的工具：拒绝掉，并在 Trace 里留一条失败记录。

    这条记录的价值恰恰在于「失败也记」：只记成功的话，
    「模型老是调一个不存在的工具」这种问题在 Trace 上完全看不出来。
    """
    caplog.set_level(logging.CRITICAL)

    _install_llm(
        monkeypatch,
        [
            _tool_call_message("call-1", "delete_everything", {}),
            _final_message("抱歉，我做不到。"),
        ],
    )
    _install_no_mcp(monkeypatch)

    result = await run_agent(question="把库清空", knowledge_base_id=KB_ID)

    assert result.answer == "抱歉，我做不到。"
    # 被拒绝的调用不进 tool_calls（那是「真执行过的」清单），但进 Trace。
    assert result.tool_calls == []
    call = result.trace.tool_calls[0]
    assert call.tool == "delete_everything"
    assert call.success is False
    assert call.error == "tool not allowed"


async def test_forced_convergence_records_final_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型一直要求调工具直到轮数用尽：强制收敛，最后那次 chat 也要记进 Trace。

    这条路径平时跑不到，但它恰恰是最需要观测的一种情况 ——
    出现了就说明模型在打转，而「第 5 轮之后还硬要调工具」正是它陷入循环的证据。
    """
    # 每一轮都回一条「我要调工具」，永远不给最终答案。
    _install_llm(
        monkeypatch,
        [
            _tool_call_message(f"call-{i}", SEARCH_TOOL_NAME, {"query": f"第 {i} 次"})
            for i in range(MAX_TOOL_ITERATIONS)
        ],
    )
    _install_no_mcp(monkeypatch)
    _install_search_tool(monkeypatch)

    # 收敛这一步用的是【不带工具】的 chat()，和循环里不是同一个入口，
    # 所以得单独换掉。
    async def fake_chat(messages: Any, temperature: float = 0.7) -> str:
        await asyncio.sleep(LLM_SLEEP_SECONDS)
        return "根据当前知识库无法确定。"

    monkeypatch.setattr(agent.llm, "chat", fake_chat)

    result = await run_agent(question="问题", knowledge_base_id=KB_ID)

    # 原有行为不变：不报错，用一个体面的回答收尾。
    assert result.answer == "根据当前知识库无法确定。"
    # iterations 停在 5：最后一次收敛【不算新一轮】，否则会把「5 轮失控」
    # 读成「6 轮正常往返」。
    assert result.trace.iterations == MAX_TOOL_ITERATIONS
    # 5 次带工具的调用 + 1 次收敛用的调用。
    assert len(result.trace.llm_calls) == MAX_TOOL_ITERATIONS + 1
    assert result.trace.llm_calls[-1].duration_ms >= LLM_SLEEP_MS
    assert result.trace.tool_calls and len(result.trace.tool_calls) == MAX_TOOL_ITERATIONS


async def test_agent_error_is_recorded_in_trace(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """模型调用抛异常：Trace 记下 error，异常原样抛出。

    这里用一个「探针」把 finish() 的结果截下来 ——
    失败时 run_agent 会把异常抛出去，Trace 不会随返回值交出来，
    只能从 finish() 这一步观察。顺带也验了失败路径确实调用了 finish()。
    """
    caplog.set_level(logging.CRITICAL)

    captured: list[Trace] = []
    original_finish = TraceContext.finish

    def spy_finish(self: TraceContext, error: str | None = None) -> Trace:
        trace = original_finish(self, error)
        captured.append(trace)
        return trace

    monkeypatch.setattr(TraceContext, "finish", spy_finish)
    _install_no_mcp(monkeypatch)

    async def failing_chat_with_tools(messages: Any, tools: Any) -> Any:
        raise RuntimeError("DeepSeek 接口返回错误（HTTP 429）")

    monkeypatch.setattr(agent.llm, "chat_with_tools", failing_chat_with_tools)
    monkeypatch.setattr(agent.llm, "model_name", lambda: "deepseek-chat")

    # 原异常必须原样抛出去 —— Trace 不能改变错误语义。
    with pytest.raises(RuntimeError, match="HTTP 429"):
        await run_agent(question="问题", knowledge_base_id=KB_ID)

    assert len(captured) == 1
    trace = captured[0]
    assert trace.error == "RuntimeError: DeepSeek 接口返回错误（HTTP 429）"
    assert trace.finished_at is not None
    assert trace.total_duration_ms > 0
    # 跑到第 1 轮就崩了，这一轮的调用同样留下记录（且是失败的）。
    assert trace.iterations == 1
    assert trace.llm_calls[0].success is False
    assert trace.llm_calls[0].error == "RuntimeError: DeepSeek 接口返回错误（HTTP 429）"


def test_api_response_maps_trace_fields() -> None:
    """接口层把内部 Trace 映射成 response schema，而不是直接塞出去。

    不启 FastAPI、不连数据库：这里验的是「映射这一成」——
    内部结构改了、接口字段却没跟着改（或反过来多漏了字段），
    在构造 AgentResponse 的这一刻就会报错。
    """
    # 直接构造内部记录，模拟 service 层产出的对象。
    trace = Trace(iterations=2, total_duration_ms=1820.4)
    trace.llm_calls.append(
        LLMCallTrace(model="deepseek-chat", duration_ms=640.2, success=True)
    )
    trace.tool_calls.append(
        ToolCallTrace(
            tool=SEARCH_TOOL_NAME,
            arguments={"query": "Milvus", "top_k": 3},
            duration_ms=120.0,
        )
    )

    response = AgentResponse(
        conversation_id=uuid4(),
        answer="答",
        tool_calls=[_to_agent_tool_call(record) for record in trace.tool_calls],
        trace_id=trace.trace_id,
        iterations=trace.iterations,
        llm_calls=[_to_agent_llm_call(record) for record in trace.llm_calls],
        total_duration_ms=trace.total_duration_ms,
    )

    assert response.trace_id == trace.trace_id
    assert response.iterations == 2
    assert response.total_duration_ms == 1820.4
    assert response.llm_calls[0].model == "deepseek-chat"
    assert response.llm_calls[0].duration_ms == 640.2
    # 便捷字段仍然从 arguments 里取，没有被 Trace 影响。
    assert response.tool_calls[0].query == "Milvus"
    assert response.tool_calls[0].top_k == 3
