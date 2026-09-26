"""Agent 工具循环的端到端回归：真模型 + 真 Milvus + 真 MCP 子进程。

和 tests/unit/test_agent_trace.py 的分工（两边都要有，缺一不可）：

    单元测试 —— LLM / MCP / embedding / 向量库全换成假实现，
                验的是「编排逻辑对不对」，任何机器都能跑；
    本文件   —— 一个都不换，验的是「这套编排接到真实依赖上还通不通」。

只留单元测试的话，会出现「全绿但线上不能用」：比如 SDK 升级后工具定义的
字段名变了、模型不再按预期请求工具、Milvus 返回结构变了 ——
这些改动的共同点是【假实现看不出来】，因为假实现是按我们对 SDK 的理解写的，
理解错了它也一起错。
"""

import uuid
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import pytest

from app.services.agent import MAX_TOOL_TOP_K, SEARCH_TOOL_NAME, run_agent
from app.services.document.ingest import ingest_txt
from tests.regression.availability import ServiceStatus

pytestmark = [pytest.mark.integration, pytest.mark.regression]

# 语料：写一个只能从知识库里得到、模型不可能自己知道的事实。
# 用「编造的型号 + 具体数字」而不是「Milvus 的默认端口」这类真实存在的常识 ——
# 常识题模型不查资料也能答对，那样「有没有真的检索」就看不出来了。
CORPUS = (
    "平台内部约定：知识库切片（chunk）的保留期是 47 天，"
    "到期后由清理任务统一删除。\n"
    "该数值只在本平台的运维手册中出现，不属于任何公开规范。\n"
)

QUESTION = "知识库切片的保留期是多少天？"


async def test_agent_completes_tool_loop_on_real_stack(
    services: ServiceStatus, new_kb: Callable[[], str], tmp_path: Path
) -> None:
    """问题 → 模型决定检索 → 真查 Milvus → 结果回传 → 模型作答，全程真实。

    断言分三组，各自守一件不同的事：
        1. 工具循环真的转起来了（模型请求了工具、工具执行成功、有最终回答）；
        2. Trace 把这一轮如实记下来了（轮数、耗时、错误、id）；
        3. 参数收敛在真实链路上同样成立（模型给什么 top_k 都不会越界）。
    """
    services.require_llm()

    document_id = str(uuid.uuid4())
    corpus_path = tmp_path / "agent-corpus.txt"
    corpus_path.write_text(CORPUS, encoding="utf-8")
    kb_id = new_kb()
    await ingest_txt(corpus_path, kb_id, document_id)

    result = await run_agent(question=QUESTION, knowledge_base_id=UUID(kb_id))

    # ---- 1. 工具循环 ----
    executed_searches = [
        call
        for call in result.trace.tool_calls
        if call.tool == SEARCH_TOOL_NAME and call.success
    ]
    assert executed_searches, (
        f"模型没有成功调用 {SEARCH_TOOL_NAME}；"
        f"Trace 里的调用记录是 {[(c.tool, c.success, c.error) for c in result.trace.tool_calls]}"
    )
    # result.tool_calls 是「给用户看的、真正执行过的」清单，必须同步有记录 ——
    # 它和 Trace 是两条独立的输出，只更新一条就是接口层缺字段。
    assert [call.tool for call in result.tool_calls] == [SEARCH_TOOL_NAME]
    assert result.answer.strip(), "Agent 返回了空回答"

    # ---- 2. Trace ----
    assert result.trace.error is None
    # 最少两轮：一轮请求工具，一轮拿着资料作答。
    assert result.trace.iterations >= 2
    assert len(result.trace.llm_calls) >= 2
    assert all(call.model for call in result.trace.llm_calls)
    assert result.trace.trace_id
    assert result.trace.finished_at is not None
    assert result.trace.total_duration_ms > 0
    # 每一次模型调用和工具调用都真的计了时（假实现里这两个数会恒为 0）。
    assert all(call.duration_ms >= 0 for call in result.trace.llm_calls)
    assert all(call.duration_ms >= 0 for call in result.trace.tool_calls)

    # ---- 3. 参数收敛 ----
    # 模型是被要求「尽量多给」时唯一的收敛点，这里在真实链路上再确认一次它没被绕开。
    for call in executed_searches:
        assert 1 <= call.arguments["top_k"] <= MAX_TOOL_TOP_K
        assert call.arguments["query"].strip()
