"""tool_calling grader 的离线测试。

只测 grader 本身：喂手搓的 Trace 和 Artifact，验判定结果。
不联网、不调模型、不连 Milvus —— grader 是纯函数，本来就该能这样测。

这里专门覆盖 tool_selection 的 F1 计算，因为这里出过一个真实的 bug：
把「precision 和 recall 都为 0」时的兜底值写成了 1.0，
于是「工具全选错」被判成满分。这个 bug 特别值得留一条用例，
因为它正好是最危险的形态 —— 报告上会显示成一片绿，
而且只有 selection 这一项失真，其它检查照常报警，
看起来像「只有参数有点小问题」。
"""

from app.services.agent import ToolCallRecord
from app.services.trace import ToolCallTrace, Trace
from tests.evals.graders.tool_calling import grade
from tests.evals.schemas import (
    ToolCallingArtifact,
    ToolCallingCase,
    ToolExpectation,
)


def _make_case(expected_tools: list[str]) -> ToolCallingCase:
    return ToolCallingCase(
        id="case",
        question="问题",
        corpus="product_docs",
        expect=ToolExpectation(tools=expected_tools),
    )


def _make_artifact(requested: list[str]) -> ToolCallingArtifact:
    """按「模型请求了这些工具、且都成功执行」构造证据。"""
    trace = Trace()
    executed: list[ToolCallRecord] = []
    for name in requested:
        trace.tool_calls.append(ToolCallTrace(tool=name, arguments={}, duration_ms=1.0))
        executed.append(ToolCallRecord(tool=name, arguments={}))
    return ToolCallingArtifact(trace=trace, executed=executed, answer="答")


def _selection_metric(artifact: ToolCallingArtifact, case: ToolCallingCase) -> tuple[bool, float]:
    result = grade(case, artifact)
    check = next(c for c in result.checks if c.name == "tool_selection")
    assert check.metric is not None
    return bool(check.passed), check.metric


def test_selection_f1_is_one_when_tool_matches() -> None:
    """选对了工具：F1 = 1，判定通过。"""
    passed, metric = _selection_metric(
        _make_artifact(["search_knowledge_base"]),
        _make_case(["search_knowledge_base"]),
    )

    assert metric == 1.0
    assert passed is True


def test_selection_f1_is_zero_when_tool_is_completely_wrong() -> None:
    """工具完全选错：F1 必须为 0，而不是 1。

    这是回归点。precision 和 recall 同时为 0 时，
    兜底值写成 1.0 就会把「一个都没选对」记成满分。
    """
    passed, metric = _selection_metric(
        _make_artifact(["delete_everything"]),
        _make_case(["search_knowledge_base"]),
    )

    assert metric == 0.0
    assert passed is False


def test_selection_f1_is_one_when_both_sides_are_empty() -> None:
    """期望不调工具、实际也没调：F1 = 1，判定通过。

    和上一个用例是一对：两边都为空是「完美」，一边为空另一边非空是「全错」。
    它们的 F1 分子分母都长得像 0/0，必须靠分支顺序分开，不能共用一个兜底值。
    """
    passed, metric = _selection_metric(_make_artifact([]), _make_case([]))

    assert metric == 1.0
    assert passed is True


def test_selection_metric_between_zero_and_one_for_partial_overlap() -> None:
    """部分命中：F1 落在 (0, 1) 之间，且判定不通过。

    同时验证多调了工具（precision < 1）也会导致不通过 ——
    默认不容忍 expect.tools 之外的调用。
    """
    passed, metric = _selection_metric(
        _make_artifact(["search_knowledge_base", "delete_everything"]),
        _make_case(["search_knowledge_base"]),
    )

    assert 0.0 < metric < 1.0
    assert passed is False


def test_selection_ignores_extra_tools_when_allowed() -> None:
    """allow_extra_tools=True 时只要求不漏调，多调不扣通过判定。"""
    case = ToolCallingCase(
        id="case",
        question="问题",
        corpus="product_docs",
        expect=ToolExpectation(tools=["search_knowledge_base"], allow_extra_tools=True),
    )

    passed, metric = _selection_metric(
        _make_artifact(["search_knowledge_base", "mcp_get_current_time"]), case
    )

    assert passed is True
    # 分数仍然如实反映「多调了一个」。
    assert metric < 1.0
