"""Tool Calling 评测：模型有没有选对工具、传对参数、调对次数。

**这个 grader 是纯函数**：输入用例和证据，输出判定，不联网、不调模型、不读文件。
所以它完全可复现，也能离线单测（手搓一个 Trace 就能跑）。
这一点不是洁癖 —— 工具选择本来就该是确定性的（chat_with_tools 的
temperature 固定 0.0），如果连判定都要靠另一个模型来猜，那评测结果里
就再也分不清「模型选错了」和「判官看走眼了」。

五项检查，对应要求里的 Selection / Arguments / Call Count / Order / Error：

    tool_selection   —— 模型请求了哪些工具（含被拒绝的）
    arguments        —— 每次成功调用实际用的参数（按数据集声明的字段逐项评）
    call_count       —— 实际执行了几次
    order            —— 多工具时的相对顺序
    error_free_rate  —— 有没有失败/被拒绝的调用

为什么 selection 与 count 用【不同】的来源：
    selection 看 trace.tool_calls（模型请求过的全部，含被白名单拒绝的）——
    这样「模型臆造了一个工具」才会在「选错工具」上体现出来；
    count 看 artifact.executed（参数合法、真正进入执行的）——
    被拒绝的调用连参数都没解析，算进「调用次数」会误导归因。
两者混用会让某一类错误在两层同时漏掉，理由详见 §代码注释处的具体分支。
"""

from typing import Any

from tests.evals.graders import build_grade_result, normalize_text
from tests.evals.schemas import (
    ArgMatcher,
    CheckResult,
    GradeResult,
    ToolCallingArtifact,
    ToolCallingCase,
)

# 各项权重。加起来恰好 1.0，跳过某项时由 build_grade_result 自动重新归一化。
#
# 为什么 selection 和 arguments 各占 0.3：它们是「工具调用」这件事的两个核心
# ——选错了工具，参数再准也没有意义；选对了工具但参数离谱，等于没查。
# count / order / error 更像「行为规范」，出问题的概率低，权重相应小一些。
WEIGHT_TOOL_SELECTION: float = 0.30
WEIGHT_ARGUMENTS: float = 0.30
WEIGHT_CALL_COUNT: float = 0.15
WEIGHT_ORDER: float = 0.10
WEIGHT_ERROR_FREE: float = 0.15

# 用来区分「参数里没有这个键」和「这个键的值是 None」。
# 模型完全可能不传 top_k（走默认值），那和「传了个 null」是两回事，
# 不能都当成「取出来是 None」处理。
_MISSING: object = object()


def _subsequence_contains(sequence: list[str], expected: list[str]) -> bool:
    """expected 是否是 sequence 的子序列（保持相对顺序即可，允许中间插别的）。

    用子序列而不是全等匹配，是因为模型被允许重复调用同一个工具
    （比如先粗查一次、改写后再查一次）。要求全等的话，
    「查了两次、顺序正确」会被判成顺序错误 —— 那评的就不是顺序了。
    """
    if not expected:
        return True
    index = 0
    for name in sequence:
        if name == expected[index]:
            index += 1
            if index == len(expected):
                return True
    return False


def _match_argument(matcher: ArgMatcher, actual: Any) -> tuple[bool, str]:
    """按匹配器判定一个参数值，返回 (是否通过, 说明)。"""
    if actual is _MISSING:
        return False, "模型没有传这个参数"

    if matcher.match == "exact":
        passed = normalize_text(str(actual)) == normalize_text(str(matcher.value))
        return passed, f"期望 exact {matcher.value!r}，实际 {actual!r}"

    if matcher.match == "any_of":
        text = normalize_text(str(actual))
        hit = [value for value in matcher.values if normalize_text(value) in text]
        return bool(hit), f"期望包含 {matcher.values} 中任一，命中 {hit or '无'}"

    if matcher.match == "contains_all":
        text = normalize_text(str(actual))
        missing = [value for value in matcher.values if normalize_text(value) not in text]
        return not missing, f"期望包含全部 {matcher.values}，缺失 {missing or '无'}"

    # 以下是数值类匹配。bool 是 int 的子类，先排除它 ——
    # 模型传 true 当 top_k 时，不该被当成「1」通过。
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False, f"期望数值，实际 {actual!r}（{type(actual).__name__}）"

    if matcher.match == "lte":
        return actual <= matcher.value, f"期望 <= {matcher.value}，实际 {actual}"
    if matcher.match == "gte":
        return actual >= matcher.value, f"期望 >= {matcher.value}，实际 {actual}"
    if matcher.match == "range":
        passed = matcher.min <= actual <= matcher.max  # type: ignore[operator]
        return passed, f"期望 [{matcher.min}, {matcher.max}]，实际 {actual}"

    # ArgMatcher.match 是 Literal，Pydantic 已经挡住了别的取值，走不到这里。
    return False, f"未知的匹配方式：{matcher.match}"


def _check_tool_selection(case: ToolCallingCase, requested: list[str]) -> CheckResult:
    """Selection：模型请求的工具集合与期望集合的吻合程度。"""
    expected = set(case.expect.tools)
    actual = set(requested)
    common = expected & actual

    # 空集合的 precision/recall 定为 1.0：
    # 「期望不调工具，实际也没调」应当算完美，而不是 0/0。
    precision = 1.0 if not actual else len(common) / len(actual)
    recall = 1.0 if not expected else len(common) / len(expected)

    # F1。这里的 else 分支必须是 0.0，不能是 1.0 ——
    # 只有 precision 和 recall 都为 0 时才会走到 else（两边非空且毫无交集，
    # 也就是「工具全选错了」），那是最差的选法，给 1.0 会把最严重的错误
    # 记成满分。而「双方都为空」的情形早在上面就被 precision=recall=1.0 接住了，
    # 根本不会落到 else 里 —— 这两个分支各自负责一种情况，不能互相顶替。
    metric = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # allow_extra_tools 只放过「多调了」，不放过「漏调了」。
    # 「多调一个」有时是合理的探索（模型先查了 A 又查了 B），
    # 但「该查却没查」永远是能力问题。
    if case.expect.allow_extra_tools:
        passed = recall == 1.0
    else:
        passed = precision == 1.0 and recall == 1.0

    extra = sorted(actual - expected)
    missing = sorted(expected - actual)

    return CheckResult(
        name="tool_selection",
        passed=passed,
        metric=metric,
        weight=WEIGHT_TOOL_SELECTION,
        expected={"tools": case.expect.tools, "allow_extra_tools": case.expect.allow_extra_tools},
        actual={"requested": requested, "unique": sorted(actual)},
        detail=(
            "选择正确"
            if passed
            else f"多调了 {extra or '无'}；漏调了 {missing or '无'}"
        ),
    )


def _check_arguments(case: ToolCallingCase, artifact: ToolCallingArtifact) -> list[CheckResult]:
    """Arguments：按数据集声明的字段逐个评。

    只对 trace 里 success=True 的调用取参数。理由是 agent.py 的
    _execute_search_tool 把 call.arguments 的赋值放在了参数校验【之后】——
    被拒绝、JSON 非法、校验失败的调用，其 arguments 都是空的 {}。
    拿它们来评参数，会把「参数根本没能解析出来」记成「参数为空」，
    归因就错了（那属于 Error 项的问题，不属于 Arguments）。
    """
    declared_total = sum(len(fields) for fields in case.expect.args.values())
    if declared_total == 0:
        return []

    # 每个字段平分 arguments 这一项的总权重。
    # 平均分而不是让某个字段独占，是因为数据集里声明了几个字段，
    # 就说明这几个字段同等重要（要突出某一个，应该拆成独立用例）。
    weight_per_field = WEIGHT_ARGUMENTS / declared_total

    successful = [call for call in artifact.trace.tool_calls if call.success]
    checks: list[CheckResult] = []

    for tool_name, field_matchers in case.expect.args.items():
        calls = [call for call in successful if call.tool == tool_name]

        for field, matcher in field_matchers.items():
            values = [
                call.arguments[field] if field in call.arguments else _MISSING
                for call in calls
            ]

            if not calls:
                checks.append(
                    CheckResult(
                        name=f"arguments:{tool_name}.{field}",
                        passed=False,
                        metric=0.0,
                        weight=weight_per_field,
                        expected=matcher.model_dump(),
                        actual={"successful_calls": 0},
                        detail=f"没有任何一次成功的 {tool_name} 调用，无法核对参数",
                    )
                )
                continue

            results = [_match_argument(matcher, value) for value in values]
            # 「至少一次满足」即通过：模型可以先粗查再改写重查，
            # 只要有一次把参数给对了，就说明它知道该传什么。
            # 每次的实际值都记进 actual，便于人工复核是哪一次对上的。
            passed = any(ok for ok, _ in results)

            checks.append(
                CheckResult(
                    name=f"arguments:{tool_name}.{field}",
                    passed=passed,
                    metric=1.0 if passed else 0.0,
                    weight=weight_per_field,
                    expected=matcher.model_dump(),
                    actual=[None if v is _MISSING else v for v in values],
                    detail=(
                        f"通过（{len(values)} 次调用）"
                        if passed
                        else "；".join(detail for _, detail in results)
                    ),
                )
            )

    return checks


def _check_call_count(case: ToolCallingCase, artifact: ToolCallingArtifact) -> CheckResult:
    """Call Count：实际进入执行的调用次数是否落在允许区间。"""
    count = len(artifact.executed)
    low = case.expect.min_calls or 0
    high = case.expect.max_calls if case.expect.max_calls is not None else count
    passed = low <= count <= high

    return CheckResult(
        name="call_count",
        passed=passed,
        metric=1.0 if passed else 0.0,
        weight=WEIGHT_CALL_COUNT,
        expected={"min_calls": low, "max_calls": high},
        actual={"calls": count, "tools": [r.tool for r in artifact.executed]},
        detail="调用次数符合预期" if passed else f"期望 {low}~{high} 次，实际 {count} 次",
    )


def _check_order(case: ToolCallingCase, requested: list[str]) -> CheckResult:
    """Order：多工具调用的相对顺序。

    数据集没声明 order 时返回 skipped（passed/metric 都是 None），
    而不是默认通过 —— 默认通过会让「没测顺序」在报告里看起来像「顺序正确」。
    """
    if not case.expect.order:
        return CheckResult(
            name="order",
            passed=None,
            metric=None,
            weight=WEIGHT_ORDER,
            detail="数据集未声明顺序期望，跳过",
        )

    passed = _subsequence_contains(requested, case.expect.order)

    return CheckResult(
        name="order",
        passed=passed,
        metric=1.0 if passed else 0.0,
        weight=WEIGHT_ORDER,
        expected=case.expect.order,
        actual=requested,
        detail="顺序正确" if passed else f"期望（子序列）{case.expect.order}，实际 {requested}",
    )


def _check_errors(artifact: ToolCallingArtifact) -> CheckResult:
    """Error：有没有失败的调用。

    被白名单拒绝的调用也在这里（agent.py 里 _failed_call 会 mark_failed），
    所以「模型臆造了一个工具」会同时体现在 selection 和这里。
    这是刻意接受的重叠：两项的语义本来就都对，
    为了去重而让某一项漏掉一类错误，得不偿失。
    """
    all_calls = artifact.trace.tool_calls
    failed = [call for call in all_calls if not call.success]

    total = len(all_calls)
    metric = 1.0 if total == 0 else 1.0 - len(failed) / total
    passed = not failed and artifact.trace.error is None

    return CheckResult(
        name="error_free_rate",
        passed=passed,
        metric=metric,
        weight=WEIGHT_ERROR_FREE,
        expected={"failed_calls": 0, "trace_error": None},
        actual={
            "failed_calls": [{"tool": c.tool, "error": c.error} for c in failed],
            "trace_error": artifact.trace.error,
        },
        detail=(
            "无失败调用"
            if passed
            else (
                f"{len(failed)} 次调用失败："
                + "；".join(f"{c.tool} -> {c.error}" for c in failed)
                + (f"；run 级错误：{artifact.trace.error}" if artifact.trace.error else "")
            )
        ),
    )


def grade(case: ToolCallingCase, artifact: ToolCallingArtifact) -> GradeResult:
    """判定一条 tool_calling 用例。

    参数：
        case:     数据集里的一行。
        artifact: runner 跑完 run_agent 之后整理出的证据。

    返回：
        GradeResult。全程不联网、不调模型。
    """
    requested = [call.tool for call in artifact.trace.tool_calls]

    checks: list[CheckResult] = [
        _check_tool_selection(case, requested),
        *_check_arguments(case, artifact),
        _check_call_count(case, artifact),
        _check_order(case, requested),
        _check_errors(artifact),
    ]

    return build_grade_result(
        case_id=case.id,
        layer="tool_calling",
        checks=checks,
        evidence={
            # 用 mode="json" 把 datetime 之类转成 JSON 原生类型，
            # 保证整份报告可以 model_dump_json() 直接落盘。
            "trace": artifact.trace.model_dump(mode="json"),
            # 与 final_answer 层的 evidence 保持同一个键名，
            # runner 才能不分层地统一统计成本。
            "llm_calls": len(artifact.trace.llm_calls),
            "executed": [
                {"tool": record.tool, "arguments": record.arguments}
                for record in artifact.executed
            ],
            "requested": requested,
            # 把回答原文也留下。工具没被调用时，报告上只有「漏调了 xxx」，
            # 看不出模型当时在干什么 —— 而「它直接答了」和「它拒答了」
            # 是两种完全不同的问题，对应完全不同的修法。
            # 这是模型自己的输出，不是提示词，与 Trace 不记 prompt 的约定不冲突。
            "answer": artifact.answer or "",
            "answer_length": len(artifact.answer or ""),
        },
    )
