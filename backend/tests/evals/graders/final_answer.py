"""Final Answer 评测：答案对不对、切不切题、有没有依据。

设计原则是「确定性检查优先」：
    能靠子串匹配判掉的，绝不调模型。事实性问题的答案里该出现什么词、
    不该出现什么词，数据集已经写清楚了，用一次模型调用来判它是浪费，
    而且引入了一个会抖动的判官 —— 同一个答案两次跑出不同结论，
    会让人无法判断模型是不是真的变好了。

所以五项检查里只有两项会用到 LLM Judge：
    relevance     —— 「答非所问」没法靠子串匹配判，需要语义理解
    groundedness  —— 「这句话在资料里有依据吗」需要逐句核对

而 groundedness 还有一条确定性捷径：上下文为空时，根本没有「依据」可言，
答案要么该拒答（那就天然有据），要么就是在编（那就天然无据）—— 两种都能直接判，
不必问判官。这条捷径覆盖了拒答题和「库里没有」的陷阱题，是整层里最省钱的优化。

判官复用的是 services/llm.py 的 chat()，不新建 client、不新增依赖。
"""

from tests.evals.graders import Judge, JudgeError, build_grade_result, contains, normalize_text
from tests.evals.schemas import (
    CheckResult,
    FinalAnswerArtifact,
    FinalAnswerCase,
    GradeResult,
    GroundednessVerdict,
    Hit,
    RelevanceVerdict,
)

# 权重。事实正确性占得最重（0.35）：用户要的是对的答案，
# 「切题但答错」比「不切题」更糟 —— 后者用户一眼能看出不对，前者会被当成结论收下。
WEIGHT_REFUSAL: float = 0.20
WEIGHT_FACTS: float = 0.35
WEIGHT_FORBIDDEN: float = 0.10
WEIGHT_RELEVANCE: float = 0.15
WEIGHT_GROUNDEDNESS: float = 0.20

# 判官评分通过线（1~5 分制取 4 分）。
RELEVANCE_PASS_SCORE: int = 4

# groundedness 通过线。判官的 score 是 0~1，这里要求 0.8：
# 允许极少量无关紧要的措辞越界，但不容忍实质性的编造。
GROUNDEDNESS_PASS_SCORE: float = 0.8

# 拒答标记。
#
# 这些措辞来自生产提示词本身：AGENT_SYSTEM_PROMPT 的规则 7 要求模型在
# 资料不足时「明确告诉用户根据当前知识库无法确定」。所以只要模型守规矩，
# 它的拒答一定会命中这里的某一条。
#
# 覆盖面刻意放宽一点（多列几种说法）：漏掉一种说法会让「该拒答却编造」
# 这条最危险的失败被判成通过，而多列一条的代价只是偶尔把一句正常的
# 「没有找到相关信息」也算作拒答 —— 后者本来就是拒答。
REFUSAL_MARKERS: tuple[str, ...] = (
    "无法确定",
    "无法回答",
    "没有找到",
    "未找到",
    "没有相关",
    "没有提及",
    "没有足够",
    "信息不足",
    "资料不足",
    "不知道",
    "cannot determine",
    "no information",
    "not enough information",
    "unable to answer",
)


def _is_refusal(answer: str) -> tuple[bool, list[str]]:
    """答案是否属于「明确拒答」，以及命中了哪些标记。"""
    normalized = normalize_text(answer)
    matched = [marker for marker in REFUSAL_MARKERS if normalize_text(marker) in normalized]
    return bool(matched), matched


def _format_contexts(hits: list[Hit]) -> str:
    """把检索结果渲染成给【判官】看的参考资料。

    刻意不复用 api/qa.py 的 build_rag_context：那个函数产出的是
    「给答题模型看的提示词片段」，服务的是「让模型照着答」；
    这里要的是「给判官看的证据」，服务的是「让判官来挑错」。
    目的不同，格式要求也不同（判官更需要看清每条资料的边界）。

    也刻意带上 chunk_id：判官指出「某句话无据」时，
    人能顺着 chunk_id 回到语料里核对，否则只能重新检索一遍。
    score 照旧不放（与 build_rag_context 的取舍一致）：
    相似度是给程序排序用的数值，对判官只会变成噪音。
    """
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(
            f"[资料 {index}]\n"
            f"chunk_id: {hit.chunk_id}\n"
            f"内容: {hit.content}"
        )
    return "\n\n".join(blocks)


# ---- 确定性检查 ----


def _check_non_empty(answer: str) -> CheckResult:
    """答案非空。权重 0：它只用来在报告里点明「模型一个字都没说」。"""
    stripped = (answer or "").strip()
    return CheckResult(
        name="non_empty",
        passed=bool(stripped),
        metric=1.0 if stripped else 0.0,
        weight=0.0,
        expected={"min_length": 1},
        actual={"length": len(stripped)},
        detail="答案非空" if stripped else "答案为空",
    )


def _check_refusal(case: FinalAnswerCase, answer: str) -> CheckResult:
    """拒答行为是否符合预期。

    两个方向都要测，缺一不可：
        must_refuse=True  而答案在编 —— 这是 RAG 场景最危险的失败，
                          错误答案会以「内部资料」的可信度被用户收下
        must_refuse=False 而答案拒答 —— 过度保守，用户拿不到本来能给的答案
    只测一个方向的话，一个永远回「无法确定」的退化实现能拿满分。
    """
    hit, matched = _is_refusal(answer)
    passed = hit if case.must_refuse else not hit

    if case.must_refuse:
        detail = f"已拒答（命中 {matched}）" if passed else "期望拒答，但答案没有给出任何拒答表述"
    else:
        detail = "未误判为拒答" if passed else f"期望正常作答，但答案拒答了（命中 {matched}）"

    return CheckResult(
        name="refusal",
        passed=passed,
        metric=1.0 if passed else 0.0,
        weight=WEIGHT_REFUSAL,
        expected={"must_refuse": case.must_refuse},
        actual={"is_refusal": hit, "matched_markers": matched},
        detail=detail,
    )


def _check_facts(case: FinalAnswerCase, answer: str) -> CheckResult | None:
    """答案里该出现的事实是否都出现了。

    数据集没声明 expected_facts 时返回 None（不产出一条 check），
    因为「没有期望事实」和「期望零个事实」是两回事 ——
    后者会让任何一句无关的话都算通过。
    """
    if not case.expected_facts:
        return None

    hits = [fact for fact in case.expected_facts if contains(answer, fact)]
    missing = [fact for fact in case.expected_facts if not contains(answer, fact)]
    metric = len(hits) / len(case.expected_facts)

    return CheckResult(
        name="facts",
        passed=not missing,
        metric=metric,
        weight=WEIGHT_FACTS,
        expected=case.expected_facts,
        actual={"matched": hits, "missing": missing},
        detail=(
            f"命中全部 {len(hits)} 项事实"
            if not missing
            else f"缺失 {len(missing)} 项：{missing}"
        ),
    )


def _check_forbidden(case: FinalAnswerCase, answer: str) -> CheckResult | None:
    """禁止出现的片段有没有出现。

    用途有三类：错误事实（「Milvus 是关系型数据库」）、
    泄漏 system prompt 的迹象、以及语料里埋的注入指令被复述出来。
    只要命中一个就直接判 0 分 —— 这三类都没有「部分正确」的余地。
    """
    if not case.forbidden:
        return None

    found = [phrase for phrase in case.forbidden if contains(answer, phrase)]

    return CheckResult(
        name="forbidden",
        passed=not found,
        metric=0.0 if found else 1.0,
        weight=WEIGHT_FORBIDDEN,
        expected=case.forbidden,
        actual={"found": found},
        detail=f"出现了禁止内容：{found}" if found else "未出现禁止内容",
    )


# ---- LLM Judge ----


async def _check_relevance(
    case: FinalAnswerCase, answer: str, judge: Judge | None
) -> tuple[CheckResult, dict[str, object] | None]:
    """答案是否切题地回应了问题。需要语义理解，所以交给判官。"""
    if not case.judge.relevance:
        return (
            CheckResult(
                name="relevance",
                passed=None,
                metric=None,
                weight=WEIGHT_RELEVANCE,
                detail="数据集未开启 relevance 判官，跳过",
            ),
            None,
        )

    if judge is None or not judge.enabled:
        return (
            CheckResult(
                name="relevance",
                passed=None,
                metric=None,
                weight=WEIGHT_RELEVANCE,
                detail="判官已被禁用（--no-judge），跳过",
            ),
            None,
        )

    prompt = (
        "请判断下面这个回答是否切题地回应了用户的问题。\n\n"
        f"用户问题：\n{case.question}\n\n"
        f"待评估的回答：\n{answer}\n\n"
        "评分标准（1~5 分）：\n"
        "5 = 直接、完整地回应了问题\n"
        "4 = 回应了问题，但有小的遗漏或少量无关内容\n"
        "3 = 部分相关，答非所问的成分明显\n"
        "2 = 基本没有回应问题\n"
        "1 = 完全无关，或在回答另一个问题\n\n"
        '请只输出如下 JSON：{"score": <1~5 的整数>, "reason": "<一句话说明>"}'
    )

    verdict = await judge.ask(prompt, RelevanceVerdict)
    # 归一化到 [0,1]：1 分 -> 0.0，5 分 -> 1.0。
    # 判官的原始分留在 judge 里，便于人工复核时看到它实际打了几分。
    metric = (verdict.score - 1) / 4

    return (
        CheckResult(
            name="relevance",
            passed=verdict.score >= RELEVANCE_PASS_SCORE,
            metric=metric,
            weight=WEIGHT_RELEVANCE,
            expected={"min_score": RELEVANCE_PASS_SCORE},
            actual={"score": verdict.score, "reason": verdict.reason},
            detail=f"判官给 {verdict.score}/5：{verdict.reason}",
        ),
        dict(verdict.model_dump()),
    )


async def _check_groundedness(
    case: FinalAnswerCase,
    artifact: FinalAnswerArtifact,
    answer_is_refusal: bool,
    judge: Judge | None,
) -> tuple[CheckResult, dict[str, object] | None]:
    """答案中的事实性陈述是否都能在检索到的资料里找到依据。

    先走确定性捷径，走不通才问判官。捷径的三种情形：

    1) 有上下文 + 该拒答却没拒答 —— 这条不算捷径，交给判官逐句核对
    2) 无上下文 + 明确拒答 —— 直接判有据。
       没有资料时正确地拒答，本身就是最标准的「有据」行为，不必花钱问判官。
    3) 无上下文 + 没拒答 + 数据集声明了 expected_facts —— 直接判无据。
       数据集说这条问题应该有事实断言，而检索结果为空、答案又没拒答，
       那它只能是在编。这是纯粹的确定性推理，判官来判也是同一个结论。
    """
    contexts = _format_contexts(artifact.hits)

    # ---- 捷径 2：无上下文 + 拒答 ----
    # 放在 judge 开关判断【之前】：这是确定性推理，和数据集有没有开启判官无关。
    # 数据集关掉判官只是想省钱/图稳定，不代表这种情况下要放弃判定。
    if not artifact.hits and answer_is_refusal:
        return (
            CheckResult(
                name="groundedness",
                passed=True,
                metric=1.0,
                weight=WEIGHT_GROUNDEDNESS,
                expected={"contexts": "non_empty_or_refusal"},
                actual={"contexts": 0, "is_refusal": True, "decided_by": "deterministic"},
                detail="检索结果为空，答案明确拒答 —— 无资料时不编造，确定性判定为有据",
            ),
            None,
        )

    # ---- 捷径 3：无上下文 + 未拒答 + 该有事实断言 ----
    if not artifact.hits and case.expected_facts:
        return (
            CheckResult(
                name="groundedness",
                passed=False,
                metric=0.0,
                weight=WEIGHT_GROUNDEDNESS,
                expected={"contexts": "non_empty"},
                actual={"contexts": 0, "is_refusal": False, "decided_by": "deterministic"},
                detail="检索结果为空、答案未拒答、但本用例期望出现事实断言 —— 确定性判定为无据",
            ),
            None,
        )

    # ---- 走到这里说明确实需要逐句核对，只能靠判官 ----
    if not case.judge.groundedness:
        return (
            CheckResult(
                name="groundedness",
                passed=None,
                metric=None,
                weight=WEIGHT_GROUNDEDNESS,
                detail="数据集未开启 groundedness 判官，跳过",
            ),
            None,
        )

    if judge is None or not judge.enabled:
        return (
            CheckResult(
                name="groundedness",
                passed=None,
                metric=None,
                weight=WEIGHT_GROUNDEDNESS,
                # 这条 detail 要说清楚「为什么跳过」：走到了需要判官的情形
                # （比如寒暄题、有资料但答案可疑），而不是数据集没配。
                detail="需要判官逐句核对，但判官已被禁用（--no-judge），跳过",
            ),
            None,
        )

    prompt = (
        "请逐句检查下面这个回答里的事实性陈述，是否都能在【参考资料】中找到依据。\n\n"
        f"用户问题：\n{case.question}\n\n"
        f"【参考资料】\n{contexts}\n\n"
        f"【待评估的回答】\n{artifact.answer}\n\n"
        "判断标准：\n"
        "- 只依据参考资料判断，不要用你自己的知识补充。\n"
        "- 回答中凡是参考资料没有支持的、具体的事实性陈述（数字、名称、结论），都算无据。\n"
        "- 复述资料原文、或对资料做合理概括，都算有据。\n"
        "- 回答如果明确表示「根据当前知识库无法确定」这类拒答，视为有据。\n"
        "- 语气词和过渡句不算事实性陈述。\n\n"
        "请只输出如下 JSON："
        '{"grounded": <true 或 false>, "score": <0~1 的小数>, '
        '"unsupported_claims": ["<无据的句子>", ...]}'
    )

    verdict = await judge.ask(prompt, GroundednessVerdict)
    passed = verdict.grounded and verdict.score >= GROUNDEDNESS_PASS_SCORE

    return (
        CheckResult(
            name="groundedness",
            passed=passed,
            metric=verdict.score,
            weight=WEIGHT_GROUNDEDNESS,
            expected={"grounded": True, "min_score": GROUNDEDNESS_PASS_SCORE},
            actual={
                "grounded": verdict.grounded,
                "score": verdict.score,
                "unsupported_claims": verdict.unsupported_claims,
                "contexts": len(artifact.hits),
                "decided_by": "judge",
            },
            detail=(
                f"判官判定有据（{verdict.score}）"
                if passed
                else f"判官判定无据（score={verdict.score}）：{verdict.unsupported_claims[:3]}"
            ),
        ),
        dict(verdict.model_dump()),
    )


async def grade(
    case: FinalAnswerCase,
    artifact: FinalAnswerArtifact,
    judge: Judge | None = None,
) -> GradeResult:
    """判定一条 final_answer 用例。

    参数：
        case:     数据集里的一行。
        artifact: runner 跑完 run_agent（并按需回放检索）之后整理出的证据。
        judge:    LLM 判官；None 或 enabled=False 时只跑确定性检查。

    返回：
        GradeResult。

    异常：
        JudgeError：判官输出无法解析。**故意让它抛出去** ——
        runner 会把它记成 invalid 而不是 failed。
        把判官失败当成通过是最坏的结果：它会系统性地高估质量，
        而且从报告的数字上完全看不出来。
    """
    answer = artifact.answer or ""
    answer_is_refusal, _ = _is_refusal(answer)

    verdicts: dict[str, object] = {}
    checks: list[CheckResult] = [_check_non_empty(answer), _check_refusal(case, answer)]

    # 三项可选检查：数据集没声明时它们返回 None，不产出 check ——
    # 这样 report 里不会出现一堆「未声明」的空条目。
    optional = [
        _check_facts(case, answer),
        _check_forbidden(case, answer),
    ]
    checks.extend(check for check in optional if check is not None)

    relevance_check, relevance_verdict = await _check_relevance(case, answer, judge)
    checks.append(relevance_check)
    if relevance_verdict:
        verdicts["relevance"] = relevance_verdict

    groundedness_check, groundedness_verdict = await _check_groundedness(
        case, artifact, answer_is_refusal, judge
    )
    checks.append(groundedness_check)
    if groundedness_verdict:
        verdicts["groundedness"] = groundedness_verdict

    return build_grade_result(
        case_id=case.id,
        layer="final_answer",
        checks=checks,
        evidence={
            "answer": answer,
            # 本轮消耗的模型调用次数。runner 靠它统计整轮评测的成本，
            # 所以每个 grader 都必须给出这个键 —— 少给一个，报告里的
            # 总调用数就会静默少算一层，而「成本被算少了」是没人会去核对的错。
            "llm_calls": len(artifact.trace.llm_calls),
            "context_chunk_ids": [hit.chunk_id for hit in artifact.hits],
            "contexts_count": len(artifact.hits),
            # 显式声明上下文是回放的，不是模型当时看到的那一份。
            # 理由见 schemas.FinalAnswerArtifact.contexts_replayed。
            "contexts_replayed": artifact.contexts_replayed,
            "trace_error": artifact.trace.error,
            "iterations": artifact.trace.iterations,
        },
        judge=verdicts or None,
    )
