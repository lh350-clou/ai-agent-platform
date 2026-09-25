"""RAG 评测：直接评检索结果本身。

**绝不通过最终答案来判断 RAG。** 这是本模块最重要的约束。

理由：一个 RAG 链路的失败可能出在两处 —— 检索没找到资料，或者找到了但模型
没用好。如果拿最终答案打分，「答案错了」会被笼统地记成「RAG 不行」，
而调优方向完全相反：检索的问题该去调切分和向量模型，生成的问题该去调提示词。
两者混在一起时，改哪边都看不出效果。所以这里只看 hits，
数据集里也刻意没有 answer 字段（见 schemas.RagCase 的说明）。

调用的是生产链路本身：embed_text + vector_store.search，
一行检索逻辑都没有重写。改天换了向量模型或检索策略，
这里评的就自动是新的那一套。

四项指标：
    hit@K        —— 前 K 条里有没有至少一条相关（答得出 / 答不出）
    recall@K     —— 前 K 条覆盖了多少标注的相关切片（覆盖得全不全）
    mrr          —— 第一条相关结果排在第几位（排序质量）
    non_empty    —— 有没有返回结果（用它区分「检索挂了」和「排序不好」）
"""

from tests.evals.graders import build_grade_result
from tests.evals.schemas import CheckResult, GradeResult, Hit, RagArtifact, RagCase

# 权重。hit@K 与 recall@K 各占大头：前者回答「能不能找到」，
# 后者回答「找得全不全」，这两个是检索质量的核心。
# mrr 只反映排序，权重给小一些 —— 排序差一点通常还能用，找不到就直接废了。
WEIGHT_HIT_AT_K: float = 0.40
WEIGHT_RECALL_AT_K: float = 0.40
WEIGHT_MRR: float = 0.20

# 固定观察档位：不管数据集声明的 top_k 是多少，都额外报这几个位置上的命中情况。
# 它的用途是对比「答案是不是就在第一条」——只看 hit@5 的话，
# 一个把正确答案排在第 5 位的检索和一个排在第 1 位的检索看起来一样好，
# 而它们的实际体验差得很远。
FIXED_K_VALUES: tuple[int, ...] = (1, 3, 5)


def _is_relevant(hit: Hit, relevant_chunk_ids: set[str], snippets: list[str]) -> bool:
    """一条结果算不算相关。

    两个口径，命中任一即可：
        chunk_id  —— 主口径。精确，靠固定 document_id 保证可复现。
        snippet   —— 兜底。语料被改写后 chunk 序号会漂移，
                     这时只有内容匹配还站得住；也用于「相关内容不止一段」的场景。
    """
    if hit.chunk_id is not None and hit.chunk_id in relevant_chunk_ids:
        return True

    content = hit.content or ""
    return any(snippet in content for snippet in snippets if snippet)


def _relevant_positions(
    hits: list[Hit], relevant_chunk_ids: set[str], snippets: list[str]
) -> list[int]:
    """相关结果的 1-based 名次，按出现顺序。"""
    return [
        index
        for index, hit in enumerate(hits, start=1)
        if _is_relevant(hit, relevant_chunk_ids, snippets)
    ]


def _hit_at(positions: list[int], k: int) -> float:
    """前 k 条里有没有相关结果。1.0 / 0.0。"""
    return 1.0 if any(position <= k for position in positions) else 0.0


def grade(case: RagCase, artifact: RagArtifact) -> GradeResult:
    """判定一条 rag 用例。

    参数：
        case:     数据集里的一行（query / top_k / 标注）。
        artifact: runner 跑完 embed_text + search 之后整理出的证据。

    返回：
        GradeResult。全程不联网（embedding 与检索已经在 adapter 里跑完了）、
        不调模型、不看任何答案文本。
    """
    hits = artifact.hits
    k = case.top_k
    positions = _relevant_positions(hits, artifact.relevant_chunk_ids, case.relevant_snippets)

    # top_k 内的命中情况。hits 不足 k 条时按实际条数算 ——
    # 用 min 而不是硬取 k，是为了让 detail 里报出的「实际 K」是诚实的。
    effective_k = min(k, len(hits)) if hits else 0

    checks: list[CheckResult] = []

    # ---- ① non_empty：不做评分，只做归因 ----
    # 权重 0：它不参与分数，因为「检索为空」的后果已经由 hit@K / recall@K
    # 各扣一次分了，再算一次是重复惩罚。
    # 它存在的唯一意义是把「Milvus 没连上」和「排序不好」分开 ——
    # 只看 hit@K=0 的话，这两种情况长得一模一样。
    checks.append(
        CheckResult(
            name="non_empty",
            passed=bool(hits),
            metric=1.0 if hits else 0.0,
            weight=0.0,
            expected={"min_hits": 1},
            actual={"hits": len(hits)},
            detail=(
                f"返回 {len(hits)} 条"
                if hits
                else "检索结果为空：请先确认 Milvus 可用、语料已入库、知识库 ID 正确"
            ),
        )
    )

    # ---- ② hit@K ----
    hit_at_k = _hit_at(positions, k)
    checks.append(
        CheckResult(
            name=f"hit@{k}",
            passed=hit_at_k == 1.0,
            metric=hit_at_k,
            weight=WEIGHT_HIT_AT_K,
            expected={"any_relevant_within": k, "relevant_chunk_ids": sorted(artifact.relevant_chunk_ids)},
            actual={"relevant_positions": positions, "effective_k": effective_k},
            detail=(
                f"前 {k} 条命中（相关结果名次 {positions[:3]}）"
                if hit_at_k == 1.0
                else f"前 {k} 条内没有相关结果"
            ),
        )
    )

    # ---- ③ recall@K ----
    # 只有给了 relevant_chunk_seqs 才算得出来；只给 snippet 时欠一个分母，
    # 与其编一个数，不如标记为 skipped。
    if artifact.relevant_chunk_ids:
        recalled = {
            hit.chunk_id
            for hit in hits[:k]
            if hit.chunk_id is not None
            and hit.chunk_id in artifact.relevant_chunk_ids
        }
        recall = len(recalled) / len(artifact.relevant_chunk_ids)
        passed = recall >= case.min_recall

        detail = f"召回 {len(recalled)}/{len(artifact.relevant_chunk_ids)}，阈值 {case.min_recall}"
        if not passed:
            missing = sorted(artifact.relevant_chunk_ids - recalled)
            detail += f"；漏掉 {missing}"
    else:
        recall = None
        passed = None
        recalled = set()
        detail = "数据集只给了 relevant_snippets，没有精确分母，无法计算 Recall@K"

    checks.append(
        CheckResult(
            name=f"recall@{k}",
            passed=passed,
            metric=recall,
            weight=WEIGHT_RECALL_AT_K,
            expected={
                "relevant_chunk_ids": sorted(artifact.relevant_chunk_ids),
                "min_recall": case.min_recall,
            },
            actual={"recalled": sorted(recalled)},
            detail=detail,
        )
    )

    # ---- ④ mrr：只看排序质量，不参与通过与否 ----
    # passed=None（skipped）是有意的：MRR 衡量的是「排得好不好」，
    # 而一条用例该不该通过，取决于「找没找到、找全没找全」。
    # 让排序质量直接决定通过与否，会把「检索可用但排序一般」判成失败。
    mrr = 1.0 / positions[0] if positions else 0.0
    checks.append(
        CheckResult(
            name="mrr",
            passed=None,
            metric=mrr,
            weight=WEIGHT_MRR,
            expected={"first_relevant_rank": 1},
            actual={"first_relevant_rank": positions[0] if positions else None},
            detail=(
                f"第一条相关结果排在第 {positions[0]} 位"
                if positions
                else "没有任何相关结果，MRR 记 0"
            ),
        )
    )

    # ---- ⑤ 固定档位：只呈现，不评分 ----
    for fixed_k in FIXED_K_VALUES:
        # top_k 比 fixed_k 小时，这个档位没有意义（检索本来就没取那么多条），
        # 直接跳过而不是按实际条数算 —— 后者会把 hit@5 混进 hit@3 里。
        if fixed_k > k:
            checks.append(
                CheckResult(
                    name=f"hit@{fixed_k}",
                    passed=None,
                    metric=None,
                    weight=0.0,
                    detail=f"top_k={k} 小于 {fixed_k}，本档位不适用",
                )
            )
            continue

        value = _hit_at(positions, fixed_k)
        checks.append(
            CheckResult(
                name=f"hit@{fixed_k}",
                passed=None,
                metric=value,
                weight=0.0,
                expected={"any_relevant_within": fixed_k},
                actual={"relevant_positions": positions},
                detail="命中" if value == 1.0 else "未命中",
            )
        )

    return build_grade_result(
        case_id=case.id,
        layer="rag",
        checks=checks,
        evidence={
            "query": artifact.query,
            "top_k": artifact.top_k,
            # 只记 chunk_id / document_id / score，不记 content：
            # 报告要能进版本库、能贴进 issue，正文可能很长且属于语料内容，
            # 逐条抄一份进报告没有额外价值（要看正文可以直接看语料文件）。
            "hits": [
                {
                    "chunk_id": hit.chunk_id,
                    "document_id": hit.document_id,
                    "score": hit.score,
                }
                for hit in hits
            ],
            "relevant_chunk_ids": sorted(artifact.relevant_chunk_ids),
            "relevant_snippets": case.relevant_snippets,
        },
    )
