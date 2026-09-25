"""Evaluation V1 的命令行入口。

跑法（必须在 backend/ 目录下，用 -m 而不是直接跑文件，
这样 tests.evals 才是一个完整的包、import app.* 才成立）：

    cd backend
    python -m tests.evals.runner --layer all
    python -m tests.evals.runner --layer rag
    python -m tests.evals.runner --layer tool_calling --limit 10
    python -m tests.evals.runner --layer final_answer --no-judge --limit 10

本模块是【唯一的编排处】。它做四件事：
    1. 加载数据集并校验（坏用例带着行号立刻报错，绝不静默跳过）
    2. 按固定的 uuid5 ID 重建评测语料（先清空再入库，保证可重复）
    3. 把每一层跑成证据（Agent / 检索 / Agent+回放），交给对应的 grader
    4. 汇总、打印、落盘 reports/<run_id>.json

三层各自的 adapter 都在这里，因为「怎么把生产链路跑出证据」是编排问题，
不是判定问题。判定逻辑全在 graders/ 下，那些模块不联网、可离线单测。

关于「评测与生产解耦」：本模块只 import app.services.* 和 app.models.message，
app/ 下没有任何一处反过来 import 本模块。评测跑的是真实的生产链路
（run_agent / embed_text / vector_store.search / ingest_txt），
一行检索或 Agent 逻辑都没有重写。
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from app.core.config import settings
from app.models.message import Message
from app.services import vector_store
from app.services.agent import (
    DEFAULT_TOOL_TOP_K,
    SEARCH_TOOL_NAME,
    AgentResult,
    run_agent,
)
from app.services.document.ingest import build_chunk_id, ingest_txt
from app.services.embedding import embed_text
from app.services.llm import model_name
from app.services.trace import format_error
from tests.evals.graders import Judge, JudgeError
from tests.evals.graders import final_answer as final_answer_grader
from tests.evals.graders import rag as rag_grader
from tests.evals.graders import tool_calling as tool_calling_grader
from tests.evals.schemas import (
    CaseResult,
    CorpusSeedInfo,
    EvalReport,
    FinalAnswerArtifact,
    FinalAnswerCase,
    Hit,
    HistoryTurn,
    LayerSummary,
    RagArtifact,
    RagCase,
    TagSummary,
    ToolCallingArtifact,
    ToolCallingCase,
    corpus_filename,
    eval_document_id,
    eval_kb_id,
)

logger = logging.getLogger(__name__)

# 层名 -> 数据集文件名。
# 写死映射，不允许「随便指一个 jsonl」：用 tool_calling 的 grader 去评
# rag 的数据集只会产出一堆无意义的红，而那种红看起来像模型退化。
LAYER_DATASETS: dict[str, str] = {
    "tool_calling": "tool_calling.jsonl",
    "rag": "rag.jsonl",
    "final_answer": "final_answer.jsonl",
}

# 需要跑 Agent 的层（相对地，rag 层只跑检索，不碰模型）。
AGENT_LAYERS: frozenset[str] = frozenset({"tool_calling", "final_answer"})

EVAL_DIR: Path = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR: Path = EVAL_DIR / "datasets"
DEFAULT_REPORT_DIR: Path = EVAL_DIR / "reports"
FIXTURE_DIR: Path = EVAL_DIR / "fixtures"

# --no-mcp 时替换成的模块名。它当然不存在 —— 目的就是让 mcp_client 起子进程
# 失败，从而走 agent._open_mcp 里那条「MCP 不可用就降级成只有内置工具」的
# 既定分支。用配置去触发降级，而不是 monkeypatch 生产代码的内部函数：
# 后者等于评测自己改被测对象，跑出来的结论说明不了生产环境的行为。
NO_MCP_MODULE: str = "tests.evals.__no_such_mcp_module__"

CASE_MODEL: dict[str, type] = {
    "tool_calling": ToolCallingCase,
    "rag": RagCase,
    "final_answer": FinalAnswerCase,
}


# ---- 数据集加载 ----


def _load_cases(layer: str, dataset_dir: Path) -> list[Any]:
    """加载并校验一层的全部用例。

    坏用例【立刻抛错】并带上行号，不做「跳过坏行继续跑」。
    静默跳过会让通过率虚高，而且没人会注意到有几条用例根本没跑 ——
    对一个用来判断「系统有没有变好」的工具来说，这是最不可接受的失败方式。
    """
    path = dataset_dir / LAYER_DATASETS[layer]
    if not path.exists():
        raise FileNotFoundError(f"数据集不存在：{path}")

    model = CASE_MODEL[layer]
    cases: list[Any] = []
    seen_ids: set[str] = set()

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = model.model_validate(json.loads(line))
        except Exception as exc:
            raise ValueError(f"{path.name} 第 {line_no} 行不是合法的用例：{exc}") from exc

        # 语料名写错时在这里就报错。等到 seeding 阶段才发现的话，
        # 报出来的是「找不到 fixture 文件」，离真正的原因（数据集里少写了个字母）很远。
        try:
            corpus_filename(case.corpus)
        except ValueError as exc:
            raise ValueError(f"{path.name} 第 {line_no} 行：{exc}") from exc

        if case.id in seen_ids:
            raise ValueError(f"{path.name} 第 {line_no} 行：用例 id 重复：{case.id}")
        seen_ids.add(case.id)
        cases.append(case)

    return cases


def _select_cases(cases: list[Any], args: argparse.Namespace) -> list[Any]:
    """按 --case / --tag / --limit 过滤。"""
    selected = cases

    if args.case:
        wanted = set(args.case)
        selected = [case for case in selected if case.id in wanted]
        # 显式指出「你点名的用例没找到」，而不是让用户对着空结果猜。
        missing = wanted - {case.id for case in selected}
        if missing:
            raise ValueError(f"数据集里没有这些用例 id：{sorted(missing)}")

    if args.tag:
        wanted_tags = set(args.tag)
        selected = [case for case in selected if wanted_tags & set(case.tags)]

    if args.limit > 0:
        selected = selected[: args.limit]

    return selected


# ---- 语料准备（seeding）----


async def _seed_corpus(corpus: str, print_chunks: bool) -> CorpusSeedInfo:
    """把一个评测语料重建进 Milvus。

    「先清空再入库」不是可选的优化：vector_store.insert_chunk 是插入而不是
    upsert，重复运行会让同一个 chunk_id 在集合里出现多份，
    Recall@K 的分母随之失真（明明召回 1 条，却算出 0.5）。
    清一次的开销远小于让评测悄悄算错。

    kb_id 与 document_id 都由 uuid5 固定，所以每台机器、每次运行
    得到的 chunk_id 完全一致 —— 这是数据集里能写 chunk 序号的前提。
    """
    kb_id = str(eval_kb_id(corpus))
    document_id = str(eval_document_id(corpus))
    fixture = FIXTURE_DIR / corpus_filename(corpus)

    if not fixture.exists():
        raise FileNotFoundError(f"评测语料不存在：{fixture}")

    # 清理上一轮的残留。首次运行时返回 0（库里本来就什么都没有），不是错误。
    removed = await vector_store.delete_by_knowledge_base_id(kb_id)
    if removed:
        logger.info("已清理语料 %s 的旧向量：%d 条", corpus, removed)

    chunks = await ingest_txt(
        file_path=fixture,
        knowledge_base_id=kb_id,
        document_id=document_id,
    )

    if print_chunks:
        _print_chunks(corpus, document_id, chunks)

    return CorpusSeedInfo(
        corpus=corpus,
        kb_id=kb_id,
        document_id=document_id,
        chunks=chunks,
        taskset=str(fixture),
    )


def _print_chunks(corpus: str, document_id: str, count: int) -> None:
    """打印每个 chunk 的序号与前 60 字。

    用途是人工核对数据集里的 relevant_chunk_seqs 有没有指对位置 ——
    语料一旦改动，切分序号会整体漂移，而漂移不会报错，
    只会让 Recall 悄悄变成 0，看起来像检索退化。
    这里必须重新切一遍（而不是从 Milvus 里查回来）：Milvus 里只有向量和正文，
    序号要靠 build_chunk_id 反推，重新切分是唯一可靠的对照方式。
    """
    from app.services.document.parser import parse_txt
    from app.services.document.splitter import split_text

    text = parse_txt(FIXTURE_DIR / corpus_filename(corpus))
    pieces = split_text(text)
    print(f"  [chunks] {corpus}: {count} 条已入库，逐条预览如下（序号用于数据集标注）")
    for index, piece in enumerate(pieces, start=1):
        preview = piece[:60].replace("\n", " ")
        print(f"    seq {index:02d} {build_chunk_id(document_id, index)[-13:]} | {preview}")


# ---- Adapter：把生产链路跑成证据 ----


def _build_history(turns: Sequence[HistoryTurn], conversation_id: uuid.UUID) -> list[Message]:
    """把数据集里的历史消息转成 Message 对象。

    只构造对象、不落库：history_to_messages 只读 role 和 content 两个属性，
    根本不需要数据库。conversation_id 是必填外键，这里填 kb_id 占位 ——
    它不会参与任何判断，只是让模型构造合法。
    """
    return [
        Message(conversation_id=conversation_id, role=turn.role, content=turn.content)
        for turn in turns
    ]


async def _run_agent_case(case: Any, kb_id: uuid.UUID) -> AgentResult:
    """跑一次真实的 Agent。tool_calling 与 final_answer 两层共用。"""
    history = _build_history(case.history, kb_id) if case.history else None
    return await run_agent(
        question=case.question,
        knowledge_base_id=kb_id,
        history=history,
    )


async def _run_rag_case(case: RagCase, kb_id: uuid.UUID) -> RagArtifact:
    """跑检索链路：embed_text + vector_store.search。

    【完全不经过 Agent、不看任何答案】。理由见 graders/rag.py 的模块说明：
    检索的问题和生成的问题必须分开归因，否则调优方向会被带偏。
    """
    query_vector = await embed_text(case.query)
    raw_hits = await vector_store.search(
        knowledge_base_id=str(kb_id),
        query_vector=query_vector,
        top_k=case.top_k,
    )

    document_id = str(eval_document_id(case.corpus))
    return RagArtifact(
        query=case.query,
        top_k=case.top_k,
        hits=[Hit.model_validate(hit) for hit in raw_hits],
        relevant_chunk_ids={
            build_chunk_id(document_id, seq) for seq in case.relevant_chunk_seqs
        },
    )


async def _replay_contexts(result: AgentResult, kb_id: uuid.UUID) -> list[Hit]:
    """用模型实际用过的查询参数，把检索结果重新取一遍。

    为什么要「回放」而不是直接拿：services/trace.py 刻意不记录工具返回内容
    （它的安全约定明确要求不抄一份用户数据），所以 AgentResult 里没有任何
    出口能拿到「答案依据的那批资料」。而 groundedness 判定必须要这个上下文。

    回放是等价的：search 是确定性检索，同一份语料 + 同一个查询向量 + 同一个
    top_k 一定得到同一批结果，而语料在同一轮评测里不会变。
    这个事实在报告里由 contexts_replayed 字段显式标注，
    免得有人把回放的上下文当成「模型当时看到的那一份」。

    只回放【成功】的调用：失败调用的 arguments 是空的（见 _execute_search_tool
    里 arguments 的赋值时机），拿它们去检索只会查到一个空字符串。
    """
    queries: list[tuple[str, int]] = []
    for call in result.trace.tool_calls:
        if call.tool != SEARCH_TOOL_NAME or not call.success:
            continue
        query = call.arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            continue
        top_k = call.arguments.get("top_k")
        queries.append((query, top_k if isinstance(top_k, int) else DEFAULT_TOOL_TOP_K))

    if not queries:
        # 模型压根没检索（寒暄、MCP 工具题等）。返回空上下文是正确的结果，
        # 不是错误 —— grader 的确定性捷径会正确处理这种情况。
        return []

    merged: dict[str, Hit] = {}
    for query, top_k in queries:
        query_vector = await embed_text(query)
        raw_hits = await vector_store.search(
            knowledge_base_id=str(kb_id),
            query_vector=query_vector,
            top_k=top_k,
        )
        for raw in raw_hits:
            hit = Hit.model_validate(raw)
            # 按 chunk_id 去重：模型改写重查时，两次检索很可能命中同一批切片，
            # 不去重会让同一份资料在判官眼里出现多次，看起来像「有多个来源支持」。
            key = hit.chunk_id or f"__anonymous_{len(merged)}"
            merged.setdefault(key, hit)

    return list(merged.values())


# ---- 单条用例的执行 + 判定 ----


async def _run_case(
    layer: str,
    case: Any,
    judge: Judge | None,
) -> CaseResult:
    """跑一条用例并判定。任何异常都转成 invalid，不往上抛。

    不往上抛是刻意的：一条用例因为网络抖动失败，不该让后面几十条都不跑 ——
    那样一次抖动就会毁掉整轮评测，而且报告里只剩一个异常，什么信息都没有。
    """
    kb_id = eval_kb_id(case.corpus)
    started = time.perf_counter()

    try:
        if layer == "tool_calling":
            result = await _run_agent_case(case, kb_id)
            grade = tool_calling_grader.grade(
                case,
                ToolCallingArtifact(
                    trace=result.trace,
                    executed=result.tool_calls,
                    answer=result.answer,
                ),
            )
            # 耗时用 Trace 记的，不自己计时：两边都算一遍迟早会对不上，
            # 而且 Trace 的 total_duration_ms 才是生产环境里真正被观测到的那个数。
            duration_ms = result.trace.total_duration_ms

        elif layer == "rag":
            artifact = await _run_rag_case(case, kb_id)
            grade = rag_grader.grade(case, artifact)
            # 检索层没有 Trace，只能自己计时。
            duration_ms = round((time.perf_counter() - started) * 1000, 2)

        elif layer == "final_answer":
            result = await _run_agent_case(case, kb_id)
            hits = await _replay_contexts(result, kb_id)
            grade = await final_answer_grader.grade(
                case,
                FinalAnswerArtifact(answer=result.answer, trace=result.trace, hits=hits),
                judge,
            )
            duration_ms = result.trace.total_duration_ms

        else:
            raise ValueError(f"未知的评测层：{layer}")

    except JudgeError as exc:
        # 判官坏了 ≠ 模型答错了。判官失败必须记成 invalid，
        # 一旦被当成 failed（或者更糟，当成 passed），报告就会系统性失真。
        return CaseResult(
            case_id=case.id,
            layer=layer,
            tags=case.tags,
            status="invalid",
            passed=False,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            error=str(exc),
        )
    except Exception as exc:
        logger.debug("用例 %s 执行失败", case.id, exc_info=True)
        return CaseResult(
            case_id=case.id,
            layer=layer,
            tags=case.tags,
            status="invalid",
            passed=False,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            # format_error 只取「异常类型 + 消息」，不带堆栈 ——
            # 报告可能被贴进 issue 或提交进仓库，堆栈里的本机路径不该跟着走。
            error=format_error(exc),
        )

    return CaseResult(
        case_id=case.id,
        layer=layer,
        tags=case.tags,
        status="passed" if grade.passed else "failed",
        passed=grade.passed,
        score=grade.score,
        duration_ms=duration_ms,
        grade=grade,
    )


async def _run_layer(
    layer: str,
    cases: list[Any],
    judge: Judge | None,
    concurrency: int,
) -> list[CaseResult]:
    """跑完一层。默认串行。"""
    if concurrency <= 1:
        results: list[CaseResult] = []
        for case in cases:
            result = await _run_case(layer, case, judge)
            results.append(result)
            _print_case_line(result)
        return results

    # 并发模式：用信号量限流，而不是无脑 gather。
    # 评测打的是真实的付费接口，不限流很容易自己把自己打到 429，
    # 于是整轮评测变成一堆 invalid，白花钱。
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(case: Any) -> CaseResult:
        async with semaphore:
            result = await _run_case(layer, case, judge)
        _print_case_line(result)
        return result

    return list(await asyncio.gather(*(guarded(case) for case in cases)))


# ---- 汇总 ----


def _summarize(layer: str, results: list[CaseResult]) -> LayerSummary:
    """算一层的通过率与各 metric 均值。"""
    total = len(results)
    passed = sum(1 for r in results if r.status == "passed")
    failed = sum(1 for r in results if r.status == "failed")
    invalid = sum(1 for r in results if r.status == "invalid")

    # pass_rate 的【分母排除 invalid】：invalid 的含义是「这次没测出来」
    # （接口挂了、判官输出看不懂），不是「模型不行」。把它算进分母，
    # 等于让一次网络抖动看起来像能力退化 —— 那就违背了把 invalid 单列出来的初衷。
    measurable = total - invalid
    pass_rate = passed / measurable if measurable > 0 else 0.0

    # metric 按 check 名聚合。名里带工具/字段名的那些（arguments:xxx.query）
    # 会各自成为一项 —— 这正好回答「是哪个参数在错」，比一个笼统的平均值有用。
    buckets: dict[str, list[float]] = {}
    for result in results:
        if result.grade is None:
            continue
        for check in result.grade.checks:
            if check.metric is None:
                continue
            buckets.setdefault(check.name, []).append(check.metric)

    metrics = {
        name: round(sum(values) / len(values), 4)
        for name, values in sorted(buckets.items())
    }

    return LayerSummary(
        layer=layer,
        total=total,
        passed=passed,
        failed=failed,
        invalid=invalid,
        pass_rate=round(pass_rate, 4),
        metrics=metrics,
    )


def _summarize_tags(results: list[CaseResult]) -> dict[str, TagSummary]:
    """按标签分组统计，让失败可归因。

    不是 dashboard（那是后续阶段的事），只是为了让「寒暄类全过、多跳类全挂」
    这种结论能直接从报告里看出来 —— 只看总通过率是看不出来的。
    """
    summary: dict[str, TagSummary] = {}
    for result in results:
        for tag in result.tags or ["<untagged>"]:
            entry = summary.setdefault(tag, TagSummary())
            entry.total += 1
            if result.status == "passed":
                entry.passed += 1
            elif result.status == "failed":
                entry.failed += 1
            else:
                entry.invalid += 1
    return dict(sorted(summary.items()))


# ---- 输出 ----


def _print_case_line(result: CaseResult) -> None:
    """一条用例一行。失败时把第一个没过的 check 打出来，省得去翻 JSON。"""
    marker = {"passed": "PASS", "failed": "FAIL", "invalid": "INVALID"}[result.status]
    line = f"  {marker:>7}  {result.case_id:<10} score={result.score:<6} {result.duration_ms:>8.0f}ms"

    if result.status == "invalid":
        line += f"  {result.error}"
    elif result.status == "failed" and result.grade is not None:
        bad = next(
            (c for c in result.grade.checks if c.passed is False),
            None,
        )
        if bad is not None:
            line += f"  [{bad.name}] {bad.detail}"

    print(line)


def _print_summary(report: EvalReport) -> None:
    """控制台汇总表。"""
    print("\n" + "=" * 78)
    print("Evaluation V1 汇总")
    print("=" * 78)
    print(f"run_id      : {report.run_id}")
    print(f"model       : {report.model}")
    print(f"judge       : {report.judge_model}{'' if report.judge_enabled else '（已禁用）'}")
    print(f"mcp         : {'启用' if report.mcp_enabled else '禁用'}")
    print(f"llm calls   : {report.total_llm_calls}（不含判官 {report.judge_calls} 次）")

    header = f"\n{'layer':<14}{'total':>6}{'pass':>6}{'fail':>6}{'inval':>7}{'pass_rate':>11}"
    print(header)
    print("-" * len(header))
    for layer in report.layers:
        print(
            f"{layer.layer:<14}{layer.total:>6}{layer.passed:>6}"
            f"{layer.failed:>6}{layer.invalid:>7}{layer.pass_rate:>11.2%}"
        )

    for layer in report.layers:
        if not layer.metrics:
            continue
        print(f"\n[{layer.layer}] 各检查项均值")
        for name, value in layer.metrics.items():
            print(f"    {name:<42} {value:.3f}")

    if report.by_tag:
        print("\n按标签")
        for tag, entry in report.by_tag.items():
            print(
                f"    {tag:<20} {entry.passed}/{entry.total} 过"
                f"（失败 {entry.failed}，无效 {entry.invalid}）"
            )


# ---- 主流程 ----


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tests.evals.runner",
        description="Evaluation V1：Tool Calling / RAG / Final Answer 三层评测",
    )
    parser.add_argument(
        "--layer",
        choices=[*LAYER_DATASETS, "all"],
        default="all",
        help="跑哪一层，默认 all",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="ID",
        help="只跑指定用例 id，可重复",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="TAG",
        help="只跑带该标签的用例，可重复（命中任一即可）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="每层最多跑前 N 条，0 表示全部（调试省钱用）",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="关闭 MCP（把 MCP_SERVER_MODULE 指向不存在的模块，触发生产代码里的降级分支）",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="禁用 LLM 判官，只跑确定性检查",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="同一层内的并发用例数，默认 1（串行，避免自触发限流）",
    )
    parser.add_argument(
        "--keep-vectors",
        action="store_true",
        help="跑完不清理评测语料的向量（默认清理）",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        help="任一层的 pass_rate 低于该值时以退出码 1 结束（给 CI 用），默认 0（不失败）",
    )
    parser.add_argument(
        "--print-chunks",
        action="store_true",
        help="入库后打印每个 chunk 的序号与前 60 字，用于核对数据集的 relevant_chunk_seqs",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"数据集目录，默认 {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"报告输出目录，默认 {DEFAULT_REPORT_DIR}",
    )
    return parser.parse_args(argv)


async def _run(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    started_at = datetime.now(timezone.utc)

    if args.no_mcp:
        # 走生产代码自己的降级分支（agent._open_mcp 里 try/except 那一段），
        # 而不是 monkeypatch 内部函数 —— 评测要观察的是生产行为。
        settings.MCP_SERVER_MODULE = NO_MCP_MODULE
        # MCP 连不上会走 logger.exception 打一整段堆栈。这是预期内的失败，
        # 不打出来只会让输出看不清真正的问题。降级的行为本身不受影响。
        logging.getLogger("app.services.agent").setLevel(logging.CRITICAL)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    layers = list(LAYER_DATASETS) if args.layer == "all" else [args.layer]

    # ---- 1. 加载 + 过滤 ----
    per_layer_cases: dict[str, list[Any]] = {}
    for layer in layers:
        cases = _select_cases(_load_cases(layer, args.dataset_dir), args)
        if not cases:
            print(f"[{layer}] 没有符合条件的用例，跳过")
            continue
        per_layer_cases[layer] = cases

    if not per_layer_cases:
        print("没有可执行的用例。请检查 --layer / --case / --tag 是否写对了。")
        return 2

    # ---- 2. Seeding ----
    # 只给真正用到的语料建向量。三层都要：tool_calling 的用例大多期望模型去检索，
    # 语料不在库里的话检索会返回空，模型只能答「无法确定」，那样评的就不是
    # 工具调用能力了。
    corpora = sorted({case.corpus for cases in per_layer_cases.values() for case in cases})
    seeding: list[CorpusSeedInfo] = []

    print(f"准备评测语料：{', '.join(corpora)}")
    for corpus in corpora:
        info = await _seed_corpus(corpus, args.print_chunks)
        seeding.append(info)
        print(f"  {corpus}: kb_id={info.kb_id} chunks={info.chunks}")

    judge = Judge(enabled=not args.no_judge)
    results: list[CaseResult] = []

    try:
        # ---- 3. 逐层执行 ----
        for layer, cases in per_layer_cases.items():
            print(f"\n[{layer}] {len(cases)} 条用例")
            results.extend(await _run_layer(layer, cases, judge, args.concurrency))
    finally:
        # ---- 5. 清理 ----
        # 放在 finally：评测中途崩了同样要清。残留的向量虽然不会被任何真实
        # 查询命中（search 强制按 kb 过滤），但白占存储，而且下次运行还得再清一遍。
        if not args.keep_vectors:
            for info in seeding:
                try:
                    removed = await vector_store.delete_by_knowledge_base_id(info.kb_id)
                    logger.info("已清理语料 %s 的向量：%d 条", info.corpus, removed)
                except Exception:
                    # 清理失败只记日志，不改变退出码：真正的评测结论不该
                    # 被一个善后动作的失败顶掉（和 ingest.py 里 _discard_partial 一个道理）。
                    logger.warning("清理语料 %s 的向量失败，Milvus 中可能残留", info.corpus)

    # ---- 4. 汇总 + 落盘 ----
    summaries = [
        _summarize(layer, [r for r in results if r.layer == layer])
        for layer in per_layer_cases
    ]
    finished_at = datetime.now(timezone.utc)

    report = EvalReport(
        run_id=f"{started_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:4]}",
        started_at=started_at,
        finished_at=finished_at,
        model=model_name(),
        judge_model=judge.model,
        judge_enabled=judge.enabled,
        mcp_enabled=not args.no_mcp,
        seeding=seeding,
        layers=summaries,
        by_tag=_summarize_tags(results),
        # 借 Trace 已有的 llm_calls 直接统计模型调用次数 —— 这本来就是
        # 「这一轮花了多少钱」最直接的代理指标。
        # 用 .get(..., 0) 逐条累加而不是按层特判：grader 那边漏给这个键时，
        # 这里会安静地少算，所以每个 grader 的 evidence 里都显式写了 llm_calls。
        total_llm_calls=sum(
            int(r.grade.evidence.get("llm_calls", 0))
            for r in results
            if r.grade is not None
        ),
        judge_calls=judge.calls,
        cases=results,
    )

    _print_summary(report)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"{report.run_id}.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n报告已写入：{report_path}")

    # ---- 6. 退出码 ----
    invalid_total = sum(layer.invalid for layer in summaries)
    if invalid_total:
        # 单独提示一句。invalid 得看，但不该自动让 CI 变红 ——
        # 它多半是环境问题（限流、服务没起），不是这次改动的问题。
        print(f"注意：有 {invalid_total} 条用例为 invalid（环境或判官问题，已从 pass_rate 分母中排除）")

    if args.fail_under > 0:
        below = [layer for layer in summaries if layer.pass_rate < args.fail_under]
        if below:
            print(
                f"pass_rate 低于阈值 {args.fail_under}："
                + ", ".join(f"{layer.layer}={layer.pass_rate:.2%}" for layer in below)
            )
            return 1

    return 0


def main() -> None:
    # Windows 控制台默认不是 UTF-8，中文输出会变成乱码。
    # 报告文件始终是 UTF-8 写的，这里只是让控制台别花屏。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
