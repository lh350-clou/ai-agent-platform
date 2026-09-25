"""Evaluation V1 的用例模型与结果模型。

这个模块只放【数据形状】，不放任何判定逻辑：
    用例模型   —— datasets/*.jsonl 里每一行长什么样（人工标注）
    Artifact   —— 跑完生产链路之后拿到的证据长什么样
    结果模型   —— 判定结果与报告长什么样

为什么要用 Pydantic 定义数据集，而不是直接读 dict：
jsonl 是手写的，字段名打错（写成 expected_tool 而不是 tools）在运行时不会报错，
只会让那条断言变成「永远通过」。Pydantic 会把这种错误在【加载阶段】就拦下来，
并且能报出是第几行、哪个字段 —— 这是手工维护的评测集最容易出的问题。

所有 id 派生函数都集中在这里，是为了让「kb_id 怎么来的」只有一个实现处：
runner 用它做 seeding，数据集标注靠它保证可复现，两处一旦分叉就会出现
「标注指向的 chunk 根本不在被检的库里」，而那种错误表现为分数为 0，
看起来像检索退化，实际是评测自己错了。
"""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.services.agent import ToolCallRecord
from app.services.trace import Trace

# ---- 评测语料的 id 派生 ----


# 用 uuid5（基于命名空间的确定性 uuid）而不是 uuid4：
# 评测语料的知识库 ID 必须【每次运行、每台机器都一样】，否则
# 数据集里标注的 chunk_id 就无法预先算出来。这是整套 RAG 评测成立的前提。
EVAL_ID_NAMESPACE: uuid.UUID = uuid.NAMESPACE_URL

# 语料名 -> fixtures/ 下的文件名。
# 数据集里只写语料名（"product_docs"），不写文件名也不写路径 ——
# 文件名和磁盘布局是 runner 的实现细节，不该漏进人工维护的标注里。
CORPUS_FILES: dict[str, str] = {
    "product_docs": "product_docs.txt",
    "engineering_notes": "engineering_notes.txt",
}


def corpus_filename(corpus: str) -> str:
    """语料名 -> fixtures 下的文件名。名字不认识时给出可用的取值。"""
    try:
        return CORPUS_FILES[corpus]
    except KeyError:
        known = ", ".join(sorted(CORPUS_FILES))
        raise ValueError(f"未知的评测语料 {corpus!r}；可用的有：{known}") from None


def eval_kb_id(corpus: str) -> uuid.UUID:
    """评测语料使用的知识库 ID。

    刻意【不】在 PostgreSQL 里建这个知识库：
    vector_store.search 只按 knowledge_base_id 过滤，Milvus 里每一行都带着它，
    所以检索链路完全不需要那条业务记录。不建记录 = 不落库 = 不动 schema。
    """
    return uuid.uuid5(EVAL_ID_NAMESPACE, f"eval-kb:{corpus}")


def eval_document_id(corpus: str) -> uuid.UUID:
    """评测语料使用的文档 ID。

    它决定了所有 chunk_id：ingest.build_chunk_id 的产物是
    f"{document_id}-chunk-{序号:06d}"，文档 ID 一固定，序号就能写进数据集。
    """
    filename = corpus_filename(corpus)
    return uuid.uuid5(EVAL_ID_NAMESPACE, f"eval-doc:{corpus}:{filename}")


# ---- 共用小模型 ----


class Hit(BaseModel):
    """一条检索结果，字段与 vector_store.search 的返回严格一致。

    chunk_id / content 允许为 None，是因为 vector_store._normalize_hits 用的是
    entity.get(...)：Milvus 少返回一个字段时它给 None 而不是报错。
    这里如实表达这一点，而不是假装它们一定有值 ——
    评测工具把「数据有洞」伪装成「数据正常」是最坏的做法。
    """

    chunk_id: str | None = None
    document_id: str | None = None
    content: str | None = None
    score: float | None = None


class HistoryTurn(BaseModel):
    """数据集里的一条历史消息（仅多轮用例使用）。

    角色只允许 user / assistant：这和 services/conversation.py 的
    ALLOWED_HISTORY_ROLES 是同一条规则。让 system 出现在标注里，
    等于在数据集层面开了一个注入入口。
    """

    role: Literal["user", "assistant"]
    content: str


# ---- 用例模型：Tool Calling ----


class ArgMatcher(BaseModel):
    """一个工具参数的匹配器。

    为什么要做成「匹配器」而不是直接写期望值：
    期望 query 和模型实际生成的 query 不可能逐字相同（它是模型自己组织的措辞）。
    硬要求全等，评的就不是「参数对不对」而是「模型会不会恰好写出我这句话」，
    那种评测只会一直红。所以这里提供几种更贴近意图的判定方式。
    """

    match: Literal["exact", "any_of", "contains_all", "lte", "gte", "range"]
    value: Any = None
    values: list[str] = Field(default_factory=list)
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _require_fields_for_match(self) -> "ArgMatcher":
        """校验每种匹配方式需要的字段都给了。

        必须在这里拦，而不是等 grader 运行时判失败：
        漏写 value 的匹配器在运行时会「永远匹配不上」，看起来像模型参数传错了，
        实际是数据集自己写坏了 —— 这种错误在报告里表现为一个稳定的红，
        会让人花几个小时去查模型，而真正的问题在一行 JSON 里。
        """
        if self.match in ("exact", "lte", "gte") and self.value is None:
            raise ValueError(f"match={self.match!r} 需要提供 value")
        if self.match in ("any_of", "contains_all") and not self.values:
            raise ValueError(f"match={self.match!r} 需要提供非空的 values")
        if self.match == "range" and (self.min is None or self.max is None):
            raise ValueError("match='range' 需要同时提供 min 和 max")
        return self


class ToolExpectation(BaseModel):
    """一条 tool_calling 用例的期望。"""

    # 期望模型【请求】的工具集合，顺序无关。空列表表示期望不调用任何工具
    # （寒暄类问题）—— 这是 Agent 相对固定 RAG 的核心卖点，值得专门有用例。
    tools: list[str] = Field(default_factory=list)

    # 实际【执行】的调用次数区间。
    # 缺省是严格的等值（min == max == len(tools)）；需要容忍模型改写重查时
    # 由数据集显式放宽，而不是让 grader 自作主张地宽容。
    min_calls: int | None = None
    max_calls: int | None = None

    # 按工具名分组的参数匹配器：{"search_knowledge_base": {"top_k": ArgMatcher(...)}}
    args: dict[str, dict[str, ArgMatcher]] = Field(default_factory=dict)

    # 期望的相对顺序（子序列匹配，不是全等）：模型被允许重复调用同一个工具。
    # 缺省表示不检查顺序 —— 这时该项 check 会被标记为 skipped 而非通过。
    order: list[str] = Field(default_factory=list)

    # 是否容忍 expect.tools 之外的调用。默认不容忍：多调一个不该调的工具
    # 是真实的能力问题，不该被默认放过。
    allow_extra_tools: bool = False

    @model_validator(mode="after")
    def _fill_call_bounds(self) -> "ToolExpectation":
        """把没写的次数上下界补成「与工具数一致」。

        在模型里补而不是在 grader 里补，是为了让报告里的期望值就是最终生效的值 ——
        否则日志上显示 min_calls=null，而实际按 1 判定，事后对不上账。
        """
        if self.min_calls is None:
            self.min_calls = len(self.tools)
        if self.max_calls is None:
            self.max_calls = len(self.tools)
        return self


class ToolCallingCase(BaseModel):
    """datasets/tool_calling.jsonl 的一行。"""

    id: str
    question: str
    corpus: str
    history: list[HistoryTurn] = Field(default_factory=list)
    expect: ToolExpectation
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None


# ---- 用例模型：RAG ----


class RagCase(BaseModel):
    """datasets/rag.jsonl 的一行。

    刻意【没有 answer / expected_answer 字段】：
    RAG 层评的是检索结果本身，一旦在这里放开 answer，就会有人用它去评生成质量，
    而「答案对不对」和「检索准不准」是两个必须分开归因的问题 ——
    混在一起时，检索明明命中了却因为模型没答好而被判失败，调优方向会被带偏。
    """

    id: str
    corpus: str
    query: str
    top_k: int = 5

    # Ground Truth 的主口径：chunk 序号（1-based），由 build_chunk_id 组装成 chunk_id。
    # 写序号而不是写完整 chunk_id，是因为完整 ID 是
    # "<document_id>-chunk-000003" 这种几十个字符的串，人工维护极易抄错。
    relevant_chunk_seqs: list[int] = Field(default_factory=list)

    # 补充口径：命中 content 子串即算相关。用于语料被改写、序号漂移时兜底，
    # 也用于「相关但不是唯一答案」的场景。
    relevant_snippets: list[str] = Field(default_factory=list)

    # Recall@K 的通过阈值。默认要求全召回。
    min_recall: float = 1.0
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None


# ---- 用例模型：Final Answer ----


class JudgeOptions(BaseModel):
    """逐项开关 LLM Judge。

    默认两个都关，数据集必须显式打开。
    用意是「确定性检查优先」：能靠子串匹配判掉的事实题，不该花一次模型调用，
    更不该把判定权交给一个可能抖动的判官。
    """

    relevance: bool = False
    groundedness: bool = False


class FinalAnswerCase(BaseModel):
    """datasets/final_answer.jsonl 的一行。"""

    id: str
    corpus: str
    question: str
    history: list[HistoryTurn] = Field(default_factory=list)

    # 答案里必须出现的事实片段（子串匹配）。空列表表示不做正确性内容检查。
    expected_facts: list[str] = Field(default_factory=list)

    # 出现即判失败的片段：错误事实、幻觉短语、或泄漏 system prompt 的迹象。
    forbidden: list[str] = Field(default_factory=list)

    # True  -> 答案必须命中拒答标记（该拒答，考察「不编造」）
    # False -> 答案不得命中拒答标记（不该拒答，考察「不过度保守」）
    # 两个方向都要有用例：只测一个方向的话，一个永远回「无法确定」的
    # 退化实现能拿到满分。
    must_refuse: bool = False

    judge: JudgeOptions = Field(default_factory=JudgeOptions)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None


# ---- Artifact：跑完链路之后的证据 ----


class ToolCallingArtifact(BaseModel):
    """Tool Calling 层跑完 run_agent 拿到的证据。

    trace 和 executed 都带上，是因为它们回答的是两个不同问题：
        trace    —— 模型【请求】了哪些工具（含被拒绝、参数非法的）
        executed —— 哪些调用【真的进入了执行阶段】
    只看一个都会漏掉一半事实。它们的差别见 agent.py 里
    _ToolContext.executed 与 TraceContext.tool_call 的写入时机。
    """

    trace: Trace
    executed: list[ToolCallRecord] = Field(default_factory=list)
    answer: str = ""


class RagArtifact(BaseModel):
    """RAG 层跑完 embed_text + vector_store.search 拿到的证据。"""

    query: str
    top_k: int
    hits: list[Hit] = Field(default_factory=list)
    # 由 runner 用 build_chunk_id 组装好的 Ground Truth。
    # 不让 grader 自己去拼：拼 ID 要读语料配置，那是 runner 的职责，
    # grader 拿到手就该是可直接比较的集合。
    relevant_chunk_ids: set[str] = Field(default_factory=set)


class FinalAnswerArtifact(BaseModel):
    """Final Answer 层跑完 run_agent（并按需回放检索）拿到的证据。"""

    answer: str
    trace: Trace
    hits: list[Hit] = Field(default_factory=list)

    # 上下文是不是「回放」出来的。
    # Trace 刻意不记录工具返回内容（见 trace.py 的安全约定），所以答案真正的
    # 依据只能靠 trace 里模型用过的 query/top_k 重新检索一次得到。
    # 这个字段把这个事实显式写进报告，避免有人把回放的上下文
    # 当成「模型当时看到的那一份」。
    contexts_replayed: bool = True


# ---- 结果模型 ----


class CheckResult(BaseModel):
    """一项检查的结果。

    weight 直接挂在每一项上，而不是在 grader 里维护一张「名字 -> 权重」的表：
    参数检查是按字段拆成多项的（arguments:xxx.query / arguments:xxx.top_k），
    字段数由数据集决定，用名字查表就得多写一层模糊匹配。
    权重放在数据里，报告里也能看出「这一分是怎么来的」。
    """

    name: str
    # passed=None 表示 skipped：数据集没声明这一项，或判官失败。
    # 它既不参与 passed 的判定，也不参与分数 —— 「没测」不该长得像「做对了」。
    passed: bool | None = None
    # 归一化到 [0,1]，越大越好；None 表示不计入汇总。
    metric: float | None = None
    weight: float = 0.0
    expected: Any = None
    actual: Any = None
    detail: str | None = None


class GradeResult(BaseModel):
    """一条用例的判定结果。"""

    case_id: str
    layer: Literal["tool_calling", "rag", "final_answer"]
    passed: bool
    score: float
    checks: list[CheckResult] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    # 判官的原始返回，便于人工复核「模型为什么这么判」。
    judge: dict[str, Any] | None = None


class CaseResult(BaseModel):
    """一条用例的完整执行结果（含链路层的信息）。

    status 三态是刻意的：
        passed  —— 判定通过
        failed  —— 跑了，但没达标（能力问题）
        invalid —— 根本没跑成（模型 API 挂了、判官输出不是 JSON、Milvus 连不上）
    invalid 与 failed 必须分开计数。混在一起的话，一次网络抖动会让
    「通过率下降」看起来像模型退化，而人对着报告根本查不出原因。
    """

    case_id: str
    layer: str
    tags: list[str] = Field(default_factory=list)
    status: Literal["passed", "failed", "invalid"]
    passed: bool
    score: float = 0.0
    duration_ms: float = 0.0
    # 形如 "RuntimeError: DeepSeek 接口返回错误（HTTP 429）"，
    # 由 services/trace.py 的 format_error 产出（不带堆栈）。
    error: str | None = None
    grade: GradeResult | None = None


class TagSummary(BaseModel):
    """按标签分组的统计。

    它的用途不是做 dashboard（那是后续阶段的事），而是让失败可归因 ——
    「寒暄类全过、多跳类全挂」这种结论只靠一个总通过率是看不出来的。
    """

    total: int = 0
    passed: int = 0
    failed: int = 0
    invalid: int = 0


class LayerSummary(BaseModel):
    """一层的统计。"""

    layer: str
    total: int
    passed: int
    failed: int
    invalid: int
    pass_rate: float
    # 各 metric 在所有【有效】用例上的均值（跳过 None）。
    metrics: dict[str, float] = Field(default_factory=dict)


class CorpusSeedInfo(BaseModel):
    """一次 seeding 的结果，写进报告备查。"""

    corpus: str
    kb_id: str
    document_id: str
    chunks: int
    taskset: str = Field(default="")


class EvalReport(BaseModel):
    """一次评测运行的完整报告，直接落盘成 reports/<run_id>.json。"""

    run_id: str
    started_at: datetime
    finished_at: datetime
    model: str
    judge_model: str
    judge_enabled: bool
    mcp_enabled: bool
    seeding: list[CorpusSeedInfo] = Field(default_factory=list)
    layers: list[LayerSummary] = Field(default_factory=list)
    by_tag: dict[str, TagSummary] = Field(default_factory=dict)
    # 被评测链路消耗的模型调用次数（从每轮的 Trace 里数出来的）。
    total_llm_calls: int = 0
    # 判官额外消耗的模型调用次数。单独记，因为它是评测自身的成本：
    # 把两者混在一起，就再也说不清「这一轮花了多少钱」里有多少是在测、多少是在跑。
    judge_calls: int = 0
    cases: list[CaseResult] = Field(default_factory=list)


# ---- 判官的结构化输出 ----


class RelevanceVerdict(BaseModel):
    """判官对「答案是否切题」的判定。"""

    score: int = Field(ge=1, le=5)
    reason: str = ""


class GroundednessVerdict(BaseModel):
    """判官对「答案是否有据」的判定。"""

    grounded: bool
    score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
