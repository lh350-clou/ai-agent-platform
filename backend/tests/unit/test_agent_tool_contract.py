"""Agent 工具契约的单元测试：参数收敛、范围校验、多余字段丢弃。

为什么这些断言必须放在单元测试里，而不是放进 LLM 的 Tool Calling 评测：

顺着 agent.py 的执行路径走一遍就会发现，「top_k 超过上限时会被收敛到 5」
这件事在 Agent 这条链路上【根本无法观测】——

    模型给出 top_k=1000
      ├─ 有收敛逻辑：_normalize_tool_arguments 改成 5，校验通过，trace 里记的是 5
      └─ 没有收敛逻辑：SearchToolArgs 的范围校验（le=5）直接拒绝整次调用，
                       _failed_call 返回 "invalid tool arguments"，
                       而 call.arguments 的赋值在校验【之后】，所以它保持 {}，
                       这次调用根本不会被记成「成功调用」

两种情况下，观测到的 top_k 永远不超过 5。也就是说，在评测数据集里写
「top_k <= 5」是一条【不可能失败】的断言：删掉它，通过率一模一样；
留着它，只会让人以为边界已经被覆盖了。真正能区分「有收敛」和「没收敛」的，
是这里 —— 直接调那个函数，看它到底返回了什么。

所以本文件测的是工具的【契约】，评测数据集测的是模型的【行为】，两者不混。
"""

import pytest
from pydantic import ValidationError

from app.services.agent import (
    DEFAULT_TOOL_TOP_K,
    MAX_TOOL_QUERY_LENGTH,
    MAX_TOOL_TOP_K,
    MIN_TOOL_TOP_K,
    SEARCH_TOOL_NAME,
    SearchToolArgs,
    _normalize_tool_arguments,
)

# 两条正交的标记：unit 说明它不依赖外部服务，regression 说明它属于
# 「守住已有能力」的那批（见 tests/regression/README.md）。
pytestmark = [pytest.mark.unit, pytest.mark.regression]


def _validated(raw: dict) -> SearchToolArgs:
    """走一遍生产代码的完整参数处理路径：先收敛，再校验。

    刻意不单独调 SearchToolArgs.model_validate：生产代码（_execute_search_tool）
    永远是「收敛 → 校验」这两步连着走的，只测其中一步会漏掉它们之间的配合。
    """
    return SearchToolArgs.model_validate(_normalize_tool_arguments(raw))


# ---- 参数收敛 ----


def test_top_k_above_limit_is_clamped_to_max() -> None:
    """模型要 1000 条：收敛到上限，而不是报错。

    「要多了」和「要错了」性质不同：要 1000 条最合理的理解是「尽量多给」，
    收敛到上限既满足它又守住边界；而要 0 条没有任何合理解释，属于参数错误。
    """
    args = _validated({"query": "Milvus", "top_k": 1000})

    assert args.top_k == MAX_TOOL_TOP_K


def test_top_k_at_limit_is_unchanged() -> None:
    """正好等于上限：原样保留，不该被误改。"""
    args = _validated({"query": "Milvus", "top_k": MAX_TOOL_TOP_K})

    assert args.top_k == MAX_TOOL_TOP_K


def test_top_k_within_range_is_unchanged() -> None:
    """范围内的值必须原样保留 —— 收敛逻辑只能动「超出上限」那一种情况。

    这条是在守一条边界：如果哪天有人把收敛写成「一律取上限」，
    上面几条用例都还是绿的，只有这一条会红。
    """
    args = _validated({"query": "Milvus", "top_k": 3})

    assert args.top_k == 3


def test_top_k_missing_uses_default() -> None:
    """不传 top_k：用默认值，而且是有效范围内的值。"""
    args = _validated({"query": "Milvus"})

    assert args.top_k == DEFAULT_TOOL_TOP_K
    assert MIN_TOOL_TOP_K <= args.top_k <= MAX_TOOL_TOP_K


def test_top_k_zero_is_rejected_not_clamped() -> None:
    """要 0 条：属于参数错误，必须被拒绝，而不是「收敛」成 1。

    把 0 也当成「要多了」那样收敛掉，等于替模型猜意图 ——
    而「返回 0 条」没有任何合理解释。
    """
    with pytest.raises(ValidationError):
        _validated({"query": "Milvus", "top_k": 0})


def test_top_k_negative_is_rejected() -> None:
    """负数同样必须被拒绝。"""
    with pytest.raises(ValidationError):
        _validated({"query": "Milvus", "top_k": -1})


def test_top_k_non_integer_is_rejected() -> None:
    """非整数（比如布尔值、字符串）必须被拒绝。"""
    with pytest.raises(ValidationError):
        _validated({"query": "Milvus", "top_k": "三"})


# ---- query 校验 ----


def test_query_is_stripped_before_length_check() -> None:
    """先去掉首尾空白再做长度校验。

    顺序很关键：默认校验器在字段校验【之后】运行，那样 "   " 会以长度 3
    通过 min_length=1，清洗后却成了空串，等于拿一个空查询去调 embedding ——
    那是要花钱的一次真实调用。
    """
    args = _validated({"query": "  Milvus 是什么  "})

    assert args.query == "Milvus 是什么"


def test_blank_query_is_rejected() -> None:
    """纯空白必须被判为非法，而不是清洗后变成空串放行。"""
    with pytest.raises(ValidationError):
        _validated({"query": "   "})


def test_query_at_max_length_is_accepted() -> None:
    """长度上限是闭区间：正好等于上限应当通过。"""
    args = _validated({"query": "a" * MAX_TOOL_QUERY_LENGTH})

    assert len(args.query) == MAX_TOOL_QUERY_LENGTH


def test_query_over_max_length_is_rejected() -> None:
    """超过上限一个字符就应当被拒绝。"""
    with pytest.raises(ValidationError):
        _validated({"query": "a" * (MAX_TOOL_QUERY_LENGTH + 1)})


# ---- 多余字段的处理（安全边界）----


def test_knowledge_base_id_in_arguments_is_dropped() -> None:
    """模型自己加上 knowledge_base_id：静默丢弃，绝不进检索路径。

    这是本模块最需要防的一件事。工具定义里没有这个参数，但模型完全可以
    自己加上去（或被提示词注入诱导），而真正的保障必须落在解析这一步 ——
    实际检索用的是请求路径里传来的知识库，不是参数里的这个。
    """
    args = _validated(
        {
            "query": "Milvus",
            "knowledge_base_id": "11111111-1111-1111-1111-111111111111",
        }
    )

    assert not hasattr(args, "knowledge_base_id")
    assert args.model_dump() == {"query": "Milvus", "top_k": DEFAULT_TOOL_TOP_K}


def test_unknown_arguments_are_ignored_not_rejected() -> None:
    """其他多余字段同样被忽略而不是报错。

    如果多余字段直接导致校验失败，模型多传一个无关字段就会让整次调用
    被记为「参数非法」，而它本来只是想查个资料 —— 那种失败对用户毫无意义。
    """
    args = _validated({"query": "Milvus", "verbose": True, "limit": 99})

    assert args.model_dump() == {"query": "Milvus", "top_k": DEFAULT_TOOL_TOP_K}


# ---- 工具名常量 ----


def test_search_tool_name_matches_dataset_expectations() -> None:
    """工具名是一个被多处引用的契约：工具定义、白名单、评测数据集都靠它对齐。

    写错一个字母的后果是「模型请求了工具，但执行时匹配不到」，而白名单
    会把它当成未知工具拒绝掉，表现为「Agent 从来不调用工具」，很难查。
    """
    assert SEARCH_TOOL_NAME == "search_knowledge_base"
