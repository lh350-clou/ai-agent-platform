"""RAG 检索链路的回归：真 embedding + 真 Milvus。

和 Evaluation V1 的 `rag` 层分工不同，两者都要有：

    Evaluation  —— 拿 12 条标注语料算 Hit@K / Recall@K / MRR，
                   回答的是「检索质量有没有变差」；
    本文件      —— 只问「检索这条路还通不通」：写进去的能查出来吗？
                   会不会查到别的库的数据？删了之后还在不在？

所以这里【不评分数、不比排序】，只断言那几条一旦坏了就出大事的性质。
特别是「跨库查不到」—— 隔离条件一旦被删掉，检索照样返回结果、照样有分数，
只是返回的是别人的数据，看结果是发现不了的。
"""

import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from app.services import vector_store
from app.services.document.ingest import ingest_txt
from app.services.embedding import embed_text

pytestmark = [pytest.mark.integration, pytest.mark.regression]

# 语料写得短（对 500 字符的切分参数来说只有 1 个 chunk），
# 这样「命中哪一条」没有歧义，断言不依赖切分细节 ——
# 切分细节变了不该让回归变红，那是 splitter 自己的单测该管的事。
CORPUS = (
    "Milvus 的默认端口是 19530，控制台端口是 9091。\n"
    "本平台把向量存在 Milvus，把业务事实存在 PostgreSQL。\n"
    "检索时必须带上知识库过滤条件，否则会跨库返回别人的数据。\n"
)

# 语料里唯一出现「19530」的地方就是第一段，用它做锚点。
QUERY = "Milvus 的默认端口是多少？"
ANCHOR = "19530"


async def _seed(corpus_path: Path, kb_id: str) -> str:
    """把语料入库，返回 document_id。"""
    document_id = str(uuid.uuid4())
    written = await ingest_txt(corpus_path, kb_id, document_id)
    assert written >= 1, "入库没有写入任何 chunk"
    return document_id


def _write_corpus(tmp_path: Path) -> Path:
    # 写在 pytest 的 tmp_path 里，由 pytest 自己清理 ——
    # 不往 tests/ 下放固定文件，就不会出现「跑完测试多了几个待提交文件」。
    path = tmp_path / "regression-corpus.txt"
    path.write_text(CORPUS, encoding="utf-8")
    return path


async def test_seeded_corpus_is_retrievable(new_kb: Callable[[], str], tmp_path: Path) -> None:
    """入库 → 检索：写进去的内容必须能被查回来。

    这是 RAG 整条链路的地基。它红了说明「文档处理或向量检索」断了，
    后面所有关于检索质量的结论都不用看了。
    """
    kb_id = new_kb()
    document_id = await _seed(_write_corpus(tmp_path), kb_id)

    hits = await vector_store.search(kb_id, await embed_text(QUERY), top_k=3)

    assert hits, "刚入库的内容一条都检索不到"
    top = hits[0]
    # 返回的字段要能对回业务数据：chunk_id 是 PostgreSQL 里 chunk 的主键，
    # document_id 是它属于哪篇文档。少一个，检索结果就没法溯源。
    assert top["document_id"] == document_id
    assert top["chunk_id"].startswith(f"{document_id}-chunk-")
    assert ANCHOR in top["content"]
    # score 是余弦相似度（越接近 1 越相似），不是距离。
    # 断言它的范围而不是具体值：具体值随 embedding 模型变，范围不该变。
    assert 0.0 < top["score"] <= 1.0


async def test_search_is_isolated_by_knowledge_base(new_kb: Callable[[], str], tmp_path: Path) -> None:
    """A 库的向量，在 B 库里一条都不该被查到。

    这条是回归集里最该有的一条负向断言。检索时那个
    `knowledge_base_id == "..."` 过滤条件如果被删掉或写错，
    接口不会报错、结果也不会为空 —— 只是返回别人的资料，
    而且相似度分数看起来一切正常。线上表现为数据泄漏，
    看检索结果永远看不出来。
    """
    owner_kb, other_kb = new_kb(), new_kb()
    await _seed(_write_corpus(tmp_path), owner_kb)

    query_vector = await embed_text(QUERY)

    # 自己的库：查得到（否则下一条断言就是「两个库都查不到」的假通过）。
    assert await vector_store.search(owner_kb, query_vector, top_k=3)
    # 别人的库：一条都不该有。
    assert await vector_store.search(other_kb, query_vector, top_k=3) == []


async def test_delete_by_knowledge_base_id_removes_vectors(new_kb: Callable[[], str], tmp_path: Path) -> None:
    """按知识库删除：返回删除条数，且删完立刻查不到。

    这条守的是整个回归套件（以及 Evaluation）依赖的那个清理动作本身。
    删除写错的表现是「删了但还在」，而它不会报错 ——
    残留的向量会一直堆在库里，越积越多。
    """
    kb_id = new_kb()
    await _seed(_write_corpus(tmp_path), kb_id)

    query_vector = await embed_text(QUERY)
    assert await vector_store.search(kb_id, query_vector, top_k=3)

    removed = await vector_store.delete_by_knowledge_base_id(kb_id)
    assert removed >= 1

    # 紧接着就查：删除必须立刻可见（vector_store 内部做了 flush，
    # 不做的话 Milvus 的删除是有可见性延迟的，这里会假红）。
    assert await vector_store.search(kb_id, query_vector, top_k=3) == []

    # 重复删除不是错误：返回 0 表示「目标状态已达成」。
    # 这条保证 fixture 的清理可以无条件调用，不用先判断有没有数据。
    assert await vector_store.delete_by_knowledge_base_id(kb_id) == 0
