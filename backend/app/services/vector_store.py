"""Milvus 向量存储服务：chunk 的写入与检索。

职责边界：本模块只管「把向量存进去」和「按知识库把相似向量取出来」，
不负责切分文本、不负责生成向量（那是 splitter / embedding 的事），
也不负责拼 prompt（那是 RAG 的事）。

约定（见 CLAUDE.md）：业务代码只使用本模块导出的 insert_chunk() / search()，
不自己拿 MilvusClient，也不在别处硬编码 collection 名和向量维度 ——
两者分别来自 core/milvus.py 的常量和 settings.EMBEDDING_DIM。
"""

import asyncio
import logging

from pymilvus import MilvusException

from app.core import milvus
from app.core.config import settings

logger = logging.getLogger(__name__)

# 字段名提出来做常量：检索要返回它、删除要按它过滤，两处必须完全一致。
# 写错一个字母的后果是「删除时匹配不到任何数据」，而且不会报错 ——
# 接口返回成功，数据却还在，是最难发现的一类问题。
_DOCUMENT_ID_FIELD: str = "document_id"

# 检索时返回的字段。刻意不包含 vector：
# 一次检索动辄返回上千个浮点数，而调用方（拼 prompt 的 RAG 层）只需要文本。
# 把向量带回来只会白白占用带宽和内存。
_OUTPUT_FIELDS: list[str] = ["chunk_id", _DOCUMENT_ID_FIELD, "content"]


# ---- 参数校验 ----


def _validate_vector(vector: list[float]) -> None:
    """插入前校验向量维度。

    这是最后一道闸门。前面 embedding.py 已经校验过一次，这里再查一遍不是重复：
    维度对不上的向量写进 Milvus 不一定报错，但和任何查询向量的距离都是无意义的，
    表现为「数据明明写进去了，就是搜不出来」—— 这种问题极难定位。
    在写入前拦下，代价最小。
    """
    if len(vector) != settings.EMBEDDING_DIM:
        raise ValueError(
            f"向量维度不符：收到 {len(vector)} 维，但 collection 的维度是 "
            f"{settings.EMBEDDING_DIM}（settings.EMBEDDING_DIM）。"
        )


def _validate_filter_id(value: str, field_name: str) -> None:
    """校验要拼进 Milvus 过滤表达式的 ID 值。

    这里只挡一件事：值里包含双引号。因为检索和删除都要把它拼进过滤表达式
    （形如 document_id == "..."), 一个引号就能把表达式的结构撑坏，
    让它变成另一个含义完全不同的过滤条件 —— 也就是表达式注入。
    删除场景下这个后果尤其严重：本该只删一篇文档，可能删掉一整批。
    ID 正常情况下都是 UUID，永远不该含引号，所以这个校验不会误伤。
    """
    if not value:
        raise ValueError(f"{field_name} 不能为空")
    if '"' in value:
        raise ValueError(f"{field_name} 不能包含双引号")


# ---- 同步实现（真正调用 pymilvus 的部分）----
#
# 为什么要单独抽出来、再用 asyncio.to_thread 包一层：
# pymilvus 的 MilvusClient 是「同步」客户端，每个方法都会阻塞当前线程直到拿到结果。
# 如果直接在 async 函数里调用它，这段等待时间会卡死整个事件循环 ——
# 和当初给 DeepSeek 选 AsyncOpenAI 是同一个道理。
# pymilvus 虽然也提供 AsyncMilvusClient，但那样就得再建一个客户端，
# 与「不重复创建 MilvusClient」的要求冲突。所以这里复用已有的同步客户端，
# 把阻塞调用丢到线程里去执行。


def _insert_sync(row: dict) -> None:
    client = milvus.get_milvus_client()
    client.insert(collection_name=milvus.COLLECTION_NAME, data=[row])


def _delete_by_document_sync(document_id: str) -> int:
    client = milvus.get_milvus_client()

    result = client.delete(
        collection_name=milvus.COLLECTION_NAME,
        # 用 == 精确匹配，不用 like/in。
        # 这不是风格问题：like 是模糊匹配，一个 document_id 恰好是另一个的前缀时
        # 会把别人的数据一起删掉 —— 而删除是不可逆的。
        filter=f'{_DOCUMENT_ID_FIELD} == "{document_id}"',
    )

    # 删除同样有可见性问题（和写入那次一样）：
    # 不 flush 的话，紧接着的 query 仍可能查到「已经删掉」的数据，
    # 于是调用方会以为删除失败了。这里同步 flush 一次，保证删除立即可见。
    client.flush(milvus.COLLECTION_NAME)

    return int(result.get("delete_count", 0))


def _delete_by_knowledge_base_sync(knowledge_base_id: str) -> int:
    client = milvus.get_milvus_client()

    result = client.delete(
        collection_name=milvus.COLLECTION_NAME,
        filter=f'{milvus.KNOWLEDGE_BASE_ID_FIELD} == "{knowledge_base_id}"',
    )

    client.flush(milvus.COLLECTION_NAME)

    return int(result.get("delete_count", 0))


def _search_sync(knowledge_base_id: str, query_vector: list[float], top_k: int) -> list[list[dict]]:
    client = milvus.get_milvus_client()

    # 检索前必须先把 collection 加载进内存，否则会报 collection not loaded。
    # 这里不做「先查状态再决定是否加载」的优化：get_load_state 本身也是一次 RPC，
    # 和直接调 load_collection 的代价一样，反而多一层逻辑。
    client.load_collection(milvus.COLLECTION_NAME)

    return client.search(
        collection_name=milvus.COLLECTION_NAME,
        # data 是「一批查询向量」，这里每次只查一个，所以是单元素列表
        data=[query_vector],
        # 过滤条件 —— 这是知识库隔离发生的地方，不是可选优化。
        # 不加它，检索会在全部知识库范围内找最近邻：既会把别的库的内容
        # 返回给当前用户（数据泄漏），又会让无关内容挤占 top_k 名额。
        filter=f'{milvus.KNOWLEDGE_BASE_ID_FIELD} == "{knowledge_base_id}"',
        limit=top_k,
        output_fields=_OUTPUT_FIELDS,
        # 显式指定 COSINE，和建索引时的 metric_type 保持一致。
        search_params={"metric_type": "COSINE"},
    )


def _normalize_hits(raw: list[list[dict]]) -> list[dict]:
    """把 Milvus 的原始返回整理成扁平、只含必要字段的 dict 列表。

    原始结构的形状是「按查询分组的二维结构」：
        [[{id, distance, entity: {...}}, ...]]
    外层对应「第几个查询向量」，内层才是命中结果。我们每次只传一个查询向量，
    所以取 [0]。这层结构对调用方毫无意义，在这里剥掉，
    让 service 层的返回值就是「一组结果」本身。
    """
    if not raw:
        return []

    hits: list[dict] = []
    for hit in raw[0]:
        entity = hit.get("entity") or {}
        hits.append(
            {
                "chunk_id": entity.get("chunk_id"),
                "document_id": entity.get("document_id"),
                "content": entity.get("content"),
                # 字段名是 distance，但用 COSINE 时它其实是「余弦相似度」：
                # 越接近 1 越相似。这是 Milvus 沿用「距离」这个字段名造成的，
                # 不是我们算错了。统一改名成 score，避免调用方误以为是「越小越好」。
                "score": hit.get("distance"),
            }
        )
    return hits


# ---- 对外接口 ----


async def insert_chunk(
    knowledge_base_id: str,
    document_id: str,
    chunk_id: str,
    content: str,
    vector: list[float],
) -> None:
    """写入一个 chunk 的向量及其正文。

    参数：
        knowledge_base_id: 所属知识库，检索时靠它做隔离。
        document_id:      所属文档，删除整篇文档时靠它批量清理。
        chunk_id:         该切片在 PostgreSQL 中的主键，用于把检索结果对回业务数据。
        content:          切片正文（在 Milvus 里冗余存一份，让检索一次就能拿到文本）。
        vector:           长度必须等于 settings.EMBEDDING_DIM。

    返回：
        None。刻意不返回 Milvus 的原始响应（里面是 auto_id、插入计数这类
        只有 Milvus 自己关心的信息），调用方拿到也没有用。

    异常：
        ValueError：   参数不合法（维度不符、ID 为空或含引号）。
        RuntimeError： Milvus 不可用或写入失败。
    """
    _validate_vector(vector)
    _validate_filter_id(knowledge_base_id, "knowledge_base_id")

    # id 不传：它在 schema 里是 auto_id=True，由 Milvus 自己生成。
    # 业务标识是 chunk_id，它对应 PostgreSQL 里 chunks 表的主键。
    row = {
        "knowledge_base_id": knowledge_base_id,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "content": content,
        "vector": vector,
    }

    try:
        await asyncio.to_thread(_insert_sync, row)
    except MilvusException as exc:
        logger.exception("写入 Milvus 失败：collection=%s chunk_id=%s", milvus.COLLECTION_NAME, chunk_id)
        raise RuntimeError(f"写入 Milvus 失败（chunk_id={chunk_id}），详情见服务端日志") from exc
    except OSError as exc:
        # Milvus 没启动时，底层会抛连接类错误（ConnectionError 是 OSError 的子类）
        logger.exception("无法连接 Milvus：%s", milvus.MILVUS_URI)
        raise RuntimeError(
            f"无法连接 Milvus（{milvus.MILVUS_URI}），请确认容器是否在运行"
        ) from exc


async def delete_by_document_id(document_id: str) -> int:
    """删除某个文档在 Milvus 中的全部 chunk 向量。

    参数：
        document_id: 要删除的文档 ID。只会删这个 ID 下的数据，
                     其它文档的向量不受影响。

    返回：
        实际删除的条数。文档本来就没有向量时返回 0（不是错误）——
        「删除到 0 条」和「本来就没有」对调用方是同一件事：目标状态已达成。

    异常：
        ValueError：   document_id 为空或含双引号。
        RuntimeError： Milvus 不可用或删除失败。
    """
    _validate_filter_id(document_id, "document_id")

    try:
        return await asyncio.to_thread(_delete_by_document_sync, document_id)
    except MilvusException as exc:
        logger.exception(
            "删除 Milvus 数据失败：collection=%s document_id=%s",
            milvus.COLLECTION_NAME, document_id,
        )
        raise RuntimeError(f"删除 Milvus 数据失败（document_id={document_id}）") from exc
    except OSError as exc:
        logger.exception("无法连接 Milvus：%s", milvus.MILVUS_URI)
        raise RuntimeError(
            f"无法连接 Milvus（{milvus.MILVUS_URI}），请确认容器是否在运行"
        ) from exc


async def delete_by_knowledge_base_id(knowledge_base_id: str) -> int:
    """删除整个知识库在 Milvus 中的全部向量（该库下所有文档的所有切片）。

    用在「删除知识库」上。一次调用就能清干净，不需要先查出库下有哪些文档
    再逐个删 —— 因为 collection 里每一行都带着 knowledge_base_id，
    按它过滤是最直接的。

    参数：
        knowledge_base_id: 要清空的知识库。

    返回：
        实际删除的条数。库里本来就没有向量时返回 0（不是错误）。

    异常：
        ValueError：   knowledge_base_id 为空或含双引号。
        RuntimeError： Milvus 不可用或删除失败。
    """
    _validate_filter_id(knowledge_base_id, "knowledge_base_id")

    try:
        return await asyncio.to_thread(_delete_by_knowledge_base_sync, knowledge_base_id)
    except MilvusException as exc:
        logger.exception(
            "删除 Milvus 数据失败：collection=%s knowledge_base_id=%s",
            milvus.COLLECTION_NAME, knowledge_base_id,
        )
        raise RuntimeError(
            f"删除 Milvus 数据失败（knowledge_base_id={knowledge_base_id}）"
        ) from exc
    except OSError as exc:
        logger.exception("无法连接 Milvus：%s", milvus.MILVUS_URI)
        raise RuntimeError(
            f"无法连接 Milvus（{milvus.MILVUS_URI}），请确认容器是否在运行"
        ) from exc


async def search(
    knowledge_base_id: str,
    query_vector: list[float],
    top_k: int = 5,
) -> list[dict]:
    """在指定知识库内检索与查询向量最相似的 chunk。

    参数：
        knowledge_base_id: 只在这个知识库范围内检索 —— 不是可选项，是隔离的保证。
        query_vector:      查询文本的向量，长度必须等于 settings.EMBEDDING_DIM。
        top_k:             返回的最大条数。

    返回：
        dict 列表，按相似度从高到低排列。每项包含：
            chunk_id / document_id / content / score
        score 是余弦相似度（越接近 1 越相似）。不返回 vector 字段。

    异常：
        ValueError：   参数不合法。
        RuntimeError： Milvus 不可用或查询失败。
    """
    _validate_vector(query_vector)
    _validate_filter_id(knowledge_base_id, "knowledge_base_id")
    if top_k <= 0:
        raise ValueError(f"top_k 必须为正整数，收到 {top_k}")

    try:
        raw = await asyncio.to_thread(_search_sync, knowledge_base_id, query_vector, top_k)
    except MilvusException as exc:
        logger.exception("检索 Milvus 失败：collection=%s kb=%s", milvus.COLLECTION_NAME, knowledge_base_id)
        raise RuntimeError("检索 Milvus 失败，详情见服务端日志") from exc
    except OSError as exc:
        logger.exception("无法连接 Milvus：%s", milvus.MILVUS_URI)
        raise RuntimeError(
            f"无法连接 Milvus（{milvus.MILVUS_URI}），请确认容器是否在运行"
        ) from exc

    return _normalize_hits(raw)
