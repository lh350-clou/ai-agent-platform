"""知识库的增删查：业务逻辑都在这一层，API 路由只负责转发。

单独抽出来的理由和之前几个 service 一致：删除知识库要同时处理
PostgreSQL 和 Milvus 两个存储，这段编排既有顺序要求又有失败回滚的取舍，
塞在路由函数里会让「接口长什么样」和「数据怎么保证一致」两件事混在一起。
"""

import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.services import vector_store

logger = logging.getLogger(__name__)


async def _document_counts(
    db: AsyncSession, knowledge_base_ids: list[UUID]
) -> dict[UUID, int]:
    """统计每个知识库下的文档数量。

    用一条 GROUP BY 查出全部，而不是在循环里逐个查：
    列表接口一次要返回 N 个知识库，逐个查就是 N 次数据库往返（经典的 N+1），
    知识库一多就会明显变慢。

    返回的字典里【没有】文档数为 0 的知识库 —— 它们不会出现在 GROUP BY 的结果里，
    所以调用方要用 .get(id, 0) 取值，不能直接下标。
    """
    if not knowledge_base_ids:
        return {}

    rows = await db.execute(
        select(Document.knowledge_base_id, func.count(Document.id))
        .where(Document.knowledge_base_id.in_(knowledge_base_ids))
        .group_by(Document.knowledge_base_id)
    )
    return {kb_id: count for kb_id, count in rows.all()}


async def list_knowledge_bases(db: AsyncSession) -> list[tuple[KnowledgeBase, int]]:
    """列出全部知识库，附带各自的文档数量。

    按创建时间倒序：新建的排在前面，符合「刚建的马上要用」的直觉。
    不按名称排序是因为中文名称的编码序对用户没有意义。
    """
    result = await db.execute(select(KnowledgeBase).order_by(KnowledgeBase.created_at.desc()))
    knowledge_bases = list(result.scalars().all())
    counts = await _document_counts(db, [kb.id for kb in knowledge_bases])
    return [(kb, counts.get(kb.id, 0)) for kb in knowledge_bases]


async def get_knowledge_base(
    db: AsyncSession, knowledge_base_id: UUID
) -> tuple[KnowledgeBase, int] | None:
    """按 ID 取单个知识库，附带文档数量。不存在返回 None。"""
    knowledge_base = await db.get(KnowledgeBase, knowledge_base_id)
    if knowledge_base is None:
        return None

    counts = await _document_counts(db, [knowledge_base_id])
    return knowledge_base, counts.get(knowledge_base_id, 0)


async def create_knowledge_base(
    db: AsyncSession, name: str, description: str | None
) -> tuple[KnowledgeBase, int]:
    """创建知识库。名称与描述的清洗由 schema 层完成，这里只负责落库。"""
    knowledge_base = KnowledgeBase(name=name, description=description)
    db.add(knowledge_base)
    await db.commit()
    # 新建的库必然没有文档，这里直接给 0 而不是再查一次数据库。
    return knowledge_base, 0


async def delete_knowledge_base(db: AsyncSession, knowledge_base_id: UUID) -> int:
    """删除知识库，连同它的文档、会话和 Milvus 向量。

    删除顺序是「先 Milvus，再 PostgreSQL」，和删除单个文档时一致，理由也一样：

        Milvus 里没有外键、没有级联，向量只能靠 knowledge_base_id 这个值来找。
        如果先删了 PostgreSQL 记录，那些向量就永远失去了线索 ——
        没人再知道它们属于哪个库，也就再也删不掉了。它们会一直留在
        collection 里，检索时照样被召回，成为清不掉的垃圾数据。

    反过来先删 Milvus、失败就中止，最坏情况只是「向量删了但记录还在」——
    这个状态是可见、可重试的，用户再点一次删除即可，不会积累隐形垃圾。

    返回：
        Milvus 中实际删除的向量条数（用于日志和测试断言）。

    异常：
        RuntimeError：Milvus 删除失败。此时 PostgreSQL 不动，调用方可重试。
    """
    # ---- 1. 先清 Milvus ----
    # 失败会直接抛出去，PostgreSQL 那一侧完全不碰。
    deleted_vectors = await vector_store.delete_by_knowledge_base_id(str(knowledge_base_id))

    # ---- 2. 先把磁盘文件路径读出来 ----
    # 必须在删记录【之前】读：documents 表跟着知识库一起级联删除，
    # 删完就再也查不到这些文件在哪了 —— 它们会变成永远清理不掉的孤儿文件，
    # 和 Milvus 向量是同一类问题（记录没了，线索就断了）。
    rows = await db.execute(
        select(Document.file_path).where(Document.knowledge_base_id == knowledge_base_id)
    )
    file_paths = [Path(path) for path in rows.scalars().all()]

    # ---- 3. 再删 PostgreSQL ----
    # documents 和 conversations 都配了 ondelete="CASCADE" 的外键，
    # 由数据库一条语句连带删除，这里不需要手动逐个清理 ——
    # 手动删反而容易漏（比如将来又加了一张属于知识库的表）。
    knowledge_base = await db.get(KnowledgeBase, knowledge_base_id)
    if knowledge_base is None:
        # 理论上到不了：调用方已经先确认过存在。
        # 但真出现说明「检查存在」和「执行删除」之间被别处删掉了，
        # 此时 Milvus 已经清空，直接返回 0 而不是抛错。
        logger.warning("删除知识库时记录已不存在：%s", knowledge_base_id)
        return deleted_vectors

    await db.delete(knowledge_base)
    await db.commit()

    # ---- 4. 最后删磁盘文件 ----
    # 放在 PostgreSQL 提交【之后】：反过来的话，万一提交失败，
    # 记录还在而文件没了 —— 文档列表里显示得好好的，点开却什么都没有。
    #
    # 这里和「删除单个文档」的处理不同：那边文件删不掉会返回 500，
    # 这边只记日志、不影响结果。因为此刻知识库已经真的删掉了，
    # 再回一个 500 会让用户以为删除失败而重试（然后拿到 404），
    # 而实际后果只是磁盘上多留了几个文件。
    # 一个文件删不掉，也不该让整个删除操作看起来失败了。
    for file_path in file_paths:
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("删除知识库时清理文件失败：%s", file_path)

    logger.info(
        "知识库已删除：knowledge_base_id=%s，清理向量 %d 条，文件 %d 个",
        knowledge_base_id, deleted_vectors, len(file_paths),
    )
    return deleted_vectors
