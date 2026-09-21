"""知识库检索接口：把一个问题转成向量，在指定知识库里找最相似的切片。

本层只做三件事：校验请求、确认知识库存在、把两个 service 串起来
（embedding 生成查询向量 → vector_store 检索）。真正的向量化逻辑和
Milvus 查询逻辑都在 services 层，这里不复制任何一份。

这是 RAG 的「检索」那一半。「生成答案」那一半（把检索结果拼进提示词
交给大模型）不在本接口里，属于后续阶段。
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.knowledge_base import KnowledgeBase
from app.schemas.search import SearchRequest, SearchResponse, SearchResultItem
from app.services import vector_store
from app.services.embedding import embed_text

logger = logging.getLogger(__name__)

# 路由前缀走 /api/knowledge-bases，是因为「检索」在语义上从属于某个知识库：
# 检索永远发生在某一个库内，把 id 放在路径上比放在请求体里更能体现这层归属，
# 也让「跨库检索」这种危险的用法在 URL 层面就不可能表达出来。
router = APIRouter(prefix="/api/knowledge-bases", tags=["知识库检索"])


@router.post(
    "/{knowledge_base_id}/search",
    response_model=SearchResponse,
    summary="在指定知识库内做语义检索",
)
async def search_knowledge_base(
    knowledge_base_id: UUID,
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """把一个查询文本向量化，在指定知识库内检索最相似的切片。

    参数：
        knowledge_base_id: 只在这个知识库内检索。
        request:           query（1~2000 字符）和 top_k（1~20，默认 5）。

    返回：
        200 + 检索结果，按相似度从高到低。知识库存在但还没有任何文档时，
        results 为空列表而不是报错 —— 「库里没东西」是正常状态，不是故障。

    异常：
        404 知识库不存在；500 向量化或检索失败。
    """
    # ---- 1. 知识库必须存在 ----
    # 这一步除了给出明确的 404，还有个实际作用：把「库不存在」和
    # 「库存在但没有内容」区分开。少了它，两种情况都会返回空结果，
    # 调用方分不清是自己传错了 ID，还是这个库还没上传文档。
    knowledge_base = await db.get(KnowledgeBase, knowledge_base_id)
    if knowledge_base is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"知识库不存在：{knowledge_base_id}",
        )

    # ---- 2. 查询文本 → 向量 ----
    # 注意这里不需要对 query 做任何「转义」：它不会进入 Milvus 的过滤表达式，
    # 只是被送去 embedding，得到的向量再参与相似度计算。
    # 表达式注入的风险只存在于过滤条件（knowledge_base_id）上，
    # 而那个值来自 UUID 解析，且在 vector_store 里还有一道双引号校验。
    try:
        query_vector = await embed_text(request.query)
    except Exception as exc:
        logger.exception("检索失败（向量化阶段）：knowledge_base_id=%s", knowledge_base_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="检索失败：生成查询向量时出错，请稍后重试",
        ) from exc

    # ---- 3. 在指定知识库内检索 ----
    try:
        hits = await vector_store.search(
            # 强制只搜这个库：这是隔离的保证，不是可选项。
            # 一旦这里传错或漏传，检索会跨库返回别人的内容 —— 既是数据泄漏，
            # 也会让结果质量下降。所以它由 URL 路径参数一路传下来，没有默认值。
            knowledge_base_id=str(knowledge_base_id),
            query_vector=query_vector,
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.exception("检索失败（向量检索阶段）：knowledge_base_id=%s", knowledge_base_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="检索失败：查询向量库时出错，请稍后重试",
        ) from exc

    # ---- 4. 整理返回值 ----
    # hits 里已经只有 chunk_id / document_id / content / score 四个字段
    # （vector_store 刻意不返回向量本身），直接映射即可。
    # 顺序保持 Milvus 返回的顺序，这里不排序。
    return SearchResponse(
        query=request.query,
        results=[SearchResultItem(**hit) for hit in hits],
    )
