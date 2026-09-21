"""知识库接口：建、查、列、删。

本层刻意很薄 —— 校验请求、调用 service、把结果映射成响应模型。
删除知识库要同时协调 Milvus 和 PostgreSQL，那段编排在 services/knowledge_base.py 里。
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.knowledge_base import KnowledgeBase
from app.schemas.knowledge_base import KnowledgeBaseCreate, KnowledgeBaseResponse
from app.services import knowledge_base as kb_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge-bases", tags=["知识库"])


def _to_response(knowledge_base: KnowledgeBase, document_count: int) -> KnowledgeBaseResponse:
    """把 ORM 对象和文档数量拼成响应模型。

    document_count 不在 ORM 对象上（它是实时统计出来的），所以不能直接用
    model_validate 从对象转换，得显式传进来。
    """
    return KnowledgeBaseResponse(
        id=knowledge_base.id,
        name=knowledge_base.name,
        description=knowledge_base.description,
        created_at=knowledge_base.created_at,
        updated_at=knowledge_base.updated_at,
        document_count=document_count,
    )


@router.post(
    "",
    response_model=KnowledgeBaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建知识库",
)
async def create_knowledge_base(
    request: KnowledgeBaseCreate,
    db: AsyncSession = Depends(get_db),
) -> KnowledgeBaseResponse:
    """创建一个知识库。名称必填，描述可空。"""
    knowledge_base, document_count = await kb_service.create_knowledge_base(
        db, name=request.name, description=request.description
    )
    return _to_response(knowledge_base, document_count)


@router.get("", response_model=list[KnowledgeBaseResponse], summary="列出全部知识库")
async def list_knowledge_bases(db: AsyncSession = Depends(get_db)) -> list[KnowledgeBaseResponse]:
    """列出全部知识库，按创建时间倒序，附带各自的文档数量。

    空列表是正常结果（还没建过知识库），返回 200 + []，不是 404。
    """
    rows = await kb_service.list_knowledge_bases(db)
    return [_to_response(kb, count) for kb, count in rows]


@router.get("/{knowledge_base_id}", response_model=KnowledgeBaseResponse, summary="查询知识库")
async def get_knowledge_base(
    knowledge_base_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> KnowledgeBaseResponse:
    """按 ID 查询单个知识库。"""
    found = await kb_service.get_knowledge_base(db, knowledge_base_id)
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"知识库不存在：{knowledge_base_id}",
        )
    knowledge_base, document_count = found
    return _to_response(knowledge_base, document_count)


@router.delete(
    "/{knowledge_base_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除知识库（含文档、会话与向量数据）",
)
async def delete_knowledge_base(
    knowledge_base_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """删除知识库，同时清理它在 Milvus 里的向量。

    删除是不可逆的，且会连带删掉这个库下的全部文档、会话和向量，所以：
      - 知识库不存在返回 404（而不是静默成功）——
        静默成功会让调用方以为删掉了一个真实存在的库；
      - Milvus 清理失败时返回 500 且【不删 PostgreSQL】，保证数据不留半截。
    """
    # 先确认存在，把 404 和「删除失败」区分开。
    # 少了这一步，删一个不存在的 ID 会因为 service 里「记录已不存在」而返回 204，
    # 调用方无从知道自己删了个不存在的东西。
    exists = await kb_service.get_knowledge_base(db, knowledge_base_id)
    if exists is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"知识库不存在：{knowledge_base_id}",
        )

    try:
        await kb_service.delete_knowledge_base(db, knowledge_base_id)
    except Exception as exc:
        # 具体原因只写日志：Milvus 的连接信息、主机名、异常堆栈都不该回给客户端。
        logger.exception("删除知识库失败：knowledge_base_id=%s", knowledge_base_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除知识库失败：清理向量数据时出错，知识库未被删除，请稍后重试",
        ) from exc
