"""Agent 接口：让模型自主决定是否检索知识库，再基于结果作答。

本层刻意写得很薄 —— 按 CLAUDE.md 的约定，Agent 的逻辑（工具定义、循环、
参数校验、安全边界）全在 services/agent.py 里。这里只做三件事：
校验知识库存在、调用 service、把结果映射成对外的响应模型。

**知识库 ID 只从路径取**，然后原样传给 service。它不会出现在工具参数里，
模型也就没有任何途径去检索别的知识库 —— 隔离的保证在这一层就定死了。
"""

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.knowledge_base import KnowledgeBase
from app.schemas.agent import AgentRequest, AgentResponse, AgentToolCall
from app.services.agent import run_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge-bases", tags=["知识库 Agent"])


@router.post(
    "/{knowledge_base_id}/agent",
    response_model=AgentResponse,
    summary="知识库 Agent（模型自主决定是否检索）",
)
async def run_knowledge_base_agent(
    knowledge_base_id: UUID,
    request: AgentRequest,
    db: AsyncSession = Depends(get_db),
) -> AgentResponse:
    """把问题交给 Agent，由模型决定要不要检索知识库。

    与 /ask 的区别：/ask 每次都固定先检索再作答；这里由模型自己判断。
    对「你好」这类不需要查资料的问题，Agent 会直接回答而不做无谓检索。

    参数：
        knowledge_base_id: 允许检索的知识库，由路径指定，模型无法更改。
        request:           question（1~2000 字符）。

    返回：
        200 + answer 和本次实际执行过的 tool_calls。

    异常：
        404 知识库不存在；500 Agent 执行失败（模型调用失败等）。
    """
    # 知识库不存在就没有「可以检索的范围」可言，直接在入口挡掉。
    # 少了这一步，一个不存在的 ID 会被原样传进 Agent 循环，
    # 检索阶段查不到东西，模型最后只能回答「无法确定」——
    # 调用方会以为「这个库里没有相关内容」，而实际上这个库根本不存在。
    knowledge_base = await db.get(KnowledgeBase, knowledge_base_id)
    if knowledge_base is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"知识库不存在：{knowledge_base_id}",
        )

    try:
        result = await run_agent(
            question=request.question,
            knowledge_base_id=knowledge_base_id,
        )
    except Exception as exc:
        # Agent 内部已经处理了「工具执行失败」（作为工具结果回给模型），
        # 能跑到这里的只有真正致命的错误：模型调不通、循环失控等。
        # 具体原因只写日志，回给客户端一句通用文案。
        logger.exception("Agent 执行失败：knowledge_base_id=%s", knowledge_base_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent 处理失败，请稍后重试",
        ) from exc

    return AgentResponse(
        answer=result.answer,
        # 显式转换而不是直接塞 service 的对象：接口的返回结构由 schema 决定，
        # 不该跟着 service 的内部模型走。
        tool_calls=[_to_agent_tool_call(record) for record in result.tool_calls],
    )


def _to_agent_tool_call(record: Any) -> AgentToolCall:
    """把 service 层的调用记录映射成对外的响应结构。

    query / top_k 从 arguments 里取而不是让 service 单独维护一份 ——
    它们是「同一个事实的两种呈现」，分开存就有不一致的可能。
    非检索工具的参数里没有这两个键，取出来自然是 None。
    """
    arguments = record.arguments or {}
    return AgentToolCall(
        tool=record.tool,
        arguments=arguments,
        query=arguments.get("query"),
        top_k=arguments.get("top_k"),
    )
