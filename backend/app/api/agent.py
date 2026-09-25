"""Agent 接口：让模型自主决定是否调用工具，再基于结果作答。

本层刻意写得很薄 —— Agent 的逻辑（工具定义、循环、参数校验、安全边界）
全在 services/agent.py 里。这里负责：校验知识库与会话、读写消息、把结果
映射成对外的响应模型。

**知识库 ID 只从路径取**，然后原样传给 service。它不会出现在工具参数里，
模型也就没有任何途径去检索别的知识库 —— 隔离的保证在这一层就定死了。
"""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.conversation import Conversation
from app.models.knowledge_base import KnowledgeBase
from app.models.message import Message
from app.schemas.agent import AgentLLMCall, AgentRequest, AgentResponse, AgentToolCall
from app.services.agent import run_agent
from app.services.conversation import load_recent_messages
from app.services.trace import LLMCallTrace

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge-bases", tags=["知识库 Agent"])

# 会话标题取问题开头的字符数。255 是列宽上限，这里留足余量即可 ——
# 标题只用于列表展示，不需要装下完整问题。
TITLE_MAX_LENGTH: int = 50


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


def _to_agent_llm_call(record: LLMCallTrace) -> AgentLLMCall:
    """把 service 层的 LLM 调用记录映射成对外的响应结构。

    逐字段搬运而不是直接把内部模型塞进响应：接口的返回结构由 schema 决定，
    将来 Trace 里多加一个字段（比如 token 用量），也不会自动漏到接口上 ——
    要暴露就得有人在这里显式写一行，这个「多一步」正是它存在的意义。
    """
    return AgentLLMCall(
        model=record.model,
        duration_ms=record.duration_ms,
        success=record.success,
        error=record.error,
    )


@router.post(
    "/{knowledge_base_id}/agent",
    response_model=AgentResponse,
    summary="知识库 Agent（模型自主决定是否调用工具，支持多轮）",
)
async def run_knowledge_base_agent(
    knowledge_base_id: UUID,
    request: AgentRequest,
    db: AsyncSession = Depends(get_db),
) -> AgentResponse:
    """把问题交给 Agent，由模型决定要不要调用工具。

    与 /ask 的区别：/ask 每次都固定先检索再作答；这里由模型自己判断。
    对「你好」这类不需要查资料的问题，Agent 会直接回答而不做无谓检索。

    参数：
        knowledge_base_id: 允许检索的知识库，由路径指定，模型无法更改。
        request:           question（1~2000 字符）和可选的 conversation_id。

    返回：
        200 + conversation_id、answer、本次实际执行过的 tool_calls，
        以及本次运行的 trace_id / iterations / llm_calls / total_duration_ms。

    异常：
        404 知识库不存在，或会话不存在 / 不属于该知识库；
        500 Agent 执行失败（模型调用失败、消息保存失败等）。
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

    # ---- 1. 取出或新建会话 ----
    # 归属校验只认数据库里的 knowledge_base_id，不用任何内存状态或缓存 ——
    # 那些在重启和多实例部署下都会失效。
    if request.conversation_id is None:
        conversation = Conversation(
            knowledge_base_id=knowledge_base_id,
            # 用第一句话做标题，方便在会话列表里认出来。
            # 截断到 TITLE_MAX_LENGTH：完整问题可能有几千字，标题不需要那么长。
            title=request.question[:TITLE_MAX_LENGTH],
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
    else:
        conversation = await db.get(Conversation, request.conversation_id)
        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"会话不存在：{request.conversation_id}",
            )
        # 会话存在，但属于别的知识库 —— 同样返回 404 而不是 403。
        # 403 等于告诉调用方「这个 ID 是真实存在的，只是你没权限」，
        # 那本身就是一次信息泄漏。也不能「纠正」成用当前库继续，
        # 那会让 A 库的对话内容跑到 B 库的上下文里。
        if conversation.knowledge_base_id != knowledge_base_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"会话不存在：{request.conversation_id}",
            )

    # ---- 2. 先读历史，再保存本轮问题 ----
    # 顺序很关键：这里读到的必须是【不含本轮问题】的之前的消息。
    # 如果先保存再读，本轮问题会既出现在历史里、又被 run_agent 追加一次，
    # 同一个问题在提示词里出现两遍。
    history = await load_recent_messages(db, conversation.id)

    # ---- 3. 保存用户消息 ----
    # 先落库再调用外部服务：即使后面全部失败，用户问过什么也留下了记录 ——
    # 失败时最需要排查的就是「他到底问了什么」。
    #
    # 这一步失败就不该继续往下走：Agent 的回答会失去对应的提问，
    # 存进库里就是一条没有前文的 assistant 消息，下一轮读历史时莫名其妙。
    try:
        db.add(
            Message(
                conversation_id=conversation.id,
                role="user",
                content=request.question,
            )
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.exception("Agent 会话保存用户消息失败：conversation_id=%s", conversation.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="保存提问失败，请稍后重试",
        ) from exc

    # ---- 4. 执行 Agent ----
    try:
        result = await run_agent(
            question=request.question,
            knowledge_base_id=knowledge_base_id,
            history=history,
        )
    except Exception as exc:
        # Agent 内部已经处理了「工具执行失败」（作为工具结果回给模型），
        # 能跑到这里的只有真正致命的错误：模型调不通、循环失控等。
        #
        # 这里【不保存 assistant 消息】：失败时没有任何回答可存，
        # 存一条空消息或错误文案只会污染历史，让下一轮的上下文变得莫名其妙。
        logger.exception("Agent 执行失败：knowledge_base_id=%s", knowledge_base_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent 处理失败，请稍后重试",
        ) from exc

    # ---- 5. 保存回答并刷新会话时间 ----
    try:
        db.add(
            Message(
                conversation_id=conversation.id,
                role="assistant",
                content=result.answer,
            )
        )
        # 显式更新时间戳。只插入 message 是不会碰 conversations 这一行的，
        # 而 updated_at 带 onupdate 只在「本行确实被 UPDATE」时才生效 ——
        # 不主动赋值的话，会话列表按更新时间排序会永远停在创建时刻。
        conversation.updated_at = datetime.now(timezone.utc)
        await db.commit()
    except Exception as exc:
        # 回答已经生成出来了，但存不进去。只能回一个安全错误 ——
        # 不能把回答照常返回：那样用户以为这一轮正常，下一轮却发现
        # 模型完全不记得刚才说过什么（因为历史里缺了这一轮）。
        #
        # 回滚保证不会留下半提交状态：事务要么两条消息都在，要么只有 user。
        await db.rollback()
        logger.exception("Agent 会话保存回答失败：conversation_id=%s", conversation.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="保存回答失败，请稍后重试",
        ) from exc

    # run_agent 保证返回前已经 finish() 过，所以这里的 total_duration_ms
    # 一定是真实耗时，而不是「还没开始计」的 0。
    trace = result.trace

    return AgentResponse(
        conversation_id=conversation.id,
        answer=result.answer,
        # 显式转换而不是直接塞 service 的对象：接口的返回结构由 schema 决定，
        # 不该跟着 service 的内部模型走。
        tool_calls=[_to_agent_tool_call(record) for record in result.tool_calls],
        trace_id=trace.trace_id,
        iterations=trace.iterations,
        llm_calls=[_to_agent_llm_call(record) for record in trace.llm_calls],
        total_duration_ms=trace.total_duration_ms,
    )
