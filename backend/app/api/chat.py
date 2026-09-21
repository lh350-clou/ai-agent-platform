"""对话路由：把一次 HTTP 请求转成一次 DeepSeek 调用。

本层故意写得很薄，只做三件事：接参数 → 调 services 层 → 包装返回值。
不 import openai、不拼 prompt、不碰 settings。
这样将来把「单轮对话」换成 RAG 检索问答或 Agent 多步推理时，
改动都发生在 services 层，路由和对外契约（请求 / 响应 JSON）都不用动。
"""

import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.chat import ChatRequest, ChatResponse
from app.services import llm

logger = logging.getLogger(__name__)

# 统一挂 /api 前缀，让业务接口和 /health 这类运维接口在路径上区分开。
router = APIRouter(prefix="/api", tags=["对话"])


@router.post("/chat", response_model=ChatResponse, summary="与 DeepSeek 对话")
async def chat(request: ChatRequest) -> ChatResponse:
    """接收一条用户消息，交给 DeepSeek 生成回复。

    参数名 request 会和 FastAPI 的 Request 对象同名，但这里用的是自定义的
    ChatRequest，FastAPI 见到 Pydantic 模型就会把它当作请求体解析，不会冲突。
    """
    # 本轮先只发当前这一条消息。模型本身不记上下文，
    # 等做多轮对话时，这里会改成把历史消息一并传入，接口契约保持不变。
    messages = [{"role": "user", "content": request.message}]

    try:
        reply = await llm.chat(messages)
    except RuntimeError as exc:
        # llm.chat 已经把底层异常统一翻译成了不含密钥的中文说明，
        # 这里只需要把它映射成一个 HTTP 状态码。
        #
        # 用 503（服务暂时不可用）而不是 400：问题出在服务端这一侧
        # —— 可能是没配 Key，也可能是 DeepSeek 侧限流或故障，
        # 而调用方发来的请求本身是合法的，让它改请求没有意义。
        #
        # 具体原因只写日志、不放进响应体：这个接口目前不鉴权，
        # 把「未配置 API Key」这类内部状态回显出去属于信息泄漏。
        # 这与 api/health.py 里处理数据库异常的做法是一致的。
        logger.exception("对话接口调用 DeepSeek 失败")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型服务暂时不可用，请稍后重试",
        ) from exc

    return ChatResponse(reply=reply)
