"""对话接口的请求 / 响应模型。

用 Pydantic 模型而不是裸 dict 来定义接口契约，好处有两个：
1. FastAPI 会据此自动校验入参、生成 /docs 文档；字段类型不对时直接返回 422，
   不用在路由里手写「这个字段是不是字符串」这类判断。
2. 字段名、类型、说明都集中在一处，前后端对齐时只看这个文件就够了。
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /api/chat 的请求体。"""

    # min_length=1 是为了挡掉空消息：空字符串发给模型既浪费一次调用，
    # 也拿不到有意义的回复，不如让 FastAPI 直接以 422 拒掉，错误更早也更清楚。
    message: str = Field(
        min_length=1,
        description="用户本轮输入的消息内容",
        examples=["你好"],
    )


class ChatResponse(BaseModel):
    """POST /api/chat 的响应体。

    目前只返回一个 reply 字段。将来即使前端需要 token 用量、引用来源等信息，
    也应该是在这里「加字段」，而不是改 reply 的含义 —— 加字段对老客户端是兼容的。
    """

    reply: str = Field(
        description="DeepSeek 返回的回复文本",
        examples=["DeepSeek连接成功"],
    )
