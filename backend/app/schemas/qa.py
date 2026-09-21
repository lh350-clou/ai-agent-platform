"""RAG 问答接口的请求 / 响应模型。

和检索接口一样，边界校验放在这一层不是形式主义：question 超长会
直接变成一次真实的 embedding 调用 + 一次真实的 DeepSeek 调用，
两道都是花钱的。在本地挡掉明显不合理的请求，成本最低。
"""

from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.schemas.search import SearchResultItem

# 问题文本的最大长度（字符）。与检索接口的 query 上限保持一致 ——
# 两者在语义上是同一种输入（用户问的一句话），没有理由不一样。
MAX_QUESTION_LENGTH: int = 2000

# top_k 的范围。上限比检索接口（20）更紧，是 10。
# 原因：检索接口返回的是「结果列表」，调用方自己决定怎么用；
# 而问答接口的每条结果都会【被拼进提示词送给大模型】——
# 拼得越多，上下文越长、越贵、越慢，而且无关内容还会干扰模型判断。
# 10 条已经足够覆盖一般问题所需的资料量。
MIN_TOP_K: int = 1
MAX_TOP_K: int = 10
DEFAULT_TOP_K: int = 5


class AskRequest(BaseModel):
    """POST /api/knowledge-bases/{id}/ask 的请求体。"""

    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description="用户的问题",
        examples=["什么是向量数据库？"],
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=MIN_TOP_K,
        le=MAX_TOP_K,
        description=f"检索多少条资料用于回答，范围 {MIN_TOP_K}~{MAX_TOP_K}",
        examples=[5],
    )
    # 会话 ID。为 null 表示「开一个新会话」，服务端会创建并返回它的 ID；
    # 传上一次的返回值则延续同一个会话，模型能看到之前的问答。
    #
    # 用 UUID 而不是字符串：格式不合法的 ID 会在这一层就被 422 挡掉，
    # 不会带着一个乱七八糟的字符串跑到数据库查询里。
    conversation_id: UUID | None = Field(
        default=None,
        description="会话 ID；不传或传 null 表示新建会话",
        examples=[None],
    )

    @field_validator("question", mode="before")
    @classmethod
    def _strip_question(cls, value: object) -> object:
        """先去首尾空白，再做长度校验。

        mode="before" 的理由同检索接口的 query：默认校验器在字段校验之后运行，
        那样 "   " 会以长度 3 通过 min_length=1，清洗后却成了空串 ——
        等于放一个空问题去调用 embedding 和 DeepSeek 两个付费接口。
        """
        return value.strip() if isinstance(value, str) else value


class AskResponse(BaseModel):
    """POST /api/knowledge-bases/{id}/ask 的响应体。"""

    # 本次问答所属的会话 ID。新建会话时这里是新生成的 ID，
    # 调用方把它存下来，下一轮原样传回来就能接上上下文。
    conversation_id: UUID = Field(description="会话 ID，下一轮请求带上它即可延续对话")
    question: str = Field(description="回显清洗后的问题")
    answer: str = Field(
        description="根据知识库内容生成的回答；资料不足时会明确说明无法确定",
        examples=["Milvus 是一个开源的向量数据库。"],
    )
    # 直接复用检索接口的结果模型：一条「问答用到的资料」和一条「检索结果」
    # 本来就是同一个东西（chunk_id / document_id / content / score），
    # 没有必要定义两个字段完全相同的类再让它们各自演化。
    # 将来问答的来源确实需要额外字段（比如「是否被引用」）时再拆也不迟。
    sources: list[SearchResultItem] = Field(
        default_factory=list,
        description="本次回答实际使用的检索结果，按相似度降序；没有检索到时为空列表",
    )
