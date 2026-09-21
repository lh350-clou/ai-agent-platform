"""知识库检索接口的请求 / 响应模型。

用 Pydantic 定义契约的理由同其它 schema：自动校验、自动生成 /docs 文档、
字段说明集中在一处。这里的校验还有一层实际作用 —— query 和 top_k 的边界
如果不在这一层挡住，就会一路传到 embedding API 和 Milvus 上，
变成「拿钱换来的报错」（一次超长 query 照样会消耗一次 embedding 调用）。
"""

from pydantic import BaseModel, Field, field_validator

# query 的最大长度（字符）。
#
# 这个上限不是为了防「算力被拖慢」，而是因为超长 query 会直接变成一次
# 真实的 embedding 调用 —— 那是要花钱的。设一个上限，让明显不合理的请求
# 在本地就被 422 挡掉，不要让它走到外部 API。
#
# 2000 字符对「一个问题」来说已经非常宽松：正常提问几十字，长一点的一段话
# 也就几百字。真正的长文本检索是另一种用法（文档对文档的相似度比较），
# 不在本接口的范围内。
MAX_QUERY_LENGTH: int = 2000

# top_k 的取值范围。
# 下限 1：返回 0 条没有意义，调用方应该知道自己在要什么。
# 上限 20：这是个「喂给大模型」的接口，返回越多，拼进提示词的内容越长、
# 成本和延迟越高。20 已经远超一般的需要，再大基本是在浪费上下文窗口。
MIN_TOP_K: int = 1
MAX_TOP_K: int = 20
DEFAULT_TOP_K: int = 5


class SearchRequest(BaseModel):
    """POST /api/knowledge-bases/{id}/search 的请求体。"""

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description="检索用的问题文本",
        examples=["什么是向量数据库"],
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=MIN_TOP_K,
        le=MAX_TOP_K,
        description=f"返回的最大结果数，范围 {MIN_TOP_K}~{MAX_TOP_K}",
        examples=[5],
    )

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query(cls, value: object) -> object:
        """先去掉首尾空白，再交给长度校验。

        mode="before" 是这里的关键：默认的校验器在字段校验【之后】运行，
        那样 "   " 会先被 min_length=1 判为合格（长度是 3），
        清洗后却变成空串，等于放进去一个空 query 去调 embedding API。
        放在 before 阶段，清洗后的空串会正常触发 min_length 报错、返回 422。
        """
        return value.strip() if isinstance(value, str) else value


class SearchResultItem(BaseModel):
    """一条检索结果。"""

    chunk_id: str = Field(description="切片 ID")
    document_id: str = Field(description="所属文档 ID")
    content: str = Field(description="切片正文")
    # 直接沿用 Milvus 返回的相似度：用 COSINE 时它其实是余弦相似度，
    # 越接近 1 越相似。这里【不重新计算、也不改类型】——
    # 一旦在 API 层做转换，就多出一处可能和检索层不一致的逻辑，
    # 而且调用方再也无法把它们对起来。
    score: float = Field(description="余弦相似度，越大越相似")


class SearchResponse(BaseModel):
    """POST /api/knowledge-bases/{id}/search 的响应体。"""

    query: str = Field(description="回显清洗后的查询文本")
    # 顺序就是 Milvus 返回的顺序（相似度降序），API 层不再排序 ——
    # 排序规则属于检索层，挪到这里会让两处逻辑各说各话。
    results: list[SearchResultItem] = Field(
        default_factory=list,
        description="检索结果，按相似度从高到低；没有匹配时为空列表",
    )
