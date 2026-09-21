"""文档接口的请求 / 响应模型。

用 Pydantic 模型而不是裸 dict 定义契约，理由同 schemas/chat.py：
自动校验、自动生成 /docs 文档、字段说明集中在一处。
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# 文档的处理状态。
#
# 这里用 Literal 把取值收死，是刻意补上数据库没有的那道约束：
# Document.status 在库里是普通 VARCHAR（当初为了「加状态不用改表」才不用 enum）。
# 好处是灵活，代价是写错状态名数据库不会拦 —— 比如手滑写成 "completd"，
# 插入照样成功，只是这条文档永远查不出「已完成」。
# 所以把校验放在业务层：接口进出这一层就挡住，数据库保持灵活。
DocumentStatus = Literal["pending", "processing", "completed", "failed"]


class DocumentResponse(BaseModel):
    """文档的对外表示。

    model_config 里的 from_attributes=True 是让 FastAPI 能直接把
    SQLAlchemy 的 Document 对象转成本模型：它允许按属性名取值，
    而不是要求传一个 dict。没有这一行，返回 ORM 对象时会报校验错误。
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(description="文档 ID")
    knowledge_base_id: UUID = Field(description="所属知识库 ID")
    name: str = Field(description="文档名称（原始文件名）")
    file_type: str = Field(description="文件类型，目前为 txt", examples=["txt"])
    status: DocumentStatus = Field(
        description="处理状态：pending / processing / completed / failed",
        examples=["completed"],
    )
    chunk_count: int = Field(description="成功写入向量库的切片数量")
    error_message: str | None = Field(
        default=None,
        description="失败原因，仅当 status 为 failed 时有值",
    )
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="最后更新时间")


class DocumentUploadResponse(DocumentResponse):
    """POST /api/documents/upload 的返回体。

    字段目前和 DocumentResponse 完全一致 —— 上传成功后直接把创建好的文档返回。
    之所以单独定义一个名字而不是直接复用：上传接口将来大概率会带上
    「只有它才有」的信息（比如后台任务 ID、排队位置），
    到那时它会和查询接口的返回体分叉。现在把名字分开，
    那次改动就只影响上传接口，不会波及查询接口的契约。
    """
