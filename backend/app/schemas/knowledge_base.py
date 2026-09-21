"""知识库接口的请求 / 响应模型。"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

# 名称长度上限。刻意与数据库列宽（VARCHAR(255)）保持一致：
# 校验比列宽松的话，超长输入会一路走到数据库才报错，
# 而那个错误的措辞是给运维看的，不是给用户看的。
MAX_NAME_LENGTH: int = 255

# 描述用 Text 列（数据库没有长度限制），但接口仍然设上限。
# 理由和 name 不同 —— 这里不是为了对齐列宽，而是防止有人
# 往描述里塞一整篇文档：描述是给人快速了解这个库用的，不是内容存储。
MAX_DESCRIPTION_LENGTH: int = 2000


class KnowledgeBaseCreate(BaseModel):
    """POST /api/knowledge-bases 的请求体。"""

    name: str = Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
        description="知识库名称",
        examples=["向量数据库资料"],
    )
    description: str | None = Field(
        default=None,
        max_length=MAX_DESCRIPTION_LENGTH,
        description="知识库描述，可为空",
        examples=["Milvus 相关文档与笔记"],
    )

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, value: object) -> object:
        """先去首尾空白再校验长度。

        mode="before" 是关键：默认校验器在字段校验【之后】运行，
        那样 "   " 会以长度 3 通过 min_length=1，清洗后却成了空名 ——
        数据库里就会出现一个没有名字、界面上点不出区别的知识库。
        """
        return value.strip() if isinstance(value, str) else value

    @field_validator("description", mode="before")
    @classmethod
    def _normalize_description(cls, value: object) -> object:
        """把空白描述统一成 None。

        不清洗的话，库里会同时存在 None、""、"   " 三种「没有描述」，
        前端就得写三段判断才能渲染出同一个效果。
        """
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class KnowledgeBaseResponse(BaseModel):
    """单个知识库的对外表示。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(description="知识库 ID")
    name: str = Field(description="知识库名称")
    description: str | None = Field(default=None, description="知识库描述")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="最后更新时间")
    # 文档数量来自对 documents 表的实时统计，不是存在知识库表上的计数字段。
    # 存计数字段的话，每次上传/删除文档都得记得同步更新它，
    # 一旦有一处漏了，界面上的数字就会永远对不上，而且很难发现。
    document_count: int = Field(description="该知识库下的文档数量")
