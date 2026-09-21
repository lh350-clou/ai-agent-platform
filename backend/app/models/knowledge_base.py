"""KnowledgeBase：知识库。

本阶段只建「知识库」这个容器本身，还没有文档表和向量切片表 ——
那两张表会在接入 Milvus 的 RAG 阶段再加，通过 knowledge_base_id 关联过来。

之所以现在就先有这一层，是因为向量库里存的每个向量都必须能回答
「我属于哪个知识库」：向量库只认 ID，没法像关系库那样做关联查询。
把知识库的元信息（名字、描述）留在 PostgreSQL，把向量留在 Milvus，
两边靠这个表的主键对上，是整个 RAG 检索链路的基础。
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class KnowledgeBase(Base):
    """知识库表。"""

    __tablename__ = "knowledge_bases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # 名字不允许为空：知识库是给人挑的，列表里出现一堆无名条目没法用。
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # 描述可以为空：建库时常常先起个名字就用了，描述是后来才补的。
    # 类型用 Text 而不是 String(255) —— 描述可能写得很长（说明这个库收了哪些资料、
    # 适合问什么类型的问题），没必要设一个会突然撞上的长度上限。
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 时间戳的写法和 Conversation 完全一致：TimeZone-aware + Python 默认值 + 数据库兜底，
    # 详见 conversation.py 里的注释。
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return f"<KnowledgeBase id={self.id} name={self.name!r}>"
