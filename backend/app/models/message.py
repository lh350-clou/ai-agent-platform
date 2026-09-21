"""Message：对话里的一条消息。

每条消息只属于一个对话，role 区分它是用户说的、模型答的，还是系统预设的。
把 role 存成字段而不是拆成三张表，是因为对话历史回放时永远是
「按时间顺序把所有角色的消息拉出来」——存在一张表里天然就是有序的。
"""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    # 理由同 conversation.py：只在类型检查时导入，避免两个模块互相导入。
    from app.models.conversation import Conversation


class Message(Base):
    """消息表。"""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # 外键指向 conversations.id。这一列有两个额外设置，都是有实际原因的：
    #
    # ondelete="CASCADE" 是「数据库层」的删除规则：删掉一个对话时，
    # PostgreSQL 自己把它的消息一并删掉。它是 Conversation 那边 ORM cascade 的兜底 ——
    # 只要有人绕过 ORM 直接执行 DELETE FROM conversations，消息也不会变成孤儿数据。
    # 两处都写不是重复：ORM 那层管「通过 ORM 删」，这一层管「怎么删都算数」。
    #
    # index=True 会在这一列上建索引。这一点容易被忽略：
    # 外键列「不会」自动获得索引（MySQL 会隐式建，PostgreSQL 不会）。
    # 而下面查询几乎永远带着这个条件 ——「取某个对话的全部消息」是最常用的操作，
    # 没有索引就只能全表扫描，对话一多就明显变慢。
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # role 只会是 user / assistant / system。
    # 用字符串而不是数据库 enum 类型：以后加一种角色（比如 tool）时不用改数据库类型，
    # 取值校验交给上层的 Pydantic schema 做 —— 校验规则属于业务层，不该焊死在表结构里。
    role: Mapped[str] = mapped_column(String(20), nullable=False)

    # 正文用 Text（PG 里是 text，长度无上限），而不是 String(255)。
    # 一次带 RAG 引用的回答动辄几千字，用带长度的类型迟早会被截断或报错。
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # 消息不设 updated_at：消息是「已经发生的事实」，写进去就不会再改。
    # 留一个永远等于 created_at 的字段只会让读代码的人以为它有意义。
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    # 多对一的反向关系：每条消息属于一个对话。
    # 这里刻意不写 cascade —— 删除的传播方向是「从父到子」，
    # 删掉一条消息不应该对它所属的对话产生任何影响。
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        # 正文只截前 30 个字符：消息动辄上千字，整条打出来会刷屏，
        # 而调试时通常也只需要看一眼开头就能认出是哪条。
        return f"<Message id={self.id} role={self.role!r} content={self.content[:30]!r}>"
