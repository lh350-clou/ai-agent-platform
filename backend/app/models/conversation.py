"""Conversation：一次对话会话。

会话本身不存任何内容，内容都在 messages 表里 —— 它只是「消息的容器」。
拆成两张表而不是把消息塞进一个大字段，是为了让每条消息能单独被检索、
单独带上自己的 role 和时间戳，而「会话」这一层则负责标题、创建时间这类整体属性。
"""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    # 只在类型检查（mypy / IDE）时导入 Message，运行时靠 SQLAlchemy 按类名去注册表里找。
    # 如果这里写成真的 import，conversation.py 和 message.py 会互相导入，
    # Python 加载到一半就会因为「对方还没定义完」而报 ImportError。
    from app.models.message import Message


class Conversation(Base):
    """对话会话表。"""

    # 表名用复数，和 SQLAlchemy / Alembic 的惯例保持一致：
    # 一张表存的是「一堆同类实体」，复数读起来更自然（SELECT * FROM conversations）。
    __tablename__ = "conversations"

    # 主键用 UUID 而不是自增整数，原因有二：
    #   1. 自增 ID 会把业务规模暴露出去 —— 用户看到 id=1024 就知道平台一共只有一千条对话；
    #   2. 以后要做数据迁移、多库合并时，自增主键极容易撞号，UUID 不会。
    #
    # default=uuid.uuid4 是「Python 侧」生成：对象还没写进数据库时就已经有 id 了。
    # 这一点在两步写入时很有用 —— 先建对话拿到 id，再用这个 id 批量插入消息，
    # 全程不用为了拿主键而多查一次数据库。
    id: Mapped[uuid.UUID] = mapped_column(
        # PostgreSQL 原生 uuid 类型（16 字节），而不是存成 char(36) 字符串。
        # as_uuid=True 表示读写时都用 Python 的 uuid.UUID 对象，不用手动做字符串转换。
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # 标题允许为空。新建对话时用户往往还没想好名字，
    # 常见做法是先留空，等第一轮问答结束再用问题内容回填。
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # 时间统一用「带时区」的 datetime（对应 PG 的 timestamptz）。
    # 绝不用不带时区的 timestamp：同一个库里混进本地时间和 UTC，
    # 是后期最难排查的一类问题 —— 你永远说不清那条 2026-09-21 08:00 到底是哪个时区。
    #
    # 这里同时给了 Python 默认值和数据库默认值，两者分工不同：
    #   - default：正常走 ORM 插入时由 Python 填值，插完对象上立刻就有时间，
    #     不需要再回查一次数据库（异步场景下这点尤其重要，见 database.py 的 expire_on_commit 注释）；
    #   - server_default：兜底。万一有人绕过 ORM 直接写 SQL，也不会插进 NULL。
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    # updated_at 比 created_at 多一个 onupdate：每次 UPDATE 这一行时，
    # SQLAlchemy 会自动把它刷成当前时间，不需要业务代码手动赋值。
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    # 一对多：一个对话有多条消息。
    #
    # 这里没有写 relationship("Message", ...) 把目标类名写死，
    # 而是靠 Mapped[list["Message"]] 这个类型标注推导 —— 这正是 SQLAlchemy 2.x
    # typed ORM 的写法：类型标注本身就是配置。
    #
    # 两个 cascade 参数分工不同：
    #   - cascade="all, delete-orphan" 是 ORM 层的规则。
    #     delete 表示「删对话时连带删消息」；delete-orphan 表示「消息一旦从
    #     conversation.messages 里被移出去，就当孤儿删掉」，避免留下没有归属对话的脏数据。
    #   - passive_deletes=True 则是告诉 SQLAlchemy「别再把消息一条条加载进内存再逐条 DELETE 了」：
    #     外键上已经声明了 ON DELETE CASCADE，交给数据库一条语句删干净，消息多时快得多。
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        # 只列关键字段，方便在调试器里一眼认出对象；不要把 messages 也打出来，
        # 那样会触发一次懒加载查询，也可能刷屏。
        return f"<Conversation id={self.id} title={self.title!r}>"
