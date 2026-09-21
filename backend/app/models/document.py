"""Document：知识库下的一份文档。

这张表记录的是「文档这份资料本身的业务信息」—— 叫什么、存在哪、处理到哪一步了。
它不存文件内容，也不存切分后的片段：原文留在磁盘上，向量留在 Milvus 里，
chunk 的正文与向量由后续的 chunks 表 / Milvus collection 承担。

一张表只回答一个问题的好处是：文档处理失败时，这张表能明确告诉你
「哪份文件、失败在哪一步、错在哪」，而不用去翻日志。
"""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    # 只在类型检查（mypy / IDE）时导入，运行时靠 SQLAlchemy 按类名去注册表里找。
    # 理由同 conversation.py：真写成 import 会让两个模块互相导入，
    # Python 加载到一半就会因为「对方还没定义完」而报 ImportError。
    from app.models.knowledge_base import KnowledgeBase


class Document(Base):
    """文档表。"""

    __tablename__ = "documents"

    # 主键用 UUID，理由同 Conversation：不暴露业务规模，且多库合并时不会撞号。
    # default=uuid.uuid4 是 Python 侧生成 —— 对象还没写库就已经有 id 了。
    # 这一点在本项目里尤其有用：入库流程要用 document_id 去拼 chunk_id
    # （{document_id}-chunk-000001），必须在插入 Milvus 之前就拿到它。
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # 外键指向所属知识库。两个设置都是有实际作用的：
    #
    # ondelete="CASCADE" 是数据库层的删除规则：删掉一个知识库时，
    # PostgreSQL 自己把它下面的文档一并删掉。它和 KnowledgeBase 那边的
    # ORM cascade 是配合关系而不是重复 —— ORM 那层管「通过 ORM 删」，
    # 这一层管「怎么删都算数」（比如有人直接执行 DELETE FROM knowledge_bases）。
    #
    # index=True 会在这一列上建索引。这点容易忽略：外键列「不会」自动获得索引
    # （MySQL 会隐式建，PostgreSQL 不会）。而「列出某个知识库下的所有文档」
    # 是最常用的查询，没有索引就只能全表扫描。
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # 文档名（给人看的）。用文件原始名或用户自定义的名称，不参与寻址。
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # 服务器上的文件路径（给程序用的）。寻址一律走这个字段，而不是 name ——
    # name 可能重名、可能含斜杠、可能被用户改过，拿它当路径迟早出事。
    #
    # 长度给到 1024 而不是 255：路径由「存储根目录 + 知识库 UUID + 文档 UUID + 扩展名」
    # 拼成，如果部署时把根目录设成绝对路径，很容易超过 255。
    # 这个字段没有索引，长度对性能没有影响。
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)

    # 文件类型（txt / pdf / ...），当前主要支持 txt。
    # 用字符串而不是数据库 enum：以后加新格式不用改数据库类型，取值校验交给上层。
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)

    # 处理状态：pending / processing / completed / failed。
    #
    # 为什么需要状态字段：解析 + 切分 + 向量化是秒级到分钟级的耗时操作，
    # 不可能塞在 HTTP 请求里同步做完。有了它，上传接口可以先记一条 pending
    # 就立刻返回，由后台任务推进状态，前端轮询即可。
    #
    # 同样用 String 而不是 enum：状态集合大概率还会增加
    # （比如 partial、cancelled），enum 每次加值都要改数据库类型，
    # 而这类「流程状态」恰恰是变化最频繁的。
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        # 数据库侧兜底：万一有人绕过 ORM 直接插入，也不会写进 NULL。
        server_default="pending",
    )

    # 成功入库的 chunk 数量。处理完成后回填，用于「上传后立刻展示结果」。
    #
    # 注意：这个值应当以 ingest_txt() 的返回值为准（它返回的是真实写入 Milvus 的数量），
    # 而不是切分出来的片段数 —— 两者在部分失败时可能不一致。
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        # 用 text("0") 而不是字符串 "0"：这一列是整数类型，
        # 显式给出类型可以避免迁移脚本里生成 DEFAULT '0' 这种看着别扭的写法。
        server_default=text("0"),
    )

    # 失败原因。只有 status='failed' 时才有值，所以 nullable。
    # 用 Text 而不是 String(n)：异常信息动辄几百字（含调用栈摘要），
    # 定长类型迟早会被截断，而截断掉的恰好是最关键的那几行。
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 时间戳写法和其它表完全一致：TimeZone-aware + Python 默认值 + 数据库兜底，
    # 详见 conversation.py 里的注释。
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    # updated_at 比 created_at 多一个 onupdate：每次 UPDATE 这一行时自动刷新，
    # 业务代码不需要手动赋值。状态从 pending 走到 completed 时它就变了。
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    # 多对一的反向关系：每份文档属于一个知识库。
    # 这里不写 cascade —— 删除的传播方向是「从父到子」，
    # 删掉一份文档不应该影响它所属的知识库。
    knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="documents")

    def __repr__(self) -> str:
        # 只列关键字段，方便在调试器里一眼认出对象。
        # 不打印 file_path 全文（可能很长），只留文件名和状态。
        return (
            f"<Document id={self.id} name={self.name!r} "
            f"status={self.status!r} chunks={self.chunk_count}>"
        )
