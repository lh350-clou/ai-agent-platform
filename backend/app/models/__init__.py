"""数据库模型（PostgreSQL）。

这里必须 import 每一个模型类 —— 哪怕本模块自己一行都用不到它们。

原因见 alembic/env.py：autogenerate 是拿 Base.metadata 和数据库现状做对比来生成迁移的，
而只有被 import 过的模型类，才会在定义时把自己的表注册到 Base.metadata 上。
少写一个 import，autogenerate 就会认为「这张表不该存在」，从而生成一份删表迁移。
"""

from app.models.base import Base
from app.models.conversation import Conversation
from app.models.knowledge_base import KnowledgeBase
from app.models.message import Message

__all__ = ["Base", "Conversation", "KnowledgeBase", "Message"]
