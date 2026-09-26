"""PostgreSQL 持久化的回归：ORM 真的能写、真相能读回来、库里确实什么都不留。

两条用例守的东西不一样：

    test_orm_round_trip_is_rolled_back —— 模型定义和数据库表还对得上吗？
    test_database_schema_covers_all_models —— 模型里定义的表真的都建出来了吗？

为什么两条都要有：前者在「列名改了但迁移没跑」时会因为 UnknownColumn 报错，
后者在「新加了表但忘了生成迁移」时才会红。只留前者的话，
一张没被任何用例碰过的新表可以一直是缺的。

**不写生产数据、不建表**：所有写入都在一个不提交的事务里做完然后回滚，
schema 那条只读 information_schema。所以这个文件可以在任何环境反复跑，
跑完数据库和跑之前一模一样。
"""

import uuid

import pytest
from sqlalchemy import func, inspect, select

from app.core.database import async_session_factory, check_connection, engine
from app.models import Base
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.message import Message
from tests.regression.availability import ServiceStatus

pytestmark = [pytest.mark.integration, pytest.mark.regression]


async def test_orm_round_trip_is_rolled_back(services: ServiceStatus) -> None:
    """写一行 → 读回来 → 回滚 → 确认库里没有。

    最后一组断言是重点：它同时证明了两件事 ——
    「这些字段真的落到了数据库」（否则中间那组读不到），
    和「这条用例真的没留下东西」（否则最后一组会读到残留）。
    只断言前半段的话，一个悄悄 commit 掉的实现会让测试通过，
    而它每次运行都往生产库里塞一行垃圾数据。
    """
    services.require_postgres()

    kb_id = uuid.uuid4()
    document_id = uuid.uuid4()

    async with async_session_factory() as session:
        session.add(KnowledgeBase(id=kb_id, name="regression-临时知识库"))
        session.add(
            Document(
                id=document_id,
                knowledge_base_id=kb_id,
                name="regression.txt",
                file_path="/tmp/regression.txt",
                file_type="txt",
            )
        )
        # flush 会把 INSERT 真的发到数据库（外键约束也在这里被真正校验），
        # 但事务还没提交 —— 这正是我们要的效果：验真，但不落盘。
        await session.flush()

        rows = (
            await session.execute(
                select(Document)
                .where(Document.knowledge_base_id == kb_id)
                # populate_existing 强制拿数据库返回的值覆盖内存对象。
                # 不加它的话，SQLAlchemy 会直接把身份映射里那个 Python 对象还给你，
                # 断言就变成了「我自己刚放进去的东西还在」，跟数据库没关系。
                .execution_options(populate_existing=True)
            )
        ).scalars().all()

        assert [row.id for row in rows] == [document_id]
        assert rows[0].name == "regression.txt"
        # status / chunk_count 是模型里的默认值，落库后要能读出来 ——
        # 这两个字段的 server_default 写错时，只有真写一次才发现得了。
        assert rows[0].status == "pending"
        assert rows[0].chunk_count == 0

    # 退出 async with 时没有 commit → 事务回滚。
    async with async_session_factory() as session:
        assert await session.get(KnowledgeBase, kb_id) is None
        remaining = (
            await session.execute(
                select(func.count())
                .select_from(Document)
                .where(Document.knowledge_base_id == kb_id)
            )
        ).scalar_one()
        assert remaining == 0, "用例在数据库里留下了数据"


async def test_database_schema_covers_all_models(services: ServiceStatus) -> None:
    """模型里定义的每一张表，数据库里都必须存在。

    它抓的是「加了模型但没生成/没执行迁移」——那种情况下代码能 import、
    应用能启动，只有在真去查那张表时才报错，而那时往往已经在别的用例里了。
    这类问题的表现是「本地好好的，部署一次就 500」，因为本地库是手工建过的。
    """
    services.require_postgres()

    # 读一次 app.models（它本身的作用就是把所有模型类 import 进来）。
    # 只有被 import 过的模型才会把表注册到 Base.metadata 上。
    defined = set(Base.metadata.tables)

    async with engine.connect() as conn:
        existing = set(
            await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
        )

    missing = sorted(defined - existing)
    assert not missing, (
        f"这些表在模型里定义了，但数据库里没有：{missing}。"
        "通常是加完模型后忘了生成迁移或没执行 alembic upgrade head。"
    )


async def test_message_and_conversation_tables_are_writable(services: ServiceStatus) -> None:
    """会话与消息这两张表按外键串起来写一次（同样回滚）。

    单列一条是因为它们是「多轮对话持久化」这条能力的载体，
    而外键关系写反了、级联方向错了这类问题，只有真写一次才暴露得出来。
    """
    services.require_postgres()

    kb_id = uuid.uuid4()
    conversation_id = uuid.uuid4()

    async with async_session_factory() as session:
        session.add(KnowledgeBase(id=kb_id, name="regression-临时知识库"))
        await session.flush()
        session.add(
            Conversation(
                id=conversation_id,
                knowledge_base_id=kb_id,
                title="regression",
            )
        )
        session.add(
            Message(
                conversation_id=conversation_id,
                role="user",
                content="回归用的一条消息",
            )
        )
        await session.flush()

        count = (
            await session.execute(
                select(func.count())
                .select_from(Message)
                .where(Message.conversation_id == conversation_id)
            )
        ).scalar_one()
        assert count == 1

    async with async_session_factory() as session:
        assert await session.get(Conversation, conversation_id) is None


async def test_check_connection_succeeds(services: ServiceStatus) -> None:
    """应用自己那条探活查询（SELECT 1）还能跑通。

    /health 依赖它，而它比上面几条更底层：它走的是 engine.connect()
    而不是会话工厂，两者在连接池上的配置并不相同。
    """
    services.require_postgres()

    await check_connection()
