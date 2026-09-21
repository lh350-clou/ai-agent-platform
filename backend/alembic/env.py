"""Alembic 迁移环境配置。

这是 Alembic 的入口脚本：无论执行 upgrade / downgrade 还是 revision，
Alembic 都会先导入本文件，由它决定两件事 ——
  1. 连哪个数据库（url / engine）；
  2. 拿谁的 metadata 和数据库现状做对比（autogenerate 的依据）。

本项目的两条硬性约定（见 CLAUDE.md）：
  - 数据库地址只能来自 app.core.config.settings，绝不写死在 alembic.ini 里，
    否则密码就会跟着配置文件一起进版本库；
  - 使用项目现有的异步引擎（asyncpg 驱动），而不是另起一套同步连接。
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy.engine import Connection

from alembic import context

from app.core.config import settings

# app.core.database 里的 engine 就是「整个应用共用的那个 AsyncEngine」，
# 它内部用的正是 settings.database_url。
# 迁移脚本直接复用它，可以保证「迁移改的库」和「服务读写的库」永远是同一个 ——
# 如果这里再单独拼一个连接串，很容易出现服务连 A 库、迁移改 B 库的隐蔽错误。
from app.core.database import engine

# 未来的业务模型必须被导入，表定义才会注册到 Base.metadata 上。
# 只写 `from app.models.base import Base` 是不够的：那只能拿到空的元数据容器，
# 具体的模型类没被 import 过就不会执行，autogenerate 会以为「这些表不存在」，
# 进而生成一份想要删表的迁移脚本。所以新增模型后，要记得在 app/models/__init__.py
# 里 import 一次，下面的 `import app.models` 负责把整包加载起来。
import app.models  # noqa: F401
from app.models.base import Base

# Alembic 的 Config 对象，对应 alembic.ini 里的内容
config = context.config

# 按 alembic.ini 的 [loggers] 等配置初始化 Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# autogenerate 的「对照物」：拿它和数据库里现存的结构做 diff，得出该生成什么迁移。
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式：不连数据库，只把「将要执行的 SQL」打印出来。

    用法：alembic upgrade head --sql > migration.sql
    适合生产环境 —— 由 DBA 审核 SQL 后再手动执行，而不是让程序直接改库。
    因为没有真实连接，这里只需要一个 url 字符串。
    """
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """在一条已经建立好的连接上执行迁移。

    这是个同步函数，但它操作的是异步连接 —— 由下面的 run_sync 负责调用，
    SQLAlchemy 会在异步连接所在的线程里安全地把它跑起来。
    Alembic 内部是同步写法，所以必须这样过渡一层。
    """
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """在线模式：真正连上数据库执行迁移。"""
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)

    # 迁移是一次性任务，跑完就把连接池释放掉。
    # 不释放的话，池里的连接会一直被本进程占着，直到进程退出才归还给数据库。
    await engine.dispose()


def run_migrations_online() -> None:
    """命令行的入口：Alembic 是同步调用的，这里用一个事件循环把异步逻辑跑完。"""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
