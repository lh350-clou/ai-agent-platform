"""数据库基础设施：异步引擎、会话工厂，以及 FastAPI 的会话依赖。

约定（见 CLAUDE.md）：业务代码只使用本模块导出的 engine / async_session_factory / get_db，
不在别处自己创建引擎，也不直接调用 asyncpg。
"""

from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings

# ---- 引擎 ----
# 引擎内部维护一个连接池，所以整个进程只应该有一个实例：
# 如果每个请求都 create_async_engine，连接池会被反复创建和销毁，比不用池还慢。
# 另外 create_async_engine 是惰性的 —— 导入本模块并不会真的去连数据库。
engine: AsyncEngine = create_async_engine(
    settings.database_url,
    # 本地开发（DEBUG=true）时把实际执行的 SQL 打印到日志，方便排查
    echo=settings.DEBUG,
    # 取用连接前先 ping 一次。数据库重启或空闲超时会把连接掐断，
    # 没有这个选项时，池子里那些已经失效的连接会导致偶发的报错。
    pool_pre_ping=True,
)

# ---- 会话工厂 ----
# 每个请求从连接池借一个连接，开一个 AsyncSession，用完归还。
#
# expire_on_commit=False 在异步场景下几乎是必须的：
# 默认值 True 会让 commit() 之后对象上的所有属性失效，之后再读 obj.id 这类
# 最简单的属性都会触发一次新的数据库查询。同步代码里这只是变慢，
# 但在异步函数里，这种「隐式发起的 IO」会直接抛 MissingGreenlet 异常。
# 关掉它，commit 之后的对象仍然可以安全读取。
async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：为每个请求提供一个数据库会话。

    用 yield 而不是 return，是为了让 FastAPI 在请求处理结束后回到这里，
    无论请求成功还是抛异常，都会执行退出逻辑关闭会话、把连接还给连接池。

    用法（等写业务接口时）：在路由函数参数里写 db: AsyncSession = Depends(get_db)。
    本阶段还没有业务表，所以暂时没有调用方。
    """
    async with async_session_factory() as session:
        yield session


async def check_connection() -> None:
    """执行一次最轻量的查询，确认数据库真的可用。

    只跑 SELECT 1：不碰任何业务表，所以在还没建表的阶段也能用来探活。
    连不上时直接把异常抛给调用方，由调用方决定怎么呈现（见 api/health.py）。
    """
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
