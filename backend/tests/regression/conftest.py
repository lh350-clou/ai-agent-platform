"""回归套件的公共 fixture。

这里只放「跨模块复用」的东西：服务探活、临时知识库。单个文件里只用一个
用例的辅助函数就写在那个文件里 —— 没有第二个调用方就不要提前抽出来。
"""

import logging
import uuid
from collections.abc import AsyncGenerator, Callable

import pytest

from app.core.database import engine
from app.services import vector_store
from tests.regression.availability import ServiceStatus

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
async def dispose_engine_after_each_test() -> AsyncGenerator[None, None]:
    """每条用例跑完后释放数据库连接池。

    为什么必须有：pytest-asyncio 默认【每条用例一个事件循环】，而
    core/database.py 里的 engine 是进程级的单例，它的连接池里存着
    「在某个循环里建立」的连接。上一条用例的循环一关，那条连接就废了，
    下一条用例再取到它，报错是连接池内部抛的 `RuntimeError: Event loop is closed`
    —— 位置在 sqlalchemy / asyncpg 里，看起来和被测代码毫无关系。

    在用例结束时（此时循环还开着）主动 dispose，池子清空，就不会有连接
    跨循环存活。生产环境不存在这个问题：一个进程一个循环，服务停止时
    lifespan 里本来就会 dispose 一次。

    autouse 覆盖本目录全部用例：离线的那些调用它只是空操作
    （没连过数据库的池子本来就是空的）。
    """
    yield
    await engine.dispose()


@pytest.fixture(scope="session")
def services() -> ServiceStatus:
    """本轮回归内只探活一次的外部依赖状态。

    session 级而不是 function 级：探活有 1 秒超时，每条用例都探一遍，
    光是「服务在不在」就会成为整个套件里最慢的一环。
    """
    return ServiceStatus()


@pytest.fixture
async def new_kb(services: ServiceStatus) -> AsyncGenerator[Callable[[], str], None]:
    """工厂 fixture：每次调用返回一个新的 knowledge_base_id，用例结束后清空。

    两件事必须由它保证：

    1. **隔离**：ID 每次随机（uuid4），不会碰到任何真实知识库的数据，
       也不会和上一次运行残留的向量混淆。
    2. **清理**：退出时按 ID 删除 Milvus 里这一批向量。放在 yield 之后的
       finally 语义里，所以用例【失败时同样会清理】—— 清理这件事依赖
       「用例跑成功了」是最不可靠的写法，而残留向量会一直堆在库里。

    做成「工厂」而不是「一个 ID」，是因为「知识库隔离」这条能力本身
    要靠两个不同的库来验：只给一个 ID，就没法表达「在 A 库里查不到 B 库的数据」。

    依赖 Milvus + Embedding：本 fixture 的每个调用方都要么入库、要么检索，
    两者都要向量化。DeepSeek 不在这里要求 —— 检索层根本不碰模型。
    """
    services.require_milvus()
    services.require_embedding()

    created: list[str] = []

    def _new() -> str:
        kb_id = str(uuid.uuid4())
        created.append(kb_id)
        return kb_id

    try:
        yield _new
    finally:
        for kb_id in created:
            try:
                removed = await vector_store.delete_by_knowledge_base_id(kb_id)
                logger.info("回归清理：知识库 %s 的向量已删除 %d 条", kb_id, removed)
            except Exception:
                # 清理失败要吼一声，但不能改变用例本身的结果 ——
                # 一个善后动作的失败不该把「能力是否被破坏」的结论顶掉。
                # 和 ingest.py 的 _discard_partial 是同一个取舍。
                logger.warning("回归清理失败：知识库 %s 的向量可能残留在 Milvus", kb_id)
