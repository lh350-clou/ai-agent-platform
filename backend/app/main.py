"""FastAPI 应用入口。

本地启动（先 cd 到 backend/ 目录，这样 app 包才能被正确导入）：
    pip install -r requirements.txt
    uvicorn app.main:app --reload

启动后访问：
    http://127.0.0.1:8000/health                  健康检查
    http://127.0.0.1:8000/api/chat                对话接口（POST）
    http://127.0.0.1:8000/api/documents/upload    文档上传（POST, multipart）
    http://127.0.0.1:8000/api/documents/{id}      文档查询（GET）/ 删除（DELETE）
    http://127.0.0.1:8000/api/knowledge-bases/{id}/search   知识库检索（POST）
    http://127.0.0.1:8000/api/knowledge-bases/{id}/ask      RAG 问答（POST）
    http://127.0.0.1:8000/api/knowledge-bases/{id}/agent    知识库 Agent（POST）
    http://127.0.0.1:8000/docs                    自动生成的接口文档

数据库配置从仓库根目录的 .env 读取（见 app/core/config.py），
仓库里只有 .env.example 模板，需要自己复制一份并填上真实密码。
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.knowledge_bases import router as knowledge_bases_router
from app.api.qa import router as qa_router
from app.api.search import router as search_router
from app.core.config import settings
from app.core.database import check_connection, engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期钩子：yield 之前是启动逻辑，之后是关闭逻辑。

    启动时只「探测一次并记录日志」，不强制要求数据库可用。
    因为如果这里直接抛异常，数据库一挂服务就起不来，/health 也就返回不了任何东西，
    反而失去了它「报告数据库状态」的意义。真正的实时状态由 /health 负责。
    """
    # 日志里只打主机、端口、库名，绝不打 settings.database_url ——
    # 那个字符串里含密码，一旦进日志就等于泄漏了。
    logger.info(
        "正在探测 PostgreSQL 连接：%s:%s/%s",
        settings.POSTGRES_HOST,
        settings.POSTGRES_PORT,
        settings.POSTGRES_DB,
    )
    try:
        await check_connection()
        logger.info("PostgreSQL 连接正常")
    except Exception:
        logger.warning(
            "PostgreSQL 连接失败，服务仍会启动；请检查 .env 中的数据库配置，"
            "或确认容器是否在运行",
            exc_info=True,
        )

    yield

    # 进程退出前释放连接池，把所有连接还给 PostgreSQL。
    # 不做这一步，反复重启服务会慢慢耗尽数据库的最大连接数。
    await engine.dispose()
    logger.info("PostgreSQL 连接池已释放")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。

    把创建过程包在函数里，是为了让测试代码也能拿到一个干净的应用实例
    （将来写 pytest 时可以直接 create_app()）。
    """
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    # 跨域配置。前端跑在 5173 端口，和后端不是同一个源，
    # 浏览器默认会拦掉这类请求，必须在这里显式放行。
    #
    # 允许的来源来自配置（默认只有开发用的两个本地地址），
    # 不使用 "*"：这些接口没有鉴权，通配符等于让任意网站都能
    # 借用户的浏览器调它们，其中包括删知识库这种破坏性操作。
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        # 允许带 Cookie 之类的凭据。只有在 allow_origins 不是 "*" 时
        # 才能开启这一项（浏览器规范禁止二者同时使用）。
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册各模块的路由。后续新增模块时，在这里追加 include_router 即可，
    # 不需要改动已有代码。
    application.include_router(health_router)
    application.include_router(chat_router)
    application.include_router(documents_router)
    application.include_router(search_router)
    application.include_router(qa_router)
    application.include_router(agent_router)
    application.include_router(knowledge_bases_router)

    return application


# uvicorn 通过 "app.main:app" 找到的就是这个变量
app = create_app()
