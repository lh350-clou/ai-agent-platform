"""健康检查路由：供容器编排 / 负载均衡探活使用。"""

import logging
from typing import Literal

from fastapi import APIRouter

from app.core.config import settings
from app.core.database import check_connection
from app.schemas.health import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["健康检查"])


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check() -> HealthResponse:
    """返回服务存活状态，以及 PostgreSQL 是否可用。

    数据库检查失败时依然返回 200，只是把 database 标成 error。
    这样探活方能区分「进程没起来（连不上端口）」和「进程起来了但数据库连不上」；
    如果这里直接抛 500，这两种故障在调用方看来就是一样的了。
    """
    try:
        # SELECT 1 是最轻量的探活方式：不碰业务表，还没建表时也能用
        await check_connection()
        database_status: Literal["ok", "error"] = "ok"
    except Exception:
        # 异常详情只写日志，不放进响应体：报错信息里可能带上主机名、端口、
        # 甚至认证细节，而健康检查接口通常不鉴权，回显出去属于信息泄漏。
        # 需要排查时看 uvicorn 的日志即可。
        logger.exception("健康检查：PostgreSQL 连接失败")
        database_status = "error"

    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        database=database_status,
    )
