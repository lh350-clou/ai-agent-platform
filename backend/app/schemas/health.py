"""健康检查接口的响应模型。"""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """GET /health 的返回体。"""

    # status 描述的是「Web 服务进程本身」，database 描述的是「下游依赖」。
    # 两者分开报，是为了让调用方能区分「服务没起来」和「服务起来了但数据库连不上」。
    status: str = Field(description="服务状态，正常时为 ok", examples=["ok"])
    app_name: str = Field(description="应用名称")
    version: str = Field(description="应用版本号")
    database: Literal["ok", "error"] = Field(
        description="PostgreSQL 连接状态：ok 表示 SELECT 1 执行成功",
        examples=["ok"],
    )
