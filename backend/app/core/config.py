"""应用配置：统一从环境变量 / .env 读取。

约定（见 CLAUDE.md）：代码里只引用本模块的 settings 对象，
不在业务代码中直接调用 os.environ，更不允许硬编码密钥。
"""

from pathlib import Path
from urllib.parse import quote_plus

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py 位于 backend/app/core/ 下，向上四级即仓库根目录。
# 这里用绝对路径而不是 ".env" 相对路径，是为了保证无论从哪个目录启动服务
# （根目录执行 uvicorn backend.app.main:app，或 backend/ 下执行 uvicorn app.main:app），
# 读到的都是同一份仓库根目录的 .env。
BASE_DIR: Path = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """全部配置项。

    新增配置时：在这里加字段（给默认值即可），并同步更新根目录的 .env.example。
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # .env 中出现本类未声明的变量时忽略而不报错，方便本地临时调试
        extra="ignore",
    )

    # ---- 应用基础信息 ----
    APP_NAME: str = "AI 智能知识库 Agent 平台"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # ---- 服务监听地址 ----
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ---- PostgreSQL 数据库 ----
    # 拆成多个字段而不是一个完整的 DATABASE_URL，好处是密码能单独用 SecretStr 包起来：
    # 即使有人打印了整个 settings 对象，密码也只会显示成 **********，
    # 而不是明文躺在连接串里。
    #
    # HOST 的取值取决于「谁在连数据库」：
    #   本机直接跑 uvicorn  -> localhost（走宿主机映射出来的 5432 端口）
    #   backend 也进 Docker -> 容器名，例如 pg16（容器之间不经过宿主机端口映射）
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    # 真实密码只写在 .env 里，这里给空字符串只是为了让「还没建 .env」时应用也能启动，
    # 从而让 /health 能明确告诉你「连不上数据库」，而不是在导入配置时就崩掉。
    POSTGRES_PASSWORD: SecretStr = SecretStr("")
    POSTGRES_DB: str = "agent_db"

    @property
    def database_url(self) -> str:
        """拼接 SQLAlchemy 异步连接串。

        形如 postgresql+asyncpg://用户:密码@主机:端口/库名。

        两个容易踩的点：
        1. 驱动必须显式写成 +asyncpg。不写时 SQLAlchemy 默认去找同步驱动 psycopg2，
           而本项目没装它，会直接连接失败。
        2. 用户名和密码要做 URL 编码。密码里如果出现 @ : / # 这类字符，
           不编码会把连接串的结构撑坏（例如 @ 会被当成「主机名开始」）。
        """
        user = quote_plus(self.POSTGRES_USER)
        password = quote_plus(self.POSTGRES_PASSWORD.get_secret_value())
        return (
            f"postgresql+asyncpg://{user}:{password}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


# pydantic-settings 在导入时会自动读取 .env，并让环境变量覆盖默认值。
settings = Settings()
