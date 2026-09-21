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

    # ---- DeepSeek 大模型 ----
    # DeepSeek 走 OpenAI 兼容接口，所以这三个配置项和官方 openai SDK 的参数一一对应，
    # 名字也刻意取得和 SDK 一致（api_key / base_url / model），减少对应关系的记忆成本。
    #
    # 为什么用 SecretStr 而不是 str：SecretStr 的 __str__ / __repr__ 只会输出 **********，
    # 所以哪怕有人不小心 print(settings)、或者把 settings 整个丢进日志，
    # 密钥也不会跟着泄漏。要用真实值时必须显式调用 .get_secret_value()，
    # 于是「代码里哪几处碰了明文密钥」一搜就能搜出来。
    #
    # 这里给空字符串默认值，是为了让「还没配 Key」时应用照样能启动、能被 import，
    # 真正的报错推迟到第一次调用模型时（见 services/llm.py），
    # 而不是在导入配置阶段就把整个服务拦死。
    DEEPSEEK_API_KEY: SecretStr = SecretStr("")
    # DeepSeek 官方 API 地址。SDK 会自己在这后面拼 /chat/completions，
    # 所以这里不要写成 .../v1/chat/completions 这种带路径的形式。
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    # deepseek-chat 是通用对话模型（deepseek-reasoner 是推理模型，本项目暂不用）。
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # ---- 硅基流动 Embedding ----
    # 为什么向量化不用 DeepSeek：DeepSeek 官方 API 只有 chat / reasoner 这类生成模型，
    # 没有 embedding 接口（调用 /embeddings 会返回 404）。
    # 所以 RAG 的两半分别由两家承担：「把文本转向量」用硅基流动，「根据检索结果生成答案」用 DeepSeek。
    #
    # 硅基流动同样是 OpenAI 兼容接口，所以下面两个配置项和 openai SDK 的参数一一对应，
    # 调用侧可以复用同一个 SDK，只是把 base_url 换掉。
    SILICONFLOW_API_KEY: SecretStr = SecretStr("")
    # 这个地址带 /v1 后缀，和上面 DEEPSEEK_BASE_URL 的写法不一样 ——
    # 各家的路径约定不同，不是笔误。SDK 会在这后面拼 /embeddings。
    SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"

    # 向量模型名。做成配置而不写死在调用处，将来换模型只改 .env，不用动代码。
    SILICONFLOW_EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # 向量维度。显式配置出来不是为了方便改，恰恰相反 —— 是要让「改错」这件事尽早暴露：
    # Milvus collection 的维度在创建时就固定了，建完改不了，只能重建集合并把全部文档
    # 重新向量化一遍。如果换模型后维度对不上却没人察觉，检索会照常返回结果，
    # 只是那些相似度全是无意义的 —— 这种错误极难排查。
    # 有了这个配置项，就能在写入前拿它和 collection 的实际维度比一次，不一致就直接报错。
    # bge-m3 的输出维度是 1024。
    EMBEDDING_DIM: int = 1024

    # ---- 文件上传限制 ----
    # 单个上传文件的大小上限，单位字节，默认 10 MiB。
    #
    # 放在配置里而不是写成接口内的字面量，有两个原因：
    # 一是部署环境不同上限也不同（本地开发可以松，公网部署必须紧），
    # 二是测试需要把它临时调小，才能在不构造 10 MiB 真实数据的前提下
    # 覆盖「恰好等于上限」「超过 1 字节」这些边界情况。
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024

    # ---- MCP（Model Context Protocol）----
    # MCP Server 的启动方式。
    #
    # 这里用「模块名」而不是「完整的命令行」作为配置，是因为 MCP Client 通过
    # stdio 与之通信：它需要自己拉起 Server 进程，而启动用的解释器必须是
    # 当前这个（sys.executable）—— 否则换个环境就会出现「命令能跑但依赖不对」。
    # 把解释器交给代码去填、只把「跑哪个模块」暴露成配置，既保留了可配置性，
    # 又不会因为配错解释器而启动失败。
    MCP_SERVER_MODULE: str = "app.mcp_server.server"

    # 单次 MCP 工具调用的超时（秒）。
    # 必须有这个上限：Server 是本机子进程，万一它卡死（死循环、等待输入），
    # 没有超时的话整个 Agent 请求会一直挂在那里，直到 HTTP 层超时。
    MCP_TOOL_TIMEOUT_SECONDS: float = 15.0

    # ---- 跨域（CORS）----
    # 允许访问后端的前端地址，逗号分隔。
    #
    # 用逗号分隔的字符串而不是 JSON 数组：写环境变量时
    # CORS_ALLOW_ORIGINS=http://a,http://b 比
    # CORS_ALLOW_ORIGINS=["http://a","http://b"] 顺手得多，
    # 而后者只要少一个引号，应用启动就会因为解析失败而挂掉。
    #
    # 默认值只放开发用的两个地址。【绝不使用 "*"】：
    # 通配符意味着任何网站都能带着用户的浏览器直接调这些接口，
    # 而这些接口既没有鉴权、又能删数据。
    CORS_ALLOW_ORIGINS: str = "http://127.0.0.1:5173,http://localhost:5173"

    @property
    def cors_allow_origins(self) -> list[str]:
        """把逗号分隔的配置拆成 CORS 中间件要的列表，顺带去掉空项。"""
        return [origin.strip() for origin in self.CORS_ALLOW_ORIGINS.split(",") if origin.strip()]

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
