"""Milvus 基础设施：客户端、Collection 的 schema 与初始化。

约定（见 CLAUDE.md）：业务代码只使用本模块导出的 get_milvus_client() /
ensure_collection_exists() / COLLECTION_NAME，
不在别处自己 new MilvusClient，也不在别处硬编码 collection 名或向量维度。

本模块目前只负责「把 collection 准备好」，不包含任何读写数据的方法 ——
那些属于 services/vector_store.py。
"""

import logging

from pymilvus import DataType, MilvusClient
from pymilvus.orm.schema import CollectionSchema

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---- 常量 ----

# Milvus 地址。暂时写在这里是因为本轮的改动范围不含 config.py，
# 等接入 Milvus 的代码稳定后应挪到 settings 里（和 POSTGRES_HOST 一样从 .env 读），
# 这样切 Docker 部署时改 .env 即可，不用改代码。
MILVUS_URI: str = "http://localhost:19530"

# Collection 名在这里定义一次，其他地方一律引用这个常量。
# 写成常量而不是散落的字符串字面量，是为了避免「一处改名、另一处漏改」——
# 那种错误会表现为「数据写进去了但搜不到」，很难查。
COLLECTION_NAME: str = "knowledge_chunks"

# 向量字段名单独提出来，因为建索引、插入、检索三处都要用到它。
VECTOR_FIELD_NAME: str = "vector"

# 检索时的过滤字段。给它建倒排索引的理由见 build_index_params() 里的说明。
KNOWLEDGE_BASE_ID_FIELD: str = "knowledge_base_id"

# Milvus 的 VARCHAR 必须显式给 max_length，没有「不限长」这个选项。
# UUID 字符串固定 36 字符，留到 64 足够且不会浪费。
MAX_ID_LENGTH: int = 64

# chunk 正文的长度上限。Milvus 对 VARCHAR 的 max_length 上限就是 65535，
# 这里直接给到顶：给定小了（比如 2000），超出部分会被截断或写入报错，
# 而且截断是静默的 —— 检索出来的内容缺一截，很难发现。
MAX_CONTENT_LENGTH: int = 65535

# ---- 客户端 ----
# 和 llm.py / embedding.py 一样用「模块级变量 + 惰性创建」：
# 如果在模块导入时就连接 Milvus，那么 Milvus 没启动时连 import 本模块都会失败，
# FastAPI 直接起不来。我们希望的失败方式是「用到时才报错」，而不是「服务启动不了」。
_milvus_client: MilvusClient | None = None


def get_milvus_client() -> MilvusClient:
    """返回全局唯一的 Milvus 客户端，第一次调用时才真正建立连接。

    整个进程共用一个客户端：它内部维护到 Milvus 的连接，
    每次调用都新建会反复建连，而且容易把 Milvus 的连接数耗尽。
    """
    global _milvus_client

    if _milvus_client is None:
        _milvus_client = MilvusClient(uri=MILVUS_URI)
        logger.info("已连接 Milvus：%s", MILVUS_URI)

    return _milvus_client


def build_schema() -> CollectionSchema:
    """构造 collection 的字段定义。

    注意 id 用的是 Milvus 自增主键（auto_id=True），不是业务 ID：
    - Milvus 自己维护这个整数主键，插入时不用（也不能）指定它；
    - 真正的业务标识放在 chunk_id 里，它对应 PostgreSQL 中 chunks 表的主键，
      两边靠它对照。

    enable_dynamic_field=False 是刻意关掉的。开着的话，插入时带了 schema 里
    没有的字段不会报错，而是被悄悄塞进一个隐藏的动态字段里 ——
    等发现「字段写错了、数据查不出来」时，往往已经写进去一大批了。
    """
    schema = get_milvus_client().create_schema(auto_id=True, enable_dynamic_field=False)

    # 主键：Milvus 自增，插入时不需要也不允许指定
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)

    # 三个 ID 字段。维度从 settings 读，不在这里写死 1024 ——
    # 见文件末尾对「为什么」的说明。
    schema.add_field(
        field_name="knowledge_base_id",
        datatype=DataType.VARCHAR,
        max_length=MAX_ID_LENGTH,
    )
    schema.add_field(
        field_name="document_id",
        datatype=DataType.VARCHAR,
        max_length=MAX_ID_LENGTH,
    )
    schema.add_field(
        field_name="chunk_id",
        datatype=DataType.VARCHAR,
        max_length=MAX_ID_LENGTH,
    )

    # 正文直接存在 Milvus 里（PostgreSQL 那边也有一份）。
    # 这份是冗余的，目的是让检索一次就能拿到要喂给模型的文本，
    # 不用再回 PostgreSQL 查一轮 —— 那会增加最热路径上的延迟。
    schema.add_field(
        field_name="content",
        datatype=DataType.VARCHAR,
        max_length=MAX_CONTENT_LENGTH,
    )

    # 向量字段。维度必须等于 embedding 模型实际输出的维度，
    # 所以从 settings.EMBEDDING_DIM 读，保证全项目只有这一个来源。
    schema.add_field(
        field_name=VECTOR_FIELD_NAME,
        datatype=DataType.FLOAT_VECTOR,
        dim=settings.EMBEDDING_DIM,
    )

    return schema


def build_index_params():
    """构造向量字段的索引参数。

    索引方式用 AUTOINDEX：这是 Milvus 3.x 的标准做法 ——
    pymilvus 自己的快速建表路径（create_collection 只给 dimension 参数时）
    内部用的就是 AUTOINDEX。交给服务端根据数据规模自动选具体索引算法，
    不用我们预先猜 HNSW 的 M / efConstruction 该设多少。

    metric_type 用 COSINE：主流 embedding 模型（bge 系列、OpenAI 系列）
    都以余弦相似度为设计目标，和它们配合最自然。
    """
    index_params = get_milvus_client().prepare_index_params()
    index_params.add_index(
        field_name=VECTOR_FIELD_NAME,
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    # 标量字段 knowledge_base_id 上的倒排索引。
    #
    # 为什么必须有它：检索时永远带着 `knowledge_base_id == "..."` 这个过滤条件
    # （不加就会跨知识库检索，既泄漏数据又拖低准确率）。Milvus 执行这种带过滤的
    # 向量检索时，是「先按标量条件筛出候选集，再在候选集里算向量距离」。
    # 没有倒排索引，第一步就只能全表扫描 —— 数据量小的时候感觉不出来，
    # 一旦上量，它会成为整个检索链路里最慢的一环。
    index_params.add_index(
        field_name=KNOWLEDGE_BASE_ID_FIELD,
        index_type="INVERTED",
    )

    return index_params


def ensure_scalar_index_exists() -> bool:
    """确保 knowledge_base_id 上的标量索引存在。幂等：已有就什么都不做。

    返回 True 表示本次新建，False 表示本来就有。

    单独抽成函数，是因为 collection 和索引的生命周期并不总是一致：
    这个 collection 就是先建好、后来才补上标量索引的。
    没有这个函数的话，「给存量 collection 补索引」就只能靠手工执行，
    下次换台机器部署又会漏掉。
    """
    client = get_milvus_client()

    # 先查已有的索引名。Milvus 里索引名默认等于字段名，
    # 所以直接判断字段名在不在索引列表里即可。
    existing = client.list_indexes(COLLECTION_NAME)
    if KNOWLEDGE_BASE_ID_FIELD in existing:
        logger.info("Collection %r 的标量索引已存在，跳过创建", COLLECTION_NAME)
        return False

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=KNOWLEDGE_BASE_ID_FIELD,
        index_type="INVERTED",
    )
    # create_index 对这个 collection 是「只补这一个字段的索引」：
    # pymilvus 内部是逐字段循环调用底层接口，不会顺带删掉别的索引，
    # 所以已有的向量索引不受影响。
    client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
    logger.info("已为 Collection %r 的 %r 创建 INVERTED 索引", COLLECTION_NAME, KNOWLEDGE_BASE_ID_FIELD)
    return True


def _get_existing_dim(client: MilvusClient) -> int | None:
    """读出已存在 collection 的向量维度，读不到返回 None。"""
    description = client.describe_collection(COLLECTION_NAME)
    for field in description.get("fields", []):
        if field.get("name") == VECTOR_FIELD_NAME:
            params = field.get("params") or {}
            dim = params.get("dim")
            return int(dim) if dim is not None else None
    return None


def ensure_collection_exists() -> bool:
    """确保 collection 和它需要的索引都存在。幂等。

    做两件事：
    1. collection 不存在就建，已存在就跳过；
    2. 无论哪种情况，都确认标量索引在位（见 ensure_scalar_index_exists）。

    返回 True 表示本次新建了 collection，False 表示本来就有
    （用于让调用方和验证脚本能区分「建了」和「跳过」，而不是靠日志猜）。

    刻意不做的事：不删除、不清空、不重建。初始化逻辑最忌讳「顺手重建」——
    那会在某次上线时把生产数据无声地抹掉。维度不一致时也只报错、不动数据。
    """
    client = get_milvus_client()

    if client.has_collection(COLLECTION_NAME):
        # 到这里说明 collection 已经存在，什么都不创建。
        # 但要额外查一件事：它的向量维度是否还和当前配置一致。
        #
        # 这是本模块最重要的一道检查。维度是建 collection 时写死的，事后改不了；
        # 如果换了 embedding 模型却没重建 collection，数据照样能写进去、检索也照样返回结果，
        # 只是相似度全是无意义的 —— 这种错误几乎不可能靠「看结果」发现。
        # 与其等到检索阶段，不如在初始化时就把它拦下来。
        existing_dim = _get_existing_dim(client)
        if existing_dim is not None and existing_dim != settings.EMBEDDING_DIM:
            raise RuntimeError(
                f"Collection {COLLECTION_NAME!r} 已存在，但它的向量维度是 {existing_dim}，"
                f"与当前配置 EMBEDDING_DIM={settings.EMBEDDING_DIM} 不一致。"
                "维度建好后无法修改，需要重建 collection 并重新向量化全部文档。"
                "（本函数不会自动重建，以免误删数据，请确认后手工处理。）"
            )
        logger.info("Collection %r 已存在，跳过创建", COLLECTION_NAME)
        created = False

    else:
        # 不存在才创建。schema 和 index 一起传进去，
        # create_collection 会在建表后自动建索引并加载，不需要额外调 load_collection。
        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=build_schema(),
            index_params=build_index_params(),
        )
        logger.info("已创建 Collection %r（dim=%s）", COLLECTION_NAME, settings.EMBEDDING_DIM)
        created = True

    # 不论 collection 是刚建的还是早就存在的，都补一次标量索引（函数内部自带幂等判断）。
    # 刻意放在 if/else 外面，是为了让「给存量 collection 补索引」和「全新部署」
    # 走同一条代码路径 —— 否则补索引这件事只会在老环境上被漏掉。
    ensure_scalar_index_exists()

    return created
