"""Embedding 服务：把文本转成向量。

为什么向量化不用 DeepSeek：DeepSeek 官方 API 只提供生成模型（chat / reasoner），
没有 embedding 接口，调用 /embeddings 会返回 404。所以 RAG 的两半分别由两家承担 ——
「把文本转向量」用硅基流动，「根据检索结果生成答案」用 DeepSeek（见 services/llm.py）。

硅基流动同样是 OpenAI 兼容接口，所以这里可以复用 openai 官方 SDK，
只把 base_url 指向 https://api.siliconflow.cn/v1。

约定（见 CLAUDE.md）：业务代码只使用本模块导出的 embed_text()，
不自己 new AsyncOpenAI，也不直接读 settings.SILICONFLOW_API_KEY。
"""

import logging

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---- 客户端 ----
# 和 llm.py 一样用「模块级变量 + 惰性创建」，原因也一样：
# AsyncOpenAI 在 api_key 为空时会直接抛 OpenAIError，如果在模块导入时创建，
# 那么「还没配 Key」的环境下连 import 本模块都会失败，FastAPI 都起不来。
# 我们希望的失败方式是「调用时报错」，而不是「服务启动不了」。
#
# 另外整个进程共用一个 client 就够了：它内部维护 HTTP 连接池，
# 每次调用都新建会反复重建连接，反而更慢。
_embedding_client: AsyncOpenAI | None = None


def get_embedding_client() -> AsyncOpenAI:
    """返回全局唯一的 AsyncOpenAI 客户端（用于 embedding），第一次调用时才创建。

    创建客户端本身不发任何网络请求，也不产生费用；真正的调用只发生在 embed_text() 里。
    """
    global _embedding_client

    if _embedding_client is None:
        # 只判断「有没有配」，绝不把 Key 本身写进日志或异常信息。
        if not settings.SILICONFLOW_API_KEY.get_secret_value():
            raise RuntimeError(
                "未配置 SILICONFLOW_API_KEY：请在仓库根目录的 .env 中填写后重试"
                "（可参考 .env.example）"
            )

        # 显式调用 get_secret_value() 取出明文交给 SDK —— 这是唯一需要明文的地方。
        _embedding_client = AsyncOpenAI(
            api_key=settings.SILICONFLOW_API_KEY.get_secret_value(),
            base_url=settings.SILICONFLOW_BASE_URL,
        )

    return _embedding_client


async def embed_text(text: str) -> list[float]:
    """把一段文本转成向量。

    参数：
        text: 待向量化的文本。空白字符串会被拒绝 —— 见下方说明。

    返回：
        浮点数列表，长度必然等于 settings.EMBEDDING_DIM；不满足就直接抛异常。

    异常：
        ValueError：  text 为空或只有空白字符（调用方传错了参数）。
        RuntimeError：未配置 Key、网络不通、接口返回 4xx/5xx（服务端或环境问题）。
    """
    # 先挡掉空输入。这里刻意抛错而不是返回一个全零向量：
    # 全零向量和任何文本的余弦相似度都是 0，写进 Milvus 后它永远不会被检索到，
    # 但也不会报错 —— 等于悄悄多了一条「查不到的垃圾数据」，比直接失败更难发现。
    if not text or not text.strip():
        raise ValueError("embed_text 收到空文本，无法向量化")

    client = get_embedding_client()

    try:
        response = await client.embeddings.create(
            # 模型名从配置读，不写死在调用处：将来换模型只改 .env。
            # 但要注意，换模型往往会改变向量维度，而 Milvus collection 的维度建好就改不了。
            model=settings.SILICONFLOW_EMBEDDING_MODEL,
            input=text,
        )
    except APIConnectionError as exc:
        # 网络层失败：DNS 解析不了、超时、连不上 api.siliconflow.cn。
        # 这类问题要看到完整堆栈才好排查，所以用 logger.exception 记全。
        logger.exception("调用 SiliconFlow Embedding 失败：无法建立连接")
        raise RuntimeError(
            "无法连接 SiliconFlow 服务，请检查网络或 .env 中的 SILICONFLOW_BASE_URL"
        ) from exc
    except APIStatusError as exc:
        # 服务端有响应但状态码非 2xx：401 密钥无效、402 余额不足、429 限流、5xx 服务异常。
        # 只把状态码和接口给的错误说明写进日志 —— 它们能定位问题，又不含密钥。
        logger.error("调用 SiliconFlow Embedding 失败：HTTP %s - %s", exc.status_code, exc.message)
        raise RuntimeError(
            f"SiliconFlow Embedding 接口返回错误（HTTP {exc.status_code}），详情见服务端日志"
        ) from exc

    # 正常情况下 data 里至少有一项；为空属于异常响应，
    # 与其让后面 data[0] 抛出难懂的 IndexError，不如在这里给出明确提示。
    if not response.data:
        raise RuntimeError("SiliconFlow 返回结果中没有 data，无法取出向量")

    vector: list[float] = response.data[0].embedding

    # 维度校验：这是一道「防止静默出错」的闸门。
    # 如果换了 embedding 模型却没同步改 EMBEDDING_DIM，或者 collection 是按别的维度建的，
    # 向量照样能算出来、照样能写进 Milvus —— 只是检索出来的相似度全是无意义的，
    # 而那种错误几乎不可能通过「看结果」发现。所以在这里立刻拦下，把话说死。
    if len(vector) != settings.EMBEDDING_DIM:
        raise RuntimeError(
            f"向量维度不匹配：模型 {settings.SILICONFLOW_EMBEDDING_MODEL} 返回了 "
            f"{len(vector)} 维，但配置 EMBEDDING_DIM={settings.EMBEDDING_DIM}。"
            "请确认 .env 中的 SILICONFLOW_EMBEDDING_MODEL 与 EMBEDDING_DIM 是否对应；"
            "注意 Milvus collection 的维度建好之后无法更改，改维度需要重建集合并重新向量化。"
        )

    return vector


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量把多段文本转成向量，一次请求完成。

    为什么需要批量版本：单条 embed_text() 一次要等 1~2 秒，入库时动辄几百上千个
    chunk，逐个调用会慢到不可用。批量接口一次能传多条，网络往返只发生一次，
    整体耗时能降一个数量级。

    参数：
        texts: 待向量化的文本列表。

    返回：
        向量列表，长度与 texts 相同，且**顺序一一对应**。
        输入为空列表时返回空列表。

    异常：
        ValueError：   列表中任意一条为空或只有空白字符（即使只有一条不合规，
                       整个调用也会失败 —— 见下方说明）。
        RuntimeError： 未配置 Key、网络不通、接口返回 4xx/5xx、
                       返回数量与请求不符，或任意一条向量维度不正确。
    """
    # 空列表直接返回，不发起请求。这不是省事的写法，而是因为：
    # 拿一个空列表去调 API，不同服务商的反应不一致（有的报错、有的返回空），
    # 与其依赖对方的实现，不如在本地把语义定死：没有输入就没有输出。
    if not texts:
        return []

    # 先整体校验再发请求：任何一条不合规就整批拒绝。
    #
    # 为什么不做「过滤掉空串、只向量化有效的那些」：那样返回的向量数量会和输入对不上，
    # 调用方必须再去猜「第几条被丢掉了」——而这个对应关系一旦错位，
    # 存进 Milvus 的就是「A 的向量配 B 的文本」，检索结果会静默地变得毫无意义。
    # 报错位置（第几条）也一并给出，方便定位。
    for index, text in enumerate(texts):
        if not text or not text.strip():
            raise ValueError(
                f"embed_texts 收到第 {index} 条为空或只有空白字符的文本，无法向量化"
            )

    client = get_embedding_client()

    try:
        response = await client.embeddings.create(
            model=settings.SILICONFLOW_EMBEDDING_MODEL,
            # 关键：把整个列表传进去，SDK 会把它序列化成一个请求。
            # 这里绝不能写成 for 循环逐条调用 —— 那会把「一次请求」变成 N 次，
            # 批量接口的意义就没了。
            input=texts,
        )
    except APIConnectionError as exc:
        logger.exception("批量调用 SiliconFlow Embedding 失败：无法建立连接")
        raise RuntimeError(
            "无法连接 SiliconFlow 服务，请检查网络或 .env 中的 SILICONFLOW_BASE_URL"
        ) from exc
    except APIStatusError as exc:
        logger.error(
            "批量调用 SiliconFlow Embedding 失败：HTTP %s - %s", exc.status_code, exc.message
        )
        raise RuntimeError(
            f"SiliconFlow Embedding 接口返回错误（HTTP {exc.status_code}），详情见服务端日志"
        ) from exc

    if not response.data:
        raise RuntimeError("SiliconFlow 返回结果中没有 data，无法取出向量")

    # 按 index 排序，而不是直接用返回顺序。
    # OpenAI 兼容接口的响应里每条都带一个 index 字段，它才是「这条对应输入中的第几条」
    # 的权威标识。绝大多数情况下返回顺序就是输入顺序，但协议并不保证这一点 ——
    # 一旦顺序错位，存进 Milvus 的向量就会和文本对不上，而这种错误从检索结果上
    # 完全看不出来（相似度照样算得出来，只是算错了对象）。所以这里显式排序。
    ordered = sorted(response.data, key=lambda item: item.index)
    vectors: list[list[float]] = [item.embedding for item in ordered]

    # 数量核对：少了或多了都说明前面的 index 对应关系已经不可信，直接失败。
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"SiliconFlow 返回的向量数量与请求不符：请求 {len(texts)} 条，收到 {len(vectors)} 条"
        )

    # 逐条校验维度。任何一条不合格就整批失败、不返回部分结果 ——
    # 因为「部分成功」会让调用方拿到一个和输入等长的列表，
    # 但其中某些位置的向量是错的，这种数据一旦入库极难排查。
    for index, vector in enumerate(vectors):
        if len(vector) != settings.EMBEDDING_DIM:
            raise RuntimeError(
                f"第 {index} 条向量维度不匹配：模型 {settings.SILICONFLOW_EMBEDDING_MODEL} "
                f"返回了 {len(vector)} 维，但配置 EMBEDDING_DIM={settings.EMBEDDING_DIM}。"
                "注意 Milvus collection 的维度建好之后无法更改，"
                "改维度需要重建集合并重新向量化。"
            )

    return vectors
