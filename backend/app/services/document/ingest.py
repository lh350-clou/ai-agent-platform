"""文档入库编排：把一个 TXT 文件完整处理成 Milvus 里的向量。

这一步的角色是「编排」而不是「实现」—— 解析、切分、向量化、写入四件事
分别由 parser / splitter / embedding / vector_store 完成，
本模块只负责按正确顺序把它们串起来，并处理分批。

这样分层的好处是每一段都能单独替换和测试：换向量模型只动 embedding.py，
换向量库只动 vector_store.py，这个文件不用改。

本模块不做的事（属于后续阶段）：写 PostgreSQL 的 documents/chunks 表、
记录处理状态、文件上传接口、失败回滚。
"""

import asyncio
import logging
from pathlib import Path

from app.core import milvus
from app.services import vector_store
from app.services.document.parser import parse_txt
from app.services.document.splitter import split_text
from app.services.embedding import embed_texts

logger = logging.getLogger(__name__)

# 切分参数。暂时放在这里，等接入 config.py 后应该挪到 settings 里，
# 和 DEEPSEEK_MODEL 一样从 .env 读 —— 这两个值直接影响检索效果，
# 是会被反复调的参数，写死在代码里意味着每调一次都得改代码重新部署。
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50

# 每批送去向量化的 chunk 数量。
# 为什么不一次把全部 chunk 发过去：SiliconFlow 单次请求有条数上限，
# 而且条数越多请求体越大、越容易超时。32 是个折中值 ——
# 既明显减少了请求次数，又不会让单个请求变得又大又慢。
EMBED_BATCH_SIZE: int = 32


def build_chunk_id(document_id: str, sequence: int) -> str:
    """按「文档 ID + 6 位序号」生成稳定、可读的 chunk_id。

    序号从 1 开始（000001），补零到 6 位：
    补零是为了让 ID 定长 —— 定长之后，按字符串排序的结果和按数值排序一致
    （chunk-000002 排在 chunk-000010 前面）。不补零的话 "chunk-10" 会排在
    "chunk-2" 前面，在列表和日志里看着很别扭。

    「稳定」指的是：同一份文档反复入库，第 N 个 chunk 拿到的 ID 永远相同。
    正因为稳定，它才能安全地当作 PostgreSQL 中 chunks 表的主键来用，
    让检索结果能对回业务数据。
    """
    return f"{document_id}-chunk-{sequence:06d}"


async def ingest_txt(
    file_path: str | Path,
    knowledge_base_id: str,
    document_id: str,
) -> int:
    """把一个 TXT 文件解析、切分、向量化并写入 Milvus。

    参数：
        file_path:          TXT 文件路径。
        knowledge_base_id:  写入哪个知识库，检索时靠它做隔离。
        document_id:        这批 chunk 属于哪篇文档。

    返回：
        实际写入 Milvus 的 chunk 数量。

    ── 返回即代表「可查」──
        函数成功返回时，本次写入的数据已经 flush 过了，可以立刻被 Milvus 的
        query（标量过滤/聚合）查到。

        这一点必须显式保证，因为 Milvus 对「新写入但尚未 flush 的数据」有一个
        容易踩的行为：search（向量检索）能看到它，query（按条件查询/统计）看不到。
        实测过：插入后立刻 query 得到 0，同一时刻 search 却能返回这条数据，
        flush 之后 query 才变成 1。

        也就是说，如果不做这件事，调用方在入库后立刻统计「这篇文档有几个 chunk」
        会得到 0 —— 数据明明写进去了，看起来却像失败了。
        与其让每个调用方各自记得 flush，不如在入库流程结束时统一做掉。

    异常：
        任何一步失败都直接向上抛，不吞、不降级：
            FileNotFoundError / ValueError  解析阶段（文件不存在、编码不对、内容为空）
            ValueError                      切分参数非法
            ValueError / RuntimeError       向量化与写入阶段（空文本、维度不符、网络或服务端错误）
        调用方拿到异常就知道「这次入库没有完整成功」。

    ── 失败时自动清理 ──
        写入是分批进行的，如果第 3 批失败，前 2 批已经进了 Milvus。
        本函数会在抛出异常【之前】先把该 document_id 的向量全部删掉，
        所以调用方拿到异常时，Milvus 里不会留下这篇文档的任何残留数据。

        清理本身失败不会影响异常语义：往外抛的始终是最初那个失败原因，
        清理错误只记进日志（见 _discard_partial 的说明）。

        调用方捕获异常后，把文档标记为 failed 即可；
        重试也是安全的 —— 不会因为上一次的残留而产生重复数据。
    """
    path = Path(file_path)

    # ---- 1. 解析：文件 → 纯文本 ----
    # parse_txt 已经保证了内容非空，所以这里不需要再判一次空。
    text = parse_txt(path)

    # ---- 2. 切分：纯文本 → chunk 列表 ----
    chunks = split_text(
        text,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    # parse_txt 保证了文本非空，而 split_text 对非空文本一定至少产出一个 chunk。
    # 所以走到这里 chunks 为空说明上游契约被破坏了 —— 与其返回 0 让调用方以为
    # 「处理成功了，只是没有内容」，不如明确报错。
    if not chunks:
        raise RuntimeError(
            f"切分后没有得到任何 chunk：{path}（这不该发生，请检查解析与切分逻辑）"
        )

    # ---- 3/4. 分批向量化、写入、flush；任何一步失败都先清理残留 ----
    #
    # 整段只包一层 try，而不是在每个可能失败的位置各补一段清理。
    # 原因是失败点有好几个（embedding、数量核对、insert、flush），逐个补容易漏，
    # 而且以后往中间插入新步骤时还会再漏一次。
    # 包住整段就只有一个出口，「任何失败都会先清理」由代码结构本身保证，
    # 而不是靠人每次改代码时记得。
    try:
        written = await _embed_and_store(chunks, knowledge_base_id, document_id)
    except Exception:
        # 清理残留，然后把【原始异常】原样抛出去。
        #
        # 这里刻意不写 `except Exception as exc`、也不做任何包装：
        # 往外抛的必须还是原来那一个异常。上层 api/documents.py 依赖它
        # 把文档标成 failed、把原因写进 error_message —— 一旦被替换成别的异常，
        # 用户看到的失败原因就会变成「清理出错」之类的干扰信息，真正的病因反而丢了。
        await _discard_partial(document_id)
        raise

    logger.info("入库完成：document_id=%s，共写入 %d 条", document_id, written)
    return written


async def _embed_and_store(
    chunks: list[str],
    knowledge_base_id: str,
    document_id: str,
) -> int:
    """分批向量化并写入 Milvus，最后 flush 一次；返回实际写入条数。

    单独抽成函数，是为了让调用方能用一段 try 覆盖「全部写入步骤」，
    从而只在一个地方处理失败清理。
    """
    total_batches = (len(chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    logger.info(
        "开始入库：document_id=%s，%d 个 chunk，分 %d 批（每批最多 %d）",
        document_id, len(chunks), total_batches, EMBED_BATCH_SIZE,
    )

    written = 0

    # 批次按顺序处理（不是并发），保证写入顺序和原文顺序一致。
    # 并发虽然更快，但会让 chunk 的写入次序变得不确定，而 chunk_id 是按序号生成的，
    # 顺序错乱会让「同一篇文档先后两次入库」得到不一致的结果。
    for batch_no, start in enumerate(range(0, len(chunks), EMBED_BATCH_SIZE), start=1):
        batch = chunks[start : start + EMBED_BATCH_SIZE]

        try:
            vectors = await embed_texts(batch)
        except Exception:
            # 记下是哪一批、哪些序号失败的，再原样抛出。
            # 注意这里是裸 raise，抛出的还是原来那个异常对象，没有包装、没有吞掉 ——
            # 只是给日志补一条定位信息（几十批的情况下，没有这条很难查是哪里断的）。
            logger.exception(
                "第 %d/%d 批向量化失败（chunk 序号 %d~%d）",
                batch_no, total_batches, start + 1, start + len(batch),
            )
            raise

        # 数量核对。embed_texts 本身已经保证等长，这里再查一次是因为
        # 下面的 zip 在长度不一致时会「静默截断」—— 少写几条却不报错，
        # 而本函数的契约是返回真实写入数量，不能容忍这种静默偏差。
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"第 {batch_no} 批向量数量与 chunk 数量不符："
                f"chunk {len(batch)} 条，向量 {len(vectors)} 条"
            )

        # 逐条写入。zip 的两边都来自同一个 batch，顺序严格对应：
        # 第 i 个向量就是第 i 个 chunk 的向量。
        for offset, (content, vector) in enumerate(zip(batch, vectors)):
            await vector_store.insert_chunk(
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                # 序号 = 全局位置 + 1（1-based），跨批次连续
                chunk_id=build_chunk_id(document_id, start + offset + 1),
                content=content,
                vector=vector,
            )
            written += 1

        logger.info("第 %d/%d 批完成，累计写入 %d 条", batch_no, total_batches, written)

    # ---- flush：让写入的数据立刻可被 query 查到 ----
    # 只有在「确实写进去了东西」且「前面每一步都成功」的前提下才做。
    # 走到这里本来就意味着所有 embedding 和 insert_chunk 都成功了，
    # 这个判断是把这个前提显式写出来，避免以后有人在中间加了 continue 之类
    # 的跳过逻辑后，flush 在一个空写入上白跑一趟。
    if written > 0:
        try:
            # 复用 core/milvus.py 里那个全局唯一的客户端，不另建一个。
            #
            # 用 asyncio.to_thread 包一层：flush 是同步阻塞调用，会一直等到
            # Milvus 把数据落盘才返回。直接在 async 函数里调用会把事件循环
            # 卡住这段时间，和 vector_store.py 里处理其它 Milvus 调用的做法一致。
            await asyncio.to_thread(
                milvus.get_milvus_client().flush, milvus.COLLECTION_NAME
            )
        except Exception:
            # flush 失败意味着「数据写进去了，但暂时查不到」这种半吊子状态。
            # 不能吞掉、更不能照常返回成功数量 —— 那会让调用方以为一切正常。
            logger.exception(
                "flush 失败：已写入 %d 条，但它们可能暂时无法被 query 查到", written
            )
            raise

        logger.info("已 flush，本次写入的数据现在可被 query 查询")

    return written


async def _discard_partial(document_id: str) -> None:
    """清理某个文档已经写进 Milvus 的向量。只在入库失败的路径上调用。

    **这个函数自己绝不抛异常。** 它是在异常处理过程中被调用的，
    此时「原始异常正在往外传播」；如果这里再抛一个，就会把原始异常顶掉
    （异常链上只看最后一个），上层拿到的失败原因就成了「清理出错」，
    而真正导致失败的原因反而丢失 —— 排查时会被带到完全错误的方向上。
    所以清理失败只记日志，绝不往上抛。
    """
    try:
        # 复用 vector_store 里已有的实现，不在 ingest 里再写一遍 Milvus 删除。
        #
        # 无脑调用即可，不需要先判断「到底写进去过没有」：
        # 没有向量时它返回 0，这不算错误。反过来，如果因为 written == 0 就跳过清理，
        # 会漏掉一种情况 —— insert_chunk 已经写进 Milvus、但在返回途中才失败
        # （网络抖动、响应超时），此时 written 还没自增，清理却被跳过了，
        # 数据就永久残留。失败路径上多一次 Milvus 调用的代价，远小于残留数据的代价。
        deleted = await vector_store.delete_by_document_id(document_id)
        logger.warning(
            "入库失败，已清理该文档的残留向量：document_id=%s，清理 %d 条",
            document_id, deleted,
        )
    except Exception:
        # 清理失败必须留痕，否则数据会悄无声息地残留。
        # 但只记录，不再抛 —— 理由见上面的函数说明。
        logger.exception(
            "入库失败后的清理也失败了，Milvus 中可能残留该文档的向量：document_id=%s",
            document_id,
        )
