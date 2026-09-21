"""文档接口：上传 TXT、同步完成入库、查询处理结果。

本层只做三件事：校验请求、驱动 Document 的状态流转、把 services 层串起来。
解析、切分、向量化、写向量库都在 services/document/ingest.py 里，这里不重复实现。

关于「同步完成」：上传请求会一直等到入库跑完才返回（HTTP 201 或 500），
没有用后台任务。这是本轮刻意的选择 —— 先把整条链路跑通、把状态流转定下来，
再考虑改成异步。代价是前端要等较久（几十秒级），后面会换成 BackgroundTasks
或真正的任务队列。
"""

import logging
import uuid
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import BASE_DIR, settings
from app.core.database import get_db
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.schemas.document import DocumentResponse, DocumentUploadResponse
from app.services import vector_store
from app.services.document.ingest import ingest_txt

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["文档"])

# 上传文件的存储目录。
#
# 用 config.BASE_DIR 推导（= 仓库根目录），而不是写相对路径 "./storage" ——
# 相对路径取决于「从哪个目录启动服务」，从根目录启动和从 backend/ 启动
# 会落到两个不同的地方，这种问题往往要到线上才发现。
# 这和 .env 的读取用的是同一个基准，行为保持一致。
STORAGE_DIR: Path = BASE_DIR / "storage" / "documents"

# 当前只支持 TXT。等接入 PDF / Word 时，这里会变成一张「扩展名 → 解析器」的分派表。
ALLOWED_SUFFIX: str = ".txt"

# error_message 落库前的截断长度。异常信息动辄几百字，全存进去没有意义，
# 排查时看日志更完整。
MAX_ERROR_MESSAGE_LENGTH: int = 500


def _storage_dir() -> Path:
    """返回存储目录，不存在就创建。

    刻意不在模块导入时创建：导入模块是「读操作」，不应该有建目录这种副作用，
    否则任何 import 这个模块的脚本（比如跑测试）都会在磁盘上留下目录。
    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return STORAGE_DIR


def _safe_error_message(exc: Exception) -> str:
    """把异常转成既能落库、又能安全回给客户端的简短说明。

    这个字段会通过 DocumentResponse 返回给客户端，所以必须过滤一道：
      - 我们自己抛的 ValueError / RuntimeError，消息是专门写给人看的中文说明，
        且刻意不含密钥，可以原样保留；
      - 其它异常（数据库驱动、第三方库）的消息可能夹带连接串、主机名、
        文件路径等内部信息 —— 这些一旦回给客户端就是信息泄漏。
        所以只保留异常类型名，细节留在服务端日志里。
    """
    if isinstance(exc, (ValueError, RuntimeError)):
        return f"{type(exc).__name__}: {exc}"[:MAX_ERROR_MESSAGE_LENGTH]
    return f"{type(exc).__name__}: 内部错误，详情见服务端日志"


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="上传 TXT 文档并入库",
)
async def upload_document(
    file: UploadFile = File(..., description="要上传的 TXT 文件"),
    knowledge_base_id: UUID = Form(..., description="目标知识库 ID"),
    db: AsyncSession = Depends(get_db),
) -> DocumentUploadResponse:
    """上传一份 TXT 文档，同步完成解析、切分、向量化并写入 Milvus。

    状态流转：
        pending -> processing -> completed
        ingest 抛异常时 -> failed（error_message 记录原因）

    返回：
        201 + 文档信息（含 chunk_count）。

    异常：
        404 知识库不存在；400 文件不合法；500 入库失败。
    """
    # ---- a. 知识库必须存在 ----
    # 先查这一步不是多余的：documents.knowledge_base_id 有外键约束，
    # 不查的话会等到 commit 时才报数据库层面的外键错误，
    # 那个错误既难懂、又会变成 500，而这里本该是明确的 404。
    knowledge_base = await db.get(KnowledgeBase, knowledge_base_id)
    if knowledge_base is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"知识库不存在：{knowledge_base_id}",
        )

    # ---- b. 校验文件名与扩展名 ----
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="缺少文件名")

    # 只取扩展名来判断类型。注意：filename 里可能带路径分隔符
    # （例如 "..\\..\\etc\\passwd"），这里只把它当字符串看后缀，
    # 真正的磁盘路径完全由 UUID 生成（见下面 c），所以这种输入伤不到我们。
    if Path(filename).suffix.lower() != ALLOWED_SUFFIX:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"目前只支持 {ALLOWED_SUFFIX} 文件，收到：{filename}",
        )

    # ---- b. 校验内容是 UTF-8 且非空 ----
    # 在这里先解一次码，是为了把「文件本身有问题」和「入库流程出问题」区分开：
    # 不先查，编码错误会一路传到 ingest 里才炸，最终返回 500（服务端错误），
    # 而实际上这是客户端传了个坏文件，应该是 400。
    raw = await file.read()

    # ---- b. 大小限制 ----
    # 放在解码、落盘、建记录之前，超限就立刻返回。
    # 位置很关键：这一步之后的所有动作（写磁盘、建 Document、embedding、写 Milvus）
    # 都不会发生，所以一次超限的上传在系统里【不留任何痕迹】——
    # 没有文件、没有数据库记录、没有向量。
    if len(raw) > settings.MAX_UPLOAD_SIZE_BYTES:
        # 用 HTTP_413_CONTENT_TOO_LARGE：这个 Starlette 版本里
        # 旧的 HTTP_413_REQUEST_ENTITY_TOO_LARGE 已被标记废弃，
        # 用旧名字虽然还能跑，但每次响应都会带上 DeprecationWarning。
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"文件过大：{len(raw)} 字节，"
                f"超过上限 {settings.MAX_UPLOAD_SIZE_BYTES} 字节"
                f"（{settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MiB）"
            ),
        )

    if not raw:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件内容为空")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="文件不是合法的 UTF-8 编码。"
                   "Windows 记事本保存的中文文档常是 GBK，请先转成 UTF-8 再上传。",
        ) from exc
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="文件内容为空（只有空白字符）"
        )

    # ---- c. 落盘：磁盘文件名一律由 UUID 生成 ----
    # 绝不使用客户端传来的 filename 作为磁盘路径。那个字符串是用户可控的，
    # 直接拿来拼路径就是「路径穿越」漏洞的经典成因（../../ 就能写到目录外）。
    # 这里生成 document_id，用它命名，扩展名也用我们校验过的常量而不是原字符串。
    document_id = uuid.uuid4()
    target = _storage_dir() / f"{document_id}{ALLOWED_SUFFIX}"
    try:
        target.write_bytes(raw)
    except OSError as exc:
        logger.exception("保存上传文件失败：%s", target)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="文件保存失败"
        ) from exc

    # ---- d/e/f. 建记录（pending）并提交，再推进到 processing ----
    document = Document(
        id=document_id,
        knowledge_base_id=knowledge_base_id,
        # name 用原始文件名（给人看的），file_path 用实际的 UUID 路径（给程序用的）。
        # 两者分开，才能既保留可读性又不信任用户输入。
        name=filename,
        file_path=str(target),
        file_type="txt",
        status="pending",
        chunk_count=0,
    )

    try:
        db.add(document)
        # 先提交一次：状态是 pending 的记录此刻就落库了。
        # 这样即使进程随后崩掉（入库是很重的操作），数据库里也留下一条
        # 「这份文档来过、但没处理完」的痕迹，而不是什么都没有。
        await db.commit()

        document.status = "processing"
        await db.commit()
    except Exception as exc:
        # 记录都没建起来，那刚写下去的文件就是垃圾，删掉。
        await db.rollback()
        target.unlink(missing_ok=True)
        logger.exception("创建文档记录失败：document_id=%s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="创建文档记录失败"
        ) from exc

    # ---- g. 入库 ----
    try:
        chunk_count = await ingest_txt(
            file_path=target,
            # ingest_txt 要的是字符串：chunk_id 由它拼成 "{document_id}-chunk-000001"，
            # 用字符串更直接，也避免 UUID 对象在 Milvus 那边被转换成别的形式。
            knowledge_base_id=str(knowledge_base_id),
            document_id=str(document_id),
        )
    except Exception as exc:
        logger.exception("文档入库失败：document_id=%s", document_id)

        document.status = "failed"
        document.error_message = _safe_error_message(exc)
        await db.commit()

        # 处理失败，原始文件没有留存价值，删掉，避免 storage 目录堆积垃圾。
        target.unlink(missing_ok=True)

        # 回给客户端的是通用文案：具体原因已经写进 error_message 和日志了，
        # 这里再重复一遍既啰嗦，又容易顺手把内部细节带出去。
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档处理失败，请查看该文档的 error_message 或服务端日志",
        ) from exc

    # ---- h. 成功 ----
    document.status = "completed"
    document.chunk_count = chunk_count
    document.error_message = None
    await db.commit()

    logger.info("文档入库完成：document_id=%s，chunk_count=%d", document_id, chunk_count)
    return DocumentUploadResponse.model_validate(document)


@router.get("/{document_id}", response_model=DocumentResponse, summary="查询文档处理结果")
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """按 ID 查询文档的处理状态与结果。

    这是「上传后确认到底成没成」的入口：上传接口是同步返回的，
    但客户端仍然可以用这个接口复查状态（尤其是失败后看 error_message）。
    """
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文档不存在：{document_id}",
        )
    return DocumentResponse.model_validate(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除文档（清 Milvus 向量 + PostgreSQL 记录 + 原始文件）",
)
async def delete_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """删除一份文档，同时清理它在三个地方留下的数据。

    删除顺序是「先 Milvus，再 PostgreSQL，最后磁盘」，这个顺序不能随便调：

    1. **先删 Milvus**。如果这一步失败就直接返回 500，PostgreSQL 记录保持不动 ——
       用户还能重试。反过来先删 PostgreSQL 的话，Milvus 里的向量就永远失去了
       document_id 这条线索（记录都没了，谁也不知道要删哪些向量），
       它们会变成清不掉的垃圾数据：检索时照样被召回，但已经没有任何办法定位。

    2. **再删 PostgreSQL**。此时向量已经清干净，记录可以安全删掉。

    3. **最后删磁盘文件**。放在最后是因为它最容易失败（文件被占用、权限问题），
       而前两步已经保证了数据层的一致性。

    返回：
        204 No Content。

    异常：
        404 文档不存在；500 任一步失败。
    """
    # 先把需要的信息读出来。注意要在删除之前读 —— 记录一旦删掉，
    # file_path 就再也拿不到了，磁盘上那份文件会永久残留。
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文档不存在：{document_id}",
        )

    file_path = Path(document.file_path)
    knowledge_base_id = document.knowledge_base_id

    # ---- 1. Milvus ----
    try:
        deleted_chunks = await vector_store.delete_by_document_id(str(document_id))
    except Exception as exc:
        # 这里刻意不删 PostgreSQL 记录：留着它，用户才能重试，
        # 也才能知道「这份文档的向量还在，需要清理」。
        logger.exception("删除文档失败（Milvus 阶段）：document_id=%s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除文档失败：清理向量数据时出错，文档未被删除",
        ) from exc

    # ---- 2. PostgreSQL ----
    try:
        await db.delete(document)
        await db.commit()
    except Exception as exc:
        logger.exception("删除文档失败（PostgreSQL 阶段）：document_id=%s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="删除文档记录失败"
        ) from exc

    # ---- 3. 磁盘 ----
    try:
        # missing_ok=True：文件已经不在了不算错误。这是正常情况 ——
        # 上传失败时接口已经把文件删掉了，而记录还在；
        # 此时用户来删文档，不该因为「文件本来就没有」而失败。
        file_path.unlink(missing_ok=True)
    except OSError as exc:
        # 注意此时 Milvus 和 PostgreSQL 都已经删完了，文件却还在。
        # 这是个不一致状态，但删除已经不可回退，只能报错让运维知道去手工处理。
        logger.exception("删除文档失败（磁盘阶段）：file_path=%s", file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档数据已删除，但原始文件清理失败，请联系管理员",
        ) from exc

    logger.info(
        "文档已删除：document_id=%s, knowledge_base_id=%s, 清理向量 %d 条",
        document_id, knowledge_base_id, deleted_chunks,
    )
