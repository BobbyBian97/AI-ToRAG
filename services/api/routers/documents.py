"""文档接入路由（F1）：批量上传 / 查询 / 软删除。

设计要点（对应设计书 §2.1 F1、§4.1 入库流水线、§8 API）：
- 批量上传为"部分成功"语义：HTTP 恒 200，逐文件结果落在
  uploaded / skipped / rejected 三个数组里，单文件失败不影响其余文件；
- 状态机：上传成功即插入 status="parsing" 的 document 行，随后
  enqueue(TASK_PARSE)；parsing→ready/failed 的推进由 F2 worker 负责；
  deleted 为软删标记（本模块 DELETE 接口设置，F2 worker 异步清向量库）；
- 外部依赖（MinIO / 队列）一律通过模块属性访问（infra.minio / infra.queue），
  便于测试 monkeypatch，不引入额外间接层。

注意：本模块只管"入队"，TASK_PARSE / TASK_PURGE 对应的任务函数本体
（services.worker.tasks.parse_document / purge_document）由 F2 实现。
"""
import hashlib
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

import infra.minio as minio_storage
import infra.queue as task_queue
from infra.models import AuditLog, Document, KnowledgeBase
from services.api.deps import UserCtx, get_current_user, get_db
from services.api.schemas.documents import (
    DocumentListOut,
    DocumentOut,
    FileNotice,
    UploadedFileInfo,
    UploadResult,
)

router = APIRouter(prefix="/api/v1", tags=["documents"])

DB = Annotated[Session, Depends(get_db)]
User = Annotated[UserCtx, Depends(get_current_user)]

# 文档状态常量（与 infra.models.Document.status 的枚举一致）
STATUS_PARSING = "parsing"
STATUS_FAILED = "failed"
STATUS_DELETED = "deleted"

# 扩展名白名单：PDF / Word / Excel / PPT / Markdown / HTML / 纯文本
ALLOWED_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".docx", ".doc",
        ".xlsx", ".xls",
        ".pptx", ".ppt",
        ".md", ".markdown",
        ".html", ".htm",
        ".txt",
    }
)


def _clean_filename(raw: str | None) -> str:
    """去掉客户端可能带上的路径成分，只留文件名本身（防对象 key 出现斜杠）。"""
    if not raw:
        return ""
    return raw.replace("\\", "/").rsplit("/", 1)[-1]


def _audit(db: Session, user_id: str, action: str, resource: str, detail: dict) -> None:
    """写操作审计（与业务同事务提交；合规要求保留 >= 180 天）。"""
    db.add(AuditLog(user_id=user_id, action=action, resource=resource, detail=detail))


@router.post("/documents", response_model=UploadResult)
async def upload_documents(
    kb_id: Annotated[int, Form(description="目标知识库 id")],
    files: Annotated[list[UploadFile], File(description="批量上传的文件列表")],
    db: DB,
    user: User,
) -> UploadResult:
    """批量上传文档（multipart，部分成功语义，HTTP 恒 200）。

    逐文件处理流程：
    1. 扩展名不在白名单 → rejected(unsupported_format)，跳过；
    2. 计算 sha256，同 kb 内已有相同指纹且未软删的文档 → skipped(duplicate)；
    3. 否则插入 status="parsing" 的 document 行 → 写 MinIO（key=kb/{kb_id}/{doc_id}/{filename}）
       → enqueue(TASK_PARSE, document_id)；
    4. MinIO / 入队异常 → 该文档行标 failed 并进 rejected（storage_error / enqueue_error），
       不影响其余文件（MinIO 已写成功但入队失败时可能残留一个对象，属可接受的边界损耗）。

    kb_id 不存在 → 404（整单失败，不落任何行）。
    """
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail=f"knowledge_base {kb_id} not found")

    # 幂等确保桶存在；失败则整单 500（此时尚未写入任何 document 行，无脏数据）
    minio_storage.ensure_bucket()

    result = UploadResult()
    for f in files:
        filename = _clean_filename(f.filename)
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        # 1) 扩展名白名单
        if f".{ext}" not in ALLOWED_EXTENSIONS:
            result.rejected.append(FileNotice(filename=filename, reason="unsupported_format"))
            continue

        content = await f.read()
        digest = hashlib.sha256(content).hexdigest()
        content_type = f.content_type or "application/octet-stream"

        # 2) 同 kb 内容去重（软删的不算重复，允许重新上传）
        dup = (
            db.query(Document.id)
            .filter(
                Document.kb_id == kb_id,
                Document.sha256 == digest,
                Document.status != STATUS_DELETED,
            )
            .first()
        )
        if dup is not None:
            result.skipped.append(FileNotice(filename=filename, reason="duplicate"))
            continue

        # 3) 落库（先 flush 拿自增 id，才能拼 MinIO 对象 key）
        doc = Document(
            kb_id=kb_id,
            source_type=ext,
            sha256=digest,
            title=filename,
            status=STATUS_PARSING,
            filename=filename,
            content_type=content_type,
            size_bytes=len(content),
        )
        db.add(doc)
        db.flush()
        doc.minio_object_key = minio_storage.object_key(kb_id, doc.id, filename)

        try:
            minio_storage.put_object(doc.minio_object_key, content, content_type=content_type)
        except Exception:  # noqa: BLE001 -- 部分成功语义：任一文件失败不中断整批
            doc.status = STATUS_FAILED
            db.flush()
            result.rejected.append(FileNotice(filename=filename, reason="storage_error"))
            continue

        try:
            task_queue.get_ingest_queue().enqueue(task_queue.TASK_PARSE, doc.id)
        except Exception:  # noqa: BLE001 -- 部分成功语义：入队失败仅该文件标 failed
            doc.status = STATUS_FAILED
            db.flush()
            result.rejected.append(FileNotice(filename=filename, reason="enqueue_error"))
            continue

        result.uploaded.append(UploadedFileInfo(id=doc.id, filename=filename))
        _audit(
            db,
            user.user_id,
            "document.upload",
            f"document:{doc.id}",
            {"kb_id": kb_id, "filename": filename, "sha256": digest, "size_bytes": len(content)},
        )

    db.commit()
    return result


@router.get("/documents", response_model=DocumentListOut)
def list_documents(
    db: DB,
    user: User,
    kb_id: Annotated[int | None, Query(description="按知识库过滤")] = None,
    status: Annotated[str | None, Query(description="parsing|ready|failed|deleted")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> DocumentListOut:
    """文档分页列表：{total, items}，按 id 倒序（新文档在前）。

    默认不返回软删文档；显式传 status=deleted 可查软删（管理/排查用）。
    """
    query = db.query(Document)
    if kb_id is not None:
        query = query.filter(Document.kb_id == kb_id)
    if status is not None:
        query = query.filter(Document.status == status)
    else:
        query = query.filter(Document.status != STATUS_DELETED)

    total = query.count()
    items = (
        query.order_by(Document.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return DocumentListOut(total=total, items=items)  # type: ignore[arg-type]


@router.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, db: DB, user: User) -> Document:
    """文档详情；不存在或已软删 → 404（软删文档对外视为不存在）。"""
    doc = (
        db.query(Document)
        .filter(Document.id == document_id, Document.status != STATUS_DELETED)
        .first()
    )
    if doc is None:
        raise HTTPException(status_code=404, detail=f"document {document_id} not found")
    return doc


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: int, db: DB, user: User) -> None:
    """软删除文档：status 置 deleted + enqueue(TASK_PURGE)，向量清理由 F2 worker 异步执行。

    幂等语义（本实现选定 404 方案）：文档不存在 **或已软删** 一律返回 404，
    与 GET 详情的可见性语义保持一致；重复 DELETE 因此天然幂等。
    入队失败则整体 500 且不提交（文档仍为原状态），客户端可安全重试。
    """
    doc = (
        db.query(Document)
        .filter(Document.id == document_id, Document.status != STATUS_DELETED)
        .first()
    )
    if doc is None:
        raise HTTPException(status_code=404, detail=f"document {document_id} not found")

    doc.status = STATUS_DELETED
    db.flush()
    task_queue.get_ingest_queue().enqueue(task_queue.TASK_PURGE, doc.id)
    _audit(
        db,
        user.user_id,
        "document.delete",
        f"document:{doc.id}",
        {"kb_id": doc.kb_id, "filename": doc.filename},
    )
    db.commit()
