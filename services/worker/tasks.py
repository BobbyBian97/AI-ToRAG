"""rq 入库任务（F2 流水线）：解析 -> 切分 -> 向量化 -> 入库（PG chunk 行 + Milvus 向量）。

与 infra.queue 的契约对齐：
- TASK_PARSE = "services.worker.tasks.parse_document"（F1 上传入队）
- TASK_PURGE = "services.worker.tasks.purge_document"（F1 删除文档后入队清理）

设计要点（设计书 §4.1）：
- 自包含会话：函数内自己 SessionLocal() 开会话（rq worker 进程与 API 无共享会话）；
- 幂等：入库前先删该 doc 旧 chunks（Milvus 先删、PG 后删，PG 为事实源），可安全重跑；
- 父块行（parent_id=None）不进向量库；子块批量向量化后写入 Milvus（chunk_id=PG chunk.id）；
- 阶段化：每阶段完成即落库（status 推进 + parse_meta.stages 合并指标），后台管理可实时
  观察进度；失败时 parse_meta.failed_stage 记录失败阶段（parse | chunk | embed | index）；
- 任何异常：回滚、清理本任务已写入的 Milvus 孤儿向量，置 status="failed" 后
  不向外抛（rq 重试无意义，失败的文档由人工/上游重新入队处理）。
"""
from __future__ import annotations

import logging
import time

from sqlalchemy.orm import Session

from core.chunking import ChunkDraft, chunk_document
from core.parsing import ParsedDoc, parse_document_file
from infra import milvus, minio
from infra.config import get_settings
from infra.models import Chunk, Document
from infra.pg import SessionLocal
from services import model_gateway
from services.model_gateway.base import Embedding

logger = logging.getLogger(__name__)

# 批量向量化上限（与 BGE-M3 批大小习惯一致；测试也依赖此值断言分批）
EMBED_BATCH_SIZE = 32


def parse_document(document_id: int) -> None:
    """解析入库主任务（阶段化）：MinIO 取文件 -> 解析 -> 切分 -> 向量化 -> PG + Milvus。

    每阶段完成即落库（status 沿 PIPELINE_STAGES 推进 + parse_meta.stages 合并指标），
    后台管理可实时观察进度；失败时 parse_meta.failed_stage 记录失败阶段。
    幂等：重跑前清空该 doc 的旧 chunks（先 Milvus 后 PG），同文档重跑不产生重复行。
    """
    db: Session = SessionLocal()
    inserted_milvus_ids: list[int] = []  # 本任务已写入 Milvus 的 id，失败时回滚清理
    stage = "parse"
    try:
        doc = db.get(Document, document_id)
        if doc is None:
            logger.warning("文档 %s 不存在，跳过解析", document_id)
            return
        if doc.status == "deleted":
            logger.info("文档 %s 已软删，跳过解析", document_id)
            return

        # 重跑清零：去掉上一轮残留的 failed_stage / error，成功后不留脏标记（随首个阶段提交落库）
        doc.parse_meta = {"stages": {}}

        _stage_cleanup(db, doc)

        stage = "parse"
        parsed = _stage_fetch_and_parse(db, doc)

        stage = "chunk"
        child_rows, child_drafts, parent_count = _stage_chunk(db, doc, parsed)

        stage = "embed"
        embeddings = _stage_embed(db, doc, child_drafts)

        stage = "index"
        _stage_index(db, doc, child_rows, child_drafts, parent_count, embeddings, inserted_milvus_ids)
        logger.info(
            "文档 %s 解析入库完成：%s 子块 / %s 父块", document_id, len(child_rows), parent_count
        )
    except Exception as exc:
        logger.exception("文档 %s 入库失败（阶段=%s）", document_id, stage)
        db.rollback()
        # PG 已回滚；Milvus 无法回滚事务，手动清掉本任务写入的孤儿向量，保持两侧一致
        if inserted_milvus_ids:
            try:
                milvus.delete_chunks(inserted_milvus_ids)
            except Exception:
                logger.exception("清理 Milvus 孤儿向量失败 chunk_ids=%s", inserted_milvus_ids)
        try:
            doc = db.get(Document, document_id)
            if doc is not None:
                doc.status = "failed"
                # 合并写入（保留已完成阶段的指标），绝不整体覆盖 parse_meta
                meta = dict(doc.parse_meta or {})
                meta["failed_stage"] = stage
                meta["error"] = str(exc)
                doc.parse_meta = meta
                db.commit()
        except Exception:
            logger.exception("标记文档 %s 为 failed 失败", document_id)
        # 不向外抛：rq 重试无意义（可重试的瞬时错误由上游重新入队）
    finally:
        db.close()


def _commit_stage(db: Session, doc: Document, status: str, stage: str, **metrics) -> None:
    """推进阶段：status 置新值 + parse_meta.stages[stage] 记指标，立即提交（复制后重赋值，JSON 列不追踪原地修改）。"""
    doc.status = status
    meta = dict(doc.parse_meta or {})
    stages = dict(meta.get("stages") or {})
    stages[stage] = metrics
    meta["stages"] = stages
    doc.parse_meta = meta
    db.commit()


def _stage_cleanup(db: Session, doc: Document) -> None:
    """幂等清理：先删该 doc 旧 chunks（Milvus 先删；PG 删除单独提交，保证失败态下不留半数据）。"""
    old_ids = [cid for (cid,) in db.query(Chunk.id).filter(Chunk.doc_id == doc.id).all()]
    if old_ids:
        milvus.delete_chunks(old_ids)
        db.query(Chunk).filter(Chunk.doc_id == doc.id).delete(synchronize_session=False)
        db.commit()


def _stage_fetch_and_parse(db: Session, doc: Document) -> ParsedDoc:
    """阶段 1（parse）：MinIO 取原始文件 -> 解析为结构化中间格式 ParsedDoc。"""
    t0 = time.perf_counter()
    data = minio.get_object(doc.minio_object_key)
    parsed = parse_document_file(doc.filename or f"{doc.id}.txt", data)
    filename = doc.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    _commit_stage(db, doc, "parsed", "parse", parser=ext,
                  duration_ms=int((time.perf_counter() - t0) * 1000))
    return parsed


def _stage_chunk(
    db: Session, doc: Document, parsed: ParsedDoc
) -> tuple[list[Chunk], list[ChunkDraft], int]:
    """阶段 2（chunk）：标题感知切分 -> 落 PG（父块先行拿 id，子块回填 parent_id）。"""
    t0 = time.perf_counter()
    drafts: list[ChunkDraft] = chunk_document(parsed)
    parent_ids: dict[int, int] = {}  # parent_seq -> 父块行 id
    child_rows: list[Chunk] = []
    child_drafts: list[ChunkDraft] = []
    for d in drafts:
        row = Chunk(
            doc_id=doc.id,
            kb_id=doc.kb_id,
            parent_id=parent_ids.get(d.parent_seq),
            seq=d.seq,
            content=d.content,
            token_count=d.token_count,
            page_no=d.page_no,
            headings=d.headings.split(" > ") if d.headings else None,
        )
        db.add(row)
        db.flush()  # 立即取自增主键，供父子关联与 Milvus chunk_id 使用
        if d.parent_seq is None:
            parent_ids[d.seq] = row.id
        else:
            child_rows.append(row)
            child_drafts.append(d)
    _commit_stage(db, doc, "chunked", "chunk",
                  parent_count=len(parent_ids), child_count=len(child_rows),
                  duration_ms=int((time.perf_counter() - t0) * 1000))
    return child_rows, child_drafts, len(parent_ids)


def _stage_embed(db: Session, doc: Document, child_drafts: list[ChunkDraft]) -> list[Embedding]:
    """阶段 3（embed）：子块分批向量化（只计算，不写 Milvus）。"""
    t0 = time.perf_counter()
    gateway = model_gateway.get_gateway()
    embeddings: list[Embedding] = []
    batch_count = 0
    for i in range(0, len(child_drafts), EMBED_BATCH_SIZE):
        batch = child_drafts[i : i + EMBED_BATCH_SIZE]
        embs = gateway.embed([d.content for d in batch])
        if len(embs) != len(batch):
            raise ValueError(f"embedding 返回数量不符：期望 {len(batch)}，实际 {len(embs)}")
        embeddings.extend(embs)
        batch_count += 1
    _commit_stage(db, doc, "embedded", "embed", batch_count=batch_count,
                  duration_ms=int((time.perf_counter() - t0) * 1000))
    return embeddings


def _stage_index(
    db: Session,
    doc: Document,
    child_rows: list[Chunk],
    child_drafts: list[ChunkDraft],
    parent_count: int,
    embeddings: list[Embedding],
    inserted_milvus_ids: list[int],
) -> None:
    """阶段 4（index）：向量分批写 Milvus（chunk_id=PG chunk.id）-> status=ready。

    ready 的提交紧邻最后一次 Milvus insert：PG 事实源与向量库尽可能同时达成一致视图。
    """
    t0 = time.perf_counter()
    for i in range(0, len(child_drafts), EMBED_BATCH_SIZE):
        rows = [
            {
                "chunk_id": row.id,
                "kb_id": doc.kb_id,
                "doc_id": doc.id,
                "dense": emb.dense,
                "sparse": emb.sparse,
                "text": d.content,
            }
            for row, d, emb in zip(
                child_rows[i : i + EMBED_BATCH_SIZE],
                child_drafts[i : i + EMBED_BATCH_SIZE],
                embeddings[i : i + EMBED_BATCH_SIZE],
            )
        ]
        milvus.insert_chunks(rows)
        inserted_milvus_ids.extend(r["chunk_id"] for r in rows)
    meta = dict(doc.parse_meta or {})
    stages = dict(meta.get("stages") or {})
    stages["index"] = {"duration_ms": int((time.perf_counter() - t0) * 1000)}
    meta["stages"] = stages
    meta["chunk_count"] = len(child_rows)  # 子块数（进向量库的块）
    meta["parent_count"] = parent_count  # 父块数（章节块，不进向量库）
    doc.parse_meta = meta
    doc.status = "ready"
    db.commit()


def purge_document(document_id: int) -> None:
    """删除文档后的清理任务：Milvus 向量 -> PG chunks -> MinIO 原始文件。

    全程幂等：对象不存在 / 无 chunk 均不报错。Milvus 删除失败则不动 PG
    （PG 为事实源，保留行以便之后重试清理）。
    """
    db: Session = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        chunk_ids = [cid for (cid,) in db.query(Chunk.id).filter(Chunk.doc_id == document_id).all()]
        if chunk_ids:
            milvus.delete_chunks(chunk_ids)
            db.query(Chunk).filter(Chunk.doc_id == document_id).delete(synchronize_session=False)
            db.commit()
        if doc is not None and doc.minio_object_key:
            try:
                minio.get_minio().remove_object(get_settings().minio_bucket, doc.minio_object_key)
            except Exception:
                # 容忍对象不存在 / MinIO 瞬时不可用：原始文件不阻碍元数据清理
                logger.warning("删除 MinIO 对象失败（容忍不存在）key=%s", doc.minio_object_key, exc_info=True)
        logger.info("文档 %s 清理完成（chunks=%s）", document_id, len(chunk_ids))
    except Exception:
        db.rollback()
        logger.exception("文档 %s 清理失败", document_id)
    finally:
        db.close()
