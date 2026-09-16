"""services.worker.tasks 端到端单测：内存 sqlite + fake 网关 + mock MinIO/Milvus（零外部服务）。

rq 任务是普通函数，直接同步调用即可，无需起 worker。
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import infra.milvus
import infra.minio
from infra.models import Base, Chunk, Document
from services.worker import tasks

MD = """# 部署指南

本指南介绍系统的部署流程与注意事项。

## 安装

执行安装脚本即可完成基础安装。

| 步骤 | 命令 |
|---|---|
| 1 | pip install -e . |
| 2 | python -m services.worker |

### 注意事项

安装完成后需要重启服务才能生效。
"""
# MD 切分结果固定为：3 个叶子章节（概要/安装/注意事项）-> 3 父块；子块 1+2+1=4
EXPECTED_PARENTS = 3
EXPECTED_CHILDREN = 4


@pytest.fixture
def session_factory(monkeypatch):
    """内存 sqlite 会话工厂，并替换 tasks 的 SessionLocal（任务自包含开会话）。"""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(tasks, "SessionLocal", factory)
    return factory


class _MilvusRecorder:
    def __init__(self):
        self.inserted: list[dict] = []
        self.deleted: list[list[int]] = []


@pytest.fixture
def milvus_recorder(monkeypatch):
    rec = _MilvusRecorder()
    monkeypatch.setattr(infra.milvus, "insert_chunks", lambda rows: rec.inserted.extend(rows))
    monkeypatch.setattr(infra.milvus, "delete_chunks", lambda ids: rec.deleted.append(list(ids)))
    return rec


class _FakeMinioClient:
    def __init__(self, removed: list):
        self._removed = removed

    def remove_object(self, bucket: str, key: str):
        self._removed.append((bucket, key))


@pytest.fixture
def minio_store(monkeypatch):
    objects: dict[str, bytes] = {}
    removed: list[tuple[str, str]] = []
    monkeypatch.setattr(infra.minio, "get_object", lambda key: objects[key])
    monkeypatch.setattr(infra.minio, "get_minio", lambda: _FakeMinioClient(removed))
    return objects, removed


def _make_doc(factory, *, filename="guide.md", status="parsing", key="kb/1/1/guide.md", kb_id=1):
    with factory() as db:
        doc = Document(
            kb_id=kb_id,
            source_type="upload",
            status=status,
            filename=filename,
            title="任意标题",
            sha256="0" * 64,
            minio_object_key=key,
        )
        db.add(doc)
        db.commit()
        return doc.id


def test_parse_document_success(session_factory, milvus_recorder, minio_store, fake_gateway):
    objects, _ = minio_store
    key = "kb/1/1/guide.md"
    objects[key] = MD.encode("utf-8")
    doc_id = _make_doc(session_factory, key=key)

    tasks.parse_document(doc_id)

    with session_factory() as db:
        doc = db.get(Document, doc_id)
        assert doc.status == "ready"
        assert doc.parse_meta["parent_count"] == EXPECTED_PARENTS
        assert doc.parse_meta["chunk_count"] == EXPECTED_CHILDREN
        chunks = db.query(Chunk).filter(Chunk.doc_id == doc_id).order_by(Chunk.seq).all()
        parents = [c for c in chunks if c.parent_id is None]
        children = [c for c in chunks if c.parent_id is not None]
        assert len(parents) == EXPECTED_PARENTS
        assert len(children) == EXPECTED_CHILDREN
        # seq 从 0 连续
        assert [c.seq for c in chunks] == list(range(len(chunks)))
        # 子块 parent_id 均指向本文档父块行
        parent_ids = {c.id for c in parents}
        assert all(c.parent_id in parent_ids for c in children)
        # headings 落库为路径列表
        assert ["部署指南", "安装"] in [c.headings for c in chunks]
        # small-to-big：父块内容包含其全部子块
        for p in parents:
            kids = [c for c in children if c.parent_id == p.id]
            assert kids and all(k.content in p.content for k in kids)
        contents = {c.id: c.content for c in children}
        child_ids = set(contents)

    inserted = milvus_recorder.inserted
    assert milvus_recorder.deleted == []  # 首次入库无旧数据
    milvus_ids = {r["chunk_id"] for r in inserted}
    # 向量库恰好收到全部子块，且父块不进向量库
    assert milvus_ids == child_ids
    assert milvus_ids == {r["chunk_id"] for r in inserted}
    parent_only_ids = {c.id for c in parents}
    assert milvus_ids.isdisjoint(parent_only_ids)
    for r in inserted:
        assert set(r.keys()) == {"chunk_id", "kb_id", "doc_id", "dense", "sparse", "text"}
        assert r["kb_id"] == 1
        assert r["doc_id"] == doc_id
        assert len(r["dense"]) == 1024  # fake 网关 dim 与 settings.embed_dim 一致
        assert r["sparse"] and all(isinstance(k, int) and v > 0 for k, v in r["sparse"].items())
        assert r["text"] == contents[r["chunk_id"]]  # text 与 PG 子块内容一致


def test_parse_document_idempotent_rerun(session_factory, milvus_recorder, minio_store, fake_gateway):
    objects, _ = minio_store
    key = "kb/1/1/guide.md"
    objects[key] = MD.encode("utf-8")
    doc_id = _make_doc(session_factory, key=key)

    tasks.parse_document(doc_id)
    first_run_rows = len(milvus_recorder.inserted)
    first_ids = {r["chunk_id"] for r in milvus_recorder.inserted}
    tasks.parse_document(doc_id)

    with session_factory() as db:
        chunks = db.query(Chunk).filter(Chunk.doc_id == doc_id).all()
        # 不产生重复行：总数不变、(doc_id, seq) 唯一
        assert len(chunks) == EXPECTED_PARENTS + EXPECTED_CHILDREN
        assert len({c.seq for c in chunks}) == len(chunks)
    # 第二轮重新写入等量向量（sqlite 会复用 rowid，PG Identity 下是新 id，不断言新旧）
    assert len(milvus_recorder.inserted) == first_run_rows + EXPECTED_CHILDREN
    # 且重跑前先清理了第一轮写入的旧向量
    assert milvus_recorder.deleted
    assert first_ids <= set(milvus_recorder.deleted[-1])


def test_parse_document_skips_deleted(session_factory, milvus_recorder, minio_store):
    # MinIO store 里故意不放对象：若未跳过会因取不到对象而 failed
    doc_id = _make_doc(session_factory, status="deleted")
    tasks.parse_document(doc_id)
    assert milvus_recorder.inserted == []
    with session_factory() as db:
        assert db.query(Chunk).count() == 0
        assert db.get(Document, doc_id).status == "deleted"


def test_parse_document_failure_marks_failed(session_factory, milvus_recorder, minio_store):
    objects, _ = minio_store
    key = "kb/1/2/scan.pdf"
    objects[key] = b"%PDF-1.4 fake"
    doc_id = _make_doc(session_factory, filename="scan.pdf", key=key)

    tasks.parse_document(doc_id)  # 不向外抛

    with session_factory() as db:
        doc = db.get(Document, doc_id)
        assert doc.status == "failed"
        assert "mineru" in doc.parse_meta["error"]
        assert db.query(Chunk).filter(Chunk.doc_id == doc_id).count() == 0
    assert milvus_recorder.inserted == []


def test_parse_document_rerun_failure_clears_old_chunks(
    session_factory, milvus_recorder, minio_store, fake_gateway, monkeypatch
):
    objects, _ = minio_store
    key = "kb/1/1/guide.md"
    objects[key] = MD.encode("utf-8")
    doc_id = _make_doc(session_factory, key=key)
    tasks.parse_document(doc_id)
    with session_factory() as db:
        assert db.query(Chunk).filter(Chunk.doc_id == doc_id).count() == EXPECTED_PARENTS + EXPECTED_CHILDREN

    # 重新解析时解析器抛错：旧 chunks（PG + Milvus）应被幂等清理，文档置 failed
    def _boom(filename, data):
        raise ValueError("boom")

    monkeypatch.setattr(tasks, "parse_document_file", _boom)
    tasks.parse_document(doc_id)

    with session_factory() as db:
        doc = db.get(Document, doc_id)
        assert doc.status == "failed"
        assert doc.parse_meta["error"] == "boom"
        assert db.query(Chunk).filter(Chunk.doc_id == doc_id).count() == 0
    assert {r["chunk_id"] for r in milvus_recorder.inserted} <= set(milvus_recorder.deleted[-1])


def test_parse_document_embed_batches(session_factory, milvus_recorder, minio_store, fake_gateway, monkeypatch):
    objects, _ = minio_store
    paras = [
        f"第{i}段落说明。" + "这是用于批量向量化测试的中文长句子，覆盖切分与嵌入全流程。" * 6
        for i in range(200)
    ]
    objects["kb/1/9/long.md"] = ("# 长文档\n\n" + "\n\n".join(paras)).encode("utf-8")
    doc_id = _make_doc(session_factory, key="kb/1/9/long.md")

    batch_sizes = []
    original_embed = fake_gateway.embed

    def _record(texts):
        batch_sizes.append(len(texts))
        return original_embed(texts)

    monkeypatch.setattr(fake_gateway, "embed", _record)

    tasks.parse_document(doc_id)

    with session_factory() as db:
        doc = db.get(Document, doc_id)
        assert doc.status == "ready"
        chunk_count = doc.parse_meta["chunk_count"]
        assert chunk_count > tasks.EMBED_BATCH_SIZE  # 足以触发多批
    assert len(batch_sizes) >= 2
    assert all(s <= tasks.EMBED_BATCH_SIZE for s in batch_sizes)
    assert sum(batch_sizes) == chunk_count


def test_purge_document(session_factory, milvus_recorder, minio_store, fake_gateway):
    objects, removed = minio_store
    key = "kb/1/1/guide.md"
    objects[key] = MD.encode("utf-8")
    doc_id = _make_doc(session_factory, key=key)
    tasks.parse_document(doc_id)
    assert milvus_recorder.inserted

    tasks.purge_document(doc_id)

    with session_factory() as db:
        assert db.query(Chunk).filter(Chunk.doc_id == doc_id).count() == 0
    # Milvus 删除收到该 doc 全部 chunk id（至少包含全部进过向量库的子块 id）
    inserted_ids = {r["chunk_id"] for r in milvus_recorder.inserted}
    assert inserted_ids <= set(milvus_recorder.deleted[-1])
    # MinIO 原始对象被删除
    assert removed and removed[0][1] == key

    # 幂等：重跑不报错、无多余删除
    n_deletes = len(milvus_recorder.deleted)
    tasks.purge_document(doc_id)
    assert len(milvus_recorder.deleted) == n_deletes


def test_purge_document_nonexistent_doc(session_factory, milvus_recorder, minio_store):
    tasks.purge_document(99999)  # 不存在：静默幂等
    assert milvus_recorder.deleted == []
