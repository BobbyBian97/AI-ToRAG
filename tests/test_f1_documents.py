"""F1 文档接入测试：批量上传 / 去重 / 状态机 / 列表过滤 / 软删幂等。

外部依赖全部打桩（不依赖真实 MinIO / Redis）：
- fake_minio：monkeypatch infra.minio 的 put/get/ensure_bucket，用 dict 存对象；
- fake_queue：monkeypatch infra.queue.get_ingest_queue，记录 enqueue 调用参数。
路由侧通过 `import infra.minio as minio_storage` / `import infra.queue as task_queue`
以模块属性访问外部依赖，因此上述打桩直接生效。
"""

import pytest

import infra.minio
import infra.queue
from infra.queue import TASK_PARSE, TASK_PURGE

DOC_PATH = "/api/v1/documents"
KB_PATH = "/api/v1/knowledge-bases"


class FakeQueue:
    """假入库队列：只记录 enqueue 的任务名与位置参数。"""

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def enqueue(self, task: str, *args, **kwargs) -> None:
        self.calls.append((task, args))

    def calls_for(self, task: str) -> list[tuple]:
        return [args for name, args in self.calls if name == task]


@pytest.fixture
def fake_queue(monkeypatch):
    q = FakeQueue()
    monkeypatch.setattr(infra.queue, "get_ingest_queue", lambda: q)
    return q


@pytest.fixture
def fake_minio(monkeypatch):
    """对象存储桩：{key: (bytes, content_type)}。"""
    store: dict[str, tuple[bytes, str]] = {}

    def _put(key, data, content_type="application/octet-stream"):
        store[key] = (data, content_type)

    monkeypatch.setattr(infra.minio, "put_object", _put)
    monkeypatch.setattr(infra.minio, "get_object", lambda key: store[key][0])
    monkeypatch.setattr(infra.minio, "ensure_bucket", lambda: None)
    return store


def _create_kb(client, name="默认库"):
    resp = client.post(KB_PATH, json={"name": name})
    assert resp.status_code == 201
    return resp.json()["id"]


def _upload(client, kb_id: int, files: list[tuple[str, bytes, str]]):
    """files: [(filename, content, content_type), ...]"""
    return client.post(
        DOC_PATH,
        data={"kb_id": str(kb_id)},
        files=[("files", (name, content, ctype)) for name, content, ctype in files],
    )


def test_upload_partial_success(client, fake_minio, fake_queue):
    """3 个合法 + 1 个非法扩展 → 部分成功；合法文档 status=parsing 且正确入队。"""
    kb_id = _create_kb(client)
    resp = _upload(
        client,
        kb_id,
        [
            ("a.md", b"# hi", "text/markdown"),
            ("b.txt", b"hello", "text/plain"),
            ("c.html", b"<p>x</p>", "text/html"),
            ("d.exe", b"MZ...", "application/octet-stream"),
        ],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["uploaded"]) == 3
    assert body["skipped"] == []
    assert body["rejected"] == [{"filename": "d.exe", "reason": "unsupported_format"}]

    # 每个成功文档：status=parsing、MinIO key 符合 kb/{kb_id}/{doc_id}/{filename} 约定
    for item in body["uploaded"]:
        detail = client.get(f"{DOC_PATH}/{item['id']}").json()
        assert detail["status"] == "parsing"
        assert detail["title"] == item["filename"]
        assert detail["minio_object_key"] == f"kb/{kb_id}/{item['id']}/{item['filename']}"
        assert detail["minio_object_key"] in fake_minio
    # 对象内容与上传内容一致
    assert fake_minio[f"kb/{kb_id}/{body['uploaded'][0]['id']}/a.md"][0] == b"# hi"

    # 入队契约：TASK_PARSE + document_id 位置参数，每个成功文档恰好一次
    assert sorted(fake_queue.calls_for(TASK_PARSE)) == sorted(
        (item["id"],) for item in body["uploaded"]
    )
    assert fake_queue.calls_for(TASK_PURGE) == []


def test_upload_sha256_dedup_skipped_and_no_enqueue(client, fake_minio, fake_queue):
    """同 kb 同内容 → skipped(duplicate) 且不再入队；不同内容同名文件不误伤。"""
    kb_id = _create_kb(client)
    first = _upload(client, kb_id, [("a.md", b"same content", "text/markdown")])
    assert first.json()["uploaded"] and first.json()["skipped"] == []

    second = _upload(client, kb_id, [("a.md", b"same content", "text/markdown")])
    assert second.status_code == 200
    body = second.json()
    assert body["uploaded"] == []
    assert body["skipped"] == [{"filename": "a.md", "reason": "duplicate"}]
    assert len(fake_queue.calls_for(TASK_PARSE)) == 1  # 未因重复文件再次入队

    # 内容不同（sha256 不同）即使同名也正常入库
    third = _upload(client, kb_id, [("a.md", b"different content", "text/markdown")])
    assert third.json()["uploaded"] and third.json()["skipped"] == []
    assert len(fake_queue.calls_for(TASK_PARSE)) == 2


def test_upload_kb_not_found_404(client, fake_minio, fake_queue):
    resp = _upload(client, 99999, [("a.md", b"x", "text/markdown")])
    assert resp.status_code == 404
    assert fake_queue.calls == []  # 未入队
    assert fake_minio == {}  # 未写对象


def test_minio_failure_marks_failed_and_rejected(client, fake_minio, fake_queue, monkeypatch):
    """MinIO 异常 → 该文档行标 failed 并进 rejected，不影响其余文件。"""
    kb_id = _create_kb(client)

    real_put = infra.minio.put_object

    def flaky_put(key, data, content_type="application/octet-stream"):
        if key.endswith("bad.pdf"):
            raise RuntimeError("minio down")
        real_put(key, data, content_type)

    monkeypatch.setattr(infra.minio, "put_object", flaky_put)

    resp = _upload(
        client,
        kb_id,
        [("ok.md", b"fine", "text/markdown"), ("bad.pdf", b"%PDF", "application/pdf")],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [u["filename"] for u in body["uploaded"]] == ["ok.md"]
    assert body["rejected"] == [{"filename": "bad.pdf", "reason": "storage_error"}]

    # failed 文档仍可查（仅 deleted 不可见），ok.md 正常入队
    failed_id = client.get(DOC_PATH, params={"kb_id": kb_id, "status": "failed"}).json()["items"][0]["id"]
    assert client.get(f"{DOC_PATH}/{failed_id}").json()["status"] == "failed"
    assert len(fake_queue.calls_for(TASK_PARSE)) == 1


def test_list_filter_by_kb_and_status(client, fake_minio, fake_queue):
    kb1, kb2 = _create_kb(client, "库一"), _create_kb(client, "库二")
    assert _upload(client, kb1, [("a.md", b"aaa", "text/markdown")]).json()["uploaded"]
    assert _upload(client, kb1, [("b.txt", b"bbb", "text/plain")]).json()["uploaded"]
    assert _upload(client, kb2, [("c.txt", b"ccc", "text/plain")]).json()["uploaded"]

    # 按 kb_id 过滤
    r1 = client.get(DOC_PATH, params={"kb_id": kb1})
    assert r1.status_code == 200
    assert r1.json()["total"] == 2
    assert all(item["kb_id"] == kb1 for item in r1.json()["items"])

    # 按 status 过滤：上传后全部为 parsing；ready 为 0
    rp = client.get(DOC_PATH, params={"kb_id": kb1, "status": "parsing"})
    assert rp.json()["total"] == 2
    rr = client.get(DOC_PATH, params={"status": "ready"})
    assert rr.json() == {"total": 0, "items": []}

    # 分页
    page = client.get(DOC_PATH, params={"page": 1, "page_size": 2})
    assert page.json()["total"] == 3
    assert len(page.json()["items"]) == 2


def test_soft_delete_get_404_and_idempotent(client, fake_minio, fake_queue):
    """软删：DELETE 204 + TASK_PURGE 入队；GET 404；重复 DELETE 仍 404（选定语义）。"""
    kb_id = _create_kb(client)
    doc_id = _upload(client, kb_id, [("del.md", b"bye", "text/markdown")]).json()["uploaded"][0]["id"]

    assert client.get(f"{DOC_PATH}/{doc_id}").status_code == 200
    assert client.delete(f"{DOC_PATH}/{doc_id}").status_code == 204

    # TASK_PURGE 入队参数 = (document_id,)
    assert fake_queue.calls_for(TASK_PURGE) == [(doc_id,)]
    # 软删后：详情 404、默认列表不可见、显式 status=deleted 可见
    assert client.get(f"{DOC_PATH}/{doc_id}").status_code == 404
    assert client.get(DOC_PATH, params={"kb_id": kb_id}).json()["total"] == 0
    deleted = client.get(DOC_PATH, params={"status": "deleted"})
    assert deleted.json()["total"] == 1
    assert deleted.json()["items"][0]["status"] == "deleted"

    # 幂等：再次删除已 deleted 的文档 → 404
    assert client.delete(f"{DOC_PATH}/{doc_id}").status_code == 404
    assert len(fake_queue.calls_for(TASK_PURGE)) == 1  # 不重复入队


def test_get_document_404(client, fake_minio, fake_queue):
    assert client.get(f"{DOC_PATH}/99999").status_code == 404
