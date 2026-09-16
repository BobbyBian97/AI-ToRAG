"""F3 端到端测试：POST /api/v1/chat 的 SSE 流、落库与参数校验。

复用 conftest 的 client（fake 网关 + 内存 sqlite），Milvus 用 monkeypatch 固定返回，
全程零真实外部服务。
"""
import json
import re

import pytest
from sqlalchemy import select

import infra.milvus
from core.prompt import NO_RESULT_TEXT
from infra.models import Chunk, Conversation, Document, KnowledgeBase, Message
from infra.pg import get_db
from services.model_gateway import RerankHit


@pytest.fixture
def db(client):
    """拿到 client 背后同一个内存库的会话，用于预置/校验数据（StaticPool 单连接共享）。"""
    session = next(client.app.dependency_overrides[get_db]())
    yield session
    session.close()


@pytest.fixture
def seeded(db):
    kb = KnowledgeBase(name="测试知识库")
    db.add(kb)
    db.commit()
    db.refresh(kb)
    doc = Document(kb_id=kb.id, title="权限过滤手册", status="ready")
    db.add(doc)
    db.commit()
    db.refresh(doc)
    c1 = Chunk(doc_id=doc.id, kb_id=kb.id, seq=1, content="权限过滤必须在召回阶段执行", page_no=3)
    c2 = Chunk(doc_id=doc.id, kb_id=kb.id, seq=2, content="混合检索用 RRF 融合两路结果", page_no=5)
    db.add_all([c1, c2])
    db.commit()
    db.refresh(c1)
    db.refresh(c2)
    return {"kb": kb, "doc": doc, "c1": c1, "c2": c2}


# ---------------- SSE 解析工具 ----------------

def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """把响应文本解析为 [(event, data)]，块格式非法即断言失败。"""
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        m_event = re.search(r"^event: (.+)$", block, re.MULTILINE)
        m_data = re.search(r"^data: (.+)$", block, re.MULTILINE)
        assert m_event and m_data, f"非法 SSE 块: {block!r}"
        events.append((m_event.group(1), json.loads(m_data.group(1))))
    return events


def _of(events: list[tuple[str, dict]], name: str) -> list[dict]:
    return [data for evt, data in events if evt == name]


def _patch_search(monkeypatch, dense, sparse):
    monkeypatch.setattr(
        infra.milvus, "search_hybrid",
        lambda kb_ids, d, s, top_k, doc_ids=None: (
            [infra.milvus.Hit(cid, sc) for cid, sc in dense],
            [infra.milvus.Hit(cid, sc) for cid, sc in sparse],
        ),
    )


# ---------------- 正常流 ----------------

def test_chat_sse_happy_path(client, db, seeded, fake_gateway, monkeypatch):
    s = seeded
    # 两路各命中一个 => RRF 并列（各 1/61）按 chunk_id 升序得候选 [c1, c2]
    _patch_search(monkeypatch, dense=[(s["c1"].id, 0.9)], sparse=[(s["c2"].id, 0.8)])
    # rerank 固定为逆序（c2 第一），同时把分数覆盖为固定值，保证 refs 完全确定
    monkeypatch.setattr(
        fake_gateway, "rerank",
        lambda q, docs, top_n: [RerankHit(index=1, score=0.9), RerankHit(index=0, score=0.4)],
    )

    resp = client.post(
        "/api/v1/chat",
        json={"kb_ids": [s["kb"].id], "query": "权限过滤应该在哪一步做？"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    names = [evt for evt, _ in events]
    # 事件顺序：delta+ -> refs -> done（refs 是流结束前的引用列表）
    assert names[0] == "delta"
    assert names[-2] == "refs"
    assert names[-1] == "done"
    assert "error" not in names

    joined = "".join(d["text"] for d in _of(events, "delta"))
    assert joined  # 非空
    assert "[1]" in joined or "[2]" in joined  # fake 网关输出带引用标记

    refs = _of(events, "refs")[0]
    assert refs == [
        {"chunk_id": s["c2"].id, "doc_id": s["doc"].id, "doc_title": "权限过滤手册",
         "page_no": 5, "score": 0.9},
        {"chunk_id": s["c1"].id, "doc_id": s["doc"].id, "doc_title": "权限过滤手册",
         "page_no": 3, "score": 0.4},
    ]

    done = _of(events, "done")[0]
    conv_id, msg_id = done["conversation_id"], done["message_id"]
    assert isinstance(conv_id, int) and isinstance(msg_id, int)
    assert isinstance(done["latency_ms"], int) and done["latency_ms"] >= 0

    # ---- 落库校验 ----
    conv = db.get(Conversation, conv_id)
    assert conv is not None
    assert conv.title == "权限过滤应该在哪一步做？"[:30]
    messages = list(
        db.execute(
            select(Message).where(Message.conversation_id == conv_id).order_by(Message.id)
        ).scalars()
    )
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "权限过滤应该在哪一步做？"
    assistant = messages[1]
    assert assistant.id == msg_id
    assert assistant.content == joined          # 完整回答
    assert assistant.refs == refs               # refs JSON 与事件一致
    assert assistant.latency_ms is not None and assistant.latency_ms >= 0


def test_chat_empty_recall(client, db, seeded, monkeypatch):
    s = seeded
    _patch_search(monkeypatch, dense=[], sparse=[])
    resp = client.post(
        "/api/v1/chat", json={"kb_ids": [s["kb"].id], "query": "完全不相关的问题"}
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)

    deltas = [d["text"] for d in _of(events, "delta")]
    assert deltas == [NO_RESULT_TEXT]           # 只发一条固定 delta
    assert _of(events, "refs") == [[]]          # 空检索 refs 为空列表
    done = _of(events, "done")[0]
    assert isinstance(done["message_id"], int)

    conv_id = done["conversation_id"]
    messages = list(
        db.execute(
            select(Message).where(Message.conversation_id == conv_id).order_by(Message.id)
        ).scalars()
    )
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].content == NO_RESULT_TEXT
    assert messages[1].refs == []


def test_chat_multi_turn_reuses_conversation(client, db, seeded, fake_gateway, monkeypatch):
    s = seeded
    _patch_search(monkeypatch, dense=[(s["c1"].id, 0.9)], sparse=[])
    monkeypatch.setattr(
        fake_gateway, "rerank", lambda q, docs, top_n: [RerankHit(index=0, score=0.9)]
    )
    body1 = {"kb_ids": [s["kb"].id], "query": "权限过滤在哪一步做？"}
    first = client.post("/api/v1/chat", json=body1)
    conv_id = _of(_parse_sse(first.text), "done")[0]["conversation_id"]

    # 第二轮带 conversation_id：走多轮改写路径，消息追加到同一会话
    second = client.post(
        "/api/v1/chat",
        json={"conversation_id": conv_id, "kb_ids": [s["kb"].id], "query": "为什么不能检索后再过滤？"},
    )
    assert second.status_code == 200
    events = _parse_sse(second.text)
    assert events[-1][0] == "done"
    assert _of(events, "done")[0]["conversation_id"] == conv_id

    messages = list(
        db.execute(
            select(Message).where(Message.conversation_id == conv_id).order_by(Message.id)
        ).scalars()
    )
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
    total_convs = len(list(db.execute(select(Conversation.id)).scalars()))
    assert total_convs == 1  # 未新建会话


def test_chat_unknown_conversation_404(client, seeded, monkeypatch):
    s = seeded
    _patch_search(monkeypatch, dense=[], sparse=[])
    resp = client.post(
        "/api/v1/chat",
        json={"conversation_id": 424242, "kb_ids": [s["kb"].id], "query": "问题"},
    )
    assert resp.status_code == 404


# ---------------- 参数校验 ----------------

def test_chat_validation_422(client, seeded):
    kb_id = seeded["kb"].id
    # 缺 kb_ids
    assert client.post("/api/v1/chat", json={"query": "q"}).status_code == 422
    # kb_ids 为空列表
    assert client.post("/api/v1/chat", json={"kb_ids": [], "query": "q"}).status_code == 422
    # kb_ids 超限（>10 个）
    assert (
        client.post(
            "/api/v1/chat", json={"kb_ids": list(range(kb_id, kb_id + 11)), "query": "q"}
        ).status_code
        == 422
    )
    # query 为空 / 缺失
    assert client.post("/api/v1/chat", json={"kb_ids": [kb_id], "query": ""}).status_code == 422
    assert client.post("/api/v1/chat", json={"kb_ids": [kb_id]}).status_code == 422
