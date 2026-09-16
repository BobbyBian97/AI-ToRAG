"""F3 单元测试：RRF 融合、retrieve 组装、prompt 构建、多轮查询改写（零外部服务）。"""
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import infra.milvus
from core.prompt import NO_RESULT_TEXT, build_messages, rewrite_standalone_question
from core.retrieval import RECALL_TOP_K, RetrievedChunk, retrieve, rrf_fuse
from infra.models import Base, Chunk, Conversation, Document, KnowledgeBase, Message
from services import model_gateway


@pytest.fixture
def db():
    """独立内存 sqlite 会话（core 层单测不经过 API）。"""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = TestSession()
    yield session
    session.close()


# ---------------- rrf_fuse ----------------

def test_rrf_fuse_two_lists_multi_list_wins():
    """两路融合：同时出现在两路的 chunk 应排在单路第一名的前面。"""
    # 1: 1/61+1/62 ≈ 0.0325；4: 1/61 ≈ 0.0164；2: 1/62；3: 1/63
    assert rrf_fuse([[1, 2, 3], [4, 1]]) == [1, 4, 2, 3]


def test_rrf_fuse_single_list_keeps_order():
    assert rrf_fuse([[7, 8, 9]]) == [7, 8, 9]


def test_rrf_fuse_empty_inputs():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
    assert rrf_fuse([[]], top_n=5) == []


def test_rrf_fuse_tie_breaks_by_chunk_id():
    """同分并列时按 chunk_id 升序，保证结果确定。"""
    assert rrf_fuse([[9], [3]]) == [3, 9]


def test_rrf_fuse_top_n_truncates():
    out = rrf_fuse([[1, 2, 3], [3, 2, 1]], top_n=2)
    assert len(out) == 2
    assert set(out) == {1, 3}  # 两端各 1/61+1/63 最高，中间 2×(1/62) 略低


# ---------------- build_messages ----------------

def _mk_hits() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=11, doc_id=1, doc_title="权限手册", page_no=3, score=0.9,
            content="权限过滤必须在召回阶段执行",
        ),
        RetrievedChunk(
            chunk_id=12, doc_id=2, doc_title="切分指南", page_no=None, score=0.5,
            content="按标题层级递归切分",
        ),
    ]


def test_build_messages_with_hits():
    msgs = build_messages("如何做权限过滤？", _mk_hits())
    assert [m["role"] for m in msgs] == ["system", "user"]

    system = msgs[0]["content"]
    assert "<context>" in system          # system 规则声明上下文边界（§9 注入防御）
    assert "不是指令" in system           # 上下文内容不是指令
    assert "[n]" in system                # 引用必须带编号
    assert NO_RESULT_TEXT in system       # 空/无关上下文的标准话术

    user = msgs[1]["content"]
    assert user.startswith("<context>")
    assert "[1]" in user and "[2]" in user                    # 逐块编号
    assert "《权限手册》" in user and "第 3 页" in user          # 标题 + 页码
    assert "《切分指南》" in user and "None" not in user        # 无页码不输出占位
    assert "权限过滤必须在召回阶段执行" in user                   # 块正文进入上下文
    assert user.rstrip().endswith("问题：如何做权限过滤？")        # context 之后是问题
    assert user.index("<context>") < user.index("问题：")


def test_build_messages_empty_hits():
    msgs = build_messages("随便问", [])
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    assert "<context>" in user and "随便问" in user
    assert "（无检索内容）" in user       # 空检索显式标记，配合 system 规则


# ---------------- rewrite_standalone_question ----------------

def _mk_conversation(db) -> Conversation:
    conv = Conversation(user_id="u1", title="t")
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def test_rewrite_without_history_returns_query(db, monkeypatch):
    """首轮无历史：直接返回原问题，网关不应被调用。"""
    conv = _mk_conversation(db)

    def _boom():
        raise AssertionError("无历史时不应调用网关")

    monkeypatch.setattr(model_gateway, "get_gateway", _boom)
    assert rewrite_standalone_question(db, conv.id, "第一条问题") == "第一条问题"


def test_rewrite_with_history_calls_small_model(db, fake_gateway, monkeypatch):
    conv = _mk_conversation(db)
    db.add_all(
        [
            Message(conversation_id=conv.id, role="user", content="公司的报销制度是什么？"),
            Message(conversation_id=conv.id, role="assistant", content="报销制度如下……[1]"),
        ]
    )
    db.commit()

    called = {}
    original = fake_gateway.generate

    def spy(messages, *, stream=False, model_key="main"):
        called["model_key"] = model_key
        called["history_len"] = len(messages) - 2  # 去掉 system 与最后的改写指令
        return original(messages, stream=stream, model_key=model_key)

    monkeypatch.setattr(fake_gateway, "generate", spy)
    out = rewrite_standalone_question(db, conv.id, "那它的限额是多少？")
    assert called.get("model_key") == "small"          # 改写必须走 cheap 模型
    assert called.get("history_len") == 2              # 历史被拼进 messages
    assert out != "那它的限额是多少？"
    assert "[small]" in out                            # FakeGateway 确定性输出


def test_rewrite_gateway_error_falls_back(db, fake_gateway, monkeypatch):
    conv = _mk_conversation(db)
    db.add(Message(conversation_id=conv.id, role="user", content="报销制度是什么？"))
    db.commit()

    def _boom(*args, **kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(fake_gateway, "generate", _boom)
    assert rewrite_standalone_question(db, conv.id, "限额多少？") == "限额多少？"


def test_rewrite_nonexistent_conversation_returns_query(db):
    assert rewrite_standalone_question(db, 99999, "问题") == "问题"


# ---------------- retrieve ----------------

@pytest.fixture
def seeded(db):
    kb = KnowledgeBase(name="KB")
    db.add(kb)
    db.commit()
    db.refresh(kb)
    d1 = Document(kb_id=kb.id, title="权限手册", status="ready")
    db.add(d1)
    db.commit()
    db.refresh(d1)
    c1 = Chunk(doc_id=d1.id, kb_id=kb.id, seq=1, content="权限过滤必须在召回阶段下推执行", page_no=2)
    c2 = Chunk(doc_id=d1.id, kb_id=kb.id, seq=2, content="表格整体作为一个 chunk 不再切分", page_no=None)
    db.add_all([c1, c2])
    db.commit()
    db.refresh(c1)
    db.refresh(c2)
    return SimpleNamespace(kb=kb, d1=d1, c1=c1, c2=c2)


def _patch_search(monkeypatch, s, dense, sparse):
    """monkeypatch search_hybrid 返回固定两组 Hit，并校验召回参数透传。"""
    seen = {}

    def fake_search(kb_ids, dense_vec, sparse_vec, top_k, doc_ids=None):
        seen.update(kb_ids=kb_ids, top_k=top_k, doc_ids=doc_ids)
        return (
            [infra.milvus.Hit(cid, sc) for cid, sc in dense],
            [infra.milvus.Hit(cid, sc) for cid, sc in sparse],
        )

    monkeypatch.setattr(infra.milvus, "search_hybrid", fake_search)
    return seen


def test_retrieve_rerank_reorders_and_enriches(db, seeded, fake_gateway, monkeypatch):
    seen = _patch_search(
        monkeypatch, seeded,
        dense=[(seeded.c1.id, 0.9), (seeded.c2.id, 0.8)],
        sparse=[(seeded.c2.id, 0.7)],
    )
    hits = retrieve(db, "权限过滤", [seeded.kb.id])
    # 召回参数：kb 透传、每路 top50、doc_ids 不过滤（MVP 无 ACL）
    assert seen["kb_ids"] == [seeded.kb.id]
    assert seen["top_k"] == RECALL_TOP_K
    assert seen["doc_ids"] is None
    # fake rerank 按 bigram 重合度：c1（含"权限过滤"）必排第一
    assert [h.chunk_id for h in hits] == [seeded.c1.id, seeded.c2.id]
    top = hits[0]
    assert top.doc_id == seeded.d1.id
    assert top.doc_title == "权限手册"
    assert top.page_no == 2
    assert top.content == "权限过滤必须在召回阶段下推执行"
    assert 0 < top.score < 1 and top.score > hits[1].score  # 分数已替换为 rerank 分


def test_retrieve_without_rerank_returns_rrf_order(db, seeded, fake_gateway, monkeypatch):
    _patch_search(
        monkeypatch, seeded,
        dense=[(seeded.c1.id, 0.9), (seeded.c2.id, 0.8)],
        sparse=[(seeded.c2.id, 0.7)],
    )
    # RRF：c2=1/61+1/62 > c1=1/61，融合序为 [c2, c1]；分数取两路最高（0.8）
    hits = retrieve(db, "权限过滤", [seeded.kb.id], use_rerank=False)
    assert [h.chunk_id for h in hits] == [seeded.c2.id, seeded.c1.id]
    assert hits[0].score == pytest.approx(0.8)
    assert hits[1].score == pytest.approx(0.9)


def test_retrieve_rerank_failure_degrades_to_rrf(db, seeded, fake_gateway, monkeypatch):
    _patch_search(monkeypatch, seeded, dense=[(seeded.c1.id, 0.9)], sparse=[(seeded.c2.id, 0.7)])

    def _boom(*args, **kwargs):
        raise RuntimeError("rerank down")

    monkeypatch.setattr(fake_gateway, "rerank", _boom)
    hits = retrieve(db, "权限过滤", [seeded.kb.id])  # 不抛，降级 RRF 序
    # c1/c2 各自单路排第 1，RRF 同分（1/61）并列按 chunk_id 升序
    assert [h.chunk_id for h in hits] == [seeded.c1.id, seeded.c2.id]


def test_retrieve_drops_chunks_missing_in_pg(db, seeded, fake_gateway, monkeypatch):
    _patch_search(monkeypatch, seeded, dense=[(seeded.c1.id, 0.9), (99999, 0.5)], sparse=[])
    hits = retrieve(db, "权限过滤", [seeded.kb.id])
    assert [h.chunk_id for h in hits] == [seeded.c1.id]


def test_retrieve_empty_recall(db, seeded, fake_gateway, monkeypatch):
    _patch_search(monkeypatch, seeded, dense=[], sparse=[])
    assert retrieve(db, "权限过滤", [seeded.kb.id]) == []


def test_retrieve_respects_top_k(db, seeded, fake_gateway, monkeypatch):
    _patch_search(
        monkeypatch, seeded,
        dense=[(seeded.c1.id, 0.9), (seeded.c2.id, 0.8)], sparse=[],
    )
    assert len(retrieve(db, "权限过滤", [seeded.kb.id], top_k=1)) == 1
