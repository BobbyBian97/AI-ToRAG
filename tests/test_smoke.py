"""冒烟测试：healthz / FakeGateway 三接口行为 / 路由自动发现生效。"""
import math


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["app"] == "ai-torag"


def test_fake_gateway_embed_deterministic_and_normalized(fake_gateway):
    embs = fake_gateway.embed(["知识库权限过滤设计", "文档解析切分流水线"])
    assert len(embs) == 2
    for e in embs:
        assert len(e.dense) == 1024
        # L2 归一化：与 IP 度量配套
        assert math.isclose(sum(x * x for x in e.dense), 1.0, rel_tol=1e-6)
        assert e.sparse  # 稀疏向量非空（bigram 词频）
    # 确定性：同文本同结果
    again = fake_gateway.embed(["知识库权限过滤设计"])
    assert again[0].dense == embs[0].dense
    assert again[0].sparse == embs[0].sparse


def test_fake_gateway_rerank_sorted_by_score(fake_gateway):
    docs = [
        "文档解析与切分流水线设计",
        "知识库权限过滤实现",
        "权限过滤必须在召回阶段执行",
        "MinIO 对象存储配置",
    ]
    hits = fake_gateway.rerank("权限过滤召回", docs, top_n=3)
    assert len(hits) == 3
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert all(0 <= h.index < len(docs) for h in hits)
    # 与 query 最相关的文档（bigram 重合度最高）必须排第一
    assert hits[0].index in (1, 2)


def test_fake_gateway_generate_with_citations(fake_gateway):
    messages = [{"role": "user", "content": "如何做权限过滤？"}]
    text = fake_gateway.generate(messages)
    assert isinstance(text, str)
    assert "[1]" in text and "[2]" in text  # 带引用标记
    assert "如何做权限过滤" in text

    # 流式：返回迭代器，拼接后与整体输出完全一致
    stream = fake_gateway.generate(messages, stream=True)
    joined = "".join(stream)
    assert joined == text

    # model_key="small"（查询改写等轻任务）也可用
    small = fake_gateway.generate(messages, model_key="small")
    assert isinstance(small, str) and "[1]" in small


def test_router_autodiscovery(tmp_path, monkeypatch):
    """在 routers 包里注入一个临时路由模块，验证 create_app 能自动挂载。"""
    from fastapi.testclient import TestClient

    import services.api.routers as routers_pkg
    from services.api.main import create_app

    module_code = (
        "from fastapi import APIRouter\n"
        "router = APIRouter(prefix='/api/v1', tags=['smoke'])\n"
        "\n"
        "@router.get('/smoke-ping')\n"
        "def smoke_ping():\n"
        "    return {'pong': True}\n"
    )
    (tmp_path / "smoke_dummy_router.py").write_text(module_code, encoding="utf-8")
    monkeypatch.setattr(routers_pkg, "__path__", [str(tmp_path)])

    app = create_app()
    with TestClient(app) as c:
        resp = c.get("/api/v1/smoke-ping")
    assert resp.status_code == 200
    assert resp.json() == {"pong": True}
