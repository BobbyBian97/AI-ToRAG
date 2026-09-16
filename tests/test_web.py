"""Web 简页测试：GET / 返回静态单页，且不进 OpenAPI 文档。"""


def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "AI-ToRAG" in resp.text
    # 页面必须自包含：不依赖任何外链脚本/样式（私有化内网原则）
    assert "http://" not in resp.text.replace("http://localhost", "")
    assert "https://" not in resp.text


def test_index_hidden_from_openapi(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/" not in paths          # include_in_schema=False
    assert "/api/v1/chat" in paths   # API 路由不受影响
