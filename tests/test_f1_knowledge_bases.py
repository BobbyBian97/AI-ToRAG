"""F1 知识库最小 CRUD 测试（创建 / 分页列表 / 详情 404）。"""

KB_PATH = "/api/v1/knowledge-bases"


def _create_kb(client, name="测试知识库", dept_id=None):
    payload = {"name": name}
    if dept_id is not None:
        payload["dept_id"] = dept_id
    return client.post(KB_PATH, json=payload)


def test_create_kb_returns_201(client):
    resp = _create_kb(client, name="研发知识库", dept_id="dept-rd")
    assert resp.status_code == 201
    body = resp.json()
    assert isinstance(body["id"], int)
    assert body["name"] == "研发知识库"
    assert body["dept_id"] == "dept-rd"
    assert body["embedding_model"] is None
    assert body["created_at"]  # server 生成时间已回读


def test_create_kb_without_name_422(client):
    assert client.post(KB_PATH, json={}).status_code == 422


def test_list_kbs_pagination(client):
    for i in range(3):
        assert _create_kb(client, name=f"kb-{i}").status_code == 201

    page1 = client.get(KB_PATH, params={"page": 1, "page_size": 2})
    assert page1.status_code == 200
    body1 = page1.json()
    assert body1["total"] == 3
    assert len(body1["items"]) == 2

    page2 = client.get(KB_PATH, params={"page": 2, "page_size": 2})
    body2 = page2.json()
    assert body2["total"] == 3
    assert len(body2["items"]) == 1
    # 倒序：第 1 页是新建的 kb-2/kb-1
    assert [item["name"] for item in body1["items"]] == ["kb-2", "kb-1"]
    assert body2["items"][0]["name"] == "kb-0"


def test_get_kb_by_id_and_404(client):
    kb_id = _create_kb(client, name="详情库").json()["id"]

    ok = client.get(f"{KB_PATH}/{kb_id}")
    assert ok.status_code == 200
    assert ok.json()["id"] == kb_id
    assert ok.json()["name"] == "详情库"

    missing = client.get(f"{KB_PATH}/99999")
    assert missing.status_code == 404
