"""测试脚手架：

- fake_gateway：替换 services.model_gateway.get_gateway 为全新 FakeGateway（并清 lru_cache）
- client：FastAPI TestClient，依赖 fake 网关；get_db 覆写为内存 sqlite（StaticPool）+ create_all
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import services.model_gateway as model_gateway_module
from infra.models import Base
from infra.pg import get_db
from services.api.main import create_app
from services.model_gateway import FakeGateway


@pytest.fixture
def fake_gateway(monkeypatch):
    """提供确定性的 FakeGateway，并让 get_gateway() 返回它。"""
    gw = FakeGateway()
    original = model_gateway_module.get_gateway
    model_gateway_module.get_gateway.cache_clear()
    monkeypatch.setattr(model_gateway_module, "get_gateway", lambda: gw)
    yield gw
    original.cache_clear()  # 恢复真实 get_gateway 后清掉测试期间可能产生的缓存


@pytest.fixture
def client(fake_gateway) -> TestClient:
    """FastAPI TestClient（含路由自动发现；数据库为内存 sqlite）。"""
    app = create_app()

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # 内存库需单连接共享，否则每个连接各建一个空库
    )
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
