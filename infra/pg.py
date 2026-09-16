"""PostgreSQL / SQLAlchemy 引擎与会话。

懒连接：create_engine 是惰性的，import 本模块不会建立任何数据库连接。
"""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from infra.config import get_settings

settings = get_settings()

# pool_pre_ping：取连接前先探活，避免长连接被数据库掐断后报"连接已关闭"
engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    future=True,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：每请求一个会话，请求结束自动归还连接。

    用法：db: Session = Depends(get_db)（路由侧建议从 services.api.deps 导入）。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """按 ORM 元数据建全部表（create_all）。仅开发/测试使用，生产环境走迁移脚本。"""
    from infra.models import Base

    Base.metadata.create_all(bind=engine)
