"""ORM 模型：严格对应设计书 §7 的 7 张核心表。

统一约定：
- 主键一律 BigInteger identity 自增；SQLite 单测库通过 with_variant 退化为
  INTEGER PRIMARY KEY（rowid 别名，可自增）；PG 生产库仍为 BIGINT IDENTITY；
- JSON 字段一律用 sqlalchemy.JSON（不用 PG 方言 JSONB），保证 sqlite 测试兼容；
- created_at / updated_at 由数据库生成（server_default=func.now()），
  注意 insert 后对象上不会自动带出该值，需要时请 db.refresh(obj)。
"""
from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Identity,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# 主键专用：PG 下 BIGINT（IDENTITY），SQLite 测试库下 INTEGER（rowid 别名自增）。
# 注意：SQLite 不把 BIGINT PRIMARY KEY 当 rowid 别名，直接用会导致 insert 报 NOT NULL。
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")

# 入库流水线阶段（worker 与管理 API 共用的单一事实源；worker 不能 import services.api）。
# 语义：parsing 兼作"已上传"（上传即写入）；ready 为终态（Milvus+PG 双写完成）。
PIPELINE_STAGES: tuple[str, ...] = ("parsing", "parsed", "chunked", "embedded", "ready")
# 失败时 parse_meta.failed_stage 的取值（与阶段函数一一对应，index 对应 Milvus 双写）
FAILED_STAGES: tuple[str, ...] = ("parse", "chunk", "embed", "index")


class Base(DeclarativeBase):
    """全项目唯一 declarative base（infra.pg.init_db 与测试均用它 create_all）。"""


class KnowledgeBase(Base):
    """知识库：业务域隔离单位（F6 管理路由使用）。"""

    __tablename__ = "knowledge_base"

    id: Mapped[int] = mapped_column(BigIntPK, Identity(), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    dept_id: Mapped[str | None] = mapped_column(String(64), index=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Document(Base):
    """文档：上传 / 同步的元数据（F1/F2 使用）。

    status 生命周期（PIPELINE_STAGES，每阶段完成即落库，供后台管理观察进度）：
        parsing(已上传,解析中) → parsed(结构化完成) → chunked(切片落PG)
        → embedded(向量化完成) → ready(Milvus+PG 双写完成)
    终态：failed（parse_meta.failed_stage 记录失败阶段）| deleted（软删，异步清向量库）。
    parse_meta 契约（各阶段合并写入，不得整体覆盖）：
        {"stages": {"parse": {...}, "chunk": {...}, "embed": {...}, "index": {...}},
         "failed_stage": "embed", "error": "...", "chunk_count": n, "parent_count": n}
    """

    __tablename__ = "document"

    id: Mapped[int] = mapped_column(BigIntPK, Identity(), primary_key=True)
    kb_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), default="upload", nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)  # 内容指纹，幂等去重
    title: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(16), default="parsing", index=True, nullable=False)
    parse_meta: Mapped[dict | None] = mapped_column(JSON)  # 解析器/页数/耗时等中间信息
    # ---- 原始文件信息（上传场景冗余存储，便于列表展示与从 MinIO 回取） ----
    filename: Mapped[str | None] = mapped_column(String(512))
    content_type: Mapped[str | None] = mapped_column(String(128))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    minio_object_key: Mapped[str | None] = mapped_column(String(512))  # kb/{kb_id}/{doc_id}/{filename}
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class Chunk(Base):
    """切片：元数据事实源（向量本体在 Milvus；删改以 PG 为准）。"""

    __tablename__ = "chunk"
    __table_args__ = (UniqueConstraint("doc_id", "seq", name="uq_chunk_doc_seq"),)

    id: Mapped[int] = mapped_column(BigIntPK, Identity(), primary_key=True)
    doc_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    kb_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    parent_id: Mapped[int | None] = mapped_column(BigInteger, index=True)  # 父块（small-to-big 检索用）
    seq: Mapped[int] = mapped_column(nullable=False)  # 文档内序号
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column()
    page_no: Mapped[int | None] = mapped_column()  # 溯源用：原文页码
    headings: Mapped[list | None] = mapped_column(JSON)  # 标题层级路径，如 ["三章", "3.2 权限"]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class DocAcl(Base):
    """文档级 ACL：谁能看这篇文档（F4 检索过滤的依据）。

    principal_type: user | group | dept | public
    """

    __tablename__ = "doc_acl"
    __table_args__ = (
        UniqueConstraint("doc_id", "principal_type", "principal_id", name="uq_doc_acl"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, Identity(), primary_key=True)
    doc_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    principal_type: Mapped[str] = mapped_column(String(16), nullable=False)
    principal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    allow: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class Conversation(Base):
    """会话（F5 多轮对话）。"""

    __tablename__ = "conversation"

    id: Mapped[int] = mapped_column(BigIntPK, Identity(), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    title: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Message(Base):
    """消息：问答对 + 引用与反馈（F3/F7 使用）。"""

    __tablename__ = "message"

    id: Mapped[int] = mapped_column(BigIntPK, Identity(), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    refs: Mapped[list | None] = mapped_column(JSON)  # [{chunk_id, doc_title, page_no, score}, ...]
    feedback: Mapped[int] = mapped_column(SmallInteger, default=0, nullable=False)  # 1 好 / -1 差 / 0 无
    latency_ms: Mapped[int | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class AuditLog(Base):
    """审计日志：保留 >= 180 天（合规要求）。"""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigIntPK, Identity(), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)  # 如 document.upload / chat.query
    resource: Mapped[str | None] = mapped_column(String(255))  # 如 document:123
    detail: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
