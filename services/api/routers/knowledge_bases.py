"""知识库路由（F1 最小 CRUD：仅创建 / 分页列表 / 详情）。

范围说明：F1 只需要"能建库、能查库"以支撑文档上传；知识库的更新 / 删除 /
部门管理等完整能力属 F6，本模块刻意不做 update / delete。
"""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from infra.models import AuditLog, KnowledgeBase
from services.api.deps import UserCtx, get_current_user, get_db
from services.api.schemas.knowledge_bases import KBCreate, KBListOut, KBOut

router = APIRouter(prefix="/api/v1", tags=["knowledge-bases"])

DB = Annotated[Session, Depends(get_db)]
User = Annotated[UserCtx, Depends(get_current_user)]


@router.post("/knowledge-bases", status_code=201, response_model=KBOut)
def create_knowledge_base(payload: KBCreate, db: DB, user: User) -> KnowledgeBase:
    """创建知识库；写操作顺手记审计日志。"""
    kb = KnowledgeBase(name=payload.name, dept_id=payload.dept_id)
    db.add(kb)
    db.flush()  # 拿自增 id，供审计 resource 使用

    db.add(
        AuditLog(
            user_id=user.user_id,
            action="knowledge_base.create",
            resource=f"knowledge_base:{kb.id}",
            detail={"name": payload.name, "dept_id": payload.dept_id},
        )
    )
    db.commit()
    db.refresh(kb)  # created_at 为 server 生成，需回读
    return kb


@router.get("/knowledge-bases", response_model=KBListOut)
def list_knowledge_bases(
    db: DB,
    user: User,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> KBListOut:
    """知识库分页列表（按创建顺序倒序，新库在前）。"""
    query = db.query(KnowledgeBase)
    total = query.count()
    items = (
        query.order_by(KnowledgeBase.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return KBListOut(total=total, items=items)  # type: ignore[arg-type]


@router.get("/knowledge-bases/{kb_id}", response_model=KBOut)
def get_knowledge_base(kb_id: int, db: DB, user: User) -> KnowledgeBase:
    """知识库详情；不存在返回 404。"""
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail=f"knowledge_base {kb_id} not found")
    return kb
