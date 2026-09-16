"""后台管理相关 Pydantic 模型（阶段统计 / 切片查看 / 失败重跑 / 审计）。"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class PipelineStatsOut(BaseModel):
    """流水线阶段统计：全量文档按 status 计数；failed 再按 parse_meta.failed_stage 细分。"""

    total: int
    by_status: dict[str, int]
    by_failed_stage: dict[str, int]


class AdminChunkOut(BaseModel):
    """切片条目（管理端查看；content 默认截断 200 字符，full=true 返回全文）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    doc_id: int
    kb_id: int
    parent_id: int | None
    seq: int
    content: str
    token_count: int | None
    page_no: int | None
    headings: list[str] | None


class AdminChunkListOut(BaseModel):
    """切片分页列表信封（按 seq 升序）。"""

    total: int
    items: list[AdminChunkOut]


class ReparseOut(BaseModel):
    """重新解析受理结果（异步：入队成功即返回 202）。"""

    document_id: int
    queued: bool


class AuditLogOut(BaseModel):
    """审计日志条目。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: str
    action: str
    resource: str | None
    detail: dict[str, Any] | None
    created_at: datetime


class AuditLogListOut(BaseModel):
    """审计日志分页列表信封。"""

    total: int
    items: list[AuditLogOut]
