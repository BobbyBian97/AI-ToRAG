"""知识库相关 Pydantic 模型（F1 最小 CRUD 用；完整管理属 F6）。"""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class KBCreate(BaseModel):
    """创建知识库入参。"""

    name: str = Field(min_length=1, max_length=255, description="知识库名称")
    dept_id: str | None = Field(default=None, max_length=64, description="归属部门（可选）")


class KBOut(BaseModel):
    """知识库实体输出。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    dept_id: str | None
    embedding_model: str | None
    created_at: datetime


class KBListOut(BaseModel):
    """知识库分页列表：与 documents 列表统一使用 {total, items} 信封。"""

    total: int
    items: list[KBOut]
