"""文档相关 Pydantic 模型（F1 上传 / 查询 / 软删）。"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    """文档实体输出（含上传冗余的原始文件信息）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    kb_id: int
    source_type: str
    source_uri: str | None
    sha256: str | None
    title: str | None
    status: str  # parsing | ready | failed | deleted
    filename: str | None
    content_type: str | None
    size_bytes: int | None
    minio_object_key: str | None
    parse_meta: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime


class DocumentListOut(BaseModel):
    """文档分页列表信封。"""

    total: int
    items: list[DocumentOut]


class UploadedFileInfo(BaseModel):
    """上传成功条目：新文档 id + 文件名。"""

    id: int
    filename: str


class FileNotice(BaseModel):
    """上传未成功条目：文件名 + 机器可读原因。"""

    filename: str
    reason: str  # unsupported_format | duplicate | storage_error | enqueue_error


class UploadResult(BaseModel):
    """批量上传响应（部分成功语义，HTTP 恒为 200）。

    - uploaded: 已落库（status=parsing）并成功写入 MinIO / 入队的文档
    - skipped:  同 kb 内 sha256 重复而被跳过的文件
    - rejected: 扩展名不支持，或存储/入队失败（对应 document 行标 failed）
    """

    uploaded: list[UploadedFileInfo] = Field(default_factory=list)
    skipped: list[FileNotice] = Field(default_factory=list)
    rejected: list[FileNotice] = Field(default_factory=list)
