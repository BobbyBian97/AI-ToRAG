"""API Pydantic 模型包：按资源域拆分（knowledge_bases / documents）。

路由模块统一从 `services.api.schemas` 导入，避免直接依赖子模块路径。
"""
from .documents import (
    DocumentListOut,
    DocumentOut,
    FileNotice,
    UploadedFileInfo,
    UploadResult,
)
from .knowledge_bases import KBCreate, KBListOut, KBOut

__all__ = [
    "DocumentListOut",
    "DocumentOut",
    "FileNotice",
    "KBCreate",
    "KBListOut",
    "KBOut",
    "UploadResult",
    "UploadedFileInfo",
]
