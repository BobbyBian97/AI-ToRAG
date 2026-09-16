"""Web 简页托管（设计书 §11 P0：带引用问答 Web 简页）。

静态单页 services/api/static/index.html，由本路由在 GET / 返回；
页面与 API 同源，无 CORS 问题。借助 main.py 的路由自动发现，新增本文件即可挂载。
"""
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["web"])

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@router.get("/", include_in_schema=False, summary="Web 简页")
def index() -> FileResponse:
    """返回问答单页（上传文档 / 多轮问答 / 引用卡片）。"""
    return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")
