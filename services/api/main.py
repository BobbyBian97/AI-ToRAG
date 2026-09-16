"""FastAPI 应用工厂。

路由自动发现：用 pkgutil 遍历 services/api/routers/ 下所有子模块，
凡含模块级 `router`（fastapi.APIRouter 实例）则自动 include。
后续 agent 只需新增路由文件并暴露：
    router = APIRouter(prefix="/api/v1", tags=["xxx"])
无需改动本文件。

启动：uvicorn services.api.main:app
"""
import importlib
import pkgutil

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from infra.config import get_settings


def _discover_routers() -> list[APIRouter]:
    """扫描 services/api/routers/ 包，收集所有模块级 APIRouter。"""
    import services.api.routers as routers_pkg

    found: list[APIRouter] = []
    for mod_info in pkgutil.iter_modules(routers_pkg.__path__):
        module = importlib.import_module(f"{routers_pkg.__name__}.{mod_info.name}")
        router = getattr(module, "router", None)
        if isinstance(router, APIRouter):
            found.append(router)
    return found


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, debug=settings.debug)

    # 自动挂载全部路由模块（约定见模块 docstring）
    for router in _discover_routers():
        app.include_router(router)

    @app.get("/healthz", tags=["meta"], summary="存活探针")
    def healthz() -> dict:
        return {"status": "ok", "app": settings.app_name}

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """兜底异常处理器：统一返回 {"detail": ...}（debug 模式带异常类型便于排查）。"""
        detail = f"{type(exc).__name__}: {exc}" if settings.debug else "internal server error"
        return JSONResponse(status_code=500, content={"detail": detail})

    return app


# 供 `uvicorn services.api.main:app` 使用
app = create_app()
