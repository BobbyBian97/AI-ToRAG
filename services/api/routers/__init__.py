"""路由包：自动发现（由 services.api.main._discover_routers 负责）。

约定：本包下每个路由模块暴露模块级
    router = APIRouter(prefix="/api/v1", tags=["..."])
新增业务路由（documents / knowledge-bases / chat / conversations / debug 等）只需新建
xxx.py 文件，无需注册。无 `router` 属性的模块会被跳过。
"""
