"""API 公共依赖：数据库会话 + 当前用户上下文 + 管理员门禁。"""
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException

from infra.pg import get_db  # re-export，路由统一从本模块导入

__all__ = ["UserCtx", "get_current_user", "get_db", "require_admin"]


@dataclass
class UserCtx:
    """请求用户上下文（MVP 阶段来自 Header，无密码学保证）。"""

    user_id: str
    is_admin: bool


def get_current_user(
    x_user_id: str = Header(default="dev-user", alias="X-User-Id"),
    x_is_admin: bool = Header(default=False, alias="X-Is-Admin"),
) -> UserCtx:
    """当前用户依赖。

    MVP 无真实鉴权：直接信任 X-User-Id 头（缺省 "dev-user"），管理员标记读 X-Is-Admin。
    TODO(F4): 接入企业 SSO（OIDC）/网关统一鉴权后，此处改为解析网关注入的用户身份。
    """
    return UserCtx(user_id=x_user_id or "dev-user", is_admin=bool(x_is_admin))


def require_admin(user: UserCtx = Depends(get_current_user)) -> UserCtx:
    """管理端点门禁：非管理员一律 403（后台管理路由统一使用）。"""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="admin only")
    return user
