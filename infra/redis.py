"""Redis 连接（懒创建：import 本模块不建连）。"""
from functools import lru_cache

import redis as redis_lib

from infra.config import get_settings


@lru_cache
def get_redis() -> redis_lib.Redis:
    """Redis 客户端单例。redis-py 是惰性连接，首次命令才真正建连。"""
    return redis_lib.Redis.from_url(get_settings().redis_url, decode_responses=True)
