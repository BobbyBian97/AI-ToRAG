"""MinIO 对象存储封装（原始文件）。懒连接：import 本模块不建连。"""
import io
from functools import lru_cache

from minio import Minio

from infra.config import get_settings


@lru_cache
def get_minio() -> Minio:
    """MinIO 客户端单例（构造时不发网络请求）。"""
    s = get_settings()
    return Minio(
        endpoint=s.minio_endpoint,
        access_key=s.minio_access_key,
        secret_key=s.minio_secret_key,
        secure=s.minio_secure,
    )


def object_key(kb_id: int, document_id: int, filename: str) -> str:
    """对象 key 全局约定：kb/{kb_id}/{document_id}/{filename}（F1/F2/F3 共用）。"""
    return f"kb/{kb_id}/{document_id}/{filename}"


def ensure_bucket() -> None:
    """确保配置的桶存在（幂等）。worker 启动 / 上传前调用一次即可。"""
    s = get_settings()
    client = get_minio()
    if not client.bucket_exists(s.minio_bucket):
        client.make_bucket(s.minio_bucket)


def put_object(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    """上传字节流到配置桶。调用方需先 ensure_bucket()。"""
    s = get_settings()
    get_minio().put_object(
        bucket_name=s.minio_bucket,
        object_name=key,
        data=io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )


def get_object(key: str) -> bytes:
    """下载对象为字节串。"""
    s = get_settings()
    resp = get_minio().get_object(s.minio_bucket, key)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()
