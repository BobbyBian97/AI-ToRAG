"""全局配置：pydantic-settings，从环境变量 / .env 文件加载（字段名大小写不敏感）。

关键约定：所有 provider 默认 "fake"（确定性假实现），保证零外部依赖即可跑通全部测试。
换真实模型 = 改 .env 里对应 provider 为 "openai" 并填 base_url / api_key。
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---------- 应用 ----------
    app_name: str = "ai-torag"
    debug: bool = False

    # ---------- PostgreSQL（元数据唯一事实源） ----------
    database_url: str = "postgresql+psycopg://torag:torag@localhost:5432/torag"

    # ---------- Redis（入库队列 / 缓存） ----------
    redis_url: str = "redis://localhost:6379/0"

    # ---------- Milvus（向量库） ----------
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "torag_chunk"

    # ---------- MinIO（原始文件存储） ----------
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "torag-raw"
    minio_secure: bool = False

    # ---------- Embedding（BGE-M3：dense + sparse） ----------
    embed_provider: str = "fake"  # fake | openai
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = "bge-m3"
    embed_dim: int = 1024

    # ---------- Rerank ----------
    rerank_provider: str = "fake"  # fake | openai
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = "bge-reranker-v2-m3"

    # ---------- LLM（main=回答生成，small=查询改写等轻任务） ----------
    llm_provider: str = "fake"  # fake | openai
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "qwen2.5-7b-instruct"
    llm_small_model: str = "qwen2.5-1.5b-instruct"


@lru_cache
def get_settings() -> Settings:
    """配置单例（lru_cache）。测试若改了环境变量，先 get_settings.cache_clear()。"""
    return Settings()
