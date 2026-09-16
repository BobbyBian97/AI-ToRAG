"""基础设施层：config / pg / models / minio / redis / queue / milvus。

约定：所有模块懒连接 —— import 本包任何子模块不会建立任何外部服务连接，
测试无需真实 PostgreSQL / Redis / Milvus / MinIO。
"""
