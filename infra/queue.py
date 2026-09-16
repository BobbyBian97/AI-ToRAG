"""入库任务队列（RQ over Redis）。

共享契约（F1 上传 / F2 流水线 / F3 问答的删除清理均按此入队）：
- 队列名：INGEST_QUEUE = "ingest"
- 任务字符串（enqueue 的第一个参数，worker 端按此路径 import 函数）：
    TASK_PARSE = "services.worker.tasks.parse_document"   # 解析入库主流程
    TASK_PURGE = "services.worker.tasks.purge_document"   # 删除文档后清理向量库
  即 worker 侧需提供 services/worker/tasks.py 中的同名函数 parse_document / purge_document。

用法：
    from infra.queue import get_ingest_queue, TASK_PARSE
    get_ingest_queue().enqueue(TASK_PARSE, args=(document_id,), job_timeout="30m")
"""
import rq

from infra.redis import get_redis

INGEST_QUEUE = "ingest"
TASK_PARSE = "services.worker.tasks.parse_document"
TASK_PURGE = "services.worker.tasks.purge_document"


def get_ingest_queue() -> rq.Queue:
    """返回入库队列（依赖 get_redis，懒连接）。"""
    return rq.Queue(name=INGEST_QUEUE, connection=get_redis())
