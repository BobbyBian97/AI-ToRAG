"""`python -m services.worker`：启动 RQ worker 消费入库队列（ingest）。

用法（先起 Redis，且 Milvus/MinIO/模型网关按 .env 配置好）：
    .venv/Scripts/python.exe -m services.worker

worker 进程内通过任务字符串（infra.queue 的 TASK_PARSE / TASK_PURGE）
import services.worker.tasks 中的同名函数执行，故本包必须可被 import。
"""
import logging

import rq

from infra.queue import get_ingest_queue
from infra.redis import get_redis

logger = logging.getLogger(__name__)


def main() -> None:
    """启动 worker：消费 get_ingest_queue() 对应的 ingest 队列。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    queue = get_ingest_queue()
    worker = rq.Worker([queue], connection=get_redis())
    logger.info("入库 worker 启动，监听队列：%s", queue.name)
    worker.work()


if __name__ == "__main__":
    main()
