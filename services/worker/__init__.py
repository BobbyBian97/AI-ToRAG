"""入库流水线 worker（独立进程，可多实例）。

注意：tasks.parse_document / tasks.purge_document 由 F2 agent 在本包下实现，
任务字符串契约见 infra.queue（TASK_PARSE / TASK_PURGE）。
"""
