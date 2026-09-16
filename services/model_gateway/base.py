"""模型网关抽象（设计书 §6：全系统唯一必须坚持的抽象）。

三个模型角色各一个方法，业务代码只依赖 BaseGateway ——
换模型 = 改 .env 配置，不改业务代码。
"""
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class Embedding:
    """一次向量化结果：dense 稠密向量 + sparse 稀疏词权重（键为维度 id，值非负）。"""

    dense: list[float]
    sparse: dict[int, float]


@dataclass
class RerankHit:
    """rerank 单条结果：index 指向输入 documents 的下标，列表按 score 降序。"""

    index: int
    score: float


class BaseGateway(ABC):
    """Embedding / Rerank / LLM 统一网关接口。

    实现要求：构造函数不得发起网络请求（懒创建客户端）；同一实例应可复用（线程安全由实现保证）。
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[Embedding]:
        """批量向量化，返回顺序与 texts 一一对应。dense 已 L2 归一化（与 IP 度量配套）。"""

    @abstractmethod
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankHit]:
        """对 documents 按与 query 的相关度打分，返回 top_n 条（score 降序）。"""

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        *,
        stream: bool = False,
        model_key: str = "main",
    ) -> str | Iterator[str]:
        """对话生成。

        messages：OpenAI 格式 [{"role": "system|user|assistant", "content": str}, ...]
        model_key："main"（回答生成）| "small"（多轮改写等轻任务）
        stream=False 返回完整文本 str；stream=True 返回增量文本迭代器（逐段 str）。
        """
