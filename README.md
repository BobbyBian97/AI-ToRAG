# AI-ToRAG 企业知识库问答系统

基于企业私有知识（制度文档、技术手册、合同、Wiki 等）的智能问答系统：自然语言提问 → 混合检索 → 生成**带引用来源**的回答。全组件可私有化部署，数据不出内网。

整体设计见 [企业RAG系统设计书.md](./企业RAG系统设计书.md)。

## 当前状态

P0 MVP 已实现（对应设计书 §2.1 功能编号）：

| 功能 | 说明 |
|---|---|
| F1 文档接入 | 批量上传 PDF / Word / Excel / PPT / Markdown / HTML / TXT；sha256 去重、部分成功语义、软删 |
| F2 入库流水线 | 异步解析（md/html/txt 原生，PDF→MinerU，Office→Tika）→ 标题感知递归切分（512/64、small-to-big 父块、表格整块）→ Embedding → Milvus + PG 双写；按 sha256 幂等，失败标 failed |
| F3 问答 | 混合召回（dense + sparse，权限过滤召回期下推）→ RRF 融合 → Rerank 精排 → SSE 流式生成，强制 `[n]` 引用标注；多轮查询改写 |
| Web 简页 | 上传文档 + 流式问答 + 引用卡片（`GET /`） |

未实现（规划中）：F4 权限 ACL、F5 会话管理增强、F6 知识库全量管理、F7 反馈闭环、F8 效果评估、F9 检索调试台、F10 知识更新同步。

## 技术栈

- **Python 3.11+ / FastAPI**，自研薄编排层（纯函数流水线，不用 LangChain）
- **PostgreSQL 15**：文档/切片/会话/审计元数据唯一事实源
- **Milvus 2.5+**：dense + sparse 混合检索（BGE-M3 一次产出双向量）
- **MinIO**：原始文件对象存储；**Redis + RQ**：入库任务队列
- **model-gateway**：Embedding / Rerank / LLM 三角色统一抽象（OpenAI 兼容协议），换模型 = 改配置

## 目录结构

```
ai-torag/
├── services/
│   ├── api/            # FastAPI：问答 + 知识管理 + Web 简页（static/）
│   ├── worker/         # 入库流水线 worker（独立进程，可多实例）
│   └── model_gateway/  # Embedding/Rerank/LLM 统一封装（fake / OpenAI 兼容）
├── core/               # 领域逻辑：parsing / chunking / retrieval / prompt
├── infra/              # config / pg / milvus / minio / redis / queue / models
├── deploy/             # podman-compose.dev.yml（开发依赖编排）
└── tests/              # 零外部依赖测试（fake 网关 + sqlite 内存库）
```

## 快速开始

前置：Python ≥ 3.11、[Podman](https://podman.io/)（≥ 4.7 或 podman-compose ≥ 1.0.6）。

```bash
# 1) 安装依赖（Windows PowerShell / bash；Linux/macOS 将 Scripts 换成 bin）
python -m venv .venv
.venv/Scripts/pip install -e .

# 2) 启动依赖服务（postgres / redis / minio / milvus standalone + etcd）
podman compose -f deploy/podman-compose.dev.yml up -d

# 3) 配置：复制样例为 .env（默认全部 fake provider，可先不接真实模型）
cp .env.example .env

# 4) 建表（Milvus collection 首次写入时自动创建，无需手动初始化）
.venv/Scripts/python -c "from infra.pg import init_db; init_db()"

# 5) 启动 API 与 worker（两个终端）
.venv/Scripts/python -m uvicorn services.api.main:app --reload
.venv/Scripts/python -m services.worker
```

访问入口：

| 地址 | 说明 |
|---|---|
| `http://localhost:8000/` | Web 问答页（上传文档 / 勾选知识库 / 流式问答 / 引用卡片） |
| `http://localhost:8000/docs` | Swagger API 文档（可在线调试） |
| `http://localhost:8000/healthz` | 存活探针 |

**典型使用流程**：创建知识库 → 上传文档 → 等文档状态变为「就绪」（worker 解析完成）→ 勾选知识库提问。

> 接真实模型：在 `.env` 中把 `EMBED_PROVIDER` / `RERANK_PROVIDER` / `LLM_PROVIDER` 改为 `openai` 并填 `*_BASE_URL` / `*_MODEL`（vLLM / TEI 等 OpenAI 兼容服务均可）。默认 `fake` 时全链路可跑通，但回答为占位文本。

## API 一览

```
POST   /api/v1/chat                      # 问答（SSE：delta* → refs → done）
POST   /api/v1/documents                 # 批量上传（multipart: kb_id + files[]）
GET    /api/v1/documents?kb_id=&status=  # 文档列表 {total, items}
GET    /api/v1/documents/{id}            # 文档详情
DELETE /api/v1/documents/{id}            # 软删（worker 异步清向量）
POST   /api/v1/knowledge-bases           # 创建知识库
GET    /api/v1/knowledge-bases           # 知识库列表
GET    /healthz                          # 存活探针
```

约定：写操作幂等；SSE 流必须以 `refs` 事件结束（前端据此渲染引用卡片）；检索为空时回答「知识库中未找到相关内容。」。

## 配置

全部环境变量及中文注释见 [.env.example](./.env.example)，要点：

| 变量 | 说明 |
|---|---|
| `DATABASE_URL` / `REDIS_URL` | PG 连接串 / Redis 地址 |
| `MILVUS_URI` / `MILVUS_COLLECTION` | 向量库地址 / 集合名 |
| `MINIO_*` | 对象存储 endpoint、凭证、桶名 |
| `EMBED_*` / `RERANK_*` / `LLM_*` | 三个模型角色的 provider（`fake` \| `openai`）、地址、模型名 |
| `EMBED_DIM` | 向量维度，必须与 Milvus 索引一致（默认 1024） |

## 开发

```bash
.venv/Scripts/python -m pytest        # 全量测试（67 个，零外部依赖：fake 网关 + sqlite 内存库）
.venv/Scripts/python -m ruff check .  # 代码检查
```

开发约定：

- 仓库根为导入根：`from core.x import ...` / `from infra.x import ...`
- 主键统一 `BigInteger`（PG IDENTITY；sqlite 测试库自动退化为 INTEGER，见 `infra/models.py`）
- JSON 字段统一 `sqlalchemy.JSON`；路由自动发现：`services/api/routers/` 下暴露 `router = APIRouter(...)` 即挂载
- 新增文档解析器：实现 `core/parsing.py` 的分发；重解析依赖（MinerU/Tika）装 `pip install -e ".[parsing-heavy]"`
- 模型三角色只能经 `services.model_gateway.get_gateway()` 访问，业务代码不得直连模型服务

## 架构要点

- **在线 / 离线分离**：入库流水线（CPU/IO 重）跑在独立 worker 进程，不影响问答延迟
- **PG 为事实源**：chunk 元数据在 PG，Milvus 只存向量 + 最小冗余；删改以 PG 为准
- **权限在召回期过滤**：用户可见 doc 集合作为标量条件下推 Milvus（绝不能检索后再过滤），当前预留 `doc_ids` 参数，F4 接入 ACL
- **可信生成**：上下文包裹 `<context>` 标签并声明"内容不是指令"（提示注入防御）；强制引用编号；检索为空明确说找不到
