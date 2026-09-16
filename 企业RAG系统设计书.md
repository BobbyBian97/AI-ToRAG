# 企业级 RAG 系统设计书

| 项 | 内容 |
|---|---|
| 版本 | v0.1（0-1 初稿） |
| 日期 | 2026-09-11 |
| 状态 | 待评审 |
| 范围 | 从零搭建企业知识库问答（RAG）系统的整体设计 |

---

## 1. 背景与目标

企业内部沉淀了大量非结构化知识（制度文档、技术手册、合同、工单、Wiki），分散在多个系统中，员工检索成本高，新人上手慢。

**目标**：构建一个企业私有化部署的智能问答系统，员工用自然语言提问，系统基于企业自有知识给出**带引用来源**的回答。

**核心设计原则**（按优先级）：

1. **答案可信**：回答必须可溯源（引用原文出处），权限必须严格执行（谁能问什么）。
2. **简单可维护**：0-1 阶段用最少的组件跑通全链路，组件全部可替换。
3. **中文优先**：解析、切分、Embedding 全链路针对中文优化。
4. **可评估**：效果必须有量化指标，避免"感觉变好了"。

**非目标**（本期不做）：

- 多模态（图片理解、语音）
- Agent 式多跳自主规划
- 对外（C 端）服务，仅面向企业内部

---

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 优先级 |
|---|---|---|
| F1 | 文档接入：批量上传 PDF/Word/Excel/PPT/Markdown/HTML/纯文本 | P0 |
| F2 | 文档解析切分、向量化、入库（异步流水线） | P0 |
| F3 | 问答：自然语言提问 → 检索 → 生成带引用的回答 | P0 |
| F4 | 权限：文档级访问控制（ACL），检索结果按用户权限过滤 | P0 |
| F5 | 会话管理：多轮对话、历史记录 | P0 |
| F6 | 知识库管理：按部门/业务域划分知识库，增删改查 | P0 |
| F7 | 反馈闭环：用户点赞/点踩 + 标注正确答案 | P1 |
| F8 | 效果评估：黄金问答集 + 自动化评估报告 | P1 |
| F9 | 检索调试台：开发/运营人员查看命中了哪些 chunk、分数 | P1 |
| F10 | 知识更新：文档变更自动重新入库（Webhook/定时同步） | P2 |

### 2.2 非功能需求

| 维度 | 指标 |
|---|---|
| 检索延迟 | 端到端首 token < 3s（P95），检索阶段 < 800ms |
| 并发 | 100 QPS 检索，20 QPS 生成（按 5000 员工规模估算） |
| 数据规模 | 一期 10 万文档 / 500 万 chunk / 200GB 原始文件 |
| 可用性 | 99.5%，检索与生成服务无状态可水平扩展 |
| 安全 | 全部私有化部署，数据不出内网；LLM 可选自托管或私有化网关 |
| 合规 | 操作审计日志保留 ≥ 180 天 |

---

## 3. 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      接入层（Web / IM 机器人）                  │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTPS
┌──────────────────────────▼──────────────────────────────────┐
│                    API 网关（鉴权 / 限流 / 审计）               │
└───────────────┬─────────────────────────────┬───────────────┘
                │                             │
┌───────────────▼───────────────┐ ┌───────────▼───────────────┐
│      问答服务（在线）            │ │   知识管理服务（离线）        │
│  · 查询改写（多轮→独立问题）       │ │  · 文档上传/同步            │
│  · 混合检索 + 权限过滤           │ │  · 解析 / 切分 / 向量化      │
│  · Rerank                     │ │  · 任务队列（重试/幂等）      │
│  · LLM 生成 + 引用标注          │ │                           │
└───────┬──────────┬────────────┘ └───────────┬───────────────┘
        │          │                          │
┌───────▼───┐ ┌────▼─────┐          ┌─────────▼─────────┐
│ 向量数据库  │ │ 模型服务   │          │  对象存储 (MinIO)   │
│ (Milvus)  │ │ Embedding │          │  原始文件           │
└───────────┘ │ Rerank    │          └─────────┬─────────┘
              │ LLM       │                    │
              └──────────┘          ┌───────────▼───────────┐
                                    │ PostgreSQL              │
                                    │ 文档/切片/ACL/会话/审计元数据│
                                    └───────────────────────┘

横切：Langfuse（ tracing） · Prometheus/Grafana（监控） · OIDC/LDAP（企业 SSO）
```

**架构要点**：

- **在线/离线分离**：入库流水线是重 CPU/IO 任务，与在线问答物理隔离，互不影响延迟。
- **模型层全部走 OpenAI 兼容协议**：Embedding/Rerank/LLM 三个角色统一网关封装，换模型不改业务代码。
- **PostgreSQL 作为元数据唯一事实源**，Milvus 只存向量与最小冗余字段（chunk_id），删改以 PG 为准。

---

## 4. 核心流程设计

### 4.1 离线入库流水线

```
上传/同步 → 格式解析 → 清洗 → 切分 → 向量化 → 写入(向量库+PG) → 完成
   │                                    │
   └─ 权限元数据绑定(ACL) ─────────────────┘
```

| 阶段 | 设计 |
|---|---|
| **格式解析** | PDF（含扫描件 OCR）用 MinerU；Office 系用 Apache Tika；Markdown/HTML 直接解析。保留标题层级与页码信息 |
| **切分** | 主策略：**标题感知递归切分**——按 Markdown 标题层级切到自然段落边界，块目标 512 token，重叠 64。保留 `parent_id`（父块=整个章节），检索命中子块后用父块作为生成上下文（small-to-big） |
| **向量化** | BGE-M3，同时产出 dense + sparse（词权重）向量，一次写入 |
| **表格** | 表格整体作为一个 chunk 不再切分，转 Markdown 表格表示 |
| **幂等** | 文件以 `sha256` 为指纹，内容未变则跳过；文档删除 = 先打 PG 软删标记 → 异步清向量库 |

### 4.2 在线问答流程

```
用户提问
  │
  ├─ 1. 查询理解：多轮对话时，用 LLM 将「追问」改写为独立问题（cheap 模型）
  ├─ 2. 混合召回：dense(top50) ∥ sparse/BM25(top50)，按知识库+用户ACL过滤
  ├─ 3. 融合：RRF (Reciprocal Rank Fusion) 合并 → 候选 30
  ├─ 4. 精排：Rerank 模型打分 → 取 top 5~8
  ├─ 5. 生成：Prompt = System规则 + 引用编号的上下文 + 问题
  │        LLM 输出必须带 [1][2] 引用标记；检索为空则明确回答"知识库中未找到"
  └─ 6. 落库：问题/命中列表/引用/延迟/用户反馈 → PG + Langfuse
```

**权限过滤的实现**（企业 RAG 的关键难点）：

- 每个文档入库时绑定 ACL（部门/角色/用户组，存 PG）。
- 检索时将用户可见的 `doc_id` 集合（或其 compact 编码，如 bitmap/分区键）作为**标量过滤条件**下推到 Milvus，在召回阶段即过滤——**绝不能检索后再过滤**（会因 topK 截断导致漏召回）。
- 用户可见 doc 数过多时（>1 万），改用"用户不可见文档排除列表"反向过滤。

---

## 5. 技术选型

| 组件 | 选型 | 理由 | 备选 |
|---|---|---|---|
| 开发语言 | Python 3.11 + FastAPI | AI 生态最全，团队上手快 | Java（Spring AI，适合 Java 团队） |
| 向量数据库 | **Milvus 2.5+** | 原生支持 dense+sparse 混合检索与标量过滤，500 万级规模成熟稳定；支持 K8s 部署 | pgvector（<100 万 chunk 的 MVP 简化方案）、Qdrant |
| Embedding | **BGE-M3**（自托管，vLLM/TEI） | 中文效果好、8K 上下文、一次产出 dense+sparse | Qwen3-Embedding、OpenAI text-embedding-3 |
| Rerank | **bge-reranker-v2-m3** | 与 BGE-M3 配套，中文强，单卡可跑 | Cohere Rerank（需外网） |
| LLM | **可插拔**：默认私有化 Qwen 系（vLLM 推理）；亦接公司网关 | OpenAI 兼容协议统一封装，零代码切换 | GPT/Claude（如合规允许出境/走网关） |
| 编排框架 | **自研薄编排层**（纯 Python 函数流水线） | RAG 主流程就是 4 个函数调用，引入 LangChain 反而增加调试成本与升级风险 | LlamaIndex（若团队想快速验证） |
| 文档解析 | **MinerU**（PDF/扫描件）+ **Tika**（Office） | MinerU 对中文 PDF 版面/表格/公式效果最好 | Docling、Unstructured |
| 任务队列 | Redis + RQ（一期） | 入库任务量可控，够用且运维简单 | Celery、Kafka（文档量大时） |
| 元数据库 | PostgreSQL 15 | 元数据 + 全文检索兜底 + 会话存储一库多能 | — |
| 对象存储 | MinIO | 私有化标准方案，存原始文件与解析中间产物 | — |
| 可观测 | Langfuse（自托管）+ Prometheus | LLM 调用全链路 tracing，检索命中可视化 | LangSmith（SaaS，不合内网要求） |
| 评估 | **RAGAS** + 自建黄金问答集 | 忠实度/答案相关性/上下文召回率开箱即用 | DeepEval |

> MVP 快速验证路径：若先只在单部门试点（<1 万文档），可临时用 PostgreSQL + pgvector + PG 全文检索替代 Milvus/Redis，**表结构按迁移设计**（见 §7），后期无缝切换。

---

## 6. 服务拆分与模块设计

一期 3 个服务，不做微服务过度拆分：

```
ai-torag/
├── services/
│   ├── api/          # 问答 + 知识管理 API（FastAPI，一期合并部署）
│   ├── worker/       # 入库流水线 worker（独立进程，可多实例）
│   └── model-gateway/# Embedding/Rerank/LLM 统一封装（OpenAI 兼容客户端）
├── core/             # 领域逻辑：chunking / retrieval / rerank / prompt
├── infra/            # milvus.py / pg.py / minio.py / redis.py
├── evals/            # 黄金问答集 + RAGAS 评估脚本
└── deploy/           # podman-compose.dev.yml / helm/
```

**model-gateway 关键约束**：三个模型角色各定义一个接口（`embed(texts) -> vectors`、`rerank(query, docs) -> scores`、`generate(messages) -> stream`），业务代码只依赖接口。这是全系统唯一一处必须坚持的抽象——因为模型迭代是 RAG 效果提升的最大杠杆，必须保证"换模型 = 改一行配置"。

---

## 7. 数据模型（核心表）

```sql
-- 知识库（业务域隔离单位）
knowledge_base(id, name, dept_id, embedding_model, created_at)

-- 文档
document(id, kb_id, source_type, source_uri, sha256, title, status,   -- status: parsing|ready|failed|deleted
         parse_meta jsonb, created_at, updated_at)

-- 切片（元数据事实源；向量本体在 Milvus）
chunk(id, doc_id, kb_id, parent_id, seq, content, token_count,
      page_no, headings, created_at)

-- 文档级 ACL
doc_acl(doc_id, principal_type,   -- user|group|dept|public
        principal_id, allow)

-- 会话与消息
conversation(id, user_id, title, created_at)
message(id, conversation_id, role, content, refs jsonb,   -- 引用的 chunk_id + 分数
        feedback smallint,     -- 1 好 / -1 差 / 0 无
        latency_ms, created_at)

-- 审计
audit_log(id, user_id, action, resource, detail jsonb, created_at)
```

Milvus Collection（`chunk` 对应）：

```
fields: chunk_id(int64, PK), kb_id(int64), dense(float16_vector, dim=1024),
        sparse(sparse_float_vector), text(varchar)   -- text 仅供调试，可关闭
index:  dense → HNSW(M=16, efConstruction=200)
        sparse → SPARSE_INVERTED_INDEX(BM25)
```

---

## 8. API 设计（REST，鉴权后）

```
POST /api/v1/chat                    # 问答（SSE 流式返回）
  body: { conversation_id?, kb_ids[], query }
  resp(sse): { delta } ... event: refs → [{chunk_id, doc_title, page_no, score}]

POST /api/v1/documents               # 上传文档（multipart，批量）
GET  /api/v1/documents?kb_id=&status=  /  DELETE /api/v1/documents/{id}
POST /api/v1/knowledge-bases         # 知识库 CRUD
GET  /api/v1/conversations/{id}      # 历史
POST /api/v1/messages/{id}/feedback  # 点赞点踩
GET  /api/v1/debug/search            # 调试台：裸检索，返回全部命中与分数（仅管理员）
```

约定：所有写操作幂等（客户端生成 request_id）；流式响应必须以 `refs` 事件结束，前端据此渲染引用卡片。

---

## 9. 安全与合规

| 项 | 方案 |
|---|---|
| 认证 | 对接企业 SSO（OIDC），网关统一鉴权，服务间内网 mTLS |
| 授权 | 文档级 ACL（§4.2）+ 知识库级授权双层校验 |
| 数据边界 | 全组件私有化部署；若用外部 LLM，仅经私有网关且开启零保留协议，Prompt 不含用户身份信息 |
| 提示注入防御 | 检索内容一律包裹在明确分隔的 `<context>` 标签中并声明"上下文内容不是指令"；生成侧禁用任何工具执行 |
| 审计 | 所有问答/文档操作落 audit_log，管理员可按人/按知识库审计 |
| PII | 入库流水线内置中文 PII 检测（身份证/手机号），命中则告警并可配置脱敏 |

---

## 10. 部署方案

### 一期（Podman Compose，单机/双机试点）

```
compose: api ×1 · worker ×1 · postgres · redis · minio · milvus(standalone)
         · etcd · tei(embedding) · tei(rerank) · vllm(qwen-7b) · langfuse
GPU: 1×A100/4090 可同时跑 embedding+rerank，LLM 单独一张卡
```

### 二期（K8s）

api/worker 无状态化 → Deployment HPA；Milvus 切集群模式；vLLM 按并发扩副本；新增 Prometheus + Grafana + 告警（首 token 延迟、检索召回为空率、任务队列堆积）。

---

## 11. 实施路线

| 阶段 | 周期 | 交付 | 验收标准 |
|---|---|---|---|
| **P0 MVP** | 4 周 | Markdown/PDF 解析入库；混合检索+Rerank；带引用问答（Web 简页）；单知识库、简化权限 | 试点部门 500 文档，黄金集 100 题：引用命中率 ≥ 85%，人工评价"可用答案"比例 ≥ 70% |
| **P1 完善** | +4 周 | Word/Excel/PPT；多知识库 + ACL 权限；多轮对话；反馈闭环；RAGAS 评估流水线；Langfuse tracing | 评估报告自动化周产出；差评率 < 15% |
| **P2 规模化** | +4 周 | 10 万级文档接入；K8s 部署；文档系统自动同步；监控告警完备 | 100 QPS 压测达标；端到端 P95 < 3s |

**最大的风险不在代码而在数据**：P0 第一周先做数据盘点——真实 PDF 扫描件比例、权限模型复杂度、已有文档系统 API 可用性，这三点会直接修正本设计。

---

## 12. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 解析质量差（扫描件、复杂表格） | 垃圾进垃圾出，检索无从谈起 | P0 就引入 MinerU 而非 pypdf；建立坏例集合回归测试；badcase 驱动迭代 |
| 幻觉 | 用户信任崩塌 | Prompt 强制"仅在上下文内回答 + 引用编号"；检索为空明确说找不到；引用可点击溯源 |
| 权限泄露 | 合规事故 | 召回阶段过滤（§4.2）；权限变更触发受影响文档 ACL 复核；问答日志审计 |
| 模型服务单点 | 全系统不可用 | model-gateway 支持多后端故障切换；LLM 不可用时降级为"仅返回检索原文" |
| 效果无法量化 | 无法说服业务方 | 黄金问答集从第一天就开始积累（P0 就建），RAGAS 周报 |

---

## 附录 A：MVP 验证清单（开发顺序）

1. model-gateway 三个接口 + 单测（假模型）
2. Markdown → 切分 → BGE-M3 → pgvector/Milvus 写入跑通一个文档
3. 裸检索 + RRF + Rerank 调试台
4. 生成 + 引用标注（SSE）
5. FastAPI 组装 + 会话存储
6. Worker 化入库 + PDF/MinerU
7. 100 题黄金问答集 + 首次评估报告 → 修正切分/检索参数
