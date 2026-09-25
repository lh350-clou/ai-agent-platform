# Evaluation V1

三层评测：**Tool Calling** → **RAG** → **Final Answer**。

它评的是生产链路本身（`run_agent` / `embed_text` / `vector_store.search` / `ingest_txt`），一行检索或 Agent 逻辑都没有重写。`app/` 下没有任何一处 import 本目录 —— 依赖方向只有 `evals → app`。

---

## 怎么跑

必须在 `backend/` 目录下，用 `-m` 而不是直接跑文件：

```bash
cd backend

python -m tests.evals.runner --layer all
python -m tests.evals.runner --layer rag
python -m tests.evals.runner --layer tool_calling --limit 10
python -m tests.evals.runner --layer final_answer --no-judge --limit 10
python -m tests.evals.runner --layer rag --print-chunks      # 核对 chunk 序号
python -m tests.evals.runner --layer all --fail-under 0.8    # CI 用：低于阈值退出码 1
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--layer` | `all` | `tool_calling` / `rag` / `final_answer` / `all` |
| `--case ID` | 无 | 只跑指定用例，可重复 |
| `--tag TAG` | 无 | 只跑带该标签的用例，可重复（命中任一即可） |
| `--limit N` | 0（全部） | 每层最多跑前 N 条 |
| `--no-mcp` | 关 | 关闭 MCP，触发生产代码里的降级分支 |
| `--no-judge` | 关 | 禁用 LLM 判官，只跑确定性检查 |
| `--concurrency N` | 1 | 层内并发用例数。默认串行，避免自己把自己打到 429 |
| `--keep-vectors` | 关 | 跑完不清理评测语料的向量 |
| `--fail-under X` | 0 | 任一层的 `pass_rate` 低于 X 时退出码 1 |
| `--print-chunks` | 关 | 入库后打印每个 chunk 的序号与前 60 字 |
| `--dataset-dir` / `--report-dir` | `tests/evals/{datasets,reports}` | 路径覆盖 |

报告写到 `tests/evals/reports/<run_id>.json`（已进 `.gitignore`）。

### 前置条件

| 层 | DeepSeek | SiliconFlow | Milvus | PostgreSQL |
| --- | --- | --- | --- | --- |
| `tool_calling` | ✅ | ✅ | ✅ | ❌ |
| `rag` | ❌ | ✅ | ✅ | ❌ |
| `final_answer` | ✅ | ✅ | ✅ | ❌ |

**不需要 PostgreSQL**：评测语料的 `knowledge_base_id` 是 uuid5 生成的裸 UUID，只往 Milvus 写向量，一行业务记录都不落 —— 所以不需要建库、不需要跑 Alembic。隔离由 `vector_store.search` 的 `knowledge_base_id` 过滤保证，这些向量永远不会被任何真实业务查询命中。

---

## 三层各评什么

### 1. Tool Calling —— Selection / Arguments / Count / Order / Error

| 检查项 | 权重 | 判定 |
| --- | --- | --- |
| `tool_selection` | 0.30 | 模型**请求**的工具集合 vs 期望集合，metric 用 F1 |
| `arguments:<工具>.<字段>` | 0.30（按字段均分） | 逐字段匹配器判定，取 `success=True` 的调用 |
| `call_count` | 0.15 | 实际**执行**的调用次数是否落在 `[min_calls, max_calls]` |
| `order` | 0.10 | 工具序列是否包含期望的**子序列**（数据集未声明则跳过） |
| `error_free_rate` | 0.15 | 有无失败/被拒绝的调用 + `trace.error` 是否为 null |

两个关键口径：

- **selection 看 `trace.tool_calls`（模型请求的全部，含被白名单拒绝的），count 看 `AgentResult.tool_calls`（真正进入执行的）。** 用前者才能让「模型臆造了一个工具」体现在「选错工具」上；用后者才能让「参数根本没解析出来」不被记成「调用次数不对」。
- **arguments 只取 `success=True` 的调用。** `agent.py` 把 `call.arguments` 的赋值放在参数校验之后，失败调用的 `arguments` 是 `{}`，拿它们评参数会把「参数没解析出来」记成「参数为空」。

> **为什么这里不断言 `top_k` 的上限。** 顺着执行路径走一遍会发现它无法观测：模型给出 `top_k=1000` 时，**有**收敛逻辑就改成 5 并通过校验，**没有**收敛逻辑就被 Pydantic 的范围校验（`le=5`）整次拒绝、`arguments` 保持 `{}`。两种情况下观测到的 `top_k` 都不超过 5，所以「`top_k <= 5`」是一条**不可能失败**的断言 —— 删掉它通过率一模一样，留着只会让人误以为覆盖了边界。真正能区分「有收敛」和「没收敛」的测试在 `tests/unit/test_agent_tool_contract.py`（14 条，含 query 长度、多余字段丢弃、非法值拒绝），那里直接调函数、看它到底返回了什么。**Agent 行为归评测，工具契约归单测，两者不混。**

三层都是**纯函数、零网络、零 LLM**，完全可复现。回归测试见 `tests/unit/test_eval_tool_calling_grader.py`。

### 2. RAG —— Hit@K / Recall@K / MRR

**直接评检索结果，绝不通过最终答案判断 RAG。** 数据集里连 `answer` 字段都没有 —— 一旦放开，就会有人拿它去评生成质量，而「检索没找到」和「找到了但没用好」是两个必须分开归因的问题，混在一起时改哪边都看不出效果。

| 检查项 | 权重 | 判定 |
| --- | --- | --- |
| `hit@K` | 0.40 | 前 K 条里有没有至少一条相关 |
| `recall@K` | 0.40 | 前 K 条覆盖了多少标注的相关切片（只给 snippet 时跳过） |
| `mrr` | 0.20 | 第一条相关结果的名次倒数；**不参与通过判定** |
| `non_empty` | 0 | 只用于归因：区分「检索挂了」和「排序不好」 |
| `hit@1` / `hit@3` / `hit@5` | 0 | 固定档位，只呈现不评分 |

相关性判定两个口径，命中任一即可：`relevant_chunk_seqs`（主口径，精确）+ `relevant_snippets`（兜底，语料改写后序号漂移时仍站得住）。

调用的是 `embed_text` + `vector_store.search` —— 换成别的向量模型或检索策略，这里评的就是新的那一套。

### 3. Final Answer —— Correctness / Relevance / Groundedness

**确定性检查优先**，只有判不动的才交给 LLM 判官。

| 检查项 | 权重 | 判定 | 是否用判官 |
| --- | --- | --- | --- |
| `refusal` | 0.20 | `must_refuse` 与拒答标记命中情况是否一致 | 否 |
| `facts` | 0.35 | `expected_facts` 逐条子串匹配（归一化后） | 否 |
| `forbidden` | 0.10 | `forbidden` 任一命中即 0 分 | 否 |
| `relevance` | 0.15 | 判官给 1~5 分，≥4 通过 | ✅ |
| `groundedness` | 0.20 | 判官逐句核对 + 两条确定性捷径 | 必要时 |
| `non_empty` | 0 | 只用于归因 | 否 |

**拒答要往两个方向测**：`must_refuse=true` 抓「该拒答却编造」（RAG 场景最危险的失败），`must_refuse=false` 抓「不该拒答却拒了」。只测一个方向的话，一个永远回「无法确定」的退化实现能拿满分。

**Groundedness 的两条确定性捷径**（省掉判官调用）：

1. 检索结果为空 + 答案明确拒答 → 直接判**有据**（无资料时不编造，就是最标准的有据行为）
2. 检索结果为空 + 未拒答 + 数据集声明了 `expected_facts` → 直接判**无据**（没有资料却在陈述事实，只能是编的）

判官复用 `services/llm.py` 的 `chat(temperature=0.0)`，不新建 client、不新增依赖。

---

## 数据集

`datasets/*.jsonl`，一行一条，字段定义见 `schemas.py`（Pydantic 校验，坏用例带着**行号**立刻报错，绝不静默跳过 —— 静默跳过会让通过率虚高）。

| 文件 | 条数 | 标签 |
| --- | --- | --- |
| `tool_calling.jsonl` | 10 | `retrieval` / `no_tool` / `mcp` / `multi_tool` / `order` / `args` / `injection` |
| `rag.jsonl` | 12 | `single_hop` / `multi_hop` / `paraphrase` / `other_corpus` |
| `final_answer.jsonl` | 10 | `retrieval` / `refusal` / `trap` / `no_tool` / `mcp` / `injection` |

**一条用例只考察一件事。** `tc-008` 原本问的是「给我 1000 条 Milvus 资料」，同时考察「要不要调工具」和「`top_k` 会不会被收敛」—— 前者是 Agent 行为，后者是工具契约，而且后者在这条链路上根本无法观测（理由见上一节）。现已改成明确的检索请求，只考察 Selection + query 组织；clamp 移到工具契约单测。

### 语料与 chunk 序号

`fixtures/*.txt` 是评测语料，经**生产切分链路**（`split_text`，500 字符 / 50 重叠）切成 7 + 2 个 chunk。

`kb_id` 和 `document_id` 由 uuid5 固定派生（`schemas.eval_kb_id` / `eval_document_id`），而 `chunk_id = f"{document_id}-chunk-{序号:06d}"`（`ingest.build_chunk_id`），所以**chunk_id 在每台机器、每次运行都完全一致** —— 这是数据集里能写 `relevant_chunk_seqs: [3, 4]` 而不是几十个字符的 UUID 的前提。

每次运行 **先清空再入库**：`insert_chunk` 是插入不是 upsert，不清的话同一个 chunk_id 会有多份，`Recall@K` 的分母随之失真。运行结束在 `finally` 里清理（`--keep-vectors` 可保留）。

> ⚠️ **改 `fixtures/*.txt` 之后必须重算全部 `relevant_chunk_seqs`。** 语料改一个字，切分序号就可能整体漂移，而漂移**不会报错** —— 只会让 Recall 悄悄变成 0，看起来像检索退化。改完用 `--print-chunks` 核对。

---

## 结果与统计口径

每条用例有三种状态，**`invalid` 与 `failed` 必须分开**：

| 状态 | 含义 |
| --- | --- |
| `passed` | 判定通过 |
| `failed` | 跑了，但没达标 —— 能力问题 |
| `invalid` | 根本没跑成：模型 API 挂了、判官输出不是 JSON、Milvus 连不上 |

`pass_rate = passed / (total - invalid)`，**分母排除 invalid**。invalid 的含义是「这次没测出来」，不是「模型不行」；把它算进分母等于让一次网络抖动看起来像能力退化，那就违背了把它单列出来的初衷。

`metrics` 按 check 名聚合（`arguments:search_knowledge_base.top_k` 会单独成为一项），所以报告能直接回答「是哪个参数在错」。`by_tag` 让失败可归因 —— 「寒暄类全过、多跳类全挂」这种结论只靠总通过率是看不出来的。

---

## 已知限制（V1 明确接受）

1. **Groundedness 的上下文是「回放」的，不是模型当时看到的那一份。** `services/trace.py` 刻意不记录工具返回内容（它的安全约定要求不抄一份用户数据），所以 runner 只能用 trace 里模型实际用过的 `query` / `top_k` 重新检索一次。同一轮内语料不变，`search` 又是确定性检索，因此结果几乎等价；报告里由 `contexts_replayed` 字段显式标注。**如果模型压根没检索，就没有上下文可回放**，此时只能走上面那两条确定性捷径。
2. **模型非确定性。** 工具选择走 `chat_with_tools(temperature=0.0)`，是稳定的；但强制收敛那一步走 `llm.chat(messages)`，默认 `temperature=0.7`，触发该路径的用例稳定性明显更低。建议连跑 3 次看波动（V1 不做 multi-run 自动化）。
3. **判官与被评模型同源**（都是 `deepseek-chat`），存在自我偏好偏差。确定性检查承担 65% 的权重，判官只占 35%。
4. **grader 与生产常量耦合。** grader 引用 `SEARCH_TOOL_NAME` / `DEFAULT_TOOL_TOP_K` / `build_chunk_id`。这是有意的单一事实来源，但生产改常量时数据集里写死的边界值会静默过时（例如 `MAX_TOOL_TOP_K` 从 5 改成 10，`tc-008` 的 `lte: 5` 就不再检验收敛）。
5. **没有「不该检索到」的负向用例。** 当前 `hit@K` 只会因为「没命中」变红，无法表达「期望检索不到任何相关结果」。要加需要引入反向语义。
6. **不在 V1 范围**：Reliability（多次运行方差）、Regression Dashboard、CI 集成、`/ask` 路径的 Final Answer、多格式语料、MCP 工具的错误注入。
7. **`--no-mcp` 会打出一段 MCP 连接失败的堆栈日志。** 那是预期内的失败（配置指向不存在的模块，靠生产代码自己的降级分支生效），日志级别已临时抬高到 CRITICAL，行为不受影响。

---

## 加一条用例

1. 在对应的 `datasets/*.jsonl` 里加一行（字段见 `schemas.py`）。
2. 若用到新语料，先放 `fixtures/<name>.txt` 并在 `schemas.CORPUS_FILES` 里登记 —— 不登记会以「未知的评测语料」报错，这是刻意的，避免数据集里出现指向不存在文件的标注。
3. 若标注了 chunk 序号，用 `--print-chunks` 核对。
4. 先 `--case <新ID>` 单独跑通，再跑全层。

改了 grader 逻辑的话，跑 `cd backend && pytest`（`tests/unit/test_eval_tool_calling_grader.py` 覆盖了 F1 计算的三类边界）。
