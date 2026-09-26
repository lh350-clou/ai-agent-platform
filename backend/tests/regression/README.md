# Regression V1（核心能力回归）

**一条命令，回答「这次改动有没有把已经能用的东西弄坏」。**

```bash
cd backend
pytest -m regression
```

跑完的判读方式只有一句：**全绿 = 已有能力都还在；有红 = 看红在哪。**

---

## 和其他几套测试的分工

四者互不替代，也互不复制：

| | 位置 | 跑的是什么 | 红了说明 | 要不要外部服务 |
| --- | --- | --- | --- | --- |
| **Unit Test** | `tests/unit/` | LLM / MCP / embedding / 向量库全部换成假实现 | 编排逻辑写错了 | 不要 |
| **Integration Test** | 本目录里的 `integration` 用例 | 真服务、真模型、真子进程 | 接线断了 / 环境不对 | 要 |
| **Evaluation V1** | `tests/evals/` | 真实链路 + grader，输出通过率与指标 | 系统变好还是变差（**不是**通过/失败） | 要 |
| **Regression** | `pytest -m regression` | 上面两部分里守住核心能力的那批 | 已有能力被破坏 | 视用例而定 |

两条正交的标记（定义见 `backend/pytest.ini`）：

- `unit` / `integration` —— **怎么跑**（要不要外部服务）
- `regression` —— **为什么跑**（守住已有能力）

于是 `pytest -m regression` 目前选中的就是仓库里的全部用例。这不是巧合：
现在每一条用例都是为了守住某个已有能力写的。标记的意义在于以后新增
「先跑跑看」的探索性用例时，它可以不进回归集 —— 而不需要大家每次手动挑文件。

### 为什么不把 grader 逻辑抄进来

评测层「判得准不准」由 `tests/unit/test_eval_tool_calling_grader.py` 守着；
回归层只验「评测这条链路还跑不跑得通」（`test_eval_pipeline.py`：跑一次真入口，
看退出码和报告）。抄一份 grader 进来等于有两份「什么算通过」的定义，
改了一处没改另一处时两边会给出矛盾结论，而没人知道该信哪个。

---

## 覆盖范围

| 用例文件 | 守的能力 | 依赖 |
| --- | --- | --- |
| `test_health.py` | `/health` 的字段契约；**数据库挂了也返回 200**；数据库正常时报 ok | 无 / PostgreSQL |
| `test_agent_loop.py` | Agent 工具循环端到端：模型请求检索 → 真查 Milvus → 作答，且 Trace 如实记录轮数/耗时/错误 | DeepSeek + SiliconFlow + Milvus |
| `test_rag_retrieval.py` | 入库即可检索、**跨知识库查不到**、按知识库删除立即生效 | SiliconFlow + Milvus |
| `test_mcp_tool.py` | MCP 工具发现、参数 schema、正常调用、非法参数返回**可读错误**而非协议失败、名字前缀可逆 | 无（起 Python 子进程） |
| `test_postgres.py` | ORM 写读回滚、模型定义与数据库 schema 一致、会话/消息外键可写、`SELECT 1` 探活 | PostgreSQL |
| `test_eval_pipeline.py` | Evaluation V1 入口跑通并落盘报告（只跑 `rag` 层，不评质量） | SiliconFlow + Milvus |
| `tests/unit/*`（4 个文件） | Agent Trace、工具契约、grader 的 F1 计算 | 无 |

Agent 循环与工具契约的分工值得单独说一句：`top_k` 超限收敛这类边界
**在 Agent 链路上无法观测**（有收敛就改成 5，没有就被范围校验整次拒绝，
两种情况观测到的 `top_k` 都不超过 5），所以它归 `tests/unit/test_agent_tool_contract.py`；
真实链路能观测到的是「工具到底有没有被真的调用、结果有没有回到模型」。

---

## 数据安全

回归套件不改任何生产数据：

- **Milvus**：知识库 ID 每条用例随机生成（`uuid4`），用例结束【含失败】时按 ID 清空；
- **PostgreSQL**：只做「写一行 → 读回来 → 回滚」，不 commit、不建表、不改 schema；
- **磁盘**：临时语料写在 pytest 的 `tmp_path` 里，由 pytest 自动删除；
- **评测报告**：写到 `tmp_path`，不落进仓库。

跑完可以自查：Milvus 里的 `knowledge_base_id` 集合、PostgreSQL 各表的行数
都和跑之前一致。

---

## 服务不可用时怎么办

集成用例在外部服务不可用时会 **skip，并写明原因**（`pytest.ini` 里的 `-rs`
保证 skip 理由一定打印出来）。这样做的取舍是：

    直接失败 —— 「今天没开 Docker」看起来像「代码坏了」，红几次之后就没人看了；
    静默跳过 —— 「一条都没测」和「全部通过」在输出上一模一样，更危险。

所以第三种做法：**跳过，但必须看得见，并且可以关掉。**

探活在会话开始时做一轮（`availability.py`），覆盖四件事：PostgreSQL 和 Milvus
能不能建 TCP 连接、两个 API Key 配没配、两个 API 域名能不能完成一次 HTTPS 请求。
后两项缺一不可 —— Key 配了但网络断了（VPN 掉线、公司网络拦了）与 Key 没配
是两种不同的原因，只探一样，另一种情况下用例照样会红，而红出来的是一串
`ConnectError`，看到的人只会以为是代码坏了。

```bash
# 只跑不依赖外部服务的那部分
pytest -m "regression and unit"

# 只跑集成部分
pytest -m "regression and not unit"

# CI 用：服务不可用不再跳过，直接失败
REGRESSION_REQUIRE_SERVICES=1 pytest -m regression
```

---

## 两个已知的坑

1. **集成用例打的是真实外部接口，偶发网络失败会让它红。** 开始前能探到的网络故障
   （HTTPS 都连不上）会在探活阶段变成 skip 并写明原因；但**跑到一半**才断的情况探活
   看不到，那时会红，判断方法是看失败类型：连接错误 / 超时
   （`APIConnectionError`、`TimeoutExpired`）是环境问题，重跑即可；
   断言失败才是真的坏了。评测那条用例超时会把子进程已抓到的输出一起打出来。
2. **事件循环是按用例重建的。** pytest-asyncio 默认每条用例一个新循环，
   而 `core/database.py` 的 engine 是进程级单例，连接池里的连接会和创建它的
   循环绑死 —— 上一条用例的循环一关，下一条用例就会炸在连接池内部的
   `Event loop is closed`。`conftest.py` 里那条 autouse fixture 在每条用例
   结束时 dispose 一次就是为了这个。生产环境不存在这个问题（一个进程一个循环）。

## 加一条回归用例

1. 先问：**这条已经有人守了吗？** 有就不要再写一条 —— 重复的断言不会多挡住什么，
   只会让同一个原因造成多处红。
2. 放在本目录，按上面的表选一个文件；需要临时知识库就用 `new_kb` fixture。
3. 打上标记：要外部服务用 `integration`，不要用 `unit`，两者都加 `regression`。
4. 跑 `pytest -m regression` 之前，先单独跑这一个文件 —— 别让整轮回归等一条新用例。
