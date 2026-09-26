"""Regression V1：核心能力回归集。

它要回答的问题只有一个 —— **这次改动有没有把已经能用的东西弄坏。**

和另外两套测试的分工（三者不重叠，也不互相复制）：

| | 位置 | 跑的是什么 | 失败说明 |
| --- | --- | --- | --- |
| 单元测试 | `tests/unit/` | 把 LLM / MCP / 向量库全换成假实现，验编排逻辑 | 逻辑写错了 |
| 集成测试 | 本目录 `test_*.py` 里的 integration 用例 | 真服务、真模型、真子进程 | 接线断了 / 环境不对 |
| 评测 | `tests/evals/` | 真实链路 + grader，输出通过率与指标 | 系统变好还是变差 |
| 回归 | `pytest -m regression` | 上面两部分里守住核心能力的那批 | 已有能力被破坏 |

评测的 grader 逻辑一行都没有复制进来 —— 判定正确性由
`tests/unit/test_eval_tool_calling_grader.py` 守，评测链路的连通性由
`test_eval_pipeline.py` 用「跑一次真入口、看退出码和报告」的方式守。
「评得准不准」和「评测还能不能跑」是两件事。

本目录的用例【不修改任何生产数据】：
    - Milvus：知识库 ID 每次随机生成，用例结束（含失败）时按 ID 清空；
    - PostgreSQL：只做「写一行 → 读回来 → 回滚」，不 commit，不建表；
    - 磁盘：临时语料写在 pytest 的 tmp_path 里，由 pytest 自动删除；
    - 评测报告写到 tmp_path，不落进仓库。

跑法见 tests/regression/README.md。
"""
