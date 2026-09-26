"""Evaluation V1 编排链路的冒烟回归。

【本文件不碰任何 grader 逻辑】—— 这是刻意的分工：

    grader 判定得对不对      -> tests/unit/test_eval_tool_calling_grader.py（纯函数，离线）
    评测链路还跑不跑得通     -> 本文件（跑真入口，看退出码和报告）

把 grader 的判定抄一份进来是典型的重复：抄完就有两份「什么算通过」的定义，
改了一处没改另一处时，两边会给出互相矛盾的结论，而没人知道该信哪个。

为什么选 rag 层做冒烟：三层里只有它【不调用任何 LLM】（只用 embedding + Milvus），
既便宜又稳定，适合每次回归都跑。tool_calling / final_answer 的端到端由
Evaluation 自己负责，回归不重复买单。

为什么用子进程跑而不是 import 进来调 `_run()`：那是个下划线开头的内部函数，
入口是 `python -m tests.evals.runner`。子进程跑的就是用户敲的那条命令，
退出码、控制台输出、报告落盘全都一起验了；顺带还避免了评测的
settings 改写和 Milvus 连接污染本进程。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.config import BASE_DIR
from tests.regression.availability import ServiceStatus

pytestmark = [pytest.mark.integration, pytest.mark.regression]

# 只跑检索层的前 2 条：足够走完「加载数据集 → 语料入库 → 逐条执行 → 判定 →
# 汇总 → 落盘 → 清理」整条编排链路，又不会让回归慢到没人愿意跑。
EVAL_LAYER = "rag"
CASE_LIMIT = 2

# 真去调 embedding 和 Milvus，给足时间；超过这个时长说明卡住了，
# 与其一直挂着，不如让 subprocess 把它杀掉并报错。
TIMEOUT_SECONDS = 300


def _text(value: str | bytes | None) -> str:
    """把可能为 None / bytes 的子进程输出统一成字符串。"""
    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode("utf-8", "replace")


def test_eval_runner_rag_layer_smoke(services: ServiceStatus, tmp_path: Path) -> None:
    """跑一次真的评测入口，看它能不能自己走完并写出报告。"""
    services.require_milvus()
    services.require_embedding()

    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.evals.runner",
                "--layer",
                EVAL_LAYER,
                "--limit",
                str(CASE_LIMIT),
                # 报告写进 tmp_path：评测报告是运行产物，不该往仓库里丢
                # （仓库里那份 reports/ 已经在 .gitignore 里，这里是双保险）。
                "--report-dir",
                str(tmp_path),
            ],
            # 必须在 backend/ 下跑：评测入口是用 `-m tests.evals.runner` 调的，
            # 它要能 import 到 tests 和 app 两个包。
            cwd=BASE_DIR / "backend",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        # 本地正常跑完大约 30 秒，所以走到这里说明它卡住了（多半是外部接口没响应）。
        # 把已经抓到的输出一起抛出来：否则只剩一句「超时」，
        # 完全看不出它停在哪一步，只能手工再跑一遍。
        raise AssertionError(
            f"评测入口在 {TIMEOUT_SECONDS} 秒内没有结束。已抓到的输出：\n"
            f"--- stdout ---\n{_text(exc.stdout)}\n--- stderr ---\n{_text(exc.stderr)}"
        ) from exc

    # 评测入口自己的约定是「跑不通就非 0」。把两段输出一起打出来，
    # 否则失败时只能看到一句「退出码 1」，还得手工再跑一次才知道为什么。
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"

    reports = list(tmp_path.glob("*.json"))
    assert len(reports) == 1, f"应当恰好落盘一份报告，实际有 {[p.name for p in reports]}"

    report = json.loads(reports[0].read_text(encoding="utf-8"))

    assert [layer["layer"] for layer in report["layers"]] == [EVAL_LAYER]
    layer = report["layers"][0]
    assert layer["total"] == CASE_LIMIT
    # invalid 的含义是「这条用例根本没测出来」（判官输出看不懂、服务调用失败）。
    # 它为 0 才说明这条链路是真的走通了。
    #
    # 这里【不断言 passed 的条数】：那等于在回归里评估检索质量，
    # 检索质量变差由 Evaluation 报告回答，而且它会因为语料和模型波动而红 ——
    # 一个会随机变红的回归套件，很快就没人看了。
    assert layer["invalid"] == 0, [
        case for case in report["cases"] if case["status"] == "invalid"
    ]
