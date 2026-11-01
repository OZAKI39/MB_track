"""真实硬件错误队列测试：STOP 并读回关闭后，注入一条不存在的命令。

本文件只引用正式实现，不模拟 AFG，不伪造查询回复；不能与其他实验并行运行。
从项目根目录通过已有 Windows Python 的标准输入运行。
"""

import json
import threading
from pathlib import Path

from customer_workspace.afg_timing import RoundUnreliable, TimedRig
from customer_workspace.experiment_runner import load_plan, run_experiments
from microbubble_experiment.afg import load_config

ROOT = Path("customer_workspace/validation/write_only_20261023-190015")


class ErrorQueueTestRig(TimedRig):
    """在真实 STOP 后注入错误；其余操作和核查均使用正式实现。"""

    def check_state(self, settings, phase):
        if phase == "after":
            actual = self.status()
            if any(channel["output"] for channel in actual.values()):
                raise RuntimeError("错误注入前未确认全部关闭")
            self.sessions["A"].write("CUSTOMER:INVALID:COMMAND")
            with (ROOT / "error_injection_command.json").open(
                "x", encoding="utf-8"
            ) as file:
                json.dump(
                    {
                        "device": "A",
                        "command": "CUSTOMER:INVALID:COMMAND",
                        "all_outputs_off_before_injection": True,
                        "status_before_injection": actual,
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
        return super().check_state(settings, phase)


def main():
    config = load_config()
    plan = load_plan("customer_workspace/configs/pzt1_pulse.json", config)
    expected_error = None
    try:
        run_experiments(
            ErrorQueueTestRig(config), plan, ROOT / "error_test", 1, threading.Event()
        )
    except RoundUnreliable as exc:
        expected_error = str(exc)
    summary = json.loads((ROOT / "error_test/summary.json").read_text(encoding="utf-8"))
    report = json.loads(
        (ROOT / "error_test/timing_summary.json").read_text(encoding="utf-8")
    )
    success = (
        expected_error is not None
        and summary["status"] == "unreliable"
        and summary["data_reliable"] is False
        and summary["all_outputs_off_confirmed"] is True
        and report["unreliable_runs"] == 1
        and report["paired_intervals"] == 0
        and not (ROOT / "error_test/timing.csv").exists()
    )
    result = {
        "passed": success,
        "expected_error": expected_error,
        "summary_status": summary["status"],
        "excluded_from_timing_statistics": report["paired_intervals"] == 0,
        "all_outputs_off_confirmed": summary["all_outputs_off_confirmed"],
        "audit": summary.get("validation_failure"),
    }
    with (ROOT / "error_injection_result.json").open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not success:
        raise RuntimeError("错误队列与不可靠轮次验证未通过")


if __name__ == "__main__":
    main()
