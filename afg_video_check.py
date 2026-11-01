"""从 WSL 启动录像、新在线追踪与独立 Windows AFG 执行器。"""

import argparse
import json
import subprocess
import time
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from customer_workspace.experiment_runner import IDLE_AMPLITUDE_VPP, load_plan
from customer_workspace.four_pzt_validation import check_times
from customer_workspace.parallel_video_check import add_tracking_arguments, run_check
from customer_workspace.streamcopy_recording import write_json
from customer_workspace.video_clock import read_ready
from customer_workspace.wsl_launcher import windows_path
from microbubble_experiment.afg import DEFAULT_CONFIG, load_config
from microbubble_experiment.configuration import ROOT

PLAN = ROOT / "customer_workspace/configs/four_pzt_video_20s.json"
CONFIG = DEFAULT_CONFIG


def load_video_plan(path: Path, config: Path = CONFIG):
    """设备启动前验证本入口支持的20秒调幅计划，并检查稳定读回的时间余量。"""
    plan = load_plan(path, load_config(config))
    if plan.duration_s != 20 or any(step.frequencies_hz for step in plan.steps):
        raise ValueError("联合录像入口目前支持20秒、频率固定的调幅时间表")
    check_times(plan, None)
    return plan


class AFGControl:
    """
    相机可以单独停止；AFG 子进程必须自行关断并退出，父进程不强杀它。
    封装了对 Windows 侧 AFG（音频信号发生器）子进程 的启动、状态查询、优雅停止等全部操作
    """
    def __init__(self, plan=PLAN, config=CONFIG, mode="normal"):
        self.plan, self.config, self.mode = plan, config, mode
        self.process = self.log = None
        self.output = None

    def start(self, output):
        """
        启动 AFG 子进程
        """
        # 保存并执行同一份已验证时间表，避免准备期间源文件变化导致记录与执行不符。
        self.output = output
        plan = load_video_plan(self.plan, self.config)
        selected_plan = output / "selected_plan.json"
        write_json(selected_plan, asdict(plan))

        # 提取通道幅值，生成会话清单
        amplitudes = {
            name: setting.amplitude_vpp for name, setting in plan.settings.items()
        }
        manifest = {
            "experiment_id": uuid.uuid4().hex,
            "stream_id": uuid.uuid4().hex,
            "input_kind": "camera",
            "mode": self.mode,
            "action_seconds": 20,
            "recording_seconds": 22,
            "clock_domain": "same Windows host monotonic; never subtract WSL counter",
            "active_vpp": next(iter(amplitudes.values()))
            if len(set(amplitudes.values())) == 1
            else None,
            "active_vpp_by_channel": amplitudes,
            "frequency_hz_by_channel": {
                name: setting.frequency_hz for name, setting in plan.settings.items()
            },
            "idle_vpp": IDLE_AMPLITUDE_VPP,
            "requested_plan": str(self.plan.resolve()),
            "selected_plan": selected_plan.name,
        }
        write_json(output / "link_session.json", manifest)

        # 拼 Windows 侧启动命令
        machine = json.loads((ROOT / "configs/machine.local.json").read_text())
        command = [
            machine["windows_afg_python"],
            "-X",
            "utf8",
            "-u",
            "-B",
            "-m",
            "customer_workspace.afg_video_control",
            "--output",
            windows_path(output),
            "--plan",
            windows_path(selected_plan),
            "--config",
            windows_path(self.config),
            "--mode",
            self.mode,
        ]
        write_json(output / "afg_launch.json", {"args": command})

        # 开日志文件、启动子进程
        self.log = (output / "afg_console.log").open("xb")
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=self.log,
            stderr=self.log,
            start_new_session=True,
        )

    def ready(self):
        return read_ready(self.output / "afg_ready.json") is not None

    def finished(self):
        return self.process.poll() is not None

    def poll(self):
        if self.finished() and self.process.returncode:
            summary = read_ready(self.output / "afg_link_summary.json")
            raise RuntimeError(
                f"AFG 进程失败，退出码 {self.process.returncode}；原因：{summary.get('error') if summary else '见 afg_console.log'}"
            )

    def stop(self):
        if self.process is None:
            if self.log:
                self.log.close()
            return
        requested = None
        if not self.finished():
            requested = time.monotonic()
            try:
                self.process.stdin.write(b"cancel\n")
                self.process.stdin.flush()
            except OSError:
                pass  # 子进程可能刚结束；EOF 守护仍是独立的取消路径。
        while not self.finished():
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                continue
            except KeyboardInterrupt:
                print(
                    "正在等待 AFG 关闭输出并保存记录，不强制终止设备进程。", flush=True
                )
        self.process.stdin.close()
        self.log.close()
        write_json(
            self.output / "afg_process_exit.json",
            {
                "exit_code": self.process.returncode,
                "parent_cancel_requested_wsl_s": requested,
                "note": "WSL时间仅说明父进程行为，不用于动作与视频相减",
            },
        )


def main():
    """
    运行一次音视频同步（AV sync）自动化校验，根据结果成功或失败返回退出码
    目前 “真实相机录像 + 新微泡识别追踪 + PZT 自动控制” 的联合实验入口
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("normal", "cancel", "fault"), default="normal")
    parser.add_argument("--output-dir", type=Path)
    add_tracking_arguments(parser, offline=False)       # 追加一批与 "追踪 / 定位" 相关的参数
    parser.add_argument("--plan", type=Path, default=PLAN, help="20秒调幅时间表 JSON，默认沿用四路联动")
    args = parser.parse_args()
    
    # 自动生成输出目录
    if args.output_dir is None:
        stamp = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y%m%d-%H%M%S")
        args.output_dir = (
            ROOT / "customer_workspace/validation" / f"avsync_{stamp}_{args.mode}"
        )

    # 创建控制对象并执行检查
    control = AFGControl(plan=args.plan, mode=args.mode)
    error = None
    try:
        report = run_check(
            SimpleNamespace(
                output_dir=args.output_dir,
                background=args.background,
                camera_config=args.camera_config,
                opencv_threads=args.opencv_threads,
                mode="parallel",
                seconds=22,
            ),
            control=control,
        )
        if not report.get("recording_passed"):
            raise RuntimeError("录像完整性检查未通过；记录已保留，请查看 recording_check.json")
    except BaseException as exc:  
        # noqa: BLE001 -- run_check已先等待AFG收尾；保留真实中断原因。
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if control.output is not None:
            write_json(control.output / "integration_failure.json", error)

    if error:
        print(json.dumps(error, ensure_ascii=False), flush=True)
        return 1

    print(f"录像和AFG记录已保存：{args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
