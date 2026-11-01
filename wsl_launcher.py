"""从 WSL 启动现有 Windows Python，转换路径并传递取消请求。

不复制源码、不安装驱动，也不在 Linux 端打开仪器。Windows 子进程读取
共享项目中的同一份执行器；标准输出直接回到当前 WSL 终端。
"""

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from microbubble_experiment.desktop import is_wsl


def windows_path(path: Path) -> str:
    """将 WSL 路径转换为 Windows 可读的绝对路径，保留空格和中文。"""
    result = subprocess.run(
        ["wslpath", "-w", str(path.resolve())],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    converted = result.stdout.strip()
    if not converted:
        raise OSError(f"无法将路径转换给 Windows：{path}")
    return converted


def watch_launcher_input(cancel: Callable[[], None]) -> None:
    """Windows 端等待取消通知；WSL 端退出导致管道关闭时也请求停止。"""
    try:
        # 此专用管道不接收其他交互；任意一行或 EOF 都表示请求停止。
        sys.stdin.readline()
    except (OSError, UnicodeError):
        pass  # 通信中断按取消处理，不继续无人管理的实验。
    finally:
        cancel()


def wait_for_windows(command: list[str], project_root: Path) -> int:
    """等待 Windows 进程；Ctrl+C 请求正常收尾，不强杀设备控制进程。"""
    process = subprocess.Popen(
        command,
        cwd=project_root,
        stdin=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        # 避免终端 Ctrl+C 直接终止子进程，改由管道通知执行器取消。
        start_new_session=True,
    )
    cancel_sent = False
    try:
        while True:
            try:
                return process.wait()
            except KeyboardInterrupt:
                if not cancel_sent:
                    print(
                        "已请求停止，正在等待 Windows 端关闭输出并保存记录。",
                        flush=True,
                    )
                    try:
                        process.stdin.write("cancel\n")
                        process.stdin.flush()
                        cancel_sent = True
                    except OSError as exc:
                        print(
                            f"停止请求未能发送：{exc}；正在等待 Windows 进程结束。",
                            flush=True,
                        )
                else:
                    print("仍在等待 Windows 端收尾，请勿强行关闭终端。", flush=True)
    finally:
        # 子进程已结束时管道可能已断开；清理不能覆盖它的退出状态。
        try:
            process.stdin.close()
        except OSError:
            pass


def launch_from_wsl(
    project_root: Path,
    plan: Path,
    config: Path,
    output_dir: Path | None = None,
    windows_python: Path | None = None,
    *,
    repeat: int = 1,
    module: str = "customer_workspace.experiment_runner",
    extra_args: list[str] | None = None,
) -> int:
    """
    通过现有 Windows 环境执行实验或专项测试，返回 Windows 进程退出码。

    默认解释器取自现有 machine.local.json 的 windows_afg_python。
    调用方明确指定模块及额外参数；共用路径转换、取消通知和进程收尾。
    """
    if not is_wsl():
        raise OSError("当前不是 WSL；请在 WSL 终端或 Windows Python 中运行。")

    if windows_python is None:
        machine_path = project_root / "configs/machine.local.json"

        # 检查配置文件是否存在
        if not machine_path.is_file():
            raise FileNotFoundError(
                f"缺少 {machine_path}；请设置 windows_afg_python 或使用 --windows-python。"
            )

        # 读取配置文件里的内容
        machine = json.loads(machine_path.read_text(encoding="utf-8-sig"))

        # 找到 Windows Python 的位置，并检查有没有填好
        configured = machine.get("windows_afg_python")
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError("machine.local.json 缺少 windows_afg_python 路径。")
        windows_python = Path(configured)

    windows_python = windows_python.resolve()
    if not windows_python.is_file():
        raise FileNotFoundError(f"找不到 Windows Python：{windows_python}")
    
    if output_dir is not None and output_dir.exists():
        raise FileExistsError(f"结果目录已存在：{output_dir}")

    # 不经过 shell；文件名中的空格、中文和符号按单个参数传递。
    command = [
        str(windows_python),
        "-X",
        "utf8",
        "-u",
        "-B",
        "-m",
        module,
        windows_path(plan),
        "--config",
        windows_path(config),
    ]
    if output_dir is not None:
        command.extend(["--output-dir", windows_path(output_dir)])
    if repeat != 1:
        command.extend(["--repeat", str(repeat)])
    if extra_args:
        command.extend(extra_args)
    command.append("--launcher-stdin")
    print(f"从 WSL 调用 Windows Python：{windows_python}", flush=True)
    return wait_for_windows(command, project_root)
