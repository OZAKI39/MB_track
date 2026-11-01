"""限时 Windows GPU/内存诊断，同时复放已有 A/B；不连接实验设备。"""

import argparse
import base64
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

from customer_workspace.tools.analysis_host_trace import POWERSHELL, clock_bridge
from customer_workspace.tools.gpu_decode_check import save_json
from customer_workspace.wsl_launcher import windows_path
from microbubble_experiment.configuration import ROOT


def run(output, profile="Gpu", rounds=1, native_root=None, conditions=False):
    """诊断 WSL 或显式指定的独立 Windows 解码目录；原 WSL 默认保持不变。"""
    output.mkdir(parents=True, exist_ok=False)
    host = (
        Path("/mnt/c/Users/lzy/AppData/Local/Temp") / output.parent.name / output.name
    )
    host.mkdir(parents=True, exist_ok=False)
    source = ROOT / "customer_workspace/microbubble_tracking/windows_stall_capture.ps1"
    shutil.copyfile(source, host / source.name)
    if profile == "GpuLight":
        preset = source.with_name("windows_gpu_light.wprp")
        shutil.copyfile(preset, host / preset.name)
    elif profile == "Acpi":
        preset = ROOT / "customer_workspace/tools/windows_acpi_trace.wprp"
        shutil.copyfile(preset, host / preset.name)
    capture = host / "capture"
    limit_s = 180 if conditions else min(180, 40 + 50 * rounds)
    arguments = (
        f'-NoProfile -ExecutionPolicy Bypass -File "{windows_path(host / source.name)}" '
        f'-Output "{windows_path(capture)}" -DurationSeconds {limit_s} -ProfilePreset {profile}'
    )
    script = (
        "$ErrorActionPreference='Stop'; Start-Process powershell.exe -Verb RunAs "
        "-PassThru -ArgumentList '"
        + arguments.replace("'", "''")
        + "' | Select-Object -ExpandProperty Id"
    )
    command = [
        POWERSHELL,
        "-NoProfile",
        "-EncodedCommand",
        base64.b64encode(script.encode("utf-16le")).decode("ascii"),
    ]
    save_json(
        output / "capture_location.json",
        {
            "host": str(host),
            "capture": str(capture),
            "limit_s": limit_s,
            "profile": profile,
            "elevation_command": command,
        },
    )
    try:
        with (output / "elevation.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=log, timeout=90, check=True)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        # 若管理员窗口迟到，现存目录让录制脚本拒绝启动；若已启动则请求停止。
        capture.mkdir(exist_ok=True)
        (capture / "stop.request").touch()
        save_json(output / "launch_failure.json", {"error": str(error)})
        raise
    deadline = time.perf_counter() + 45
    while not (capture / "recording.json").exists():
        if (capture / "error.json").exists() or time.perf_counter() > deadline:
            raise RuntimeError(f"Host trace did not start; inspect {capture}")
        time.sleep(0.2)
    print(f"HOST RECORDING: automatic stop within {limit_s} seconds", flush=True)
    stop = threading.Event()
    clock = None
    if native_root is None:
        clock = threading.Thread(target=clock_bridge, args=(output, stop), daemon=True)
        clock.start()
    commands = []
    try:
        time.sleep(2)
        if conditions:
            native_output = native_root / "runs" / output.name
            cmd = [
                str(native_root / "python312/python.exe"),
                "-u",
                windows_path(ROOT / "customer_workspace/tools/gpu_acpi_conditions.py"),
                "--native-root",
                windows_path(native_root),
                "--output-dir",
                windows_path(native_output),
            ]
            save_json(output / "commands.json", [cmd])
            with (output / "conditions.log").open("x") as log:
                subprocess.run(cmd, stdout=log, stderr=log, timeout=170, check=True)
            for path in native_output.iterdir():
                if path.is_dir():
                    shutil.copytree(path, output / path.name)
                else:
                    shutil.copyfile(path, output / path.name)
            if (capture / "finished.json").exists():
                raise RuntimeError("Host trace stopped before conditions completed")
            return
        runs = [
            (video, video if rounds == 1 else f"{video}_r{repeat + 1}")
            for repeat in range(rounds)
            for video in ("A", "B")
        ]
        for video, name in runs:
            cmd = [
                "nsys",
                "profile",
                "--trace=cuda,nvvideo,osrt",
                "--sample=none",
                "--cpuctxsw=process-tree",
                "--osrt-backtrace-threshold=20000",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                f"--output={output}/profile_{name}",
                "--export=sqlite",
                str(ROOT / "customer_workspace/.venv_nvdec/bin/python"),
                "-u",
                "-m",
                "customer_workspace.tools.gpu_decode_diagnosis",
                "--video",
                str(
                    ROOT
                    / f"customer_workspace/microbubble_tracking/data/raw/exp{video}.mp4"
                ),
                "--frames",
                "1320",
                "--wait",
                "hybrid",
                "--profile",
                "--output-dir",
                str(output / f"traced_{name}"),
            ]
            native_output = None
            if native_root is not None:
                native_output = native_root / "runs" / f"{output.name}_{name}"
                cmd = [
                    str(native_root / "python312/python.exe"),
                    "-u",
                    "-m",
                    "customer_workspace.tools.gpu_native_decode",
                    "--video",
                    windows_path(native_root / f"videos/exp{video}.mp4"),
                    "--output-dir",
                    windows_path(native_output),
                ]
            commands.append(cmd)
            save_json(output / "commands.json", commands)
            print("START", name, flush=True)
            with (output / f"profile_{name}.log").open("x") as log:
                subprocess.run(
                    cmd, cwd=ROOT, stdout=log, stderr=log, timeout=65, check=True
                )
            if native_output is not None:
                shutil.copytree(native_output, output / f"traced_{name}")
            if (capture / "finished.json").exists():
                raise RuntimeError("Host trace stopped before both videos completed")
            print("DONE", name, flush=True)
    finally:
        (capture / "stop.request").touch()
        stop.set()
        if clock is not None:
            clock.join(timeout=10)
        deadline = time.perf_counter() + 90
        while not (capture / "finished.json").exists():
            if time.perf_counter() > deadline:
                raise RuntimeError(f"Check host trace shutdown in {capture}")
            time.sleep(0.25)
        state = json.loads((capture / "finished.json").read_text(encoding="utf-8-sig"))
        save_json(output / "host_finished.json", state)
        if state["stop_exit_code"]:
            raise RuntimeError("WPR could not save the host trace")
        print("HOST RECORDING SAVED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("Gpu", "GpuLight", "Acpi"), default="Gpu")
    parser.add_argument("--rounds", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--conditions", action="store_true", help="原生解码静置/单路/双路对照"
    )
    parser.add_argument(
        "--native-root", type=Path, help="显式指定 F 盘独立原生测试目录的 WSL 路径"
    )
    args = parser.parse_args()
    native_root = args.native_root.resolve() if args.native_root else None
    if native_root is not None and not (native_root / "python312/python.exe").is_file():
        parser.error("The native test Python executable does not exist")
    if args.conditions and (native_root is None or args.profile != "Acpi"):
        parser.error("--conditions requires --native-root and --profile Acpi")
    run(
        args.output_dir.resolve(),
        args.profile,
        args.rounds,
        native_root,
        args.conditions,
    )


if __name__ == "__main__":
    main()
