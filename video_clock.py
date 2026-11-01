"""同一 Windows 主机的时钟核对与一次实验身份；不转换 WSL 计时器。"""

import json
import os
import platform
import socket
import threading
import time
from pathlib import Path

from customer_workspace.streamcopy_recording import write_json


def read_ready(path: Path):
    """标记文件可能正写到一半；尚未完整时返回 None，由调用方限定等待时间。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def clock_identity(experiment_id: str) -> dict:
    return {
        "experiment_id": experiment_id,
        "host": platform.node(),
        "os": os.name,
        "pid": os.getpid(),
        "python": platform.python_version(),
        "clock": "windows_monotonic",
        "monotonic": vars(time.get_clock_info("monotonic")),
        "perf_counter": vars(time.get_clock_info("perf_counter")),
        "windows_s": time.monotonic(),
    }


def clock_interval(sample: dict) -> tuple[float, float]:
    """往返消息得到两时钟偏移的保守包络，不假定去程与回程耗时相等。"""
    return sample["b3"] - sample["a4"], sample["b2"] - sample["a1"]


class ClockServer:
    """仅监听 Windows 本机回环端口，回答时间核对；不接收仪器操作指令。"""

    def __init__(self, output: Path, experiment_id: str, health_provider=None):
        self.output, self.experiment_id = output, experiment_id
        self.stop_event = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(1)
        self.socket.settimeout(0.2)
        self.thread = threading.Thread(
            target=self._serve, name="video-clock", daemon=True
        )
        self.records = []
        self.health_provider = health_provider

    def start(self):
        identity = clock_identity(self.experiment_id)
        identity["port"] = self.socket.getsockname()[1]
        write_json(self.output / "camera_clock.json", identity)
        self.thread.start()
        return self

    def _serve(self):
        while not self.stop_event.is_set():
            try:
                connection, _ = self.socket.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with connection:
                connection.settimeout(1)
                try:
                    with connection.makefile("rb") as file:
                        request = json.loads(file.readline(4096))
                    received = time.monotonic()
                    if request["experiment_id"] != self.experiment_id:
                        raise ValueError("时间核对不属于本次实验")
                    reply = {
                        "experiment_id": self.experiment_id,
                        "b2": received,
                        "b3": time.monotonic(),
                    }
                    if request.get("kind") == "health" and self.health_provider:
                        reply["health"] = self.health_provider()
                    connection.sendall(json.dumps(reply).encode() + b"\n")
                    self.records.append(reply)
                except (OSError, ValueError, KeyError) as exc:
                    self.records.append({"error": str(exc)})

    def close(self):
        self.stop_event.set()
        self.socket.close()
        self.thread.join(2)
        write_json(self.output / "camera_clock_exchanges.json", self.records)


def read_health(camera: dict, experiment_id: str) -> dict:
    """本机消息读取实时状态，避免 Windows 共享文件的打开/替换锁冲突。"""
    with socket.create_connection(
        ("127.0.0.1", camera["port"]), timeout=0.5
    ) as connection:
        connection.sendall(
            json.dumps({"experiment_id": experiment_id, "kind": "health"}).encode()
            + b"\n"
        )
        with connection.makefile("rb") as file:
            reply = json.loads(file.readline(8192))
    if reply["experiment_id"] != experiment_id:
        raise ValueError("相机实时状态的实验身份不符")
    return reply["health"]


def probe_clock(camera: dict, experiment_id: str, count: int = 12) -> dict:
    """两端均须为同一 Windows 主机的 QPC 时钟；往返观测必须与零偏移相容。"""
    local = clock_identity(experiment_id)
    if any(camera[k] != local[k] for k in ("experiment_id", "host", "os", "clock")):
        raise ValueError("不是同一 Windows 主机、同一实验的时钟")
    if local["os"] != "nt" or any(
        not row["monotonic"]["monotonic"]
        or row["monotonic"]["adjustable"]
        or "QueryPerformanceCounter" not in row["monotonic"]["implementation"]
        for row in (camera, local)
    ):
        raise ValueError("本入口需要已验证的 Windows QueryPerformanceCounter 时钟")
    samples = []
    for _ in range(count):
        with socket.create_connection(
            ("127.0.0.1", camera["port"]), timeout=2
        ) as connection:
            start = time.monotonic()
            connection.sendall(
                json.dumps({"experiment_id": experiment_id}).encode() + b"\n"
            )
            with connection.makefile("rb") as file:
                reply = json.loads(file.readline(4096))
            end = time.monotonic()
        if reply["experiment_id"] != experiment_id:
            raise ValueError("时间核对回复身份不符")
        sample = {"a1": start, "b2": reply["b2"], "b3": reply["b3"], "a4": end}
        lower, upper = clock_interval(sample)
        if not lower <= 0 <= upper or not start <= reply["b2"] <= reply["b3"] <= end:
            raise ValueError("共同 Windows 时钟核对不通过，不能进行时间关联")
        samples.append(sample)
    best = min(samples, key=lambda r: r["a4"] - r["a1"])
    lower, upper = clock_interval(best)
    return {
        "identity": local,
        "samples": samples,
        "offset_interval_s": [lower, upper],
        "applied_offset_s": 0,
        "meaning": "同机Windows公共时钟；包络仅为软件核对，不是曝光同步误差",
    }
