"""诊断用确认消息与耗时分解；只比较同一系统的时间或两端各自的时长。"""

import json
import platform
import time


def clock_description():
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "monotonic": vars(time.get_clock_info("monotonic")),
        "perf_counter": vars(time.get_clock_info("perf_counter")),
    }


def send_ack(stream, lock, kind, seq, timestamps):
    """通过原 stdin 控制管道确认，不等待下一张视频帧，也不更换视频通道。"""
    with lock:
        row = {"kind": kind, "seq": seq, **timestamps}
        row["ack_sent_wsl_s"] = time.perf_counter()
        stream.write(b"ACK " + json.dumps(row, separators=(",", ":")).encode() + b"\n")


def parse_ack(command, received_windows_s):
    row = json.loads(command[4:])
    if row["kind"] not in ("received", "complete") or not isinstance(row["seq"], int):
        raise ValueError("Invalid latency acknowledgement")
    row["ack_received_windows_s"] = received_windows_s
    return row


def forward_bounds_ms(sent, received_ack):
    """单向应用层传输的因果界限；不假设两方向对称，不直接相减两端时钟。

    下界：WSL 收齐前缀后，读完剩余头和 H.264 数据的本机时长。
    上界：Windows 发送→收到确认，扣除 WSL 收齐→发确认的本机时长。
    上界仍含确认回程，因此只能报告范围，不能当成精确单向值。
    """
    lower = received_ack["arrived_wsl_s"] - received_ack["prefix_received_wsl_s"]
    roundtrip = received_ack["ack_received_windows_s"] - sent["write_start_windows_s"]
    preparation = received_ack["ack_sent_wsl_s"] - received_ack["arrived_wsl_s"]
    return lower * 1000, (roundtrip - preparation) * 1000


def feedback_components_ms(sent, completed_ack):
    """Windows 同钟总时间，减去 WSL 各自测得的时长；返回严格可加的组成。"""
    total = completed_ack["ack_received_windows_s"] - sent["received_windows_s"]
    before_send = sent["write_start_windows_s"] - sent["received_windows_s"]
    wsl_processing = completed_ack["completed_wsl_s"] - completed_ack["arrived_wsl_s"]
    ack_queue = completed_ack["ack_sent_wsl_s"] - completed_ack["completed_wsl_s"]
    return {
        "feedback_total_ms": total * 1000,
        "windows_before_send_ms": before_send * 1000,
        "wsl_arrival_to_result_ms": wsl_processing * 1000,
        "completion_ack_queue_ms": ack_queue * 1000,
        "forward_plus_ack_return_ms": (total - before_send - wsl_processing - ack_queue)
        * 1000,
    }
