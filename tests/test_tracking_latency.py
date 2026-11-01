"""不同起点、非对称往返的计时测试；不能用 RTT/2 伪造单向时间。"""

import io
import threading

import pytest

from customer_workspace.tracking_latency import (
    feedback_components_ms,
    forward_bounds_ms,
    parse_ack,
    send_ack,
)


def test_forward_bounds_use_local_durations_and_allow_asymmetry():
    # 去程 3 ms，回程 8 ms，WSL 确认准备 1 ms；两个时钟起点完全不同。
    sent = {"write_start_windows_s": 100.0}
    ack = {
        "arrived_wsl_s": 9000.003,
        "prefix_received_wsl_s": 9000.001,
        "ack_sent_wsl_s": 9000.004,
        "ack_received_windows_s": 100.012,
    }
    lower, upper = forward_bounds_ms(sent, ack)
    assert lower == pytest.approx(2)
    assert upper == pytest.approx(11)
    assert lower <= 3 <= upper


def test_feedback_is_same_windows_clock_and_components_add_up():
    sent = {"received_windows_s": 100.0, "write_start_windows_s": 100.002}
    ack = {
        "arrived_wsl_s": 9000.005,
        "completed_wsl_s": 9000.020,
        "ack_sent_wsl_s": 9000.021,
        "ack_received_windows_s": 100.029,
    }
    result = feedback_components_ms(sent, ack)
    assert result["feedback_total_ms"] == pytest.approx(29)
    assert result["windows_before_send_ms"] == pytest.approx(2)
    assert result["wsl_arrival_to_result_ms"] == pytest.approx(15)
    assert result["completion_ack_queue_ms"] == pytest.approx(1)
    assert result["forward_plus_ack_return_ms"] == pytest.approx(11)
    assert sum(
        value for key, value in result.items() if key != "feedback_total_ms"
    ) == pytest.approx(29)


def test_ack_preserves_matching_frame_and_separate_host_timestamps(monkeypatch):
    monkeypatch.setattr(
        "customer_workspace.tracking_latency.time.perf_counter", lambda: 9012.0
    )
    stream = io.BytesIO()
    send_ack(stream, threading.Lock(), "complete", 87, {"completed_wsl_s": 9011.0})
    message = stream.getvalue()
    assert message.startswith(b"ACK ") and message.endswith(b"\n")
    received = parse_ack(message, 5.0)
    assert received == {
        "kind": "complete",
        "seq": 87,
        "completed_wsl_s": 9011.0,
        "ack_sent_wsl_s": 9012.0,
        "ack_received_windows_s": 5.0,
    }


@pytest.mark.parametrize(
    "message", [b'ACK {"kind":"unknown","seq":1}', b'ACK {"kind":"complete","seq":"1"}']
)
def test_invalid_ack_is_rejected(message):
    with pytest.raises(ValueError):
        parse_ack(message, 1.0)
