"""用本机假服务验证模拟相机控制会话，不接触任何真实设备。"""

import socket
import threading

import pytest

from customer_workspace.streamcopy_recording import RecordingRtspControl
from microbubble_experiment.camera_rtsp import split_message


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/a",
        "rtsp://192.168.0.10/a",
        "rtsp://user:secret@127.0.0.1/a",
        "rtsp://127.0.0.1:99999/a",
    ],
)
def test_invalid_endpoint_fails_before_connecting(url):
    with pytest.raises(ValueError):
        RecordingRtspControl("127.0.0.1", rtsp_url=url)


def test_requests_keep_one_connection_and_record_real_responses():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5)
    port = listener.getsockname()[1]
    requests, errors = [], []

    def server():
        try:
            with listener.accept()[0] as peer:
                peer.settimeout(5)
                pending = b""
                for index in range(3):
                    while (message := split_message(pending)) is None:
                        chunk = peer.recv(8192)
                        if not chunk:
                            raise ConnectionError(
                                "Client incorrectly closed its session"
                            )
                        pending += chunk
                    request, pending = message
                    requests.append(request.split(b"\r\n")[0].decode())
                    # 第二条拒绝请求，确认日志保留真实错误而非统一写成功。
                    status = "404 Not Found" if index == 1 else "200 OK"
                    peer.sendall(
                        f"RTSP/1.0 {status}\r\nCSeq: {index + 1}\r\n\r\n".encode()
                    )
        except (OSError, ValueError) as exc:
            errors.append(repr(exc))

    worker = threading.Thread(target=server)
    worker.start()
    relay = RecordingRtspControl(
        "127.0.0.1", rtsp_url=f"rtsp://127.0.0.1:{port}/expA"
    ).start()
    try:
        assert relay.url.endswith("/expA")
        for index, method in enumerate(("OPTIONS", "DESCRIBE", "TEARDOWN"), 1):
            relay._camera_request(
                f"{method} {relay.url} RTSP/1.0\r\nCSeq: {index}\r\n\r\n".encode()
            )
        assert relay.stop_event.is_set()
        assert relay.exchanges[1]["response"] == "RTSP/1.0 404 Not Found"
        assert relay.exchanges[-1]["response"] == "RTSP/1.0 200 OK"
    finally:
        relay.close()
        worker.join(6)
        listener.close()
    assert not worker.is_alive()
    assert not errors
    assert len(requests) == 3
    assert all(f"rtsp://127.0.0.1:{port}/expA " in request for request in requests)
