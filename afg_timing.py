"""真实 AFG 的计时及实验专用调频/调幅；连接、核查和 STOP 复用原有 Rig。

发送时间取自调用 VISA write 之前；回复时间取自 VISA read/query 返回之后。
它们是电脑端时间，不是 USB 总线抓包时间或实际电压变化时间。
"""

from collections import deque
from dataclasses import dataclass, replace
from time import monotonic

from microbubble_experiment.afg import AFGError, ChannelSettings, Rig


class RoundUnreliable(AFGError):
    """边界核查发现错误或参数不符；保留整份核查结果。"""

    def __init__(self, report: dict):
        self.report = report
        super().__init__("本轮不可靠：" + "; ".join(report["problems"]))


@dataclass
class OutputTiming:
    """一次输出开关或参数写入的通信时间；失败或未收到回复时保留空值。"""

    command: str
    query: str | None
    command_sent: float | None = None
    write_returned: float | None = None
    state_reply_received: float | None = None
    state_reply: str | None = None
    succeeded: bool = False

    def relative_to(self, start_s: float, *, reply_name: str = "state") -> dict:
        """转换到实验的共同起点；只有驱动核查通过才给出确认时间。"""
        confirmed = self.state_reply_received if self.succeeded else None
        return {
            "command": self.command,
            "query": self.query,
            "command_sent_s": None
            if self.command_sent is None
            else self.command_sent - start_s,
            "command_write_returned_s": None
            if self.write_returned is None
            else self.write_returned - start_s,
            f"{reply_name}_reply_received_s": None
            if self.state_reply_received is None
            else self.state_reply_received - start_s,
            f"{reply_name}_reply": self.state_reply,
            "confirmed_s": None if confirmed is None else confirmed - start_s,
            "command_to_confirmation_s": None
            if confirmed is None or self.command_sent is None
            else confirmed - self.command_sent,
        }


class TimedSession:
    """转发已连接的真实 VISA 会话，只观察现有命令，不增加查询。

    原驱动连接时已设置超时和行结束符，此处仅包装它实际使用的通信方法。
    批量查询按原驱动的一问一行规则，把输出开关或幅值回复对应到查询命令。
    """

    def __init__(self, session):
        self._session = session
        self.resource_name = session.resource_name
        self.timing: OutputTiming | None = None
        self._queries: deque[str] = deque()
        self._capture_reply = False
        self.error_replies: list[str] = []
        self.io_counts = {
            "write_calls": 0,
            "read_calls": 0,
            "query_calls": 0,
            "query_commands": 0,
        }

    def write(self, command: str):
        timing = self.timing
        commands = command.split(";")
        self.io_counts["write_calls"] += 1
        self.io_counts["query_commands"] += sum(item.endswith("?") for item in commands)
        self._queries = deque(commands) if command.endswith("?") else deque()
        is_switch = (
            timing is not None
            and command == timing.command
            and timing.command_sent is None
        )
        if not command.endswith("?"):
            # 后续关断不能被误记成先前失败开关的确认。
            self._capture_reply = False
        if is_switch:
            timing.command_sent = monotonic()
        try:
            result = self._session.write(command)
        except Exception as exc:
            exc.add_note(f"AFG {self.resource_name}：VISA write 失败，命令：{command}")
            raise
        if is_switch:
            timing.write_returned = monotonic()
            self._capture_reply = True
        return result

    def read(self):
        self.io_counts["read_calls"] += 1
        try:
            result = self._session.read()
        except Exception as exc:
            command = self._queries[0] if self._queries else "未知查询"
            exc.add_note(
                f"AFG {self.resource_name}：VISA read 失败，等待回复：{command}"
            )
            raise
        received = monotonic()
        command = self._queries.popleft() if self._queries else None
        self._record_reply(command, result, received)
        return result

    def query(self, command: str):
        self.io_counts["query_calls"] += 1
        self.io_counts["query_commands"] += 1
        self._queries.clear()
        try:
            result = self._session.query(command)
        except Exception as exc:
            exc.add_note(f"AFG {self.resource_name}：VISA query 失败，命令：{command}")
            raise
        received = monotonic()
        self._record_reply(command, result, received)
        return result

    def _record_reply(self, command: str | None, result: str, received: float) -> None:
        if command is not None and command.upper() in (
            "SYSTEM:ERROR:NEXT?",
            "SYSTEM:ERROR?",
        ):
            self.error_replies.append(result.strip())
        timing = self.timing
        if timing is not None and self._capture_reply and command == timing.query:
            timing.state_reply_received = received
            timing.state_reply = result.strip()
            self._capture_reply = False

    def close(self):
        return self._session.close()


class TimedRig(Rig):
    """复用现有连接和核查；轮前核查后支持只发送频率或幅值命令。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.last_output_timing: OutputTiming | None = None
        self.last_amplitude_timing: OutputTiming | None = None
        self.last_frequency_timing: OutputTiming | None = None
        self._round_checked = False
        self._enabled_channels: set[str] = set()

    def connect(self):
        identities = super().connect()
        try:
            for alias, session in self.sessions.items():
                self.sessions[alias] = TimedSession(session)
        except BaseException:
            self._close_sessions()
            raise
        return identities

    def set_output(self, name, on: bool, *, return_state=False):
        with self.lock:
            self.last_output_timing = None
            session, channel = self._target(name)
            timing = OutputTiming(
                command=f"OUTPut{channel['channel']}:STATe {'ON' if on else 'OFF'}",
                query=f"OUTPut{channel['channel']}:STATe?",
            )
            self.last_output_timing = timing
            session.timing = timing
            try:
                result = super().set_output(name, on, return_state=return_state)
                if on:
                    self._enabled_channels.add(name)
                else:
                    self._enabled_channels.discard(name)
                timing.succeeded = True
                return result
            finally:
                session.timing = None

    def all_off(self) -> list[str]:
        """STOP 同时使只发送模式失效；下轮必须重新准备并核查。"""
        with self.lock:
            self._round_checked = False
            self._enabled_channels.clear()
            return super().all_off()

    def check_state(self, settings: dict[str, ChannelSettings], phase: str) -> dict:
        """在轮前或 STOP 后核查四路，保存错误回复；不证明中途没有瞬态变化。

        每路复用原驱动的 20 项查询和参数核验。轮末先逐台查错误队列，
        无错误时每台一次；有错误时原驱动最多读 65 次，直到零或报告未排空。
        两个检查阶段均预期输出 OFF，不能推迟 STOP 来做检查。
        """
        with self.lock:
            self._round_checked = False
            self._enabled_channels.clear()
            if phase not in ("before", "after") or set(settings) != set(
                self.config["channels"]
            ):
                raise ValueError("核查需要 before/after 阶段及完整四路设置")
            counts_before = {
                alias: dict(session.io_counts)
                for alias, session in self.sessions.items()
            }
            for session in self.sessions.values():
                session.error_replies.clear()
            report = {"phase": phase, "channels": {}, "problems": []}
            if phase == "after":
                for alias, session in self.sessions.items():
                    try:
                        self._errors(session)
                    except Exception as exc:  # noqa: BLE001 - 保存第三方通信错误，仍核查另一台设备。
                        report["problems"].append(
                            f"{alias} 错误队列：{type(exc).__name__}: {exc}"
                        )
            for name, target in settings.items():
                try:
                    actual = self.read_channel(
                        name, check_errors=True, check_peer_coupling=True
                    )
                    report["channels"][name] = actual
                    self._verify(name, target, actual)
                    if actual["output"] or actual["peer_concurrent"]:
                        raise AFGError("输出未关闭，或另一通道的频率联动未关闭")
                except Exception as exc:  # noqa: BLE001 - 保存该路错误，继续核查其余通道。
                    notes = "; ".join(getattr(exc, "__notes__", []))
                    report["problems"].append(
                        f"{name} 核查：{type(exc).__name__}: {exc} {notes}".strip()
                    )
            report["error_queue_replies"] = {
                alias: list(session.error_replies)
                for alias, session in self.sessions.items()
            }
            report["io_counts"] = {
                alias: {
                    key: value - counts_before[alias][key]
                    for key, value in session.io_counts.items()
                }
                for alias, session in self.sessions.items()
            }
            report["passed"] = not report["problems"]
            self._round_checked = phase == "before" and report["passed"]
            return report

    def send_amplitude(self, name: str, amplitude_vpp: float) -> dict:
        """只写一条幅值命令，不查询；返回发送信息，不宣称设备已确认。"""
        return self._send_parameter(name, "amplitude", amplitude_vpp)

    def send_frequency(self, name: str, frequency_hz: float) -> dict:
        """只写一条连续波频率命令，不查询；沿用原驱动的频率写法和限制。"""
        return self._send_parameter(name, "frequency", frequency_hz)

    def _send_parameter(self, name: str, parameter: str, value: float) -> dict:
        """共用本地保护、一次写入和时间记录，不添加通信或更新已核查缓存。

        prepared 仅用于确认准备模式和检查参数上限，不代表中途的实际设置。
        """
        with self.lock:
            if parameter == "frequency":
                self.last_frequency_timing = None
                field, header, unit = "frequency_hz", "FREQuency:FIXed", "Hz"
                result_key = "frequency_requested_hz"
            elif parameter == "amplitude":
                self.last_amplitude_timing = None
                field = "amplitude_vpp"
                header, unit = "VOLTage:LEVel:IMMediate:AMPLitude", "Vpp"
                result_key = "amplitude_requested_vpp"
            else:
                raise ValueError("只支持 frequency 或 amplitude")
            if not self._round_checked or self._enabled_channels != set(
                self.config["channels"]
            ):
                raise AFGError("只发送参数需要轮前核查通过且四路已开启")
            session, channel = self._target(name)
            previous = self.prepared.get(name)
            if previous is None or previous.mode != "continuous":
                raise AFGError("只发送参数需要已准备的连续正弦波")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("频率或幅值必须为数值")
            replace(previous, **{field: value}).validate(channel)
            command = f"SOURce{channel['channel']}:{header} {value:.12g}{unit}"
            timing = OutputTiming(command=command, query=None)
            if parameter == "frequency":
                self.last_frequency_timing = timing
            else:
                self.last_amplitude_timing = timing
            counts_before = dict(session.io_counts)
            session.timing = timing
            try:
                self.touched = True
                self._write(session, command)
                timing.succeeded = True
                return {
                    result_key: value,
                    "device_confirmed": False,
                    "io_delta": {
                        key: value - counts_before[key]
                        for key, value in session.io_counts.items()
                    },
                }
            except BaseException:
                self.all_off()
                raise
            finally:
                session.timing = None

    def apply_live(self, name: str, settings: ChannelSettings) -> dict:
        """观察原驱动的幅值写入和幅值读回，不把较早的输出 ON 回复当成确认。"""
        with self.lock:
            self.last_amplitude_timing = None
            session, channel = self._target(name)
            amplitude = f"SOURce{channel['channel']}:VOLTage:LEVel:IMMediate:AMPLitude"
            timing = OutputTiming(
                command=f"{amplitude} {settings.amplitude_vpp:.12g}Vpp",
                query=f"{amplitude}?",
            )
            self.last_amplitude_timing = timing
            session.timing = timing
            try:
                result = super().apply_live(name, settings)
                timing.succeeded = True
                return result
            finally:
                session.timing = None
