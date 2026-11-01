"""离线记录 GC 事件；normal 观察正式流程，其余为历史诊断选项。"""

import argparse
import csv
import gc
import json
import platform
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

if __package__:
    from .bubble_tracker import video_info
    from .paced_replay import paced_replay
else:
    from bubble_tracker import video_info
    from paced_replay import paced_replay


def object_summary(objects):
    """只在独立对象审计中扫描，返回类型/函数来源，不保留对象或执行 repr。"""
    types = Counter()
    functions = Counter()
    for obj in objects:
        cls = type(obj)
        types[f"{cls.__module__}.{cls.__qualname__}"] += 1
        if cls.__name__ == "function":
            code = obj.__code__
            functions[f"{code.co_filename}:{code.co_firstlineno}:{code.co_name}"] += 1
    return {
        "count": sum(types.values()),
        "types": types.most_common(30),
        "functions": functions.most_common(20),
    }


class GCTrial:
    """有明确起止的进程级对照；不关闭自动 GC，不改变送帧子进程。

    normal 不额外清理；正式模块自行执行开始前和保存后的完整清理。
    precollect 和 defer_full 是历史诊断选项，会额外执行清理；后者还提高
    最老一代的触发门槛。两者都不是当前正式策略的验证入口。
    必须在送帧开始前 begin_stream，保存线程退出后 finish。外层 with
    提供异常兜底。正式默认入口不创建此对象，也不安装它的回调。
    """

    def __init__(self, output, mode="normal", *, audit_objects=False):
        if mode not in ("normal", "precollect", "defer_full"):
            raise ValueError("Unknown GC trial mode")
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError(
                "This GC threshold trial is validated for Python 3.12 only"
            )
        self.output = Path(output)
        self.mode, self.audit = mode, audit_objects
        self.rows = deque(maxlen=20000)
        self.overflow = 0
        self.phase = "preparation"
        self.started = self.closed = False
        self.current = None
        self.original_thresholds = gc.get_threshold()
        self.original_enabled = gc.isenabled()
        self.original_debug = gc.get_debug()
        self.garbage_start = len(gc.garbage)
        self.report = {
            "mode": mode,
            "python": platform.python_version(),
            "audit_objects": audit_objects,
            "original_thresholds": self.original_thresholds,
            "original_enabled": self.original_enabled,
        }

    def _callback(self, phase, info):
        # 仅在实际发生 GC 时调用；不扫描线程、对象列表或逐帧写文件。
        if phase == "start":
            stack = []
            if info["generation"] == 2 or self.audit:
                frame = sys._getframe(1)
                try:
                    for _ in range(6):
                        if frame is None:
                            break
                        stack.append(
                            f"{frame.f_code.co_filename}:{frame.f_lineno}:"
                            f"{frame.f_code.co_name}"
                        )
                        frame = frame.f_back
                finally:
                    del frame  # 只存字符串，不能把运行栈留到以后制造新的引用环。
            self.current = (
                self.phase,
                threading.get_native_id(),
                info["generation"],
                " | ".join(stack),
                time.perf_counter(),
                time.thread_time(),
            )
        elif self.current is not None:
            ended, cpu = time.perf_counter(), time.thread_time()
            phase_name, thread, generation, stack, started, old_cpu = self.current
            self.current = None
            if len(self.rows) == self.rows.maxlen:
                self.overflow += 1
            self.rows.append(
                (
                    phase_name,
                    thread,
                    generation,
                    started,
                    ended,
                    (ended - started) * 1000,
                    (cpu - old_cpu) * 1000,
                    info["collected"],
                    info["uncollectable"],
                    stack,
                )
            )

    def __enter__(self):
        if not gc.isenabled():
            raise RuntimeError("GC must initially be enabled for this comparison")
        if gc.get_debug() & gc.DEBUG_SAVEALL:
            raise RuntimeError("An existing SAVEALL audit must be finished first")
        gc.callbacks.append(self._callback)
        return self

    def begin_stream(self):
        if self.started or self.closed:
            raise RuntimeError("GC trial already started or closed")
        if self.audit:
            self.report["tracked_before"] = object_summary(gc.get_objects())
            self.report["generation_sizes_before"] = [
                len(gc.get_objects(g)) for g in range(3)
            ]
        if self.mode != "normal" or self.audit:
            self.phase = "pre_stream_collection"
            self.report["precollected"] = gc.collect(2)
        if self.mode == "defer_full":
            first, second, _ = self.original_thresholds
            gc.set_threshold(first, second, 1_000_000)
        if self.audit:
            gc.set_debug(self.original_debug | gc.DEBUG_SAVEALL)
        self.report["stream_thresholds"] = gc.get_threshold()
        self.report["stream_started_s"] = time.perf_counter()
        self.phase = "stream"
        self.started = True

    def finish(self):
        """先恢复设置，再做收尾清理；任何异常都不能把临时设置遗留到下一轮。"""
        if self.closed:
            return
        self.closed = True
        self.report["stream_ended_s"] = time.perf_counter()
        self.phase = "shutdown"
        try:
            gc.set_threshold(*self.original_thresholds)
            if self.audit:
                gc.collect(2)
                self.report["unreachable_objects"] = object_summary(
                    gc.garbage[self.garbage_start :]
                )
                self.report["tracked_after"] = object_summary(gc.get_objects())
                gc.set_debug(self.original_debug)
                del gc.garbage[self.garbage_start :]
            if self.mode != "normal" or self.audit:
                self.report["final_collected"] = gc.collect(2)
        finally:
            gc.set_threshold(*self.original_thresholds)
            gc.set_debug(self.original_debug)
            if self.audit:
                # 审计本身出错也要释放本轮暂存的对象，不能遗留 SAVEALL 数据。
                del gc.garbage[self.garbage_start :]
            if self.original_enabled:
                gc.enable()
            else:
                gc.disable()
            if self._callback in gc.callbacks:
                gc.callbacks.remove(self._callback)
            self.report.update(
                restored_thresholds=gc.get_threshold(),
                restored_enabled=gc.isenabled(),
                overflow_events=self.overflow,
                garbage_after=len(gc.garbage),
            )
            with (self.output / "gc_events.csv").open("x", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "phase",
                        "thread_id",
                        "generation",
                        "start_s",
                        "end_s",
                        "wall_ms",
                        "thread_cpu_ms",
                        "collected",
                        "uncollectable",
                        "trigger_stack",
                    ]
                )
                writer.writerows(self.rows)
            (self.output / "gc_report.json").write_text(
                json.dumps(self.report, indent=2) + "\n"
            )

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.finish()
        except Exception as close_error:
            if exc is None:
                raise
            exc.add_note(f"Also failed to close GC trial: {close_error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("normal", "precollect", "defer_full"), default="normal"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--video-name", choices=("expA", "expB"), default="expA")
    parser.add_argument("--audit-objects", action="store_true")
    parser.add_argument("--cancel-after-s", type=float, default=0)
    args = parser.parse_args()
    code = Path(__file__).resolve().parent
    video = code / "data/raw" / (args.video_name + ".mp4")
    fps, frames = video_info(str(video))
    if args.repeats < 1 or args.repeats * frames / fps > 360:
        parser.error("Use a positive replay duration no longer than 360 seconds")
    if args.audit_objects and args.repeats > 1:
        parser.error("Object retention audit is limited to one video; not a speed test")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with GCTrial(args.output_dir, args.mode, audit_objects=args.audit_objects) as trial:
        paced_replay(
            video,
            code / "results/B_offline/background.npy",
            code / "data/pzt" / args.video_name,
            args.output_dir / "replay",
            repeats=args.repeats,
            opencv_threads=24,
            conversion_threads=1,
            cancel_after_s=args.cancel_after_s,
            gc_control=trial,
        )


if __name__ == "__main__":
    main()
