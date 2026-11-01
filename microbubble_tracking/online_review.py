"""第 1 轮结果的只读可视化：复用已保存轮廓，不重新检测或追踪。"""

import argparse
import csv
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Patch
from matplotlib.text import Text
from PIL import ImageFont

if __package__:
    from .online_replay import file_hash
    from .online_review_render import render_video, write_gallery
else:
    from online_replay import file_hash
    from online_review_render import render_video, write_gallery


TEAL = "#087F8C"
AMBER = "#D59316"
PURPLE = "#8055A2"
INK = "#233443"
GRAY = "#CDD5DC"


class ReviewRun:
    """只读已关闭的数据库；每次只取一张画面的轮廓。"""

    def __init__(self, directory: Path, label: str):
        self.directory, self.label = directory.resolve(), label
        database = self.directory / "tracking.sqlite"
        wal = database.with_name(database.name + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise ValueError("Close the tracking writer before rendering its records")
        self.db = sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)
        self.db.row_factory = sqlite3.Row
        self.metadata = {
            k: json.loads(v)
            for k, v in self.db.execute("SELECT key,value FROM session")
        }
        if self.metadata["status"] != "completed":
            raise ValueError("This review requires a completed replay")
        session = json.loads((self.directory / "session.json").read_text())
        for key in ("experiment_id", "stream_id", "run_id"):
            if session[key] != self.metadata[key]:
                raise ValueError(f"Session identity mismatch: {key}")
        provenance = self.metadata["provenance"]
        self.video = Path(provenance["video"])
        if file_hash(self.video) != provenance["video_sha256"]:
            raise ValueError("The raw video no longer matches these tracking records")
        self.hashes = {
            str(database): file_hash(database),
            str(self.video): file_hash(self.video),
        }
        for original in ("actions.csv", "video_frames.csv"):
            path = self.directory / ("source_" + original)
            actual = file_hash(path)
            if actual != provenance["synchronization_hashes"][original]:
                raise ValueError(f"Changed source timing record: {path}")
            self.hashes[str(path)] = actual
        with (self.directory / "source_video_frames.csv").open(
            encoding="utf-8-sig"
        ) as f:
            self.recording = list(csv.DictReader(f))
        with (self.directory / "source_actions.csv").open(encoding="utf-8-sig") as f:
            self.actions = list(csv.DictReader(f))
        if any(
            r["experiment_id"] != self.metadata["experiment_id"]
            for r in self.recording + self.actions
        ):
            raise ValueError("Mixed experiment identities in source records")
        self.times = np.array([float(r["video_s"]) for r in self.recording])
        self.duration = self.times[-1] + float(np.median(np.diff(self.times)))
        self.frames = {
            r["video_frame"]: dict(r) for r in self.db.execute("SELECT * FROM frames")
        }
        for index, row in self.frames.items():
            original = self.recording[index]
            if (
                row["source_frame"] != int(original["source_seq"])
                or abs(row["video_time_s"] - float(original["video_s"])) > 1e-9
            ):
                raise ValueError("Tracking frame identity disagrees with the recording")
        self.source = {r["source_frame"]: r for r in self.frames.values()}
        self.events = {
            r["event_id"]: dict(r) for r in self.db.execute("SELECT * FROM events")
        }
        self.updates = {}
        for row in self.db.execute("SELECT * FROM event_updates ORDER BY sequence"):
            self.updates.setdefault(row["event_id"], []).append(dict(row))
        self.report = json.loads((self.directory / "replay_report.json").read_text())

    def observations(self, video_frame: int) -> list[dict]:
        frame = self.frames.get(video_frame)
        if frame is None:
            return []
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT * FROM observations WHERE source_frame=? ORDER BY track_id",
                (frame["source_frame"],),
            )
        ]

    def event_status(self, event_id: int, source_frame: int) -> str | None:
        """回到当时的判断，不提前使用数据库中的最终确认结果。"""
        status = None
        for update in self.updates[event_id]:
            if update["reported_frame"] > source_frame:
                break
            status = update["status"]
        return status

    def pzt_intervals(self):
        """按已存发送范围绘制命令时间轴；不是实际声场测量。"""
        intervals = []
        for channel in ("PZT1", "PZT2", "PZT3", "PZT4"):
            start, state = 0.0, "IDLE"
            actions = [
                a
                for a in self.actions
                if a["channel"] == channel and a["kind"] != "stop_repeat"
            ]
            actions.sort(key=lambda a: float(a["command_windows_s"]))
            for action in actions:
                if action["command_video_mapped"].lower() != "true":
                    raise ValueError("Cannot draw an unmapped PZT command")
                low = float(action["command_video_video_from_s"])
                high = float(action["command_video_video_to_s"])
                intervals.extend(
                    [(channel, start, low, state), (channel, low, high, "SWITCHING")]
                )
                start, state = high, action["state"]
            intervals.append((channel, start, self.duration, state))
        return intervals

    def export_events(self, path: Path):
        fields = [
            "experiment_id",
            "run_id",
            "event_id",
            "kind",
            "participants",
            "anchor_video_frame",
            "reported_video_frame",
            "decided_video_frame",
            "anchor_video_s",
            "reported_video_s",
            "decided_video_s",
            "interval_from_s",
            "interval_to_s",
            "evidence_gap",
            "final_status",
        ]
        with path.open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fields)
            writer.writeheader()
            for event in self.events.values():
                row = {
                    k: event[k]
                    for k in (
                        "event_id",
                        "kind",
                        "participants",
                        "interval_from_s",
                        "interval_to_s",
                        "evidence_gap",
                    )
                }
                row.update(
                    experiment_id=self.metadata["experiment_id"],
                    run_id=self.metadata["run_id"],
                    final_status=event["status"],
                )
                for key in ("anchor", "reported", "decided"):
                    frame = self.source[event[key + "_frame"]]
                    row[key + "_video_frame"] = frame["video_frame"]
                    row[key + "_video_s"] = frame["video_time_s"]
                writer.writerow(row)

    def close(self):
        self.db.close()


def configure_arial(font: Path):
    if ImageFont.truetype(str(font), 20).getname()[0] != "Arial":
        raise ValueError("Actual Arial is required; substitutes are not accepted")
    fontManager.addfont(str(font))
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig, output: Path, stem: str, font: Path):
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        text.set_fontproperties(
            FontProperties(fname=str(font), size=text.get_fontsize())
        )
    fig.savefig(output / (stem + ".png"), dpi=180)
    fig.savefig(output / (stem + ".pdf"))
    plt.close(fig)


def make_charts(a: ReviewRun, b: ReviewRun, drop: ReviewRun, output: Path, font: Path):
    """只绘制记录中的计时、生命周期和数量；不推算实际内存或物理同步误差。"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.3), layout="constrained")
    for run, color in [(a, TEAL), (b, PURPLE)]:
        frames = list(run.frames.values())[1:]
        axes[0].plot(
            [r["video_time_s"] for r in frames],
            [r["delta_time_s"] * 1000 for r in frames],
            color=color,
            lw=1,
            label=run.label,
        )
    axes[0].set(
        title="Recorded intervals: complete input",
        ylabel="Frame interval (ms)",
        ylim=(0, 37),
    )
    axes[0].legend(loc="upper right")
    frames = list(drop.frames.values())[1:]
    axes[1].scatter(
        [r["video_time_s"] for r in frames],
        [r["delta_time_s"] * 1000 for r in frames],
        s=3,
        color=AMBER,
    )
    axes[1].set(
        title="Exp. A: every third frame withheld",
        ylabel="Frame interval (ms)",
        ylim=(0, 37),
    )
    choose = np.where((a.times >= 3.0) & (a.times <= 3.2))[0]
    omitted = [i for i in choose if i not in drop.frames]
    kept = [i for i in choose if i in drop.frames]
    axes[2].scatter(
        a.times[kept],
        [int(a.recording[i]["source_seq"]) for i in kept],
        s=35,
        color=TEAL,
        label="Processed",
    )
    axes[2].scatter(
        a.times[omitted],
        [int(a.recording[i]["source_seq"]) for i in omitted],
        s=45,
        marker="x",
        color=AMBER,
        label="Withheld",
    )
    axes[2].set(
        title="Source numbers are preserved",
        ylabel="Source frame",
        xlabel="Video time (s)",
    )
    axes[2].legend(loc="upper left", ncol=2)
    for ax in axes:
        ax.grid(axis="y", color="#E5E9ED", lw=0.6)
    save_figure(fig, output, "01_frame_timing", font)

    fig, axes = plt.subplots(1, 2, figsize=(13, 8), layout="constrained")
    for ax, run in zip(axes, [a, b]):
        count = run.report["tracks"]
        matrix = np.zeros((count, len(run.recording)), np.uint8)
        for row in run.db.execute(
            "SELECT source_frame,track_id,status_at_frame FROM observations"
        ):
            frame = run.source[row["source_frame"]]["video_frame"]
            matrix[row["track_id"] - 1, frame] = (
                1 if row["status_at_frame"] == "tentative" else 2
            )
        ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            extent=(0, run.duration, 0.5, count + 0.5),
            cmap=ListedColormap(["white", AMBER, TEAL]),
            vmin=0,
            vmax=2,
        )
        ax.set(title=run.label, xlabel="Video time (s)", ylabel="Stable track ID")
        ax.set_yticks([1, *range(25, count + 1, 25)])
    fig.legend(
        handles=[
            Patch(color=AMBER, label="Provisional"),
            Patch(color=TEAL, label="Confirmed track"),
            Patch(facecolor="white", edgecolor=GRAY, label="Not observed"),
        ],
        loc="outside upper center",
        ncol=3,
    )
    save_figure(fig, output, "02_track_lifetimes", font)

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 7.1), sharex="col", layout="constrained"
    )
    for column, run in enumerate([a, b]):
        frames = list(run.frames.values())
        counts = dict(
            run.db.execute(
                "SELECT source_frame,count(*) FROM observations GROUP BY source_frame"
            )
        )
        times = [r["video_time_s"] for r in frames]
        axes[0, column].plot(
            times, [r["active_tracks"] for r in frames], color=TEAL, lw=1.4
        )
        axes[0, column].set(
            title=run.label, ylabel="Targets in active state", ylim=(0, 30)
        )
        cumulative = np.cumsum([counts.get(r["source_frame"], 0) for r in frames])
        axes[1, column].plot(times, cumulative, color=PURPLE, lw=1.5)
        axes[1, column].set(
            ylabel="Observations saved to disk",
            xlabel="Video time (s)",
            ylim=(0, 24000),
        )
        for row in range(2):
            axes[row, column].grid(axis="y", color="#E5E9ED", lw=0.6)
    save_figure(fig, output, "03_history_and_active_targets", font)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8.5), layout="constrained")
    for ax, run in zip(axes, [a, b]):
        labels = []
        for i, action in enumerate(run.actions):
            labels.append(
                f"{i + 1:02d}  {action['channel']}  {action['state'].title()}"
            )
            centers = []
            for kind, color, marker in [
                ("command", TEAL, "o"),
                ("confirmation", PURPLE, "s"),
            ]:
                low = float(action[kind + "_video_video_from_s"])
                high = float(action[kind + "_video_video_to_s"])
                centers.append((low + high) / 2)
                ax.errorbar(
                    (low + high) / 2,
                    i,
                    xerr=(high - low) / 2,
                    fmt=marker,
                    color=color,
                    capsize=3,
                    ms=4,
                    lw=1,
                )
            ax.plot(centers, [i, i], color=GRAY, ls=":", zorder=0)
        ax.set_yticks(range(len(labels)), labels)
        ax.set(
            title=run.label,
            xlabel="Mapped video time (s)",
            xlim=(0, 22),
            ylim=(len(labels) - 0.5, -0.5),
        )
        ax.grid(axis="x", color="#E5E9ED", lw=0.6)
    axes[0].plot([], [], "o", color=TEAL, label="Command range")
    axes[0].plot([], [], "s", color=PURPLE, label="Readback range")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.07), ncol=2)
    save_figure(fig, output, "04_pzt_video_alignment", font)


def choose_events(a, b):
    """按记录选取较可见的实例；这些只是审核候选，不能当成人工标准答案。"""

    def area(run, event, child=False):
        ids = json.loads(event["participants"])
        tid = ids[-1] if child else event["result_track"]
        row = run.db.execute(
            "SELECT max(area) FROM observations WHERE track_id=? AND source_frame BETWEEN ? AND ?",
            (tid, event["anchor_frame"] - 2, event["anchor_frame"]),
        ).fetchone()
        return row[0] or 0

    merge = max(
        (
            e
            for e in a.events.values()
            if e["kind"] == "merge" and e["status"] == "confirmed_tracks"
        ),
        key=lambda e: area(a, e),
    )
    split = max(
        (
            e
            for e in b.events.values()
            if e["kind"] == "split" and e["status"] == "confirmed_tracks"
        ),
        key=lambda e: area(b, e, True),
    )
    lost = max(
        (
            e
            for e in a.events.values()
            if e["kind"] == "lost" and e["status"] == "confirmed_tracks"
        ),
        key=lambda e: area(a, e),
    )
    unresolved = next(
        e for e in a.events.values() if e["status"] == "unresolved_at_end"
    )
    return [
        (a, merge, "04_merge_candidate", "Merge candidate"),
        (b, split, "05_split_confirmation", "Split candidate"),
        (a, lost, "06_tracking_loss", "Tracking loss"),
        (a, unresolved, "07_unresolved_target", "Unresolved target"),
    ]


def generate(source: Path, output: Path, font: Path):
    configure_arial(font)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("videos", "charts", "data", "assets", "previews"):
        (output / name).mkdir()
    shutil.copyfile(font, output / "assets/arial.ttf")
    runs = []
    try:
        for folder, label in [
            ("final_A", "Exp. A"),
            ("final_B", "Exp. B"),
            ("final_A_drop3", "Exp. A: withheld frames"),
        ]:
            runs.append(ReviewRun(source / folder, label))
        a, b, drop = runs
        if a.metadata["stream_id"] != drop.metadata["stream_id"]:
            raise ValueError("The skipped-frame comparison uses a different video")
        videos = []
        for run, stem in [(a, "01_expA_full_review"), (b, "02_expB_full_review")]:
            videos.append(
                render_video(
                    run,
                    output,
                    stem,
                    font,
                    title=run.label,
                    start_s=0,
                    end_s=run.duration,
                )
            )
            run.export_events(output / "data" / f"{run.label[-1]}_events.csv")
            shutil.copyfile(
                run.directory / "action_links.csv",
                output / "data" / f"{run.label[-1]}_action_links.csv",
            )
        videos.append(
            render_video(
                a,
                output,
                "03_skipped_frame_comparison",
                font,
                title="Withheld-frame check",
                start_s=2.7,
                end_s=4.7,
                speed=0.25,
                comparison=drop,
            )
        )
        selected = []
        for run, event, stem, title in choose_events(a, b):
            anchor_s = run.source[event["anchor_frame"]]["video_time_s"]
            reported_s = run.source[event["reported_frame"]]["video_time_s"]
            start, end = max(0, anchor_s - 0.6), min(run.duration, reported_s + 0.7)
            roi = (
                int(np.clip(event["x"] - 200, 0, 880)),
                int(np.clip(event["y"] - 160, 0, 704)),
                400,
                320,
            )
            videos.append(
                render_video(
                    run,
                    output,
                    stem,
                    font,
                    title=f"{run.label} | {title}",
                    start_s=start,
                    end_s=end,
                    speed=0.2,
                    roi=roi,
                    event=event,
                )
            )
            selected.append(
                dict(
                    experiment=run.label,
                    event=event,
                    crop_xywh=roi,
                    start_s=start,
                    end_s=end,
                    video=stem + ".mp4",
                )
            )
        make_charts(a, b, drop, output / "charts", font)
        hashes = {}
        for run in runs:
            hashes.update(run.hashes)
        if any(file_hash(Path(path)) != value for path, value in hashes.items()):
            raise ValueError("A source changed during rendering")
        manifest = dict(
            source=str(source.resolve()),
            font=str(font),
            font_family="Arial",
            font_sha256=file_hash(font),
            inputs=hashes,
            videos=videos,
            selected_events=selected,
            hardware_access=False,
            contours="Read directly from existing SQLite records; no new tracking run",
            status="Status at each source frame, not final hindsight confirmation",
            memory_plot="Active target count, not RAM measurements",
        )
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        write_gallery(output, videos)
        print(f"Review package: {output.resolve()}", flush=True)
    finally:
        for run in runs:
            run.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Round 1 validation directory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--font", type=Path, default=Path("/mnt/c/Windows/Fonts/arial.ttf")
    )
    args = parser.parse_args()
    output = args.output_dir or args.source.parent / (
        "online_review_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    generate(args.source, output, args.font)


if __name__ == "__main__":
    main()
