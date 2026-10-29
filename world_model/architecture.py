#!/usr/bin/env python3
"""绘制当前世界动作模型的架构，不运行训练或实验。

从项目根目录运行：
    ../../.venv/bin/python -m world_model.architecture

若需要同时核对已保存的模型，使用装有 PyTorch 的环境：
    ../.venv_baseline312/bin/python -m world_model.architecture --verify-checkpoint

默认写入 figures/world_model_architecture_refined。目录已存在时拒绝覆盖；可用
--output-dir 指定另一个目录。所有文字使用真实 Arial；SVG 将文字保存为
轮廓，PDF 嵌入字体，避免换电脑后字体被替换。可用 --palette 更换用户提供的
六组色卡，用 --palette-sheet 同时生成对照预览。
"""

import argparse
import ast
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = ROOT / "world_model/models.py"
DATA_SOURCE = ROOT / "world_model/dataset.py"
INK = "#292321"
MUTED = "#6A6460"
LINE = "#BFBAB6"
NEUTRAL = "#EFEFEF"
WIDTH, HEIGHT = 20, 10.5
# 色值直接来自用户上传的色卡；重复的薄荷绿色保留在 swatches 中。
PALETTES = {
    "mint_gold": ("#803028", "#E1C48C", "#C8E0D8", "#C8E0D8", "#DAEDE7"),
    "rose_yellow": ("#F7CD17", "#F24774", "#F97896", "#F799A7", "#FCB895"),
    "lime": ("#F3F2B2", "#C2CF87", "#D5C943", "#829E63", "#CED35B"),
    "burgundy_cream": ("#7F1E0E", "#B75B66", "#E3B75E", "#C8A8A8", "#F2E4A3"),
    "ice_red": ("#E1F5F3", "#FFB50A", "#E1D0D6", "#D80C0D", "#4B4B41"),
    "coral": ("#FAE8E4", "#F0D0D1", "#E2845E", "#FB6A71", "#FB565A"),
}
# 同一绘图角色始终使用同一颜色，不把不同色卡随意混为一组。
PALETTE_ROLES = {
    "mint_gold": (0, 2, 1, 4, 3),
    "rose_yellow": (1, 2, 0, 3, 4),
    "lime": (3, 1, 2, 0, 4),
    "burgundy_cream": (0, 3, 2, 4, 1),
    "ice_red": (3, 0, 1, 2, 4),
    "coral": (4, 1, 2, 0, 3),
}


def sha256(path):
    """记录来源文件内容，供后续追溯；不改变原文件。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def argument_defaults(function):
    """只解析 Python 源码中的字面默认值，不导入或执行训练模块。"""
    names = [arg.arg for arg in function.args.args]
    return {
        name: ast.literal_eval(value)
        for name, value in zip(names[-len(function.args.defaults):], function.args.defaults)
    }


def read_architecture():
    """从当前源代码读取尺寸及默认配置，并检查本图适用的模型版本。"""
    tree = ast.parse(MODEL_SOURCE.read_text(encoding="utf-8"))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"NODE_IN", "EDGE_IN", "GLOBAL_IN", "FOURIER_CYCLES", "POS_SCALE"}:
                constants[name] = ast.literal_eval(node.value)
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GraphWorldModel")
    init = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    defaults = argument_defaults(init)
    data_tree = ast.parse(DATA_SOURCE.read_text(encoding="utf-8"))
    data_node = next(
        n for n in data_tree.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "DEFAULTS"
    )
    data = {item.arg: ast.literal_eval(item.value) for item in data_node.value.keywords}
    spec = {
        "node_base": constants["NODE_IN"],
        "fourier_features": 4 * len(constants["FOURIER_CYCLES"]),
        "node_input": constants["NODE_IN"] + 4 * len(constants["FOURIER_CYCLES"]),
        "edge_input": constants["EDGE_IN"],
        "global_input": constants["GLOBAL_IN"],
        "latent": defaults["latent"],
        "message_blocks": defaults["steps"],
        "members": defaults["members"],
        "radius_px": defaults["radius_px"],
        "history_steps": data["n_history"],
        "future_steps": data["n_future"],
        "frames_per_step": data["step_frames"],
        "position_scale_px": constants["POS_SCALE"],
        "history_dropout": defaults["history_dropout"],
        "beta": defaults["beta"],
        "context": defaults["context"],
    }
    # 语义标签对应这一版 _features/_rollout；尺寸变化时要求更新图，不能照旧输出。
    expected = {
        "node_base": 21, "fourier_features": 24, "node_input": 45,
        "edge_input": 5, "global_input": 12, "latent": 64, "message_blocks": 3,
        "members": 5, "radius_px": 150.0, "history_steps": 3,
        "future_steps": 10, "frames_per_step": 6, "position_scale_px": 5.0,
        "context": "motion",
    }
    if not defaults["interactions"] or not defaults["fourier"]:
        raise ValueError("This diagram requires interactions=True and fourier=True.")
    for key, value in expected.items():
        if spec[key] != value:
            raise ValueError(f"Architecture changed: {key}={spec[key]!r}; update the diagram.")
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_net")
    mlp = next(n for n in build.body if isinstance(n, ast.FunctionDef) and n.name == "mlp")
    calls = [
        n.func.attr for n in ast.walk(mlp)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    if calls.count("Linear") != 3 or calls.count("SiLU") != 2:
        raise ValueError("MLP layers changed; update the detailed diagram.")
    return spec


def verify_checkpoint(spec):
    """仅在明确启用时读取权重，检查五个成员每一层的实际尺寸。"""
    import torch

    path = ROOT / "results/world_model/model/graph_world_model.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    expected_config = {
        "interactions": True, "fourier": True, "members": spec["members"],
        "latent": spec["latent"], "steps": spec["message_blocks"],
        "radius_px": spec["radius_px"], "context": spec["context"],
    }
    for key, value in expected_config.items():
        if saved["config"].get(key) != value:
            raise ValueError(f"Checkpoint and source disagree: {key}.")
    shapes = {}
    for prefix, first, last in [
        ("node_enc", 45, 64), ("edge_enc", 5, 64), ("glob_enc", 12, 64),
        ("dec", 128, 8),
    ] + [
        (f"{kind}.{index}", first, 64)
        for index in range(3)
        for kind, first in [("edge_upd", 256), ("node_upd", 192)]
    ]:
        for layer, rows, cols in [(0, 64, first), (2, 64, 64), (4, last, 64)]:
            shapes[f"{prefix}.{layer}.weight"] = (rows, cols)
            shapes[f"{prefix}.{layer}.bias"] = (rows,)
    if len(saved["nets"]) != spec["members"]:
        raise ValueError("Checkpoint member count does not match source.")
    for member in saved["nets"]:
        if set(member) != set(shapes):
            raise ValueError("Checkpoint has unexpected or missing layers.")
        for key, shape in shapes.items():
            if tuple(member[key].shape) != shape:
                raise ValueError(f"Unexpected checkpoint tensor shape: {key}.")
    return {
        "status": "passed", "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
        "members_checked": len(saved["nets"]),
        "parameters_per_member": sum(v.numel() for v in saved["nets"][0].values()),
        "tensor_shapes": shapes,
    }


def arial_fonts(font_dir=None):
    """定位真实 Arial 常规和粗体字体；找不到时停止，不使用近似字体。"""
    if font_dir is not None:
        regular, bold = font_dir / "arial.ttf", font_dir / "arialbd.ttf"
    else:
        windows = Path("/mnt/c/Windows/Fonts")
        if (windows / "arial.ttf").is_file():
            regular, bold = windows / "arial.ttf", windows / "arialbd.ttf"
        else:
            regular = Path(font_manager.findfont("Arial", fallback_to_default=False))
            bold = Path(font_manager.findfont(
                font_manager.FontProperties(family="Arial", weight="bold"),
                fallback_to_default=False,
            ))
    for path in (regular, bold):
        if not path.is_file():
            raise FileNotFoundError(f"Arial font missing: {path}")
        prop = font_manager.FontProperties(fname=str(path))
        if prop.get_name() != "Arial":
            raise ValueError(f"Expected Arial, found {prop.get_name()}: {path}")
        font_manager.fontManager.addfont(str(path))
    return regular, bold


class Diagram:
    """固定画布和少量绘图原语；统一字号、箭头和字体检查。"""

    def __init__(self, title, fonts, palette="mint_gold"):
        self.fig, self.ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=120)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT))
        self.ax.axis("off")
        self.fonts = fonts
        roles = PALETTE_ROLES[palette]
        self.colors = dict(zip(
            ("accent", "linear", "activation", "state", "context"),
            [PALETTES[palette][index] for index in roles],
        ))
        self.labels = []
        self.text(0.4, 10.15, title, 21, bold=True)

    def text(self, x, y, value, size=12, color=INK, bold=False, ha="left", va="center", rotation=0):
        font_path = str(self.fonts[int(bold)])
        available = font_manager.get_font(font_path).get_charmap()
        missing = {char for char in value if not char.isspace() and ord(char) not in available}
        if missing:
            raise ValueError(f"Characters unavailable in Arial: {sorted(missing)}")
        prop = font_manager.FontProperties(fname=font_path, size=size)
        label = self.ax.text(
            x, y, value, fontproperties=prop, color=color, ha=ha, va=va,
            linespacing=1.2, zorder=5, parse_math=False, rotation=rotation,
        )
        self.labels.append(label)
        return label

    def rectangle(self, x, y, w, h, fill="white", edge=LINE, lw=1.0, radius=0.10):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2,
        )
        self.ax.add_patch(patch)
        return patch

    def panel(self, x, y, w, h, label):
        self.rectangle(x, y, w, h, lw=0.85, radius=0.13)
        self.text(x + 0.22, y + h - 0.35, label, 16, bold=True)

    def box(self, x, y, w, h, title, body="", fill=NEUTRAL, edge=LINE, color=INK,
            size=12, body_size=11):
        self.rectangle(x, y, w, h, fill, edge)
        if body:
            self.text(x + w / 2, y + h * 0.73, title, size, color, True, ha="center")
            self.text(x + w / 2, y + h * 0.31, body, body_size, color, ha="center")
        else:
            self.text(x + w / 2, y + h / 2, title, size, color, True, ha="center")

    def arrow(self, points, color=INK, lw=1.05, dashed=False, head=True):
        style = "--" if dashed else "-"
        if len(points) > 2:
            xs, ys = zip(*points[:-1])
            self.ax.plot(xs, ys, color=color, lw=lw, ls=style, zorder=3)
        self.ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>" if head else "-",
            mutation_scale=11, linewidth=lw, color=color, linestyle=style,
            shrinkA=0, shrinkB=0, zorder=3,
        ))

    def junction(self, x, y, symbol, color=INK):
        self.ax.add_patch(Circle((x, y), 0.13, facecolor="white", edgecolor=color, lw=1.05, zorder=4))
        self.text(x, y, symbol, 12, color, ha="center")

    def save(self, out, stem, dpi):
        """检查文字范围/字体并输出位图和嵌入字体或字形轮廓的矢量图。"""
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        canvas = self.fig.bbox
        for label in self.labels:
            if label.get_fontproperties().get_name() != "Arial":
                raise ValueError(f"Non-Arial text: {label.get_text()}")
            bounds = label.get_window_extent(renderer)
            if not canvas.contains(bounds.x0, bounds.y0) or not canvas.contains(bounds.x1, bounds.y1):
                raise ValueError(f"Text outside canvas: {label.get_text()}")
        # 记录文字之间的真实相交；返回结果用于检查，避免只凭脚本无报错验收。
        overlaps = []
        boxes = [label.get_window_extent(renderer).padded(-1.0) for label in self.labels]
        for i, first in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                if first.overlaps(boxes[j]):
                    overlaps.append([self.labels[i].get_text(), self.labels[j].get_text()])
        files = []
        for suffix in ("png", "pdf", "svg"):
            path = out / f"{stem}.{suffix}"
            self.fig.savefig(path, dpi=dpi, facecolor="white", transparent=False)
            files.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
        plt.close(self.fig)
        return {"files": files, "text_count": len(self.labels), "font": "Arial", "text_overlaps": overlaps}


def load_example():
    """读取真实 B 实验的一帧和已保存的预测，只作为架构图中的示例。"""
    import cv2

    prediction_path = ROOT / "results/world_model/predictions.csv"
    objects_path = ROOT / "results/B_offline/objects.csv"
    contours_path = ROOT / "results/B_offline/contours.npz"
    video_path = ROOT / "data/raw/expB.mp4"
    frame = 240
    with prediction_path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["fold"] == "train A → test B" and int(row["frame"]) == frame]
    predictions = {}
    for horizon in (0.1, 0.5, 1.0):
        predictions[horizon] = {
            int(row["track"]): (float(row["Graph world model: x"]),
                                float(row["Graph world model: y"]))
            for row in rows if float(row["horizon_s"]) == horizon
        }
    common = set.intersection(*(set(value) for value in predictions.values()))
    if not common:
        raise ValueError("The illustration frame has no complete saved predictions.")
    with objects_path.open(newline="") as handle:
        objects = [row for row in csv.DictReader(handle) if int(row["frame"]) == frame]
    positions = {int(row["id"]): (float(row["cx"]), float(row["cy"])) for row in objects}
    with np.load(contours_path, allow_pickle=False) as packed:
        contours = {}
        for index in np.flatnonzero(packed["frame"] == frame):
            left, right = packed["offset"][index:index + 2]
            contours[int(packed["id"][index])] = packed["xy"][left:right].copy()
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if not ok:
            raise ValueError(f"Cannot read frame {frame}: {video_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    return {
        "image": image, "positions": positions, "contours": contours,
        "predictions": predictions, "common": sorted(common), "crop": (480, 520, 1110, 1024),
        "provenance": {
            "experiment": "B", "frame": frame, "fold": "train A → test B",
            "crop_xyxy": [480, 520, 1110, 1024], "displayed_horizons_s": [0, 0.1, 0.5, 1.0],
            "prediction_tracks": sorted(common),
            "note": "Forecast centroids on the SAME initial frame; not future video frames. "
                    "The examples are saved held-out predictions, not an accuracy claim.",
            "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                              (prediction_path, objects_path, contours_path, video_path)},
        },
    }


def scene(d, example, x, y, w, h, horizon=None, relations=False):
    """在初始画面上显示真实轮廓或保存的预测中心；坐标不放大、不平移。"""
    x0, y0, x1, y1 = example["crop"]
    d.ax.imshow(example["image"][y0:y1, x0:x1], extent=(x, x + w, y, y + h),
                aspect="auto", zorder=2)
    border = d.rectangle(x, y, w, h, fill="none", edge=INK, radius=0, lw=0.9)

    def points(values):
        values = np.asarray(values, dtype=float)
        return np.column_stack((x + (values[:, 0] - x0) * w / (x1 - x0),
                                y + (y1 - values[:, 1]) * h / (y1 - y0)))

    def inside(point):
        return x0 <= point[0] <= x1 and y0 <= point[1] <= y1

    if relations:
        visible = [point for point in example["positions"].values() if inside(point)]
        for index, a in enumerate(visible):
            for b in visible[index + 1:]:
                if np.linalg.norm(np.subtract(a, b)) < 150:
                    q = points([a, b])
                    d.ax.plot(q[:, 0], q[:, 1], color="white", lw=0.8, zorder=3, clip_path=border)
        for contour in example["contours"].values():
            q = points(np.vstack([contour, contour[:1]]))
            d.ax.plot(q[:, 0], q[:, 1], color=d.colors["accent"], lw=0.75,
                      zorder=3, clip_path=border)
    for track in example["common"]:
        current = example["positions"][track]
        target = current if horizon is None else example["predictions"][horizon][track]
        if not inside(current) or not inside(target):
            continue
        q = points([current, target])
        if horizon is not None:
            d.ax.plot(q[:, 0], q[:, 1], color=d.colors["accent"], lw=1.0, zorder=4)
        d.ax.plot(q[-1, 0], q[-1, 1], "o", ms=4.2, color=d.colors["accent"],
                  markeredgecolor="white", markeredgewidth=0.65, zorder=4)


def vertical_layer(d, x, y, w, h, label, fill, size=13):
    """采用参考图中的细长竖向层；所有文字仍使用真实 Arial。"""
    d.rectangle(x, y, w, h, fill, d.colors["accent"], lw=0.85, radius=0.055)
    rgb = matplotlib.colors.to_rgb(fill)
    luminance = sum(value * weight for value, weight in zip(rgb, (0.2126, 0.7152, 0.0722)))
    foreground = "white" if luminance < 0.42 else INK
    d.text(x + w / 2, y + h / 2, label, size, foreground, ha="center", rotation=270)


def mlp_layers(d, x, y, w, h, first, last, title):
    """按实际代码展开 Linear / SiLU / Linear / SiLU / Linear 五个层。"""
    d.rectangle(x, y, w, h, "white", d.colors["activation"], lw=1.0, radius=0.10)
    d.text(x + w / 2, y + h - 0.21, title, 13, bold=True, ha="center")
    widths = [0.20, 0.08, 0.20, 0.08, 0.20]
    labels = [f"Linear\n{first} → 64", "SiLU", "Linear\n64 → 64", "SiLU", f"Linear\n64 → {last}"]
    cursor = x + 0.03 * w
    centre = y + (h - 0.48) / 2 + 0.14
    for index, (share, label) in enumerate(zip(widths, labels)):
        layer_width = share * w
        fill = d.colors["activation"] if index % 2 else d.colors["linear"]
        vertical_layer(d, cursor, y + 0.14, layer_width, h - 0.48, label, fill, 11.5)
        if index < 4:
            d.arrow([(cursor + layer_width, centre), (cursor + layer_width + 0.045 * w, centre)])
        cursor += layer_width + 0.045 * w
    return centre


def processor_detail(d, x=0.6, y=0.55):
    """一个真实消息传递块：更新边、按接收节点求和、更新节点，含两条残差。"""
    c = d.colors
    # 三个块的内部结构相同，但参数不同；这一处仅展开一个块。
    d.text(x + 0.12, y + 4.92, "Message-passing block", 18, bold=True)
    d.text(x + 15.75, y + 4.92, "×3  distinct weights", 12, ha="right")
    d.rectangle(x + 0.05, y + 4.02, 1.65, 0.52, c["context"], c["accent"], radius=0.07)
    rgb = matplotlib.colors.to_rgb(c["context"])
    dark = sum(rgb) / 3 < 0.45
    d.text(x + 0.875, y + 4.28, "Global · 64", 12, "white" if dark else INK, ha="center")
    d.arrow([(x + 1.7, y + 4.28), (x + 9.15, y + 4.28)], head=False)
    d.arrow([(x + 2.35, y + 4.28), (x + 2.35, y + 2.65)])
    d.arrow([(x + 9.15, y + 4.28), (x + 9.15, y + 2.65)])
    d.rectangle(x + 0.02, y + 1.53, 1.63, 1.94, "none", LINE, lw=0.7, radius=0.035)
    for i, name in enumerate(("Edge", "Sender", "Receiver")):
        vertical_layer(d, x + 0.1 + i * 0.53, y + 1.65, 0.4, 1.7, name, c["state"], 11.5)
    d.arrow([(x + 1.56, y + 2.52), (x + 2.22, y + 2.52)])
    d.junction(x + 2.35, y + 2.52, "C")
    d.arrow([(x + 2.48, y + 2.52), (x + 2.9, y + 2.52)])
    mlp_layers(d, x + 2.9, y + 1.36, 3.85, 2.52, 256, 64, "Edge update")
    d.arrow([(x + 6.75, y + 2.52), (x + 7.07, y + 2.52)])
    d.junction(x + 7.2, y + 2.52, "+")
    d.arrow([(x + 0.3, y + 3.35), (x + 0.3, y + 3.89),
             (x + 7.2, y + 3.89), (x + 7.2, y + 2.65)])
    d.arrow([(x + 7.33, y + 2.52), (x + 7.8, y + 2.52)])
    vertical_layer(d, x + 7.8, y + 1.65, 0.65, 1.7, "Sum by receiver", c["activation"], 12)
    d.arrow([(x + 8.45, y + 2.52), (x + 9.02, y + 2.52)])
    d.junction(x + 9.15, y + 2.52, "C")
    d.arrow([(x + 9.28, y + 2.52), (x + 9.65, y + 2.52)])
    mlp_layers(d, x + 9.65, y + 1.36, 3.85, 2.52, 192, 64, "Node update")
    d.arrow([(x + 13.5, y + 2.52), (x + 13.92, y + 2.52)])
    d.junction(x + 14.05, y + 2.52, "+")
    d.box(x + 8.55, y + 0.58, 1.2, 0.45, "Node · 64", fill=c["state"], edge=c["accent"], size=11)
    d.arrow([(x + 9.15, y + 1.03), (x + 9.15, y + 2.39)])
    d.arrow([(x + 9.75, y + 0.805), (x + 14.05, y + 0.805), (x + 14.05, y + 2.39)])
    d.arrow([(x + 14.18, y + 2.52), (x + 14.65, y + 2.52)])
    vertical_layer(d, x + 14.65, y + 1.65, 0.8, 1.7, "Updated node · 64", c["state"], 12)
    d.text(x + 0.1, y + 0.45, "C  Concatenation       +  Residual addition", 11, MUTED)
    d.text(x + 15.75, y + 0.1, "64-D hidden states", 11, MUTED, ha="right")


def overview(fonts, palette="mint_gold", example=None):
    """参考完整配图：上方总流程、下方处理块展开、右侧真实预测的时间推进。"""
    d = Diagram("World–Action Model", fonts, palette)
    c = d.colors
    d.text(19.5, 10.15, "0.1 s per step · 10-step rollout", 12, ha="right")
    groups = [(0.3, 6.2, 6.2, 3.45, "Scene inputs", c["state"]),
              (6.65, 6.2, 3.25, 3.45, "Encoders", c["state"]),
              (10.05, 6.2, 2.65, 3.45, "Processor", NEUTRAL),
              (12.85, 6.2, 2.1, 3.45, "Decoder", NEUTRAL),
              (15.1, 6.2, 1.65, 3.45, "Ensemble", c["state"])]
    for x, y, w, h, title, fill in groups:
        d.rectangle(x, y, w, h, fill, fill, radius=0.16)
        d.text(x + w / 2, y + h - 0.34, title, 17 if w > 2 else 14, bold=True, ha="center")
    scene(d, example, 0.5, 6.75, 2.7, 2.16, relations=True)
    d.text(1.85, 6.48, "Exp. B · frame 240", 11, ha="center")
    for y, title, encoder in [(8.55, "Regions · 45-D", "Node MLP · 45 → 64"),
                              (7.6, "Relations · 5-D", "Edge MLP · 5 → 64"),
                              (6.65, "PZT + motion · 12-D", "Global MLP · 12 → 64")]:
        d.box(3.55, y, 2.55, 0.6, title, fill="white", edge=c["accent"], size=12)
        d.arrow([(6.1, y + 0.3), (6.9, y + 0.3)])
        d.box(6.9, y, 2.75, 0.6, encoder, fill=c["linear"], edge=c["accent"], size=12)
    d.arrow([(3.2, 8.5), (3.37, 8.5), (3.37, 8.85), (3.55, 8.85)])
    d.arrow([(3.2, 7.9), (3.55, 7.9)])
    d.text(4.8, 6.35, "Tracks + action log", 10.5, ha="center")
    d.arrow([(9.65, 8.85), (10.25, 8.85)])
    d.arrow([(9.65, 7.9), (10.25, 7.9)])
    d.rectangle(10.25, 7.15, 2.25, 1.9, "white", c["activation"], radius=0.09)
    for x, text in [(10.43, "Edge MLP"), (11.15, "Sum"), (11.68, "Node MLP")]:
        width = 0.35 if text == "Sum" else 0.6
        vertical_layer(d, x, 7.35, width, 1.5, text,
                       c["activation"] if text == "Sum" else c["linear"], 11.5)
    d.arrow([(11.03, 8.05), (11.15, 8.05)], lw=0.85)
    d.arrow([(11.5, 8.05), (11.68, 8.05)], lw=0.85)
    d.text(12.35, 6.96, "×3", 14, ha="right")
    d.arrow([(12.5, 8.05), (12.97, 8.05)])
    d.junction(13.1, 8.05, "C")
    d.arrow([(13.23, 8.05), (13.4, 8.05)])
    vertical_layer(d, 13.4, 7.35, 1.0, 1.5, "MLP\n128 → 8", c["linear"], 12.5)
    d.arrow([(13.9, 7.35), (13.9, 7.0)])
    d.box(13.15, 6.55, 1.55, 0.45, "State update", fill="white", edge=c["accent"], size=10.5)
    d.arrow([(9.65, 6.95), (9.85, 6.95), (9.85, 6.3), (13.1, 6.3), (13.1, 7.92)])
    d.arrow([(11.38, 6.3), (11.38, 7.15)])
    d.text(10.08, 6.5, "Global context", 9.5)
    d.arrow([(14.7, 6.775), (15.0, 6.775), (15.0, 8.05), (15.32, 8.05)])
    for index in range(4, -1, -1):
        d.rectangle(15.32 + index * 0.045, 7.55 + index * 0.12, 0.95, 0.64,
                    c["linear"], c["accent"], radius=0.025)
    d.text(15.9, 8.98, "5 members", 11, ha="center")
    d.arrow([(15.8, 7.55), (15.8, 7.0)])
    d.box(15.3, 6.55, 1.2, 0.45, "Mean", fill="white", edge=c["accent"], size=11)
    d.arrow([(16.5, 6.775), (16.93, 6.775), (16.93, 8.05), (17.25, 8.05)])

    # 灰色展开区域只连接处理器，明确下方不是另一个串联网络。
    d.ax.add_patch(Polygon([(10.05, 6.2), (12.7, 6.2), (16.75, 5.8), (0.3, 5.8)],
                           facecolor=NEUTRAL, edgecolor="none", zorder=1))
    d.rectangle(0.3, 0.38, 16.45, 5.42, NEUTRAL, NEUTRAL, radius=0.17)
    processor_detail(d)
    d.text(0.5, 0.16, "Shared weights across time; independent weights across members", 11, MUTED)

    d.rectangle(17.08, 0.38, 2.62, 9.27, c["state"], c["state"], radius=0.16)
    d.text(18.39, 9.31, "Centroid rollout", 14, bold=True, ha="center")
    for y, horizon, label in [(7.4, None, "Observed t"), (5.2, 0.1, "t + 0.1 s"),
                               (3.0, 0.5, "t + 0.5 s"), (0.8, 1.0, "t + 1.0 s")]:
        scene(d, example, 17.28, y, 2.06, 1.648, horizon=horizon)
        d.text(18.31, y - 0.18, label, 12, ha="center")
    d.arrow([(19.51, 9.03), (19.51, 0.8)], lw=1.35)
    return d


def network_detail(fonts, palette="mint_gold", example=None):
    """细节图使用同一层条风格，展开三个编码器及精确输出含义。"""
    d = Diagram("World–Action Model | Features and Readout", fonts, palette)
    c = d.colors
    d.text(19.5, 10.15, "One member · one step", 12, ha="right")
    d.rectangle(0.3, 4.7, 19.4, 4.95, c["state"], c["state"], radius=0.16)
    d.text(0.6, 9.3, "Feature encoders", 18, bold=True)
    configs = [
        (0.65, "Node · 45-D", 45,
         "Position 2  |  Fourier 24  |  History 9\nShape 6  |  Wall 3  |  Image edge 1"),
        (7.05, "Edge · 5-D", 5,
         "Relative position 2  |  Distance 1\nSurface gap 1  |  Log-area difference 1"),
        (13.45, "Global · 12-D", 12,
         "Current + previous PZT 8\nTime since switch 1  |  Field motion 3"),
    ]
    for x, title, dims, label in configs:
        d.text(x + 2.9, 8.76, title, 14, bold=True, ha="center")
        d.text(x + 2.9, 8.18, label, 11.5, ha="center")
        mlp_layers(d, x + 0.15, 5.35, 5.5, 2.25, dims, 64, "")
        d.text(x + 2.9, 5.02, "64-D encoding", 12, ha="center")

    d.rectangle(0.3, 0.38, 19.4, 4.0, NEUTRAL, NEUTRAL, radius=0.16)
    d.text(0.6, 4.02, "Node decoder", 18, bold=True)
    d.rectangle(0.67, 0.86, 1.88, 2.58, "none", LINE, lw=0.7, radius=0.035)
    vertical_layer(d, 0.8, 1.0, 0.65, 2.3, "Final node · 64", c["linear"], 13)
    vertical_layer(d, 1.75, 1.0, 0.65, 2.3, "Global · 64", c["context"], 13)
    d.arrow([(2.4, 2.15), (2.92, 2.15)])
    d.junction(3.05, 2.15, "C")
    d.arrow([(3.18, 2.15), (3.7, 2.15)])
    mlp_layers(d, 3.7, 0.865, 5.8, 2.77, 128, 8, "128 → 64 → 64 → 8")
    d.arrow([(9.5, 2.15), (10.15, 2.15)])
    vertical_layer(d, 10.15, 1.0, 0.7, 2.3, "8 outputs", c["state"], 13)
    d.arrow([(10.85, 2.15), (11.25, 2.15), (11.25, 2.95), (11.65, 2.95)])
    d.arrow([(11.25, 2.15), (11.25, 1.45), (11.65, 1.45)])
    d.box(11.65, 2.4, 7.35, 1.12, "4 state increments",
          "dx, dy, log(area), aspect", fill=c["linear"], edge=c["accent"], size=14, body_size=13)
    d.box(11.65, 0.9, 7.35, 1.12, "4 log-variance terms",
          "x, y, log(area), aspect", fill="white", edge=c["accent"], size=14, body_size=13)
    d.text(0.6, 0.16, "Position increment × 5 px; log-area and aspect increments × 0.1", 11, MUTED)
    return d


def palette_sheet(fonts, example, out):
    """仅生成色卡对照位图，全部使用同一布局和同一份实验记录。"""
    from PIL import Image, ImageDraw, ImageFont

    cell_w, cell_h = 1600, 925
    sheet = Image.new("RGB", (cell_w * 2, cell_h * 3), "white")
    painter = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(fonts[1]), 25)
    regular = ImageFont.truetype(str(fonts[0]), 18)
    layouts = {}
    for index, name in enumerate(PALETTES):
        drawing = overview(fonts, name, example)
        drawing.fig.set_dpi(80)
        drawing.fig.canvas.draw()
        pixels = np.asarray(drawing.fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(drawing.fig)
        image = Image.fromarray(pixels)
        left, top = (index % 2) * cell_w, (index // 2) * cell_h
        sheet.paste(image, (left, top + 72))
        painter.text((left + 28, top + 13), name.replace("_", " ").title(), font=font, fill=INK)
        for i, colour in enumerate(PALETTES[name]):
            x = left + 400 + i * 225
            painter.rounded_rectangle((x, top + 14, x + 33, top + 47), radius=5, fill=colour)
            painter.text((x + 43, top + 19), colour.upper(), font=regular, fill=INK)
        layouts[name] = {"swatches": PALETTES[name], "roles": drawing.colors}
    path = out / "03_palette_comparison.png"
    sheet.save(path)
    return {"file": path.name, "sha256": sha256(path), "palettes": layouts}


def write_notes(out):
    """详细说明留在配套文件，避免将限制和出处挤到图片里。"""
    notes = """# World–Action Model Architecture

- `01_world_model_overview`: scene inputs, encoders, processor, decoder, ensemble,
  an expanded message-passing block and real saved centroid predictions.
- `02_world_model_network_detail`: feature dimensions and fully expanded MLP layers.
- `03_palette_comparison.png`: optional comparison of the six user-provided palettes.
- PNG: 20 × 10.5 inches at the requested DPI (6000 × 3150 at the default 300 dpi).
  PDF: vector with embedded Arial. SVG: vector with Arial glyphs converted to paths,
  preserving appearance on other machines.
- `architecture_manifest.json`: source/font hashes, architecture, optional checkpoint checks,
  output hashes and text-layout checks. These figures describe implementation, not performance.

Run from the microbubble_tracking project directory:

```bash
../../.venv/bin/python -m world_model.architecture --palette mint_gold --palette-sheet \
  --output-dir figures/world_model_architecture_new
```

To also verify every saved weight tensor, use an existing PyTorch environment:

```bash
../.venv_baseline312/bin/python -m world_model.architecture --verify-checkpoint \
  --output-dir figures/world_model_architecture_verified
```

The output directory must not already exist. `--font-dir` accepts a directory containing
`arial.ttf` and `arialbd.ttf`. Missing Arial causes an error, never a substitute font.
Palette names: mint_gold, rose_yellow, lime, burgundy_cream, ice_red, coral.
Colours are taken directly from the six supplied colour cards; white/grey/black are
neutral backgrounds and line/text colours. No colours from separate cards are mixed.

The illustrated input is experiment B, frame 240, cropped to x=480..1110 and y=520..1024.
The right column uses the saved `train A → test B` graph-world-model predictions at
0.1, 0.5 and 1.0 seconds on the SAME initial image. Markers are predicted centroids;
the line from the initial to the predicted centroid is a visual connector, not a
measured continuous path. The background is not a future observation or generated video.
Only tracks available at all three saved horizons are used for the centroid examples.
These illustration data belong to the held-out B model, not the final A+B training fit.
The column does not claim that these forecasts are accurate. File hashes are recorded.

## Reading the architecture

One region is one node. Edges connect region centres less than 150 pixels apart in both
directions; self-edges are excluded. Edge gap uses radii estimated from area, not exact
contour-to-contour distance. Node input is 21 base features plus 24 Fourier position features.
The six shape features are log(area), aspect ratio, circularity, solidity, sin(2 angle)
and cos(2 angle). Current and previous actions are step-averaged amplitudes, scaled by
0.8 Vpp; inactive outputs and amplitudes below the current 0.1 Vpp activity threshold
are encoded as zero. Frequency is not an input.

Each encoder/update/decoder MLP has three linear layers and two SiLU activations.
The processor has three blocks with distinct parameters. Each edge update includes the
old edge, sender node, receiver node and encoded global input. Updated incoming edges
are summed by receiving node. Both edge and node updates retain residual connections.
The global encoding is broadcast unchanged within a step and concatenated again at
the node decoder. There is no learned global-state update inside the processor.

The exact decoder channel order is dx, dy, log-variance-x, log-variance-y,
log-area increment, log-area log-variance, aspect increment, aspect log-variance.
The detail figure groups these channels by meaning rather than their tensor order.
Position increments are multiplied by 5; log-area/aspect increments by 0.1.
Within each member, per-step variances are accumulated during the rollout.

The same full network is reused for all ten time steps within a member. Each step is
six source frames, approximately 0.1 seconds for the current 60 fps recordings.
Positions, log-area, aspect, displacement history, wall lookup, global motion and
neighbour edges are updated from predictions. Circularity, solidity, orientation and
the image-border flag remain at their initial values. No future measured positions
are fed back during prediction, and no merge/split/birth/death module is present.
Wall features are inputs, not a hard wall constraint. Node count stays fixed within
one rollout. The diagrams contain no invented experimental data.

Each of the five separately trained members completes its own rollout. The code
evaluates members in a Python loop; the side-by-side branches indicate independent
predictions, not guaranteed parallel hardware execution. Position/log-area/aspect
means are averaged across members. Returned position variance combines mean accumulated
within-member variance (averaged over x/y) and between-member position variance
(also averaged over x/y). Although shape uncertainty is trained internally, the public
predict() result does not currently return a combined shape-uncertainty array.

Training is outside the inference diagram: the complete ten-step prediction is scored
against eligible automatic tracks; recent motion is hidden for approximately half of
nodes during training. The no-neighbour and reviewed-training variants, and the
persistence/constant-velocity/response-map/gradient-boosting baselines, are separate
comparisons rather than stages of this core network.

## Visual references

The user-provided full Figure 1 of Diffusion Planner supplies the visual reference:
compact top-level modules, vertical layer strips, routed condition/residual lines,
an enlarged processor panel and a vertical time column. The six supplied moth/butterfly
colour cards supply the palettes. The artwork is newly drawn with Matplotlib.

The authoritative architecture is the local `world_model/models.py`. No attention,
layer normalization, diffusion timestep or denoising module was added to the diagram:
those belong to the reference model, not the current world model.
"""
    (out / "README.md").write_text(notes, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Render the current world-model architecture using Arial.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/world_model_architecture_refined")
    parser.add_argument("--font-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--palette", choices=tuple(PALETTES), default="mint_gold")
    parser.add_argument("--palette-sheet", action="store_true", help="Render the six supplied palette options.")
    parser.add_argument("--verify-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.dpi < 150:
        parser.error("Use at least 150 dpi for readable output.")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    spec = read_architecture()
    fonts = arial_fonts(args.font_dir)
    checked = verify_checkpoint(spec) if args.verify_checkpoint else {"status": "not requested"}
    example = load_example()
    matplotlib.rcParams.update({
        "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "path", "axes.unicode_minus": False,
    })
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rendered = {}
    for stem, draw in [("01_world_model_overview", overview), ("02_world_model_network_detail", network_detail)]:
        rendered[stem] = draw(fonts, args.palette, example).save(args.output_dir, stem, args.dpi)
    comparison = palette_sheet(fonts, example, args.output_dir) if args.palette_sheet else None
    write_notes(args.output_dir)
    manifest = {
        "created": datetime.now(ZoneInfo("Europe/Zurich")).isoformat(timespec="seconds"),
        "architecture": spec, "checkpoint_verification": checked,
        "palette": {"name": args.palette, "swatches": PALETTES[args.palette]},
        "palette_comparison": comparison,
        "illustration": example["provenance"],
        "source_files": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (MODEL_SOURCE, DATA_SOURCE, Path(__file__))
        },
        "fonts": [{"path": str(p), "family": "Arial", "sha256": sha256(p)} for p in fonts],
        "png_size_px": [WIDTH * args.dpi, int(HEIGHT * args.dpi)],
        "figures": rendered,
    }
    (args.output_dir / "architecture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    overlaps = sum(len(value["text_overlaps"]) for value in rendered.values())
    print(f"Written: {args.output_dir.resolve()}")
    print(f"2 figures × 3 formats; Arial; text-overlap pairs: {overlaps}")
    if overlaps:
        raise RuntimeError("Text overlaps found. Inspect the manifest before using the figures.")


if __name__ == "__main__":
    main()
