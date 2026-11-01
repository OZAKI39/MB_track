# 微泡实验工作区：从实验录像到动作预测

这里保存本项目的实验执行、视频采集、微泡识别与追踪、人工审核、数据分析和图表导出代码。目标是把三个问题连起来：**什么时候施加了什么超声动作，画面里的微泡怎样变化，能否据此预测下一步变化。**

目前的识别与追踪主线是 [microbubble_tracking](microbubble_tracking/README.md)。早期的深色区域识别、Mix A Mask、SAM 2.1、Cutie 和联合追踪代码仍保留，供追溯与比较；它们不是当前主线的必经步骤。

本文按实际操作顺序说明入口与模块。历史实验的数值、日志和验证证据仍在 [validation](validation/) 中，本文不把某次测试通过写成对所有环境的保证。

## 当前在线开发：先看这里

当前相机采集入口已接入新逐帧识别、追踪和保存模块。已有视频可经 Windows 接收、
独立录像、跨系统传图和 WSL 分析进行整条路线的模拟验证；真实相机尚未连接验证。
本轮录像保存与算法结果一致性检查通过，但仍有少量分析跳帧，**60 fps 每帧都处理尚未全部通过**。
完整数据见[相机接入验证报告](validation/camera_tracking_20261031_final/report/REPORT.md)。
新增的本机 RTSP 接入检查见[Windows 模拟相机→WSL 报告](validation/rtsp_tracking_20261031-203916/report_final/REPORT.md)：
A、B 各运行 22 秒，分别完整录像并解码 1320 帧、分析保存 1318 帧；各有 2 帧
在最新画面交接处被替换，全部有帧号和时间记录。复现命令见第 7.5.1 节。
等待区容量的后续对照见[一帧/两帧重复测试](validation/queue_compare_20261031-211619/report/REPORT.md)：
A、B 各在两种设置下重复三次。新增 `--waiting-frames 2` 测试选项，正式默认仍是
一帧；如何比较跳帧与反馈延迟见第 7.5.3 节。
日常开发先读
[ONLINE_TRACKING.md](microbubble_tracking/ONLINE_TRACKING.md) 的相关章节，再看对应模块：

| 当前任务                             | 主要模块                                                                               |
| ------------------------------------ | -------------------------------------------------------------------------------------- |
| 送来一帧，处理并记录一帧             | `microbubble_tracking/online_tracking.py`、`bubble_tracker.py`                     |
| 分析与保存并行、结束时核对保存完整性 | `microbubble_tracking/async_store.py`、`online_store.py`                           |
| 按真实播放速度验证是否跟得上         | `microbubble_tracking/paced_replay.py`、`paced_source.py`、`frame_conversion.py` |
| 真实相机入口及同路线视频模拟         | `parallel_video_check.py`、`parallel_camera.py`、`parallel_camera_bridge.py`     |
| 仅在需要时深入排查停顿               | `microbubble_tracking/stall_probe.py`、`native_stall_watch.py`                     |

已采用编号图和分析端计算预热、送帧转换及平滑结果内存复用、异步保存，以及开始前和保存后的
主动清理；运行中仍使用原有自动清理规则。分析端准备完成后才启动正式送帧计时。
来源排查见[剩余停顿报告](validation/remaining_stalls_20261030-175436/REPORT.md)，
正式接入及启动对照见[分析端预热验证](validation/analysis_startup_20261030-183814/REPORT.md)，
平滑图复用的像素、追踪及回放核对见[平滑内存复用验证](validation/blur_reuse_20261030-185211/REPORT.md)。

`.ignore` 让日常 `rg` 搜索跳过验证输出、数据、图表、权重和外部依赖，避免成千上万
份历史文件淹没当前源码。它不删除文件、不改变 Git 同步范围，也不影响程序读取数据。
需要旧记录时，直接打开本页链接；搜索记录内容时，明确指定那一次运行：

```bash
rg --no-ignore -n '14.789' customer_workspace/validation/remaining_stalls_20261030-175436/REPORT.md
```

早期 Mask、SAM、Cutie 和联合方法见第 10.2 节；只有追溯这些方法时才需要读取。

## 1. 先找到自己要做的事

| 想做什么                           | 从哪里开始                                                                                                 | 会得到什么                                                     |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 开发逐帧接口、测真实速度或排查停顿 | [ONLINE_TRACKING.md](microbubble_tracking/ONLINE_TRACKING.md)                                               | 当前逐帧处理、保存、跳帧及性能验证方法                         |
| 查看已经整理好的图表和动画         | [实验 A](microbubble_tracking/figures/expA/README.md)、[实验 B](microbubble_tracking/figures/expB/README.md) | 识别、追踪、运动、形变、群体指标、动作响应及 baseline 对比材料 |
| 检查已有追踪结果                   | [轨迹审查工具](microbubble_tracking/review/review_tool.py)，见第 5 节                                       | 人工判断哪些轨迹可信，哪些编号属于同一个物体                   |
| 处理一段新录像                     | [bubble_tracker.py](microbubble_tracking/bubble_tracker.py)，见第 4 节                                      | 每帧轮廓、编号、位置、形状及融合/分裂等事件                    |
| 分析微泡怎样移动、变形、聚集       | [analyze.py](microbubble_tracking/analysis/analyze.py)，见第 6 节                                           | 可用于统计的 CSV 表格、曲线与核对图                            |
| 对照 PZT 动作与微泡变化            | [pzt_analysis.py](microbubble_tracking/analysis/pzt_analysis.py)，见第 6 节                                 | 每帧 PZT 状态、动作后的响应、软件时间对应范围                  |
| 采集新实验并自动控制四路 PZT       | [afg_video_check.py](afg_video_check.py)，见第 7 节                                                         | 原始录像、分析记录、动作日志和设备读回                         |
| 只测试录像或控制时序               | 第 7、10 节的专项入口                                                                                      | 录像完整性、延迟、错误原因及收尾记录                           |
| 比较自己的方法与 baseline          | [baseline/](microbubble_tracking/baseline/)、[baseline_animations.py](baseline_animations.py)，见第 8 节     | 同一录像下的识别、追踪与事件对比                               |
| 查看或研究动作预测模型             | [world_model/](microbubble_tracking/world_model/)，见第 9 节                                                | 位置与形状、合并/分裂概率、全视野分布预测及评估                |
| 重新制作 PPT 图表                  | 第 8 节                                                                                                    | 英文、Arial 的图片、动画及来源记录                             |

## 2. 整体工作流程，以及 Windows 和 WSL 各自做什么

```text
真实实验
  ├─ Windows：接收 Chronos 视频
  │    ├─ 录像支路：保存压缩视频，优先保留接收到的全部画面
  │    └─ 分析支路：传递压缩帧 → WSL 独立解码 → 最新画面 → OnlineProcessor
  └─ Windows：按时间表控制 AFG → 四路 PZT
       └─ 保存实际发送、读回和关断记录
                         ↓
              原始视频 + 动作日志 + 帧时间
                         ↓
              microbubble_tracking/bubble_tracker.py
              找出微泡区域，并连接前后帧中的同一区域
                         ↓
              review/review_tool.py
              人工审核可信度，修正同一物体的编号对应
                         ↓
              analysis/analyze.py → analysis/pzt_analysis.py
              运动、形变、群体变化 → 与 PZT 动作对应
                         ↓
              图表与动画 / baseline 比较 / world_model 预测研究
```

**在 WSL 终端输入命令，不等于仪器由 WSL 直接控制。** WSL 负责启动、传递路径和处理数据；AFG 的 USB/VISA 通信实际由 Windows Python 执行，视频接收也由 Windows 进程执行。

| 环节                   | 当前对应模块                                                                                                     | 职责                                                         |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| 实验总入口             | [afg_video_check.py](afg_video_check.py)                                                                          | 协调录像、分析和独立的 AFG 控制进程                          |
| 四路控制               | [experiment_runner.py](experiment_runner.py)、[afg_video_control.py](afg_video_control.py)                         | 按固定计划时刻发送命令，记录结果，结束或异常时尝试关闭输出   |
| 底层 AFG 驱动          | [../microbubble_experiment/afg.py](../microbubble_experiment/afg.py)                                              | 设备连接、参数写入、状态查询；工作区直接复用它               |
| WSL 启动 Windows       | [wsl_launcher.py](wsl_launcher.py)                                                                                | 路径转换、启动现有解释器、传递取消请求                       |
| 同一视频分成两路       | [parallel_camera.py](parallel_camera.py)、[parallel_camera_bridge.py](parallel_camera_bridge.py)                   | GStreamer 接收与分流；录像和最新画面分析使用独立队列         |
| 现场画面分析           | [parallel_video_check.py](parallel_video_check.py) 调用 [OnlineProcessor](microbubble_tracking/online_tracking.py) | 使用当前背景识别与多区域追踪方法，记录轮廓、编号、速度及事件 |
| 动作与视频时间对应     | [video_clock.py](video_clock.py)、[afg_video_report.py](afg_video_report.py)                                       | 用同一 Windows 主机的时间记录，寻找动作前后的录像帧          |
| 保存后的微泡识别与追踪 | [microbubble_tracking/bubble_tracker.py](microbubble_tracking/bubble_tracker.py)                                  | 当前采用的背景分割与多区域追踪方法                           |

必须区分两件事：

- 工作区的录像检查、持续检查和 PZT 联合入口共用新 `OnlineProcessor`。工程原 GUI、早期 `video_recording_check.py` 仍保留旧 `Pipeline` 作历史用途，不是本文新的相机追踪入口。
- `bubble_tracker.py --bg online` 表示只用已有信息逐帧处理；它目前仍从视频文件取帧。它不是相机接入命令，也没有自动接入世界模型后控制 PZT。

## 3. 运行位置、环境与文件位置

### 3.1 两种命令位置

**工作区顶层模块**从整个工程根目录运行，例如：

```bash
cd /home/lzy/projects/new_microbubble_experiment
.venv/bin/python -m customer_workspace.pzt_video_review --help
```

**识别追踪子项目**的命令从子项目根目录运行，例如：

```bash
cd /home/lzy/projects/new_microbubble_experiment/customer_workspace/microbubble_tracking
../../.venv/bin/python bubble_tracker.py --help
```

下文每组命令都写明了 `cd`。使用明确的解释器路径，可以避免终端同时显示 `.venv`、`base` 时用错 Python。仅执行 `--help` 显示用法，不开始实验。

### 3.2 现有环境的分工

| 环境或配置                                                                                                                                                   | 用途                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| 工程根目录 `.venv/bin/python`                                                                                                                              | WSL 中的常用离线处理、审查、分析和图表导出；工程要求 Python 3.12 及以上                 |
| [../configs/machine.local.json](../configs/machine.local.json)                                                                                                | 本机解释器位置；`windows_afg_python` 指定 AFG 环境，`windows_python` 指定原相机环境 |
| `customer_workspace/.venv_gstreamer/Scripts/python.exe`                                                                                                    | Windows 视频分流接收器使用的独立环境                                                    |
| `customer_workspace/.venv_baseline312/bin/python`                                                                                                          | 已有 baseline 独立环境，包含 OpenCV contrib 和 PyTorch；也可用于当前世界模型代码        |
| [microbubble_tracking/requirements.txt](microbubble_tracking/requirements.txt)                                                                                | 独立识别追踪子项目的依赖列表，不涵盖整个硬件采集链路                                    |
| [baseline/requirements.txt](microbubble_tracking/baseline/requirements.txt)、[world_model/requirements.txt](microbubble_tracking/world_model/requirements.txt) | 比较方法与世界模型的额外依赖                                                            |
| [GStreamer 环境说明](tools/gstreamer/INSTALL.md)、[FFmpeg 工具说明](tools/ffmpeg/INSTALL.md)                                                                   | 已验证安装位置、版本及来源                                                              |

在当前电脑上优先使用已有环境。换电脑时，应按对应依赖和安装说明重新准备；不要把 Windows 仪器依赖、WSL 分析依赖和 baseline 的 OpenCV 版本混装到一个环境中。

### 3.3 数据与代码放在哪里

```text
customer_workspace/
├── configs/                 时间表、专项配置、早期识别与 Mask 配置
├── experiment_runner.py     通用 PZT 时间表执行器
├── afg_video_*.py            PZT + 录像联合实验与报告
├── parallel_*.py             视频分流、分析与录像检查
├── *_ppt.py                  从实际结果生成演示图片和动画
├── baseline_animations.py    baseline 与当前方法的对比视频
├── microbubble_tracking/     当前识别、追踪、审核、分析和预测主线
│   ├── bubble_tracker.py
│   ├── review/              轨迹审查网页
│   ├── analysis/            运动、形变、群体及 PZT 响应分析
│   ├── baseline/            对照方法
│   ├── world_model/         动作预测模型与架构绘图
│   ├── data/raw/            expA.mp4、expB.mp4 原始录像
│   ├── data/pzt/            各录像对应的 PZT 与时间记录
│   ├── results/             自动结果、人工审核、分析表与模型评估
│   └── figures/             已整理的图表、动画和相关数据
├── validation/              各次实验、诊断、对照及导出记录
├── tests/                   软件测试与界面检查脚本
├── tools/                   录像工具和安装说明
├── dependencies/、models/   比较方法所需的外部代码和模型资源
└── bubble_*.py、sam21_*.py 等 早期区域标注、Mask 和模型对照工具
```

两个当前常用实验不能混用：

| 实验 | 原始视频                                   | 已有离线结果                                | 对应 PZT 记录                           | 内容                                |
| ---- | ------------------------------------------ | ------------------------------------------- | --------------------------------------- | ----------------------------------- |
| A    | `microbubble_tracking/data/raw/expA.mp4` | `microbubble_tracking/results/A_offline/` | `microbubble_tracking/data/pzt/expA/` | 四路依次工作，随后 PZT1 + PZT3 工作 |
| B    | `microbubble_tracking/data/raw/expB.mp4` | `microbubble_tracking/results/B_offline/` | `microbubble_tracking/data/pzt/expB/` | PZT1 → PZT3 → PZT1 双动作实验     |

两段录像各 1320 帧、22 秒，动作时间表为 20 秒，末尾保留关断后的录像。工作幅值为 0.8 Vpp、待机幅值为 0.001 Vpp，频率为 2.5 MHz；这些是 AFG 设定值。来源和文件校验见 [data/pzt/README.md](microbubble_tracking/data/pzt/README.md) 与 [manifest.json](microbubble_tracking/data/pzt/manifest.json)。

## 4. 从原始录像得到轮廓和轨迹

### 4.1 当前算法做什么

[bubble_tracker.py](microbubble_tracking/bubble_tracker.py) 集中实现以下过程，使用 OpenCV、NumPy 和 SciPy：

1. **估计空通道的亮度。** 离线模式从整段录像抽取画面，按像素取第 90 百分位，估计没有微泡遮挡时的背景。可以加入同一次装片的其他录像或已有背景，先对齐再合并。
2. **找比背景明显暗的区域。** 用相对暗度比较画面与背景；较低阈值寻找候选，较高阈值确认区域有足够深的核心。再连接小断口、填补内部亮孔并去除太小的区域。
3. **判断前后是不是同一区域。** 按过去运动估计下一位置，将上帧轮廓平移，比较它与当前区域的重叠，再用匈牙利算法求整体的一一对应；无重叠时补充距离匹配。
4. **记录特殊情况。** 多个区域合成一个、一个分成多个、短暂跟丢或离开画面，都留下事件记录。融合、分裂的编号继承规则见子项目 README。

这里识别的是图像中的微泡相关区域，不一定是一颗独立微泡。自动事件也是图像判断，不能仅凭 `lost` 认定微泡已经破灭。

### 4.2 处理一份新的结果

以下示例复用已有 A、B 原始视频，但写入新的试运行目录。**`A_readme_trial` 是示例名称，运行前确认它尚不存在；下一次请换新名称。**

```bash
cd /home/lzy/projects/new_microbubble_experiment/customer_workspace/microbubble_tracking
../../.venv/bin/python bubble_tracker.py data/raw/expA.mp4 \
  --ref data/raw/expB.mp4 \
  --out results/A_readme_trial --save-contours --render
```

- `--ref`：同一装片、同一固定视野的背景参考，不能随便拿另一视野的视频代替。
- `--save-contours`：保存轮廓，后面的人工审查需要它。
- `--render`：另外生成叠加结果视频，便于快速检查。
- `--lo`、`--hi`：相对暗度阈值，当前默认 0.06、0.12；`--min-area` 默认 12 像素。调整后应重新检查结果，不能直接套用旧审核记录。

主要输出：

| 文件               | 内容                                                   |
| ------------------ | ------------------------------------------------------ |
| `objects.csv`    | 每帧每个区域的位置、轨迹编号、面积、形状及追踪速度估计 |
| `events.csv`     | 出现、融合、分裂、离开或跟丢等自动事件                 |
| `contours.npz`   | 每帧轮廓，供审核与后续绘图使用                         |
| `background.npy` | 本次估计的背景，后续同一装片可作为参考                 |
| `stats.json`     | 参数、帧率及处理耗时等运行记录                         |
| `overlay.mp4`    | 指定 `--render` 后生成的检查视频                     |

`bubble_tracker.py` 允许向已有目录写入，会覆盖同名结果；它没有统一的“拒绝覆盖”保护。不要对已经人工审核的 `A_offline`、`B_offline` 直接重跑。

### 4.3 按在线方式处理录像

```bash
cd /home/lzy/projects/new_microbubble_experiment/customer_workspace/microbubble_tracking
../../.venv/bin/python bubble_tracker.py data/raw/expB.mp4 \
  --bg online --ref results/A_offline/background.npy \
  --out results/B_online_trial --save-contours
```

在线方式从已有背景出发，逐帧缓慢更新背景，不用本段未来画面估计背景。使用预先保存的 `.npy` 可以明确参考已在实验前准备好；没有参考而只用第一帧时，静止微团和壁面容易混淆。换装片或改变视野后需要重新建立参考。

追踪器的速度只用当前与过去位置计算，并做平滑，用于下一帧匹配；`objects.csv` 中 `vx/vy` 为像素/帧。第 6 节离线分析的速度采用前后窗口，以像素/秒记录，两者不是同一个量。

## 5. 人工审核：确认哪些数据能用于分析

查看现有 A 实验：

```bash
cd /home/lzy/projects/new_microbubble_experiment/customer_workspace/microbubble_tracking
../../.venv/bin/python review/review_tool.py \
  --video data/raw/expA.mp4 --result results/A_offline
```

终端运行期间，浏览器访问 `http://127.0.0.1:8800/`。检查 B 时换为 `expB.mp4`、`B_offline`；同时开第二个服务需加 `--port 8801`。检查第 4 节新结果时，把 `--result` 改成 `results/A_readme_trial`。

操作顺序：

1. 选择一个物体或轨迹，查看它在原始视频中的轮廓和位置。
2. 把已检查的时间段标为 **Trusted**（可信）、**Untrusted**（不可信）或保留 **Unreviewed**（未检查）；不可信时说明原因。
3. 若同一物体被程序换了编号，检查建议后建立对应关系；不能仅为了编号少而合并不同物体。
4. 使用 **Export & check** 核对导出结果，必要时制作检查视频。

这是轨迹审核界面，不是早期逐帧重画轮廓的界面。操作会保存到对应结果目录的 `review.json`，原始 `objects.csv`、`events.csv` 不改写。导出包含 `trusted.csv`、`untrusted.csv`、`unreviewed.csv`、`objects_reviewed.csv` 和 `events_reviewed.csv`。

**轨迹编号 `id`** 是自动追踪的连续片段编号；**物体编号 `object_id`** 是人工核对后用于连接同一物体的编号。后续单物体分析需要考虑后者，也要在真实融合、分裂或不可信片段处切开。

如果页面一直 Loading，先看终端是否还在运行及是否报错；确认 `--video` 是原始视频、目录内有 `contours.npz`，并等待启动解码完成。端口被占用时换 `--port`。不要用重新生成追踪结果的方式覆盖已有审核来解决页面问题。

## 6. 运动、形变、群体变化与 PZT 响应

### 6.1 先分析微泡本身

下面接续第 4、5 节，假设新结果已经人工审核：

```bash
cd /home/lzy/projects/new_microbubble_experiment/customer_workspace/microbubble_tracking
../../.venv/bin/python analysis/analyze.py \
  --result results/A_readme_trial --video data/raw/expA.mp4
../../.venv/bin/python analysis/viewer.py \
  --result results/A_readme_trial --video data/raw/expA.mp4
```

第一条生成 `results/A_readme_trial/analysis/`；第二条在 `http://127.0.0.1:8810/` 打开曲线与视频对照页面。也可仅运行第二条查看已有 `results/A_offline` 或 `results/B_offline`。

| 输出                                 | 回答的问题                                               |
| ------------------------------------ | -------------------------------------------------------- |
| `segments.csv`                     | 哪些可信轨迹段可以连起来计算，在哪里因分合或不可信而切开 |
| `motion.csv`                       | 移动了多远、往哪里走、速度多大                           |
| `shape.csv`                        | 面积、长宽比、圆度、凸度等如何变化，哪些形状测量不可靠   |
| `collective.csv`                   | 整个视野有多少区域，微团面积占比、分布范围如何变化       |
| `field_events.csv`、`onsets.csv` | 群体事件与运动开始时刻                                   |
| `state.csv`、`report.json`       | 统一状态表、参数、数据来源、自检和质量说明               |

离线速度默认用前后各 0.05 秒的位置差计算；60 fps 时通常对应前三帧到后三帧，片段边缘缩短窗口，不跨片段强行连接。知道显微镜标定后可加 `--um-per-px` 输出微米单位；不知道时保留像素单位。`--fps` 只应填写有依据的真实时间换算，不能为了让速度好看而修改。

单物体运动与形变主要使用可信段。群体统计不能简单只剩下可信目标：当前实现排除明确标成“不是微泡”的区域，其他拿不准的区域保留并记录不确定性。小区域、贴壁或被画面截断时，形状指标也有局限。

### 6.2 再把 PZT 动作接进来

```bash
cd /home/lzy/projects/new_microbubble_experiment/customer_workspace/microbubble_tracking
../../.venv/bin/python analysis/pzt_analysis.py \
  --result results/A_readme_trial --pzt data/pzt/expA --video data/raw/expA.mp4
```

输出在该结果目录的 `analysis/pzt_analysis/`：

- `commands.csv`、`actions.csv`：实际命令、动作分段与对应时间。
- `aligned_frames.csv`：每帧对应的四路 PZT 状态和过渡范围。
- `responses.csv`：每次动作后各物体的变化。
- `model_samples.csv`、`model_steps.csv`：供进一步建模使用的整理表；世界模型当前还会按自己的规则重新从自动轨迹构造样本。
- 图表和 `report.json`：动作响应、自检及时间精度说明。

新采集的视频必须配它自己的动作记录；上述 `expA` 只适用于同一 A 原始录像，不能拿来给新实验“套时间轴”。分析会核对视频身份。缺少时间对应数据时，仍可分析微泡运动，但不能声称知道每帧对应哪个 PZT 动作。

**重复运行 `analyze.py` 或 `pzt_analysis.py` 会更新其固定子目录内的同名分析文件。** 它们不会重写原始录像和审核，但不是只读查看命令。已有验收结果优先用 `viewer.py` 查看；需要新计算时按新的结果批次保存，不直接覆盖已归档科学结果。`tools/run_experiments.py` 是批量复现入口，也会写结果，不应当作普通查看按钮。

### 6.3 时间对应能说明什么

目前的关联使用同一 Windows 主机记录的命令时间、接收/解码时间以及视频帧时间，**不直接相减 Windows 与 WSL 各自的计时器**。

AFG 的“确认”是在稳定时段读回参数，不是测到模拟电压改变的瞬间；相机接收/解码时间也不是曝光时间。动作日志能定位对应的视频帧范围，但尚不能给出曝光与超声实际生效之间的物理同步误差。不要把晚约一秒安排的读回当成一秒仪器响应延迟。

## 7. 如果要开展真实实验，先选时间表，再运行实验入口

本节命令会连接真实设备。只查看用法时加 `--help`，不会启动实验。可以全程在 WSL 终端操作：启动程序会调用已有 Windows Python，由 Windows 与 AFG、相机通信，图像分析在 WSL 中运行。

操作顺序是：**先选“哪几路、什么时候工作”的时间表，再选“只控制 PZT”或“同时录像”的入口，最后检查保存的结果。** 设备配置已经统一，不需要在两份幅值上限不同的配置之间选择。

### 7.1 配置和状态的含义

| 文件                                                                  | 用途                                                                                     |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| [configs/afg_video.json](configs/afg_video.json)                       | 统一 AFG 设备配置：设备序列号、通道对应、通信设置；四路幅值上限 0.8 Vpp、频率上限 10 MHz |
| [configs/four_pzt_video_20s.json](configs/four_pzt_video_20s.json)     | 四路联动的 20 秒动作表                                                                   |
| [configs/two_action_video_20s.json](configs/two_action_video_20s.json) | PZT1 → PZT3 → PZT1 的 20 秒动作表                                                      |
| [../configs/chronos_config.json](../configs/chronos_config.json)       | Chronos 接收配置                                                                         |
| [configs/video_stability.json](configs/video_stability.json)           | 62 秒 × 4 次的持续运行检查条件                                                          |

AFG 驱动、控制界面、时间表执行器、联合录像和 AFG 专项测试都默认读取 `customer_workspace/configs/afg_video.json`。旧的 `configs/afg.json` 已移除，`afg_video_800mv.json` 已重命名；旧命令若明确写了这些路径，需要换成新路径。历史实验记录保留当时的路径和参数，不随本次修改重写。

`max_input_vpp: 0.8` 表示**允许的最大幅值**，不表示一启动就输出 0.8 Vpp。实际工作幅值和频率由时间表的 `settings` 决定，超过配置上限会在执行前被拒绝。这里的幅值是 AFG 输出、送入功放的信号幅值，不是功放之后 PZT 两端的电压。

- **IDLE**：输出保持开启，幅值为 0.001 Vpp，不等于电气关闭。
- **ACTIVE**：选定通道提高到时间表要求的工作幅值；其余通道待机。
- **STOP**：真正关闭全部四路输出。

时间表中的 `duration_s` 是动作总时长，`settings` 是各路频率和工作幅值，`steps` 是每个 `at_s` 时刻进入哪个状态。第一步从 0 秒 IDLE 开始，最后在总时长进入 STOP。通用执行器支持每步的 `frequencies_hz`；未填写的频率保持不变。

目前下面两张 20 秒时间表都使用 **2.5 MHz、工作 0.8 Vpp、待机 0.001 Vpp**：

| 时间表                        | 实际动作                                                                                                                                |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `four_pzt_video_20s.json`   | 0–3 秒全部待机；3–6 秒 PZT1；6–9 秒 PZT2；9–12 秒 PZT3；12–15 秒 PZT4；15–18 秒 PZT1 + PZT3；18–20 秒全部待机；20 秒关闭全部输出 |
| `two_action_video_20s.json` | 0–4 秒全部待机；4–8 秒 PZT1；8–12 秒 PZT3；12–16 秒 PZT1；16–20 秒全部待机；20 秒关闭全部输出                                      |

以下命令分别展示两种入口，不需要把两条实验命令都运行一遍。每次使用一个新的结果目录；不要覆盖以前的实验。

### 7.2 只控制 PZT，不录像

```bash
cd /home/lzy/projects/new_microbubble_experiment
.venv/bin/python -m customer_workspace.experiment_runner \
  customer_workspace/configs/four_pzt_video_20s.json \
  --output-dir customer_workspace/validation/pzt_trial
```

这里先给出时间表路径，程序自动使用统一的 `afg_video.json`，无需再填写 `--config`。`pzt_trial` 必须不存在。需要双动作实验时，将时间表换成 `customer_workspace/configs/two_action_video_20s.json`。`--repeat N` 可重复同一计划。不指定输出目录时，通用执行器默认保存到工程 `records/年月日-时分秒/`。

执行过程是：先关断、设置初始参数和核查，四路进入待机，再按同一个时间起点执行。远离计划时刻时可取消地等待，最后约 22 ms 用高精度计时循环等到计划时刻；这不是提前发命令，也不是硬件保证。普通切换只发送命令，不逐条插入查询。结束、取消或捕获异常后尝试 `all_off()`，并保存核查和失败原因。

同一计划时刻要求改频率和幅值时，程序连续发送两类命令；它们并非仪器内部原子地同时生效。日志保留各条实际发送时间。正常关断确认也不能代替物理电压测量；通信断线、强杀进程或断电时，软件无法保证关断。

结果中可查看 `plan.json`、`events.jsonl`、`summary.json`、`parameter_commands.csv` 和 `all_attempts.csv`。特别检查 `data_reliable`、`all_outputs_off_confirmed` 及失败原因，不只看流程是否结束。

### 7.3 录像、现场分析与 PZT 同时运行

先确认 Windows AFG 环境、GStreamer 环境、相机连接，以及当前视野对应的背景图可用。
新入口使用 `--background` 指定准备好的 `.npy` 背景，不再读取旧 `--geometry` 范围文件。
背景应与当前视野、尺寸和亮度范围一致；程序只能检查尺寸及数值，不能替用户判断是否换了显微镜视野。
下例的 `YOUR_BACKGROUND.npy` 是需要替换的路径。

```bash
cd /home/lzy/projects/new_microbubble_experiment
.venv/bin/python -m customer_workspace.afg_video_check \
  --background YOUR_BACKGROUND.npy \
  --plan customer_workspace/configs/two_action_video_20s.json \
  --output-dir customer_workspace/validation/avsync_trial
```

当前联合入口限定 **20 秒、频率固定的调幅时间表**，实际录像为 22 秒。默认计划是四路联动；上例显式选择双动作计划。它在动作之间安排稳定读回，结束时先等待 AFG 收尾，再结束视频接收。取消时在启动终端按 Ctrl+C，等待关闭与保存完成，不要强杀设备进程。

这一入口也自动使用同一份 `afg_video.json`。需要四路联动时，仅把 `--plan` 后的文件换成 `customer_workspace/configs/four_pzt_video_20s.json`。`avsync_trial` 必须不存在。

此入口已通过 `parallel_video_check.py` 接入新的 `microbubble_tracking.OnlineProcessor`，负责逐帧识别、追踪和结果保存。Windows 接收并保存压缩视频；独立的 WSL 进程解码，再把最新画面交给分析模块。

**验证范围：新模块已完成代码接入和已有视频模拟验证，但接入后的“真实相机＋AFG”联合实验尚未运行。** 模拟测试中录像完整保存，分析仍有少量跳帧，尚不能保证 60 fps 每帧都处理。详见[相机接入验证报告](validation/camera_tracking_20261031_final/report/REPORT.md)。

主要记录包括 `recording/video.mp4`、录像帧时间、`tracking/tracking.sqlite`（识别与追踪结果）、`analysis_frames.csv`（实际分析的帧）、`frame_outcomes.csv`（逐帧去向与缺口）、`afg_experiment/`、`afg_link_summary.json` 和 `link_session.json`。时间零点关联到正式录像窗口的首个压缩帧接收记录。

实验完成后的报告和合成视频操作见[第 7.4 节](#74-视频检查实验报告与观看视频)。

### 7.4 视频检查、实验报告与观看视频

本节集中放置视频采集检查和实验后的结果整理命令，均不发送 PZT 控制命令。前半部分需要连接相机；后半部分只读取已有实验记录，不连接设备。不需要每次实验都运行全部检查。

#### 7.4.1 视频采集检查：需要连接相机

单独检查相机接收、录像保存和画面分析。接入相机或排查视频问题时使用。

| 入口                                                | 用途                                                               | 是否控制 AFG             |
| --------------------------------------------------- | ------------------------------------------------------------------ | ------------------------ |
| [streamcopy_recording.py](streamcopy_recording.py)   | 只保存相机压缩视频，检查录像是否完整；不分析微泡                   | 否，但连接相机           |
| [parallel_video_check.py](parallel_video_check.py)   | `--mode baseline` 仅分析对照；`--mode parallel` 同时录像与分析 | 单独运行时否，但连接相机 |
| [stability_check.py](stability_check.py)             | 默认 62 秒 × 4 次，检查录像、分析速度、延迟、队列与内存变化       | 否，但连接相机           |
| [video_recording_check.py](video_recording_check.py) | 早期旧录像链路的对照检查                                           | 否，但连接相机           |

**20 秒独立录像检查：只录像，不分析微泡**

使用 `streamcopy_recording.py` 单独检查“相机传来的视频能否正常保存”。适合首次检查录像，或在录像出现问题时排除微泡分析的影响；不需要每次联合实验前都运行。

从 WSL 启动后，程序调用已配置的 Windows Python 和 FFmpeg，连接 Chronos 相机，把收到的压缩视频直接保存，不重新编码，也不经过只保留最新画面的分析通道。它不识别或追踪微泡，不需要 `--background`，不发送 AFG 控制命令。

确认相机和 Windows 录像环境可用，并先结束占用同一相机接收端口的其他采集程序，再运行：

```bash
cd /home/lzy/projects/new_microbubble_experiment
.venv/bin/python -m customer_workspace.streamcopy_recording \
  --seconds 20 \
  --expected-fps 60 \
  --output-dir customer_workspace/validation/streamcopy_trial
```

`--seconds 20` 是目标视频时长；`--expected-fps 60` 表示当前已确认的输入帧率为每秒 60 帧。程序据此计划保存 1200 帧，结束后再核对实际播放时间；这个参数不会设置相机帧率，也不会补帧。若相机实际帧率不同，应填写真实值。连接、等待可用起始画面和结束检查另需时间，命令总运行时间会超过视频时长。

相机配置默认读取工程根目录的 `configs/chronos_config.json`，可用 `--config` 指定另一份相机配置。`streamcopy_trial` 必须是尚不存在的新目录；不指定 `--output-dir` 时，程序自动新建 `customer_workspace/validation/streamcopy_年月日-时分秒/`。

运行结束后，程序自动检查并保存结果：

| 文件（位于本次输出目录）                     | 看什么                                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `video.mp4`                                | 本次保存的原始压缩视频，可打开检查播放是否正常                                                    |
| `recording_check.json`                     | `passed` 表示检查是否通过，`problems` 列出失败原因；`counts` 核对录像窗口内收到和保存的帧数 |
| `frame_times.csv`、`recording_check.png` | 检查每帧播放时间和相邻帧间隔，寻找明显时间中断；取得有效帧数据后生成                              |
| `capture_session.json`、`ffmpeg.log`     | 记录本次配置、采集过程及异常，便于排查失败原因                                                    |

这里能核对电脑收到的画面是否完整保存，不能证明相机或网络在此之前没有丢帧。由于本次没有运行微泡分析，结果不能作为第 7.4.2 节报告的分析速度参照；需要该参照时，使用下面的“22 秒录像与分析检查”。

**22 秒录像与分析检查**

这条命令会连接真实相机，连续采集 22 秒，同时保存录像、识别和追踪微泡。它不发送 AFG 控制命令，也不会自动改变 AFG 已有的输出状态。

`--mode parallel` 表示录像与分析同时运行，`--seconds 22` 表示正式采集时长。将 `YOUR_BACKGROUND.npy` 换为当前视野准备好的背景路径，并使用尚不存在的输出目录：

```bash
cd /home/lzy/projects/new_microbubble_experiment
.venv/bin/python -m customer_workspace.parallel_video_check \
  --background YOUR_BACKGROUND.npy \
  --mode parallel --seconds 22 \
  --output-dir customer_workspace/validation/video_baseline_trial
```

这次运行会记录录像是否完整、每秒分析多少帧以及延迟多大。结果目录 `video_baseline_trial/` 也可作为本节后续报告命令的 `--baseline` 速度参照；它与用于识别微泡的 `--background` 背景图是两种不同输入。

用于速度比较时，应使用相同的新分析程序、图像尺寸、帧率、背景和识别设置，画面复杂程度与电脑负载也应尽量接近。已有符合条件的结果时可以复用；旧分析程序的结果不能作为新追踪的速度参照。

**较长时间的稳定性检查：默认 62 秒 × 4 次**

需要检查持续运行是否逐渐变慢、积压或丢帧时，再运行：

```bash
cd /home/lzy/projects/new_microbubble_experiment
.venv/bin/python -m customer_workspace.stability_check \
  --background YOUR_BACKGROUND.npy \
  --output-dir customer_workspace/validation/stability_trial
```

录像支路完整保存“已收到的帧”，不代表网络或相机从未丢帧；报告同时检查接收序列、保存帧数和时间。分析支路只处理最新画面，跳过旧画面与录像额外丢帧是不同问题。

#### 7.4.2 实验结束后整理结果：不连接设备

完成第 7.3 节的联合实验后，`avsync_trial/` 中保存了本次实验的原视频、分析结果和 PZT 动作记录。下面两步整理这些已有记录，生成报告和便于观看的视频，均不连接设备。

当前报告程序还要求提供一份不执行 PZT 控制任务的速度参照。采集方法见本节前面的“22 秒录像与分析检查”；已有符合条件的参照结果时可直接使用，不必每次重拍。

**1. 整理已经完成的实验：只读已有记录，不连接设备。**

这条命令读取两个目录：`avsync_trial/` 是有 PZT 动作的实验，`video_baseline_trial/` 是本节“22 秒录像与分析检查”采集的速度参照。它完成两件事：把本次实验每条 PZT 命令及读回确认对应到视频中的时间范围；比较加入 PZT 控制后，分析速度和延迟是否明显变差。报告写入新目录 `avsync_trial_report/`。

**当前程序的限制：报告命令必须填写 `--baseline`，因为时间对应和性能比较写在同一个报告流程中。动作与视频的时间对应本身只依赖本次实验的记录，不依赖另一段视频；这里的参照也不是用来比较超声对微泡的作用。** 尚无速度参照时，实验原始记录仍已保存，但不能直接运行下面这条完整报告命令。

```bash
cd /home/lzy/projects/new_microbubble_experiment
.venv/bin/python -m customer_workspace.afg_video_report \
  customer_workspace/validation/avsync_trial \
  --baseline customer_workspace/validation/video_baseline_trial \
  --output-dir customer_workspace/validation/avsync_trial_report
```

报告中的 `actions.csv` 和 `video_frames.csv` 可查动作及对应视频时间，`link_report.json` 和时间轴汇总检查结果。新追踪结果还会生成 `actions_tracking.csv`，列出动作范围中实际分析过的帧，缺口不会冒充已分析。报告生成不等于所有检查都通过，仍需查看其中的失败原因。

**2. 制作观看视频：只读本次实验和第 1 步报告，不连接设备。**

这条命令把原视频与 PZT 时间轴放在一起，让人能边看微泡变化，边看当时执行了哪些 PZT 动作。`--report-dir` 指向第 1 步生成的报告目录：

```bash
cd /home/lzy/projects/new_microbubble_experiment
.venv/bin/python -m customer_workspace.pzt_video_review \
  customer_workspace/validation/avsync_trial \
  --report-dir customer_workspace/validation/avsync_trial_report
```

合成脚本在本次实验目录中生成新的 `review_年月日-时分秒/`，含 `review.mp4`、预览、逐帧对应表和来源清单。观看其中的 `review.mp4` 即可；原始录像保留，合成视频只用于审查。

### 7.5 没有相机时，验证新追踪接入

**这一节的目的是**，在没有连接相机时，把已有录像当作相机输入，检查新识别追踪程序能否接入实际实验使用的录像、传图和保存流程。适合接机前检查软件，或修改程序后重复验证。

**对于下面这条命令**，它读取实验 A 的原始视频 `expA.mp4`，按原来的播放速度送入 22 秒画面，模拟相机不断传来新画面。Windows 一路将视频保存到新目录，另一路把视频数据交给 WSL；WSL 解码、识别和追踪微泡，并保存分析结果。原视频不修改。本次不连接相机、不启动 AFG，但仍需要已配置好的 Windows 和 WSL 运行环境。

**运行后主要看三件事：**

1. 送入的画面是否全部保存到新录像，有没有额外丢帧。
2. 识别追踪能否跟上送帧速度，有没有跳帧、长时间停顿或延迟增加。
3. 已分析的轮廓、编号、速度和事件是否全部保存。

这项检查通过后，才能说明本次模拟条件下的软件流程正常；真实相机的网络传输和现场成像仍需接机测试。下面给出单次检查命令，文件位置和结果查看方法列在命令之后：

```bash
cd /home/lzy/projects/new_microbubble_experiment
.venv/bin/python -m customer_workspace.parallel_video_check \
  --offline-video customer_workspace/microbubble_tracking/data/raw/expA.mp4 \
  --background customer_workspace/validation/camera_tracking_20261031_final/background_A.npy \
  --seconds 22
```

这里的背景仅用于这段已验证的 A 视频。模拟结果默认进入新建的
`validation/camera_tracking_年月日-时分秒/`。接相机时去掉 `--offline-video`，换成该视野
经检查的背景；相机地址和 UDP 设置默认读 `configs/chronos_config.json`，也可用
`--camera-config` 指定。当前接口支持已验证的 60 fps、无 B 帧的 H.264、yuv420p 输入；
尺寸、格式或时间异常会报错，不能把错误流继续当成有效实验。

| 保存文件                                                 | 如何检查                                                 |
| -------------------------------------------------------- | -------------------------------------------------------- |
| `recording/video.mp4`、`recording_check.json`        | 原始压缩录像及收到/保存帧数核对                          |
| `tracking/tracking.sqlite`                             | 轮廓、目标编号、速度、临时目标、合并/分离及延迟确认事件  |
| `analysis_frames.csv`                                  | 每张实际分析画面的帧号、视频时间、处理耗时和软件延迟上界 |
| `frame_outcomes.csv`                                   | 录像窗口内每帧是否已分析；已解码但未分析、未解码分别标记 |
| `video_frames.csv`                                     | 原始视频、源帧编号和两端时间的对应关系；两端时间分列     |
| `analysis_check.json`、`tracking/writer_report.json` | 实际速度、跳帧、保存数量、等待队列是否清空               |

`all_window_frames_analyzed` 表示本次窗口是否每帧都分析了；`recording_passed` 表示录像是否完整。
先打开 `analysis_check.json` 看这两项，再核对 `processed`（已分析帧数）与 `saved`（已保存分析结果的帧数）是否一致，以及 `pipeline_error` 是否为空。程序正常退出不等于所有检查都通过。

**帧号和时间记录用来做什么？** 用它们找出哪张画面已分析、哪张被跳过，以及对应原视频的哪个时间。源编号是本次电脑接收的压缩帧编号，不是相机内部曝光计数。Windows 接收时间和 WSL 解码时间分别保存；计算微泡运动速度使用视频自身的时间，不直接相减两套系统的计时器。这些记录不代表实际曝光时间。

启动时先准备分析和解码空间，再开始正式窗口。解码后的最新画面使用三块有明确使用顺序的
共享空间：正在分析的空间不能覆盖。压缩帧有前后依赖，不在解码前随意丢弃；等待过多时明确
报告错误。停止或取消时先完成 PZT 收尾（若有），再结束接收，并等分析结果全部写盘。
`--mode baseline` 只分析、不保存视频，帧编号只是本次观察窗口的位置。

**需要更完整的检查时**，再运行下面的命令：除了多次回放和较长时间测试，还会比较保存的视频与原视频是否一致，以及同一批实际输入帧的追踪结果是否与离线处理一致。它会生成新数据，耗时数分钟，不操作设备：

```bash
.venv/bin/python -m customer_workspace.tools.camera_tracking_validation \
  --output customer_workspace/validation/camera_tracking_recheck --rounds 2 --long
```

它先用原离线回放测 A/B，再各进行两次 22 秒 Windows 录像与分析，以及各一段 62 秒重复素材
负载检查，最后逐像素核对录像、逐表核对追踪结果。重复素材只是负载测试，不是新的科学实验。
真实相机网络、现场曝光与背景适配仍须接机验证；软件时间对应不等于曝光与超声的物理同步。
持续运行配置已改为新模块的 **60 fps 逐帧处理目标**；出现分析跳帧会如实判为未达到该项，
不再使用旧 PIV 的 5.6 次/秒门槛。录像完整、分析完成和分析没有跳帧是三项分别核对的结果。

### 7.5.1 用 Windows 本地 RTSP 模拟相机验证正式链路

上面的 `--offline-video` 让 Windows 直接读取文件；这里改为从
`F:\MB_camera_sim` 接收正在发送的 RTSP 视频，经过正式网络接收、分流录像和 WSL 分析流程。
识别、追踪、解码和保存算法保持原样；使用独立配置，不覆盖 `configs/chronos_config.json`。

**第一步：在一个 WSL 终端启动 Windows 模拟相机，并保持运行。**

```bash
/mnt/f/MB_camera_sim/.venv/Scripts/python.exe -X utf8 -B \
  F:/MB_camera_sim/windows_camera_sim.py serve
```

这条命令实际启动 Windows 程序，同时发布 `rtsp://127.0.0.1:8554/expA` 和
`rtsp://127.0.0.1:8554/expB`。Windows 接收器访问自己的 `127.0.0.1`；不需要
把模拟源开放到局域网，也不要求 WSL 直接访问该地址。

**第二步：在另一个 WSL 终端，依次测试 A、B。** 每个输出目录必须是新目录：

```bash
cd /home/lzy/projects/new_microbubble_experiment
.venv/bin/python -m customer_workspace.parallel_video_check \
  --camera-config customer_workspace/configs/rtsp_sim_A.json \
  --background customer_workspace/validation/camera_tracking_20261031_final/background_A.npy \
  --mode parallel --seconds 22 \
  --output-dir customer_workspace/validation/rtsp_trial_A

.venv/bin/python -m customer_workspace.parallel_video_check \
  --camera-config customer_workspace/configs/rtsp_sim_B.json \
  --background customer_workspace/validation/camera_tracking_20261031_final/background_B.npy \
  --mode parallel --seconds 22 \
  --output-dir customer_workspace/validation/rtsp_trial_B
```

这里不加 `--offline-video`。两份测试配置明确指定本机源和 UDP 5500–5501 接收端口，
模拟源的控制连接保持同一个会话；未指定 `rtsp_url` 的原相机配置仍采用已有 Chronos 兼容方式。
这只影响连接源的方式，视频数据仍走原有分流与管道。

**第三步：检查结果，再停止模拟相机。** 按第 7.5 节的文件表核对录像与分析数量。
`tracking_bridge.json` 的 `decoded` 才是新路线的 WSL 解码记录；
`recording_check.json` 中旧字段 `decoded_frames_total: 0` 指 Windows 没有进行图像解码，
不代表 WSL 没有分析。测试结束，在模拟相机终端按 `Ctrl+C`。

直播可能从原视频的任意位置接入，也可能跨过播放末尾、循环回到开头。
这只是软件负载测试，回到开头产生的画面突变不能当作新的真实微泡运动。
本次逐帧去向、源视频对应位置、跳帧前后耗时和离线结果核对，均见上述 RTSP 报告。

### 7.5.2 单独测量 Windows→WSL 延迟

**要回答的问题是：时间花在传输，还是解码和微泡分析？** 原来的
`software_age_upper_bound_ms` 包含 GET 请求等待下一帧的时间，不能把它当成
管道传输时间。需要准确检查时，在第 7.5.1 节的运行命令中增加 `--measure-latency`：

```bash
.venv/bin/python -m customer_workspace.parallel_video_check \
  --camera-config customer_workspace/configs/rtsp_sim_A.json \
  --background customer_workspace/validation/camera_tracking_20261031_final/background_A.npy \
  --mode parallel --seconds 22 --measure-latency \
  --output-dir customer_workspace/validation/latency_trial_A
```

模拟源需按上一节提前启动。B 使用对应的 B 配置、背景和新输出目录。
诊断默认关闭；开启后，每张已分析画面都会立即向 Windows 返回带帧号的完成确认，
由 Windows 自己计算“收到画面 → 收到分析完成确认”的时间，不混减 Windows 与 WSL 时钟。
视频仍走原管道，解码、识别、追踪方法不变。确认表示分析完成并提交保存，
全部写盘仍在结束时单独核对；被跳过的帧不会伪造完成确认。

新增记录如下：

| 文件 | 内容 |
| --- | --- |
| `latency_windows.json` | Windows 收帧、发出和收到逐帧确认的时间，以及 Windows 时钟说明 |
| `latency_wsl_decoder.json` | WSL 收齐、解码、转换、复制及发布图像的时间，确认线程异常 |
| `latency_wsl_analysis.json` | 每张分析画面的开始、完成时间及对应源帧编号 |

只测管道、不等下一帧时，使用下面的工具。它不需要启动 RTSP 模拟源，先把视频的真实
H.264 压缩包准备到 Windows 内存，再按原管道格式逐包发送；不解码，也不重新编码：

```bash
.venv/bin/python -m customer_workspace.tools.pipe_latency_check \
  --video customer_workspace/microbubble_tracking/data/raw/expA.mp4 \
  --output customer_workspace/validation/pipe_latency_trial_A
```

单向传输报告**上下界**，不把往返时间除以二；准确值仍无法从两台未对齐的程序时钟直接求出。
本次 A/B 统计、计时定义、逐帧明细及诊断开关对照见
[延迟实测报告](validation/latency_20261031-205734/report/REPORT.md)。
该结果从 Windows 应用程序收到压缩帧开始计时，不包括相机曝光、编码、到电脑的网络时间，
也不代表超声与曝光的物理同步。起点位于 GStreamer 完成拆包、即将分流的位置，
此前的 RTSP 接收缓冲不包含在这项时间中。

### 7.5.3 对照测试：最多等待一帧或两帧

**默认仍是 `--waiting-frames 1`。** 当分析正在忙，下一张解码画面可以等待；
再有新画面到达时，原来等待的画面会被替换。测试选项 `--waiting-frames 2`
允许多留一张：按到达顺序分析，等待区满时只替换最旧的待取画面。
**有一张就能立即分析，不会故意等到两张凑齐。** 正在分析的画面始终不能被覆盖。

先按第 7.5.1 节启动 F 盘模拟相机，再运行：

```bash
.venv/bin/python -m customer_workspace.parallel_video_check \
  --camera-config customer_workspace/configs/rtsp_sim_A.json \
  --background customer_workspace/validation/camera_tracking_20261031_final/background_A.npy \
  --mode parallel --seconds 22 --measure-latency --waiting-frames 2 \
  --output-dir customer_workspace/validation/two_waiting_frames_trial_A
```

对照组改为 `--waiting-frames 1` 并使用另一个新目录；B 使用相应的 B 配置和背景。
不要同时跑两组来比较性能，以免测试本身争抢资源。

两帧等待多准备一块图像和帧头，当前尺寸约增加 1.31 MiB。它只能容纳短时成批到达的
画面，不能解决分析长期跟不上输入的问题。应同时看分析跳帧、反馈 P95/P99、最长反馈和
交接等待，不能只追求零跳帧而忽略结果已经变旧。

`tracking_bridge.json` 新增 `waiting_capacity`、`peak_waiting_frames`、
`pending_at_shutdown` 和 `replacement_events`，分别记录等待上限、最多等了几张、
结束时剩余数量，以及每次被替换的源帧号和替换它的新帧号。
正常结束先处理完等待区，再完成原有保存收尾；取消或出错仍保留未分析帧的去向。

本次 A/B 各重复三次的计划、图表、逐帧延迟和结果核对见
[等待区对照报告](validation/queue_compare_20261031-211619/report/REPORT.md)。
测试选项没有修改识别、追踪算法，也没有改动正式默认设置。

### 7.5.4 查清画面在等待区里耗在哪里

需要短时排查时，在上一节命令中增加 `--diagnose-queue`，并使用新的输出目录。
它会自动启用完成确认，分别记录画面放入等待区、被取出、开始分析、完成分析、
发出确认和 Windows 收到确认的时间。默认不启用，正常实验无需添加。

- `queue_producer.json`：写入共享图像的次数、字节数、耗时，以及申请空位、发布时的锁等待。
- `queue_consumer.json`：取帧、归还空间的时间和锁等待；取帧直接读取共享空间，没有额外整图复制。
- `queue_loop.json`：分析循环中的状态检查、保存进度检查、内存采样与等待新画面的时间。
- `latency_wsl_analysis.json`：实际分析开始、结束和主线程运行计数。
- `latency_windows.json`：Windows 收到画面到收到分析完成确认的时间，用 Windows 同一个时钟计算。

这些明细先存在内存，短测结束后写盘；诊断本身也有小量开销，不应常开。
等待时间使用 WSL 同一个时钟；总反馈使用 Windows 同一个时钟，两边的原始时间数值不能直接相减。

[本轮等待原因报告](validation/queue_delay_20261031-215045/report/REPORT.md)
逐一检查了上一轮超过 16.67 ms 的 5 帧，并用同起点文件回放和 RTSP 模拟源分别比较。
配对时按原视频画面核对，不把两个独立运行中的同一个接收编号直接当成同一画面。

### 7.5.5 定位分析内部哪一步突然变慢

短时排查可以在第 7.5.1 节的命令中增加 `--diagnose-analysis`。它复用已有的
分析计时位置，不改变轮廓规则、追踪规则、自动清理设置或等待区容量。
默认关闭；正常实验不用添加。

- `consumer_stalls.csv`：平滑、暗度、区域识别、轮廓、追踪、记录提交等分项耗时，
  以及主线程 CPU 和缺页记录。`frame` 是接收源帧号。只有内核确实启用调度统计时
  才填写等待 CPU 的时间；当前 WSL 未启用，空值不能理解为没有等待。
- `consumer_stalls.gc.csv`：Python 自动清理发生在哪段时间；清理时间已经包含在
  上述步骤里，不能再加一次。
- `analysis_stages.json`：对应的源帧号、录像帧号和原有更细的计算分项。
- `consumer_stalls.meta.json`：实际分析进程、线程和诊断设置。

诊断最多缓存 2048 帧的分项记录后批量写入；本轮 22 秒短测在结束时集中保存。
这些诊断文件与实验数据的独立保存线程分开，退出时保留剩余记录。
诊断本身会增加负担，因此先跑不开此开关的对照，再解释分项诊断结果。

Linux 的 CPU 时间或缺页计数不能单独排除 Windows 宿主影响。系统级记录需要另外
明确启动；此开关不会自动发起管理员窗口、暂停其他程序或改变系统配置。
[分析内部慢帧结论](validation/analysis_stalls_20261031-221809/FINDINGS.md)
说明本轮 A/B 重复测试的主要发现和证据边界；
[逐帧分项报告](validation/analysis_stalls_20261031-221809/report/REPORT.md)
保存每个慢帧的具体数据。

### 7.5.6 独立验证 GPU 解码：尚未接入正式追踪

这项检查回答“WSL 能否把 H.264 直接解码到 RTX 4060 显存，并在显存里继续处理”。
它只读已有 A/B 视频，不连接相机或 AFG，不调用正式追踪，也不改变正式解码默认设置。
[本轮报告](validation/gpu_decode_20261031-224639/report_final/REPORT.md) 保存像素核对、
重复计时及数据搬运证据。
当前结果：显存内解码和图像处理可用，但按约 60 FPS 逐帧运行时，GPU 解码比 CPU 更慢，
尚未证明能改善正式流程的延迟，因此继续保留原 CPU 解码。

进一步的[耗时来源检查](validation/gpu_decode_root_20261031-231519/report_final/REPORT.md)
把主要额外时间定位到 WSL 与 Windows 显卡驱动的逐帧同步交接。
例如一帧总计 27.37 ms，其中两次资源锁定请求等待了 23.61 ms。
后续的[Windows 根因检查](validation/gpu_host_cause_20261031-235434/REPORT.md)
已在同一次系统记录中复现这类长等待：Windows 收到 `0xC5` 硬件管理通知后，
独占同一个显卡适配器，暂停并恢复其工作上下文，WSL 的解码请求因此排队。
最长例子中，两次独占共持续 33.13 ms，对应一次资源锁定请求等待 33.94 ms。
后续[通知来源检查](validation/acpi_trigger_20261101-004740/REPORT.md)已定位到本机
“机械革命电竞控制台”的 `GCUService.exe`：它通过 WMI 调用固件 `WMBC` 方法，随后出现通知。
控制台程序中存在每 1.5 秒根据 CPU 使用率调整“CPU/GPU 联合运行”设置的逻辑；
本轮未取得每次调用的参数，因此没有将每一次通知都认定为同一个策略分支。
这一结论针对已核实的长锁定请求，不代表所有慢帧都由同一原因造成。
正式解码器和显卡电源设置均未改变。

测试代码为 [gpu_decode_check.py](tools/gpu_decode_check.py)。CPU 对照使用原 `.venv`，
GPU 测试使用独立的 `customer_workspace/.venv_nvdec`；依赖版本记录在
[nvdec_requirements.txt](tools/nvdec_requirements.txt)。本机已安装，不需要修改原环境。
在其他具备 NVIDIA 驱动和 CUDA 的 WSL 环境中，首次准备方法为：

```bash
cd /home/lzy/projects/new_microbubble_experiment
python3 -m venv customer_workspace/.venv_nvdec
customer_workspace/.venv_nvdec/bin/python -m pip install -r customer_workspace/tools/nvdec_requirements.txt
```

例如，用 A 视频按原播放时间测试 GPU 解码，每帧再执行一个简单的 GPU 图像处理示例：

```bash
customer_workspace/.venv_nvdec/bin/python -m customer_workspace.tools.gpu_decode_check bench \
  --video customer_workspace/microbubble_tracking/data/raw/expA.mp4 \
  --engine gpu --paced \
  --output-dir customer_workspace/validation/gpu_decode_A_trial
```

CPU 对照使用 `.venv/bin/python`，把 `--engine gpu` 换成 `--engine cpu`，另选新的输出目录。
`--paced` 表示按原视频时间送来压缩帧；不加它则尽快处理，不能把这种吞吐量测试当作实时延迟。
每次输出目录必须不存在。`frames.csv` 中分开记录解码完成、图像操作完成、实际送帧晚到时间；
这些时间不包含相机曝光、Windows 接收、跨系统传输或微泡追踪。

需要核对整段视频的像素时另行运行：

```bash
customer_workspace/.venv_nvdec/bin/python -m customer_workspace.tools.gpu_decode_check verify \
  --video customer_workspace/microbubble_tracking/data/raw/expA.mp4 \
  --output-dir customer_workspace/validation/gpu_decode_A_verify
```

`verify` 会显式把结果下载到 CPU 逐像素比较，因此其耗时不用于速度结论。
图像示例是原始亮度上的小范围均值、暗度和阈值，**不是正式微泡识别**。
原生亮度与原追踪的“彩色转换后灰度”仍有差异；当前库的 `ZERO` 模式也不能直接提供可靠的输出 PTS，
本测试只对已验证的一包一帧、无 B 帧输入保留原包时间。正式接入前须处理这两点。
现有追踪在 CPU，单独更换 GPU 解码仍可能需要下载整图，不能据此宣称整条分析已在 GPU 上运行。

### 7.5.7 Windows 原生 GPU 解码对照：独立测试

这一项检查“绕开 WSL，在 Windows 直接解码是否更快，以及能否避开显卡管理通知造成的停顿”。
独立运行环境在 `F:\MB_gpu_decode`；权威脚本是
[gpu_native_decode.py](tools/gpu_native_decode.py)，安装范围、运行指令与记录含义见
[使用说明](tools/WINDOWS_NVDEC.md)。正式实验仍沿用原解码方式。

[2026-11-01 对照报告](validation/windows_nvdec_20261101-002553/REPORT.md)：
A/B 每端各回放三遍，Windows 的中位解码耗时约 2.58 ms，WSL 约 3.18–3.38 ms。
但 Windows 原生仍会被 `0xC5` 通知期间的显卡独占操作阻塞；系统诊断中确认了一帧
总计 42.60 ms，其中原生解码线程等锁 38.04 ms。
这些是独立解码测试，不包含相机接收、跨系统传图或微泡分析，不能当作整条实验链路的性能。

### 7.5.8 查明显卡管理通知来自哪里：只读诊断

[本轮报告](validation/acpi_trigger_20261101-004740/REPORT.md)保存通知来源、对照结果和证据位置。
两次系统采集中，58 次 `0xC5` 通知都能对应到电竞控制台的同一类硬件管理调用。
不运行测试视频时也会触发；单路 B 解码最慢一帧为 37.50 ms，其中等显卡锁占 33.22 ms。
因此，单纯换成 Windows 解码不能避免这类停顿。

本轮只增加诊断工具，未停止控制台服务，未修改电源方案、驱动、固件或正式实验流程。
目前尚未验证一个既能消除通知、又不影响电脑功耗和散热管理的设置。

- [gpu_acpi_conditions.py](tools/gpu_acpi_conditions.py)：依次静置、单路解码、双路解码，再倒序重复，保留 Windows 时间和帧记录。
- [windows_acpi_trace.wprp](tools/windows_acpi_trace.wprp)：记录通知、固件方法、WMI 调用和显卡锁；需 Windows 管理员确认。
- [acpi_trigger_report.py](tools/acpi_trigger_report.py)：只读已有记录，核对调用来源、通知和解码等待。
- [dotnet_method_inspect.py](tools/dotnet_method_inspect.py)：只读取指定 .NET 文件的程序指令，不执行或修改被检查的软件。

需要复测时，使用尚不存在的新结果目录。以下是独立 F 盘测试，最长记录 180 秒；不会连接实验设备：

```bash
.venv/bin/python -m customer_workspace.tools.gpu_decode_host_trace \
  --native-root /mnt/f/MB_gpu_decode --profile Acpi --conditions \
  --output-dir customer_workspace/validation/acpi_conditions_trial
```

2026-11-01 的[服务暂停前置检查](validation/gcu_safety_20261101-011022/REPORT.md)
已找到本机 CPU/GPU 温度和两只风扇的独立读取路径，但复查 CPU 温度仍为 93–95°C，
因此没有暂停服务，也没有运行暂停前后的解码对照。没有找到可确认能单独关闭
该定时检查的官方开关；不能把 Dynamic Boost 直接当成等价设置。
[温度探针](tools/gcu_thermal_probe.ps1)只针对已检查的本机型号和控制台版本，
需要管理员权限，只读取传感器，不会暂停服务。后续暂停仍需温度条件满足、
自动恢复机制验证以及用户明确授权。

## 8. 图表、动画与 baseline 比较

### 8.1 先看已整理结果

成品入口：[实验 A 图表目录](microbubble_tracking/figures/expA/README.md)、[实验 B 图表目录](microbubble_tracking/figures/expB/README.md)。其中按分割、追踪、运动形变与群体分析、刺激响应、baseline 比较分别组织；配套表格和来源记录也在各自目录中。

这些图说明已有数据，不自动构成独立准确率评测。baseline 目前是固定阈值识别与 CSRT 单目标追踪，多目标展示来自分别初始化的运行；它没有与本方法等价的融合/分裂事件分类器，不能把画面上的场景标题当成它识别出的事件。

### 8.2 从现有 A/B 结果重新导出

以下脚本读取 `microbubble_tracking/results/A_offline` 或 `B_offline`；不是接收任意 `--result` 的通用绘图入口。原始录像、审核及分析记录必须匹配，缺数据时先完成第 4～6 节对应流程。

| 模块                                                | 内容                                                                                  |
| --------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [segmentation_ppt.py](segmentation_ppt.py)           | 背景、相对暗度、阈值、形态处理和二值轮廓；`--background-curve` 画真实单像素亮度曲线 |
| [tracking_ppt.py](tracking_ppt.py)                   | 运动预测、重叠矩阵、匹配、融合分裂示例                                                |
| [analysis_ppt.py](analysis_ppt.py)                   | 可信轨迹切段、运动、形状、群体指标和原视频核对动画；`--step` 选择部分内容           |
| [stimulus_response_ppt.py](stimulus_response_ppt.py) | 动作后响应示例及响应比例                                                              |
| [baseline_animations.py](baseline_animations.py)     | 当前方法与 baseline 的识别、追踪、事件对比动画                                        |

例如重新导出 B 实验的群体指标动画：

```bash
cd /home/lzy/projects/new_microbubble_experiment
.venv/bin/python -m customer_workspace.analysis_ppt \
  --experiment B --step 4 \
  --output-dir customer_workspace/microbubble_tracking/figures/expB_collective_trial
```

其他示例，分别写入新的输出目录：

```bash
cd /home/lzy/projects/new_microbubble_experiment
.venv/bin/python -m customer_workspace.segmentation_ppt --experiment B --background-curve
.venv/bin/python -m customer_workspace.segmentation_ppt --experiment B --frame 253
.venv/bin/python -m customer_workspace.tracking_ppt --experiment B
.venv/bin/python -m customer_workspace.stimulus_response_ppt --experiment B
```

不指定 `--output-dir` 时，这些脚本通常在 `validation/` 新建带时间的目录；要放进 `microbubble_tracking/figures/`，应明确指定新的子目录。当前演示导出使用英文和实际 Arial，默认从 `/mnt/c/Windows/Fonts` 读取字体。缺字体时应配置相应入口支持的字体路径或准备正确字体，不用相似字体冒充。

### 8.3 重做 baseline 对比动画

同一新目录按顺序运行；`prepare` 已包含本方法的自动重算，不需要再重复运行 `ours` 阶段：

```bash
cd /home/lzy/projects/new_microbubble_experiment
.venv/bin/python -m customer_workspace.baseline_animations prepare \
  --experiment B --output-dir customer_workspace/validation/baseline_B_trial
customer_workspace/.venv_baseline312/bin/python -m customer_workspace.baseline_animations masks \
  --output-dir customer_workspace/validation/baseline_B_trial
customer_workspace/.venv_baseline312/bin/python -m customer_workspace.baseline_animations baseline \
  --output-dir customer_workspace/validation/baseline_B_trial
.venv/bin/python -m customer_workspace.baseline_animations render \
  --output-dir customer_workspace/validation/baseline_B_trial
```

需要已有 MobileSAM 代码与权重、baseline 环境，以及对应实验的审核和分析结果。此流程进行模型推理与追踪，不训练模型。单独的定量比较入口是 [baseline/compare.py](microbubble_tracking/baseline/compare.py)，掩膜生成入口是 [baseline/channel_mask.py](microbubble_tracking/baseline/channel_mask.py)；参数和方法差异见 [子项目说明](microbubble_tracking/README.md)。

## 9. 世界动作模型：先预测不同 PZT 动作会带来什么变化

本节整合《微泡世界模型.md》（2026-10-30）的说明，并与当前源码和已有结果文件核对。详细技术记录见[子项目 README 的“阶段二”](microbubble_tracking/README.md#阶段二世界动作模型world_model)。

**它要解决什么问题？** 已经知道微泡团在哪里、怎样运动后，再问：“接下来这样设置 PZT，0.1、0.5、1 秒后，它们会去哪里、面积和形状如何变化、会不会合并或分裂？”模型可以分别预测几种候选动作，为之后挑选动作提供依据。

**目前做到哪一步？** 数据整理、模型训练、对照评估和模型保存已有实现；新录像上的预测还不够可靠。模型尚未接入实时相机结果形成自动决策，也没有接管 AFG。第 7.3 节仍按预先写好的时间表执行，不会根据世界模型自行改变动作。

### 9.1 各模块负责什么

| 模块 | 对实际流程的作用 |
| --- | --- |
| [dataset.py](microbubble_tracking/world_model/dataset.py) | 把自动追踪、PZT 时间记录及原视频整理成“当前状态与动作 → 后续变化”的训练和测试样本 |
| [models.py](microbubble_tracking/world_model/models.py) | 实现图网络预测、合并/分裂概率，以及保持不动、匀速、动作规律表等比较方法 |
| [run.py](microbubble_tracking/world_model/run.py) | 训练各模型，在不同录像上比较误差，检查动作作用和预测把握程度，保存模型与报告 |
| [synthetic.py](microbubble_tracking/world_model/synthetic.py) | 用简化的模拟芯片检查：增加动作样本后，方法能否学到运动规律；不控制真实设备 |
| [wm_figures.py](microbubble_tracking/world_model/wm_figures.py) | 将已有误差、动作比较、合并概率和全视野分布结果画成图 |
| [architecture.py](microbubble_tracking/world_model/architecture.py) | 绘制网络结构；当前模板尚未适配新增输入和事件预测部分，见第 9.6 节 |

### 9.2 从实验记录到预测，具体怎样运行

**1. 准备输入和答案。** 一个样本包含某一时刻画面里的全部微泡区域：位置、面积、形状、离壁距离、过去约 0.3 秒的运动，以及计划执行的四路 PZT 幅值。训练时另外提供未来 1 秒、每隔约 0.1 秒的实际变化作为答案。这里的时间步按现有 60 fps 数据设置。

主模型读取的是追踪得到的数字，不直接把整张显微图像交给神经网络。原视频用来补充每个区域“有多暗”，并计算全视野的分布。当前命令中的 `--video` 应填写原视频；缺少它时，代码会把暗度和分布网格置零，不能当作完整的新版本输入。

**本轮新增的信息有三类：**

- **区域暗度**：比较区域内部亮度与背景亮度，再抵消整体亮度变化。它提供外观差异线索，尚不是经过标定的厚度或微泡数量测量。
- **周围拥挤程度**：统计区域中心周围 60、150 像素范围内其他区域的面积占比。模型每向前预测一步，都按预测位置重新计算。
- **全视野分布图**：将画面划成 40×40 像素的小格，记录每格的区域面积和暗度加权面积，用来检查整体分布预测是否正确。

轨迹在合并、分裂处切开，不把变化前后强行当成同一个目标。短于 0.5 秒或被画面边缘截断的轨迹段不参与主模型的训练误差计算，但仍可作为周围环境信息。主模型使用自动轨迹；人工审核结果用于可信子集评估，另有专门“只用审核数据训练”的对照版本。合并/分裂标签来自自动追踪，尚不等于人工确认的真实物理事件；离开视野或跟丢后的未知部分，不按“没有事件”计分。

**2. 用附近区域的关系预测下一步。** 图神经网络把每个区域看成一个点，将相距 150 像素以内的区域连起来。每一步先交换 3 轮信息，再结合区域自身状态、PZT 幅值、离上次切换多久和全视野最近运动，预测下一小步的位置、面积和长宽比；连续推算 10 步，得到约 1 秒的预测。

模型同时给出各步合并、分裂的可能性。事件预测还参考预测位置上的最近间隙和靠近速度，因此改变候选 PZT 动作也可能改变事件概率。训练时以实际事件发生率初始化这部分，运动部分训练结束后再单独训练事件预测部分。**预测“可能合并”不代表程序已经把两个预测目标合成一个；当前向前推算仍不会自动新建、删除或重组区域。**

**3. 估计预测的把握程度。** 默认训练 5 个网络，综合它们之间的分歧和各自估计的误差范围。分歧大提示预测不稳，但输出的“90% 范围”是否真的覆盖约 90% 的情况，还要用测试数据核查，不能直接当作可靠保证。

**4. 与简单方法比较，再用未参与训练的录像检查。** 位置预测比较“保持不动”“按原速度继续走”“查询已有动作规律表”和梯度提升树；事件预测比较平均发生率、看距离与靠近速度的规则、梯度提升树。另比较去掉邻居、去掉暗度和拥挤程度、只用审核数据训练的版本，检查各部分是否有帮助。

当前主要评估是用 A 训练、B 测试，再反过来；同一录像内按不重叠时间块评估只作补充。还会比较“按真实动作切换”和“假如不切换”的预测，检查模型是否使用了动作信息；预测会随动作变化，并不自动说明这种变化是正确的。

**5. 保存模型供后续调用。** `model/graph_world_model.pt` 可由 `GraphWorldModel.from_state(...)` 恢复，再通过 `predict(...)` 给出预测。实时追踪结果怎样组装为模型输入、怎样从预测中挑动作并安全执行，仍需后续开发。

### 9.3 已有结果说明了什么

以下是现有 `results/world_model/` 中保存的结果及子项目说明，不是本次文档更新重新训练得到的结果。数据目前只有 A、B 两段各 1320 帧的录像，记录的 PZT 切换共 10 次；自动追踪记录了 30 次合并、19 次分裂。上万个“区域×时刻”样本，不能当作上万个独立实验。

**最关键的结果：在新录像上，位置预测尚未超过“保持不动”。** 下表为未来 1 秒的平均位置误差，单位为像素，越小越好，来源是 [metrics.csv](microbubble_tracking/results/world_model/metrics.csv)：

| 训练与测试 | 保持不动 | 图网络世界模型 |
| --- | ---: | ---: |
| A 训练，B 测试 | 2.50 | 14.98 |
| B 训练，A 测试 | 15.13 | 19.60 |

其他结果应这样理解：

- **在学过的录像上能拟合动作变化**：子项目记录的切换阶段预测明显优于保持不动，但这是训练录像上的表现，不能据此认为新实验也有效。
- **合并预测有信号，但不稳定**：未来 1 秒的排序指标 AUC 在 A→B 为 0.918，B→A 为 0.661。AUC 表示能否把真正将合并的区域排在更高可能性的位置，不是“猜对了多少百分比”。A→B 的概率误差优于平均发生率，B→A 则未超过；来源见 [events.csv](microbubble_tracking/results/world_model/events.csv)。
- **分裂预测尚未表现出可靠改善**：事件少，且有些标签可能来自轮廓暂时被拆开。模型已有分裂概率输出，实际预测能力仍需验证。
- **暗度和拥挤程度**：现有一次对照未显示位置误差改善，但改善了合并概率的准确程度；事件数量少，不能直接推广到所有实验。
- **全视野分布和形状**：图网络尚未稳定超过保持不动。分布比较把预测位置和面积画成圆盘再放入网格，不能理解为已经预测出未来每个区域的真实像素轮廓。
- **把握程度有参考价值，但尚未校准好**：模型说不确定的地方通常更易出错，给出的范围仍可能过宽或过窄。
- **已有速度记录**：5 个网络对 15 个区域比较 6 种候选动作、各预测 1 秒，合计约 105.5 ms，见 [report.json](microbubble_tracking/results/world_model/report.json)。这仅是该次模型预测计时，不包含相机、识别、传输、保存和 PZT 控制，不能据此承诺整个系统每 0.1 秒完成一轮控制。

**模拟数据检查支持继续增加数据，但不保证真实实验一定成功。** 在四段未参与训练的简化模拟录像上，图网络的 1 秒位置误差随训练切换次数增加而下降：

| 模拟训练中的切换次数 | 图网络误差（像素） | 保持不动误差（像素） |
| ---: | ---: | ---: |
| 13 | 48.1 | 30.1 |
| 52 | 23.3 | 30.1 |
| 158 | 10.3 | 30.1 |

来源为 [synthetic.json](microbubble_tracking/results/world_model/synthetic/synthetic.json)。模拟中没有真实声场、合并、分裂或暗度变化，只验证了简化运动规律；不能由此断言真实实验只需 150 次切换。现有证据提示动作样本少、装片间差异大是重要问题，但仍需新数据确认。

还需保留一个评估边界：模型输入和事件训练方式已经根据 A/B 的早期结果调整过，所以这两段录像属于开发数据，不能再当作完全未见过的最终验收数据。报告的 `all_checks_passed` 表示数据划分和模型读回等程序检查通过，不代表预测效果或实机自动控制已经通过验收。

### 9.4 先查看已有结果

结果目录为 [results/world_model/](microbubble_tracking/results/world_model/)，直接打开已有表格和图即可，不需要重新训练。

| 文件或目录 | 看什么 |
| --- | --- |
| `metrics.csv` | 位置、面积和长宽比误差，按预测时长、切换阶段及目标类型分开比较 |
| `action_check.csv` | 按真实动作和假设不切换时，哪个预测更接近实际 |
| `uncertainty.csv`、`uncertainty_bins.csv` | 模型的把握程度是否与实际误差相符 |
| `events.csv`、`events_field.csv` | 每个区域的合并/分裂概率，以及整个视野的预期事件次数 |
| `field.csv`、`field_grid.csv` | 全视野中心、分布范围和分布网格的误差 |
| `predictions.csv` | 各模型逐项预测的位置和误差范围 |
| `report.json`、`training_history.csv` | 输入规模、设置、划分、耗时、自检及训练过程 |
| `model/` | 用所提供的全部实验训练得到的模型；不能用训练录像上的成绩代表新实验成绩 |
| `fold_models/` | 为跨录像比较保存的模型；与上面的全数据模型用途不同 |
| `cache/` | 原视频暗度和分布网格的计算缓存 |
| `synthetic/synthetic.json` | 简化模拟中的数据量与预测误差比较 |

建议先看 [errors.png](microbubble_tracking/results/world_model/figures/errors.png) 的误差比较，再看 [events.png](microbubble_tracking/results/world_model/figures/events.png) 的合并概率表现和 [field_map.png](microbubble_tracking/results/world_model/figures/field_map.png) 的全视野分布。`field_map.png` 选的是测试录像中未来 1 秒合并较多的时刻，用于展示，不代表平均效果；总体结论应看对应表格。[imagined.png](microbubble_tracking/results/world_model/figures/imagined.png) 展示同一状态换不同 PZT 动作的预测，不能当作真实执行后的照片。

### 9.5 训练、重新评估与模拟检查：按需执行

运行位置是 `customer_workspace/microbubble_tracking`，需要 PyTorch 和 scikit-learn，依赖见 [world_model/requirements.txt](microbubble_tracking/world_model/requirements.txt)。下面沿用现有 `customer_workspace/.venv_baseline312` 环境；这是训练命令，不是普通查看命令，本次文档合并不会执行它。

至少准备两段实验的自动追踪结果，并先完成第 6 节的 PZT 对应分析，使每段结果具有 `analysis/pzt_analysis/aligned_frames.csv`。同时提供匹配的原视频，以计算暗度和分布网格。新训练使用新输出目录，避免覆盖已有权重和评估证据：

```bash
cd /home/lzy/projects/new_microbubble_experiment/customer_workspace/microbubble_tracking
../.venv_baseline312/bin/python world_model/run.py \
  --exp A=results/A_offline --exp B=results/B_offline \
  --video A=data/raw/expA.mp4 --video B=data/raw/expB.mp4 \
  --out results/world_model_trial
```

- 新增可用于开发的数据时，成对增加 `--exp C=results/C_offline` 和 `--video C=data/raw/expC.mp4`。
- 中断后继续同一次运行，可在原命令上增加 `--resume`，保留原输入、设置和输出目录。程序复用已经完成的测试轮次，全部完成后清理本次 `partial/` 中间结果。
- 只重画已有结果图，可在对应运行命令上增加 `--figures-only`。它不会训练，但会更新指定目录的图件；只看已有图时不必运行。
- 当前 `run.py --help` 仍把 `--video` 标成“figures only”，但新 `dataset.py` 实际也用它提取暗度和分布网格。请按上面的完整输入填写。

需要重新检查模拟数据时，使用另一个新目录；它也会训练模型和花费较长时间，不连接设备：

```bash
cd /home/lzy/projects/new_microbubble_experiment/customer_workspace/microbubble_tracking
../.venv_baseline312/bin/python world_model/synthetic.py \
  --out results/world_model_synthetic_trial
```

**注意数据划分：** 当前 `run.py` 按 `--exp` 对应的录像分开测试，最后还会用所有提供的实验训练最终模型。它不会自动按“同一次装片的多段录像”分组，也不会自动保留永久封存的最终测试集。因此，后续多装片方案需要先补齐分组和冻结流程，不能把封存测试数据直接加入这个训练命令。

### 9.6 架构图的当前状态

已有 [Mint Gold 版](microbubble_tracking/figures/world_model_architecture_refined/) 和 [Lime 版](microbubble_tracking/figures/world_model_architecture_lime/) 是旧版本结构图，未包含本轮新增的暗度、拥挤程度和事件预测部分。

当前 `architecture.py` 能读出输入已增加到 48 项，但绘图模板仍要求旧版的 45 项，并假设没有事件预测部分，因此结构检查会报 `Architecture changed: node_input=48; update the diagram.`。在模板更新前，不把旧图或原来的绘图命令当作本轮可用成果。本次只更新 README，不修改绘图代码或覆盖旧图。

### 9.7 与项目目标的关系

| 项目目标 | 世界模型已经提供的部分 | 仍需完成的部分 |
| --- | --- | --- |
| 聚集控制 | 预测位置、合并概率和全视野分布 | 在新实验上提高准确度，并用这些预测实际挑选动作 |
| 导航控制 | 比较不同动作下约 1 秒的运动路线和离壁关系 | 加入不能穿墙等限制，并持续重新规划；目前预测路径仍可能穿墙 |
| 形状控制 | 预测面积和长宽比 | 预测效果尚不足，也没有预测团内部结构或完整未来轮廓 |
| AI 自动控制 | 保存/恢复模型，比较候选动作，输出不确定范围 | 在线追踪到模型的输入接口，以及挑选动作、执行和安全检查的控制程序 |
| 从 4 路扩到 16 路 PZT | 图网络的思路可以继续使用 | 当前数据代码仍为 4 路；需要修改动作输入尺寸、补数据和重新训练，不能直接控制 16 路 |
| 流体作用与血栓清除 | 当前未建模 | 新增流体、血栓等观测和任务定义 |
| k-Wave、APECSS 物理模拟 | 当前只有简化运动模拟 | 尚未接入这些物理模拟；未来可研究先用模拟学习、再用真实数据校正 |

文稿中的“已接上目标”应理解为已经提供了部分预测信息，不表示聚集、导航或 AI 控制任务已经在真实设备上完成。文稿所说的“阶段一”是让 AI 直接利用显微画面的研究路线，不是第 7 节的相机接入步骤。

### 9.8 下一批实验怎样收集：研究计划，尚未执行

下一步优先增加不同装片、不同初始状态下的动作记录。一次装片指同一次注射和芯片使用过程；同一次装片可录多段，但这些录像不能分到训练和最终测试两边。

- **先做小批，再决定总量**：文稿建议先做 5 次装片，查看预测误差是否随数据增加下降；总目标暂定至少 15 次装片、约 50–60 段录像和 4000 次切换。这是数据采集计划，不是已经验证足够的数量。
- **动作顺序随机且数量均衡**：频率、工作幅值先沿用 2.5 MHz、0.8 Vpp，待机保持 0.001 Vpp；考虑四个单路、少量双路组合及待机，每种动作保持约 0.5–3 秒。随机安排用于减少固定顺序带来的混淆，不能单凭随机化就保证因果结论。
- **每段前后记录待机状态**：计划在开头和结尾各留约 30 秒观察自发漂移，中间加入参考动作和随机切换。这个约 3 分钟方案尚未接入现有实验入口；第 7.3 节目前仍限定 20 秒动作、22 秒录像，不能直接拿该入口运行长方案。
- **保存能追溯的原始记录**：保留原视频、帧时间、PZT 发送与读回，以及芯片、装片、注射时间、曝光、帧率、标定、复位方式和异常。光照和曝光尽量固定，避免把成像变化误当作区域暗度变化；同时检查时间对应、动作执行和追踪质量，并人工抽查部分片段。
- **实验前固定最终测试装片**：这些数据不用于训练和调参数。已有 A/B 已参与方法开发，不能替代新的封存测试。按装片划分的代码流程仍需补齐。

每批数据按事先确定的标准决定下一步：如果新装片上 0.5、1 秒的位置预测超过“保持不动”和“动作规律表”，合并预测至少达到距离规则水平，再推进 AI 规划控制；如果误差仍随数据增加下降，继续采集；如果增加数据也不再改善且错误集中在贴壁、密集、变形区域，再研究直接看画面的模型。

更详细的装片、随机动作和检查方案见[子项目“后续新实验方案”](microbubble_tracking/README.md#后续新实验方案为世界动作模型收集数据)。幅值档位、更多组合、相位控制、由 AI 挑选有信息价值的动作，以及物理模拟辅助训练，均属于后续方向，不是当前已验证功能。

## 10. 专项诊断与早期工具：需要时再用

### 10.1 AFG 与录像诊断

| 模块                                                                                                                                        | 作用                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| [afg_timing.py](afg_timing.py)                                                                                                               | 为现有 AFG 驱动记录真实通信调用时间                          |
| [wait_calibration.py](wait_calibration.py)                                                                                                   | 不连接仪器，测量电脑等待晚醒，帮助选择循环等待提前量         |
| [timing_benchmark.py](timing_benchmark.py)                                                                                                   | 实机定时只发送、读回或组合时序测试                           |
| [amplitude_pair_diagnostic.py](amplitude_pair_diagnostic.py)                                                                                 | 输出关闭时连续两次调幅，检查最终读回和错误；仍会连接并写仪器 |
| [visa_diagnostics.py](visa_diagnostics.py)                                                                                                   | 输出关闭状态下排查通信错误，不把查询失败当成指令丢失         |
| [four_pzt_validation.py](four_pzt_validation.py)                                                                                             | 多路控制、调频调幅和取消等专项验收                           |
| [timing_report.py](timing_report.py)、[plot_timing.py](plot_timing.py)                                                                        | 读取已有时序记录，生成表格和图                               |
| [streamcopy_report.py](streamcopy_report.py)、[parallel_video_report.py](parallel_video_report.py)、[stability_report.py](stability_report.py) | 读取录像及分析记录，检查完整性、并行性能和持续运行情况       |

通信错误 `VI_ERROR_IO`、`Syntax error`、查询无回复有各自的原始失败记录，不能只根据错误码断言硬件根因。诊断目录保留失败和不可靠轮次；统计时不能把它们悄悄当作成功样本。

### 10.2 早期识别、Mask 与模型试验

| 一组模块                                                                                                                                                                                                       | 原来解决的问题                                                                        |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [bubble_regions.py](bubble_regions.py)、[bubble_dark_regions.py](bubble_dark_regions.py)、[bubble_dark_preview.py](bubble_dark_preview.py)                                                                        | 早期暗色区域候选与参数预览                                                            |
| [bubble_region_review.py](bubble_region_review.py)、[bubble_region_annotation.py](bubble_region_annotation.py)、[bubble_frame_review.py](bubble_frame_review.py)、[bubble_review_edits.py](bubble_review_edits.py) | 生成审阅画面、逐帧修轮廓、检查帧间关系及保存修订                                      |
| [bubble_region_calibration.py](bubble_region_calibration.py)、[bubble_region_comparison.py](bubble_region_comparison.py)                                                                                         | 根据人工标注检查识别参数及前后变化                                                    |
| [channel_mask.py](channel_mask.py)、[mask_shrink_review.py](mask_shrink_review.py)、[mask_shrink_sections.py](mask_shrink_sections.py)                                                                            | 智能剪刀绘制流道、线条光顺、比较整体或分段收缩参数，保存真实二值 Mask                 |
| [bubble_tracking.py](bubble_tracking.py)、[bubble_tracking_review.py](bubble_tracking_review.py)、[bubble_tracking_display.py](bubble_tracking_display.py)、[bubble_region_video.py](bubble_region_video.py)       | 连接早期已保存轮廓，导出编号、事件与对照视频                                          |
| [bubble_benchmark.py](bubble_benchmark.py)、[benchmark_report.py](benchmark_report.py)                                                                                                                           | 固定困难片段、人工关键帧与对应关系、冻结检查及报告                                    |
| [sam21_initial_tracking.py](sam21_initial_tracking.py)、[cutie_initial_tracking.py](cutie_initial_tracking.py)                                                                                                   | 只用首帧轮廓初始化，后续独立追踪                                                      |
| [sam21_tracking_review.py](sam21_tracking_review.py)、[initial_tracking_comparison.py](initial_tracking_comparison.py)                                                                                           | 包装 SAM/Cutie 结果，检查空预测及不同方法的差异                                       |
| [joint_sam21_runner.py](joint_sam21_runner.py)、[joint_region_fusion.py](joint_region_fusion.py)、[joint_tracking_review.py](joint_tracking_review.py)                                                            | 早期 OpenCV 与 SAM 逐帧互相检查的候选实现，详见 [JOINT_TRACKING.md](JOINT_TRACKING.md) |

旧 `bubble_tracking.py` 与当前 `microbubble_tracking/bubble_tracker.py` 名字相近，但不是同一套算法。旧 Mix A Mask 及其人工边界也不会自动成为当前背景分割器的输入；当前 `bubble_tracker.py` 没有接受该 Mask 的命令行参数。

需要继续检查旧数据时可运行：

```bash
cd /home/lzy/projects/new_microbubble_experiment
.venv/bin/python -m customer_workspace.bubble_benchmark status \
  customer_workspace/validation/bubble_benchmark_20261026-120008
.venv/bin/python -m customer_workspace.bubble_benchmark annotate \
  customer_workspace/validation/bubble_benchmark_20261026-120008
```

这是旧逐帧标注工作台，可能按其配置转到后续重识别数据；它不是第 5 节的轨迹审查网页。自动轮廓和建议编号不等于人工标准答案；只有实际检查并保存的审核才算确认。固定测试数据的 `input_lock.json`、已冻结修订与不确定区域必须保留，不能为新方法改换测试条件。

## 11. 历史证据、检查方法和使用边界

### 11.1 关键证据入口

以第四轮重跑及之后的实验作为当前 A/B 来源；更早实验供排查和方法演变追溯。

| 记录                                                                            | 内容                                             |
| ------------------------------------------------------------------------------- | ------------------------------------------------ |
| [parallel_20261025-145352](validation/parallel_20261025-145352/)                 | 完整录像与现场分析并行的对照                     |
| [stability_20261025-151417](validation/stability_20261025-151417/)               | 62 秒多次运行、队列、延迟与内存检查              |
| [四路重跑摘要](validation/avsync_20261025-161356_rerun/overview.json)            | 正常、取消及主动异常条件下的联合记录；对应实验 A |
| [双动作摘要](validation/avsync_20261025-165523_two_action/overview.json)         | PZT1 → PZT3 → PZT1 联合记录；对应实验 B        |
| [固定困难片段](validation/bubble_benchmark_20261026-120008/index.html)           | 早期统一检测追踪检查数据                         |
| [Mix A 结果](validation/bubble_detection_20261026-163801_mix_a/video/index.html) | 早期 Mask 收缩与重识别记录                       |
| [SAM 初始轮廓试验](validation/sam21_initial_20261026-170243/)                    | 独立模型追踪及空预测记录                         |
| [联合候选报告](validation/joint_sam21_20261026-190838/RUN_REPORT.md)             | 早期 OpenCV + SAM 互相检查的实现与局限           |

其他历史目录继续保存在 `validation/`。查具体失败时，先找同一次运行的摘要、原始日志、输入配置和校验清单，不混合不同批次的数字。

### 11.2 软件检查

本工作区测试不一定被工程默认 `pytest` 自动收集；按模块明确指定。例如只检查相关的软件逻辑：

```bash
cd /home/lzy/projects/new_microbubble_experiment
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest \
  customer_workspace/tests/test_experiment_runner.py \
  customer_workspace/tests/test_pzt_video_review.py -q
```

这些软件测试使用测试数据或替身对象，不是实机验收。`tests/check_*.py` 是专项界面检查，运行条件各不相同；不要把所有脚本无差别当成日常启动流程。

### 11.3 保存与解释结果的原则

- 原始录像、人工审核、PZT 读回、Mask 来源和已验收结果保留；新试验与新图表使用新的输出目录。
- 审查视频和图表用于帮助理解，统计计算应回到原始数据表和来源记录。
- 当前两段录像已经反复开发与检查，不能当作完全未见过的独立测试集。人工只审过的部分，不能代表整个视野都已有真值。
- 模型预测、图像测量、仪器读回和物理测量含义不同；不要把它们之间尚未测定的差异隐藏掉。
- Git 同步范围按 [Git 工作流](../docs/GIT_WORKFLOW.md) 和 [数据策略](../docs/DATA_POLICY.md) 执行；同步代码不等于备份所有视频、环境和权重。

本文对应当前源码入口。此次整理核对了主要模块、参数和帮助命令，没有重新采集实验、训练模型或重算已验收结果。更详细的算法参数、文件列含义与研究结果请继续阅读 [microbubble_tracking/README.md](microbubble_tracking/README.md)。
