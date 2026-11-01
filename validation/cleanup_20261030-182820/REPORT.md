# 工作区清理记录与执行结果

用户已明确批准删除清单中的 16 个文件。复查后全部删除，共 120,613,628 字节
（115.03 MiB），没有跳过项或错误。删除前检查了活动进程，没有发现目标文件被打开
或映射；删除后重新校验了 57 个保留文件，内容全部与批准清单一致。
详见 `execution_preflight.json` 和 `execution_result.json`。

本轮范围仅限 `/home/lzy/projects/new_microbubble_experiment/customer_workspace`，没有操作其他项目、Windows 设备程序、远程仓库或 Git 历史。

## 已执行

- 删除 96 个未被 Git 跟踪、存在对应源码且未观察到被进程打开的 Python 字节码缓存，共 2,437,622 字节（2.32 MiB）。逐项路径、文件摘要和对应源码见 `cache_candidates.json`、`cache_result.json`。只删除清单中的文件，再移除已经空掉的缓存目录，没有递归删除代码目录。
- 新增 `customer_workspace/.ignore`，日常 ripgrep 搜索跳过验证输出、模型、外部依赖及追踪项目的数据、结果、图表。文件仍原位保留，Git 提交范围及程序读取路径不变。
- 更新工作区 README 顶部的当前在线开发入口；在工作区 AGENTS 中说明按任务选择文档和指定实验记录，避免默认扫描整个历史目录。
- 本次普通搜索可见文件由 21,833 个降至 200 个。这个数字衡量搜索范围，不代表删除了两万多个文件，也不表示已经清除了已有聊天历史。后续通过明确路径或在指定目录使用 `rg --no-ignore` 仍能查看记录。

## 已检查但保留

- 工作区磁盘占用约 25 GiB，其中验证记录约 22 GiB。包括原始视频、人工标注、Mask、PZT 时间、对照数据库及失败证据，不能因为时间较早就删除；没有核实到覆盖它们的独立备份。
- 25 个早期识别、标注、Mask、SAM/Cutie 及联合方法模块合计 477,060 字节。静态检查发现 111 处导入关系，详见 `legacy_code_references.json`。其中包含旧模块之间及测试的引用，不意味着全部属于当前在线入口；它们也没有作为一个可删除的独立单元完成调用方、文档和历史复现核对，因此本轮保留。
- 没有删 Python 环境、第三方依赖、模型权重、SQLite 数据库或原始系统诊断记录。未修改识别、追踪、保存与仪器控制算法。

两个未通过逐字节一致性核对的压缩包保留：

- `customer_workspace/microbubble_tracking/figures/expB/slide_04_multi_object_tracking/tracking_figures.zip`：成员 `step_1_motion_prediction.pdf` 与相邻展开文件不完全一致或不存在，不能认定为纯重复包。
- `customer_workspace/microbubble_tracking/figures/expB/slide_08_stimulus_response_analysis/slide8_assets.zip`：成员 `02_response_summary.pdf` 与相邻展开文件不完全一致或不存在，不能认定为纯重复包。

## 已批准并执行的删除清单

以下 **16 个文件，共 120,613,628 字节（115.03 MiB）已全部删除**。所有路径相对工程根目录。`pending_delete.json` / `pending_delete.csv` 原样保留批准前的清单，实际执行状态以 `execution_result.json` 为准。

- 6 个交付 ZIP：每个成员均与同目录展开文件逐字节一致；保留展开后的动画、图片、数据和报告。清单逐项记录保留路径及摘要。
- 10 个 perf 文本导出：保留原始 `.data` 系统记录、已精确筛选的 `.window.txt`、计时 CSV、报告和原导出命令。已重新筛选全文并验证其关键窗口与保留文本一致。删除的是重复展开大量窗口外事件的阅读副本，不是原始测量。
- 执行前已重新检查文件摘要、修改时间、Git 状态和活动进程；本次没有文件变化或占用。执行后已确认目标路径消失、保留文件摘要不变。

| 已删除路径 | 大小（MiB） | 保留依据 |
| --- | ---: | --- |
| `customer_workspace/microbubble_tracking/figures/expB/baseline_comparison/result_animations.zip` | 20.02 | 同目录展开文件全部相同 |
| `customer_workspace/microbubble_tracking/figures/expB/slide_06_motion_shape_and_collective_analysis/collective_metrics/slide6_animations.zip` | 3.29 | 同目录展开文件全部相同 |
| `customer_workspace/microbubble_tracking/figures/expB/slide_06_motion_shape_and_collective_analysis/kinematics/slide6_animations.zip` | 0.90 | 同目录展开文件全部相同 |
| `customer_workspace/microbubble_tracking/figures/expB/slide_06_motion_shape_and_collective_analysis/morphology/slide6_animations.zip` | 0.53 | 同目录展开文件全部相同 |
| `customer_workspace/microbubble_tracking/figures/expB/slide_06_motion_shape_and_collective_analysis/raw_video_validation/slide6_animations.zip` | 0.44 | 同目录展开文件全部相同 |
| `customer_workspace/microbubble_tracking/figures/expB/slide_06_motion_shape_and_collective_analysis/trajectory_segmentation/slide6_animations.zip` | 7.03 | 同目录展开文件全部相同 |
| `customer_workspace/validation/remaining_stalls_20261030-175436/A_0_candidates.txt` | 7.21 | 原始记录、关键窗口、导出命令均保留 |
| `customer_workspace/validation/remaining_stalls_20261030-175436/A_0_tracking.txt` | 7.21 | 原始记录、关键窗口、导出命令均保留 |
| `customer_workspace/validation/remaining_stalls_20261030-175436/A_1846_blur.txt` | 7.21 | 原始记录、关键窗口、导出命令均保留 |
| `customer_workspace/validation/remaining_stalls_20261030-175436/B_2947_blur.txt` | 7.36 | 原始记录、关键窗口、导出命令均保留 |
| `customer_workspace/validation/remaining_stalls_20261030-175436/B_781_blur.txt` | 7.35 | 原始记录、关键窗口、导出命令均保留 |
| `customer_workspace/validation/remaining_stalls_20261030-175436/host_A_0_candidates.txt` | 2.36 | 原始记录、关键窗口、导出命令均保留 |
| `customer_workspace/validation/remaining_stalls_20261030-175436/host_A_606_blur.txt` | 2.36 | 原始记录、关键窗口、导出命令均保留 |
| `customer_workspace/validation/remaining_stalls_20261030-175436/host_B_5505_blur.txt` | 13.93 | 原始记录、关键窗口、导出命令均保留 |
| `customer_workspace/validation/remaining_stalls_20261030-175436/host_B_3129_tracking.txt` | 13.92 | 原始记录、关键窗口、导出命令均保留 |
| `customer_workspace/validation/remaining_stalls_20261030-175436/host_B_7883_selection_strong_core.txt` | 13.92 | 原始记录、关键窗口、导出命令均保留 |

确认依据：根目录 `AGENTS.md` 第 15 条要求：

> Large cleanup requires an exact path list, sizes, preservation evidence, and user approval before deletion.

本清单已完成路径、大小、保留依据核对。用户随后回复“批准删除这 16 个文件”，本次按这一明确范围执行，没有扩大删除范围。

## 验证

检查结果记录在 `verification.json`：已删除缓存的对应源码未变化；当前分析与保存源码仍能默认搜索到；指定路径能搜索历史证据；离线回放入口 `--help` 可正常导入。没有运行新实验或修改科学计算，因此没有把这一轮描述为算法性能测试。

追加删除的核对结果在 `execution_result.json`：16 个批准路径全部不存在，57 个保留文件
重新校验通过，原有 Git 修改状态未因删除这些忽略或未跟踪文件而变化。同步更新了
追踪项目 `.gitignore` 中“保留本地 ZIP”的过时注释，忽略规则本身不变。
