# validation 清理记录（已按批准清单执行）

用户已明确回复“批准删除清单中的 6 个目录”。于 2026-10-31 按该范围完成删除：

- 6 个目录全部删除，包括 3 个空目录和 3 个重复导出目录。
- 删除 22 个文件，共 28,155,122 字节（26.85 MiB）。
- 删除前复核目录内容、文件大小、修改时间、文件身份及 SHA-256，候选中没有 Git 跟踪文件或登记的 worktree。
- 删除后确认 6 个目录及 22 个文件均已不存在；figures/expA 中的 19 个保留文件再次校验一致。
- 没有跳过项或错误；Git 状态与执行前相同。未修改现行算法或运行真实实验。

详细结果见 `execution_preflight.json` 与 `execution_result.json`。`pending_delete.json` / `pending_delete.csv` 保留批准前原始清单，不改写当时的状态。
下面为本次清单及核对依据，所有建议删除项现已执行。

本次只检查 `/home/lzy/projects/new_microbubble_experiment/customer_workspace/validation`。
盘点到原有 138 个顶层目录、29,174 个文件或链接，文件大小合计 25.22 GiB。
没有把时间较早、名称带 legacy/final/failed、文件较大当作删除依据。

## 已批准并执行的范围

已删除下列 **6 个目录，共 22 个文件、28,155,122 字节（26.85 MiB）**。
其中 3 个为空目录；另 3 个目录的全部内容已确认在 `microbubble_tracking/figures/expA` 中保留。
普通文件逐项计算 SHA-256；ZIP 逐成员核对内容，不仅比较文件名和大小。
保留文件共 19 个，完整原位置、保留位置、大小、修改时间及摘要见 `pending_delete.json`。
这是可明确认定的冗余导出，不代表其余 132 个目录已经废弃。

| 完整目录路径 | 文件数 | 大小（MiB） | 保留依据 |
| --- | ---: | ---: | --- |
| `/home/lzy/projects/new_microbubble_experiment/customer_workspace/validation/analysis_slides_20261028-134318` | 0 | 0.000 | 目录为空，没有原始数据或失败记录 |
| `/home/lzy/projects/new_microbubble_experiment/customer_workspace/validation/analysis_slides_20261028-134324` | 0 | 0.000 | 目录为空，没有原始数据或失败记录 |
| `/home/lzy/projects/new_microbubble_experiment/customer_workspace/validation/stimulus_response_slides_20261028-144848` | 0 | 0.000 | 目录为空，没有原始数据或失败记录 |
| `/home/lzy/projects/new_microbubble_experiment/customer_workspace/validation/analysis_slides_20261028-141412` | 12 | 18.972 | 全部普通文件以及 ZIP 全部成员，与 figures 中的保留文件内容一致 |
| `/home/lzy/projects/new_microbubble_experiment/customer_workspace/validation/analysis_slides_20261028-143326` | 7 | 7.835 | 全部普通文件以及 ZIP 全部成员，与 figures 中的保留文件内容一致 |
| `/home/lzy/projects/new_microbubble_experiment/customer_workspace/validation/stimulus_response_slides_20261028-144914` | 3 | 0.045 | 全部普通文件以及 ZIP 全部成员，与 figures 中的保留文件内容一致 |

三个非空目录对应的保留位置：

- `analysis_slides_20261028-141412`：`customer_workspace/microbubble_tracking/figures/expA/slide_06_motion_shape_and_collective_analysis/`，包括动画、逐帧表、来源说明和标签记录。
- `analysis_slides_20261028-143326`：同一 Slide 6 目录，保留群体指标动画、逐帧表、原验证结论和来源说明。
- `stimulus_response_slides_20261028-144914`：`customer_workspace/microbubble_tracking/figures/expA/slide_08_stimulus_response_analysis/data/`，三张数据表均完全一致。

正式源码、配置和文档的 318 个候选文本文件中，未发现对上述六个目录的现行引用。
figures 的历史 inventory/collection/provenance 仍记录原导出路径，它们是当时的来源记录，不改写成新历史；本清单保存原路径到保留文件的对应关系。
删除后，原导出目录的旧聊天链接将不再可用，应使用上述 figures 路径；成品及配套数据仍保留。
候选目录中没有 Git 跟踪文件。执行前还必须再次核对 Git 状态、目录内容和活动进程。

## 本轮保留的内容与原因

- 两段原始实验录像、PZT 记录、视频帧时间和失败/取消实验记录：原始测量不可用新一次实验替代。
- 人工标注、Mask、修订及 `bubble_benchmark_20261026-120008` 固定数据集：是人工审核和比较依据，即使某个算法不再使用，也不能按目录年龄删除。
- 大型卡顿诊断的 ETL、perf 数据和各轮数据库：保存了具体停顿、失败与对照证据，没有核实覆盖这些内容的独立备份。
- SAM/Cutie、联合算法及未采用的试验：保留比较、失败与选择依据。
- 其他图表和界面检查目录：存在不完全相同的文件、独有验证记录或标注修订，尚不能认定为纯冗余；没有删除。
- 即使部分子目录形似重复，本轮没有拆除实验结果中的 tables、候选图或 Mask，以免破坏原实验目录的完整性。

尤其注意：部分 PDF/SVG 与 figures 中同名文件的摘要不同，不能以画面相似代替逐字节核验。具体差异清单见 `presentation_copy_checks.json`。
完整目录大小见 `inventory.json`；当前代码引用见 `current_references.json`。
本轮没有运行新实验、重做科学计算、改变 AFG/相机设置或修改现行算法。

## 批准依据

项目根目录 [AGENTS.md](/home/lzy/projects/new_microbubble_experiment/AGENTS.md) 第 15 条要求：

> Large cleanup requires an exact path list, sizes, preservation evidence, and user approval before deletion.

此清单已经完成路径、大小和保留依据检查；用户随后批准上述具体六个目录。此前批准删除的 16 个文件已于上一轮删除，不属于本次清单，也不会重复使用那次批准。

本次执行只逐项删除清单内、重新核验未变化的文件，再移除空目录，没有递归清理未列出的内容。所有项通过复查，没有需要跳过的项。

进程初查：可读取的 WSL 进程中未发现候选目录被用作工作目录、命令参数或打开文件；部分系统进程无读取权限。Windows 命令行初查未发现指向 validation 的运行任务，但未检查全部 Windows 文件句柄。删除前将再次复查，有变化或疑义的目录跳过。
