# 早期 Mask、SAM、Cutie 和联合方法：已执行清理记录

状态：用户已批准“按清单删除”，于 2026-10-31T19:48:11.874711+01:00 完成。批准的 67 项全部删除，无跳过项；README 按批准的差异更新。

创建时间：2026-10-31T19:42:50.819756+01:00

## 范围与保护要求

完整保留当前识别与追踪主线 `customer_workspace/microbubble_tracking/`，包括源码、在线处理、背景、Mask、原始视频、人工审核、分析、世界模型、results、figures 和 history。
baseline 的代码、结果、MobileSAM 模型与依赖、两个独立环境、动画工具及其依赖全部保留。`baseline/channel_mask.py` 与本次旧 `customer_workspace/channel_mask.py` 是不同文件。
整个现有 validation 数据区、所有人工标注、确认过的 Mask、冻结测试集、科研结果、原始录像和时间记录均不列入本批删除。旧的 11 份配置及权重来源 JSON 也保留，用于查证参数与来源。
当前 AFG、录像、相机接入、在线追踪、图表工具及其测试不改动。`benchmark_report.py` 仍被 AFG 等待校准和时序测试调用，明确保留。

## 本批已批准并删除

共 **67 个文件，1,188,884,208 字节（1133.81 MiB，1.107 GiB）**。以下路径均相对项目根目录 `/home/lzy/projects/new_microbubble_experiment`。

不进行整目录递归删除；只处理下面逐项列出的文件。大小是文件内容大小，不保证等于删除后实际释放的磁盘空间。

| 类别 | 文件数 | MiB |
| --- | ---: | ---: |
| 程序及配套文件 | 47 | 0.61 |
| 缓存 | 16 | 0.37 |
| 旧权重 | 4 | 1132.83 |

### 精确文件列表

| 相对路径 | 字节数 | 删除依据 |
| --- | ---: | --- |
| `customer_workspace/JOINT_TRACKING.md` | 7,087 | 废弃联合方法的开发说明（实验报告另行保留） |
| `customer_workspace/assets/mask_shrink_review.html` | 7,692 | 仅由旧 Mask 参数比较工具读取的页面模板 |
| `customer_workspace/bubble_benchmark.py` | 32,710 | 废弃程序 |
| `customer_workspace/bubble_dark_preview.py` | 20,806 | 废弃程序 |
| `customer_workspace/bubble_dark_regions.py` | 4,477 | 废弃程序 |
| `customer_workspace/bubble_frame_review.py` | 19,890 | 废弃程序 |
| `customer_workspace/bubble_region_annotation.py` | 99,309 | 废弃程序 |
| `customer_workspace/bubble_region_calibration.py` | 14,470 | 废弃程序 |
| `customer_workspace/bubble_region_comparison.py` | 19,911 | 废弃程序 |
| `customer_workspace/bubble_region_review.py` | 22,123 | 废弃程序 |
| `customer_workspace/bubble_region_video.py` | 20,943 | 废弃程序 |
| `customer_workspace/bubble_regions.py` | 9,130 | 废弃程序 |
| `customer_workspace/bubble_review_edits.py` | 7,317 | 废弃程序 |
| `customer_workspace/bubble_tracking.py` | 22,141 | 废弃程序 |
| `customer_workspace/bubble_tracking_display.py` | 5,822 | 废弃程序 |
| `customer_workspace/bubble_tracking_review.py` | 18,893 | 废弃程序 |
| `customer_workspace/channel_mask.py` | 20,856 | 废弃程序 |
| `customer_workspace/cutie_initial_tracking.py` | 11,108 | 废弃程序 |
| `customer_workspace/initial_tracking_comparison.py` | 8,834 | 废弃程序 |
| `customer_workspace/joint_region_fusion.py` | 10,527 | 废弃程序 |
| `customer_workspace/joint_sam21_runner.py` | 18,740 | 废弃程序 |
| `customer_workspace/joint_tracking_review.py` | 18,822 | 废弃程序 |
| `customer_workspace/mask_shrink_review.py` | 22,552 | 废弃程序 |
| `customer_workspace/mask_shrink_sections.py` | 4,894 | 废弃程序 |
| `customer_workspace/models/cutie-base-mega.pth` | 140,443,788 | Cutie base-mega 官方权重，仅旧 Cutie 试验使用 |
| `customer_workspace/models/cutie_resnet/resnet18-5c106cde.pth` | 46,827,520 | 旧 Cutie Mask 编码器权重 |
| `customer_workspace/models/cutie_resnet/resnet50-19c8e357.pth` | 102,502,400 | 旧 Cutie 像素编码器权重 |
| `customer_workspace/models/sam2.1_hiera_large.pt` | 898,083,611 | SAM 2.1 官方权重，仅旧独立/联合追踪使用 |
| `customer_workspace/sam21_initial_tracking.py` | 15,400 | 废弃程序 |
| `customer_workspace/sam21_tracking_review.py` | 11,189 | 废弃程序 |
| `customer_workspace/tests/__pycache__/test_bubble_benchmark.cpython-312-pytest-9.1.1.pyc` | 17,688 | 上述废弃程序或测试的 Python 缓存 |
| `customer_workspace/tests/__pycache__/test_bubble_dark_regions.cpython-312-pytest-9.1.1.pyc` | 27,118 | 上述废弃程序或测试的 Python 缓存 |
| `customer_workspace/tests/__pycache__/test_bubble_frame_review.cpython-312-pytest-9.1.1.pyc` | 19,031 | 上述废弃程序或测试的 Python 缓存 |
| `customer_workspace/tests/__pycache__/test_bubble_region_calibration.cpython-312-pytest-9.1.1.pyc` | 22,532 | 上述废弃程序或测试的 Python 缓存 |
| `customer_workspace/tests/__pycache__/test_bubble_region_video.cpython-312-pytest-9.1.1.pyc` | 24,161 | 上述废弃程序或测试的 Python 缓存 |
| `customer_workspace/tests/__pycache__/test_bubble_regions.cpython-312-pytest-9.1.1.pyc` | 45,471 | 上述废弃程序或测试的 Python 缓存 |
| `customer_workspace/tests/__pycache__/test_bubble_review_edits.cpython-312-pytest-9.1.1.pyc` | 21,187 | 上述废弃程序或测试的 Python 缓存 |
| `customer_workspace/tests/__pycache__/test_bubble_tracking.cpython-312-pytest-9.1.1.pyc` | 41,646 | 上述废弃程序或测试的 Python 缓存 |
| `customer_workspace/tests/__pycache__/test_bubble_tracking_review.cpython-312-pytest-9.1.1.pyc` | 18,028 | 上述废弃程序或测试的 Python 缓存 |
| `customer_workspace/tests/__pycache__/test_channel_mask.cpython-312-pytest-9.1.1.pyc` | 55,358 | 上述废弃程序或测试的 Python 缓存 |
| `customer_workspace/tests/__pycache__/test_cutie_initial_tracking.cpython-312-pytest-9.1.1.pyc` | 5,311 | 上述废弃程序或测试的 Python 缓存 |
| `customer_workspace/tests/__pycache__/test_initial_tracking_comparison.cpython-312-pytest-9.1.1.pyc` | 4,696 | 上述废弃程序或测试的 Python 缓存 |
| `customer_workspace/tests/__pycache__/test_joint_region_fusion.cpython-312-pytest-9.1.1.pyc` | 28,370 | 上述废弃程序或测试的 Python 缓存 |
| `customer_workspace/tests/__pycache__/test_mask_shrink.cpython-312-pytest-9.1.1.pyc` | 32,950 | 上述废弃程序或测试的 Python 缓存 |
| `customer_workspace/tests/__pycache__/test_sam21_initial_tracking.cpython-312-pytest-9.1.1.pyc` | 9,884 | 上述废弃程序或测试的 Python 缓存 |
| `customer_workspace/tests/__pycache__/test_sam21_tracking_review.cpython-312-pytest-9.1.1.pyc` | 12,215 | 上述废弃程序或测试的 Python 缓存 |
| `customer_workspace/tests/check_bubble_benchmark_gui.py` | 22,668 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/check_bubble_frame_gui.py` | 12,580 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/check_bubble_region_gui.py` | 20,289 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/check_mask_shrink_page.py` | 4,876 | 仅用于旧 Mask 对照页面的测试 |
| `customer_workspace/tests/check_scissors_mask.py` | 6,748 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/test_bubble_benchmark.py` | 7,991 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/test_bubble_dark_regions.py` | 5,078 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/test_bubble_frame_review.py` | 7,516 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/test_bubble_region_calibration.py` | 6,808 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/test_bubble_region_video.py` | 8,519 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/test_bubble_regions.py` | 12,098 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/test_bubble_review_edits.py` | 4,677 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/test_bubble_tracking.py` | 8,104 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/test_bubble_tracking_review.py` | 6,089 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/test_channel_mask.py` | 11,427 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/test_cutie_initial_tracking.py` | 1,617 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/test_initial_tracking_comparison.py` | 1,160 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/test_joint_region_fusion.py` | 6,667 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/test_mask_shrink.py` | 4,599 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/test_sam21_initial_tracking.py` | 1,952 | 仅用于废弃程序的测试 |
| `customer_workspace/tests/test_sam21_tracking_review.py` | 4,137 | 仅用于废弃程序的测试 |

## 已做的核对

- 扫描 285 个当前可见的源码/配置/脚本文件；对保留的 Python 文件解析 import，未发现指向待删旧模块的导入，解析失败为 0。
- 对当前 microbubble_tracking 内源码及文字记录另外搜索旧 validation 和模型路径，未发现这些外部旧路径引用。历史来源记录仍原位保留。
- 四个拟删权重的 SHA-256 均与旧 SAM/Cutie 来源或运行记录完全一致；baseline 使用另外的 MobileSAM 权重，未列入删除。
- 已记录 888 个受保护文件的内容校验值，用于执行前后核对。环境目录、Git 内部记录及缓存不做全面内容哈希，但同样禁止删除。
- 本次本地检查未发现正在运行这些旧模块的 Python/pytest 进程；删除前仍须重新检查进程和打开的文件。
- 现有 Git 中有用户未提交修改，本批不提交、不推送、不重置。没有启动仪器或运行新算法实验。

## 删除后的影响与仍保留的资料

删除后，24 个旧模块的命令入口及对应测试不再可运行；当前识别追踪、baseline 和真实实验入口保持原样。历史结果文件仍可直接查看。
四个权重拟作为单独明确批准的删除项。没有核实这些文件存在独立备份；保留的哈希和下载网址不是备份。删除后不能依靠本地这些权重重跑旧 SAM/Cutie/联合方法，需重新取得权重。源码同样没有在本轮制作历史副本。
README 的废弃工具说明已按批准的 `README_cleanup.patch` 更新；当前主线、baseline 的使用说明保持原样。

## 执行约束

审批对象为本文件及 delete_manifest.json 的精确文件集合。执行前重新检查每个文件的大小、修改时间、SHA-256、Git 状态、引用与活动进程；发生变化或用途不明的一律跳过。删除后再次核对受保护文件，不改动当前主线与 baseline。

批准前的文件身份与引用检查已在执行前重新完成；执行后检查和离线测试见下文。没有运行真实硬件或完整视频性能实验。

## 附件

- [逐文件删除清单与 SHA-256](delete_manifest.json)
- [保留文件的校验记录](protected_manifest.json)
- [引用检查](reference_checks.json)
- [执行前 Git 状态](git_status_before.txt)

## 为什么需要批准

[项目 AGENTS.md 第 15 条](/home/lzy/projects/new_microbubble_experiment/AGENTS.md) 要求：

> Large cleanup requires an exact path list, sizes, preservation evidence, and user approval before deletion.

本次涉及多份旧源码、测试和约 1 GiB 的权重，已在这份具体清单获批后执行；您已明确排除的当前主线和 baseline 不属于删除范围。

## 执行结果

- 删除 67 个批准文件，共 1,188,884,208 字节（1.107 GiB）；没有递归删除目录，没有跳过项。此数值为删除文件的内容大小，不声称是实测磁盘可用空间增量。
- 执行前确认文件身份和 Git 状态未变化，未发现相关本地任务；检查了可访问进程的打开文件和内存映射。部分系统进程不允许读取文件描述符，已在 execution_preflight.json 如实记录；未访问 Windows 部署或远程服务器。
- 删除后解析了 209 个保留的 Python 文件，未发现导入被删旧模块的语句。
- 888 个受保护文件在删除后及测试结束后均与盘点时的 SHA-256 一致。包括当前主线完整可检查文件集合、baseline 及 MobileSAM、源视频、当前入口和测试。虚拟环境保持原位，未修改。
- 相机在线追踪接入、视频分流验收与 AFG 时序统计的 51 项离线测试全部通过（0 失败、0 错误、0 跳过）；全部使用软件或临时数据，不连接真实设备。
- README 与批准的差异完全一致；无关 Git 状态保持原样，没有提交或推送。原始 validation 资料和科研结果没有删除。

测试命令：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider \
  --basetemp customer_workspace/validation/cleanup_legacy_methods_20261031-194246/test_tmp \
  --junitxml customer_workspace/validation/cleanup_legacy_methods_20261031-194246/tests.xml -q \
  customer_workspace/tests/test_camera_tracking.py \
  customer_workspace/tests/test_parallel_video.py \
  customer_workspace/tests/test_benchmark_report.py
```

- [执行前复查](execution_preflight.json)
- [逐文件实际删除记录](deletion_actions.jsonl)
- [执行后核对](post_delete_verification.json)
- [51 项测试结果](tests.xml)
- [最终完成记录](completion.json)
