# Windows 原生 GPU 解码对照

日期：2026-11-01，Europe/Zurich。输入为已有 A/B 视频，不连接相机、AFG 或 PZT。

## 结论

Windows 原生 PyNvVideoCodec 可以直接调用 RTX 4060 解码，A/B 全部画面的像素核对通过。
本轮对照中，它的常见解码耗时和 P99 低于 WSL，但**没有消除偶发长停顿**。
已经从同一次 Windows 系统记录中确认：`0xC5` 通知触发显卡独占时，原生 Python 解码线程
也会等待同一把显卡锁。这个阻塞机制不局限于 Windows 与 WSL 的交接。

本轮只增加独立测试工具，没有替换正式解码器，没有更改识别、追踪、录像或 PZT 程序。

## 60 FPS 公平对照

两端使用同一脚本、同版本库、相同压缩视频及相同解码参数。A/B 每端各三遍，
每遍 1320 帧，共 **15840 帧**。Windows 与 WSL 交错顺序运行，没有同时占用 GPU。
每组 3960 帧的原始样本合并后计算分位数，没有平均各遍 P99。

| 视频 | 执行系统 | 帧数 | 中位数 | P99 | 最大值 | 超过 16.67 ms |
|---|---|---:|---:|---:|---:|---:|
| A | Windows 原生 | 3960 | 2.587 ms | 5.773 ms | 37.622 ms | 3 |
| A | WSL | 3960 | 3.180 ms | 7.063 ms | 21.493 ms | 1 |
| B | Windows 原生 | 3960 | 2.578 ms | 5.891 ms | 28.450 ms | 3 |
| B | WSL | 3960 | 3.378 ms | 7.163 ms | 26.717 ms | 5 |

P99 表示本组 99% 的帧耗时不超过这一数值，不表示未来不会出现更慢的帧。
Windows 的中位数约低 19%（A）和 24%（B）；最大值没有改善，两端超过一帧预算的次数均为 6。
这些是本机本次短时对照结果，不构成长时间稳定性保证，也不能把差值全部归为通信耗时。

数据入口：[合并统计](comparison/comparison.csv)、[每一遍统计](comparison/runs.csv)、
[完整分段及晚到统计](comparison/summary.json)、[实际运行命令](timing_commands.json)。

### 究竟计时了什么

- 压缩包在开始前读入内存，不计磁盘读取；不提前解码未来帧。
- 同样先预热 8 帧，再从源视频第 0 帧正式开始。
- 固定 60 FPS，即每隔 16.6667 ms 到达一帧；提前 1 ms 从睡眠转为循环等待。
- 计时从准备、提交压缩包，到 `Decode()`、`SyncOnCUStream()` 和 CUDA 流同步完成。
  输出已在显存中准备好，而不只是命令已提交。
- 没有运行微泡算法或图像处理示例，没有在计时路径主动下载整张图像。
- 正式送帧期间记录留在内存，完成后保存；启动信息在送帧前写完。
- 不主动跳帧。若一帧变慢，下一帧仍使用原计划时刻，真实晚到量另记于 `late_ms`。
  因此“1320 帧全解完”不等于“每帧均在 16.67 ms 内完成”。

A 的标称帧率是 60；B 容器中为约 60.00006。两端统一按用户要求的 60 FPS 送帧，
原输入 PTS 另外保留，没有改写原视频时间。

### 与之前 WSL 结果的关系

之前的 [WSL 小样本对照](../gpu_decode_root_20261031-231519/report_final/REPORT.md) 中，
普通等待方式每段使用前 300 帧、两遍：A 中位 3.098 ms、P99 5.771 ms、最大 13.829 ms；
B 中位 3.019 ms、P99 5.605 ms、最大 6.480 ms。
本轮扩大到完整视频、每端三遍，并去掉解码后的图像处理示例，Windows 与 WSL 改用同一份
CUDA Driver API 测试代码。**主要结论采用本轮同条件对照，而非把不同样本长度的最大值直接比较。**

## 0xC5 期间，Windows 原生仍被阻塞

另开一次最长 140 秒、可自动停止的 Windows 系统记录，原生回放 A/B 各两遍，共 **5280 帧**。
系统记录已正常停止，报告丢失事件为 0。它有额外测量开销，下面的数据不并入上面的速度表。

| 原生回放／帧号（从 0 开始） | 本帧总耗时 | 本帧解码线程等显卡锁 | 具体等待位置 |
|---|---:|---:|---|
| A 第 1 遍／636 | 42.598 ms | 38.041 ms | 两次 `DxgkSubmitCommandToHwQueue` |
| A 第 2 遍／1165 | 23.547 ms | 20.482 ms | 两次 `DxgkLock2` |

等锁时间包含在本帧总耗时内，不能再相加。

以第一个例子说明：原生 Python 进程号 **42544**，实际解码线程号 **8928**。
Windows 在收到两次编号 197（`0xC5`）通知后，由系统线程 **47488** 独占显卡锁
`0xffffda8e2eeec000`，原生解码线程随后请求同一把锁，确实一直等到独占结束才取得它。

以下时间全是同一 Windows QPC 时钟，单位为相对系统记录开始的毫秒：

| 事件 | 开始 | 结束 | 持续 |
|---|---:|---:|---:|
| 原生第 636 帧的计时窗口 | 13927.6682 | 13970.2773 | 约 42.609 ms |
| 系统第一次独占显卡 | 13929.4182 | 13939.2285 | 9.8103 ms |
| 原生解码线程第一次等锁 | 13929.8875 | 13939.3243 | 9.4368 ms |
| 系统第二次独占显卡 | 13939.3788 | 13967.8965 | 28.5177 ms |
| 原生解码线程第二次等锁 | 13939.4311 | 13968.0353 | 28.6042 ms |

QPC 窗口包住 Python 内部计时，包含两次时钟读取附近的少量开销，故比 42.598 ms 略宽。
这里没有相减 Windows 与 WSL 的时钟，也没有把往返通信时间除以二。

原生 Windows 导出和已匹配的微软函数符号还确认了通知处理路径：

```text
ACPI notification 197 / 0xC5
  → DpiAcpiHandleAcpiEvent
  → DxgkAcquireAdapterCoreSync
  → Exclusive adapter lock
  → Native Python decode thread waits for the same adapter lock
```

第二个例子的原生进程 **24920**、线程 **47664** 也被同一显卡锁阻塞；它在第一次独占已经开始后
才提交帧，所以不能把通知处理的完整时长都算进该帧。

可核查证据：[帧与锁的对应表](native_c5_cases.json)、
[锁的请求／获得／释放记录](native_trace/analysis/adapter_locks.json)、
[ACPI 通知记录](native_trace/analysis/acpi_events.json)、
[原生导出及 8 条核心调用栈](native_trace/native/acpi_stacks.json)。
原始逐帧 QPC 在各 `native_trace/traced_*/frames.csv`，进程／线程号在对应 `environment.json`。

全程共记录 22 条 `0xC5` 通知，上面是两个明确阻塞实例；没有声称每条通知都造成同样长的停顿。
本轮也没有为不带系统记录的 37.622 ms 尖峰补写确定原因。

## 画面与环境核对

Windows 与 WSL 分别对 A/B 全部 **1320 帧**做单独 CPU/GPU 原生 NV12 逐像素比较：
四组均为 **0 个不同像素**，每个压缩包立即对应一帧，结束时没有额外延迟输出。
这是图像正确性测试，不使用其中的计时作速度结论。

- Windows：[A 核对](native_verify_A/summary.json)、[B 核对](native_verify_B/summary.json)
- WSL：[A 核对](wsl_verify_A/summary.json)、[B 核对](wsl_verify_B_02/summary.json)

PyNvVideoCodec 当前 ZERO 模式返回的输出 PTS 为 0，因此没有把它冒充来源时间。
输入帧号、原包 PTS、计划时间及输出 PTS 分列保存，通过完整像素顺序核对确认本测试的关联。
这仍不是可直接接收任意视频格式的正式接入实现。

环境：Windows 11，RTX 4060 Laptop GPU（8 GiB），驱动 616.92；
Windows Python 3.12.10，WSL Python 3.12.3。
两端 PyNvVideoCodec 2.2.3、NumPy 2.2.6、PyAV 19.0.1。
Python 补丁版本及平台二进制不同，已保留在每遍 `environment.json`；不是把两个平台假定为完全相同的系统。
没有更换驱动或系统 CUDA。新环境仅在 `F:\MB_gpu_decode`，依赖与源码来源见
[deployment.json](deployment.json)、[安装清单](install_report.json)及[wheel 校验值](wheel_checksums.json)。

## 新增工具与运行方式

- `tools/gpu_native_decode.py`：两端共用的解码、同步计时与逐像素核对。
- `tools/gpu_native_report.py`：从完整逐帧记录生成合并和每遍统计。
- 现有 `gpu_decode_host_trace.py` 增加显式 `--native-root`，WSL 原有默认保持不变。
- 现有 `gpu_host_report.py` 增加 `--native`，直接使用 Windows QPC。
- 事件读取工具保留所有 ACPI 通知，并明确计数额外字段解析失败。
- `gpu_decode_check.py` 只把 Linux 专用核对模块的导入移到其核对函数内，便于 Windows 复用压缩包读取；没有改变原解码算法。

Windows PowerShell 中重新测试，输出目录必须新建：

```powershell
F:\MB_gpu_decode\python312\python.exe -m customer_workspace.tools.gpu_native_decode `
  --video F:\MB_gpu_decode\videos\expA.mp4 `
  --output-dir F:\MB_gpu_decode\runs\A_new_test
```

完整操作说明见 [WINDOWS_NVDEC.md](../../tools/WINDOWS_NVDEC.md)，F 盘也放有对应 `README.md`。
原始数据在本报告目录留档；F 盘运行目录保留原生测试结果。

## 限制、失败记录及收尾

- 本轮测试的是独立解码，未接入真实相机、RTSP、Windows→WSL 传图或正式微泡追踪。
  因此不能据此宣称整条链路会快同样的比例，也没有验证 GPU 到后续算法的完整传递。
- 原生程序同样受 Windows 的显卡管理通知影响。通知究竟来自哪项厂商策略，仍沿用
  [上一轮报告的边界](../gpu_host_cause_20261031-235434/REPORT.md)，本轮没有改动该策略。
- 旧版 TraceEvent 对 9099 个额外载荷字段报告解析失败；本结论用到的进程、线程、显卡锁、时间和通知字段
  已与 Windows 原生 xperf 导出交叉核对。两次原生导出及核心调用栈保留，不使用解析失败字段推断原因。
- 准备时遇到默认下载源过慢、pip 尚未完成安装时启动依赖安装、过于严格的输出 PTS／标称帧率检查。
  已改为官方来源 wheel 离线安装，并沿用输入 PTS 和近 60 FPS 的真实数据条件完成验证；
  这些失败日志保留，没有混入统计。xperf 第一次传入小数时间失败，改为其要求的整数微秒后导出成功。
- 所有系统记录已正常停止。原始 ETL 已归档并校验，位置与 SHA-256 见
  [etl_archive.json](native_trace/etl_archive.json)。正式程序、原始视频和配置共 12 个保护文件均未改变，
  见 [integrity_check.json](integrity_check.json)。静态检查和运行核对见 [validation.json](validation.json)。

官方资料：[NVIDIA PyNvVideoCodec 系统要求](https://docs.nvidia.com/video-technologies/pynvvideocodec/read-me/system-requirements-common.html)、
[安装方式](https://docs.nvidia.com/video-technologies/pynvvideocodec/read-me/building-library.html)、
[Windows ACPI 通知回调](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/dispmprt/nc-dispmprt-dxgkddi_notify_acpi_event)。
官方资料用于确认支持范围；具体性能和阻塞结论来自本机实测。
