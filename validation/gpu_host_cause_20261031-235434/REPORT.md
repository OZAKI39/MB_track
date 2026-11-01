# GPU 解码长等待：Windows 根因检查

检查日期：2026-10-31 至 2026-11-01。只读已有 A/B 视频，不连接相机、AFG 或 PZT。

## 已经查清的原因

本轮复现的长资源锁定请求，主要被 **Windows 处理 ACPI 通知时的显卡独占操作**阻塞。
ACPI 是电脑硬件与操作系统之间传递管理信息的机制。本次通知编号为 **197，即 `0xC5`**。

可以理解成：电脑临时要求显卡处理一项硬件管理事务，要先让许多显卡任务暂停、等待它们停妥，
再恢复工作。期间 WSL 的解码请求要等这项事务结束，或者排在同样受阻的前一条请求后面。
这解释了为什么平常几毫秒的处理，偶尔突然多等十几到三十几毫秒。

这次不只是两个事件时间接近：已经核对到**同一显卡锁地址、独占它的系统线程、等待该锁的
WSL 服务线程，以及发起独占的 Windows 函数调用栈**。

```text
ACPI notification 197 / 0xC5
  → DpiAcpiHandleAcpiEventWork
  → DpiAcpiHandleAcpiEvent
  → DxgkAcquireAdapterCoreSync / DXGADAPTER::AcquireCoreSync
  → Exclusive adapter lock
  → SuspendScheduler → Suspend/resume GPU contexts
  → Release adapter lock
  → Blocked WSL GPU requests continue
```

## 同一次运行中的证据

有效的完整 GPU 记录位于 `host_driver`：A、B 各回放两遍，每遍 1320 帧，共 **5280 帧**。
帧号从 0 开始。下列“总耗时”为本轮诊断程序的逐帧处理时间，包含解码、同步及显存内处理；
并非 NVDEC 芯片的纯硬件解码时间，也不是正式微泡分析时间。

| 回放／帧号 | 本帧总耗时 | 最长一次 Lock2 请求 | 两次显卡独占累计 | 对应调用栈 |
|---|---:|---:|---:|---|
| A 第 1 遍／417 | 19.573 ms | 15.609 ms | 15.772 ms | 已核对 |
| A 第 2 遍／404 | 20.654 ms | 17.467 ms | 17.304 ms | 同类事件与锁记录；未另导出本帧原生调用栈 |
| A 第 2 遍／765 | 40.831 ms | 33.937 ms | 33.134 ms | 已核对 |
| B 第 2 遍／847 | 26.528 ms | 21.946 ms | 22.170 ms | 已核对 |

逐项原始数字见 [cause_cases.json](cause_cases.json)。独占时间和请求时间相互重叠，
**不能把两列相加**；各自起止点不同，也不应要求它们完全相等。

### 最长例子：A 第 2 遍第 765 帧

以下时间全部来自同一个 Windows ETW 时钟，相对该记录开始的毫秒数：

| Windows 事件 | 开始 | 结束 | 时间 |
|---|---:|---:|---:|
| 系统线程 49044 第一次独占显卡 | 72364.2749 | 72381.0235 | 16.7486 ms |
| 系统线程 49044 第二次独占显卡 | 72381.3011 | 72397.6861 | 16.3850 ms |
| WSL 服务线程 7544 等待同一显卡锁 | 72381.1399 | 72397.7696 | 16.6297 ms |

锁地址均为 `0xffffda8e2eeec000`。WSL 前一条提交命令也先等待了约 16.775 ms，
因此 Linux 侧这次 Lock2 请求的一部分时间是在等前面的命令，随后才进入自身的 Windows 锁定函数。
不能把 Linux 侧 33.937 ms 全部当作 Windows 锁定函数体的运行时间。

该帧 `Decode()` 用时 38.070 ms，但调用线程实际使用 CPU 仅 0.692 ms，记录到的 Linux
就绪队列等待为 0 ms。这与 Windows 侧的资源等待证据一致。

### 独占期间究竟在做什么

四个案例的每次独占中，都记录到 **69 次 `DdiSuspendContext` 和 69 次 `DdiResumeContext`**。
这些是显卡工作上下文的暂停与恢复，不是 69 帧视频，也不能据此认定是 69 个应用程序。
通知驱动的回调本身很短，主要开销在暂停／恢复及其等待过程。

最长案例中，持锁线程两段独占期间分别有约 **10.069 ms、12.816 ms** 未在 CPU 上执行。
其中两个明确等待点的调用栈是：

```text
DpiAcpiHandleAcpiEvent
  → DxgkAcquireAdapterCoreSync
  → ADAPTER_RENDER::SuspendScheduler
  → VidSchSuspendAdapter
  → VidSchiSuspendResumeDevicesForPowerTransition
  → VidSchiSuspendResumeHwContexts
  → VidSchWaitForCompletionEvent
  → KeWaitForMultipleObjects
```

对应等待分别为 1.821 ms 和 1.252 ms。还找到一次约 0.769 ms 的等待，发生在
`DdiSuspendContext → nvlddmkm.sys → ExAcquireResourceSharedLite`，即 NVIDIA 驱动内部等待共享资源。
不能把所有未运行时间都归为 CPU 被其他程序抢走；这里有明确的主动等待暂停完成和资源等待。
NVIDIA 私有函数没有公开符号，本轮不把其内部共享资源进一步命名。

详见 [exclusive_work.json](exclusive_work.json)、[lock_owner_scheduling.json](lock_owner_scheduling.json)
及 [owner_wait_stacks.json](host_driver/symbols/owner_wait_stacks.json)。函数耗时存在嵌套，不能直接全部求和。

## 通知来源查到了哪一层

通过 Windows 官方只读接口读取了本机 DSDT 固件表；没有执行其中的方法，没有修改 BIOS。
在本机 `MECHREVO / JiguangX Series GM6PR0C` 固件中，确实找到 10 处向显卡发送 `0xC5` 的代码，
涉及 `CTAF`、`PWUP`、`PWBT`、嵌入式控制器事件 `_Q83`、`_Q84`。
这些位置同时更新 `NPCF` 下的 `ACBT`、`AMAT`、`DBAC` 等厂商管理字段。

- [通知位置清单](firmware/notify_c5_sites.json)
- [本机固件反汇编](firmware/DSDT.dsl)，例如第 132634、135430、135450 行
- [原始固件表](firmware/DSDT.aml)及[反汇编日志](firmware/disassembly.log)

**已确认：通知编号、Windows 处理路径、显卡独占与解码请求受阻之间的关系。**
**尚未确认：运行时究竟执行了上述哪个固件分支，以及哪项厂商策略触发它。**
静态固件表只能说明哪些路径能够发送通知，不能证明某一条就是当时实际运行的路径。
因此目前不把原因直接命名为“某个省电设置”“某个后台软件”或“驱动缺陷”。

微软文档说明 [ACPI 通知回调](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/dispmprt/nc-dispmprt-dxgkddi_notify_acpi_event)
及[显卡上下文暂停](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/d3dkmddi/nc-d3dkmddi-dxgkddi_suspendcontext)
的用途；本报告的具体编号与耗时结论来自本机记录，而非根据文档猜测。

## 测量和证据的有效范围

- 有效记录导出 230179 个窗口内事件，匹配 17368 对函数起止、3998 组显卡锁记录，报告丢失事件为 0。
- 用微软匹配的公开调试符号解析了所选 Windows 内核／显卡管理模块的 1480 个地址；这不包含 NVIDIA 私有符号。
  三个原生案例的 12 条核心同步栈均包含 `DpiAcpiHandleAcpiEvent`。
- Windows 与 WSL 没有直接相减各自的时钟。通过多次请求应答选取对应窗口，本表案例附近的
  半往返时间为 0.377–0.549 ms，用来提示对应误差；不把半往返当作单向通信延迟。
  上述锁的持有、等待和释放关系全部在 Windows 同一时钟内核对。
- 详细跟踪有额外开销。本轮用于确认阻塞位置，**不能用本轮最大值与无跟踪测试比较性能优劣**。
  当前驱动下 Nsight 未提供有效的 GPU 硬件活动时间线，因此不声称测得 NVDEC 内部纯计算时间。
- 旧版 TraceEvent 对部分未使用载荷会返回 `EXCEPTION_DURING_VALUE_LOOKUP` 文本，虽然其外层异常计数为 0。
  本结论使用的锁、通知和函数字段已与 Windows 原生 xperf 导出、调用栈交叉核对；未用解析失败字段推断原因。
- 最早那两次历史长等待没有对应的完整 Windows 记录，无法事后证明它们逐条都是同一原因。
  本轮证明的是**重新复现的同类长等待机制**，不替历史缺失证据补写确定结论。

## 其他慢帧与失败尝试

`host_light`、`host_gpu` 两次前置记录各完成 A/B 两遍，但 GPU 事件未成功启用，只有内核事件。
它们的文件保留，不作为显卡独占结论的核心依据。最终使用修正的独立 WPR 配置完成 `host_driver`。
另一次尝试切换旧版 TraceEvent 的事件注册接口因编译警告失败，日志保留，未使用其输出。

慢帧也并非全是同一原因：

- `host_light / A_r1 / 920` 有约 9.411 ms 的 CUDA 同步等待，CPU 时间约 9.412 ms，属于另一类表现。
- `host_light / B_r1 / 384` 的对应窗口有 vmmemWSL 页文件读取，但缺少 GPU 事件链，
  不能把整帧 16.904 ms 全部归因于换页。
- `host_driver / B_r1 / 688` 总计 34.246 ms，包含多个 Lock/Unlock 和同步等待；
  本轮没有为它建立与上述四帧同样完整的 ACPI 因果链，保留为另一项慢帧，不混入已证明案例。

## 本次修改、保存和验证

只增加／完善独立诊断工具：

- `tools/gpu_host_report.py`：对应时钟窗口、配对函数和显卡锁记录。
- `tools/gpu_host_symbols.cs`：用匹配的微软公开符号解释已保存的调用栈。
- `tools/windows_read_dsdt.ps1`：只读提取固件表。
- `microbubble_tracking/windows_gpu_light.wprp`：限时 GPU、锁和线程诊断配置。
- `README.md` 第 7.5.6 节更新结论。

原始 ETL 已从本次 Windows 临时目录归档到 `host_light/host.etl`、`host_gpu/host.etl`、
`host_driver/host.etl`，逐个核对移动前后 SHA-256；位置与校验值见各目录 `etl_archive.json`。
正式代码、A/B 原视频和设备配置共 12 个保护文件与开始时完全一致，见 [integrity_check.json](integrity_check.json)。
所有系统诊断均已结束；没有调整电源方案、驱动、BIOS、Windows 服务或正式解码设置。
最后执行 `wpr -status` 返回 `WPR is not recording`。
Python 格式、静态检查、编译、命令行帮助和 WPR XML 解析均通过；归档后的三份 ETL
也重新通过时钟校验，见 [validation.json](validation.json)。C# 符号工具和 PowerShell
固件读取工具均已实际成功运行。本轮没有进行设备联调，也没有验证任何修复方案。

重新读取新的完整记录可运行：

```bash
.venv/bin/python -m customer_workspace.tools.gpu_host_report RECORDING_DIR \
  --output-dir NEW_ANALYSIS_DIR --traceevent-dir EXISTING_TRACEEVENT_DLL_DIR
```

`RECORDING_DIR` 必须是本诊断入口保存的记录目录；分析输出必须新建。
源数据还包括各次 Nsight `.sqlite`、逐帧 CSV、原生 xperf 导出及匹配符号，均留在对应目录内。
依赖 `acpica-tools 20230628-1` 只解包到 `customer_workspace/dependencies/acpica_tools`，未安装到系统。

## 对下一步的影响

本轮完成定位，尚未证明某个设置能消除问题。正式流程仍保持已经验证的 CPU 解码。
如果下一步要减少此类 GPU 尖峰，应围绕厂商显卡管理通知做一次只改变一项设置的可恢复对照，
检查通知次数、独占时间和解码尖峰是否一起下降；不应仅修改微泡识别算法来掩盖这一等待。
