# GPU 解码延迟：内部请求与系统等待诊断

主要额外耗时已定位到当前 WSL NVDEC 路径的驱动同步交接，特别是每帧资源锁定/释放请求；取回画面时另有等待解码完成的时间。不能把 Decode() 的总耗时全部当作显卡解码芯片的计算时间。

## 对照条件

A/B 各使用同样的前 300 帧，每种条件各两次，交错顺序，共 4800 帧。所有方案复用相同解码和图像示例、8 帧预热；只改变帧间等待方式。额外读数在解码计时以外。

- hybrid：按源视频时刻，提前 1 ms 从睡眠切到忙等。
- spin：同样的源视频时刻，间隔内 CPU 一直忙等。
- gpu：同样的源视频时刻，间隔内执行短小 GPU 空工作，最后 2 ms 停止投递并忙等。这是因果对照，不是正式优化建议。
- none：连续尽快送帧，只用来观察持续工作与间歇工作的差别，不能当作实时摄像机延迟。

| 视频/条件 | 帧数 | 中位 ms | P95 ms | P99 ms | 最大 ms |
|---|---:|---:|---:|---:|---:|
| A_gpu | 600 | 1.565 | 2.096 | 2.315 | 2.726 |
| A_hybrid | 600 | 3.098 | 4.617 | 5.771 | 13.829 |
| A_none | 600 | 1.314 | 1.788 | 2.701 | 4.915 |
| A_spin | 600 | 2.332 | 3.021 | 3.230 | 3.740 |
| B_gpu | 600 | 1.587 | 2.117 | 2.388 | 2.693 |
| B_hybrid | 600 | 3.019 | 4.687 | 5.605 | 6.480 |
| B_none | 600 | 1.324 | 1.539 | 1.636 | 1.959 |
| B_spin | 600 | 2.420 | 3.090 | 3.514 | 3.727 |

## 解码库内部发生什么

Nsight 的 NVDEC/系统 API 追踪记录了实际调用栈。解析器内部先调用 cuvidDecodePicture，再调用 cuvidMapVideoFrame64。下表为独立追踪的 180 帧中位数；追踪结果与未追踪的对照分开，不把父调用和子调用重复相加。

| 条件 | 提交解码 ms | 等待画面可用 ms |
|---|---:|---:|
| hybrid | 1.411 | 1.113 |
| spin | 0.828 | 1.021 |
| gpu | 0.533 | 0.847 |
| none | 0.366 | 0.824 |

提交阶段的主要系统调用来自 libdxcore.so 的 D3DKMTLock2 / D3DKMTUnlock2，追踪中每帧约三组。它们是显卡资源的锁定/释放请求，不是 Python 线程之间的互斥锁。取回画面阶段的主要等待来自 D3DKMTWaitForSynchronizationObjectFromCpu。

微软公开的 WSL 驱动实现显示，Lock2 / Unlock2 要通过虚拟机总线发送同步请求并等待 Windows 返回。本机实际调用栈与该路径吻合；公开源码只作为机制解释，不冒充本机闭源驱动的逐行源码。

## 完整视频捕获的慢帧

另对 A/B 各 1320 帧完整追踪，保留源帧号及原视频时间。下面的系统调用耗时是实际落在该帧 Decode 内部的记录：

- full_A，源帧 660：总计 27.368 ms；Linux 运行队列等待 0.000 ms；Linux 大缺页 0。
  主要请求：D3DKMTLock2 12.736 ms；D3DKMTLock2 10.875 ms；D3DKMTWaitForSynchronizationObjectFromCpu 1.869 ms。
- full_B，源帧 760：总计 23.619 ms；Linux 运行队列等待 0.000 ms；Linux 大缺页 0。
  主要请求：D3DKMTLock2 12.394 ms；D3DKMTLock2 7.630 ms；D3DKMTWaitForSynchronizationObjectFromCpu 1.834 ms。
- full_B，源帧 308：总计 13.404 ms；Linux 运行队列等待 0.005 ms；Linux 大缺页 0。
  主要请求：D3DKMTLock2 10.073 ms；D3DKMTLock2 1.245 ms；D3DKMTWaitForSynchronizationObjectFromCpu 0.841 ms。

## 证据边界

- 对照中普通等待也主要处于 P0、视频时钟 1995 MHz。GPU 轻负载时频率可升到 2160 MHz，不能把改善全部归因于频率，也不能用这些低频率采样排除更短的硬件门控变化。
- Linux 运行队列等待很小，长锁定等待中可为零；它不等于 Windows 宿主没有调度或内存问题。Linux 无大缺页也不能排除 Windows 页文件读取。
- 一个短测慢帧在最终 CUDA 流同步里耗时 11.373 ms，且调用线程 CPU 时间约同样长。这是另一种同步忙等，不能把全部尾部波动都归为 Lock2。
- Nsight 2024.5 与当前驱动的 GPU 活动追踪不完全兼容；这里只使用已实际记录的 CPU API、调用栈和线程事件，没有宣称已测得纯硬件解码时间。
- Windows 内部还需结合另存的 host_trace 检查；未取得对应证据之前，不将资源锁定等待断言为某个后台程序、显卡休眠或驱动缺陷。

## 资料与代码

本轮只新增独立诊断工具，并给已有 Windows 诊断脚本增加显式 Gpu 选项，原 Stalls 默认不变。正式追踪、录像与设备控制未改。原始 commands.json、逐帧 CSV、GPU 状态和 .nsys-rep/.sqlite 均保留。

机制依据：[NVIDIA NVDEC 编程指南](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvdec-video-decoder-api-prog-guide/index.html)、[微软 WSL GPU 架构](https://devblogs.microsoft.com/directx/directx-heart-linux/)、[微软公开 Lock2/Unlock2 实现](https://github.com/microsoft/WSL2-Linux-Kernel/blob/linux-msft-wsl-6.6.y/drivers/hv/dxgkrnl/dxgvmbus.c#L2920)。
