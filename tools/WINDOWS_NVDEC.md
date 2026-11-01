# Windows 原生 GPU 解码测试

这是单独的测试程序，用已有 A/B 录像检查 RTX 4060 在 Windows 上的解码速度。
不连接相机、AFG 或 PZT，也不运行微泡识别或替换正式实验入口。

权威源码是 `customer_workspace/tools/gpu_native_decode.py`。
Windows 运行副本、独立 Python 3.12、依赖和视频副本位于 `F:\MB_gpu_decode`；
`installation.json` 记录副本来源及 SHA-256。不要在运行副本里直接开发。

## 按 60 FPS 测试

Windows PowerShell 中运行：

```powershell
F:\MB_gpu_decode\python312\python.exe -m customer_workspace.tools.gpu_native_decode `
  --video F:\MB_gpu_decode\videos\expA.mp4 `
  --output-dir F:\MB_gpu_decode\runs\A_new_test
```

把 `expA.mp4` 改为 `expB.mp4` 即可测试 B。输出目录必须不存在。
默认处理全部 1320 帧；可以用 `--frames 120` 先做短测。

每帧按固定的 1/60 秒计划到达；提前 1 ms 从睡眠切到循环等待。
等待下一帧、读文件、预热和最终写盘都不计入解码时间。
如果一帧变慢，后续帧保留原定时刻，真实晚到量会写入 `late_ms`；本测试不主动丢帧。

`summary.json` 保存中位数、P95、P99、最大值及超过 16.67 ms 的次数；
`frames.csv` 保留每帧的来源编号、来源时间、提交和完成时间、分段耗时。
`decode_ready_ms` 是提交压缩包到显存图像完成同步的时间，包含驱动交接与等待，
不能把它当作解码芯片的纯计算时间。
Windows 记录还保留 QPC 计数，可直接对应 Windows 系统事件。

## 单独检查画面是否正确

```powershell
F:\MB_gpu_decode\python312\python.exe -m customer_workspace.tools.gpu_native_decode `
  --video F:\MB_gpu_decode\videos\expA.mp4 `
  --verify --output-dir F:\MB_gpu_decode\runs\A_new_pixel_check
```

这一步下载显存图像，与 CPU 解码得到的原生 NV12 图像逐像素比较，输出 `pixel_check.csv`。
它不按实时速度播放，不用于速度结论。普通计时模式不下载完整图像。
当前 ZERO 模式返回的输出 PTS 不可靠，所以保留真实输入 PTS，并要求一包立即输出一帧；
全视频的逐像素、顺序核对用于确认关联正确。

## 与 WSL 公平对照

在项目根目录使用相同脚本：

```bash
customer_workspace/.venv_nvdec/bin/python -m customer_workspace.tools.gpu_native_decode \
  --video customer_workspace/microbubble_tracking/data/raw/expA.mp4 \
  --output-dir customer_workspace/validation/wsl_nvdec_new_test
```

两端使用相同的压缩包、解码参数、8 帧预热、送帧方法和同步要求。
只测试解码，没有把后续图像处理内核放进对照；因此与早期包含图像处理的总时间不能直接混用。

## 检查 0xC5 通知

使用已有的限时 Windows 诊断入口，加上显式原生目录：

```bash
.venv/bin/python -m customer_workspace.tools.gpu_decode_host_trace \
  --native-root /mnt/f/MB_gpu_decode --profile GpuLight --rounds 2 \
  --output-dir customer_workspace/validation/native_gpu_new_trace
```

这会请求 Windows 管理员权限，仅录制最长 140 秒的诊断数据并自动停止，
期间原生回放 A/B 各两遍。无需另外启动计时任务。
系统诊断会增加开销，应与没有诊断的速度测试分开报告。

读取原生帧与系统事件时，`gpu_host_report` 使用 `--native`；它直接比较 Windows 的同一时钟，
不相减 WSL 与 Windows 的计时器。是否发生阻塞，需要检查通知期间的显卡锁和具体解码线程，
不能只根据“附近有通知”就下结论。

官方依据：[NVIDIA 安装说明](https://docs.nvidia.com/video-technologies/pynvvideocodec/read-me/building-library.html)
及[系统要求](https://docs.nvidia.com/video-technologies/pynvvideocodec/read-me/system-requirements-common.html)。
