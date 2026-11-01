# Windows 本地模拟相机

这个工具把已采集的 A、B 视频按原来的速度循环发送，供 Windows 程序当作
RTSP 相机读取。工作目录是 `F:\MB_camera_sim`。它使用视频副本，不启动微泡
分析程序，也不连接真实相机、AFG 或 PZT。

本次安装的实际验收结果见 `F:\MB_camera_sim\VALIDATION_REPORT.md`；
可复查的录像、逐帧记录和日志均在该报告注明的 `runs` 子目录中。

## 1. 开始发送

在 Windows PowerShell 中运行：

```powershell
cd F:\MB_camera_sim
.\.venv\Scripts\python.exe -X utf8 -B .\windows_camera_sim.py serve
```

保持这个窗口运行。它同时提供两个地址：

| 视频 | Windows 本机 RTSP 地址 |
| --- | --- |
| A | `rtsp://127.0.0.1:8554/expA` |
| B | `rtsp://127.0.0.1:8554/expB` |

每段原视频都是约 22 秒、1320 帧，分辨率为 1280×1024，H.264 编码，约 60 FPS。
播完后从头继续。客户端随时接入，接入位置不一定是第 0 帧；这是持续直播的正常行为。

按 `Ctrl+C` 结束，程序只关闭它自己启动的发送进程与 MediaMTX。
默认只监听 Windows 的 `127.0.0.1`，本轮没有开放给外部电脑或接入 WSL 分析端。
如果提示端口被占用，请先结束已有的模拟服务；程序不会替您关闭其他服务。

## 2. 观看画面

保留发送窗口，在另一个 Windows PowerShell 窗口中运行：

```powershell
cd F:\MB_camera_sim
.\.venv\Scripts\python.exe -X utf8 -B .\windows_camera_sim.py preview --source expA --seconds 22
```

看 B 时把 `expA` 改为 `expB`。这会打开 Windows FFplay 播放器，显示到指定
时长后正常关闭该播放器窗口。RTSP 地址应交给视频播放器，不能用普通网页方式打开。

也可以从 WSL 终端启动 Windows 工具，仍由 Windows 发送和接收：

```bash
/mnt/f/MB_camera_sim/.venv/Scripts/python.exe -X utf8 -B \
  F:/MB_camera_sim/windows_camera_sim.py serve
```

这条命令只借用 WSL 终端启动 Windows Python，不运行 WSL 图像分析。

## 3. 自动核查完整性和速度

先停止手动启动的 `serve`，再运行：

```powershell
cd F:\MB_camera_sim
.\.venv\Scripts\python.exe -X utf8 -B .\windows_camera_sim.py verify --repeats 2
```

一次约需 5～6 分钟。程序自行启动和关闭模拟服务，对 A、B 各接收两次，最后
分别打开播放器检查播放过程。验证默认让接收端使用 UDP；可以另加
`--transport tcp` 单独测试 TCP 接收。文件上传到本机 MediaMTX 始终使用 TCP。

每次结果放入新的 `runs/年月日-时分秒Z_verify` 目录；`Z` 表示目录名采用 UTC。
系统日志中的 Windows 本地时间与 UTC 目录名可能相差一个小时。

主要检查如下：

1. 接入直播后，按输入视频时间预留开头 2 秒准备，排除建立解码状态时尚无完整时间戳的开头；录像还会
   等待能独立解码的关键帧。该准备过程不计入正式接收窗口，也不视为已经保存。
   此后每次接收 2760 帧，超过两个原视频周期，允许客户端从中途接入。
2. 将实时解码结果和保存录像分别与源视频逐帧对照，整段接收序列必须连续；
   中途缺帧、重复帧或画面损坏都会失败。
3. 从录像中准确截取一整段 1320 帧的源视频周期，再次逐帧核对。
4. 检查保存的视频时间是否连续，实际接收速度是否接近 60 FPS，并记录较长间隔。
5. 检查接收、发布、服务器和播放日志中的异常。

| 文件 | 用途 |
| --- | --- |
| `summary.json` | 整次测试是否通过及 A、B 各次结果 |
| `expA_01/expA_complete.mp4` | 第一次接收中完整的一段 A；其余目录同理 |
| `expA_01/received.mp4` | 包含接入起点的整段接收录像，约 46 秒 |
| `expA_01/received_frames.csv` | Windows 每次收到解码校验结果的时间和帧内容校验值 |
| `expA_01/result.json` | 帧数、逐帧一致性、实际速度和时间间隔 |
| `mediamtx.log`、`publish_*.log`、`receive.log` | 服务器、发送和接收异常 |
| `play_expA.json`、`play_expB.json` | 播放器打开、正常关闭和错误检查结果 |
| `shutdown.json` | 本次创建的进程如何结束 |

实际接收速度依据 Windows 收到 FFmpeg 解码校验结果的时间计算，包含解码和
进程传递的影响，**不能当作纯网络延迟或相机曝光时间**。FFmpeg 的录像复制输出
和实时解码输出可能从相邻的不同帧开始，因此分别对照源视频，而不是直接比较
两个输出的第 0 行。准备时间、偏移和完整性结论均保存在报告中。程序不把缺少
时间戳的开头伪造为正常时间，也不忽略正式窗口中的缺帧。

## 4. 数据流程和依赖

```text
Windows F 盘上的 A、B 视频副本
            ↓ 按原视频速度读取，直接复制 H.264
     Windows FFmpeg 发布端
            ↓ 本机 TCP 上传
     Windows MediaMTX
            ↓ RTSP；验证默认 UDP 接收
     Windows FFmpeg / FFplay
            ↓
     接收录像、完整片段、帧核对和播放日志
```

发送和录像使用 `-c:v copy`，不重新压缩；解码只用于核对画面或显示播放。
这是 [MediaMTX 官方 FFmpeg 发布方式](https://mediamtx.org/docs/publish/ffmpeg)
的本机视频应用。播放器使用 [FFplay](https://ffmpeg.org/ffplay.html)。

本次部署使用 FFmpeg/FFprobe/FFplay 9.0.2、MediaMTX 1.21.3、Windows Python 3.14.2。
F 盘 `.venv` 仅使用 Python 标准库，依赖本机已有 `C:\Python314`；没有安装新的
Python 功能包。工具版本见 `tool_versions.json`，文件来源和校验值见
`installation.json`。没有注册后台服务或修改防火墙。

维护代码位于项目 `customer_workspace/tools/windows_camera_sim.py`；服务配置位于
`customer_workspace/configs/mediamtx_camera_sim.yml`。F 盘只部署这一个独立工具、
必要配置与视频副本，不是另一份实验项目。

本轮只核查 Windows 本机视频发送、接收和播放。真实相机网络状况、Windows 到
WSL 的传输以及微泡识别性能，需要在后续对应测试中另行验证。
