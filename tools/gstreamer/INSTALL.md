# 第二轮 Windows GStreamer 环境

本环境只供 `customer_workspace.parallel_camera` 接收和分流视频使用。实时分析仍在原 WSL `.venv` 中运行，F 盘原相机 Python 环境没有安装或修改任何包。

安装位置是 `customer_workspace/.venv_gstreamer`，当前约 411 MiB，已排除普通 Git 同步。它由现有 Windows Python 3.14.2 创建，运行于 Windows 11（10.0.26200）。使用 [GStreamer 官方提供的 Python 安装方式](https://gstreamer.freedesktop.org/download/)，包来自官方 PyPI。

这次安装的入口包是 `gstreamer-bundle==1.28.7`，其依赖允许同系列的小版本更新。实际核心、Python 绑定等为 1.28.8，插件等为 1.28.7；这组版本已经通过本次真实采集，不把它们统称为“全套 1.28.8”。最初尝试入口包 1.28.8 时，它要求的插件 1.28.8 没有可安装文件，因此改用 1.28.7。实际所有版本已固定在 [requirements.txt](requirements.txt)。完整下载地址、包大小及 SHA-256 保存在 [安装清单](../../validation/parallel_20261025-145352/gstreamer_install.json)。

若今后需要在一个**尚不存在的新目录**重建环境，从 WSL 项目根目录执行：

```bash
/mnt/f/CHRONOS_control/.venv/Scripts/python.exe -m venv customer_workspace/.venv_gstreamer
chmod u+x customer_workspace/.venv_gstreamer/Scripts/python.exe
customer_workspace/.venv_gstreamer/Scripts/python.exe -m pip install \
  --index-url https://pypi.org/simple \
  -r customer_workspace/tools/gstreamer/requirements.txt
```

上述路径对应本机原相机解释器；使用它创建新环境，不向其自身安装依赖。现有环境已经准备好，无须重复执行安装。FFmpeg 沿用第一轮 [独立工具](../ffmpeg/INSTALL.md)。

`parallel_camera.py` 在自己的进程中指定包内实际的 `Lib/girepository-1.0` 路径，解决 Windows 包默认小写 `lib` 路径在 WSL 共享目录上找不到类型信息的问题，不改全局环境变量或旧程序。

本次 Windows 启动日志仍有两类提示：可选的 `giolibproxy.dll` 加载失败，以及相机 RTSP 回复中的 `host=...` 字段未被 GStreamer 识别。实际连接、收包、完整保存和结束请求均成功；GStreamer 数据处理总线没有错误或告警。原始提示保留在每次运行的 `bridge_*/windows_stderr.log` 中，本轮没有进一步修改系统代理组件或相机协议实现。

第三轮在这个独立环境新增 `psutil==7.2.2`，每秒读取一次 Windows 接收器和 FFmpeg 的驻留、私有内存。Windows 安装文件约 137 kB；[安装来源和校验清单](../../validation/stability_20261025-151417/psutil_install.json)已保留。WSL 直接复用原 Pipeline 已有的每秒内存记录，不重复创建监测线程，也没有向 F 盘原环境安装依赖。指标含义见 [psutil 官方文档](https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info)：驻留内存反映当时驻留在物理内存中的部分，不能等同于全部私有分配量。
