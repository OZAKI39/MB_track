# 独立录像使用的 Windows FFmpeg

本目录使用 Gyan 提供的 FFmpeg 9.0.2 essentials 免安装构建，来源见
[FFmpeg 官方下载页](https://ffmpeg.org/download.html)中的 Windows 构建入口。
没有修改系统 PATH、相机虚拟环境或 F 盘设备程序。

- 下载来源、最终地址、下载包大小和 SHA-256：`download.json`。
- 实际使用的 `ffmpeg.exe`、`ffprobe.exe` 大小和 SHA-256：`binaries.json`。
- 构建方许可和说明：`LICENSE`、`README.txt`。
- 二进制放置于 `bin/`，压缩包及二进制不进入普通 Git 同步。

重新准备工具时，下载 `download.json` 记录的固定版本地址，核对其 SHA-256，
解压其中的两个程序到本目录 `bin/`，再核对 `binaries.json`。也可以在运行
录像命令时用 `--tools-dir` 指定包含这两个程序的目录。

录像核查使用 FFmpeg 的 `debug_ts` 日志，该格式与版本相关，因此当前入口
限定 9.0.2。升级版本需要先用已有录像验证逐包日志、汇总和文件计数一致，
不能只修改版本判断就宣称新版本已经验证。
