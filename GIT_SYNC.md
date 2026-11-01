# 工作区的新分支同步

本次目标是 `OZAKI39/MB_track` 的 `customer-workspace/20261101-203824` 分支。
该分支根目录对应本机的 `customer_workspace/`；`microbubble_tracking/` 是其中的子目录。
远端 `main` 和既有备份分支仍采用原来的追踪子项目结构，本次不更新它们。

## 同步范围

- 保存当前工作区的代码、测试、配置和使用说明。
- 继续保存 `microbubble_tracking/results/`、`figures/`、`data/`、`history/`
  中已纳入版本管理及本次新增的研究结果，不重新计算或覆盖它们。
- `validation/` 保存报告、汇总、核对结果、必要图表、人工修订和固定测试输入；
  同时保留 README 示例所需的两份相机回放背景。
- 完整系统跟踪、重复回放数据库、逐帧大表、重复视频、临时运行状态、
  外部工具安装目录和虚拟环境保留在本机。未上传不表示可以删除，也不表示已有备份。
- 本次具体文件选择、大小及校验值见
  [同步核对记录](validation/workspace_publish_20261101-203824/REPORT.md)。
  对未上传的大文件只记录路径、大小和原因，不凭空声明存在远端副本。

## 使用时保留父工程

这是工作区的版本快照，不是可独立安装的整个实验系统。
相机与 AFG 入口仍导入父工程的 `microbubble_experiment/`，并读取父工程配置。
使用这些入口时，应把本分支内容放在原父工程的 `customer_workspace/` 位置，
继续按 README 从父工程根目录运行。不能把本分支根目录直接当成完整父工程。
机器专用的 `configs/machine.local.json`、Windows 环境和真实设备连接不在本次同步范围。
本次未改动或部署 Windows 设备程序。

## 后续同步方法

唯一开发目录仍是 `/home/lzy/projects/new_microbubble_experiment`。
使用该目录已有 Git 仓库，不建立源码副本、嵌套仓库或额外工作树：

1. 检查状态、差异及大文件，按具体路径选择需要提交的文件。
2. 检查暂存差异，再提交本次工作区变更；其他目录的未提交修改保持原样。
3. 用 `git subtree split --prefix=customer_workspace --branch <新的本地分支名>`
   提取工作区历史，再核对分支完整文件清单。
4. 用户明确要求推送后，使用 `git push mb-track <本地分支>:<新远端分支>`。
   不强制推送，不改写 `main` 或历史备份分支。

不要直接 `git add -A` 或整目录无差别暂存：本地 `validation/` 还包含未上传的大量原始记录。
追踪子项目单独同步到 `main` 的旧方法见
[microbubble_tracking/GIT_SYNC.md](microbubble_tracking/GIT_SYNC.md)，它与本页的新分支范围不同。
