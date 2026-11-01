# customer_workspace 新分支同步核对

目标仓库： https://github.com/OZAKI39/MB_track
新分支： `customer-workspace/20261101-203824`

## 内容与目录结构

本分支根目录对应 `customer_workspace/`，其中保留 `microbubble_tracking/` 子目录。
使用当前父工程的 Git 提交和 subtree split；没有新建源码镜像、clone、嵌套仓库或工作树。
既有远端 main 和 backup 分支不更新，不使用强制推送。

本次已核对 6509 个原有/当前文件，共 603.40 MiB；
另加入本次同步说明、数据字节保护规则及本目录的核对记录。
代码、测试、配置、追踪子项目结果、图表、原始 A/B 样例视频和历史比较材料均保留。
validation 选择了报告、汇总表、核对证据、人工修订及固定测试输入，而不是把所有回放副本上传。

[inventory.json](inventory.json) 记录所选文件的大小、SHA-256 和选择依据。
[local_only_files.csv](local_only_files.csv) 列出另外 22182 个本地保留条目，
共 37.99 GiB，主要是系统跟踪、回放数据库、逐帧导出和外部工具。
它们没有删除或改写；文件清单本身不是数据备份。
此前已被 .gitignore 排除的环境、录像和大权重不计入这份新增候选清单，继续保留本机。

## 检查结果

- 177 个 Python 文件通过语法解析；4321 个所选 JSON 通过读取检查。
- 120 项离线测试通过，见 [tests.log](tests.log)。覆盖在线识别追踪接口、异步保存、内存清理收尾、相机桥接、等待区、延迟记录和配置等。
- 已扫描所选文本中的常见访问令牌及私钥格式，没有发现匹配项；这不等于对所有敏感信息的绝对保证。
- 原始逐帧数据不重算，baseline 不修改，模型不重新训练。
- 两份原始 Windows 机器诊断文件无法作为普通 UTF-8 JSON 读取；与机器进程清单一起留在本地，未修补或伪装成正常 JSON。
- 本轮没有连接相机、AFG 或 PZT，没有运行长时间性能测试，没有改动 Windows 服务或部署环境。
- 暂存内容逐文件核对，与本机文件字节一致；追踪子项目 results、figures、data、history 和 baseline 没有漏项。
- 工作区外的未提交修改及状态与同步前一致，没有进入本次提交。
- `git diff --cached --check` 仍报告现有代码和第三方说明中的行尾空格等排版提示；
  本次为版本同步，保持这些现有文件原样，没有顺便格式化算法或改写第三方文档。

## 运行边界

分支是工作区快照，不是完整父工程或整机环境备份。真实相机与 AFG 入口仍需要父工程的
microbubble_experiment 模块、父工程配置、机器专用路径和已有 Windows 环境。
[parent_dependencies.json](parent_dependencies.json) 记录本次引用到的父工程模块及其当前哈希；这些文件没有复制进本分支。

第三方 MobileSAM checkout 和工具安装缓存没有重复上传；MobileSAM 的来源及版本记录在 inventory.json。
已有 baseline 代码、结果、图表和输入文件仍随追踪子项目保留，不因本次同步被删除。

Git 提交、子树文件核对及远端分支身份在推送后单独核验；本报告不把尚未执行的设备或性能测试写成通过。
