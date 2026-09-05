# T1混合设备补丁验证记录（2026-09-05）

## 已确认

- 源码增量只有`workflow/Snakefile`和`src/substain_features/gpu_pool.py`。不改变影像算法、特征定义、环境、cleanup或其他GPU任务。
- Linux真实进程：两例T1占用GPU0两个槽，第三例立即CPU；释放槽后新T1再次GPU。
- WMH等待时持有准入锁；新T1走CPU，旧GPU T1完成后WMH独占两个槽。WMH执行期间新T1仍走CPU。
- 使用假的`nvidia-smi`覆盖显存分支：T1空闲显存低于12288 MiB、WMH低于20480 MiB或查询失败时改用CPU并释放GPU锁；阈值以上仍使用GPU0。
- 12个T1同时竞争时最多两个GPU任务，其余CPU；CPU子进程CUDA不可见且无旧GPU分配标记；GPU子进程仅可见0。
- 子进程失败、不等待槽位的CPU路径、包装器TERM以及等待中的WMH被TERM后，槽位可以重新使用。
- 使用现有Linux Snakemake **7.32.4**执行从真实源码提取的T1规则：96核同时运行24个四线程T1（26例测试任务），CPU任务不再受`gpu=2`限制。测试不运行模型。
- 完整Snakefile在最小输入夹具中，以CPU和GPU模式分别`--list`解析通过。
- 兼容工作站实测哈希并加入超时续部署后，Linux隔离夹具16项部署安全测试全部通过；Windows通过15项，并按预期跳过Linux系统Python软链接测试。停止、重启、故意解析失败与超时行为使用模拟进程，不触碰真实工作站。
- 只归档时间、内容和TERM标记均匹配的新失败；旧失败与成功状态不动；不能确认的新失败阻止重启。
- 两种已知旧补丁上下文均可应用；未知GPU锁源码在停止前拒绝；故意制造解析失败会恢复两份源码并保持停止；TERM超时不应用补丁、不使用KILL。
- 工作站实测`Snakefile`哈希`f29d004d...`已加入明确允许列表；局部diff失败改为严格的唯一片段语义修改。两种WMH令牌旧版本均通过测试，未知T1资源内容会在停止前拒绝。
- Python编译、所有轻量分支及模拟源码中的Shell脚本语法检查通过；最终改动通过`ruff check --select E9,F`及`git diff --check`。

## 全项目测试限制

在完整WS1源码模拟目录中执行了pytest。针对新调度行为更新相关三项旧断言后，剩余5项失败、1项跳过，不能宣称全项目测试全绿：

1. `test_lesion_order_only_dependencies_are_ancient`：仍要求旧的lesion调度门；保存的补丁前Snakefile已使用WMH调度门，本次没有修改输入依赖。
2. `test_ws1_scheduler_runs_priority_phases_without_audit`：仍要求旧的分波`run_phase`入口；本次没有修改滚动队列启动脚本。
3. 两项`test_mapping`：本地模拟目录缺少NiChart第三方MUSE字典。
4. `test_chung_formula_matches_matlab`：本地模拟目录缺少`Residual_Info.mat`。

随后额外执行旧基线对照时，本地WSL发生`Input/output error`，该次基线运行未完成，不能计为通过。停止了进一步WSL写入与清理，未重置WSL；此前Linux并发、调度、解析、编译和Shell验证已完成并留有输出。最后的静态检查及部署器夹具测试使用已有Windows工具完成。

这些限制不涉及向工作站安装新环境，也没有把受限制资源加入轻量发布。

## 尚待工作站确认

本地无工作站1的直接执行权限，本次发布不等于工作站已部署。部署器会在工作站上再次做进程锁测试、源码检查、编译、Shell检查和实际配置解析。

真实病例验收仍需至少两例GPU版T1、一例CPU版T1与一例WMH成功完成，核对`effective_device`（WMH为`device`）、runtime、cleanup及无CUDA OOM。设备分配日志或本地模拟测试不代表A100模型推理成功。
