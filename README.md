# WS1 rolling-window patch

这是工作站1的轻量补丁分支，不包含完整项目源码、环境、模型、影像或分析结果。

仓库内容只有：

- `ws1_rolling_window.patch`：滚动200例调度补丁；
- `deploy_ws1_rolling_window.sh`：停止旧进程、校验现场、应用补丁并重启；
- `ws1_dual_t1_wmh_exclusive.patch`：GPU0双T1、WMH独占增量补丁；
- `deploy_ws1_dual_t1.sh`：校验现场、TERM停止、备份、应用双槽补丁并重启；
- `verify_ws1_gpu_slots.py`：不运行模型的Linux文件锁行为测试；
- `SHA256SUMS`：文件校验和。

补丁保留现场已有设置：T1和WMH使用GPU0，registration使用CPU 8线程，cleanup继续删除形变场。CPU总预算仍为96核。新逻辑最多放行200例，每产生1个终态`cleanup.json`就补入下一例。

## 工作站1使用方法

```bash
cd /data/usersdir/linzhenzong

git clone --depth 1 --single-branch \
  --branch codex/ws1-rolling-window-patch \
  https://github.com/zhenzonglin/Substain-WMH-T1.git \
  Substain-ws1-rolling-patch

cd Substain-ws1-rolling-patch
sha256sum -c SHA256SUMS

bash deploy_ws1_rolling_window.sh \
  /data/usersdir/linzhenzong/Substain
```

部署脚本仅对活动项目中的三个运行文件应用补丁。它不会把此轻量仓库当作完整项目，也不会替换环境、资源、配置或`derivatives/`。

停止旧任务时只发送TERM并等待最多60秒；若仍有残留进程，脚本停止操作，不使用KILL，也不应用补丁。

## 在已运行滚动队列的工作站1启用双T1

该补丁只使用物理GPU0。`gpu=2`表示GPU0上的两个逻辑槽位，不会调用GPU1或GPU2：

- 每个T1占1槽，最多同时运行2例；
- WMH占满2槽，运行期间T1不能进入；
- Snakemake额外限制最多只有1个WMH包装任务等待或运行。

在工作站1更新轻量仓库并执行：

```bash
cd /data/usersdir/linzhenzong/Substain-ws1-rolling-patch-lite
git pull --ff-only
git rev-parse HEAD
sha256sum -c SHA256SUMS

bash deploy_ws1_dual_t1.sh \
  /data/usersdir/linzhenzong/Substain
```

部署脚本要求当前项目已经应用滚动200例补丁，并保留T1 GPU、registration 8线程及cleanup实现。它会先完成补丁上下文检查，再校验PID、PGID、cwd与启动命令；随后只发送TERM。应用前的四个文件保存在`archive/dual-t1-wmh-exclusive-*`。

补丁后的正式资源保持为96个CPU核心、普通任务4线程、registration 8线程、滚动窗口200例。启动日志应包含`gpu_slots=2`和`wmh_exclusive=1`。

继续使用窗口3：

```bash
cd /data/usersdir/linzhenzong/Substain
envs/core-venv/bin/python scripts/monitor_ws1_progress_resources.py --interval 30
```

只监测GPU0：

```bash
nvidia-smi -i 0 \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw \
  --format=csv -l 1
```

## 比较旧CPU与新GPU的T1耗时

```bash
python compare_t1_cpu_gpu_runtime.py \
  --project-root /data/usersdir/linzhenzong/Substain
```

工具读取每例`status_history.jsonl`，按`effective_device`分类。主比较只使用成功、完整、未复用、未从DLICV断点恢复的T1运行，并保存明细TSV和汇总JSON到项目`logs/`。
