# WS1 rolling-window patch

这是工作站1的轻量补丁分支，不包含完整项目源码、环境、模型、影像或分析结果。

## 最新：GPU0可共享，显存不足自动CPU（2026-09-05）

已运行滚动200例、GPU0双槽版本的工作站1，使用本节的新部署器。下方旧章节保留作为历史说明，**不要在新补丁后重复运行旧部署器**。

```bash
(
set -euo pipefail
cd /data/usersdir/linzhenzong/Substain-ws1-rolling-patch-lite
git pull --ff-only
sha256sum -c SHA256SUMS
bash deploy_ws1_t1_hybrid.sh /data/usersdir/linzhenzong/Substain
)
```

- T1在启动时非阻塞尝试GPU0的一个槽位；槽忙或GPU0空闲显存低于`12288 MiB`时，马上使用原`--profile cpu`入口。CPU任务不会中途切到GPU。
- GPU版T1最多两例。CPU版T1不另设并行上限，每例在Snakemake中占4线程，与其他阶段共享96核预算。CPU预算可能被T1占满，使上游等待。
- T1的Snakemake `gpu=0`表示不预先占用GPU调度令牌，不表示强制CPU。实际GPU占用由文件锁控制。
- WMH请求2个GPU令牌，并独占GPU0的两个项目槽；取得双槽后若GPU0空闲显存低于`20480 MiB`，该例WMH转CPU并释放GPU锁。WMH等待项目内旧T1期间不因槽忙提前转CPU。
- 其他用户可以继续使用GPU0；本项目不读取其命令、不停止其进程。显存查询失败时采用CPU。阈值可通过配置中的`execution.t1_gpu_min_free_memory_mib`和`execution.wmh_gpu_min_free_memory_mib`覆盖。
- 只使用GPU0，不检测或调用GPU1/2。不增加推理失败后的跨设备重试。
- 保留registration 8线程、滚动200例、原V1.0.9入口、现有状态/runtime和cleanup。不改环境、配置、数据、资源或V1.1标签。

`ws1_t1_hybrid.patch`只修改`workflow/Snakefile`和`src/substain_features/gpu_pool.py`。部署器支持此前WMH令牌为1或2的两个已知双槽版本；未知GPU锁源码或补丁上下文会在停止分析前中止。Python可为合法的虚拟环境软链接。

2026-09-05兼容修正：工作站实测`Snakefile`校验值`f29d004d...`的算法内容正确，但统一diff因局部文本上下文差异不能应用。部署器现在先核对该完整校验值，再对GPU包装器、WMH和T1三个已知片段逐项做唯一匹配的语义修改；任何片段缺失或重复仍会在停止分析前中止。GPU锁模块继续使用最小diff且校验应用前后哈希。

部署器核对实际derivatives、96核/4线程/200例设置及PID/PGID/cwd/启动参数，备份两个源码文件与活动阶段，只发TERM并等最多60秒。只有状态在本次停止时间内新写入、且明确含SIGTERM/退出143等标记的失败才归档；有效完成结果和历史失败保留。不能确认的新失败会阻止重启，等待人工检查。为本次中断且缺失的输出清理对应Snakemake元数据，不全局重置状态。

若部署器报告“TERM后60秒仍有活动进程”，先确认输出中的旧PID和PGID均已退出、且本项目没有分析进程，再从它输出的同一个归档续部署；不要重新执行首次部署，也不要使用KILL：

```bash
bash deploy_ws1_t1_hybrid.sh /data/usersdir/linzhenzong/Substain \
  --resume /data/usersdir/linzhenzong/Substain/archive/t1-hybrid-XXXXXX
```

续部署会重新校验陈旧PID文件、旧PID/PGID、项目进程、源码、归档备份和derivatives路径。任一项改变都会在覆盖源码前中止；校验通过后才归档可确认由该次TERM产生的失败状态、应用补丁、验证、解锁并重启。

编译、Shell检查、Snakemake解析及解锁通过后，恢复原分析入口。验证失败恢复源码并保持停止；重启后的运行错误只报告，不在活动进程下覆盖源码。归档位置输出为`archive/t1-hybrid-*`，其中`before/`可恢复原两个文件；恢复前必须先停止并核验项目进程。

### 验证和观察

不使用模型或实际GPU的Linux测试：

```bash
cd /data/usersdir/linzhenzong/Substain-ws1-rolling-patch-lite
/data/usersdir/linzhenzong/Substain/envs/core-venv/bin/python \
  verify_ws1_t1_hybrid.py /data/usersdir/linzhenzong/Substain --snakemake
```

Linux进程测试已通过两GPU T1、槽位/显存CPU回退、WMH等待/独占、显存查询失败、并发竞争、失败和TERM释放。此前在Snakemake 7.32.4中的真实调度测试同时启动24个四线程T1，证明96核预算生效且不被两个GPU令牌限制；本次未改变该调度资源。显存分支测试使用假的`nvidia-smi`数值，不执行影像、模型或A100推理，不等于工作站病例验收通过。

完整检查结果和未通过项见[验证记录](VALIDATION_T1_HYBRID.md)。全项目pytest仍有5项旧断言或本地缺少资源导致的失败，未宣称全项目测试全绿。

运行日志会出现：

```text
GPU_DISPATCH stage=t1 participant=... device=gpu0 reason=slot_acquired slots=1
GPU_DISPATCH stage=t1 participant=... device=cpu reason=slots_busy slots=0
GPU_DISPATCH stage=t1 participant=... device=cpu reason=admission_busy slots=0
GPU_DISPATCH stage=t1 participant=... device=cpu reason=free_memory_12000_lt_12288 slots=0
GPU_DISPATCH stage=wmh-seg participant=... device=gpu0 reason=slot_acquired slots=2
GPU_DISPATCH stage=wmh-seg participant=... device=cpu reason=free_memory_20000_lt_20480 slots=0
```

`admission_busy`表示准入锁被占用（通常为等待或运行的WMH，也可能是另一T1短暂的取槽操作）。选择设备后，子进程实际成功状态里的`details.effective_device`与runtime仍是最终判断依据。

新版窗口3原命令继续使用：

```bash
cd /data/usersdir/linzhenzong/Substain &&
envs/core-venv/bin/python scripts/monitor_ws1_progress_resources.py --interval 30
```

工作站验收仍需至少两例GPU版T1、一例CPU版T1和一例WMH成功完成，核对有效设备、耗时、cleanup及无CUDA OOM。仅看到设备分配日志不能视为推理成功。

仓库内容只有：

- `ws1_rolling_window.patch`：滚动200例调度补丁；
- `deploy_ws1_rolling_window.sh`：停止旧进程、校验现场、应用补丁并重启；
- `ws1_dual_t1_wmh_exclusive.patch`：GPU0双T1、WMH独占增量补丁；
- `deploy_ws1_dual_t1.sh`：校验现场、TERM停止、备份、应用双槽补丁并重启；
- `ws1_gpu_throughput_first.patch`：让WMH请求两个Snakemake GPU令牌，避免等待中的WMH占住第二个T1令牌；
- `deploy_ws1_gpu_throughput.sh`：从已部署双槽版本切换为吞吐优先并重启；
- `verify_ws1_gpu_slots.py`：不运行模型的Linux文件锁行为测试；
- `ws1_t1_hybrid.patch`、`ws1_t1_hybrid_manifest.json`：T1混合设备增量及已知GPU锁校验值；
- `deploy_ws1_t1_hybrid.sh`、`deploy_ws1_t1_hybrid.py`：混合设备部署入口与安全检查；
- `verify_ws1_t1_hybrid.py`、`test_ws1_t1_hybrid_deploy.py`：Linux并发/实际调度测试和部署安全测试；
- `SHA256SUMS`：文件校验和。

历史双槽补丁中T1和WMH均使用GPU0；最新混合设备补丁允许其他用户同时占用GPU0，并按空闲显存让T1或WMH转CPU。各版本均保留registration CPU 8线程、cleanup删除形变场、96核总预算，以及每完成一个终态`cleanup.json`补入一例的滚动200例逻辑。

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

## 启用GPU吞吐优先

若监测发现一个T1持有1槽、一个等待中的WMH包装器占用另一个Snakemake令牌，执行下列增量部署。修改后，有可运行T1时可持续填充两个T1令牌；WMH必须同时取得两个Snakemake令牌和两个物理槽，因此仍不会与T1重叠。代价是T1积压期间WMH可能延后。

```bash
cd /data/usersdir/linzhenzong/Substain-ws1-rolling-patch-lite
git pull --ff-only
sha256sum -c SHA256SUMS

bash deploy_ws1_gpu_throughput.sh \
  /data/usersdir/linzhenzong/Substain
```

该脚本仅修改活动项目的`workflow/Snakefile`。它先验证双槽版本、PID、PGID、cwd和补丁上下文，再TERM停止；随后备份、应用、解析、解锁并通过原V1.0.9入口重启。失败时不会使用KILL。

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
