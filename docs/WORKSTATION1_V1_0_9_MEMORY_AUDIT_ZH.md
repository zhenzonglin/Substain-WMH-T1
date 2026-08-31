# 工作站1 v1.0.9 内存与显存静态审查

## 结论

当前实现未发现会跨病例长期持有NIfTI数组、Python模型对象或CUDA张量的明确代码路径。每个Snakemake阶段由独立Python进程执行，ANTs、SynthStrip、WMH-SynthSeg和DLMUSE均通过阻塞式`subprocess.run()`等待子进程退出。阶段进程结束后，进程内NumPy数组和子进程CUDA上下文由操作系统回收。

这不代表96 cores运行时不会出现内存压力。`CPU_THREADS_PER_JOB=4`允许约24个CPU重任务并发；多个阶段会用`get_fdata()`同时物化三维数组，ANTs和DLMUSE子进程还会额外占用RAM。Snakefile只声明CPU线程和GPU令牌，没有声明按GB计的RAM资源，因此风险是并发峰值，不是已证实的逐病例泄漏。

## 已检查路径

- `src/substain_features/gpu_pool.py`：循环只等待文件锁；获得锁后用阻塞式`subprocess.run()`执行一个GPU任务，并在`finally`释放锁和关闭句柄。
- `src/substain_features/wmh.py`、`synthstrip.py`、`registration.py`、`t1.py`：外部工具均用阻塞式`subprocess.run()`，没有后台`Popen`队列。
- `src/substain_features/images.py`、`wmh.py`、`t1.py`：存在多个`get_fdata()`大数组，但这些对象属于单阶段短生命周期进程。
- `workflow/Snakefile`：CPU重阶段使用`threads: CPU_HEAVY_THREADS`，GPU阶段使用单个`gpu=1`令牌；没有RAM资源上限。
- 未发现`lru_cache`、全局病例结果列表、常驻模型池或未回收`Popen`对象。

## 本次采取的策略

按用户选择保持96 cores、每例4线程、200例固定波次和GPU 0，不自动降低并发。`scripts/monitor_ws1.py main`每30秒记录系统RAM、项目RSS、swap、OOM和GPU显存；出现以下条件时只告警：

- `MemAvailable < 10%`；
- 项目RSS超过总RAM的80%；
- swap相对监测基线增加至少2 GiB；
- `/proc/vmstat`的`oom_kill`计数增加；
- 出现不属于主PGID的项目进程；
- GPU利用率低于5%但显存占用超过1 GiB。

若运行中出现持续单调增长而不是随阶段退出回落，应根据`logs/ws1_resource_history.tsv`和`logs/ws1_stage_runtime_history.tsv`定位具体阶段，再决定是否降低cores；本版本不自动暂停或杀进程。
