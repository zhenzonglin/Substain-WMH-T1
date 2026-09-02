# V1.1.0-rc1 新项目源码迁移

## 发布边界

`v1.1.0-rc1`是私人GitHub源码候选版。它包含算法源码、工作流、测试、配置模板、小型标准模板与许可证，不包含原始病例、真实metadata、派生结果、已安装环境、大型模型、常模或受限制第三方源码。

该版本用于建立全新项目，不用于覆盖正在运行的V1.0.x目录。旧项目的数据与结果应保持原位。

## 建立新项目

```bash
git clone --branch v1.1.0-rc1 --depth 1 \
  https://github.com/zhenzonglin/Substain-WMH-T1.git Substain-v1.1.0-rc1
cd Substain-v1.1.0-rc1
cp config/metadata.example.tsv config/metadata.tsv
```

按`config/config.yaml`填写BIDS或folders输入根目录、病灶后缀和`metadata.tsv`。输入目录可以挂载或建立项目内符号链接；不得复制旧项目的`participants.tsv`，它由`prepare`重新生成。

## 补齐获授权资源

根据`resources/source_versions.tsv`和`resources/licenses/license_inventory.tsv`补齐模型、常模、ANTs、SynthStrip及DLMUSE/NiChart资源。不得从GitHub源码标签推断这些文件已经包含，也不得绕过上游许可。

联网准备机可以运行：

```bash
bash scripts/install_core.sh
bash scripts/download_resources.sh
bash scripts/install_full_envs.sh
```

完整离线迁移必须在资源齐备的获授权工作站另行运行`scripts/build_offline_bundle.sh`；本RC源码发布本身不生成离线tar包。

## 验证与运行

```bash
./run_pipeline.sh offline
./run_pipeline.sh prepare
./run_pipeline.sh audit all
```

audit通过后可前台运行：

```bash
CPU_THREADS_PER_JOB=4 BATCH_SIZE=200 CUDA_VISIBLE_DEVICES=0 \
  ./run_pipeline.sh run all gpu 96
```

也可用独立进程组启动：

```bash
TOTAL_CORES=96 CPU_THREADS_PER_JOB=4 BATCH_SIZE=200 \
  PIPELINE_PROFILE=gpu CUDA_VISIBLE_DEVICES=0 \
  bash scripts/start_full_run.sh
```

监测命令：

```bash
envs/core-venv/bin/python scripts/monitor_pipeline.py main --interval 30
envs/core-venv/bin/python scripts/monitor_pipeline.py jobs --interval 30
envs/core-venv/bin/python scripts/monitor_pipeline.py progress --interval 30
tail -F "$(cat logs/full_run.logpath)"
```

需要中断时运行：

```bash
bash scripts/stop_full_run.sh
```

停止脚本会核对PID、项目目录和独立进程组，只发送TERM；60秒后仍有残留时停止操作，不自动KILL。

完成计算后再独立执行人工QC、导出和结果验证：

```bash
./run_pipeline.sh qc all
./run_pipeline.sh export all
./run_pipeline.sh verify
```
