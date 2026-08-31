# 工作站1 v1.0.9 停止、部署、重排队与监测

活动项目固定为 `/data/usersdir/linzhenzong/Substain`。以下命令均在工作站1执行。

## 1. clone热修复分支并核对远端

```bash
cd /data/usersdir/linzhenzong
git clone --single-branch \
  --branch codex/v1.0.9-ws1-grid-runtime-hotfix \
  https://github.com/zhenzonglin/Substain-WMH-T1.git \
  Substain-v1.0.9-ws1-hotfix

git -C Substain-v1.0.9-ws1-hotfix rev-parse HEAD
git ls-remote --heads \
  https://github.com/zhenzonglin/Substain-WMH-T1.git \
  codex/v1.0.9-ws1-grid-runtime-hotfix
```

两个commit必须相同。认证失败或commit不一致时停止，不手工复制源码。

## 2. 安全停止full_run_v1.0.8

```bash
bash /data/usersdir/linzhenzong/Substain-v1.0.9-ws1-hotfix/scripts/stop_ws1_v1_0_8.sh \
  /data/usersdir/linzhenzong/Substain
```

脚本会校验数字PID、项目cwd和`PID=PGID`，列出整个进程组，发送TERM并等待最多60秒。仍有残留时不自动KILL。

确认活动项目没有残留阶段进程：

```bash
ps -u "$USER" -o pid,ppid,pgid,stat,etime,args | \
  grep -E '/data/usersdir/linzhenzong/Substain/(workflow/Snakefile|src)|substain_features\.cli stage|substain_features\.gpu_pool' | \
  grep -v grep || true
nvidia-smi
```

## 3. 应用带备份的最小热修复

```bash
cd /data/usersdir/linzhenzong/Substain-v1.0.9-ws1-hotfix
bash scripts/apply_v1_0_9_ws1_hotfix.sh \
  /data/usersdir/linzhenzong/Substain
```

脚本先检查现场存在`stage_skullstrip`和`error_records_only_v1`，再执行`git apply --check`。源码上下文不匹配时不会修改。备份写入活动项目的`archive/hotfix-v1.0.9-backup-*`。

## 4. 测试与Snakemake解锁

```bash
active=/data/usersdir/linzhenzong/Substain
hotfix=/data/usersdir/linzhenzong/Substain-v1.0.9-ws1-hotfix
core="$active/envs/core-venv/bin/python"

PYTHONPATH="$active/src" "$core" -m pytest -q \
  "$hotfix/tests/test_images.py" \
  "$hotfix/tests/test_wmh.py::test_wmh_order_and_ml_are_native_voxel_based" \
  "$hotfix/tests/test_wmh.py::test_wmh_volume_is_unchanged_for_subthreshold_header_rounding" \
  "$hotfix/tests/test_wmh.py::test_wmh_grid_error_reports_physical_diagnostics" \
  "$hotfix/tests/test_status_runtime.py" \
  "$hotfix/tests/test_cleanup.py" \
  "$hotfix/tests/test_monitor_ws1.py" \
  "$hotfix/tests/test_ws1_rerun.py"

PYTHONPATH="$active/src" "$core" -m pytest -q "$active/tests"

cd "$active"
PYTHONPATH="$active/src" "$core" -m snakemake \
  --snakefile "$active/workflow/Snakefile" \
  --configfile "$active/config/config.yaml" \
  --cores 1 --unlock \
  --config \
    "active_config_file=$active/config/config.yaml" \
    selected_participant=all profile=gpu gpu_devices=0
```

任一测试失败时不要重排队或启动。

## 5. 冻结名单并执行安全重排队

先只预览：

```bash
cd /data/usersdir/linzhenzong/Substain
envs/core-venv/bin/python scripts/prepare_ws1_v1_0_9_rerun.py \
  --project-root /data/usersdir/linzhenzong/Substain
```

核对动态发现的网格失败数、3个代表病例和TERM中断数后执行：

```bash
envs/core-venv/bin/python scripts/prepare_ws1_v1_0_9_rerun.py \
  --project-root /data/usersdir/linzhenzong/Substain \
  --apply
```

网格失败病例的旧`status/`、`logs/`和中央QC图会归档，残留影像中间件会删除并从头重跑。TERM中断病例只移走本次中断后产生的失败状态和`cleanup`状态，保留此前有效pass阶段与中间件。清单指针为`logs/ws1_v1.0.9_rerun_manifest.path`。

## 6. 启动v1.0.9

```bash
cd /data/usersdir/linzhenzong/Substain
bash scripts/start_ws1_v1_0_9.sh
```

固定参数为每例4线程、200例波次、96 cores和GPU 0。队列顺序为：3个代表病例、其余WMH网格失败病例、TERM中断病例、从未处理病例。其他历史真实失败不会重跑。

启动后立即检查：

```bash
run_pid="$(cat logs/full_run_v1.0.9.pid)"
run_log="$(cat logs/full_run_v1.0.9.logpath)"
ps -o pid,ppid,pgid,stat,etime,%cpu,%mem,args -p "$run_pid"
tail -n 120 "$run_log"
grep -n 'rule audit:' "$run_log" && echo '异常：出现audit' || echo '确认：未运行audit'
```

## 7. 四个监测终端

终端1，主进程与RAM/GPU/OOM告警：

```bash
cd /data/usersdir/linzhenzong/Substain
envs/core-venv/bin/python scripts/monitor_ws1.py main --interval 30
```

终端2，当前子任务：

```bash
cd /data/usersdir/linzhenzong/Substain
envs/core-venv/bin/python scripts/monitor_ws1.py jobs --interval 30
```

终端3，各阶段完成、失败、运行中、待处理、比例及最新耗时：

```bash
cd /data/usersdir/linzhenzong/Substain
envs/core-venv/bin/python scripts/monitor_ws1.py progress --interval 30
```

终端4，主日志：

```bash
cd /data/usersdir/linzhenzong/Substain
tail -F "$(cat logs/full_run_v1.0.9.logpath)"
```

资源历史写入`logs/ws1_resource_history.tsv`，阶段耗时写入`logs/ws1_stage_runtime_history.tsv`。内存、swap、OOM、孤儿进程和空闲GPU显存异常只告警，不自动限流或停止。
