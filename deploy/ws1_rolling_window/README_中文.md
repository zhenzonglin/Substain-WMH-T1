# 工作站1滚动200例热修复

本包只修改三处运行逻辑：

- 将固定的`200例全部cleanup后再开下一批`改为滚动窗口；
- 同时最多放行200例，每出现1个终态`cleanup.json`就立即补入下一例；
- 保留现场已有的T1 GPU0、WMH GPU0、registration CPU 8线程及cleanup改动。

CPU资源不扩容：Snakemake仍使用96核总预算。普通CPU重任务沿用每例4线程，registration沿用每例8线程。T1和WMH继续共享1个GPU令牌，在GPU0串行执行。

## 使用

把整个解压目录放到工作站1，例如：

```bash
/data/usersdir/linzhenzong/ws1_rolling_window_hotfix_20260904
```

然后只运行：

```bash
bash /data/usersdir/linzhenzong/ws1_rolling_window_hotfix_20260904/deploy_ws1_rolling_window.sh \
  /data/usersdir/linzhenzong/Substain
```

脚本会依次：核对现场T1 GPU和registration 8线程、检查补丁兼容性、保存当前进程组和活动阶段清单、向旧进程组发送TERM并最多等待60秒、归档被TERM打断后生成的失败状态、备份两个被修改文件、应用补丁、解析Snakemake滚动规则、解锁并使用原`start_ws1_v1_0_9.sh`重启。

脚本不会使用KILL。若TERM后仍有进程，脚本会停止操作，不应用补丁，也不重启。

部署后查看：

```bash
cd /data/usersdir/linzhenzong/Substain
tail -n 80 "$(cat logs/full_run_v1.0.9.logpath)"
envs/core-venv/bin/python scripts/monitor_ws1_progress_resources.py --interval 30
```

日志出现以下内容即表示新调度已启用：

```text
启动滚动队列: total=..., window=200, cores=96, gpu=1
```
