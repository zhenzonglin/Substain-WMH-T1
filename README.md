# WS1 rolling-window patch

这是工作站1的轻量补丁分支，不包含完整项目源码、环境、模型、影像或分析结果。

仓库内容只有：

- `ws1_rolling_window.patch`：滚动200例调度补丁；
- `deploy_ws1_rolling_window.sh`：停止旧进程、校验现场、应用补丁并重启；
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
