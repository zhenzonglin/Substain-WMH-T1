# 离线工作站迁移说明

## 1. 包的边界

`Substain_offline.tar.gz`是项目运行包，不是结果包。它包含代码、配置、测试、模型、
模板、GenMIND/Chung常模、ANTs、固定第三方源码、环境归档、核心wheel、Docker禁网
烟雾测试镜像和许可证。

以下目录明确不进入包内：

- `BIDS/`和`Lesion/`：原始影像与病灶。
- `archive/`：历史版本。
- `derivatives/`：当前两例技术结果。
- `inputs/`、`config/metadata.tsv`和`config/participants.tsv`：本机软链接和真实病例清单；
  工作站上从`metadata.example.tsv`重新建立。
- `.git/`、`build/`和`logs/`：版本库、本机构建缓存和运行日志。
- `envs/wmh`、`envs/t1`、`envs/core-venv`和`offline/envs`：绑定原机器路径的活跃环境。

## 2. 源工作站生成包

```bash
cd /path/to/Substain
./scripts/build_offline_bundle.sh
```

生成四个交付文件：

```text
Substain_offline.tar.gz
Substain_offline.tar.gz.sha256
Substain_offline.contents.txt
Substain_offline.verification.json
```

## 3. 目标工作站核验和解包

目标系统应为x86_64 Ubuntu 20.04，具备`python3`、`tar`和NVIDIA驱动。完整禁网验证还需要
Docker与NVIDIA Container Toolkit。

```bash
sha256sum -c Substain_offline.tar.gz.sha256
tar -xzf Substain_offline.tar.gz
cd Substain
./scripts/install_offline.sh
./run_pipeline.sh offline
```

安装脚本只使用包内wheel和conda-pack归档，不访问网络。环境恢复到`envs/core-venv`、
`envs/wmh`和`envs/t1`。

## 4. 禁网烟雾测试

```bash
envs/core-venv/bin/substain-features verify-offline --smoke-test
```

该命令在`Docker --network none`中检查三环境导入、全部单元测试、GPU、WMH权重、DLMUSE
权重、GenMIND文件和资源哈希，并实际运行一次SynthStrip模型与完整对侧WMH替代烟雾样例。因为迁移包不含原始数据，受试者级流程会明确记录为跳过；验证摘要和日志均保存在`offline/`。

目标工作站的受控入口是`sudo safe_docker.sh`。使用总控脚本运行：

```bash
./run_pipeline.sh offline-smoke --container-command 'sudo safe_docker.sh'
```

也可直接运行CLI：

```bash
envs/core-venv/bin/substain-features verify-offline \
  --config-file config/config.yaml \
  --smoke-test \
  --container-command 'sudo safe_docker.sh'
```

程序将该入口保存为参数数组，不会把含空格的字符串误当作一个命令。若未显式指定，则依次尝试
`sudo safe_docker.sh`、`safe_docker`和`docker`。无论使用哪个入口，烟雾测试都必须保留
`--network none --gpus all`。

本次V1.0交付已于2026年8月24日在独立Docker守护进程中以`--network none --gpus all`
完成验证。当前wheel仅使用包内资源和`PIP_NO_INDEX=1`恢复三套环境，并通过47项测试、CUDA内核、
WMH权重、SynthStrip、对侧替代、DLMUSE和GenMIND检查。权威摘要为
`offline/verification.json`，完整日志为`offline/smoke.log`；摘要记录
`network_used=false`、`network_isolation=docker_--network_none`和`smoke_test_status=pass`。
本机示例的T1/FLAIR ID仍带旧`ses01`后缀，而新MNI病灶ID不带该后缀，因此受试者级烟雾测试按
输入契约记录`subject_smoke_test skipped_no_valid_v1_inputs`，没有将不匹配病例静默拼接。目标工作站
放入符合V1.0 ID契约的正式数据后，应再次运行上述`--smoke-test`，取得该工作站自己的隔离证据。

## 5. 放入正式数据后运行

### 已安装V1.0工作站更新到V1.0.1

GitHub只传输修正源码和小型项目wheel，不替代工作站内已有的模型、环境归档和Docker镜像。工作站执行：

```bash
git clone --branch zhenzong/v1.0.1-safe-docker --depth 1 <私人仓库地址> Substain-v1.0.1-hotfix
cd Substain-v1.0.1-hotfix
./scripts/apply_v1_0_1_hotfix.sh ~/Substain-v2/Substain
```

更新脚本仅覆盖固定的热修复文件，先把旧文件备份到目标项目的`archive/hotfix-v1.0.1-backup-*`，
随后从旧项目已有的离线缓存强制重装项目wheel、重建资源清单并执行完整性检查。它不读取或修改
`BIDS/`、`Lesion/`及既有分析结果。完成后再运行：

```bash
cd ~/Substain-v2/Substain
sudo -v
./run_pipeline.sh offline-smoke --container-command 'sudo safe_docker.sh'
```

在目标项目中放入或挂载输入目录，复制并填写`config/metadata.example.tsv`，再按实际数据更新
`config/config.yaml`的输入模式、根目录和精确后缀。`participants.tsv`由程序自动生成，不手工维护。
先运行：

```bash
./run_pipeline.sh prepare
./run_pipeline.sh audit all
```

审计通过后执行：

```bash
./run_pipeline.sh all all auto 200
```

`all`会完成全部病例特征和四张QC图后退出，不启动人工QC。之后可随时运行
`./run_pipeline.sh qc all`继续可恢复审核；全部审核完成后再运行`export`和`verify`。

DLMUSE资源按非商业科研许可管理。本包仅用于本项目和获得授权的研究工作站，不公开再分发。
