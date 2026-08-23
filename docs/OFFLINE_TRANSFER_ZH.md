# 离线工作站迁移说明

## 1. 包的边界

`Substain_offline.tar.gz`是项目运行包，不是结果包。它包含代码、配置、测试、模型、
模板、GenMIND/Chung常模、ANTs、固定第三方源码、环境归档、核心wheel、Docker禁网
烟雾测试镜像和许可证。

以下目录明确不进入包内：

- `BIDS/`和`Lesion/`：原始影像与病灶。
- `archive/`：历史版本。
- `derivatives/`：当前两例技术结果。
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
权重、GenMIND文件和资源哈希，并实际运行一次SynthStrip模型与完整对侧WMH替代烟雾样例。因为迁移包不含原始数据，受试者级WMH步骤会明确记录
`subject_smoke_test skipped_no_raw_inputs`；源项目此前的完整一例烟雾证据保留在
`offline/verification.json`与`offline/smoke.log`。

## 5. 放入正式数据后运行

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
