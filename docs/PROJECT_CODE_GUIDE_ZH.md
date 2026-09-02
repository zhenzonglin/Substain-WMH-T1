# WMH–T1 V1.1.0-rc1代码与数据流导读

## 1. 功能分区

```text
Substain/
├── config/                 输入模式、元数据、资源和并行参数
├── inputs/bids_links/      自动生成的无session软链接视图
├── src/substain_features/  Python算法
├── workflow/Snakefile      阶段依赖和资源声明
├── scripts/steps/          00–05分步脚本
├── run_pipeline.sh         总控脚本
├── resources/              固定模型、模板、常模、工具和许可证
├── tests/                  单元及契约测试
└── derivatives/            受试者结果、中央QC和组级表
```

`BIDS/`、`Lesion/`是只读原始输入；`archive/`不参与运行。GitHub版本不包含这三处以及`derivatives/`。

## 2. 总控入口

```bash
./run_pipeline.sh <mode> [participant|all] [auto|gpu|cpu] [total_cores]
```

| 序号 | mode | 脚本 | 作用 |
|---|---|---|---|
| 00 | `prepare` | `00_prepare.sh` | 匹配输入并生成软链接和`participants.tsv` |
| 01 | `audit` | `01_audit.sh` | 审计影像、MNI病灶、工具和固定资源 |
| 02 | `run` | `02_features.sh` | 运行Snakemake计算、四图生成和成功清理 |
| 03 | `qc` | `03_qc.sh` | 打开可恢复四图人工QC |
| 04 | `export` | `04_export.sh` | 仅导出人工QC通过病例 |
| 05 | `verify` | `05_verify.sh` | 验收表格、QC和并行契约 |

`all`只按00–02执行，完成全部病例的特征、四张QC图和`features_computed40.tsv`后退出，不启动GUI，
也不执行正式导出。`qc`、`export`和`verify`是后续独立命令；GUI中断不影响已完成指标，重新执行
`qc`会从上次审核进度继续。

## 3. 各步骤输入、函数与输出

### 00 输入准备

输入：

- BIDS根目录，或T1/FLAIR/lesion三个递归根目录；
- 精确文件后缀；
- `metadata.tsv`：`participant_id、age、sex、site_id`。

调用：

```text
cli.prepare_inputs_command()
└── input_prep.prepare_inputs()
    ├── _scan_bids() / _scan_suffix()
    ├── _require_same_ids()
    └── _replace_symlink()
```

实际行为：

- BIDS模式解析`sub-*`和可选单一`ses-*`；多session阻断。
- folders模式从文件名末尾删除配置后缀，递归寻找ID。
- 重复、缺失、额外ID均阻断，不静默减少队列。
- 原始影像不写入；软链接更新仅允许发生在`inputs/bids_links`。

输出：`inputs/bids_links/`、`input_manifest.tsv`、`config/participants.tsv`。

### 01 审计

调用：

```text
cli.audit_command()
└── audit.run_audit()
    ├── schema.load_participants()
    ├── images.grid_info()
    ├── audit._mni152_grid()
    └── resources.sha256()
```

病灶审计同时检查：严格0/1、有限值、shape、affine、方向、qform、sform，以及与内置FSL MNI152 1 mm或2 mm参考唯一匹配。旧T1空间病灶在此阻断。

输出：`audit_report.json`、`audit_subjects.tsv`。

### 02.1 SynthStrip与配准

调用：

```text
pipeline.stage_registration()
├── synthstrip.run_synthstrip()
└── registration.register_and_warp_atlas()
```

SynthStrip v7.4.1模型v1是固定去颅骨程序：T1脑掩膜用于T1主链，FLAIR脑掩膜用于刚体配准和自动门控；不调用备用工具。随后始终估计T1→FLAIR六自由度刚体变换，并建立T1↔`ch2better`非线性链。配准自动门控失败会阻断病灶和对侧WMH处理。

### 02.2 MNI病灶进入个体空间

调用：`pipeline.stage_lesion()` → `resample_label_to_reference()`和`apply_transforms()`。

```text
FSL MNI152 lesion
→ ch2better（MNI世界坐标、最近邻）
→ T1（T1–模板逆变换、最近邻）
→ FLAIR（T1→FLAIR刚体变换、最近邻）
```

每次重采样后重新二值化并检查非空病灶没有消失。输出记录原始病灶和所有变换SHA256；不做2 mm膨胀。

### 02.3 WMH-SynthSeg

调用：`pipeline.stage_wmh_segmentation()` → `wmh.run_wmh_synthseg()`。

输入：原生FLAIR、固定模型和隔离WMH环境。该节点在病灶空间链确认后运行。模型硬分割恢复到原生FLAIR网格；概率图只是可删除中间件。输出原始硬分割和阶段状态。

### 02.4 自动对侧WMH替代与WMH20

调用：

```text
pipeline.stage_wmh()
├── symmetry.run_contralateral_replacement()
│   ├── label_lesion_components()
│   ├── mirror_world_x_zero()
│   └── compose_native_wmh()
├── wmh.extract_wmh20_ml()
└── wmh.chung_zscore()
```

26邻域区分病灶成分。只有与原始WMH重叠的成分触发替代。对侧位置由`ch2better`世界坐标`x=0`反射得到；对侧急性病灶冲突、跨中线自身冲突或脑外供体均扣除。病灶外WMH保持不变。

WMH20按原生FLAIR体素体积换算mL，固定顺序为基底节、额叶、枕叶、颞叶、顶叶，各1–4层。Chung转换为：

```text
z = (volume_ml - residual_mean) / residual_sd
```

公开公式未使用年龄，因此`age_adjustment_applied=false`。

### 02.5 T1 145/119/20区

调用：

```text
pipeline.stage_t1()
├── t1.run_nichart_dlmuse()
├── t1.extract_t1_features()
├── mapping.aggregate_macro20()
├── mapping.assert_volume_conservation()
└── normative.GenMINDGlobalV1Provider.transform()
```

DLICV/DLMUSE产生最终分割。145个原生ROI中保留119个灰质ROI，再按`muse_macro20_v1_provisional`求和到20个互斥宏区。聚合先求原始体积总和，不平均ROI z分数。技术常模计算`log(宏区体积/非脑室MUSE组织总体积)`并输出体积越小越高的萎缩z分数。

### 02.6 自动生成四图与清理

`pipeline.stage_qc()`只调用`save_overlay()`和`save_dual_overlay()`生成四张中央图。这个自动节点只表示
QC材料生成成功，不代表人工判读通过。`images._orthogonal_qc_views()`保持物理比例，并按标准临床
放射学方向直接显示冠状、矢状和轴位，不对RAS标准化后的切片追加旋转。

`pipeline.stage_cleanup()`只在四图成功且所有计算节点通过后执行。它逐个验证清理目标位于当前受试者衍生目录，保留最终分割/特征/日志，删除可重建影像和DLMUSE临时目录，并写`cleanup_manifest.json`。失败病例由`error_records_only_v1`删除影像中间件，只保留`status/`与`logs/`。

### 03 人工QC

调用：`cli.qc_command()` → `qc_review.serve_qc()`。

Python标准库`ThreadingHTTPServer`只绑定`127.0.0.1`。浏览器2×2显示四图。SQLite表是权威状态；TSV是同步副本。每次保存使用事务，支持上一例、下一例、跳转、修改和中断恢复。图像或处理状态哈希变化后，旧判定变为`stale`。

该步骤不属于特征计算的依赖链。可以先完成全部病例的`features_computed40.tsv`，之后分批审核；人工
判定只在04导出时决定哪些病例进入SuStaIn正式输入。

四图统一使用RAS标准化后的临床放射学方向：冠状位和轴位图像左侧为患者右侧，矢状位图像左侧为前方，所有平面上方为解剖学上方/前方。三个正交面分别选择叠加掩膜体素最多的层面；WMH–病灶图优先按实际重叠选择，无重叠时使用联合掩膜。选择规则写入每例`status/qc.json`。

### 04 导出

`pipeline.aggregate_outputs()`先生成`features_computed40.tsv`和全部单模态表，不生成正式主矩阵。

`pipeline.export_reviewed_outputs()`要求每例状态均为`pass`或`fail`，然后：

- 只把人工`pass`且确有完整40维的病例写入`features_primary40.tsv`；
- QC通过但缺少40维时阻断；
- 把人工判定、失败原因、备注、时间和哈希合并到`subject_qc.tsv`。

## 4. GPU与200任务并发

`run_pipeline.sh`的默认波次与GPU策略为：

```text
MAX_PARALLEL_JOBS=200
GPU_POLICY=auto_one_job_per_device
```

`cli.run_command()`调用`gpu_pool.detect_gpu_ids()`读取GPU设备编号。V1.1.0-rc1的全量GPU入口要求`CUDA_VISIBLE_DEVICES`只指定一张卡，并由`gpu_pool.run_with_gpu_lock()`使用`flock`保证同一时刻只运行一个GPU任务；本RC不纳入工作站2的双GPU专用调度。CPU/轻量规则仍可在总核心池内并行。`CPU_THREADS_PER_JOB`会作为`cpu_threads_per_job`显式传入Snakefile，保证日志、线程环境和Snakemake资源一致。没有GPU时`auto`切到CPU；显式`gpu`但未检测到设备时立即失败。

## 5. 关键表格

| 表格 | 含义 |
|---|---|
| `wmh20_raw.tsv` / `wmh20_z_chung.tsv` | WMH原始mL和20维z分数 |
| `t1_muse145_raw.tsv` / `t1_gm119_raw.tsv` | MUSE原生145和灰质119体积 |
| `t1_macro20_raw.tsv` / `t1_macro20_z_genmind.tsv` | T1宏区体积和萎缩z |
| `features_computed40.tsv` | 计算完整但未人工QC的暂存矩阵 |
| `features_primary40.tsv` | 仅人工QC通过的正式矩阵 |
| `subject_qc.tsv` | 自动节点、对侧替代指标和人工判定 |
| `qc_reviews.tsv` | 可读人工审核记录 |
| `resource_manifest.tsv` | 固定资源SHA256 |

V1.1.0-rc1保持40维特征定义不变，仅整合物理网格容差、运行时间记录、失败保留策略、通用监测和可移植调度入口。
