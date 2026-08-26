# WMH–T1 40维结构特征工程 V1.0

本项目从每例T1、FLAIR和FSL MNI152空间急性卒中二值病灶生成20个WMH特征与20个T1灰质萎缩特征。本阶段只生成特征和QC，不运行SuStaIn。

## 运行入口

先在`config/config.yaml`中选择`input.mode: bids`或`folders`，配置影像根目录、病灶根目录、三个文件后缀及`config/metadata.tsv`，然后运行：

首次从GitHub clone时先复制元数据模板；真实`metadata.tsv`和自动生成的`participants.tsv`不会提交到GitHub：

```bash
cp config/metadata.example.tsv config/metadata.tsv
```

```bash
cd /path/to/Substain
./run_pipeline.sh all all auto 200
```

`all`依次执行：

```text
prepare → audit → run（全部病例特征、四张QC图和计算完成表）
```

`all`不会启动人工QC，也不会等待人工判读，因此人工QC不影响其余病例继续提取指标。计算结束后，
`features_computed40.tsv`保存所有计算成功病例。人工QC可在之后任意时间单独启动；它只决定病例是否进入
SuStaIn正式输入`features_primary40.tsv`。

再次执行`./run_pipeline.sh qc`会回到第一例未完成或已过期的病例。全部审核完成后再执行`export`和
`verify`。也可分步运行：

```bash
./run_pipeline.sh prepare
./run_pipeline.sh audit all
./run_pipeline.sh run all auto 200
./run_pipeline.sh qc all
./run_pipeline.sh export all
./run_pipeline.sh verify
./run_pipeline.sh offline
```

`offline`只检查资源完整性。目标工作站需要通过受控Docker入口执行真正禁网烟雾测试时运行：

```bash
./run_pipeline.sh offline-smoke --container-command 'sudo safe_docker.sh'
```

容器入口会作为`sudo`和`safe_docker.sh`两个独立参数传递，不会直连Docker socket；容器仍固定使用
`--network none --gpus all`。其他机器可省略`--container-command`，程序会依次尝试
`sudo safe_docker.sh`、`safe_docker`和`docker`。

默认总任务并发上限为200。`auto`通过`nvidia-smi`检测GPU，并使用进程锁保证每张GPU一次只运行一个WMH-SynthSeg、SynthStrip或DLMUSE任务；没有GPU时自动使用CPU配置。

## 输入契约

### 从临床SAS表生成metadata.tsv

项目根目录的`00_make_metadata_c3.py`固定筛选`D_DIAG=1`且
`H_DEMEN=H_DYSPHR=H_EP=0`，并转换`code_n→participant_id`、
`GENDER 1/2→male/female`、`AGE→age`、`site_code→site_id`。

先在脚本顶部填写SAS7BDAT、T1汇总文件夹、FLAIR汇总文件夹和lesion汇总文件夹4个路径，
然后直接运行，不需要命令行参数：

```bash
envs/core-venv/bin/python 00_make_metadata_c3.py
```

脚本递归扫描三个文件夹中的NIfTI文件，以每个文件名第一个`_`之前的内容作为ID。它只保留
同时满足临床条件并且具有唯一T1、FLAIR和lesion文件的患者，直接生成：

```text
config/metadata.tsv
config/participants.tsv
```

两张表已完成匹配，因此运行后不要再执行`prepare`或`all`，以免旧输入准备步骤覆盖结果。后续从
下面两步开始：

```bash
./run_pipeline.sh audit all
./run_pipeline.sh run all auto 200
```

`site_id`是影像采集中心/医院的稳定代码，当前用于追溯和多中心接口，不进入V1.0特征计算公式。

### BIDS模式

递归读取每例唯一的T1w和FLAIR。同一病例可以没有session或只有一个session；软链接视图会展平为无session结构。发现多个session或同一模态多个文件时直接失败。

### folders模式

递归扫描T1、FLAIR和lesion三个根目录。仅删除文件名末尾配置的精确后缀，剩余字符串作为ID；三个模态和`metadata.tsv`必须严格一一对应。

统一输入视图：

```text
inputs/bids_links/sub-ID/anat/
├── sub-ID_T1w.nii.gz
└── sub-ID_FLAIR.nii.gz
```

原始文件只读。自动生成的`config/participants.tsv`固定为：

```text
participant_id age sex site_id t1w flair lesion_mask
```

病灶必须是FSL MNI152标准1 mm或2 mm网格上的严格0/1掩膜。程序核对shape、affine、qform和sform，不根据文件名猜测空间。旧的个体T1空间病灶不再接受，也不会静默转换。

## 处理链

```text
输入准备和审计
→ T1/FLAIR固定SynthStrip
→ T1–FLAIR刚体配准、T1–ch2better配准
→ MNI152病灶进入ch2better、T1和FLAIR
→ WMH-SynthSeg
→ ch2better世界坐标x=0自动对侧WMH替代
→ WMH20与Chung转换
→ DLMUSE、MUSE macro20与GenMIND技术常模
→ 自动生成四张QC图
→ 成功病例清理可重建中间件
```

上述批量计算链在`features_computed40.tsv`处结束。人工QC是独立的建模纳入步骤；它不会阻断特征提取，
只在后续`export`时筛选SuStaIn正式输入。

WMH校正不使用固定病灶膨胀。对侧供体发生急性病灶冲突或超出可靠脑区时直接扣除；病灶外WMH逐体素不变。所有WMH体积在原生FLAIR网格中按mL计算。

## 四图QC

`substain-features qc`在`127.0.0.1`启动本地网页程序，2×2同时显示：

1. `lesion_on_T1`
2. `lesion_on_FLAIR`
3. `WMH_lesion_overlap`
4. `T1_macro20`

每张图按标准临床放射学方向显示冠状位、矢状位和轴位，不再对已经转为RAS的切片追加90度旋转。三个平面分别选择掩膜体素最多的层面；`WMH_lesion_overlap`优先选择实际重叠体素最多的层面，无重叠时回退到两掩膜联合范围。可多选`t1_invalid、flair_invalid、registration_invalid、wmh_failed、macro_failed`；`qc_pass`与失败项互斥。结果每次选择后立即写入`qc_reviews.sqlite`并同步到`qc_reviews.tsv`。QC图或处理状态变化会使旧判定过期。

## 输出和清理

- 计算完成但未人工审核：`features_computed40.tsv`
- 正式主矩阵：`features_primary40.tsv`，仅包含人工`qc_pass`病例
- 单模态表：`wmh20_*.tsv`、`t1_*raw.tsv`、`t1_macro20_z_genmind.tsv`
- 处理QC：`subject_qc.tsv`
- 人工QC：`qc_reviews.sqlite`、`qc_reviews.tsv`
- 中央图像：`derivatives/substain_features/qc/`

`run`和`all`不会调用`export`。所有病例审核完成前，手动执行的`export`会拒绝生成正式主矩阵。自动处理成功并完成四图生成后，会删除概率图、模板空间临时掩膜、去颅骨强度图、非线性形变场和DLMUSE临时目录；保留最终分割、40维特征、四图、状态、日志、哈希和清理清单。自动处理失败病例保留中间件供诊断。

当前旧示例的病灶仍是个体T1空间，因此V1.0会按新契约拒绝它们；替换为FSL MNI152 1/2 mm二值病灶后再运行。详细函数调用见`docs/PROJECT_CODE_GUIDE_ZH.md`。

## GitHub边界

GitHub私人仓库用于保存V1.0源码、配置模板、测试和小型参考文件。`BIDS/`、`Lesion/`、`archive/`、`derivatives/`、已安装环境、大型模型/常模和受限制第三方源码不提交。GitHub clone不是完整离线运行包；离线工作站仍需使用已授权的项目资源副本。
