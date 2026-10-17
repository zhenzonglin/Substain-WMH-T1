# LS 健康队列：从 GitHub 克隆并安装

此目录从工作站当前 `/data02/linzhenzong/Substain_GB` 的配置与源码生成 LS 专用运行副本。GB 的源码、配置、名单、日志和结果不改写；模型、模板、Python 环境继续复用。全部真实影像计算在工作站进行。

## 安装

在工作站执行以下四行。代码和 Excel 读取依赖都通过 GitHub 获取，不需要访问 PyPI，也不需要传输压缩包或粘贴长脚本。

```bash
cd /data02/linzhenzong
git clone --depth 1 --single-branch --branch codex/ls-cohort-install https://github.com/zhenzonglin/Substain-WMH-T1.git Substain_LS_installer
cd Substain_LS_installer
bash cohorts/LS/install_LS.sh
```

先出现 `LS_CHECKOUT_VERIFIED` 表示源码摘要通过；出现 `LS_INSTALLED` 才表示安装完成。安装读取原配置、检查当前 Python/CUDA 环境并记录资源摘要，不安装或升级原分析环境。Excel依赖 openpyxl 3.1.5、et_xmlfile 2.0.0 的源码在 `vendor/` 中，原许可保留于对应 dist-info 目录。

如果已有 GB/LS 分析进程、未知源码结构或既有 LS 文件被修改，脚本会停止并报告具体原因，不终止分析、不覆盖人工修改。对相同版本可重复安装。已有 LS 分析运行后，不要直接换版本覆盖；安装器会拒绝不一致的运行代码。仅四个运行 Python 文件和 vendor 依赖进入工作站运行目录；仓库文档、测试及测试夹具不会混入运行指纹。

已有安装因缺少 T1/FLAIR 或误识别 `PRECISE_Thickness_GMV_QC_results` 目录停在准备阶段时，使用这个轻量更新，只替换输入适配器一个文件（兼容已应用缺模态补丁的版本）：

```bash
cd /data02/linzhenzong/Substain_LS_installer
git pull --ff-only &&
bash cohorts/LS/apply_missing_modalities_LS.sh &&
bash /data02/linzhenzong/Substain_GB/scripts/run_LS.sh
```

更新器先备份旧脚本和失败报告，不重装环境、模型或配置。它仅适用于后台已退出、尚无参与者名单、冻结记录和阶段结果的准备失败情况；其他情况保留原状态并说明原因。

## 首次运行

```bash
bash /data02/linzhenzong/Substain_GB/scripts/run_LS.sh
```

命令返回 `LS_STARTED PID=... log=...` 后，准备、审计、单例验证和全队列在后台依次执行。PID 已启动不等于验收通过。任何前置步骤失败，都不会进入全队列。

查看主日志：

```bash
tail -n 60 -f "$(cat /data02/linzhenzong/Substain_GB/logs/LS/full_run_LS.logpath)"
```

首次准备检查 Excel、影像匹配、SHA256 和常模资格。随后首次审计全部输入，验证空间和资源；这段可能耗时。优先验证合格的 P0001，若不存在则选排序后的第一名合格被试。单例必须八阶段通过、完整40维且健康 WMH 一致性检查通过，才接续全队列。

只准备、不开始影像计算：

```bash
bash /data02/linzhenzong/Substain_GB/scripts/prepare_LS.sh
```

## 独立监测与续跑

已经启动分析、需要更新监测器时，在 Git 克隆目录运行下列命令。新版监测器直接读取已安装的 LS 配置，不改写安装目录、运行指纹或分析进程：

```bash
cd /data02/linzhenzong/Substain_LS_installer
git pull --ff-only
bash cohorts/LS/monitor_LS.sh --interval 30
```

首次准备需要读取 Excel、匹配影像并完成输入 SHA256，`participants_LS.tsv` 在验证完成后才发布。名单尚未生成时，新版显示“名册未就绪”、可核实的进程状态及日志，而不会虚报0名被试或直接退出；名单出现后自动切换八阶段统计。`D` 通常表示等待 I/O，不能仅凭一次采样认定卡死。

安装目录中的监测入口适用于已安装的版本：

一次快照：

```bash
bash /data02/linzhenzong/Substain_GB/scripts/monitor_LS.sh --once
```

每30秒刷新：

```bash
bash /data02/linzhenzong/Substain_GB/scripts/monitor_LS.sh --interval 30
```

监测覆盖全部八阶段，并单独显示无法读取/损坏状态。`unknown` 不计为等待。Ctrl+C只退出监测。

单例已通过后的续跑：

```bash
bash /data02/linzhenzong/Substain_GB/scripts/resume_LS.sh
```

若首次流程在单例通过前中断，继续使用 `run_LS.sh`。输入、配置和代码指纹不变时复用准备结果与已通过审计，不重新推理已完成的病例。源影像、目录候选、Excel、配置或分析代码变化会停止复用，并报告差异；不要删除状态文件绕过检查。

正常阶段失败遵循原队列的 fail/cleanup 机制，保留失败记录并继续其他病例。此前 GB 的外置 T1 看守器不会被本包自动接管或改写；它原有 GB 进程筛选规则不能直接当作 LS 看守器使用。

## 输入与方法

输入根目录为 `/data02/linzhenzong/LS/`，临床表为 `LS_INFO.xlsx`。首行必须唯一包含 BIGDATA_CODE、AGE、SEX；多个匹配工作表不自动猜选。

| 字段 | 规则 |
|---|---|
| BIGDATA_CODE | 与目录 P0001/P0005 等精确匹配，保留前导零，只去除首尾空白 |
| AGE | 按岁读取，保留小数；不取整、不插补 |
| SEX | 1=male，2=female；其他编码报错 |
| site_id | 固定 LS |
| T1 | 受试者目录内唯一 `*T1w*.nii.gz` |
| FLAIR | 受试者目录内唯一 `*T2wFLAIR*.nii.gz` |

影像目录构成待核对名册。缺少 T1 或 FLAIR 者记录后跳过，只有完整配对者进入 `participants_LS.tsv`、metadata 和常模适用性表。排除记录位于 `/data02/linzhenzong/Substain_GB/inputs/cohort_LS/excluded_subjects_LS.tsv`；一人缺两种影像只记一人。`input_report_LS.json` 分别记录全部影像目录人数、实际纳入人数和缺模态排除人数。全部缺模态时会报告无可分析者，不发布空名单。

按用户指定，直接子目录 `PRECISE_Thickness_GMV_QC_results` 不作为被试扫描，内容保持原样，也不计入纳入或缺模态排除人数。跳过记录写入 `input_report_LS.json` 和 `input_issues_LS.tsv`，代码为 `ignored_non_subject_directory`。仅排除这一确切目录名，其他非法 P 开头目录仍报告错误。

临床表额外病例仍保留在完整临床表及双侧匹配报告。重复ID、非法临床字段、损坏影像或多个影像候选仍阻止准备，不自动选择或额外排除。监测器、阶段结果和 `not_exported_LS.tsv` 以实际纳入名单为分母，准备期缺模态者另见上述排除表。

临床原列、公式及其缓存值分别保留，公式缺少缓存值时需由原数据负责人保存正确的 Excel 后再准备。

所有人均按用户确认的健康、无急性病灶处理。全零 MNI mask 只是原接口适配物，有模板/文件 SHA256 和 `synthetic_zero_mask` 来源记录，不代表做过病灶分割。原 WMH 分割保留，最终逐体素核对校正前后 WMH 一致、病灶成分与修正量为0。QC中的空病灶叠加不作为病灶配准准确性的证据；其余配准与人工QC门槛仍沿用原流程。

常模沿用原技术配置：GenMIND 22–90岁、同性交叉年龄窗与原参考人数门槛。不适用者仍在参与者名册和资格报告中，原阶段会记录失败，不输出完整40维；不会外推或填零。当前 T1 常模仍是原项目的临时技术常模，本包不改变其适用性声明。

## 输出与人工QC

输出根目录严格为：

```text
/mnt/share/GPUStore2/usersdir/linzhenzong/Sustain_LS/derivatives
```

主要总表位于其 `tables/`：

- `features_computed40_LS.tsv`：计算完成的 WMH20＋T1 20，尚未替代人工QC。
- `wmh20_raw_LS.tsv`、`wmh20_z_chung_LS.tsv`。
- `t1_macro20_raw_LS.tsv`、`t1_macro20_z_genmind_LS.tsv`。
- `t1_muse145_raw_LS.tsv`、`t1_gm119_raw_LS.tsv`。
- `subject_qc_LS.tsv`、`run_summary_LS.json`、`run_summary_LS.tsv`、`not_exported_LS.tsv`。

总表在队列收尾时生成；单例完成期间先保存各自特征和状态。`cleanup`结束数不等于成功人数。`run_summary_LS.json` 的 `success` 为八阶段全部通过人数，`computed40_rows` 为按原聚合规则可导出的完整40维人数。仅清理失败但前七阶段通过者仍保留特征，在 `computed_with_cleanup_issue` 和逐人汇总中单独记录。

最终验证：

```bash
bash /data02/linzhenzong/Substain_GB/scripts/verify_LS.sh
```

人工QC使用独立数据库和8766端口；计算结束后启动：

```bash
bash /data02/linzhenzong/Substain_GB/scripts/qc_LS.sh
```

保留原流程的人工决定，完成审核后才执行：

```bash
bash /data02/linzhenzong/Substain_GB/scripts/export_LS.sh
```

该命令沿用原有人工QC规则生成 `features_primary40_LS.tsv`。对已经全部完成的队列重复续跑不会删除已有正式导出；确需重新聚合时，旧主矩阵先保存于 `exports_LS/` 下。

## 隔离布局与资源

框架内新增 `config/config_LS.yaml`、`config/participants_LS.tsv`、`inputs/cohort_LS/`、`scripts/cohort_LS/`、`runtime_LS/` 和 `logs/LS/`。单例内部状态继续使用 `sub-P0001/status/t1.json` 等既有名称，全部位于 LS derivatives 内。

LS CPU核数、每任务线程、GPU列表/槽位、显存门槛、滚动窗口和计算任务上限来自工作站当时配置。不会恢复本地旧版的50核或其他旧参数。物理GPU锁保留原绝对路径和协议，避免同卡超额占用；GB活跃时不会再启动LS全队列。

本地验收包含合成输入、源码隔离、真实 Snakemake dry-run 与主控模拟；不包含工作站真实影像推理。`validation_LS.json` 记录本次 GitHub 版本的测试结果。测试所用最小历史源码夹具位于 `tests/fixtures/`，不含模型、影像、临床数据或环境归档。

开发者可在已具备 pytest、numpy、pandas、nibabel、PyYAML、Snakemake 7.32.4 的 Linux 环境运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider cohorts/LS/tests
```

工作站安装和正式分析不要求执行测试。纯源码校验命令为 `python3 cohorts/LS/verify_checkout_LS.py`；维护者修改源码后使用 `python3 cohorts/LS/maintenance/update_checksums_LS.py` 更新清单。普通使用者不应重新生成清单来绕过内容不一致。
