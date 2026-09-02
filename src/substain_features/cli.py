"""`substain-features` 命令行入口。"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Mapping, Optional

import click

from . import __version__
from .audit import run_audit
from .gpu_pool import detect_gpu_ids
from .input_prep import prepare_inputs
from .mapping import write_macro_mapping
from .pipeline import (
    aggregate_outputs,
    export_reviewed_outputs,
    qc_dir,
    stage_cleanup,
    stage_lesion,
    stage_lowres,
    stage_qc,
    stage_registration,
    stage_skullstrip,
    stage_t1,
    stage_wmh,
    stage_wmh_segmentation,
    status_path,
)
from .qc_review import serve_qc
from .resources import build_manifest, verify_manifest
from .schema import Participant, load_config, load_participants, select_participants
from .status import guarded_stage


DEFAULT_CONFIG = Path("config/config.yaml")


def _context(config_path: Path) -> tuple:
    config = load_config(config_path.resolve())
    root = Path(str(config["project_root"]))
    participant_path = Path(str(config["participants"]))
    participant_path = participant_path if participant_path.is_absolute() else root / participant_path
    participants = load_participants(participant_path, root)
    return config, root, participants


def _derivatives_root(config: Mapping[str, object], root: Path) -> Path:
    """解析当前配置的衍生目录，禁止阶段命令回退到默认输出路径。"""

    derivatives = Path(str(config["derivatives"]))
    return derivatives if derivatives.is_absolute() else root / derivatives


@click.group()
@click.version_option(version=__version__, prog_name="substain-features")
def main() -> None:
    """构建 WMH 20 + T1 20 结构特征，并保存完整 QC。"""


@main.command("prepare-inputs")
@click.option("--config-file", type=click.Path(path_type=Path), default=DEFAULT_CONFIG, show_default=True)
def prepare_inputs_command(config_file: Path) -> None:
    """递归匹配输入并生成无session的BIDS软链接视图与participants.tsv。"""

    config = load_config(config_file.resolve())
    try:
        details = prepare_inputs(config)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(details, ensure_ascii=False, indent=2))


@main.command("audit")
@click.option("--config-file", type=click.Path(path_type=Path), default=DEFAULT_CONFIG, show_default=True)
@click.option("--participant-id", default="all", show_default=True)
@click.option("--allow-fail", is_flag=True, help="仍写报告但不以非零状态退出，供 Snakemake 状态汇总使用。")
def audit_command(config_file: Path, participant_id: str, allow_fail: bool) -> None:
    """预检输入、世界坐标网格、元数据、工具和固定资源。"""

    config, root, participants = _context(config_file)
    selected = select_participants(participants, participant_id)
    output = Path(str(config["derivatives"]))
    output = output if output.is_absolute() else root / output
    report = run_audit(root, selected, output / "audit")
    click.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "pass" and not allow_fail:
        raise click.ClickException("audit 未通过；详见 derivatives/substain_features/audit/audit_report.json")


@main.command("run")
@click.option("--config-file", type=click.Path(path_type=Path), default=DEFAULT_CONFIG, show_default=True)
@click.option("--participant-id", default="all", show_default=True)
@click.option("--profile", type=click.Choice(["auto", "gpu", "cpu"]), default="auto", show_default=True)
@click.option("--cores", type=click.IntRange(min=1), default=96, show_default=True)
@click.option("--target", default="all", show_default=True, help="Snakemake 目标规则。")
@click.option("--dry-run", is_flag=True)
@click.option("--skip-prepare", is_flag=True, help="仅供总控脚本或测试在已执行prepare-inputs后使用。")
def run_command(
    config_file: Path,
    participant_id: str,
    profile: str,
    cores: int,
    target: str,
    dry_run: bool,
    skip_prepare: bool,
) -> None:
    """运行单例或批量Snakemake；自动检测GPU并确保每张卡一次一个任务。"""

    config_file = config_file.resolve()
    initial_config = load_config(config_file)
    if not skip_prepare:
        prepare_inputs(initial_config)
    config, root, participants = _context(config_file)
    select_participants(participants, participant_id)
    configured_threads = config.get("execution", {}).get("cpu_threads_per_job", 8)
    raw_threads = os.environ.get("CPU_THREADS_PER_JOB", str(configured_threads)).strip()
    try:
        cpu_threads_per_job = int(raw_threads)
    except ValueError as exc:
        raise click.ClickException(
            "CPU_THREADS_PER_JOB必须为正整数，收到: {}".format(raw_threads)
        ) from exc
    if cpu_threads_per_job < 1:
        raise click.ClickException(
            "CPU_THREADS_PER_JOB必须为正整数，收到: {}".format(raw_threads)
        )
    # 为两个上游SynthStrip保留CPU空间；T1最多同时两例；其余重型CPU槽
    # 优先用于registration -> lesion -> WMH特征，避免DLICV铺满全部核心。
    skullstrip_slots = 2 if cores >= 3 * cpu_threads_per_job else 1
    t1_slots = 2 if cores >= 2 * cpu_threads_per_job else 1
    reserved_upstream_threads = skullstrip_slots * cpu_threads_per_job
    finish_cpu_slots = max(
        1,
        (cores - reserved_upstream_threads) // cpu_threads_per_job,
    )
    gpu_ids = detect_gpu_ids()
    selected_profile = profile
    if profile == "auto":
        selected_profile = "gpu" if gpu_ids else "cpu"
    if selected_profile == "gpu" and not gpu_ids:
        raise click.ClickException("请求GPU配置，但nvidia-smi未检测到可用设备")
    # 迁移后的虚拟环境中，console script 的shebang可能仍指向打包机器路径。
    # 直接使用当前核心Python加载模块，避免依赖不可迁移的bin/snakemake入口。
    try:
        import snakemake  # noqa: F401
    except ImportError as exc:
        raise click.ClickException("核心Python缺少Snakemake；请重新运行 scripts/install_offline.sh") from exc
    command = [
        sys.executable,
        "-m",
        "snakemake",
        "--snakefile",
        str(root / "workflow" / "Snakefile"),
        "--configfile",
        str(config_file.resolve()),
        "--cores",
        str(cores),
        "--keep-going",
        "--printshellcmds",
    ]
    # Snakemake 的 --resources/--config 都会持续吞入后续非选项参数；目标必须放在它们之前。
    if target != "all":
        command.append(target)
    if dry_run:
        command.append("--dry-run")
    command.extend(
        [
            "--resources",
            "gpu={}".format(len(gpu_ids) if selected_profile == "gpu" else 0),
            "finish_cpu={}".format(finish_cpu_slots),
            "skullstrip_cpu={}".format(skullstrip_slots),
            "t1_cpu={}".format(t1_slots),
            "--config",
            "active_config_file={}".format(config_file),
            "selected_participant={}".format(participant_id),
            "profile={}".format(selected_profile),
            "gpu_devices={}".format(",".join(gpu_ids)),
            "cpu_threads_per_job={}".format(cpu_threads_per_job),
        ]
    )
    completed = subprocess.run(command, cwd=str(root), check=False)
    if completed.returncode != 0:
        raise click.ClickException("Snakemake 失败，exit={}".format(completed.returncode))


@main.command("qc")
@click.option("--config-file", type=click.Path(path_type=Path), default=DEFAULT_CONFIG, show_default=True)
@click.option("--participant-id", default="all", show_default=True)
@click.option("--port", type=click.IntRange(min=1024, max=65535), default=8765, show_default=True)
@click.option("--no-browser", is_flag=True, help="不自动打开浏览器，只打印本机URL。")
def qc_command(config_file: Path, participant_id: str, port: int, no_browser: bool) -> None:
    """启动可中断、可恢复的四图人工QC本地网页程序。"""

    config, root, participants = _context(config_file)
    selected = select_participants(participants, participant_id)
    derivatives = Path(str(config["derivatives"]))
    derivatives = derivatives if derivatives.is_absolute() else root / derivatives
    serve_qc(selected, qc_dir(config), derivatives, port=port, open_browser=not no_browser)


@main.command("export")
@click.option("--config-file", type=click.Path(path_type=Path), default=DEFAULT_CONFIG, show_default=True)
@click.option("--participant-id", default="all", show_default=True)
def export_command(config_file: Path, participant_id: str) -> None:
    """在全部人工QC完成后生成正式features_primary40.tsv。"""

    config, _, participants = _context(config_file)
    selected = select_participants(participants, participant_id)
    try:
        details = export_reviewed_outputs(config, selected)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(details, ensure_ascii=False, indent=2))


def shutil_which(name: str) -> Optional[str]:
    # 独立函数便于单元测试替换 PATH。
    import shutil

    return shutil.which(name)


def _resolve_docker_command(configured: Optional[str] = None) -> List[str]:
    """把容器入口解析为参数数组，支持 ``sudo safe_docker.sh``。"""

    command_text = configured
    if command_text is None:
        command_text = os.environ.get("SUBSTAIN_DOCKER_COMMAND", "").strip()
    if command_text:
        try:
            command = shlex.split(command_text)
        except ValueError as exc:
            raise ValueError("Docker入口参数无法解析: {}".format(exc)) from exc
        if not command:
            raise ValueError("Docker入口不能为空")
        executable = shutil_which(command[0])
        if executable is None:
            raise FileNotFoundError("找不到Docker入口的首个命令: {}".format(command[0]))
        command[0] = executable
        return command

    # 若受控脚本在普通用户PATH中可见，按工作站要求通过sudo执行。
    safe_script = shutil_which("safe_docker.sh")
    sudo = shutil_which("sudo")
    if safe_script and sudo:
        return [sudo, safe_script]
    safe_wrapper = shutil_which("safe_docker")
    if safe_wrapper:
        return [safe_wrapper]
    docker = shutil_which("docker")
    if docker:
        return [docker]
    raise FileNotFoundError("找不到sudo safe_docker.sh、safe_docker或docker入口")


@main.command("verify-offline")
@click.option("--config-file", type=click.Path(path_type=Path), default=DEFAULT_CONFIG, show_default=True)
@click.option("--smoke-test", is_flag=True, help="额外运行一例核心烟雾测试；不会联网。")
@click.option(
    "--container-command",
    default=None,
    help="容器入口命令；工作站使用引号包围的 'sudo safe_docker.sh'。",
)
def verify_offline_command(config_file: Path, smoke_test: bool, container_command: Optional[str]) -> None:
    """不发出网络请求，核对 SHA256、源码提交、wheel 缓存和可选烟雾测试。"""

    # 迁移包不携带真实metadata/participants；离线资源核验不应要求受试者清单存在。
    config = load_config(config_file.resolve())
    root = Path(str(config["project_root"]))
    manifest = _derivatives_root(config, root) / "tables" / "resource_manifest.tsv"
    if not manifest.is_file():
        build_manifest(root, manifest)
    verification_root = Path(
        os.environ.get(
            "SUBSTAIN_OFFLINE_VERIFY_ROOT",
            str(root / "offline"),
        )
    )
    report = verify_manifest(root, manifest)
    report["smoke_test_requested"] = smoke_test
    if smoke_test:
        script = root / "scripts" / "offline_smoke.sh"
        try:
            docker_command = _resolve_docker_command(container_command)
        except (FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        completed = subprocess.run(
            [str(script), "--", *docker_command],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        report["smoke_test_status"] = "pass" if completed.returncode == 0 else "fail"
        report["smoke_test_exit_code"] = completed.returncode
        report["smoke_test_stdout"] = completed.stdout[-4000:]
        report["smoke_test_stderr"] = completed.stderr[-4000:]
        command_label = shlex.join(
            [Path(value).name if index < 2 and value.startswith("/") else value for index, value in enumerate(docker_command)]
        )
        report["container_command"] = command_label
        report["network_isolation"] = "{}_--network_none".format(command_label.replace(" ", "_"))
        report["rebuilt_environment_root"] = str(verification_root / "envs")
        if report["smoke_test_status"] != "pass":
            report["status"] = "fail"
    # 快速完整性检查不得覆盖已经通过的禁网烟雾证据。
    output = verification_root / ("verification.json" if smoke_test else "integrity.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    click.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise click.ClickException("离线完整性验证失败；见 {}".format(output))


@main.command("build-mapping", hidden=True)
@click.option("--config-file", type=click.Path(path_type=Path), default=DEFAULT_CONFIG)
def build_mapping_command(config_file: Path) -> None:
    config, root, _ = _context(config_file)
    t1_config = config["t1"]
    native = root / str(t1_config["native_mapping"])
    derived = root / str(t1_config["derived_mapping"])
    output = root / str(t1_config["macro_mapping"])
    click.echo(json.dumps(write_macro_mapping(native, derived, output), ensure_ascii=False, indent=2))


@main.group("stage", hidden=True)
def stage_group() -> None:
    """Snakemake 调用的内部阶段命令。"""


def _single(config_file: Path, participant_id: str) -> tuple:
    config, root, participants = _context(config_file)
    selected = select_participants(participants, participant_id)
    if len(selected) != 1:
        raise click.ClickException("内部 stage 必须指定一个 participant_id")
    return config, root, selected[0]


def _run_guarded(config: Mapping[str, object], participant: Participant, stage: str, function: object) -> None:
    path = status_path(config, participant, stage)
    ok = guarded_stage(path, stage, participant.participant_id, function)  # type: ignore[arg-type]
    click.echo("{} {} {}".format(participant.participant_id, stage, "pass" if ok else "fail"))
    if not ok:
        raise click.ClickException("{} {} 失败；见 {}".format(participant.participant_id, stage, path))


@stage_group.command("lesion")
@click.option("--config-file", type=click.Path(path_type=Path), required=True)
@click.option("--participant-id", required=True)
def stage_lesion_command(config_file: Path, participant_id: str) -> None:
    config, _, participant = _single(config_file, participant_id)
    _run_guarded(config, participant, "lesion", lambda: stage_lesion(config, participant))


@stage_group.command("registration")
@click.option("--config-file", type=click.Path(path_type=Path), required=True)
@click.option("--participant-id", required=True)
def stage_registration_command(config_file: Path, participant_id: str) -> None:
    config, _, participant = _single(config_file, participant_id)
    _run_guarded(config, participant, "registration", lambda: stage_registration(config, participant))


@stage_group.command("skullstrip")
@click.option("--config-file", type=click.Path(path_type=Path), required=True)
@click.option("--participant-id", required=True)
@click.option("--profile", type=click.Choice(["gpu", "cpu"]), required=True)
def stage_skullstrip_command(config_file: Path, participant_id: str, profile: str) -> None:
    config, _, participant = _single(config_file, participant_id)
    _run_guarded(config, participant, "skullstrip", lambda: stage_skullstrip(config, participant, profile))


@stage_group.command("wmh-seg")
@click.option("--config-file", type=click.Path(path_type=Path), required=True)
@click.option("--participant-id", required=True)
@click.option("--profile", type=click.Choice(["gpu", "cpu"]), required=True)
def stage_wmh_segmentation_command(config_file: Path, participant_id: str, profile: str) -> None:
    config, _, participant = _single(config_file, participant_id)
    _run_guarded(config, participant, "wmh_seg", lambda: stage_wmh_segmentation(config, participant, profile))


@stage_group.command("wmh")
@click.option("--config-file", type=click.Path(path_type=Path), required=True)
@click.option("--participant-id", required=True)
@click.option("--profile", type=click.Choice(["gpu", "cpu"]), required=True)
def stage_wmh_command(config_file: Path, participant_id: str, profile: str) -> None:
    config, _, participant = _single(config_file, participant_id)
    _run_guarded(config, participant, "wmh", lambda: stage_wmh(config, participant, profile))


@stage_group.command("t1")
@click.option("--config-file", type=click.Path(path_type=Path), required=True)
@click.option("--participant-id", required=True)
@click.option("--profile", type=click.Choice(["gpu", "cpu"]), required=True)
def stage_t1_command(config_file: Path, participant_id: str, profile: str) -> None:
    config, _, participant = _single(config_file, participant_id)
    _run_guarded(config, participant, "t1", lambda: stage_t1(config, participant, profile))


@stage_group.command("qc")
@click.option("--config-file", type=click.Path(path_type=Path), required=True)
@click.option("--participant-id", required=True)
def stage_qc_command(config_file: Path, participant_id: str) -> None:
    config, _, participant = _single(config_file, participant_id)
    _run_guarded(config, participant, "qc", lambda: stage_qc(config, participant))


@stage_group.command("cleanup")
@click.option("--config-file", type=click.Path(path_type=Path), required=True)
@click.option("--participant-id", required=True)
def stage_cleanup_command(config_file: Path, participant_id: str) -> None:
    config, _, participant = _single(config_file, participant_id)
    _run_guarded(config, participant, "cleanup", lambda: stage_cleanup(config, participant))


@stage_group.command("lowres")
@click.option("--config-file", type=click.Path(path_type=Path), required=True)
@click.option("--participant-id", required=True)
@click.option("--profile", type=click.Choice(["gpu", "cpu"]), required=True)
def stage_lowres_command(config_file: Path, participant_id: str, profile: str) -> None:
    config, _, participant = _single(config_file, participant_id)
    _run_guarded(config, participant, "lowres", lambda: stage_lowres(config, participant, profile))


@stage_group.command("aggregate")
@click.option("--config-file", type=click.Path(path_type=Path), required=True)
@click.option("--participant-id", default="all")
def stage_aggregate_command(config_file: Path, participant_id: str) -> None:
    config, root, participants = _context(config_file)
    selected = select_participants(participants, participant_id)
    details = aggregate_outputs(config, selected)
    manifest_path = _derivatives_root(config, root) / "tables" / "resource_manifest.tsv"
    build_manifest(root, manifest_path)
    click.echo(json.dumps(details, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
