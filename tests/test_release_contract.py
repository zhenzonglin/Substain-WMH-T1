import importlib.util
import re
import sys
from pathlib import Path


def _load_script(project_root: Path, name: str):
    path = project_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_versions_and_generic_entrypoints(project_root: Path) -> None:
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    package_init = (project_root / "src" / "substain_features" / "__init__.py").read_text(
        encoding="utf-8"
    )
    config = (project_root / "config" / "config.yaml").read_text(encoding="utf-8")
    assert re.search(r'^version = "1\.1\.0rc1"$', pyproject, re.MULTILINE)
    assert '__version__ = "1.1.0rc1"' in package_init
    assert 'project_version: "1.1.0-rc1"' in config
    for relative in (
        "scripts/start_full_run.sh",
        "scripts/stop_full_run.sh",
        "scripts/monitor_pipeline.py",
    ):
        assert (project_root / relative).is_file()


def test_ws1_hotfix_artifacts_and_old_wheel_are_absent(project_root: Path) -> None:
    removed = (
        "deploy/ws1_v1_0_9_source.patch",
        "scripts/apply_v1_0_9_ws1_hotfix.sh",
        "scripts/prepare_ws1_v1_0_9_rerun.py",
        "scripts/start_ws1_v1_0_9.sh",
        "scripts/stop_ws1_v1_0_8.sh",
        "scripts/monitor_ws1.py",
        "wheels/core/substain_features-1.0.0-py3-none-any.whl",
    )
    assert not [relative for relative in removed if (project_root / relative).exists()]


def test_offline_verifier_uses_clone_directory_and_current_wheel(
    project_root: Path, tmp_path: Path
) -> None:
    verifier = _load_script(project_root, "verify_offline_package.py")
    project_name = "Substain-v1.1.0-rc1"
    project_version = "1.1.0rc1"
    archive = tmp_path / "bundle.tar.gz"
    archive.write_bytes(b"test")
    members = [
        verifier._project_member(project_name, relative)
        for relative in verifier.REQUIRED_RELATIVE_MEMBERS
    ]
    members.append(
        verifier._project_member(
            project_name,
            "wheels/core/substain_features-1.1.0rc1-py3-none-any.whl",
        )
    )
    contents = tmp_path / "contents.txt"
    contents.write_text("\n".join(members) + "\n", encoding="utf-8")
    report = verifier.verify(archive, contents, project_name, project_version)
    assert report["status"] == "pass"
    assert report["project_wheel"].endswith(
        "/wheels/core/substain_features-1.1.0rc1-py3-none-any.whl"
    )

    with contents.open("a", encoding="utf-8") as handle:
        handle.write(project_name + "/derivatives/private-result.tsv\n")
    report = verifier.verify(archive, contents, project_name, project_version)
    assert report["status"] == "fail"
    assert report["forbidden_members"] == [
        project_name + "/derivatives/private-result.tsv"
    ]


def test_generic_start_passes_resource_environment(project_root: Path) -> None:
    start = (project_root / "scripts" / "start_full_run.sh").read_text(encoding="utf-8")
    stop = (project_root / "scripts" / "stop_full_run.sh").read_text(encoding="utf-8")
    assert 'export CPU_THREADS_PER_JOB="${cpu_threads}"' in start
    assert 'export BATCH_SIZE="${batch_size}"' in start
    assert 'pid_file="${root}/logs/full_run.pid"' in start
    assert 'setsid nohup bash "${root}/run_pipeline.sh" run all' in start
    assert 'kill -TERM -- "-${run_pgid}"' in stop
    assert "kill -KILL" not in stop
