import os
import subprocess
from pathlib import Path


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_hotfix_updates_only_fixed_files_and_keeps_backup(project_root: Path, tmp_path: Path) -> None:
    """热修复必须保留原始影像，并在覆盖旧项目文件前建立备份。"""

    target = tmp_path / "existing-substain"
    target.mkdir()
    (target / "pyproject.toml").write_text("old project\n", encoding="utf-8")
    (target / "README.md").write_text("old readme\n", encoding="utf-8")
    (target / "envs" / "offline").mkdir(parents=True)
    (target / "envs" / "offline" / "environment_archives.sha256").write_text("test\n", encoding="utf-8")
    (target / "resources" / "tools").mkdir(parents=True)
    (target / "resources" / "tools" / "offline-smoke-image.tar").write_bytes(b"test")
    (target / "BIDS").mkdir()
    (target / "BIDS" / "source-data-marker.txt").write_text("unchanged\n", encoding="utf-8")

    _write_executable(target / "run_pipeline.sh", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(target / "scripts" / "install_offline.sh", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        target / "envs" / "core-venv" / "bin" / "python",
        "#!/usr/bin/env bash\necho fake_python \"$@\"\nexit 0\n",
    )

    script = project_root / "scripts" / "apply_v1_0_1_hotfix.sh"
    completed = subprocess.run(
        [str(script), str(target)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HOME": str(tmp_path / "home")},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (target / "README.md").read_text(encoding="utf-8") == (
        project_root / "README.md"
    ).read_text(encoding="utf-8")
    assert (target / "BIDS" / "source-data-marker.txt").read_text(encoding="utf-8") == "unchanged\n"
    backups = list((target / "archive").glob("hotfix-v1.0.1-backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "README.md").read_text(encoding="utf-8") == "old readme\n"
    assert "V1.0.1热修复已应用" in completed.stdout
