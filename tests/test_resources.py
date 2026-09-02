import os
from pathlib import Path

import pytest

from substain_features import resources


@pytest.mark.skipif(os.name == "nt", reason="Windows未启用开发者模式时不能建立测试符号链接")
def test_build_manifest_excludes_active_and_failed_environments(tmp_path: Path) -> None:
    """正式资源清单不得依赖已解包环境或失败环境中的临时文件。"""

    keep = tmp_path / "envs" / "offline" / "environment_archives.sha256"
    active = tmp_path / "envs" / "core-venv" / "bin" / "python"
    failed = tmp_path / "envs" / "core-venv.failed-py312-test" / "bin" / "python3.12"
    keep.parent.mkdir(parents=True)
    keep.write_text("test\n", encoding="utf-8")

    # venv中的python通常是指向系统解释器的符号链接；排除判断不能跟随该链接。
    system_python = tmp_path / "system" / "python3.12"
    system_python.parent.mkdir(parents=True)
    system_python.write_text("test\n", encoding="utf-8")
    for path in (active, failed):
        path.parent.mkdir(parents=True)
        path.symlink_to(system_python)

    output = tmp_path / "derivatives" / "resource_manifest.tsv"
    table = resources.build_manifest(tmp_path, output)
    paths = set(table["relative_path"].tolist())

    assert "envs/offline/environment_archives.sha256" in paths
    assert "envs/core-venv/bin/python" not in paths
    assert "envs/core-venv.failed-py312-test/bin/python3.12" not in paths


def test_git_head_falls_back_to_loose_reference(tmp_path: Path, monkeypatch: object) -> None:
    repository = tmp_path / "repository"
    reference = repository / ".git" / "refs" / "heads" / "main"
    reference.parent.mkdir(parents=True)
    commit = "1d9c8456168fa8b3a0190c6e91ca3dfbc6c90068"
    (repository / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    reference.write_text(commit + "\n", encoding="utf-8")

    def missing_git(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("git")

    monkeypatch.setattr(resources.subprocess, "run", missing_git)  # type: ignore[attr-defined]
    assert resources.git_head(repository) == commit


def test_git_head_falls_back_to_packed_reference(tmp_path: Path, monkeypatch: object) -> None:
    repository = tmp_path / "repository"
    git_dir = repository / ".git"
    git_dir.mkdir(parents=True)
    commit = "84a977b77243a64f38f2ea0f1423daabe6cfaddd"
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n{} refs/heads/main\n".format(commit),
        encoding="utf-8",
    )

    def missing_git(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("git")

    monkeypatch.setattr(resources.subprocess, "run", missing_git)  # type: ignore[attr-defined]
    assert resources.git_head(repository) == commit
