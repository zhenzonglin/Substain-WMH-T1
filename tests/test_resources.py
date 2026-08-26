from pathlib import Path

from substain_features import resources


def test_build_manifest_excludes_active_and_failed_environments(tmp_path: Path) -> None:
    """正式资源清单不得依赖已解包环境或失败环境中的临时文件。"""

    keep = tmp_path / "envs" / "offline" / "environment_archives.sha256"
    active = tmp_path / "envs" / "core-venv" / "bin" / "python"
    failed = tmp_path / "envs" / "core-venv.failed-py312-test" / "bin" / "python3.12"
    for path in (keep, active, failed):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test\n", encoding="utf-8")

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
