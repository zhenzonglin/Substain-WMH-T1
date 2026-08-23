from pathlib import Path

from substain_features import resources


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
