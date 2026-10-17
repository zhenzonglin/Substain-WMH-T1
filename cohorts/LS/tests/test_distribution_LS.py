"""Git distribution must install only stable runtime code and dependencies."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runner_LS as runner


def test_runtime_sources_ignore_documentation_reports_and_repository_metadata(tmp_path, monkeypatch):
    required = {"inputs_LS.py", "monitor_LS.py", "overlay_LS.py", "runner_LS.py"}
    vendor = {"vendor/openpyxl/__init__.py", "vendor/openpyxl.dist-info/LICENCE.rst"}
    ignored = {
        "install_LS.sh", "README.md", "README_LS.md", "SHA256SUMS_LS",
        "validation_LS.json", "test_results_LS.xml", "tests/test_example.py",
        ".git/config", ".pytest_cache/cache", "__pycache__/runner_LS.cpython-38.pyc",
        "vendor/openpyxl/__pycache__/__init__.cpython-38.pyc",
    }
    for relative in required | vendor | ignored:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n")
    monkeypatch.setattr(runner, "HERE", tmp_path)
    sources = runner.runtime_sources()
    assert all(isinstance(path, Path) for path in sources)
    actual = {path.relative_to(tmp_path).as_posix() for path in sources}
    assert actual == required | vendor
    before = {str(path): runner.digest(path) for path in sources}
    for relative in ignored:
        (tmp_path / relative).write_text("changed documentation or report\n")
    after = {str(path): runner.digest(path) for path in runner.runtime_sources()}
    assert after == before


def test_packaged_runtime_dependencies_keep_original_licenses():
    package = Path(__file__).resolve().parents[1]
    for relative in (
        "vendor/openpyxl-3.1.5.dist-info/LICENCE.rst",
        "vendor/et_xmlfile-2.0.0.dist-info/LICENCE.rst",
        "vendor/et_xmlfile-2.0.0.dist-info/LICENCE.python",
        "vendor/et_xmlfile-2.0.0.dist-info/AUTHORS.txt",
    ):
        assert (package / relative).stat().st_size > 0
