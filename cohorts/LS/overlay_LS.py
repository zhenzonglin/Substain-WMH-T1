"""Build an isolated LS source overlay from the installed, working pipeline.

Only this overlay is changed. Models, environments and GB source remain read-only.
All edits are planned and compiled before any destination files are installed.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import tokenize
from pathlib import Path
from typing import Dict


VERSION = 1
LEDGER = "overlay_manifest_LS.json"
TABLES = (
    "wmh20_raw.tsv", "wmh20_z_chung.tsv", "t1_muse145_raw.tsv",
    "t1_gm119_raw.tsv", "t1_macro20_raw.tsv", "t1_macro20_z_genmind.tsv",
    "features_computed40.tsv", "features_primary40.tsv", "subject_qc.tsv",
    "feature_dictionary.tsv", "resource_manifest.tsv", "qc_reviews.tsv",
    "qc_reviews.sqlite", "audit_report.json", "audit_subjects.tsv",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace(text: str, old: str, new: str, count: int = 1) -> str:
    text = text.replace('\r\n', '\n')
    if text.count(old) != count:
        raise ValueError("Unrecognized pipeline source; no files changed: " + old[:120])
    return text.replace(old, new)


def _function(text: str, name: str, body: str) -> str:
    text = text.replace('\r\n', '\n')
    tree = ast.parse(text)
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError("Expected one function: " + name)
    node = matches[0]
    lines = text.splitlines(True)
    # Keep decorators and function signature; replace only the function body.
    start = node.body[0].lineno - 1
    stop = node.end_lineno
    return "".join(lines[:start]) + body.rstrip() + "\n" + "".join(lines[stop:])


def _named_literals(text: str, required: tuple = ()) -> str:
    """Rename only complete known file-name literals, never GB identifiers."""
    text = text.replace('\r\n', '\n')
    mapping = {name: Path(name).stem + "_LS" + Path(name).suffix for name in TABLES}
    seen = set()
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.STRING:
            try:
                value = ast.literal_eval(token.string)
            except (SyntaxError, ValueError):
                value = None
            if isinstance(value, str) and value in mapping:
                seen.add(value)
                token = tokenize.TokenInfo(token.type, repr(mapping[value]), token.start, token.end, token.line)
        tokens.append(token)
    absent = set(required) - seen
    if absent:
        raise ValueError("Unrecognized file-name contract: " + repr(sorted(absent)))
    return tokenize.untokenize(tokens)


def _runtime_source(root: Path, config_path: Path, destination: Path) -> str:
    constants = "PROJECT_ROOT = Path({!r})\nCONFIG_PATH = Path({!r})\nOVERLAY_ROOT = Path({!r})\n".format(
        str(root), str(config_path), str(destination))
    return '''"""LS-only guards; never imports a GB resume-audit cache."""
import csv
import hashlib
import json
from pathlib import Path

''' + constants + '''

def validate_config_path_LS(path):
    if Path(path).resolve() != CONFIG_PATH:
        raise ValueError("LS overlay accepts only " + str(CONFIG_PATH))


def validate_config_LS(config):
    candidate = Path(str(config["project_root"]))
    candidate = candidate if candidate.is_absolute() else CONFIG_PATH.parent / candidate
    if candidate.resolve() != PROJECT_ROOT:
        raise ValueError("LS project_root changed")
    if config.get("input", {}).get("cohort") != "LS":
        raise ValueError("LS overlay requires input.cohort=LS")
    if config.get("cohort_suffix", "_LS") != "_LS":
        raise ValueError("LS overlay requires cohort_suffix=_LS")


def audit_guard_LS(config):
    validate_config_LS(config)
    value = config.get("ls", {}).get("audit_guard_path")
    if not value:
        raise ValueError("Missing LS audit guard; run the LS runner audit first")
    path = Path(str(value))
    if not path.is_absolute() or not path.name.endswith("_LS.json") or path.is_symlink():
        raise ValueError("LS audit guard must be an absolute, regular *_LS.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise ValueError("LS audit guard has not passed")
    return str(path)


def existing_manifest_LS(root, config, expected):
    validate_config_LS(config)
    value = config.get("ls", {}).get("resource_manifest_path")
    if not value:
        raise ValueError("LS resource manifest is missing from config")
    path = Path(str(value))
    if not path.is_absolute() or path.resolve() != Path(expected).resolve():
        raise ValueError("LS resource manifest must match the declared DAG output")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Prebuilt LS resource manifest is missing or symlinked: " + str(path))
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\\t")
        if not {"relative_path", "sha256"}.issubset(reader.fieldnames or []):
            raise ValueError("Invalid LS resource manifest columns")
        rows = list(reader)
    if not rows or len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("Empty or duplicate LS resource manifest")
    errors = []
    for row in rows:
        resource = Path(root) / row["relative_path"]
        if not resource.is_file():
            errors.append(row["relative_path"])
            continue
        value = hashlib.sha256()
        with resource.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                value.update(block)
        if value.hexdigest() != row["sha256"]:
            errors.append(row["relative_path"])
    if errors:
        raise ValueError("LS resource manifest verification failed: " + repr(errors))
    return {"status": "pass", "checked_files": len(rows), "network_used": False}


def workdir_LS():
    path = OVERLAY_ROOT / "work_LS"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
'''


def _patch_cli(text: str, config_path: Path, destination: Path) -> str:
    text = _named_literals(text, ("resource_manifest.tsv",))
    text = _replace(text, 'DEFAULT_CONFIG = Path("config/config.yaml")',
                    'from .runtime_LS import existing_manifest_LS, workdir_LS\n\nDEFAULT_CONFIG = Path({!r})'.format(str(config_path)))
    text = _replace(text, 'str(root / "workflow" / "Snakefile"),',
                    'str(Path({!r})),'.format(str(destination / "workflow/Snakefile_LS")))
    text = _replace(text, 'completed = subprocess.run(command, cwd=str(root), check=False)',
                    'completed = subprocess.run(command, cwd=workdir_LS(), check=False)')
    text = _replace(text, '    build_manifest(root, manifest_path)',
                    '    existing_manifest_LS(root, config, manifest_path)')
    text = _replace(text, '    gpu_ids = detect_gpu_ids()',
                    '    gpu_ids = [str(value) for value in config.get("execution", {}).get("gpu_devices", [])] or detect_gpu_ids()')
    text = _function(text, "build_mapping_command", '    raise click.ClickException("LS reuses the existing mapping; rebuilding shared resources is disabled")')
    body = '''    config = load_config(config_file.resolve())
    if smoke_test:
        raise click.ClickException("LS does not rebuild or smoke-test shared environments")
    root = Path(str(config["project_root"]))
    manifest = _derivatives_root(config, root) / "tables" / "resource_manifest_LS.tsv"
    report = existing_manifest_LS(root, config, manifest)
    output = _derivatives_root(config, root) / "audit" / "integrity_LS.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    click.echo(json.dumps(report, ensure_ascii=False, indent=2))'''
    text = _function(text, "verify_offline_command", body)
    return text


def _patch_snake(text: str, root: Path, config_path: Path, destination: Path) -> str:
    text = _replace(text, 'configfile: "config/config.yaml"', 'configfile: {!r}'.format(str(config_path)))
    text = _replace(text, 'ROOT = Path(workflow.basedir).resolve().parent',
                    'ROOT = Path({!r})\nSOURCE_ROOT = Path({!r})'.format(str(root), str(destination)))
    text = _replace(text, 'CONFIG_FILE = Path(config.get("active_config_file", ROOT / "config" / "config.yaml"))',
                    'CONFIG_FILE = Path({!r})'.format(str(config_path)))
    if '# GB_RESUME_AUDIT_V1' in text:
        start = text.index('# GB_RESUME_AUDIT_V1')
        stop = text.index('\n\nif ROLLING_ENABLED:', start)
        block = text[start:stop]
        if 'GB_AUDIT_INPUT = ancient(_gb_audit.validate(ROOT, CONFIG_FILE))' not in block:
            raise ValueError("Unrecognized GB resume guard")
        audit_input = 'audit=GB_AUDIT_INPUT,'
    else:
        if text.count('rule audit:\n') != 1:
            raise ValueError("Expected original audit rule or confirmed GB resume guard")
        start = text.index('rule audit:\n')
        stop = text.index('\n\nif ROLLING_ENABLED:', start)
        if 'images=PARTICIPANT_INPUTS' not in text[start:stop]:
            raise ValueError("Unrecognized original audit rule")
        audit_input = 'audit=rules.audit.output,'
    guard = ('# LS_AUDIT_GUARD_V1\n'
             'import sys as _ls_sys\n_ls_sys.path.insert(0, str(SOURCE_ROOT / "src"))\n'
             'from substain_features.runtime_LS import audit_guard_LS\n'
             'LS_AUDIT_INPUT = ancient(audit_guard_LS(config))\n')
    text = text[:start] + guard + text[stop:]
    text = _replace(text, audit_input, 'audit=LS_AUDIT_INPUT,')
    # These are source-location references only. Keep ROOT for every model/tool path.
    if 'PYTHONPATH={ROOT}/src' not in text or 'PYTHONPATH={root}/src' not in text:
        raise ValueError("Unrecognized workflow Python dispatch")
    text = text.replace('PYTHONPATH={ROOT}/src', 'PYTHONPATH={SOURCE_ROOT}/src')
    text = text.replace('root=ROOT,', 'root=SOURCE_ROOT,')
    text = _replace(text, '"PYTHONPATH={}/src {} {}".format(ROOT, python, command)',
                    '"PYTHONPATH={}/src {} {}".format(SOURCE_ROOT, python, command)')
    # .format calls need SOURCE_ROOT; Snakemake shell templates use its global directly.
    text = text.replace('.format(ROOT=ROOT,', '.format(SOURCE_ROOT=SOURCE_ROOT, ROOT=ROOT,')
    text = _replace(text, 'ROLLING_COMPLETE = str(ROLLING_TOKEN_DIR / "all_complete.json")',
                    'ROLLING_COMPLETE = str(ROLLING_TOKEN_DIR / "all_complete_LS.json")')
    for name in ('features_computed40.tsv', 'subject_qc.tsv', 'resource_manifest.tsv'):
        if text.count('"' + name + '"') != 2:
            raise ValueError("Unrecognized workflow output contract: " + name)
        text = text.replace('"' + name + '"', repr(Path(name).stem + '_LS.tsv'))
    if 'gb_resume_audit' in text or 'GB_AUDIT_INPUT' in text or 'rules.audit.output' in text:
        raise ValueError("GB audit dependency remains in LS workflow")
    return text


def build_overlay(root: Path, config_path: Path, destination: Path, dry_run: bool = False) -> Dict[str, object]:
    """Plan, validate, then install immutable LS copies; never overwrite edits."""
    root = Path(root).resolve(strict=True)
    config_path = Path(config_path).resolve(strict=True)
    destination = Path(destination).absolute()
    if destination.is_symlink() or destination.resolve() != root / 'runtime_LS':
        raise ValueError("Overlay destination must be the regular project runtime_LS directory")
    destination = destination.resolve()
    if config_path.parent != root / 'config' or config_path.name != 'config_LS.yaml':
        raise ValueError("Expected project config/config_LS.yaml")
    source = root / 'src/substain_features'
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Source package missing or symlinked")
    originals = {}
    planned = {}
    for path in sorted(source.rglob('*')):
        if '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo'):
            continue
        if path.is_symlink():
            raise ValueError("Source package contains a symlink: " + str(path))
        if path.is_file():
            name = path.relative_to(root).as_posix()
            originals[name] = path.read_bytes()
            planned[name] = originals[name]
    required = ('__init__.py', 'schema.py', 'input_prep.py', 'pipeline.py', 'audit.py', 'qc_review.py', 'resources.py', 'cli.py')
    for name in required:
        if 'src/substain_features/' + name not in planned:
            raise ValueError("Missing source module: " + name)
    for name, mandatory in (
        ('pipeline.py', TABLES[:10]), ('audit.py', ('audit_report.json', 'audit_subjects.tsv')),
        ('qc_review.py', ('qc_reviews.sqlite', 'qc_reviews.tsv')),
    ):
        key = 'src/substain_features/' + name
        planned[key] = _named_literals(planned[key].decode('utf-8'), mandatory).encode('utf-8')
    key = 'src/substain_features/schema.py'
    text = planned[key].decode('utf-8')
    text = _replace(text, '    with path.open("r", encoding="utf-8") as handle:',
                    '    from .runtime_LS import validate_config_path_LS, validate_config_LS\n'
                    '    validate_config_path_LS(path)\n    with path.open("r", encoding="utf-8") as handle:')
    text = _replace(text, '    config["project_root"] = str(root)\n    return config',
                    '    config["project_root"] = str(root)\n    validate_config_LS(config)\n    return config')
    planned[key] = text.encode('utf-8')
    key = 'src/substain_features/input_prep.py'
    planned[key] = _function(planned[key].decode('utf-8'), 'prepare_inputs',
        '    raise ValueError("LS inputs must be prepared by the LS runner; shared GB preparation is disabled")').encode('utf-8')
    key = 'src/substain_features/resources.py'
    planned[key] = _function(planned[key].decode('utf-8'), 'build_manifest',
        '    raise ValueError("LS uses a prebuilt scoped resource manifest; recursive shared-root rebuild is disabled")').encode('utf-8')
    key = 'src/substain_features/cli.py'
    planned[key] = _patch_cli(planned[key].decode('utf-8'), config_path, destination).encode('utf-8')
    key = 'src/substain_features/runtime_LS.py'
    if key in planned:
        raise ValueError("Shared package unexpectedly contains runtime_LS.py")
    planned[key] = _runtime_source(root, config_path, destination).encode('utf-8')
    snake_path = root / 'workflow/Snakefile'
    if snake_path.is_symlink():
        raise ValueError("Shared Snakefile must not be a symlink")
    originals['workflow/Snakefile'] = snake_path.read_bytes()
    planned['workflow/Snakefile_LS'] = _patch_snake(originals['workflow/Snakefile'].decode('utf-8'), root, config_path, destination).encode('utf-8')
    for name, data in planned.items():
        if name.endswith('.py'):
            ast.parse(data.decode('utf-8'), filename=name, feature_version=(3, 8))
            compile(data, name, 'exec')
    source_hashes = {name: digest(data) for name, data in sorted(originals.items())}
    payload = {'version': VERSION, 'project_root': str(root), 'config_path': str(config_path),
               'destination': str(destination), 'source_sha256': source_hashes,
               'source_fingerprint': digest(json.dumps(source_hashes, sort_keys=True).encode('utf-8')),
               'files': {name: digest(data) for name, data in sorted(planned.items())}}
    planned[LEDGER] = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')
    # Preflight *every* target and source before installing any file.
    for managed in (destination / 'src', destination / 'workflow'):
        if managed.is_dir():
            for path in managed.rglob('*'):
                if '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo'):
                    continue
                if path.is_file() and path.relative_to(destination).as_posix() not in planned:
                    raise ValueError("Unexpected file in managed overlay; refusing overwrite: " + str(path))
    for name, data in planned.items():
        path = destination / name
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
            raise ValueError("Symlinked overlay target: " + str(path))
        if path.exists() and (not path.is_file() or path.read_bytes() != data):
            raise ValueError("Existing overlay changed or source baseline differs; refusing overwrite: " + name)
    for name, data in originals.items():
        if (root / name).read_bytes() != data:
            raise RuntimeError("Source changed during overlay planning: " + name)
    if not dry_run:
        created = []
        try:
            for name, data in planned.items():
                path = destination / name
                if path.exists():
                    if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                        raise RuntimeError("Overlay target changed during installation: " + name)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation avoids replacing a file produced after preflight.
                with path.open('xb') as handle:
                    created.append(path)
                    handle.write(data)
        except Exception:
            for path in reversed(created):
                path.unlink()
            raise
    return payload
