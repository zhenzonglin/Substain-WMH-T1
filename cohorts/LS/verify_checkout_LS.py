#!/usr/bin/env python3
"""Verify the Git-distributed LS source tree before modifying the workstation."""
import hashlib
from pathlib import Path, PurePosixPath
import re
import sys


def verify(directory: Path) -> int:
    directory = directory.resolve(strict=True)
    manifest = directory / 'SHA256SUMS_LS'
    required = {'inputs_LS.py', 'monitor_LS.py', 'overlay_LS.py', 'runner_LS.py',
                'install_LS.sh', 'verify_checkout_LS.py', 'vendor/openpyxl/__init__.py',
                'vendor/et_xmlfile/__init__.py'}
    checked = set()
    for number, line in enumerate(manifest.read_text(encoding='utf-8').splitlines(), 1):
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        if not match:
            raise ValueError('Invalid checksum line {}'.format(number))
        expected, name = match.groups()
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or name in checked:
            raise ValueError('Unsafe or duplicate checksum path: ' + name)
        path = directory.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError('Missing or symlinked checkout file: ' + name)
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError('Checkout SHA256 mismatch: ' + name)
        checked.add(name)
    if not required.issubset(checked):
        raise ValueError('Incomplete checksum manifest: ' + ', '.join(sorted(required - checked)))
    return len(checked)


def main() -> int:
    try:
        directory = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(__file__).resolve().parent
        count = verify(directory)
        print('LS_CHECKOUT_VERIFIED files={}'.format(count), flush=True)
        return 0
    except (OSError, ValueError) as error:
        print('LS_CHECKOUT_ERROR: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
