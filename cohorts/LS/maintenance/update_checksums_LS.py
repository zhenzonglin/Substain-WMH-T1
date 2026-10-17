#!/usr/bin/env python3
"""Regenerate Git checkout checksums after reviewed LS source changes."""
import hashlib
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    lines = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.name in ('SHA256SUMS_LS', 'test_results_LS.xml'):
            continue
        if any(part in ('__pycache__', '.pytest_cache', '.ruff_cache', '.git') for part in path.parts):
            continue
        if path.suffix in ('.pyc', '.pyo', '.log', '.gz', '.zip'):
            continue
        if path.is_symlink():
            raise ValueError('Do not publish symlinked source: ' + str(path))
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append('{}  {}\n'.format(checksum, path.relative_to(root).as_posix()))
    (root / 'SHA256SUMS_LS').write_bytes(''.join(lines).encode('utf-8'))
    print('LS_MANIFEST_WRITTEN files={}'.format(len(lines)))


if __name__ == '__main__':
    main()
