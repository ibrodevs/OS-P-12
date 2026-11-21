#!/usr/bin/env python3
"""Safely coalesce identical regular files on one filesystem using hard links."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile


def hash_file(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.digest()


def dedup(root):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Нужен каталог')
    indexed = {}
    replacements = saved_logical = saved_allocated = 0
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(base)/d).is_symlink())
        for name in sorted(names):
            p = Path(base)/name
            if p.is_symlink() or not p.is_file():
                continue
            st = p.stat()
            if st.st_size == 0:
                continue
            key = (st.st_dev, st.st_size, st.st_uid, st.st_gid, st.st_mode & 0o7777, hash_file(p))
            other = indexed.get(key)
            if other is None:
                indexed[key] = p
                continue
            try:
                old = other.stat()
                if (old.st_dev, old.st_ino) == (st.st_dev, st.st_ino):
                    continue
                # Recheck candidate metadata and data before replacement.
                if (old.st_size != st.st_size or old.st_uid != st.st_uid or
                    old.st_gid != st.st_gid or (old.st_mode & 0o7777) != (st.st_mode & 0o7777) or
                    hash_file(other) != hash_file(p)):
                    continue
                fd, tmp_name = tempfile.mkstemp(prefix='.pr12-link-', dir=p.parent)
                os.close(fd)
                os.unlink(tmp_name)
                try:
                    os.link(other, tmp_name)
                    os.replace(tmp_name, p)
                finally:
                    if os.path.lexists(tmp_name):
                        os.unlink(tmp_name)
                replacements += 1
                saved_logical += st.st_size
                saved_allocated += st.st_blocks * 512
                print(f'Объединено: {p.relative_to(root)} -> {other.relative_to(root)}')
            except OSError as e:
                print(f'Пропущено {p}: {e}', file=sys.stderr)
    print(f'Заменено файлов: {replacements}')
    print(f'Логический объём дубликатов: {saved_logical} байт')
    print(f'Оценка освобождённого выделенного места: {saved_allocated} байт (st_blocks; зависит от ФС)')
    return replacements

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Использование: python3 dedup.py КАТАЛОГ')
    try:
        dedup(sys.argv[1])
    except (OSError, ValueError) as e:
        sys.exit(f'Ошибка: {e}')
