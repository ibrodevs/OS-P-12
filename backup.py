#!/usr/bin/env python3
"""Incremental directory backup/restore. Python stdlib only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

JOURNAL = '.pr12_backup_journal.jsonl'

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.digest()

def is_within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False

def entries(root):
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(base)/d).is_symlink())
        for name in sorted(files):
            p = Path(base)/name
            if p.is_file() and not p.is_symlink() and p.name != JOURNAL:
                yield p.relative_to(root)

def log(root, action, rel, detail=''):
    with (root/JOURNAL).open('a', encoding='utf-8') as f:
        f.write(json.dumps({'at': datetime.now(timezone.utc).isoformat(), 'action': action,
                            'file': str(rel), 'detail': detail}, ensure_ascii=False) + '\n')

def sync(mode, src, dst):
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if not src.is_dir():
        raise ValueError(f'Каталог-источник не существует: {src}')
    if src == dst or is_within(dst, src) or is_within(src, dst):
        raise ValueError('Источник и назначение не могут быть вложены друг в друга')
    dst.mkdir(parents=True, exist_ok=True)
    copied = skipped = preserved = 0
    for rel in entries(src):
        origin, target = src/rel, dst/rel
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f'Неподдерживаемый тип файла в назначении: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and origin.stat().st_size == target.stat().st_size and digest(origin) == digest(target):
            skipped += 1
            continue
        if mode == 'restore' and target.exists():
            old = target.with_name(target.name + '.before_restore')
            if old.exists():
                # Preserve an older conflicting snapshot rather than overwriting it.
                stamp = 1
                while old.with_name(old.name + f'.{stamp}').exists():
                    stamp += 1
                old = old.with_name(old.name + f'.{stamp}')
            shutil.copy2(target, old)
            preserved += 1
            log(dst, 'before_restore', rel, str(old.relative_to(dst)))
        temp = target.with_name(target.name + '.pr12_tmp')
        try:
            shutil.copy2(origin, temp)
            os.replace(temp, target)
        finally:
            if temp.exists():
                temp.unlink()
        copied += 1
        log(dst, mode, rel, 'sha256=' + digest(target).hex())
    # Directories including empty ones are also preserved; existing directories are not deleted.
    for base, dirs, _ in os.walk(src, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(base)/d).is_symlink()]
        for name in dirs:
            source_dir = Path(base)/name
            out_dir = dst/source_dir.relative_to(src)
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copystat(source_dir, out_dir)
    print(f'{mode}: скопировано={copied}, без изменений={skipped}, сохранено предыдущих версий={preserved}')
    print(f'Журнал: {dst / JOURNAL}')

if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] not in ('backup','restore'):
        print('Использование: bash backup.sh backup ИСТОЧНИК КОПИЯ | restore КОПИЯ ИСТОЧНИК', file=sys.stderr)
        sys.exit(2)
    try:
        sync(*sys.argv[1:])
    except (OSError, ValueError) as error:
        print('Ошибка:', error, file=sys.stderr)
        sys.exit(1)
