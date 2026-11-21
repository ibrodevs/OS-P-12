#!/usr/bin/env python3
"""Educational passthrough FUSE FS with virtual /.access_counts.json.
Requires Ubuntu, fusepy, fuse3 and /dev/fuse. Not a security boundary.
"""
import errno
import json
import os
from pathlib import Path
import sys
from collections import Counter
from threading import RLock

try:
    from fuse import FUSE, FuseOSError, Operations
except ImportError:
    FUSE = None
    FuseOSError = OSError
    Operations = object

VIRTUAL = '/.access_counts.json'


class Pr12FS(Operations):
    def __init__(self, backing):
        self.root = Path(backing).resolve(strict=True)
        self.counter = Counter()
        self.lock = RLock()

    def _path(self, path):
        if path == VIRTUAL:
            raise FuseOSError(errno.EPERM)
        # FUSE paths are absolute, with root meaning the backing directory.
        p = (self.root / path.lstrip('/')).resolve(strict=False)
        if p != self.root and self.root not in p.parents:
            raise FuseOSError(errno.EACCES)
        return str(p)

    def _count(self, key):
        with self.lock:
            self.counter[key] += 1

    def _stats(self):
        with self.lock:
            return (json.dumps(dict(self.counter), ensure_ascii=False, indent=2) + '\n').encode()

    def getattr(self, path, fh=None):
        if path == VIRTUAL:
            import stat
            data = self._stats()
            return {'st_mode': stat.S_IFREG | 0o444, 'st_nlink': 1,
                    'st_size': len(data), 'st_uid': os.getuid(), 'st_gid': os.getgid(),
                    'st_atime': 0, 'st_mtime': 0, 'st_ctime': 0}
        st = os.lstat(self._path(path))
        return {name: getattr(st, name) for name in
                ('st_atime', 'st_ctime', 'st_gid', 'st_mode', 'st_mtime', 'st_nlink', 'st_size', 'st_uid')}

    def readdir(self, path, fh):
        names = os.listdir(self._path(path))
        return ['.', '..'] + sorted(names) + ([VIRTUAL.lstrip('/')] if path == '/' else [])

    def open(self, path, flags):
        if path == VIRTUAL:
            if flags & (os.O_WRONLY | os.O_RDWR):
                raise FuseOSError(errno.EACCES)
            return 0
        fd = os.open(self._path(path), flags)
        self._count('open')
        return fd

    def create(self, path, mode, fi=None):
        fd = os.open(self._path(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        self._count('create')
        return fd

    def read(self, path, size, offset, fh):
        self._count('read')
        if path == VIRTUAL:
            return self._stats()[offset:offset + size]
        return os.pread(fh, size, offset)

    def write(self, path, data, offset, fh):
        if path == VIRTUAL:
            raise FuseOSError(errno.EPERM)
        self._count('write')
        return os.pwrite(fh, data, offset)

    def truncate(self, path, length, fh=None):
        if path == VIRTUAL:
            raise FuseOSError(errno.EPERM)
        if fh is not None:
            os.ftruncate(fh, length)
        else:
            os.truncate(self._path(path), length)

    def flush(self, path, fh):
        if path != VIRTUAL:
            os.fsync(fh)
        return 0

    def release(self, path, fh):
        if path != VIRTUAL:
            os.close(fh)
        return 0

    def unlink(self, path):
        os.unlink(self._path(path))
        self._count('unlink')

    def mkdir(self, path, mode):
        os.mkdir(self._path(path), mode)

    def rmdir(self, path):
        os.rmdir(self._path(path))

    def rename(self, old, new):
        os.rename(self._path(old), self._path(new))

    def chmod(self, path, mode):
        return os.chmod(self._path(path), mode)

    def chown(self, path, uid, gid):
        return os.chown(self._path(path), uid, gid)

    def utimens(self, path, times=None):
        return os.utime(self._path(path), times)

    def access(self, path, mode):
        if path == VIRTUAL:
            if mode & os.W_OK:
                raise FuseOSError(errno.EACCES)
            return 0
        if not os.access(self._path(path), mode):
            raise FuseOSError(errno.EACCES)
        return 0

    def statfs(self, path):
        st = os.statvfs(str(self.root))
        return {key: getattr(st, key) for key in ('f_bavail','f_bfree','f_blocks','f_bsize',
                'f_favail','f_ffree','f_files','f_flag','f_frsize','f_namemax')}

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Usage: python3 level_c/fuse_fs.py BACKING_DIRECTORY MOUNTPOINT')
    if FUSE is None:
        sys.exit('fusepy not installed: pip install fusepy')
    FUSE(Pr12FS(sys.argv[1]), sys.argv[2], foreground=True, nothreads=True)
