"""Unit-level filesystem operations (without kernel FUSE mounting)."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'level_c' / 'fuse_fs.py'
spec = importlib.util.spec_from_file_location('pr12_fuse', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class TestPr12FS(unittest.TestCase):
    def test_file_lifecycle_and_access_counter(self):
        with tempfile.TemporaryDirectory() as base:
            fs = module.Pr12FS(base)
            fh = fs.create('/a.txt', 0o600)
            self.assertEqual(fs.write('/a.txt', b'abc', 0, fh), 3)
            fs.release('/a.txt', fh)
            reader = fs.open('/a.txt', 0)
            self.assertEqual(fs.read('/a.txt', 3, 0, reader), b'abc')
            fs.release('/a.txt', reader)
            self.assertIn('a.txt', fs.readdir('/', None))
            self.assertIn('.access_counts.json', fs.readdir('/', None))
            self.assertIn(b'"write": 1', fs.read('/.access_counts.json', 4096, 0, 0))
            fs.unlink('/a.txt')
            self.assertNotIn('a.txt', fs.readdir('/', None))

if __name__ == '__main__':
    unittest.main()
