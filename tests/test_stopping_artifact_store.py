import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from experiments.stopping_artifact_store import ensure


class StoreTests(unittest.TestCase):
    def entry(self):
        return dict(file='example.zip',bytes=3,sha256=hashlib.sha256(b'abc').hexdigest())

    def test_offline_never_connects(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'example.zip';p.write_bytes(b'abc')
            def fail(*args,**kwargs):raise AssertionError('network')
            self.assertEqual(ensure(self.entry(),d,download=True,opener=fail),p)
            p.unlink()
            with self.assertRaises(FileNotFoundError):ensure(self.entry(),d,opener=fail)

    def test_verified_download_and_failed_download_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                ensure(self.entry(),d,download=True,opener=lambda *a,**k:io.BytesIO(b'bad'))
            self.assertEqual(list(Path(d).iterdir()),[])
            p=ensure(self.entry(),d,download=True,opener=lambda *a,**k:io.BytesIO(b'abc'))
            self.assertEqual(p.read_bytes(),b'abc')

    def test_corrupt_local_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'example.zip';p.write_bytes(b'bad')
            with self.assertRaises(ValueError):ensure(self.entry(),d,download=True)
            self.assertEqual(p.read_bytes(),b'bad')

    def test_unsafe_path(self):
        entry=self.entry();entry['file']='../escape.zip'
        with self.assertRaises(ValueError):ensure(entry,'.cache')
