"""Small public archive checks; no downloads, propagation or checkpoints."""
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from experiments.patience_archive import PublishedArtifacts, safe_path, sha


class ArchiveTests(unittest.TestCase):
    def fixture(self, root, extra=False):
        name='trajectories/Example/10/complete.json'
        data=b'{"iterations":500000}\n'
        with ZipFile(root/'sample.zip','w') as z:
            z.writestr(name,data)
            if extra:
                z.writestr('trajectories/Example/10/unexpected.txt',b'x')
        packed=(root/'sample.zip').read_bytes()
        manifest=dict(archives=[dict(file='sample.zip',bytes=len(packed),sha256=sha(packed))],
            members=[dict(path=name,archive='sample.zip',bytes=len(data),sha256=sha(data))])
        (root/'artifacts_manifest.json').write_text(json.dumps(manifest))
        return name,data

    def test_published_roundtrip_without_local_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);name,data=self.fixture(root)
            with PublishedArtifacts(root) as published:
                self.assertEqual(published.read(name),data)
            self.assertFalse((root/'trajectories').exists())

    def test_reject_corrupt_archive_and_member(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);name,_=self.fixture(root)
            m=json.loads((root/'artifacts_manifest.json').read_text())
            m['members'][0]['sha256']='0'*64
            (root/'artifacts_manifest.json').write_text(json.dumps(m))
            with PublishedArtifacts(root) as published:
                with self.assertRaises(ValueError):published.read(name)
            (root/'sample.zip').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):PublishedArtifacts(root)

    def test_reject_unexpected_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,extra=True)
            with self.assertRaises(ValueError):PublishedArtifacts(root)

    def test_reject_unsafe_paths(self):
        for name in ('../escape','/trajectories/file','trajectories/a/../escape',
                     'trajectories\\a','trajectories/C:/a','trajectories//a'):
            with self.subTest(name=name),self.assertRaises(ValueError):safe_path(name)


if __name__=='__main__':
    unittest.main()
