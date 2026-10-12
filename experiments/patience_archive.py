"""Lossless public artifact reader; no PCC, RNG or checkpoint dependencies."""
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
from zipfile import ZipFile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(name):
    p = PurePosixPath(name)
    if (p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name
            or str(p) != name or not name.startswith('trajectories/')):
        raise ValueError('Unsafe artifact path')
    return name


class PublishedArtifacts:
    def __init__(self, root, artifact_dir=None):
        self.root = root
        self.manifest = json.loads((root/'artifacts_manifest.json').read_text())
        configured = artifact_dir or os.environ.get('PYPCC_STOPPING_ARTIFACTS')
        cache = Path(configured) if configured else Path('.cache/stopping_artifacts')
        # Transitional fallback keeps old checkouts/tests usable until external validation.
        location = cache if configured or all((cache/a['file']).exists() for a in self.manifest['archives']) else root
        self.entries = {}
        self.zips = {}
        try:
            for archive in self.manifest['archives']:
                name = archive['file']
                if (PurePosixPath(name).name != name or '\\' in name or ':' in name
                        or not name.endswith('.zip')):
                    raise ValueError('Unsafe archive path')
                data = (location/name).read_bytes()
                if len(data) != archive['bytes'] or sha(data) != archive['sha256']:
                    raise ValueError('Archive checksum mismatch')
                z = self.zips[name] = ZipFile(location/name)
                expected = {r['path'] for r in self.manifest['members'] if r['archive'] == name}
                names = z.namelist()
                if len(names) != len(set(names)) or set(names) != expected:
                    raise ValueError('Duplicate, missing or unexpected ZIP members')
            for entry in self.manifest['members']:
                name = safe_path(entry['path'])
                if name in self.entries or entry['archive'] not in self.zips:
                    raise ValueError('Duplicate member or missing archive')
                self.entries[name] = entry
        except Exception:
            self.close()
            raise

    def read(self, name):
        entry = self.entries[safe_path(name)]
        data = self.zips[entry['archive']].read(name)
        if len(data) != entry['bytes'] or sha(data) != entry['sha256']:
            raise ValueError('Member checksum mismatch')
        return data

    def close(self):
        for z in self.zips.values():
            z.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
