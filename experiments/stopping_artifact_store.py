"""Optional explicit download; pinned hashes are independent of release metadata."""
import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlopen

RELEASE_BASE='https://github.com/fbreve/pypcc/releases/download/stopping-trajectories-v1'


def verify(path, entry):
    digest=hashlib.sha256()
    size=0
    with path.open('rb') as stream:
        while block:=stream.read(1024*1024):
            digest.update(block);size+=len(block)
    if size!=entry['bytes'] or digest.hexdigest()!=entry['sha256']:
        raise ValueError(f'Artifact checksum mismatch: {path.name}')


def ensure(entry, directory, *, download=False, opener=urlopen):
    name=entry['file']
    if Path(name).name!=name or '\\' in name or ':' in name or not name.endswith('.zip'):
        raise ValueError('Unsafe artifact filename')
    directory=Path(directory);target=directory/name
    if target.exists():
        verify(target,entry);return target
    if not download:
        raise FileNotFoundError(f'{target}; copy the verified artifact or request --download')
    directory.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=directory,suffix='.part',delete=False) as output:
            temporary=Path(output.name)
            with opener(RELEASE_BASE+'/'+name,timeout=60) as response:
                while block:=response.read(1024*1024):output.write(block)
        verify(temporary,entry)
        temporary.replace(target)
        return target
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
