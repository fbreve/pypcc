"""Verify local archives offline, or explicitly download the proposed Release assets."""
import argparse
import json
import hashlib
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.stopping_artifact_store import ensure

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir',type=Path,default=Path('.cache/stopping_artifacts'))
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--stage-release',type=Path,help='Prepare local upload copies and checksums; never publish')
    args=parser.parse_args()
    source=Path('docs/benchmarks/stopping_patience')
    manifest=json.loads((source/'artifacts_manifest.json').read_text())
    paths=[]
    for entry in manifest['archives']:
        path=ensure(entry,args.artifact_dir,download=args.download)
        paths.append(path);print(path)
    if args.stage_release:
        args.stage_release.mkdir(parents=True,exist_ok=True)
        paths.extend(source/name for name in ('artifacts_manifest.json','archive_manifest.csv'))
        checks=[]
        for path in sorted(paths):
            target=args.stage_release/path.name
            shutil.copyfile(path,target)
            digest=hashlib.sha256(target.read_bytes()).hexdigest()
            checks.append(f'{digest}  {target.name}\n')
        (args.stage_release/'SHA256SUMS').write_text(''.join(checks),encoding='ascii')
