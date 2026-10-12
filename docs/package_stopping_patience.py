"""Package the original committed bytes, never checkout newline conversions.

Run before untracking the 800 individual artifacts. Does not remove any file.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_STORED, ZIP_DEFLATED

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.patience_archive import PublishedArtifacts, sha

ROOT = Path('docs/benchmarks/stopping_patience')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit')
    args = parser.parse_args()
    existing = ROOT/'artifacts_manifest.json'
    recorded = json.loads(existing.read_text())['source_commit'] if existing.exists() else 'HEAD'
    commit = subprocess.check_output(['git', 'rev-parse', args.source_commit or recorded], text=True).strip()
    prefix = ROOT.as_posix()+'/'
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit,
                                    prefix+'trajectories'], text=True).splitlines()
    if len(names) != 800:
        raise ValueError('Expected original 800 committed artifacts')
    # Four fixed alphabetical groups of five complete datasets, no outcome-based grouping.
    datasets = sorted({name[len(prefix):].split('/')[1] for name in names})
    members, archives = [], []
    for shard in range(4):
        filename = f'trajectories-{shard+1:02d}.zip'
        selected = set(datasets[shard*5:(shard+1)*5])
        with ZipFile(ROOT/filename, 'w') as z:
            for path in sorted(names):
                relative = path[len(prefix):]
                if relative.split('/')[1] not in selected:
                    continue
                data = subprocess.check_output(['git', 'show', f'{commit}:{path}'])
                info = ZipInfo(relative, (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = ZIP_STORED if path.endswith('.gz') else ZIP_DEFLATED
                z.writestr(info, data, compresslevel=9)
                members.append(dict(path=relative, archive=filename, bytes=len(data), sha256=sha(data)))
        data = (ROOT/filename).read_bytes()
        archives.append(dict(file=filename, bytes=len(data), sha256=sha(data)))
    manifest = dict(schema=1, source_commit=commit, member_count=800,
                    format='ZIP; fixed timestamps/permissions/order; stored gzip, deflated text level 9',
                    archives=archives, members=sorted(members, key=lambda r:r['path']))
    (ROOT/'artifacts_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    with PublishedArtifacts(ROOT) as published:
        for name in names:
            assert published.read(name[len(prefix):]) == subprocess.check_output(['git', 'show', f'{commit}:{name}'])
    print('Verified all 800 archived members byte-for-byte against Git commit', commit)


if __name__ == '__main__':
    main()
