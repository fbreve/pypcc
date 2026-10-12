"""Inspect published historical signals only; never run PCC or candidate policies."""
import csv
import gzip
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.patience_archive import PublishedArtifacts, sha


def inspect(root):
    protocol = json.loads((root/'protocol.json').read_text())
    schemas, checks, observations, extra = {}, 0, 0, 0
    gaps = []
    with PublishedArtifacts(root) as published:
        for info in json.loads((root/'inventory.json').read_text())['datasets']:
            for seed in protocol['seeds']:
                prefix = f"trajectories/{info['name']}/{seed}/"
                result = json.loads(published.read(prefix+'complete.json'))
                recorded = json.loads(published.read(prefix+'verification.json'))
                assert recorded == result['verification']
                schemas['complete.json'] = sorted(result)
                schemas['complete.json.rows'] = sorted(result['rows'][0])
                schemas['verification.json'] = sorted(recorded[0])
                raw = gzip.decompress(published.read(prefix+'checks.csv.gz'))
                reader = csv.DictReader(io.StringIO(raw.decode()))
                assert reader.fieldnames == ['iteration','metric','record_gain','record','counter','resets']
                schemas['checks.csv.gz'] = reader.fieldnames
                count = 0
                for row in reader:
                    assert int(row['iteration']) == 1+10*count
                    count += 1
                assert count == result['checks'] == 50000
                checks += count
                reader = csv.DictReader(io.StringIO(published.read(prefix+'samples.csv').decode()))
                assert reader.fieldnames == ['iteration','metric','correct_nodes','accuracy_unlabeled','changed_nodes','is_reference']
                schemas['samples.csv'] = reader.fieldnames
                rows = list(reader)
                times = [int(r['iteration']) for r in rows]
                assert times == sorted(set(times))
                assert all(t in times for t in range(1000,500001,1000))
                assert len(rows) == result['samples']
                gaps.extend(b-a for a,b in zip([0]+times,times))
                extra += sum(t%1000 != 0 for t in times)
                observations += len(rows)
    return dict(source_master='479952fc988d48c754202cd4d12eac898157d513',
        artifacts_manifest_sha256=sha((root/'artifacts_manifest.json').read_bytes()),
        protocol_sha256=sha((root/'protocol.json').read_bytes().replace(b'\r\n',b'\n')),
        archive_members=800,trajectories=200,checks=checks,
        check_first_iteration=1,check_spacing=10,check_last_iteration=499991,
        classification_observations=observations,regular_observations=100000,
        extra_legacy_event_observations=extra,minimum_observation_gap=min(gaps),
        maximum_observation_gap=max(gaps),schemas=schemas,
        propagation_executed=False,candidate_policies_evaluated=False)


if __name__=='__main__':
    print(json.dumps(inspect(Path('docs/benchmarks/stopping_patience')),indent=2))
