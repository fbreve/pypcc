"""Pinned, cached public data for the optional patience study (no propagation)."""
import hashlib
import csv
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import arff
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder, StandardScaler


def sha(data):
    return hashlib.sha256(data).hexdigest()


def arff_text(raw):
    # OpenML PenDigits has spaces after delimiters. SciPy's nominal parser treats
    # those as part of the value; normalize only delimiter whitespace, not rows.
    lines = raw.decode('utf-8').splitlines()
    boundary = next(i for i, line in enumerate(lines) if line.strip().lower() == '@data')
    output = io.StringIO()
    output.write('\n'.join(lines[:boundary + 1]) + '\n')
    writer = csv.writer(output, lineterminator='\n')
    for line in lines[boundary + 1:]:
        if not line.strip() or line.lstrip().startswith('%'):
            continue
        writer.writerow([value.strip() for value in next(csv.reader([line], skipinitialspace=True))])
    return output.getvalue()


def retrieve(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Finite retries; an unavailable pinned source is a failure, not a replacement.
        for attempt in range(3):
            try:
                payload = urllib.request.urlopen(url, timeout=120).read()
                temporary = path.with_suffix(path.suffix + '.tmp')
                temporary.write_bytes(payload)
                temporary.replace(path)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2)
    return path.read_bytes()


def load(entry, cache, *, fetch=None):
    """Reuse preprocessing with an optional cache-only reader; default is unchanged."""
    if fetch is None:
        fetch = retrieve
    if 'loader' in entry:
        data = getattr(datasets, entry['loader'])()
        x, y = np.asarray(data.data, dtype=np.float64), data.target
        provenance = dict(source='scikit-learn ' + entry['loader'],
                          source_url='https://scikit-learn.org/stable/modules/generated/sklearn.datasets.' + entry['loader'] + '.html',
                          numeric_features=x.shape[1], nominal_features=0)
    else:
        ident = entry['openml_id']
        url = f'https://www.openml.org/api/v1/json/data/{ident}'
        raw_meta = fetch(url, cache / f'{ident}.json')
        meta = json.loads(raw_meta)['data_set_description']
        if int(meta['version']) != entry['version']:
            raise ValueError('Pinned dataset version mismatch')
        raw = fetch(meta['url'], cache / f'{ident}.arff')
        md5 = hashlib.md5(raw).hexdigest()
        if md5 != meta['md5_checksum']:
            raise ValueError('OpenML file checksum mismatch')
        data, attributes = arff.loadarff(io.StringIO(arff_text(raw)))
        target = meta['default_target_attribute']
        ignored = set((meta.get('ignore_attribute') or '').split(','))
        ignored.update((meta.get('row_id_attribute') or '').split(','))
        columns, numeric, nominal = [], 0, 0
        for key in data.dtype.names:
            if key == target or key in ignored:
                continue
            if attributes[key][0] == 'nominal':
                nominal += 1
                values = data[key]
                # Include all observed categories, including '?' if present.
                for category in sorted(np.unique(values)):
                    columns.append((values == category).astype(np.float64))
            else:
                numeric += 1
                values = data[key].astype(np.float64)
                if not np.isfinite(values).any():
                    raise ValueError('An all-missing numeric feature needs a protocol decision')
                values[~np.isfinite(values)] = np.nanmedian(values)
                columns.append(values)
        x = np.ascontiguousarray(np.column_stack(columns))
        y = data[target]
        if any(value in (b'?', '?') for value in y):
            raise ValueError('Missing target labels')
        provenance = dict(source='OpenML', openml_id=ident, version=int(meta['version']),
                          source_name=meta['name'], source_url=f'https://www.openml.org/d/{ident}',
                          download_url=meta['url'], metadata_sha256=sha(raw_meta),
                          raw_sha256=sha(raw), raw_md5=md5, numeric_features=numeric,
                          nominal_features=nominal, raw_bytes=len(raw))
    encoder = LabelEncoder()
    y = encoder.fit_transform(y).astype(np.int64)
    counts = np.bincount(y)
    if counts.min() < 4:
        raise ValueError('Three labels/class would leave no unknown nodes in a class')
    provenance.update(nodes=len(y), processed_features=x.shape[1], classes=len(counts),
                      class_counts=counts.tolist(), class_names=[str(v) for v in encoder.classes_],
                      imbalance_ratio=float(counts.max() / counts.min()))
    return x, StandardScaler().fit_transform(x), y, provenance
