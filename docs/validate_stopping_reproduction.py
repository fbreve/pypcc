"""Compare public reproduction outputs with committed scientific references."""
import argparse
import csv
import io
import json
import subprocess
from pathlib import Path


def validate(output, reference, ref=None, required=()):
    for name in required:
        if not (output/name).is_file():
            raise ValueError('Missing reproduction output: '+name)
    def original(name):
        if ref:
            return subprocess.check_output(['git','show',ref+':'+(reference/name).as_posix()])
        return (reference/name).read_bytes()
    names = subprocess.check_output(['git','ls-tree','-r','--name-only',ref],text=True).splitlines() if ref else None
    checked=[]
    for file in sorted(output.iterdir()):
        if file.suffix not in ('.csv','.png') and file.name not in ('report.md','full_report.md'):
            continue
        exists=(reference/file.name).as_posix() in names if ref else (reference/file.name).is_file()
        if not exists:
            continue
        expected=original(file.name)
        if file.suffix=='.csv':
            actual_rows=list(csv.reader(io.StringIO(file.read_text(encoding='utf-8'))))
            expected_rows=list(csv.reader(io.StringIO(expected.decode('utf-8'))))
            equal=actual_rows==expected_rows
        elif file.suffix=='.md':
            equal=file.read_text(encoding='utf-8')==expected.decode('utf-8').replace('\r\n','\n')
        else:
            equal=file.read_bytes()==expected
        if not equal:
            raise ValueError('Scientific reproduction mismatch: '+file.name)
        checked.append(file.name)
    if not checked:
        raise ValueError('No scientific references compared')
    if not set(required).issubset(checked):
        raise ValueError('Missing required scientific reference')
    return checked


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--ref',help='Optional Git reference for an unmerged registered study')
    parser.add_argument('--required',nargs='+',default=[],help='Required output/reference filenames')
    args=parser.parse_args()
    print(json.dumps(dict(identical=validate(args.output,args.reference,args.ref,args.required)),indent=2))
