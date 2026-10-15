"""Small portable archive checks; no historical fits or backend imports."""
import hashlib
import json
from pathlib import Path
import tempfile
import struct
import unittest
import zipfile
from tools.verify_early_stopping_archive import record, digest, stats, verify_zip, loads, npz_hashes


class PortableArchiveTests(unittest.TestCase):
    def test_envelope_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'record.json';payload={'iteration':500000}
            p.write_text(json.dumps(dict(schema=1,payload=payload,sha256=digest(payload))))
            self.assertEqual(record(p),payload)
            p.write_text(json.dumps(dict(schema=1,payload=payload,sha256='0'*64)))
            with self.assertRaises(ValueError):record(p)
        with self.assertRaises(ValueError):loads('{"x":1,"x":2}')
        with self.assertRaises(ValueError):loads('{"x":NaN}')

    def test_stats(self):
        s=stats([1,2,3,4,5,100])
        self.assertEqual(s['median'],3.5)
        self.assertEqual(s['iqr'],2.5)
        self.assertEqual((s['minimum'],s['maximum']),(1,100))

    def test_verify_extract_and_refuse_existing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);data=b'original\r\nbytes\r\n'
            with zipfile.ZipFile(p/'good.zip','w') as z:
                z.writestr('data/value.txt',data)
                z.writestr('checksums.json',json.dumps({'data/value.txt':hashlib.sha256(data).hexdigest()}))
            self.assertEqual(verify_zip(p/'good.zip',p/'out')['audit'],'passed')
            self.assertEqual((p/'out/data/value.txt').read_bytes(),data)
            with self.assertRaises(ValueError):verify_zip(p/'good.zip',p/'out')

    def test_corruption_missing_and_unsafe(self):
        for name,index in [('data.txt',{'data.txt':'0'*64}),('../bad',{'../bad':'0'*64}),('extra.txt',{})]:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                p=Path(d)/'bad.zip'
                with zipfile.ZipFile(p,'w') as z:
                    z.writestr(name,b'bad');z.writestr('checksums.json',json.dumps(index))
                with self.assertRaises(ValueError):verify_zip(p)

    def test_npy_array_fingerprint_without_numpy(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'arrays.npz'
            header=repr(dict(descr='<f8',fortran_order=False,shape=(1,2))).encode()+b'\n'
            data=struct.pack('<dd',1.,2.)
            npy=b'\x93NUMPY\x01\x00'+len(header).to_bytes(2,'little')+header+data
            with zipfile.ZipFile(p,'w') as z:z.writestr('x.npy',npy)
            h=hashlib.sha256(json.dumps([[1,2],'<f8']).encode()+data).hexdigest()
            self.assertEqual(npz_hashes(p),{'x':h})
            with zipfile.ZipFile(p,'w') as z:z.writestr('x.npy',npy[:-1])
            with self.assertRaises(ValueError):npz_hashes(p)
