"""Optional bounded full-state observations, not propagation checkpoints.

Little-endian NPY v1.0 arrays + unchanged v2 summary; a final atomic manifest
publishes the enriched bundle. No propagation, RNG or native-loop code here.
"""
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import uuid

import numpy as np

from benchmarks.checkpoints import (canonical, durable_file, read_record,
                                    sync_directory, unique_pairs)
from .convergence_trajectories import array_hash
from .trajectory_io import bound_metadata, read_output, write_output

PROTOCOL = 'pypcc-mechanistic-snapshots-v1'
LIMIT = 64 * 1024 * 1024
METADATA_LIMIT = 1024 * 1024
ARRAY_DTYPES = dict(iterations='<i8', dominance='<f8', positions='<i8', strengths='<f8',
                    node_ids='<i8', particle_ids='<i8', particle_home_nodes='<i8', particle_classes='<i8')


def _array_digest(value):
    # Same canonical contract as array_hash, without a full tobytes() allocation.
    value=np.asarray(value)
    if not value.flags.c_contiguous or value.dtype.str[0]!='<':
        raise ValueError('binary snapshots require C-contiguous little-endian arrays')
    h=hashlib.sha256();h.update(json.dumps([list(value.shape),value.dtype.str]).encode())
    h.update(memoryview(value).cast('B'))
    return h.hexdigest()


def _inventory(root):
    result={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('snapshot symlink refused')
        if path.is_file():
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            result[path.relative_to(root).as_posix()]=h.hexdigest()
    return result


@dataclass(frozen=True)
class SnapshotConfig:
    max_bytes: int = LIMIT
    max_memory_bytes: int = LIMIT
    max_export_bytes: int = LIMIT

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.max_bytes,self.max_memory_bytes,self.max_export_bytes)):
            raise ValueError('snapshot limits must be positive Python integers')


class SnapshotBundle(dict):
    """Read-only mapped arrays; close promptly, particularly on Windows."""
    def close(self):
        for value in self['arrays'].values():
            value._mmap.close()

    def __enter__(self):return self

    def __exit__(self,*args):self.close()


def estimates(n, c, p, slots, backend):
    """Conservative array/workspace estimate, not process RSS or a timing claim."""
    dynamic = slots * (8*n*c + 16*p + 8)
    static = 8*n + 24*p
    native = slots * (16*n*c + 16*p + n*p + 16 + (4 if backend=='cython' else 0)) + 16 if backend!='numpy' else 0
    legacy = native + slots*(128*n + 16384) + 128*n*c
    workspace = 2*max(8*slots*n*c,8*slots*p,8*n,24*p) + METADATA_LIMIT
    return dict(snapshot_array_bytes=dynamic+static, native_buffer_bytes=native,
                estimated_memory_bytes=legacy+static+(dynamic if backend=='numpy' else 0)+workspace,
                estimated_export_bytes=dynamic+static+slots*(128*n+16384)+2*METADATA_LIMIT)


def check_budget(config, estimate):
    if (estimate['snapshot_array_bytes']>config.max_bytes or estimate['estimated_memory_bytes']>config.max_memory_bytes
            or estimate['estimated_export_bytes']>config.max_export_bytes):
        raise ValueError('full snapshot preflight exceeds array, memory or export limit')


class Snapshots:
    def __init__(self, config, labels, homes, classes, schedule, backend):
        if not isinstance(config,SnapshotConfig):raise TypeError('snapshot_config must be SnapshotConfig or None')
        self.config=config;self.backend=backend;self.used=0
        self.n=len(labels);self.c=len(np.unique(np.asarray(labels)[np.asarray(labels)!=-1]));self.p=len(homes)
        self.schedule=tuple(schedule)
        self.estimate=estimates(self.n,self.c,self.p,len(schedule),backend);check_budget(config,self.estimate)
        homes=np.asarray(homes);classes=np.asarray(classes)
        if not np.array_equal(homes,np.flatnonzero(np.asarray(labels)!=-1)) or not np.array_equal(classes,np.asarray(labels)[homes]):
            raise ValueError('particle identity/order differs from initialization contract')
        self.arrays=dict(node_ids=np.arange(self.n,dtype='<i8'),particle_ids=np.arange(self.p,dtype='<i8'),
                         particle_home_nodes=homes.astype('<i8',copy=True),particle_classes=classes.astype('<i8',copy=True))
        if backend=='numpy':
            self.arrays.update(iterations=np.empty(len(schedule),dtype='<i8'),dominance=np.empty((len(schedule),self.n,self.c),dtype='<f8'),
                positions=np.empty((len(schedule),self.p),dtype='<i8'),strengths=np.empty((len(schedule),self.p),dtype='<f8'))
        self.source_hashes={'experiments/mechanistic_snapshots.py':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()}

    def bind_native(self,state):
        if self.backend=='numpy' or 'dominance' in self.arrays:raise ValueError('native snapshots already bound or not native')
        # Keep only selected arrays, not the tuple: own/dist buffers can be freed.
        for key,index in [('iterations',1),('dominance',2),('positions',3),('strengths',4)]:
            self.arrays[key]=state[index]

    def capture(self,iteration,dominance,positions,strengths,*,native):
        if native!=(self.backend!='numpy'):raise ValueError('snapshot observation/backend mode mismatch')
        i=self.used
        if i>=len(self.schedule):raise ValueError('snapshot capacity exhausted')
        if native:
            if int(self.arrays['iterations'][i])!=iteration or any(not np.shares_memory(value,self.arrays[key][i]) for key,value in
                    [('dominance',dominance),('positions',positions),('strengths',strengths)]):
                raise ValueError('snapshot does not reference the native observation buffer')
        else:
            self.arrays['iterations'][i]=iteration
            np.copyto(self.arrays['dominance'][i],dominance,casting='no')
            np.copyto(self.arrays['positions'][i],positions,casting='no')
            np.copyto(self.arrays['strengths'][i],strengths,casting='no')
        self.used+=1

    def values(self):
        return {key:(value[:self.used] if key in ('iterations','dominance','positions','strengths') else value)
                for key,value in self.arrays.items()}


def _npy_size(value):
    header=io.BytesIO()
    np.lib.format.write_array_header_1_0(header,np.lib.format.header_data_from_array_1_0(value))
    return len(header.getvalue())+value.nbytes


def _metadata(collector,provenance):
    store=collector.snapshots
    if not collector.finished or store is None or store.used!=len(collector.rows):
        raise ValueError('export requires a finished full-snapshot collector')
    summary_metadata=bound_metadata(collector,provenance)  # reject drift before writing
    check_budget(store.config,store.estimate)
    if len(canonical(summary_metadata))>METADATA_LIMIT or len(summary_metadata['dataset'].encode('utf-8'))>256:
        raise ValueError('summary context exceeds snapshot metadata workspace')
    values=store.values()
    return dict(protocol=PROTOCOL,source_hashes=store.source_hashes,
        limits=dict(max_bytes=store.config.max_bytes,max_memory_bytes=store.config.max_memory_bytes,max_export_bytes=store.config.max_export_bytes),
        estimates=store.estimate,backend=store.backend,observations=store.used,nodes=store.n,classes=store.c,particles=store.p,
        scheduled_iterations=list(store.schedule),iterations=values['iterations'].tolist(),
        particle_order='ascending labeled node index; ids are original particle indices',
        node_order='graph row index; no reordering',
        arrays={key:dict(shape=list(value.shape),dtype=value.dtype.str,order='C',array_sha256=_array_digest(value),file_bytes=_npy_size(value)) for key,value in values.items()})


def write_snapshots(directory,collector,provenance,*,fault=lambda point:None):
    """Exclusive new destination. Missing final manifest ALWAYS means incomplete."""
    meta=_metadata(collector,provenance);store=collector.snapshots;values=store.values()
    encoded=canonical(meta)+b'\n'
    if len(encoded)>METADATA_LIMIT:raise ValueError('snapshot metadata exceeds bounded workspace')
    root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    write_output(root/'summary',collector,provenance)
    for path in (root/'summary').iterdir():
        with path.open('r+b') as f:os.fsync(f.fileno())
    fault('summary_written')
    for key,value in values.items():
        if value.dtype.str!=ARRAY_DTYPES[key] or not value.flags.c_contiguous:
            raise ValueError('snapshot dtype/layout must remain unchanged and C-contiguous')
        with (root/(key+'.npy')).open('xb') as f:
            np.lib.format.write_array(f,value,version=(1,0),allow_pickle=False);f.flush();os.fsync(f.fileno())
        fault('array_written')
    durable_file(root/'mechanism.json',encoded)
    files=_inventory(root)
    # Validate semantics, dimensions, hashes and budget BEFORE publication.
    with _read(root,files,max_export_bytes=store.config.max_export_bytes,max_memory_bytes=store.config.max_memory_bytes):pass
    from benchmarks.checkpoints import digest
    payload=dict(protocol=PROTOCOL,files=files)
    encoded=canonical(dict(schema=1,payload=payload,sha256=digest(payload)))+b'\n'
    total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())+len(encoded)
    if total>store.config.max_export_bytes:raise ValueError('complete snapshot export exceeds byte limit')
    temporary=root/('.publication-'+uuid.uuid4().hex)
    durable_file(temporary,encoded);fault('before_manifest')
    # Atomic, no-overwrite commit marker; an interrupted temp file is not valid.
    if (root/'manifest.json').exists():raise FileExistsError('snapshot manifest already exists')
    # This writer exclusively created root; rename atomically removes the temp.
    os.replace(temporary,root/'manifest.json');sync_directory(root)
    fault('manifest_published')
    return root


def _read(root,files,*,max_export_bytes=LIMIT,max_memory_bytes=LIMIT):
    expected={key+'.npy' for key in ARRAY_DTYPES}|{'mechanism.json','summary/trajectory.json','summary/trajectory.csv','summary/manifest.json'}
    if set(files)!=expected:raise ValueError('invalid mechanistic file inventory')
    if type(max_export_bytes) is not int or max_export_bytes<1:raise ValueError('invalid reader byte limit')
    total=sum((root/name).stat().st_size for name in files)+((root/'manifest.json').stat().st_size if (root/'manifest.json').exists() else 0)
    if total>max_export_bytes:raise ValueError('snapshot exceeds reader byte limit')
    current=_inventory(root);current.pop('manifest.json',None)
    if current!=files:raise ValueError('snapshot file checksum/inventory mismatch')
    if (root/'mechanism.json').stat().st_size>METADATA_LIMIT:raise ValueError('oversized metadata')
    m=json.loads((root/'mechanism.json').read_text(encoding='utf-8'),object_pairs_hook=unique_pairs)
    required={'protocol','source_hashes','limits','estimates','backend','observations','nodes','classes','particles','scheduled_iterations','iterations','particle_order','node_order','arrays'}
    if set(m)!=required or m['protocol']!=PROTOCOL:raise ValueError('invalid mechanistic metadata schema')
    config=SnapshotConfig(**m['limits'])
    if total>config.max_export_bytes:
        raise ValueError('snapshot export budget exceeded')
    n,c,p,s=m['nodes'],m['classes'],m['particles'],m['observations']
    if (any(type(v) is not int or v<1 for v in (n,c,p,s)) or c<2 or c>n or p>n or m['backend'] not in ('numpy','numba','cython')):
        raise ValueError('invalid bounded snapshot dimensions')
    if any(not isinstance(m[key],list) or any(type(i) is not int for i in m[key]) for key in ('iterations','scheduled_iterations')):
        raise ValueError('snapshot iteration lists must contain Python integers')
    planned=estimates(n,c,p,len(m['scheduled_iterations']),m['backend'])
    check_budget(config,planned)
    if type(max_memory_bytes) is not int or max_memory_bytes<1 or planned['estimated_memory_bytes']>max_memory_bytes:
        raise ValueError('snapshot exceeds reader memory estimate limit')
    summary_bytes=sum((root/name).stat().st_size for name in files if name.startswith('summary/'))
    if summary_bytes>len(m['scheduled_iterations'])*(128*n+16384)+METADATA_LIMIT:
        raise ValueError('summary exceeds bounded JSON/CSV workspace')
    summary=read_output(root/'summary');meta=summary['metadata'];rows=summary['observations']
    if (any(type(v) is not int or v<1 for v in (n,c,p,s)) or (n,c,p)!=(meta['nodes'],meta['classes'],meta['particles'])
            or m['backend']!=meta['backend'] or s!=len(rows) or m['iterations']!=[r['iteration'] for r in rows]):
        raise ValueError('snapshot dimensions/backend/iteration order differs from summary')
    from .convergence_trajectories import schedule_for
    schedule=schedule_for(meta['parameters']['max_iter'],meta['milestones'],meta['include_checks'],meta['budget']['max_observations'])
    estimate=estimates(n,c,p,len(schedule),m['backend'])
    if (m['scheduled_iterations']!=schedule or m['estimates']!=estimate
            or any(type(v) is not int for v in m['estimates'].values())):raise ValueError('snapshot preflight estimate mismatch')
    check_budget(config,estimate)
    from .trajectory_io import HASH
    if (set(m['source_hashes'])!={'experiments/mechanistic_snapshots.py'} or any(not isinstance(h,str) or not HASH.fullmatch(h) for h in m['source_hashes'].values())
            or m['particle_order']!='ascending labeled node index; ids are original particle indices' or m['node_order']!='graph row index; no reordering'):
        raise ValueError('missing/inconsistent snapshot source or ordering contract')
    shapes=dict(iterations=(s,),dominance=(s,n,c),positions=(s,p),strengths=(s,p),node_ids=(n,),particle_ids=(p,),particle_home_nodes=(p,),particle_classes=(p,))
    if set(m['arrays'])!=set(shapes):raise ValueError('snapshot array inventory mismatch')
    arrays={}
    try:
        for key,shape in shapes.items():
            path=root/(key+'.npy');desc=m['arrays'][key]
            if (not isinstance(desc,dict) or set(desc)!={'shape','dtype','order','array_sha256','file_bytes'}
                    or not isinstance(desc['shape'],list) or any(type(x) is not int for x in desc['shape'])
                    or type(desc['file_bytes']) is not int):raise ValueError('invalid array descriptor schema')
            with path.open('rb') as f:
                if np.lib.format.read_magic(f)!=(1,0):raise ValueError('only NPY v1.0 is supported')
                disk_shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
                if disk_shape!=shape or fortran or dtype.str!=ARRAY_DTYPES[key] or f.tell()+int(np.prod(shape))*dtype.itemsize!=path.stat().st_size:
                    raise ValueError('invalid NPY header/dimensions/layout/size')
            a=np.load(path,allow_pickle=False,mmap_mode='r')
            arrays[key]=a
            if (a.shape!=shape or a.dtype.str!=ARRAY_DTYPES[key] or not a.flags.c_contiguous
                    or desc!=dict(shape=list(shape),dtype=a.dtype.str,order='C',array_sha256=_array_digest(a),file_bytes=path.stat().st_size)
                    or a.offset+a.nbytes!=path.stat().st_size):
                raise ValueError('invalid snapshot NPY header, descriptor, array checksum or trailing bytes')
        if not np.array_equal(arrays['iterations'],m['iterations']) or not np.array_equal(arrays['node_ids'],np.arange(n)) or not np.array_equal(arrays['particle_ids'],np.arange(p)):
            raise ValueError('node/particle identity mismatch')
        homes,classes=arrays['particle_home_nodes'],arrays['particle_classes']
        if np.any(homes<0) or np.any(homes>=n) or not np.all(homes[1:]>homes[:-1]) or np.any(classes<0) or np.any(classes>=c):
            raise ValueError('invalid particle home/class mapping')
        labels=np.full(n,-1,dtype=np.int64);labels[homes]=np.array(meta['class_index_to_label'])[classes]
        if array_hash(labels)!=meta['fixture_hashes']['labels']:raise ValueError('particle mapping differs from input labels')
        for i,row in enumerate(rows):
            dom,pos,strength=arrays['dominance'][i],arrays['positions'][i],arrays['strengths'][i]
            if (not np.isfinite(dom).all() or not np.isfinite(strength).all() or np.any(dom< -1e-12) or np.any(dom>1+1e-12)
                    or np.any(strength< -1e-12) or np.any(strength>1+1e-12) or np.any(pos<0) or np.any(pos>=n)
                    or not np.allclose(dom.sum(axis=1),1,rtol=0,atol=1e-10)):
                raise ValueError('invalid snapshot dominance/particle state')
            for key,value in [('dominance',dom),('positions',pos),('strengths',strength)]:
                if _array_digest(value)!=row['state_hashes'][key]:raise ValueError('binary state differs from trajectory fingerprint')
            if not np.array_equal(np.argmax(dom,axis=1),row['assignments']) or not np.array_equal(dom[homes],np.eye(c)[classes]):
                raise ValueError('dominance assignments/labeled anchors differ from summary')
            if i==0 and row['iteration']==0 and (not np.array_equal(pos,homes) or not np.array_equal(strength,np.ones(p))):
                raise ValueError('snapshot initial particle order/strength mismatch')
        return SnapshotBundle(metadata=m,summary=summary,arrays=arrays)
    except BaseException:
        for value in arrays.values():value._mmap.close()
        raise


def read_snapshots(directory,*,max_export_bytes=LIMIT,max_memory_bytes=LIMIT):
    root=Path(directory)
    if root.is_symlink():raise ValueError('snapshot root cannot be a symlink')
    if (root/'manifest.json').stat().st_size>METADATA_LIMIT:raise ValueError('oversized completion manifest')
    published=read_record(root/'manifest.json')
    if set(published)!={'protocol','files'} or published['protocol']!=PROTOCOL:raise ValueError('not a completed mechanistic bundle')
    return _read(root,published['files'],max_export_bytes=max_export_bytes,max_memory_bytes=max_memory_bytes)
