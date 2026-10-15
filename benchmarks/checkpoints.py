"""Durable, append-only benchmark records. No PCC state or loop checkpoints."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import uuid


class CheckpointError(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def identifier(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{32}',value):
        raise CheckpointError('invalid record identifier')
    return value


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_record(path):
    try:
        if Path(path).is_symlink():
            raise ValueError('symlink refused')
        data = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if set(data) != {'schema', 'payload', 'sha256'} or data['schema'] != 1:
            raise ValueError('invalid envelope')
        if digest(data['payload']) != data['sha256']:
            raise ValueError('checksum mismatch')
        return data['payload']
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise CheckpointError(f'invalid record {path}: {exc}') from exc


def sync_directory(path):
    # Windows cannot portably open/fsync a directory. All files are fsynced,
    # publication is exclusive, and every record is revalidated after restart.
    if os.name != 'nt':
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def durable_file(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def publish_record(path, payload, staging, fault=lambda point: None):
    """Hard-link publication refuses an existing destination on Windows/POSIX."""
    envelope = dict(schema=1, payload=payload, sha256=digest(payload))
    temporary = Path(staging) / f'blob-{uuid.uuid4().hex}.json'
    durable_file(temporary, canonical(envelope)+b'\n')
    fault('record_flushed')
    os.link(temporary, path)  # Never os.replace: destination must be absent.
    sync_directory(Path(path).parent)
    fault('record_published')


def file_inventory(directory):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        if path.is_symlink():
            raise CheckpointError(f'symlink refused: {path}')
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def process_alive(pid):
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:  # nonexistent process
                return False
            raise CheckpointError('cannot prove lock owner is inactive')
        try:
            code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)):
                raise CheckpointError('cannot inspect lock owner')
            return code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        raise CheckpointError('cannot prove lock owner is inactive')


class Store:
    def __init__(self, root, fault=None):
        self.root = Path(root)
        self.fault = fault or (lambda point: None)
        self.token = None
        self._events_cache = None

    @classmethod
    def create(cls, root, plan, fault=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        for name in ('staging', 'events', 'results', 'fixtures', 'sessions', 'locks', 'summaries'):
            (root/name).mkdir()
        store = cls(root, fault)
        with store.lock():
            store.write(root/'plan.json', plan)
            store.write(root/'plan_identity.json',dict(campaign_id=plan['campaign_id'],plan_sha256=digest(plan)))
        return store

    def write(self, path, payload, fault=None):
        self.require_lock()
        publish_record(path, payload, self.root/'staging', fault or (lambda point: None))

    def require_lock(self):
        if self.token is None or read_record(self.root/'writer.lock')['token'] != self.token:
            raise CheckpointError('exclusive writer lock required')

    @contextmanager
    def lock(self):
        if self.token is not None:
            raise CheckpointError('nested writer lock refused')
        token = uuid.uuid4().hex
        owner = dict(token=token, pid=os.getpid(), host=socket.gethostname(), started_utc=utc())
        path = self.root/'writer.lock'
        envelope = dict(schema=1, payload=owner, sha256=digest(owner))
        durable_file(path, canonical(envelope)+b'\n')  # exclusive even during partial writes
        sync_directory(self.root)
        self.token = token
        self._events_cache = None
        try:
            self.write(self.root/'locks'/f'{token}.json', owner)
            yield
        finally:
            # Never release another writer's lock; crash leaves evidence behind.
            self.require_lock()
            path.unlink()
            sync_directory(self.root)
            self.token = None
            self._events_cache = None

    def recover_lock(self, token, reason):
        owner = read_record(self.root/'writer.lock')
        identifier(owner['token'])
        if type(owner['pid']) is not int or owner['pid']<1:
            raise CheckpointError('invalid lock PID')
        if not reason.strip() or owner['token'] != token or owner['host'] != socket.gethostname():
            raise CheckpointError('exact token, local owner and explicit reason required')
        if process_alive(owner['pid']):
            raise CheckpointError('live lock owner; recovery refused')
        target = self.root/'locks'/f'stale-{token}.json'
        if target.exists():
            raise CheckpointError('already recovered token')
        os.rename(self.root/'writer.lock', target)
        sync_directory(self.root)
        with self.lock():
            self.event('lock_recovered', token=token, reason=reason)

    def plan(self):
        plan = read_record(self.root/'plan.json')
        pin=read_record(self.root/'plan_identity.json')
        if pin!=dict(campaign_id=plan['campaign_id'],plan_sha256=digest(plan)):
            raise CheckpointError('immutable campaign plan changed')
        identifier(plan['campaign_id'])
        ids = [t['task_id'] for t in plan['tasks']]
        if len(ids) != len(set(ids)) or any(i != f't{n:04d}' for n,i in enumerate(ids)):
            raise CheckpointError('invalid task inventory/order')
        return plan

    def events(self, force=False):
        # Cache only while owning the exclusive lock; strict audits reread disk.
        if self.token is not None and self._events_cache is not None and not force:
            return list(self._events_cache)
        paths = sorted((self.root/'events').iterdir())
        records, previous = [], None
        for index, path in enumerate(paths):
            if path.name != f'{index:06d}.json':
                raise CheckpointError('event sequence gap or unknown file')
            event = read_record(path)
            if event['sequence'] != index or event['previous_sha256'] != previous:
                raise CheckpointError('broken event chain')
            previous = digest(event)
            records.append(event)
        if self.token is not None:
            self._events_cache = records
        return list(records)

    def event(self, kind, **fields):
        self.require_lock()
        events = self.events()
        payload = dict(sequence=len(events), previous_sha256=digest(events[-1]) if events else None,
                       kind=kind, timestamp_utc=utc(), **fields)
        self.write(self.root/'events'/f'{len(events):06d}.json', payload)
        self._events_cache = events + [payload]
        return payload

    def audit(self, validate_result, tolerate_summaries=False):
        """Read-only reconciliation: no fits, warm-ups, writes or automatic retries."""
        plan = self.plan()
        plan_sha=digest(plan)
        for path in (self.root/'locks').iterdir():
            owner=read_record(path)
            identifier(owner['token'])
            if (path.name not in (f"{owner['token']}.json",f"stale-{owner['token']}.json")
                    or type(owner['pid']) is not int or owner['pid']<1):
                raise CheckpointError('invalid archived lock')
        tasks = {t['task_id']:t for t in plan['tasks']}
        starts, last, retries, completions, sessions = {}, {}, {}, {}, {}
        ended, summary_events, retired = set(), [], {}
        for event in self.events(force=True):
            kind = event['kind']
            if kind in ('started','retry_authorized','completed','reconciled','failed') and event.get('task_id') not in tasks:
                raise CheckpointError('unknown task in event ledger')
            if kind == 'session_start':
                sid = event['session_id']
                identifier(sid)
                if sid in sessions:
                    raise CheckpointError('duplicate session')
                session = read_record(self.root/'sessions'/sid/'session.json')
                if session['session_id'] != sid or session['identity'] != plan['identity']:
                    raise CheckpointError('invalid session identity')
                sessions[sid] = session
            elif kind == 'started':
                tid, aid = event['task_id'], event['attempt_id']
                identifier(aid)
                if tid not in tasks or aid in starts or event['session_id'] not in sessions or event['session_id'] in ended:
                    raise CheckpointError('unknown task/session or duplicate attempt')
                if tid in last and retries.get(tid) != last[tid]:
                    raise CheckpointError('attempt repeated without explicit retry')
                if tid in completions:
                    raise CheckpointError('completed task repeated')
                warm_id=tasks[tid].get('warmup_task_id',tid)
                expected_warms=[f"sessions/{event['session_id']}/warmups/{warm_id}-{n}.json" for n in range(2)]
                if event['warmup_files']!=expected_warms:
                    raise CheckpointError('started event warm-up identity mismatch')
                for name in expected_warms:
                    warm=read_record(self.root/name)
                    if warm['session_id']!=event['session_id']:
                        raise CheckpointError('started event missing/invalid warm-up')
                retries.pop(tid, None)
                starts[aid] = event
                last[tid] = aid
            elif kind == 'retry_authorized':
                tid = event['task_id']
                if tid in completions or tid in retries or last.get(tid) != event['attempt_id'] or not event['reason'].strip():
                    raise CheckpointError('invalid retry authorization')
                retries[tid] = event['attempt_id']
            elif kind in ('completed', 'reconciled'):
                tid, aid = event['task_id'], event['attempt_id']
                if last.get(tid) != aid or tid in completions:
                    raise CheckpointError('invalid completion event')
                completions[tid] = aid
            elif kind == 'failed':
                if event['attempt_id'] not in starts or last.get(event['task_id']) != event['attempt_id']:
                    raise CheckpointError('invalid failed attempt')
            elif kind == 'fixture_retry':
                for name, hashes in event['preserved'].items():
                    if Path(name).name!=name or not name.startswith('fixture-'):
                        raise CheckpointError('invalid preserved fixture path')
                    if file_inventory(self.root/'staging'/name) != hashes:
                        raise CheckpointError('preserved partial fixture changed')
            elif kind == 'summary':
                identifier(event['generation'])
                summary_events.append(event)
            elif kind == 'summary_retired':
                gen=identifier(event['generation'])
                if gen in retired or not event['reason'].strip() or gen not in [e['generation'] for e in summary_events]:
                    raise CheckpointError('invalid summary recovery event')
                if file_inventory(self.root/'summaries'/gen)!=event['preserved']:
                    raise CheckpointError('preserved invalid summary changed')
                retired[gen]=event
            elif kind == 'session_end':
                if event['session_id'] not in sessions or event['session_id'] in ended:
                    raise CheckpointError('unknown ended session')
                end=read_record(self.root/'sessions'/event['session_id']/'end.json')
                if end['session_id']!=event['session_id']:
                    raise CheckpointError('ended session identity mismatch')
                ended.add(event['session_id'])
            elif kind != 'lock_recovered':
                raise CheckpointError(f'unknown event kind: {kind}')
        results = {}
        for path in sorted((self.root/'results').iterdir()):
            tid = path.stem
            if path.name != f'{tid}.json' or tid not in tasks:
                raise CheckpointError('unknown/duplicate result file')
            result = read_record(path)
            aid = result['attempt_id']
            if result['task_id'] != tid or last.get(tid) != aid or aid not in starts:
                raise CheckpointError('result attempt does not match ledger')
            if result['session_id'] != starts[aid]['session_id'] or result['plan_sha256'] != plan_sha:
                raise CheckpointError('result session/plan mismatch')
            if tid in retries:
                raise CheckpointError('completed result has retry authorization')
            if result['warmup_files'] != starts[aid]['warmup_files']:
                raise CheckpointError('warm-up references changed')
            validate_result(result, tasks[tid], plan)
            results[tid] = result
        if any(tid not in results or results[tid]['attempt_id'] != aid for tid,aid in completions.items()):
            raise CheckpointError('completion without valid result')
        summary_errors=[]
        for event in summary_events:
            gen=event['generation']
            if gen in retired:
                continue
            try:
                summary=read_record(self.root/'summaries'/gen/'summary.json')
                checksums=read_record(self.root/'summaries'/gen/'checksums.json')
                actual=file_inventory(self.root/'summaries'/gen)
                actual.pop('checksums.json',None)
                if checksums!=actual or set(actual)!={'summary.json'} or digest(summary)!=event['summary_sha256']:
                    raise CheckpointError('summary checksum inventory mismatch')
                if summary['plan_sha256']!=plan_sha or summary['measurements']!=len(summary['rows']):
                    raise CheckpointError('summary plan/inventory mismatch')
                if (summary['cap_hits']!=sum(r['reached_max_iter'] for r in summary['rows'])
                        or summary['censored']!=sum(r['stop_reason']=='max_iter' for r in summary['rows'])
                        or summary['status']!=('complete' if len(summary['rows'])==len(tasks) else 'partial')):
                    raise CheckpointError('summary counts/status mismatch')
                ids=[r['task_id'] for r in summary['rows']]
                if len(ids)!=len(set(ids)) or any(tid not in results for tid in ids):
                    raise CheckpointError('summary references missing/duplicate results')
                if any(row!=results[row['task_id']]['row'] for row in summary['rows']):
                    raise CheckpointError('summary differs from authoritative results')
                if 'repetitions' in plan:
                    extra=repetition_summary(summary['rows'],plan)
                    if any(summary.get(k)!=v for k,v in extra.items()):
                        raise CheckpointError('repetition/stage summary mismatch')
            except (CheckpointError,KeyError,TypeError,ValueError) as exc:
                if not tolerate_summaries:
                    raise CheckpointError(f'invalid derived summary {gen}; explicit summarize --reason required') from exc
                summary_errors.append(dict(generation=gen,error=repr(exc)))
        ambiguous = [tid for tid in tasks if tid in last and tid not in results and tid not in retries]
        return dict(plan=plan, results=results, ambiguous=ambiguous, authorized_retries=retries,
                    last_attempts=last, reconcile=[tid for tid in results if tid not in completions],
                    pending=[tid for tid in tasks if tid not in results], sessions=sessions,
                    summary_errors=summary_errors)

    def retire_summaries(self, state, reason):
        if not reason.strip():
            raise CheckpointError('explicit summary repair reason required')
        for error in state['summary_errors']:
            self.event('summary_retired',generation=error['generation'],reason=reason,error=error['error'],
                       preserved=file_inventory(self.root/'summaries'/error['generation']))

    def reconcile(self, audit):
        for tid in audit['reconcile']:
            self.event('reconciled', task_id=tid, attempt_id=audit['results'][tid]['attempt_id'])

    def authorize_retry(self, tid, reason, validate_result):
        state = self.audit(validate_result)
        if tid not in state['ambiguous'] or not reason.strip():
            raise CheckpointError('only ambiguous tasks may be explicitly retried, with a reason')
        self.event('retry_authorized', task_id=tid, attempt_id=state['last_attempts'][tid], reason=reason)

    def result(self, payload):
        self.require_lock()
        self.fault('before_result')
        self.write(self.root/'results'/f"{payload['task_id']}.json", payload,
                   fault=lambda p:self.fault('result_'+p))
        self.fault('after_result')
        self.event('completed', task_id=payload['task_id'], attempt_id=payload['attempt_id'])
        self.fault('after_completed')

    def snapshot(self, state):
        """One observation per task: raw values, no fabricated timing IQRs."""
        self.require_lock()
        if state['ambiguous']:
            raise CheckpointError('cannot summarize ambiguous tasks')
        generation = uuid.uuid4().hex
        directory = self.root/'summaries'/generation
        directory.mkdir()
        rows = [state['results'][t['task_id']]['row'] for t in state['plan']['tasks']
                if t['task_id'] in state['results']]
        summary = dict(generation=generation, plan_sha256=digest(state['plan']), timestamp_utc=utc(),
                       status='complete' if not state['pending'] else 'partial', measurements=len(rows),
                       cap_hits=sum(r['reached_max_iter'] for r in rows),
                       censored=sum(r['stop_reason']=='max_iter' for r in rows), rows=rows)
        if 'repetitions' in state['plan']:
            summary.update(repetition_summary(rows,state['plan']))
        self.write(directory/'summary.json', summary)
        self.fault('summary_written')
        self.write(directory/'checksums.json', file_inventory(directory))
        self.fault('summary_sealed')
        self.event('summary', generation=generation, summary_sha256=digest(summary))
        return summary


def repetition_summary(rows, plan):
    from .runner import summarize, TIMINGS
    groups=summarize(rows)
    for group in groups:
        group['expected_repetitions']=plan['repetitions']
        group['complete']=group['repetitions']==plan['repetitions']
        if group['repetitions']==1:
            for field in (*TIMINGS,'propagation_s_per_iteration','effective_iterations'):
                group[field].update(q1=None,q3=None,iqr=None)
    stages={}
    for seed in range(10,20):
        case_ids={c['name']+'-labels'+str(seed) for c in plan['cases'] if c['label_seed']==seed}
        expected=sum(t['case_id'] in case_ids for t in plan['tasks'])
        count=sum(r['case_id'] in case_ids for r in rows)
        stages[str(seed)]=dict(expected=expected,completed=count,
            status='complete' if count==expected else 'partial' if count else 'not_started')
    return dict(groups=groups,stages=stages)
