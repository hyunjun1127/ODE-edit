"""Finite scalar serializers and fresh capacity guards; no checkpoint storage.

Bounds are conservative protocol maxima, not measured output sizes or a shared
filesystem reservation. Exceeding a bound is a typed technical failure, never
permission to omit fields, reduce evaluation, delete originals, or retry.
"""
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from .common import TASK, require, member, sha

KiB=1024
MiB=1024**2
GiB=1024**3
ROW_BYTES=4096
CHUNK_ROWS=650
CHUNK_BYTES=3*MiB
CANDIDATE_BYTES=1024*KiB
STATIC_BYTES=512*KiB
RECEIPT_BYTES=256*KiB
REALIZATION_BYTES=8*MiB
COLLECTOR_BYTES=2*GiB
ERROR_RESERVE_BYTES=128*MiB
ATOMIC_RESERVE_BYTES=COLLECTOR_BYTES
BATCH_ATOMIC_RESERVE_BYTES=16*MiB
SOURCE_RESERVE_BYTES=256*MiB
CONSOLE_BYTES=16*MiB
CONSOLE_CHUNK_BYTES=256*KiB


def storage_plan(B=100,batches=20,arms=6):
    require((B,batches,arms)==(100,20,6),'FIXED_SIX_CELL_STORAGE_HORIZON')
    # Each arm: W0=40 chunks; every pre=2; nonmilestone post=2;
    # allseen posts at 5/10/15/20 contain 10/20/30/40 chunks.
    metric_chunks_each=40+20*2+16*2+10+20+30+40
    metric_rows_each=26000+20*1300+16*1300+6500+13000+19500+26000
    metrics=arms*metric_chunks_each*CHUNK_BYTES
    candidates=arms*batches*20*CANDIDATE_BYTES
    static=arms*batches*STATIC_BYTES
    # Sixteen independently bounded ordinary files plus one realization/batch.
    ordinary=arms*batches*16*RECEIPT_BYTES
    realization=arms*batches*REALIZATION_BYTES
    diagnostics=arms*batches*20*RECEIPT_BYTES
    runtime=arms*8*RECEIPT_BYTES
    console=7*(CONSOLE_BYTES+CONSOLE_CHUNK_BYTES)
    spool=6*256*MiB
    reserve=spool+metrics+candidates+static+ordinary+realization+diagnostics+runtime+console+COLLECTOR_BYTES+ERROR_RESERVE_BYTES+ATOMIC_RESERVE_BYTES+SOURCE_RESERVE_BYTES
    # Maximum milestone batch W20: 2 pre chunks +40 post chunks; fit streams,
    # ordinary receipts, realization and atomic work included before any fit.
    nextbatch=42*CHUNK_BYTES+20*CANDIDATE_BYTES+STATIC_BYTES+16*RECEIPT_BYTES+REALIZATION_BYTES+20*RECEIPT_BYTES+BATCH_ATOMIC_RESERVE_BYTES
    return dict(schema='INTERFERENCE_L1_BOUNDED_SERIALIZER_V1',task=TASK,
        B=B,batches=batches,arms=arms,reserve_bytes=reserve,next_batch_bytes=nextbatch,
        error_reserve_bytes=ERROR_RESERVE_BYTES,atomic_reserve_bytes=ATOMIC_RESERVE_BYTES,
        next_batch_atomic_reserve_bytes=BATCH_ATOMIC_RESERVE_BYTES,
        source_archive_reserve_bytes=SOURCE_RESERVE_BYTES,collector_reserve_bytes=COLLECTOR_BYTES,
        startup_free_bytes_min=reserve,first_W0_write_bytes=40*CHUNK_BYTES,
        metric_rows_each=metric_rows_each,metric_chunks_each=metric_chunks_each,
        maximum_candidate_records=arms*batches*20,maximum_static_price_records=arms*batches,
        maximum_record_bytes=dict(metric_row=ROW_BYTES,metric_chunk=CHUNK_BYTES,candidate=CANDIDATE_BYTES,
            entry_price=STATIC_BYTES,ordinary=RECEIPT_BYTES,realization=REALIZATION_BYTES),
        maximum_scalar_array_length=20000,maximum_dict_members=4096,maximum_nesting_depth=16,
        console_processes=7,maximum_console_bytes_per_process=CONSOLE_BYTES,maximum_console_chunk_bytes=CONSOLE_CHUNK_BYTES,
        console_error_print_reserve_bytes_per_process=CONSOLE_CHUNK_BYTES,
        console_error_print_reserve_bytes_total=7*CONSOLE_CHUNK_BYTES,
        console_bound_scope='Python stdout/stderr writes synchronously bounded; direct native FD output checked pre-batch only, not hard-reserved',
        calculated_parts_bytes=dict(wandb_spool=spool,metrics=metrics,candidates=candidates,static_prices=static,
            ordinary_receipts=ordinary,realizations=realization,diagnostic_events=diagnostics,runtime=runtime,console=console),
        assumptions='All mandatory fields retained; exceeding serializer maxima blocks, never silently drops data.',
        measured=False,shared_filesystem_reserved=False,quota='NOT_AVAILABLE_UNLESS_BOUND_BY_PREFLIGHT',
        shared_filesystem_warning='Free space is a fresh snapshot, not future guarantee or exclusive reservation; repeat prefit guard.',
        checkpoint_saved=False)


def _shape(value,depth=0):
    require(depth<=16,'SCALAR_SERIALIZER_MAX_DEPTH')
    if isinstance(value,dict):
        require(len(value)<=4096 and all(isinstance(k,str) for k in value),'SCALAR_SERIALIZER_DICT')
        for k,v in value.items():
            require(len(k.encode())<=256,'SCALAR_SERIALIZER_KEY_BYTES');_shape(v,depth+1)
    elif isinstance(value,(list,tuple)):
        require(len(value)<=20000,'SCALAR_SERIALIZER_ARRAY_LENGTH')
        for v in value:_shape(v,depth+1)
    elif isinstance(value,float):require(math.isfinite(value),'SCALAR_SERIALIZER_NONFINITE')
    elif isinstance(value,str):require(len(value.encode())<=256*KiB,'SCALAR_SERIALIZER_STRING_BYTES')
    else:require(value is None or type(value) in (int,bool),'SCALAR_ONLY_NO_TENSORS')


def encode(value,limit=RECEIPT_BYTES):
    value=_keys(value)
    _shape(value)
    data=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
    require(len(data)<=limit,'SCALAR_SERIALIZER_RECORD_BYTES')
    return data


def _keys(value):
    # Native owner-index dictionaries retain integer keys in RAM; JSON binding
    # has always used their string spelling. Reject collisions, not native rows.
    if isinstance(value,dict):
        require(all(type(k) in (str,int) for k in value),'SCALAR_SERIALIZER_DICT_KEY')
        require(len({str(k) for k in value})==len(value),'SCALAR_SERIALIZER_KEY_COLLISION')
        return {str(k):_keys(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_keys(v) for v in value]
    return value


def error_operands(value):
    """Error evidence only: explicit IEEE nonfinite tags, never metric repair."""
    if isinstance(value,float) and not math.isfinite(value):
        return dict(nonfinite='nan' if math.isnan(value) else '+inf' if value>0 else '-inf',
            original_type='IEEE754_float',not_a_finite_measurement=True)
    if isinstance(value,dict):return {str(k):error_operands(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [error_operands(v) for v in value]
    return value


def guard(folder,required_bytes,inodes=4):
    path=Path(folder).resolve()
    while not path.exists():path=path.parent
    v=os.statvfs(path);free=v.f_bavail*v.f_frsize
    require(free>=int(required_bytes),'RESOURCE_BLOCKED_STORAGE')
    require(v.f_favail>=int(inodes),'RESOURCE_BLOCKED_INODES')
    return dict(path=str(path),free_bytes=free,free_inodes=v.f_favail,required_bytes=int(required_bytes),
        measured_at_unix=time.time(),quota='NOT_AVAILABLE',quota_command_available=bool(shutil.which('quota')),
        shared_filesystem_reserved=False,quota_limit_not_inferred_from_free=True)


def _limit(path,value):
    name=Path(path).name
    if name.startswith('chunk-'):
        rows=value.get('rows');require(isinstance(rows,list) and len(rows)<=CHUNK_ROWS,'METRIC_CHUNK_ROWS')
        for row in rows:encode(row,ROW_BYTES)
        return CHUNK_BYTES
    if name=='realization.json':return REALIZATION_BYTES
    if name=='entry-price.json':return STATIC_BYTES
    return RECEIPT_BYTES


def write(path,value,limit=None):
    """Atomic create-once bounded JSON; valid existing exact bytes can be reused."""
    path=Path(path);data=encode(value,_limit(path,value) if limit is None else limit)
    write_bytes(path,data,limit=len(data))


def write_bytes(path,data,limit=RECEIPT_BYTES):
    path=Path(path);require(isinstance(data,bytes) and 0<len(data)<=limit,'BOUNDED_NONEMPTY_FILE_BYTES')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        require(path.is_file() and not path.is_symlink() and path.read_bytes()==data,'IMMUTABLE_RECEIPT_CONFLICT')
        return
    emergency=path.name in ('first-error.json','reducer-first-error.json','terminal.json','rollback.json','console-bound-failure.json',
        'SHARED_TECHNICAL_BLOCK.json','failure-inventory.json')
    guard(path.parent,len(data)+(0 if emergency else ERROR_RESERVE_BYTES))
    temporary=path.with_name(path.name+'.tmp')
    require(not temporary.exists(),'IMMUTABLE_TEMP_ALREADY_EXISTS')
    with temporary.open('xb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    require(temporary.stat().st_size==len(data),'ATOMIC_NONEMPTY_RECORD')
    os.link(temporary,path)
    temporary.unlink()


def console_files_guard(attempt,role):
    paths=sorted(Path(attempt).glob(role+'-*.out'))+sorted(Path(attempt).glob(role+'-*.err'))
    require(all(p.is_file() and not p.is_symlink() for p in paths),'UNSAFE_CONSOLE_FILE')
    require(sum(p.stat().st_size for p in paths)<=CONSOLE_BYTES,'RESOURCE_BLOCKED_STORAGE_CONSOLE_FILES')
    return dict(role=role,bytes=sum(p.stat().st_size for p in paths),limit=CONSOLE_BYTES,
        direct_native_FD_writes='Fresh file size checked before next batch; Python console calls bounded synchronously')


class ConsoleBudget:
    """Fail before oversized Python writes; never silently truncate and continue."""
    def __init__(self,path):
        self.path=Path(path);self.total=0;self.complete={'stdout':0,'stderr':0};self.counts={'stdout':0,'stderr':0}
        self.failed=False;self.original_stdout=sys.stdout;self.original_stderr=sys.stderr
    def install(self):
        sys.stdout=_ConsoleStream(self,self.original_stdout,'stdout')
        sys.stderr=_ConsoleStream(self,self.original_stderr,'stderr')
        return self
    def accept(self,stream,text):
        data=text.encode('utf-8');n=len(data)
        if self.failed:
            # First failure is already recorded in JSON; secondary Python
            # traceback printing gets a separate finite error reserve.
            if self.counts.get('error_reserve',0)+n>CONSOLE_CHUNK_BYTES:return False
            self.counts['error_reserve']=self.counts.get('error_reserve',0)+n
            return True
        if n>CONSOLE_CHUNK_BYTES or self.total+n>CONSOLE_BYTES:
            self.failed=True
            value=dict(task=TASK,status='RESOURCE_BLOCKED_STORAGE_CONSOLE_BOUND',accepted_bytes=self.total,
                rejected_chunk_bytes=n,rejected_chunk_sha256=hashlib.sha256(data).hexdigest(),
                last_complete_record_boundaries=dict(self.complete),console_limit=CONSOLE_BYTES,chunk_limit=CONSOLE_CHUNK_BYTES,
                original_raw_KEEP=True,rejected_chunk_persisted=False,no_silent_continuation=True,
                full_stdout_preservation_claim=False,automatic_retry=False,
                secondary_traceback_console='Finite256KiB error-print reserve after execution failure; fulltrace retained in first-error JSON if within its explicit bound')
            try:write(self.path,value)
            except BaseException as error:
                self.original_stderr.write('CONSOLE_BOUND_RECEIPT_NOT_VERIFIED: '+type(error).__name__+'\n')
            raise RuntimeError('RESOURCE_BLOCKED_STORAGE_CONSOLE_BOUND')
        previous=self.counts[stream];self.counts[stream]+=n;self.total+=n
        if '\n' in text:self.complete[stream]=previous+len(text[:text.rfind('\n')+1].encode('utf-8'))
        return True


class _ConsoleStream:
    def __init__(self,budget,original,name):self.budget,self.original,self.name=budget,original,name
    def write(self,text):
        if self.budget.accept(self.name,text):return self.original.write(text)
        return len(text)
    def flush(self):return self.original.flush()
    def __getattr__(self,name):return getattr(self.original,name)


class Events:
    """Entry price once, authoritative candidate once; no fit event-array copy."""
    def __init__(self,path,arm,batch):
        self.path=Path(path);self.arm=arm;self.batch=batch;self.count=0;self.other_count=0
        self.path.parent.mkdir(parents=True,exist_ok=True)
        require(not self.path.exists(),'CANDIDATE_STREAM_CREATE_ONCE')
        self.entry_path=self.path.parent/'entry-price.json';self.started=time.monotonic();self.io_seconds=0.
    def mark(self):return self.count
    def emit(self,event,payload):
        started=time.monotonic()
        row=dict(task=TASK,arm=self.arm,batch=self.batch,event=event,payload=payload)
        if event=='entry_price':
            require(not self.entry_path.exists(),'ENTRY_PRICE_ONCE');write(self.entry_path,row)
        elif event=='candidate':
            require(self.entry_path.exists() and self.count<20 and payload['candidate']==self.count,
                'AUTHORITATIVE_CANDIDATE_ORDER_ONCE')
            # Dynamic receipt must not repeat static L-by-B controller arrays.
            for name in ('controller','post_update_controller'):
                ctrl=payload.get(name,{})
                require(not any(k in ctrl for k in ('anchor_values','anchor_star','local_caps','computed_pi','effective_pi','raw_kappa')),
                    'CANDIDATE_STATIC_ARRAY_DUPLICATION')
            data=encode(row,CANDIDATE_BYTES)
            with self.path.open('ab' if self.count else 'xb') as f:
                f.write(data);f.flush()
            self.count+=1
        else:
            require(self.other_count<20,'DIAGNOSTIC_EVENT_COUNT_BOUND')
            data=encode(row,RECEIPT_BYTES)
            with (self.path.parent/'diagnostics.jsonl').open('ab') as f:f.write(data);f.flush()
            self.other_count+=1
        self.io_seconds+=time.monotonic()-started
    def __call__(self,event,payload=None):
        if isinstance(event,dict):self.emit('fit',event)
        else:self.emit(event,payload)
    def reference_since(self,mark):
        require(mark==0 and 1<=self.count<=20,'FULL_CANDIDATE_STREAM_REFERENCE')
        with self.path.open('rb') as f:os.fsync(f.fileno())
        data=self.path.read_bytes();require(data.count(b'\n')==self.count,'CANDIDATE_LINE_COUNT')
        return dict(path=str(self.path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
            line_count=self.count,first_candidate=0,last_candidate=self.count-1,
            entry_price=member(self.entry_path),authoritative=True,io_seconds=self.io_seconds)
    def stream_reference(self):return self.reference_since(0)
