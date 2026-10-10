"""Stream native W/H/RNG checkpoints over SSH; retain verified milestones remotely.

Only pointers live on the compute server. A two-phase, SHA-checked receiver commits
complete bytes atomically. A failed upload leaves the previous checkpoint usable.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import uuid

import torch

from . import checkpoint, milestones
from .prepare import digest


# Standard-library-only receiver: never loads tensors or a model on the archive host.
RECEIVER = r'''
import fcntl,hashlib,json,os,shutil,sys,tempfile
from pathlib import Path
spec=json.loads(sys.argv[1]); root=Path(spec['root'])
assert root.is_absolute() and not root.is_symlink(), 'ARCHIVE_ROOT'
root.mkdir(parents=True,exist_ok=True)
def sync(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def atomic(p,value):
 with tempfile.NamedTemporaryFile(dir=p.parent,mode='w',delete=False) as f:
  json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno());tmp=f.name
 os.replace(tmp,p);sync(p.parent)
def publish(destination,ref):
 rolling=root/'checkpoint';rolling.mkdir(exist_ok=True)
 target=rolling/ref['file']
 if not target.exists():os.link(destination/ref['file'],target)
 atomic(rolling/'latest.json',ref)
 for old in rolling.glob('batch-??-????????????????.pt'):
  if old!=target:old.unlink()
 sync(rolling)
with (root/'.archive.lock').open('a+b') as guard:
 fcntl.flock(guard,fcntl.LOCK_EX)
 pointer=root/'checkpoint/latest.json'
 previous=json.loads(pointer.read_text()) if pointer.exists() else None
 operation=spec['operation']
 if operation=='status':
  # Recover a commit interrupted between milestone rename and rolling publication.
  refs=[(p,json.loads((p/'latest.json').read_text())) for p in (root/'milestones').glob('edit-*')]
  if refs:
   p,ref=max(refs,key=lambda item:item[1]['batch'])
   assert ref['identity_sha256']==spec['identity_sha256'], 'ARCHIVE_IDENTITY_MISMATCH'
   assert sha(p/ref['file'])==ref['sha256'], 'ARCHIVE_PAYLOAD_SHA_MISMATCH'
   if previous!=ref:publish(p,ref)
   previous=ref
  print(json.dumps({'ref':previous,'milestones':[ref for p,ref in refs]}))
 else:
  stage=root/spec['stage']
  assert stage.parent==root and stage.name.startswith('.incoming-')
  if operation=='prepare':
   assert previous==spec['previous'], 'ARCHIVE_PREVIOUS_MISMATCH_RESUME_REQUIRED'
   assert shutil.disk_usage(root).free>=spec['reserve_bytes']+spec['estimated_bytes'], 'ARCHIVE_STORAGE'
   stage.mkdir(exist_ok=False)
   print(json.dumps({'stage':str(stage)}))
  elif operation=='receive':
   h=hashlib.sha256();size=0
   with (stage/'payload.pt').open('xb') as f:
    for chunk in iter(lambda:sys.stdin.buffer.read(8*1024*1024),b''):
     f.write(chunk);h.update(chunk);size+=len(chunk)
    f.flush();os.fsync(f.fileno())
   print(json.dumps({'sha256':h.hexdigest(),'bytes':size}))
  elif operation=='commit':
   ref=spec['ref'];filename=ref['file'];payload=stage/'payload.pt'
   assert previous==spec['previous'], 'ARCHIVE_PREVIOUS_MISMATCH_RESUME_REQUIRED'
   assert Path(filename).name==filename and ref['batch'] in (0,5,10,15,20)
   assert payload.is_file() and not payload.is_symlink() and payload.stat().st_size==spec['bytes']
   assert sha(payload)==ref['sha256'], 'ARCHIVE_PAYLOAD_SHA_MISMATCH'
   os.rename(payload,stage/filename);atomic(stage/'latest.json',ref);sync(stage)
   parent=root/'milestones';parent.mkdir(exist_ok=True)
   destination=parent/f"edit-{ref['batch']*100:04d}"
   assert not destination.exists(), 'ARCHIVE_ALREADY_EXISTS_RESUME_REQUIRED'
   os.rename(stage,destination);sync(parent)
   publish(destination,ref)
   print(json.dumps({'directory':str(destination),'sha256':ref['sha256'],'bytes':spec['bytes'],'ref':ref}))
  else:raise ValueError('UNKNOWN_ARCHIVE_OPERATION')
'''


def _command(archive, spec):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', archive['host'],
        shlex.join([archive['python'], '-c', RECEIVER, json.dumps(spec)])]


def _remote(archive, spec):
    result = subprocess.run(_command(archive, spec), text=True, capture_output=True, timeout=900)
    if result.returncode:
        raise RuntimeError('REMOTE_CHECKPOINT_PREVIOUS_RETAINED: ' + result.stderr[-1000:])
    return json.loads(result.stdout)


class _HashWriter:
    def __init__(self, stream):
        self.stream=stream;self.hash=hashlib.sha256();self.size=0

    def write(self, data):
        written=self.stream.write(data)
        if written != len(data):raise OSError('CHECKPOINT_SHORT_WRITE')
        self.hash.update(data);self.size+=written
        return written

    def flush(self):
        self.stream.flush()


def _stream(archive, spec, payload):
    # stdout is a small receipt. stderr is a temporary file to avoid pipe deadlock.
    with tempfile.TemporaryFile() as errors:
        proc=subprocess.Popen(_command(archive,spec),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors)
        try:
            writer=_HashWriter(proc.stdin)
            torch.save(payload,writer);writer.flush();proc.stdin.close();proc.stdin=None
            stdout,_=proc.communicate(timeout=1800)
            errors.seek(0)
            if proc.returncode:raise RuntimeError(errors.read().decode(errors='replace')[-1000:])
            received=json.loads(stdout)
            expected=dict(sha256=writer.hash.hexdigest(),bytes=writer.size)
            if received!=expected:raise ValueError('ARCHIVE_STREAM_SHA_MISMATCH')
            return expected
        finally:
            if proc.poll() is None:proc.kill();proc.wait()


def _write_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,mode='w',delete=False) as stream:
        temporary=Path(stream.name)
        json.dump(value,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
    try:
        os.replace(temporary,path);checkpoint._sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _mirror(output, archive, ref):
    info=dict(host=archive['host'],directory=archive['root']+'/milestones/'+f"edit-{ref['batch']*100:04d}",ref=ref)
    for folder in [Path(output)/'checkpoint',Path(output)/'milestones'/f"edit-{ref['batch']*100:04d}"]:
        _write_atomic(folder/'remote.json',info)
        _write_atomic(folder/'latest.json',ref)


def save(output, *, archive, lock_path, reserve_bytes=0, **kwargs):
    output,lock_path=Path(output),Path(lock_path)
    output.mkdir(parents=True,exist_ok=True);lock_path.parent.mkdir(parents=True,exist_ok=True)
    if lock_path.is_symlink():raise ValueError('CHECKPOINT_LOCK_SYMLINK')
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if shutil.disk_usage(output).free<reserve_bytes:raise ValueError('CHECKPOINT_METADATA_STORAGE')
        latest=output/'checkpoint/latest.json'
        previous=json.loads(latest.read_text()) if latest.exists() else None
        checkpoint.validate_sequence(previous,kwargs['batch'],5,kwargs['identity'])
        payload=checkpoint.make_payload(interval=5,**kwargs)
        estimated=sum(t.numel()*t.element_size() for group in ['weights','cache_c'] for t in payload[group].values())+32*2**20
        spec=dict(root=archive['root'],stage='.incoming-'+uuid.uuid4().hex,previous=previous,
            reserve_bytes=archive['reserve_bytes'],estimated_bytes=estimated)
        _remote(archive,dict(spec,operation='prepare'))
        transferred=_stream(archive,dict(spec,operation='receive'),payload)
        ref=dict(batch=kwargs['batch'],file=f"batch-{kwargs['batch']:02d}-{transferred['sha256'][:16]}.pt",
            sha256=transferred['sha256'],final_W20=kwargs['batch']==20,identity_sha256=digest(kwargs['identity']))
        receipt=_remote(archive,dict(spec,operation='commit',ref=ref,**transferred))
        if receipt['ref']!=ref or receipt['sha256']!=ref['sha256']:raise ValueError('ARCHIVE_ACK_MISMATCH_RESUME_REQUIRED')
        _mirror(output,archive,ref)
        return dict(latest=ref,milestone=dict(host=archive['host'],**receipt) if ref['batch'] in milestones.MILESTONES else None)


def _fetch(archive, ref, folder):
    subprocess.run(['rsync','--protect-args','-t','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',
        '--',archive['host']+':'+archive['root']+'/checkpoint/'+ref['file'],str(folder)+'/'],
        check=True,timeout=1800,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)


def load(output, identity, archive):
    """Fetch one verified checkpoint temporarily for resume, then reclaim local bytes."""
    checkpoint.validate_identity(identity)
    status=_remote(archive,dict(root=archive['root'],operation='status',identity_sha256=digest(identity)))
    ref=status['ref']
    if not ref or ref['identity_sha256']!=digest(identity):raise ValueError('ARCHIVE_IDENTITY_MISMATCH')
    if Path(ref['file']).name!=ref['file']:raise ValueError('INVALID_CHECKPOINT_MEMBER')
    with tempfile.TemporaryDirectory(dir=output,prefix='.resume-') as temporary:
        folder=Path(temporary);_fetch(archive,ref,folder);_write_atomic(folder/'latest.json',ref)
        state=checkpoint.load(folder,identity)
    for milestone in sorted(status['milestones'],key=lambda r:r['batch']):_mirror(output,archive,milestone)
    return state
