"""Synchronous latest-only storage without a second full native CPU H copy.

Payload/hash format matches fe_original; storage byte handling only. The host
lock covers CPU transfer, hashing, serialization, fsync, replace and metadata.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import uuid
import numpy as np
import torch
from official.experiments.prepare import digest, read
from official.experiments.checkpoint import rng_snapshot
from official.runners.server1.common import member

def bytes_sha(array):
    view=memoryview(array).cast('B');h=hashlib.sha256()
    for start in range(0,len(view),8*1024**2):h.update(view[start:start+8*1024**2])
    return h.hexdigest()

def tensor_sha(value):
    value=value.detach().cpu().contiguous()
    return bytes_sha(value.reshape(-1).view(torch.uint8).numpy())

def state_digest(value):
    def canonical(v):
        if isinstance(v,torch.Tensor):return dict(dtype=str(v.dtype),shape=list(v.shape),sha256=tensor_sha(v))
        if isinstance(v,np.ndarray):
            return dict(dtype=str(v.dtype),shape=list(v.shape),sha256=bytes_sha(np.ascontiguousarray(v)))
        if isinstance(v,dict):return {str(k):canonical(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [canonical(x) for x in v]
        if isinstance(v,np.generic):return v.item()
        return v
    return digest(canonical(value))

def finite_tensor(value):
    flat=value.detach().reshape(-1)
    return all(torch.isfinite(flat[i:i+1024**2]).all().item() for i in range(0,flat.numel(),1024**2))

def remaining_storage(config,configs):
    """Already materialized own latest files consume disk, not future reserve twice.

    Call under the same host checkpoint lock. Only allocated, manifest-bound
    regular single-link latest payload bytes receive budget credit.
    """
    credited=0;seen=set()
    for c in configs:
        path=Path(c['output'])/'checkpoints/latest.pt';receipt=path.with_name('latest.json')
        if not path.exists() or not receipt.exists():continue
        st=path.lstat()
        if path.is_symlink() or not path.is_file() or st.st_nlink!=1 or (st.st_dev,st.st_ino) in seen:continue
        meta=read(receipt)
        if meta['identity']['config_sha256']!=c['config_sha256'] or meta['bytes']!=st.st_size:continue
        seen.add((st.st_dev,st.st_ino))
        credited+=min(st.st_blocks*512,st.st_size,config['checkpoint_payload_bound'])
    required=(config['planned_latest_count']+1)*config['checkpoint_payload_bound']+config['raw_margin_bytes']+config['reserve_bytes']-credited
    return dict(required_free_bytes=required,credited_existing_latest_bytes=credited,
        reserve_bytes=config['reserve_bytes'],raw_margin_bytes=config['raw_margin_bytes'],atomic_tmp=1)

def admit_remaining(config,configs,free_bytes):
    budget=remaining_storage(config,configs);budget['actual_free_bytes']=free_bytes
    if free_bytes<budget['required_free_bytes']:raise ValueError('STORAGE_REQUIRED')
    return budget

def save_latest(folder,weights,H,contexts,z_member,batch,identity,cursor,*,lock_path,reserve_bytes):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);lock_path=Path(lock_path)
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    if lock_path.is_symlink() or folder.stat().st_dev!=lock_path.parent.stat().st_dev:raise ValueError('HOST_LOCK_FILESYSTEM')
    if batch not in (5,10,15,20):raise ValueError('CHECKPOINT_MILESTONE')
    # Native author H must already be CPU; reject an accidental giant GPU->CPU transfer.
    if H.device.type!='cpu' or not H.is_contiguous():raise ValueError('NATIVE_HISTORY_CPU_CONTIGUOUS')
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        try:
            tmp=folder/'latest.pt.partial';dest=folder/'latest.pt'
            if tmp.exists() or tmp.is_symlink() or dest.is_symlink():raise ValueError('CHECKPOINT_TARGET_EXISTS_OR_ALIAS')
            estimated=sum(v.numel()*v.element_size() for v in weights.values())+H.numel()*H.element_size()+16*1024**2
            space=os.statvfs(folder)
            if space.f_bavail*space.f_frsize<reserve_bytes+estimated:raise ValueError('CHECKPOINT_STORAGE_ADMISSION')
            before=(H._version,{k:v._version for k,v in weights.items()})
            payload=dict(weights={k:v.detach().cpu() for k,v in weights.items()},cache_c=H.detach(),
                contexts=copy.deepcopy(contexts),z_member=z_member,batch=batch,cursor=cursor,identity=identity,rng=rng_snapshot())
            payload['integrity_sha256']=state_digest(payload)
            try:
                with tmp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
                actual=tmp.stat().st_size;space=os.statvfs(folder)
                if actual>estimated or space.f_bavail*space.f_frsize<reserve_bytes:raise ValueError('CHECKPOINT_ACTUAL_STORAGE')
                if before!=(H._version,{k:v._version for k,v in weights.items()}):raise ValueError('CHECKPOINT_STATE_MUTATED')
                os.replace(tmp,dest)
                fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
            except BaseException:
                if tmp.exists() and not tmp.is_symlink():tmp.unlink()
                raise
            receipt=dict(member(dest),batch=batch,identity=identity,final_W20=batch==20,
                storage_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                native_H_copied=False,hash_chunk_bytes=8*1024**2)
            meta=folder/('latest.json.'+uuid.uuid4().hex+'.partial')
            try:
                with meta.open('x') as f:json.dump(receipt,f,sort_keys=True);f.flush();os.fsync(f.fileno())
                os.replace(meta,folder/'latest.json')
                fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
            finally:
                if meta.exists() and not meta.is_symlink():meta.unlink()
            return receipt
        finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
