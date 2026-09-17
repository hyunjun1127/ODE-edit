from pathlib import Path
import dataclasses
import math
from project.run_scripts.local_z_adaptive_allocation.common import encoded,digest,sha,identity,save,tensor_save,Timer

ROOT=Path('/data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2')
COLD=Path('/data/janghj/ODE-edit/local/local-z-adaptive-allocation/20260916-v1')
TASK='ODEEDIT-S06-SEQUENTIAL-LOCAL-Z-ALLOCATION-V2-SH4-V1'
LAYERS=(4,5,6,7,8)
ARMS=('C45678','N4','F48','G48','C4','C48')
ARM_LAYERS={'N4':(4,),'F48':(4,8),'G48':(4,8),'C4':(4,),'C48':(4,8),'C45678':LAYERS}

def serial(x):
    if dataclasses.is_dataclass(x):return serial(dataclasses.asdict(x))
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    if isinstance(x,(set,frozenset)):return sorted(serial(v) for v in x)
    if hasattr(x,'tolist'):return x.tolist()
    return x

def verify(lock):
    assert lock['instruction_id']==TASK and lock['arms']==list(ARMS)
    assert lock['seed']==20260916 and lock['gpu_cap']==2 and lock['history_layers']==list(LAYERS)
    for m in lock['members']:
        p=Path(m['path']);s=p.stat();assert s.st_size==m['bytes'],('INPUT_SIZE',str(p))
        if m.get('verification')=='PRIOR_FULL_SHA_STABLE_STAT':
            assert [s.st_dev,s.st_ino,s.st_mtime_ns]==m['stat'],('INPUT_STAT',str(p))
        else:assert sha(p)==m['sha256'],('INPUT_SHA',str(p))

class Ledger:
    def __init__(self,root):self.root=Path(root);self.n=0
    def __call__(self,event):
        p=self.root/f'{self.n:06d}.json';self.n+=1
        save(p,serial(event))
