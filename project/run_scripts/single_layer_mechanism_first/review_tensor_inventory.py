"""CPU weights_only/mmap inspection of this terminal B1's scientific tensors.

No reconstruction/model/evaluator import, no payload deletion or new checkpoint.
"""
import argparse
import json
from pathlib import Path
import torch
from .review_b1 import dump
from .review_completed_b1 import member


def inspect_payload(value,path=''):
    rows=[]
    if isinstance(value,torch.Tensor):
        if value.device.type!='cpu':raise ValueError('CPU_ONLY_INSPECTION')
        if value.numel()==4096*14336 or (value.ndim>=2 and tuple(value.shape[-2:])==(14336,14336)):
            raise ValueError('UNEXPECTED_FULL_WRITER_OR_MEMORY_TENSOR:'+path)
        finite=True
        if value.is_floating_point():
            flat=value.reshape(-1)
            finite=all(bool(torch.isfinite(flat[start:start+1048576]).all()) for start in range(0,flat.numel(),1048576))
        if not finite:raise ValueError('NONFINITE_SAVED_SCIENTIFIC_TENSOR:'+path)
        rows.append(dict(field=path,shape=list(value.shape),dtype=str(value.dtype),finite=finite,
                         logical_bytes=value.numel()*value.element_size()))
    elif isinstance(value,dict):
        for k,v in value.items():rows.extend(inspect_payload(v,path+'/'+str(k)))
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):rows.extend(inspect_payload(v,path+'/'+str(i)))
    return rows


def run(output,destination):
    output=Path(output)
    if not (output/'terminal.json').exists():raise ValueError('TERMINAL_ONLY')
    torch.set_num_threads(2);members=[]
    for path in sorted(output.rglob('*.pt')):
        if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
            raise ValueError('NO_EXTERNAL_PAYLOAD_FOLLOW')
        value=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        rows=inspect_payload(value)
        members.append(dict(**member(path),tensor_members=rows,load='CPU_WEIGHTS_ONLY_MMAP'))
        del value
    dump(Path(destination),dict(members=members,checkpoint_saved=False,
        exact_resume='NOT_AVAILABLE',GPU_continuation='NOT_TESTED',new_GPU=0,
        inspection_scope='SAVED_SCIENTIFIC_TENSORS; NOT_FULL_ENDPOINT_RECONSTRUCTION'))
    return dict(files=len(members),tensor_members=sum(len(m['tensor_members']) for m in members),
                bytes=sum(m['bytes'] for m in members))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();print(json.dumps(run(a.output,a.destination)))
