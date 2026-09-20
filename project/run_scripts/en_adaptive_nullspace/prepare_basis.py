"""Recover immutable native allowed range on CPU; never edits model weights."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='8'
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from project.run_scripts.single_layer_edit_preserving_correction.geometry import allowed_range

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 assert not (out/'basis.npy').exists();torch.set_num_threads(8)
 raw=torch.load('/data/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt',weights_only=True,map_location='cpu',mmap=True)[0]
 start=time.monotonic();space=allowed_range(raw,threads=8)
 with (out/'basis.npy').open('xb') as f:np.save(f,space.basis,allow_pickle=False)
 h=hashlib.sha256()
 with (out/'basis.npy').open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 receipt=dict(shape=list(space.basis.shape),dtype=str(space.basis.dtype),sha256=h.hexdigest(),bytes=(out/'basis.npy').stat().st_size,seconds=time.monotonic()-start,source='readonly native physical P4',derivation=space.provenance if hasattr(space,'provenance') else space.__dict__.get('receipt',{}),CUDA_initialized=torch.cuda.is_initialized(),edited_checkpoint=False)
 # Store original geometry metadata without large eigenvectors.
 receipt['geometry_metadata']={k:v for k,v in space.__dict__.items() if not isinstance(v,np.ndarray)}
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ['geometry_metadata','derivation']}),flush=True)
if __name__=='__main__':main()
