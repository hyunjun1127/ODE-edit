"""Host-only preparation for SH1-owned native FE integration. No GPU/model load."""
import importlib.metadata as md
import json,os,subprocess,zipfile,io
from pathlib import Path
import numpy as np
from official.experiments.prepare import read,write_new,file_sha
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011')
AUTHOR='478134dfb24b43f4e18b47e8500893ce3f9cc50f'
def member(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p))
def main():
 clone=LOCAL/'author-FE'
 def git(*a):return subprocess.check_output(['git','-C',str(clone),*a],text=True).strip()
 assert git('rev-parse','HEAD')==AUTHOR and not git('status','--porcelain')
 source=Path('/mnt/raid5/janghj/ODE-edit/local/fe-author-hparams-2k-20261010/preparation-r1/assets.json')
 assets=read(source);members=assets['model']['members'];verified=[]
 for m in members:
  p=Path(m['path']);s=p.stat()
  for k,v in [('bytes',s.st_size),('inode',s.st_ino),('device',s.st_dev),('mtime_ns',s.st_mtime_ns)]:
   if k in m:assert m[k]==v,(p,k)
  verified.append(dict(path=str(p),bytes=s.st_size,sha256=m['sha256'],verification='prior fullSHA + unchanged stat'))
 cov=[]
 for layer,v in assets['C0'].items():
  m=v['member'];p=Path(m['path']);s=p.stat()
  assert s.st_size==m['bytes'] and s.st_mtime_ns==m['mtime_ns'] and s.st_ino==m['inode']
  with zipfile.ZipFile(p) as z:
   count=int(np.load(io.BytesIO(z.read('mom2.count.npy')),allow_pickle=False))
  assert count==v['validation']['masked_token_vector_count']
  cov.append(dict(layer=int(layer),member=m,count=count,normalization='raw uncentered sum/count exactly once',recompute=False))
 old=Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/registration-r1')
 streams={k:member(old/'streams'/f'{k}-stream.json') for k in ('cf','zsre')}
 for m in streams.values():assert len(read(m['path']))==2000
 # Two independent jobs each retain one latest and need one atomic temporary.
 H=5*18944*18944*4;W=5*3584*18944*2;Z=2*2000*3584*5*4
 required=4*(H+W)+Z+2*1024**3+32*1024**3
 free=os.statvfs(LOCAL).f_bavail*os.statvfs(LOCAL).f_frsize
 data=dict(instruction='USER-FE-ORIGINAL-W0-RESET-20261011-R1',server='server2',author_commit=AUTHOR,author_tree=git('rev-parse','HEAD^{tree}'),
  clone=str(clone),clone_dirty=False,local_author_patch=None,shared_integration='SOURCE_INPUT_PENDING_SH1',
  author_yaml=member(clone/'configs/llms/qwen2.5-7b.yaml'),algs_yaml=member(clone/'configs/algs/memit.yaml'),
  requirements=member(clone/'requirements.txt'),model_asset_manifest=member(source),model_members=verified,C0=cov,streams=streams,
  native_contexts='GENERATE_FRESH_AUTHOR_W0_SEED0; no previous context injected',projector_required=False,
  z_policy='firstforward all2000 at coldW0 before editing; cache bound source/model/dtype/hparams/stream/context',
  runtime_python=str(LOCAL/'runtime/bin/python'),runtime={k:md.version(k) for k in ('torch','transformers','tokenizers','numpy','hydra-core','omegaconf')},
  runtime_deviations=dict(Python='3.12; author hydra1.2/numpy1.25 pins not compatible; isolated hydra1.3.2/numpy2.2.6 require shared CPU review',torch='existing 2.9.1+cu128 readonly site-packages; no torch/model download'),
  dtype='bfloat16',hparams=dict(layers=[4,5,6,7,8],clamp=1,steps=35,lr=.5,decay=.001,loss_layer=27,KL=.0625,C0_weight=15000,add_old_keys=True,L2=0),
  datasets=['cf','zsre'],seed=0,batch_size=100,batches=20,current_every_batch=True,allseen_checkpoint_edits=[500,1000,1500,2000],
  storage=dict(H_bytes=H,selected_W_BF16_bytes=W,two_latest_plus_two_atomic_bytes=4*(H+W),two_z_upper_bytes=Z,
    metrics_margin_bytes=2*1024**3,reserve_bytes=32*1024**3,required_free_bytes=required,observed_free_bytes=free,sufficient=free>=required,
    C0_no_copy=True),resources=dict(GPU=1,CPUs=6,memory_MiB=59392,wall_hours=48,project_cap=2),
  old_checkpoint_reuse=False,new_job_ids=[],GPU_observation=False)
 write_new(LOCAL/'host-preparation.json',data)
 print(json.dumps(dict(storage=data['storage'],runtime=data['runtime'],shared=data['shared_integration'])))
if __name__=='__main__':main()
