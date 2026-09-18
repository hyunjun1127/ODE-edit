"""Pair activation/key factors: exact FP64 projection, Gram and positive sum.

No bank subsampling, dense per-reference gradient archive or Gram ridge.
"""
import time
from pathlib import Path
import numpy as np
import torch
from project.run_scripts.single_layer_edit_preserving_correction.sequential_state import atomic_tensor
from .provenance import create_json

class Projection:
    def __init__(self,space,device):
        if space.status=='RANK_UNRESOLVED':raise ValueError('UNRESOLVED_PROJECTION')
        self.V=torch.as_tensor(space.basis,device=device,dtype=torch.float64)
        self.B=torch.as_tensor(space.blocked,device=device,dtype=torch.float64)
    def keys(self,K):
        x=K.double().to(self.V.device).T@self.V
        if self.B.shape[1]:x=x-(x@self.B)@self.B.T
        out=(x@self.V.T).T
        if not torch.isfinite(out).all():raise FloatingPointError('NONFINITE_QK')
        return out

def build(oracle,space,weight,WN,pairs,scan,out):
    out=Path(out);proj=Projection(space,oracle.device);files=[];b=[];receipts=[]
    delta=weight.double().to(oracle.device)-WN.double().to(oracle.device)
    for ix,pair in enumerate(pairs):
        begin=time.monotonic();i=pair['index']
        A,K,value=oracle.pair_factor(i,weight,pair['position'],pair['target'],pair['competitor'])
        U=proj.keys(K)
        # RHS deliberately uses RAW pair gradient and actual rounded center.
        center_inner=float((A.double().to(oracle.device)*(delta@K.double().to(oracle.device))).sum())
        doc=scan['documents'][i]
        retained={r['pair_id']:r for r in doc['retained_pairs']}
        if pair['pair_id'] in retained:mu=retained[pair['pair_id']]['mu']
        else:
            p=next(p for p in doc['positions'] if p['position']==pair['position'])
            if p['competitor']!=pair['competitor']:raise ValueError('MISSING_RETAINED_FULL_VOCAB_MARGIN')
            mu=p['mu']
        rhs=-mu+center_inner
        path=out/f'{ix:04d}.pt'
        atomic_tensor(path,dict(A=A,U=U.cpu(),pair=pair,mu=mu,raw_center_inner=center_inner,b=rhs,
            pair_two_head_margin=value,full_vocab_margin=mu+pair['kappa']))
        files.append(path);b.append(rhs);receipts.append(dict(pair=pair,b=rhs,mu=mu,raw_center_inner=center_inner,
            two_head_margin=value,full_vocab_margin=mu+pair['kappa'],seconds=time.monotonic()-begin,
            factor_shape=[list(A.shape),list(U.shape)]))
        if ix%32==0:print('PAIR_GRADIENT',str(out.name),ix,len(pairs),flush=True)
    del proj,delta
    create_json(out/'rows.json',receipts)
    return files,np.array(b,dtype=np.float64)

def _load_group(files,device,length):
    As=[];Us=[]
    for p in files:
        row=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        A=row['A'].to(device=device,dtype=torch.float64);U=row['U'].to(device=device,dtype=torch.float64)
        As.append(torch.nn.functional.pad(A,(0,length-A.shape[1])).T)
        Us.append(torch.nn.functional.pad(U,(0,length-U.shape[1])).T)
    return torch.stack(As),torch.stack(Us)

def gram(files,device,tile=8):
    lengths=[torch.load(p,weights_only=True,mmap=True,map_location='cpu')['A'].shape[1] for p in files]
    length=max(lengths);n=len(files);G=torch.empty((n,n),dtype=torch.float64)
    for i in range(0,n,tile):
        A,U=_load_group(files[i:i+tile],device,length);ni=A.shape[0]
        for j in range(0,i+1,tile):
            B,V=(A,U) if i==j else _load_group(files[j:j+tile],device,length)
            nj=B.shape[0]
            AA=(A.flatten(0,1)@B.flatten(0,1).T).view(ni,length,nj,length)
            UU=(U.flatten(0,1)@V.flatten(0,1).T).view(ni,length,nj,length)
            block=(AA*UU).sum((1,3)).cpu()
            G[i:i+ni,j:j+nj]=block;G[j:j+nj,i:i+ni]=block.T
            del AA,UU,B,V
        del A,U
        print('GRAM_ROWS',i,n,flush=True)
    if not torch.isfinite(G).all():raise FloatingPointError('NONFINITE_FACTOR_GRAM')
    return G.numpy()

def reconstruct(files,alpha,shape,device):
    D=torch.zeros(shape,dtype=torch.float64,device=device)
    for p,a in zip(files,alpha,strict=True):
        if a==0:continue
        row=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        D.addmm_(row['A'].to(device=device,dtype=torch.float64),row['U'].to(device=device,dtype=torch.float64).T,alpha=float(a))
    if not torch.isfinite(D).all():raise FloatingPointError('NONFINITE_RECONSTRUCTION')
    return D.cpu()
