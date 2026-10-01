"""C0 host cache only: no cross-entry adj reuse."""
import time
import numpy as np
import torch
from project.run_scripts.jlz_pilot.run import STATS,LAYERS

class CovarianceCache:
    def __init__(self):self.values={};self.records=[]
    def get(self,layer):
        start=time.monotonic();hit=layer in self.values
        if not hit:
            path=STATS/f'model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz'
            with np.load(path,allow_pickle=False) as f:
                n=int(f['mom2.count']);raw=torch.from_numpy(f['mom2.mom2'].copy())
            if n<=0 or raw.dtype!=torch.float32 or not bool(torch.isfinite(raw).all()):raise RuntimeError('C0_IDENTITY')
            self.values[layer]=raw/n
        value=self.values[layer]
        self.records.append(dict(layer=layer,hit=hit,seconds=time.monotonic()-start,bytes=value.numel()*value.element_size(),OS_cache='UNCONTROLLED'))
        return value
    def solve(self,keys,history):
        output={};records=[]
        for l in LAYERS:
            torch.cuda.synchronize();t=time.monotonic();k=keys[l].double()
            system=15000*self.get(l).to(device=k.device,dtype=torch.float64)+history[l].to(device=k.device,dtype=torch.float64)
            system.add_(k@k.T);adj=torch.linalg.solve(system,k)
            residual=float((system@adj-k).norm()/k.norm().clamp_min(1))
            if residual>1e-7 or not bool(torch.isfinite(adj).all()):raise RuntimeError('ADJ_SOLVE')
            output[l]=adj;torch.cuda.synchronize();records.append(dict(layer=l,seconds=time.monotonic()-t,residual=residual))
            del system,k
        return output,records
