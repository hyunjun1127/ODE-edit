"""Fresh native BUILD, same-layer full-owner pullback; no production reverse."""
import torch
from official.ours.core.jlz_v12r.engine import CandidateObjective as ParentObjective
from official.ours.core.jlz_native_writer_aware.physical import Adapter
from official.ours.core.jlz_native_writer_aware.builder import build as native_build
from official.ours.common import require,tensor_sha


class CandidateObjective(ParentObjective):
    def build(self,R,candidate):
        require(set(R)==set(self.a.sites),'REQUEST_LAYER_IDENTITY')
        B=self.entry['pack']['n_requests']
        require(all(v.shape==(self.a.dims[l][0],B) and v.dtype==torch.float32 and bool(torch.isfinite(v).all())
                    for l,v in R.items()),'REQUEST_FP32_SCHEMA')
        built=native_build(self.a,self.entry,R,candidate,expose_mean_M=True)
        self.calls['logical_builds']+=1
        self.calls['builder_stage_groups']+=len(self.entry['groups'])*max(0,len(self.a.sites)-1)
        built['R_hashes']={str(l):tensor_sha(v) for l,v in R.items()}
        require(all(built['mean_M'][l].shape==(B,B) for l in self.a.sites),'MEAN_M_NOT_ROW_M')
        return built
