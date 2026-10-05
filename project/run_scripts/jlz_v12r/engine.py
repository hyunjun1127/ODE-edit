"""No-grad current native writer BUILD and intentionally same-layer gradient."""
import torch
from project.run_scripts.jlz_native_writer_aware.physical import Adapter
from project.run_scripts.jlz_native_writer_aware.builder import build as native_build
from project.run_scripts.jlz_native_writer_aware.common import require,tensor_sha
from .subject import evaluate as subject_evaluate,pullback


class CandidateObjective:
    def __init__(self,a,entry,events=None):
        self.a,self.entry,self.events=a,entry,events
        self.calls=dict(logical_builds=0,logical_subject_forwards=0,logical_subject_backwards=0,
                        physical_subject_forward_groups=0,physical_subject_backward_groups=0,
                        builder_stage_groups=0,checkpoint_wrappers=0,recomputed_function_invocations=0,
                        production_builder_reverse=0,production_solve_VJP=0)

    def zeros(self):
        B=self.entry['pack']['n_requests']
        return {l:torch.zeros(self.a.dims[l][0],B,device=self.a.device,dtype=torch.float32) for l in self.a.sites}

    def build(self,R,candidate):
        require(set(R)==set(self.a.sites),'REQUEST_LAYER_IDENTITY')
        B=self.entry['pack']['n_requests']
        require(all(v.shape==(self.a.dims[l][0],B) and v.dtype==torch.float32 and bool(torch.isfinite(v).all())
                    for l,v in R.items()),'REQUEST_FP32_SCHEMA')
        built=native_build(self.a,self.entry,R,candidate);self.calls['logical_builds']+=1
        self.calls['builder_stage_groups']+=len(self.entry['groups'])*max(0,len(self.a.sites)-1)
        built['R_hashes']={str(l):tensor_sha(v) for l,v in R.items()}
        return built

    def evaluate(self,R,candidate,active_previous=None,terminal=False,capture=False,blind=False,
                 fixed_mask=None,built=None):
        physical_before=dict(getattr(self.a,'physical_calls',{}))
        built=self.build(R,candidate) if built is None else built
        result=subject_evaluate(self.a,self.entry,built,backward=not terminal,capture=capture,
            active_previous=active_previous,fixed_mask=fixed_mask,terminal=terminal)
        self.calls['logical_subject_forwards']+=1
        self.calls['logical_subject_backwards']+=result['logical_backward']
        self.calls['physical_subject_forward_groups']+=result['forward_groups']
        self.calls['physical_subject_backward_groups']+=result['backward_groups']
        gradient=None;receipt=None
        if result['adjoint'] is not None and result['logical_backward']:
            gradient,receipt=pullback(built,result['adjoint'],R,blind=blind)
        elif not terminal:gradient={l:torch.zeros_like(v,dtype=torch.float64) for l,v in R.items()}
        physical_after=getattr(self.a,'physical_calls',{})
        for local,original in (('checkpoint_wrappers','checkpoint_wrappers'),
                               ('recomputed_function_invocations','function_invocations')):
            self.calls[local]+=physical_after.get(original,0)-physical_before.get(original,0)
        return dict(built=built,observed=result,F=result['F'],active_mask=result['active_mask'],
            gradient=gradient,pullback=receipt,cost=dict(self.calls),candidate=candidate,
            terminal=terminal,production_full_gradient=False)

    def forward(self,built,capture=False):
        return subject_evaluate(self.a,self.entry,built,backward=False,capture=capture)

    def backward(self,R,built,active_mask,blind=False):
        """Diagnostic-only replay; production evaluate never uses this route."""
        result=subject_evaluate(self.a,self.entry,built,backward=True,fixed_mask=active_mask)
        gradient,record=pullback(built,result['adjoint'],R,blind)
        return dict(gradient=gradient,observed=result,pullback=record)
