"""Actual pinned production oracle, no reference reimplementation."""
import time
import resource
import torch
from project.run_scripts.jlz_sequential.oracle import Oracle

class Reference(Oracle):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.records=[];self.route_name='REF_MB2'
    def __call__(self,x):
        torch.cuda.synchronize();t=time.monotonic();e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);e0.record()
        torch.cuda.reset_peak_memory_stats()
        result=super().__call__(x)
        e1.record();torch.cuda.synchronize()
        tokens=self.spec['tokens'];width=tokens['input_ids'].shape[1]
        row=dict(route='REF_MB2',seconds=time.monotonic()-t,cuda_ms=e0.elapsed_time(e1),forward_chunks=len(self.rows),backward_chunks=len(self.rows),
            valid_tokens=int(tokens['attention_mask'].sum()),padded_tokens=tokens['input_ids'].numel(),attention_L2_positions=tokens['input_ids'].numel()*width,
            head_rows=int((self.spec['targets']!=-100).sum())+self.B,
            materialization_GEMMs=5,dense_gradient_GEMMs=5*len(self.rows),mapping_GEMMs=5,
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),host_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            components='NOT_SEPARATED in immutable production; total actual reference measured')
        self.records.append(row);result[2]['timing']=row
        return result
