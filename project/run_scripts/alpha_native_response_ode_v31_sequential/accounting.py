"""Process-local count-only instrumentation; scientific values are untouched."""
import time
import torch


class Accounting:
    def __init__(self, model):
        self.counts=dict(forward=0,forward_input_tokens=0,backward=0,linalg_solve=0)
        def forward(module,args,kwargs):
            self.counts['forward']+=1
            x=kwargs.get('input_ids',args[0] if args else None)
            if x is not None:self.counts['forward_input_tokens']+=x.numel()
        self.hook=model.register_forward_pre_hook(forward,with_kwargs=True)
        self.backward=torch.autograd.backward;self.solve=torch.linalg.solve
        def backward(*a,**kw):
            self.counts['backward']+=1
            return self.backward(*a,**kw)
        def solve(*a,**kw):
            self.counts['linalg_solve']+=1
            return self.solve(*a,**kw)
        torch.autograd.backward=backward;torch.linalg.solve=solve
    def snapshot(self):return dict(self.counts,wall=time.perf_counter())
    def difference(self,before):return {k:v-before[k] for k,v in self.snapshot().items()}
    def close(self):
        self.hook.remove();torch.autograd.backward=self.backward;torch.linalg.solve=self.solve
