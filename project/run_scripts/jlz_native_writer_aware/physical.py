"""Canonical materialized forward; qualified bounded direct R/P/input VJP."""
import torch
import torch.nn.functional as F
from project.run_scripts.jlz_realized_subject.profile import LlamaAdapter as BaseAdapter,move

def materialize(entry,R,P):
    return entry+(R.double()@P.T).to(entry.dtype)

class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,R,P,W):
        ctx.save_for_backward(x,R,P,W)
        return F.linear(x,W)
    @staticmethod
    def backward(ctx,g):
        x,R,P,W=ctx.saved_tensors
        # No persistent/dense dW. FP64 contraction order is separately qualified.
        xf=x.reshape(-1,x.shape[-1]).double()
        gf=g.reshape(-1,g.shape[-1]).double()
        gr=(gf.T@(xf@P)).to(R.dtype)
        gp=xf.T@(gf@R.double())
        return g@W,gr,gp,None

def linear(x,R,P,W,route='direct'):
    return F.linear(x,W) if route=='dense' else Linear.apply(x,R,P,W)

class Adapter(BaseAdapter):
    def stage(self,layer,next_layer,key,residual,R,P,W,kw,route='direct'):
        def step(k,r,d,p,w):
            x=r+linear(k,d,p,w,route)
            for j in range(layer+1,next_layer):
                x=self.unwrap(self.blocks[j](x,**kw))
            return self.pre_projection(next_layer,x,kw)
        return self.recompute(step,key,residual,R,P,W)
