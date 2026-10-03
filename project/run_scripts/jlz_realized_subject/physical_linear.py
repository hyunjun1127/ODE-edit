"""One FP32 linear with full D/P/input VJP and no duplicate W gradient."""
import torch
import torch.nn.functional as F

def materialize(entry, D, P):
    return entry + (D.double() @ P.T).to(entry.dtype)

class PhysicalLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, D, P, W):
        ctx.save_for_backward(x, D, P, W)
        return F.linear(x, W)
    @staticmethod
    def backward(ctx, g):
        x, d, p, w = ctx.saved_tensors
        xf = x.reshape(-1, x.shape[-1]); gf = g.reshape(-1, g.shape[-1])
        gx = g @ w if ctx.needs_input_grad[0] else None
        gw = (gf.T @ xf).double() if ctx.needs_input_grad[1] or ctx.needs_input_grad[2] else None
        gd = (gw @ p).to(d.dtype) if ctx.needs_input_grad[1] else None
        gp = gw.T @ d.double() if ctx.needs_input_grad[2] else None
        return gx, gd, gp, None

def linear(x, D, P, W, route='direct'):
    return F.linear(x, W) if route == 'dense' else PhysicalLinear.apply(x, D, P, W)
