"""Ordered full-space weight blocks and preregistered matrix-free PCG.

No rank reduction, explicit inverse, or functional block diagonalization.
Scalar inner products are FP64; vector/model storage is caller-pinned dtype.
"""
from dataclasses import dataclass, asdict
from typing import Callable
import math
import torch

WeightTree = tuple[torch.Tensor, ...]
Operator = Callable[[WeightTree], WeightTree]


def finite(x: WeightTree) -> WeightTree:
    if not all(bool(torch.isfinite(t).all()) for t in x):
        raise FloatingPointError('NONFINITE_WEIGHT_VECTOR')
    return x


def zeros(x): return tuple(torch.zeros_like(t) for t in x)
def scale(x, a): return tuple(t * a for t in x)
def add(x, y, alpha=1.0):
    if len(x) != len(y) or any(a.shape != b.shape for a,b in zip(x,y)):
        raise ValueError('WEIGHT_TREE_SHAPE_MISMATCH')
    return tuple(a + alpha*b for a,b in zip(x,y))


def dot(x, y) -> float:
    if len(x)!=len(y):raise ValueError('WEIGHT_TREE_LENGTH_MISMATCH')
    total=0.0
    for a,b in zip(x,y):
        if a.shape!=b.shape:raise ValueError('WEIGHT_TREE_SHAPE_MISMATCH')
        aa=a.reshape(-1);bb=b.reshape(-1)
        # Bounded temporary doubles, rather than cloning full weight blocks in FP64.
        for i in range(0,aa.numel(),1<<20):
            total += float((aa[i:i+(1<<20)].double()*bb[i:i+(1<<20)].double()).sum())
    if not math.isfinite(total):raise FloatingPointError('NONFINITE_DOT')
    return total


def norm(x):return math.sqrt(max(dot(x,x),0.0))


@dataclass
class PCGResult:
    solution: WeightTree
    status: str
    iterations: int
    relative_residual: float
    absolute_residual: float
    rhs_norm: float
    matvec_calls: int
    precondition_calls: int
    recursive_residuals: list[float]

    def receipt(self):
        return {k:v for k,v in vars(self).items() if k!='solution'}


def pcg(operator: Operator, rhs: WeightTree, precondition: Operator | None = None,
        *, rtol: float = 1e-4, maxiter: int = 20) -> PCGResult:
    """Independent RHS solve; finite maxiter endpoints remain typed approximate.

    The final true residual costs one extra matvec and is explicitly counted.
    Zero RHS requires no operator call. No initial-guess/state reuse across RHS.
    """
    if rtol<=0 or maxiter<1:raise ValueError('INVALID_PCG_POLICY')
    finite(rhs); bnorm=norm(rhs);x=zeros(rhs)
    if bnorm==0:return PCGResult(x,'ZERO_RHS',0,0.,0.,0.,0,0,[])
    precondition=precondition or (lambda v: v)
    r=tuple(t.detach().clone() for t in rhs);z=finite(precondition(r));p=z
    rz=dot(r,z)
    if rz<=0:raise FloatingPointError('PCG_NONPOSITIVE_PRECONDITIONER')
    calls=0;pre_calls=1;history=[];iteration=0
    for iteration in range(1,maxiter+1):
        ap=finite(operator(p));calls+=1;pap=dot(p,ap)
        if pap<=0:raise FloatingPointError('PCG_NONPOSITIVE_CURVATURE')
        alpha=rz/pap;x=add(x,p,alpha);r=add(r,ap,-alpha)
        relative=norm(r)/bnorm;history.append(relative)
        if relative<=rtol:break
        z=finite(precondition(r));pre_calls+=1;new_rz=dot(r,z)
        if new_rz<=0:raise FloatingPointError('PCG_NONPOSITIVE_PRECONDITIONER')
        p=add(z,p,new_rz/rz);rz=new_rz
    actual=add(rhs,finite(operator(x)),-1);calls+=1
    abs_res=norm(actual);rel=abs_res/bnorm
    status='CONVERGED' if rel<=rtol else 'APPROXIMATE_PCG_NONCONVERGENCE'
    return PCGResult(finite(x),status,iteration,rel,abs_res,bnorm,calls,pre_calls,history)
