"""TECH-R1: enforce the existing nonnegative dual domain in correct units.

The original step equations are reused unchanged in a private function-global
namespace. No original source/module globals are mutated. Matrix magnitude is
not a tolerance on the differently dimensioned dual variable.
"""
import itertools
import types
import torch
from . import controller as original
from .geometry import finite, dot

def joint_dual(matrix,rhs):
    matrix=finite(matrix.double());rhs=finite(rhs.double());n=len(rhs)
    if n>2:raise ValueError('ONLY_TWO_MEMORY_CHANNELS')
    candidates=[]
    for active in itertools.product((False,True),repeat=n):
        mask=torch.tensor(active,device=rhs.device,dtype=torch.bool)
        lam=torch.zeros_like(rhs)
        if mask.any():lam[mask]=torch.linalg.solve(matrix[mask][:,mask],rhs[mask])
        # Exact feasibility of the prescribed domain; never clip a solution.
        if not bool((lam>=0).all()):continue
        grad=matrix@lam-rhs
        tol=64*max(n,1)*torch.finfo(torch.float64).eps*torch.maximum(
            matrix.abs()@lam.abs()+rhs.abs(),torch.full_like(rhs,torch.finfo(torch.float64).tiny))
        if bool((grad[~mask]>=-tol[~mask]).all()) and bool((grad[mask].abs()<=tol[mask]).all()):
            objective=.5*dot(lam,matrix@lam)-dot(rhs,lam)
            candidates.append((float(objective),active,lam,grad,tol))
    if not candidates:raise FloatingPointError('JOINT_KKT_NO_FEASIBLE_ACTIVE_SET')
    _,active,lam,grad,tol=min(candidates,key=lambda x:(x[0],x[1]))
    return lam,dict(active=list(active),dual=lam.tolist(),dual_gradient=grad.tolist(),
        complementarity=float((lam*grad).abs().max()) if n else 0.,
        gradient_tolerance=tol.tolist(),dual_nonnegativity='EXACT_DOMAIN_NO_CLIP',
        repair='TECH_R1_DIMENSIONALLY_CORRECT_FEASIBILITY')

step=types.FunctionType(original.step.__code__,dict(original.step.__globals__,joint_dual=joint_dual),
    name='step',argdefs=original.step.__defaults__)
calibration=original.calibration
