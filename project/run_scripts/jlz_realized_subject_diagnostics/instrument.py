"""Read-only callback, extra component adjoints and RAM-only evaluated W copies.

The original combined-gradient path at c13/17/21 is left intact. Additional
component backwards retain the existing graph and never seed q.grad or Adam.
"""
import copy
import time
import torch
from project.run_scripts.jlz_realized_subject.subject import row_terms
from project.run_scripts.jlz_realized_subject.allocation import loss as allocation
from project.run_scripts.jlz_realized_subject.telemetry import gradient_measure
from project.run_scripts.jlz_realized_subject.writer import rng_snapshot, rng_equal
from .common import CAPTURE, require, tensor_sha, write


def optimizer_identity(q, optimizer):
    def encode(x):
        if isinstance(x,torch.Tensor): return (str(x.dtype),tuple(x.shape),tensor_sha(x))
        if isinstance(x,dict): return {str(k):encode(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [encode(v) for v in x]
        return x
    return dict(q={l:encode(v) for l,v in q.items()},
        grad={l:encode(v.grad) for l,v in q.items()},optimizer=encode(optimizer.state_dict()))


def extra_components(a,entry,R,built,arm='A'):
    """Same existing candidate graph; four diagnostic adjoints, no total update."""
    started=time.monotonic();sites=tuple(R)
    originals=[built['v'][l] for l in sites]
    v={l:x.detach().requires_grad_(True) for l,x in zip(sites,originals)}
    leaves=tuple(v.values());adj={name:[torch.zeros_like(x) for x in leaves] for name in ('nll','kl','norm')}
    calls=tokens=0
    for group in entry['groups']:
        terms,_,nt=row_terms(a,entry,group,v);tokens+=nt
        for j,name in enumerate(adj):
            require(bool(torch.isfinite(terms[name])),'NONFINITE_COMPONENT_OBSERVER')
            gs=torch.autograd.grad(terms[name],leaves,retain_graph=j<2,allow_unused=True)
            for dst,g in zip(adj[name],gs):
                if g is not None:dst.add_(g)
            calls+=1
    comp={}
    for name in adj:
        comp[name]=dict(zip(sites,torch.autograd.grad(originals,tuple(R.values()),adj[name],retain_graph=True)))
    alloc,_=allocation(R,built['geometry'],entry['anchors'],arm)
    comp['allocation']=dict(zip(sites,torch.autograd.grad(alloc,tuple(R.values()),retain_graph=True)))
    require(all(bool(torch.isfinite(g).all()) for values in comp.values() for g in values.values()),'NONFINITE_COMPONENT_GRADIENT')
    return comp,dict(seconds=time.monotonic()-started,masked_backward_calls=calls,
        builder_component_backwards=4,prediction_tokens=tokens,optimizer_bridges=0,geometry_resolves=0)


class RAMObserver:
    def __init__(self,out,capture=CAPTURE,extra=(13,17,21)):
        self.out=out;self.capture=tuple(capture);self.extra=tuple(extra)
        self.snapshots={};self.identities={};self.extra_gradient=None;self.q_before={};self.costs=[]

    def __call__(self,phase,a,entry,k,R,q,scale,optimizer,built,result):
        # Every callback is checked, including the extra adjoint boundaries.
        before=optimizer_identity(q,optimizer);rng=rng_snapshot();hooks=a.hook_signature();guard=a.guard()
        if phase=='before_evaluate' and k in self.extra:
            self.extra_gradient,cost=extra_components(a,entry,R,built)
            self.costs.append(dict(candidate=k,**cost))
        if phase=='evaluated' and k in self.capture:
            started=time.monotonic()
            require(k not in self.snapshots,'DUPLICATE_RAM_SNAPSHOT')
            self.snapshots[k]={l:w.detach().to('cpu',copy=True) for l,w in built['weights'].items()}
            self.identities[k]={str(l):tensor_sha(w) for l,w in self.snapshots[k].items()}
            nbytes=sum(w.numel()*w.element_size() for w in self.snapshots[k].values())
            write(self.out/f'ram-c{k:02d}.json',dict(candidate=k,updates_before=k-1,
                weights=self.identities[k],bytes=nbytes,storage='CPU_RAM_ONLY',checkpoint_saved=False,
                seconds=time.monotonic()-started,from_evaluated_materialized_weights=True))
            if self.extra_gradient is not None:
                # Compare diagnostic component sum to UNCHANGED production total.
                measure=gradient_measure(R,scale,dict(result,components=self.extra_gradient))
                write(self.out/f'components-c{k:02d}.json',dict(candidate=k,gradient=measure,
                    cost=self.costs[-1],production_gradient_replaced=False,existing_graph=True))
                self.extra_gradient=None
            self.q_before={l:v.detach().clone() for l,v in q.items()}
        if phase=='after_update' and k in self.capture:
            steps={str(l):dict(q_step_norm=float((q[l].detach()-old).norm()),
                R_step_norm=float((scale[l]*(q[l].detach()-old)).norm()),
                q_gradient_norm=float(q[l].grad.norm()),Adam_step=float(optimizer.state[q[l]].get('step',0)))
                for l,old in self.q_before.items()}
            write(self.out/f'step-c{k:02d}.json',dict(candidate=k,updated=k<25,layers=steps))
            self.q_before={}
        require(optimizer_identity(q,optimizer)==before,'OBSERVER_OPTIMIZER_MUTATION')
        require(rng_equal(rng) and a.guard()==guard and a.hook_signature()==hooks,'OBSERVER_RNG_HOOK_GUARD_MUTATION')

    def clear(self):
        self.snapshots.clear();self.q_before.clear();self.extra_gradient=None


@torch.no_grad()
def replace(a,weights):
    require(set(weights)==set(a.weights),'SNAPSHOT_LAYER_IDENTITY')
    for l,w in a.weights.items():
        source=weights[l]
        require(source.dtype==torch.float32 and source.shape==w.shape and bool(torch.isfinite(source).all()),'SNAPSHOT_SCHEMA')
        w.copy_(source)
        require(torch.equal(w.cpu(),source.cpu()),'SNAPSHOT_REPLACEMENT_NOT_EXACT')
