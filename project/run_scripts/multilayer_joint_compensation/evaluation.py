"""Tuple-state observation transactions and pinned CounterFact evaluation.

Temporary parameter storage is independent; the original storage, version and
bytes are restored even if the native/evaluation call raises. Not a controller.
"""
import contextlib
import torch
from .contracts import tensor_sha
from .observations import StateBoundary
from project.run_scripts.l4_two_memory_conflict_routing.evaluation import measure,generation

@contextlib.contextmanager
def materialized(model,names,weights,ledger,*,purpose='observation'):
    parameters=dict(model.named_parameters());names=tuple(names)
    if len(names)!=len(weights) or len(set(names))!=len(names):raise StateBoundary('MATERIALIZE_SUPPORT')
    selected=[parameters[n] for n in names]
    original=[p.data for p in selected]
    if len({x.untyped_storage().data_ptr() for x in original})!=len(original):
        raise StateBoundary('MATERIALIZE_STORAGE_ALIAS')
    signature=[(p.data_ptr(),p._version,tensor_sha(p)) for p in selected]
    for p,w in zip(selected,weights):
        if p.shape!=w.shape or p.dtype!=w.dtype or not torch.isfinite(w).all():
            raise StateBoundary('MATERIALIZE_SHAPE_DTYPE_FINITE')
    try:
        with torch.no_grad():
            for p,w in zip(selected,weights):p.data=w.detach().to(p.device).clone()
        ledger.add(purpose+'_temporary_materializations')
        yield
    finally:
        with torch.no_grad():
            for p,data in zip(selected,original):p.data=data
        after=[(p.data_ptr(),p._version,tensor_sha(p)) for p in selected]
        # Native in-place edits can bump the shared version counter while using
        # temporary storage. Pointer/bytes restoration is exact; version change
        # is recorded and subsequent functional views must be newly captured.
        if any((a[0],a[2])!=(b[0],b[2]) for a,b in zip(signature,after)):
            raise StateBoundary('MATERIALIZE_RESTORE_FAILED')
        ledger.add(purpose+'_restores')
        ledger.add(purpose+'_version_increments',sum(b[1]-a[1] for a,b in zip(signature,after)))

def panel(view,rows,saved_teacher,pad_token_id,role,physical_microbatch=2):
    from .observations import pack,bind_teacher
    from .functional import OutputBatch,FunctionalPanel
    batches=[]
    for b in pack(view,rows,pad_token_id,physical_microbatch):
        batches.append(OutputBatch(logits_fn=b.logits,**b.metadata(),**bind_teacher(b,saved_teacher),
          identity='|'.join(r['identity'] for r in b.rows),input_tokens=int(b.attention_mask.sum())))
    return FunctionalPanel(batches,role,tau=.1)

def observe_panel(functional_panel,weights):
    """Observation bridge for shared API v1, preserving all raw context terms."""
    from .functional import values
    result=[];total=0.;mean_nll=0.
    with torch.no_grad():
        for b in functional_panel.batches:
            logits=functional_panel._logits(weights,b)
            loss,nll,per,nlls=values(logits,b,functional_panel.role,functional_panel.tau)
            total+=float(loss);mean_nll+=float(nll)
            result.append(dict(identity=b.identity,values=per.cpu().tolist(),nll=nlls.cpu().tolist(),
              context_weights=b.context_weights.cpu().tolist()))
    return dict(value=total,mean_nll=mean_nll,context_rows=result,counts=functional_panel.counts.copy())
