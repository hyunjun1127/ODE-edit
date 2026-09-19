"""Bounded native/cache-head/batched z T0; invocation belongs to task runner.

No source/global patch, no submit/model load. The caller supplies an already
loaded FP32/eager pinned model and the four preselected Current requests.
"""
from __future__ import annotations

import hashlib
import inspect
import copy
import math
import time

import torch

from .z_hook import (NATIVE_Z_SHA, ZHookBoundary, ZHookConfig, compute_z_batch,
                     instrument_native_compute_z, normalize_requests, prepare_batch)


def _synchronize(device):
    if device.type == "cuda": torch.cuda.synchronize(device)


def _run_timed(callback, device):
    _synchronize(device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    start=time.monotonic()
    result=callback()
    _synchronize(device)
    return result, dict(seconds=time.monotonic()-start,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None)


def _compare(reference, candidate, reference_targets, targets):
    if (not reference or len(reference) != len(candidate) or
            reference_targets.shape != targets.shape or reference_targets.ndim != 2 or
            reference_targets.shape[1] != len(reference)):
        raise ZHookBoundary("T0_COMPARISON_CARDINALITY")
    reference_targets = reference_targets.detach().cpu()
    targets = targets.detach().cpu()
    if not torch.isfinite(reference_targets).all() or not torch.isfinite(targets).all():
        raise ZHookBoundary("T0_COMPARISON_NONFINITE_TARGET")
    rows=[]
    for index,(ref,other) in enumerate(zip(reference,candidate)):
        for trace in (ref,other):
            if (not trace["losses"] or any(not math.isfinite(float(row[key]))
                    for row in trace["losses"] for key in ("nll","loss"))):
                raise ZHookBoundary("T0_COMPARISON_NONFINITE_LOSS")
        equal_steps=len(ref["losses"])==len(other["losses"])
        pairs=list(zip(ref["losses"],other["losses"]))
        gradients=list(zip(ref["gradients"],other["gradients"]))
        grad_relative=[]
        gradient_rows=[]
        for step,(left,right) in enumerate(gradients):
            if left.shape != right.shape:
                raise ZHookBoundary("T0_GRADIENT_SHAPE")
            left,right=left.detach().cpu(),right.detach().cpu()
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise ZHookBoundary("T0_COMPARISON_NONFINITE_GRADIENT")
            denom=float(left.double().norm())
            error=float((left.double()-right.double()).norm())
            grad_relative.append(error/denom if denom else (0. if error==0 else float("inf")))
            gradient_rows.append(dict(step=step,reference_norm=denom,absolute_l2=error,
                max_abs=float((left.double()-right.double()).abs().max()),relative_l2=grad_relative[-1]))
        delta=targets[:,index].double()-reference_targets[:,index].double()
        rows.append(dict(request_index=index, loss_steps_reference=len(ref["losses"]),
                         loss_steps_candidate=len(other["losses"]), loss_steps_equal=equal_steps,
                         adam_steps_equal=len(ref["gradients"])==len(other["gradients"]),
                         max_NLL_abs=max((abs(a["nll"]-b["nll"]) for a,b in pairs),default=0.),
                         max_total_loss_abs=max((abs(a["loss"]-b["loss"]) for a,b in pairs),default=0.),
                         max_gradient_relative=max(grad_relative,default=0.),
                         gradient_steps=gradient_rows,
                         compared_gradient_steps=len(gradients),
                         z_bitwise_equal=torch.equal(targets[:,index],reference_targets[:,index]),
                         z_max_abs=float(delta.abs().max()),z_l2=float(delta.norm()),
                         z_relative=float(delta.norm()/reference_targets[:,index].double().norm())))
    # These thresholds are inherited and fixed before observing T0. No invented
    # vector/write ceiling; their raw differences require parent actual checks.
    passed=all(r["loss_steps_equal"] and r["adam_steps_equal"] and
               r["max_NLL_abs"]<=1e-4 and r["max_gradient_relative"]<=1e-4 for r in rows)
    return dict(pass_inherited_NLL_gradient_and_stop_gate=passed, finite_checked=True, rows=rows,
                thresholds=dict(NLL_abs=1e-4,direct_cached_gradient_relative=1e-4,
                                stop_iteration="exact",z_vector="REPORTED_NO_INVENTED_CEILING"),
                actual_write_current_parity="PARENT_T0_REQUIRED")


def _reuse_reference(native_reuse, requests, input_identities, hidden_width):
    """Validate a source-bound old reference; never infer entry equivalence.

The parent verifies old artifact fullSHA/source/model/entry/runtime before
passing the binding. Here the fresh request/tokenization identities, saved
trace completeness and target schema are independently checked without a fit.
"""
    if not isinstance(native_reuse,dict): raise ZHookBoundary("T0_NATIVE_REUSE_SCHEMA")
    binding=native_reuse.get('binding',{})
    if (binding.get('native_source_sha256')!=NATIVE_Z_SHA or
            binding.get('case_ids')!=[r.get('case_id') for r in requests] or
            binding.get('batch_input_identities')!=input_identities or
            binding.get('entry_identity_verified') is not True):
        raise ZHookBoundary("T0_NATIVE_REUSE_IDENTITY")
    artifacts=binding.get('artifacts')
    if (not isinstance(artifacts,list) or not artifacts or
            any(not isinstance(a,dict) or not isinstance(a.get('path'),str) or
                not isinstance(a.get('sha256'),str) or len(a['sha256'])!=64 for a in artifacts)):
        raise ZHookBoundary("T0_NATIVE_REUSE_ARTIFACT_BINDING")
    targets=native_reuse.get('native_targets')
    if (not isinstance(targets,torch.Tensor) or targets.device.type!='cpu' or
            targets.dtype!=torch.float32 or tuple(targets.shape)!=(hidden_width,4) or
            not torch.isfinite(targets).all()):
        raise ZHookBoundary("T0_NATIVE_REUSE_TARGET_SCHEMA")
    rows=native_reuse.get('native_rows')
    if not isinstance(rows,list) or len(rows)!=4:raise ZHookBoundary("T0_NATIVE_REUSE_ROWS")
    for row in rows:
        losses,gradients=row.get('losses'),row.get('gradients')
        if (not isinstance(losses,list) or not 1<=len(losses)<=25 or
                not isinstance(gradients,list) or len(gradients)!=len(losses)-1):
            raise ZHookBoundary("T0_NATIVE_REUSE_TRACE_CARDINALITY")
        if any(x.get('iteration')!=i or any(not math.isfinite(float(x[k]))
                for k in ('loss','nll','kl','decay')) for i,x in enumerate(losses)):
            raise ZHookBoundary("T0_NATIVE_REUSE_LOSS_TRACE")
        if any(not isinstance(g,torch.Tensor) or g.device.type!='cpu' or
               g.dtype!=torch.float32 or tuple(g.shape)!=(hidden_width,) or
               not torch.isfinite(g).all() for g in gradients):
            raise ZHookBoundary("T0_NATIVE_REUSE_GRADIENT_TRACE")
    return targets.clone(),copy.deepcopy(rows),copy.deepcopy(binding)


def compare_native_z_paths(model, tok, native_module, hp, contexts, requests, *, layer=4,
                           native_reuse=None):
    """Exactly four fixed requests, three paths, no model writes/history append.

    Returns scalar receipt and local-only tensors. Batch16 configuration is
    exercised with this bounded four-request panel (actual batch size4); a
    successful panel is NOT a full16 peak-memory or exact-bitwise certificate.
    A parent-verified native_reuse reuses only the old unhooked targets/trace;
    the repaired cache paths and every trajectory/write gate remain mandatory.
    """
    if len(requests)!=4: raise ZHookBoundary("T0_REQUIRES_FIXED_FOUR_REQUESTS")
    native=native_module.compute_z
    source=inspect.getsource(inspect.getmodule(native))
    if hashlib.sha256(source.encode()).hexdigest()!=NATIVE_Z_SHA:
        raise ZHookBoundary("T0_NATIVE_SOURCE_SHA")
    requests=normalize_requests(requests)
    device=next(model.parameters()).device
    native_rows=[]
    def run_native():
        outputs=[]
        for request in requests:
            row=dict(losses=[],gradients=[])
            def sink(kind,it,delta,total,nll,kl,decay):
                if kind=="loss":row["losses"].append(dict(iteration=it,loss=float(total.detach()),
                    nll=float(nll.detach()),kl=float(kl.detach()),decay=float(decay.detach())))
                else:row["gradients"].append(delta.grad.detach().cpu().clone())
            observed=instrument_native_compute_z(native,sink)
            outputs.append(observed(model,tok,request,hp,layer,contexts).detach())
            native_rows.append(row)
        return torch.stack(outputs,dim=1)
    reuse_binding=None
    if native_reuse is None:
        native_targets,native_timing=_run_timed(run_native,device)
    else:
        identities=[prepare_batch(tok,[request],contexts,hp,native_module.find_fact_lookup_idx,device)
                    ['identity'] for request in requests]
        hidden_width=model.get_submodule(hp.lm_head_module).weight.shape[1]
        native_targets,native_rows,reuse_binding=_reuse_reference(
            native_reuse,requests,identities,hidden_width)
        native_timing=dict(seconds=0.,peak_allocated_bytes=None,peak_reserved_bytes=None,
                           reused=True,prior_seconds=reuse_binding.get('prior_native_seconds'),
                           new_unhooked_target_calls=0)
    cache_rows=[];cache_receipts=[]
    def run_single():
        outputs=[]
        for request in requests:
            z,receipt=compute_z_batch(model,tok,[request],hp,layer,contexts,
                native_module.find_fact_lookup_idx,config=ZHookConfig(1,True))
            outputs.append(z[:,0]);cache_receipts.append(receipt)
            cache_rows.append(dict(losses=receipt["losses"][0],gradients=receipt["gradients"][0]))
        return torch.stack(outputs,dim=1)
    single_targets,single_timing=_run_timed(run_single,device)
    (batched_targets,batched_receipt),batched_timing=_run_timed(lambda:compute_z_batch(
        model,tok,requests,hp,layer,contexts,native_module.find_fact_lookup_idx,
        config=ZHookConfig(16,True)),device)
    batched_rows=[dict(losses=batched_receipt["losses"][i],gradients=batched_receipt["gradients"][i]) for i in range(4)]
    receipt=dict(status="ACTUAL_T0_PATHS_OBSERVED",source_sha256=NATIVE_Z_SHA,
        fixed_case_ids=[r.get("case_id") for r in requests],native=native_timing,
        cache_head_batch1=single_timing,cache_head_batched=batched_timing,
        configured_batched_size=16,actual_technical_batched_size=4,
        full_batch16_peak_tested=False,
        batch1_comparison=_compare(native_rows,cache_rows,native_targets,single_targets),
        batched_comparison=_compare(native_rows,batched_rows,native_targets,batched_targets),
        model_writes=0,history_appends=0,new_native_requests_technical=8 if native_reuse is not None else 12,
        native_reference_reused=native_reuse is not None,native_reuse_binding=reuse_binding,
        production_targets_reused_from_technical=False,
        limitation="Four-request technical comparison; no scientific efficacy selection, no invented z/write tolerance")
    tensors=dict(native_targets=native_targets.cpu(),cache_batch1_targets=single_targets.cpu(),
        cache_batched_targets=batched_targets.cpu(),native_rows=native_rows,
        cache_batch1_receipts=cache_receipts,cache_batched_receipt=batched_receipt)
    return receipt,tensors
