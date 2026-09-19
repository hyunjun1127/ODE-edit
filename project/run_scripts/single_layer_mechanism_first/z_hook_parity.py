"""Bounded native/cache-head/batched z T0; invocation belongs to task runner.

No source/global patch, no submit/model load. The caller supplies an already
loaded FP32/eager pinned model and the four preselected Current requests.
"""
from __future__ import annotations

import hashlib
import inspect
import time

import torch

from .z_hook import (NATIVE_Z_SHA, ZHookBoundary, ZHookConfig, compute_z_batch,
                     instrument_native_compute_z, normalize_requests)


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
    rows=[]
    for index,(ref,other) in enumerate(zip(reference,candidate)):
        equal_steps=len(ref["losses"])==len(other["losses"])
        pairs=list(zip(ref["losses"],other["losses"]))
        gradients=list(zip(ref["gradients"],other["gradients"]))
        grad_relative=[]
        for left,right in gradients:
            denom=float(left.double().norm())
            error=float((left.double()-right.double()).norm())
            grad_relative.append(error/denom if denom else (0. if error==0 else float("inf")))
        delta=targets[:,index].double()-reference_targets[:,index].double()
        rows.append(dict(request_index=index, loss_steps_reference=len(ref["losses"]),
                         loss_steps_candidate=len(other["losses"]), loss_steps_equal=equal_steps,
                         adam_steps_equal=len(ref["gradients"])==len(other["gradients"]),
                         max_NLL_abs=max((abs(a["nll"]-b["nll"]) for a,b in pairs),default=0.),
                         max_total_loss_abs=max((abs(a["loss"]-b["loss"]) for a,b in pairs),default=0.),
                         max_gradient_relative=max(grad_relative,default=0.),
                         compared_gradient_steps=len(gradients),
                         z_bitwise_equal=torch.equal(targets[:,index],reference_targets[:,index]),
                         z_max_abs=float(delta.abs().max()),z_l2=float(delta.norm()),
                         z_relative=float(delta.norm()/reference_targets[:,index].double().norm())))
    # These thresholds are inherited and fixed before observing T0. No invented
    # vector/write ceiling; their raw differences require parent actual checks.
    passed=all(r["loss_steps_equal"] and r["adam_steps_equal"] and
               r["max_NLL_abs"]<=1e-4 and r["max_gradient_relative"]<=1e-4 for r in rows)
    return dict(pass_inherited_NLL_gradient_and_stop_gate=passed, rows=rows,
                thresholds=dict(NLL_abs=1e-4,direct_cached_gradient_relative=1e-4,
                                stop_iteration="exact",z_vector="REPORTED_NO_INVENTED_CEILING"),
                actual_write_current_parity="PARENT_T0_REQUIRED")


def compare_native_z_paths(model, tok, native_module, hp, contexts, requests, *, layer=4):
    """Exactly four fixed requests, three paths, no model writes/history append.

    Returns scalar receipt and local-only tensors. Batch16 configuration is
    exercised with this bounded four-request panel (actual batch size4); a
    successful panel is NOT a full16 peak-memory or exact-bitwise certificate.
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
    native_targets,native_timing=_run_timed(run_native,device)
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
        model_writes=0,history_appends=0,new_native_requests_technical=12,
        production_targets_reused_from_technical=False,
        limitation="Four-request technical comparison; no scientific efficacy selection, no invented z/write tolerance")
    tensors=dict(native_targets=native_targets.cpu(),cache_batch1_targets=single_targets.cpu(),
        cache_batched_targets=batched_targets.cpu(),native_rows=native_rows,
        cache_batch1_receipts=cache_receipts,cache_batched_receipt=batched_receipt)
    return receipt,tensors
