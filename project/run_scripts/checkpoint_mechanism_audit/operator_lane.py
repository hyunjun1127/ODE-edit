"""Single-writer, model-free history/operator GPU lane after actual C01 gate.

Production CLI loads no Llama and performs no editing/target optimization.
Histories are explicitly partitioned by the owner across two one-GPU jobs.
Raw P is kept as stored, history LU/SVD are FP64, native dense anchors FP32.
Only key-response Y analysis caches are persisted; W/M/delta checkpoints never.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from .common import (ATTEMPT, CONTRACT, CPROOT, S4CELL, WEIGHT, EXECUTION_POLICY, mapped, read,
                     sha256, source_map, tensor_sha, write_csv, write_json)
from .geometry import load_w0
from .operators import (HistoryOperator, counterfactual_summary,
                        factor_from_response, fitting_transfer,
                        fixed_probe_scores, native_dense,
                        native_mode_decomposition, norm_discrepancy,
                        reconstruction_metrics)


NATIVE_ENTRY={1:0,2:1,6:5,11:10,21:20,51:50,91:90}
PILOTS=(0,1,10,100)
ALL_HISTORIES=(0,1,5,10,20,30,40,50,60,70,80,90,100)
TARGET_NATIVE={entry:batch for batch,entry in NATIVE_ENTRY.items()}


def sync(device: torch.device) -> None:
    if device.type == "cuda":torch.cuda.synchronize(device)


def save_analysis_tensor(path: Path, value: dict) -> dict:
    """Whitelist cached operator coordinates, never restorable edited state."""
    if not set(value)<= {"Y","history_batch","bank_order","bank_sha256","source_dtype","analysis_dtype"}:
        raise ValueError("UNAPPROVED_OPERATOR_CACHE_FIELDS")
    with path.open("xb") as handle:torch.save(value,handle)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path),
                artifact_kind="ANALYSIS_KEY_RESPONSE_NOT_MODEL_CHECKPOINT")


def validate_bank(bank: dict, *, input_dim: int = 14336, output_dim: int = 4096,
                  native_count: int = 100, geometry_count: int = 512) -> dict:
    expected={f"B{b:03d}" for b in NATIVE_ENTRY}
    if set(bank.get("native",{})) != expected:
        raise ValueError("NATIVE_BANK_MUST_CONTAIN_EXACT_SEVEN_BATCHES")
    if bank.get("w0_tensor_sha256") != CONTRACT["identity"]["w0_fp32_tensor_sha256"]:
        raise ValueError("BANK_W0_IDENTITY_MISMATCH")
    records=[]
    for name,capture in [(k,bank["native"][k]) for k in sorted(expected)]+[("geometry",bank["geometry"])]:
        n=geometry_count if name=="geometry" else native_count
        h0=capture.get("h0W0",capture.get("h0"))
        for key,tensor,shape in (("K",capture["K"],(input_dim,n)),
                                 ("bare_K",capture["bare_K"],(input_dim,n)),
                                 ("h0W0",h0,(output_dim,n))):
            if not isinstance(tensor,torch.Tensor) or tensor.shape != shape or tensor.dtype != torch.float32:
                raise ValueError(f"BANK_SHAPE_DTYPE:{name}:{key}")
            if not bool(torch.isfinite(tensor).all()):raise ValueError(f"BANK_NONFINITE:{name}:{key}")
        if len(capture["ids"]) != n or len(set(capture["ids"])) != n:
            raise ValueError(f"BANK_ID_COUNT_OR_DUPLICATE:{name}")
        if capture.get("native_group_sizes") != [1,5] or capture.get("key_weights") != [.5,.1,.1,.1,.1,.1]:
            raise ValueError(f"NATIVE_GROUP_WEIGHTS:{name}")
        if "h_entry" in capture:
            if capture["h_entry"].shape != (output_dim,n) or capture["h_entry"].dtype != torch.float32:
                raise ValueError(f"ENTRY_H_SHAPE_DTYPE:{name}")
            if not bool(torch.isfinite(capture["h_entry"]).all()):raise ValueError("NONFINITE_ENTRY_H")
        records.append(dict(name=name,requests=n,ids=list(capture["ids"]),
            K_sha256=tensor_sha(capture["K"]),bare_K_sha256=tensor_sha(capture["bare_K"]),
            h0_sha256=tensor_sha(h0),h_entry_present="h_entry" in capture))
    native_ids=[i for name in sorted(expected) for i in bank["native"][name]["ids"]]
    if len(set(native_ids)) != 7*native_count:raise ValueError("NATIVE_BANK_CROSS_BATCH_ID_DUPLICATE")
    if set(native_ids)&set(bank["geometry"]["ids"]):raise ValueError("GEOMETRY_NATIVE_ID_OVERLAP")
    return dict(status="PASS",members=records,total_native_requests=len(native_ids),
                geometry_requests=geometry_count)


def make_rhs_bank(bank: dict, device: torch.device) -> tuple[torch.Tensor,dict[str,slice]]:
    captures=[("geometry",bank["geometry"])] + [(f"B{b:03d}",bank["native"][f"B{b:03d}"]) for b in NATIVE_ENTRY]
    slices={};chunks=[];start=0
    for name,capture in captures:
        k=capture["K"].to(device=device,dtype=torch.float64)
        slices[name]=slice(start,start+k.shape[1]);start+=k.shape[1];chunks.append(k)
    return torch.cat(chunks,dim=1),slices


def validate_dependencies(histories: list[int], gate: dict,
                          pilot_receipts: list[dict]) -> dict:
    if not histories or len(histories)!=len(set(histories)) or any(h not in ALL_HISTORIES for h in histories):
        raise ValueError("INVALID_OR_DUPLICATE_HISTORY_PARTITION")
    return dict(historical_C01=gate.get('C01'),histories=histories,
                extension_requires_all_four_pilots=False,**EXECUTION_POLICY)


def source_targets(batch: int, capture: dict, mapping: dict) -> tuple[torch.Tensor,dict]:
    path=mapped(f"{S4CELL}/B{batch:03d}/native-targets.pt",mapping)
    payload=torch.load(path,weights_only=True,map_location="cpu")
    identities=payload["identities"];values=payload["values"]
    if [x["case_id"] for x in identities] != capture["ids"]:
        raise ValueError(f"TARGET_ORDER_MISMATCH:B{batch:03d}")
    if [tensor_sha(x) for x in values] != [x["sha256"] for x in identities]:
        raise ValueError(f"TARGET_TENSOR_IDENTITY_MISMATCH:B{batch:03d}")
    z=torch.stack(values,dim=1)
    if z.dtype!=torch.float32 or not bool(torch.isfinite(z).all()):raise ValueError("TARGET_DTYPE_NONFINITE")
    return z,dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path),
                  requests=len(values),target_reoptimized=False)


def residual_from_capture(z: torch.Tensor, capture: dict, entry_weight: torch.Tensor,
                          w0: torch.Tensor, entry_batch: int) -> tuple[torch.Tensor,dict]:
    """Use actual physical FP32 h_entry; affine path is separately identified."""
    h0=capture.get("h0W0",capture.get("h0"))
    affine=h0.double()+(entry_weight.double()-w0.double())@capture["bare_K"].double()
    entry_hash=tensor_sha(entry_weight)
    physical="h_entry" in capture
    if physical:
        if capture.get("entry_batch") != entry_batch or capture.get("entry_weight_sha256") != entry_hash:
            raise ValueError("PHYSICAL_ENTRY_H_BINDING_MISMATCH")
        h=capture["h_entry"]
        gate=dict(max_absolute_error=float((h.double()-affine).abs().max()),
                  numerical_validation='NOT_ESTABLISHED')
        kind="PHYSICAL_FP32_ENTRY_H"
    elif entry_batch == 0:
        h=h0;gate=None;kind="PHYSICAL_W0_CAPTURE"
    else:
        # Kept as an explicitly reconstructed observation, not original FP32 h.
        h=affine.float();gate=None;kind="AFFINE_FP64_THEN_FP32_RECONSTRUCTED_ENTRY_H"
    residual=z-h
    return residual,dict(entry_batch=entry_batch,entry_weight_sha256=entry_hash,
        h_kind=kind,original_fp32_entry_capture=(physical or entry_batch==0),
        affine_physical_parity=gate,residual_dtype=str(residual.dtype),
        h_sha256=tensor_sha(h),residual_sha256=tensor_sha(residual),
        bare_residual_key_not_mean_key=True)


def _skew(matrix: torch.Tensor) -> dict:
    mm=matrix.double();den=float(mm.norm());absolute=float((mm-mm.T).norm())
    return dict(absolute=absolute,relative=absolute/den if den else None,zero_reference=den==0)


def compute_history(projector: torch.Tensor, history: torch.Tensor,
                    bank: dict, residuals: dict[int,torch.Tensor], history_batch: int,
                    *, actual_b1: torch.Tensor | None = None,
                    original_dense_inputs: dict | None = None,
                    w0_norm: float | None = None) -> dict[str,Any]:
    """Pure tensor analysis engine, injectable small CPU fixtures for tests.

    original_dense_inputs={P,M} must retain original FP32 operands; the history
    factor itself intentionally uses explicitly recorded FP64 operands.
    """
    device=projector.device;sync(device);start=time.perf_counter()
    if projector.dtype!=torch.float64 or history.dtype!=torch.float64:
        raise ValueError("HISTORY_ANALYSIS_FACTORIZATION_REQUIRES_LOCKED_FP64")
    kbank,slices=make_rhs_bank(bank,device)
    sync(device);bank_done=time.perf_counter()
    operator=HistoryOperator(projector,history,lambda_write=1.)
    sync(device);factor_done=time.perf_counter()
    y,solve=operator.solve_bank(kbank,verify=True)
    sync(device);solve_done=time.perf_counter()
    result:dict[str,Any]=dict(history_batch=history_batch,
        status="PASS",
        status_scope="HISTORY_OPERATOR_NOT_ALL_NATIVE_DEMAND_CLAIMS",
        solve=solve,bank_prepare_seconds=bank_done-start,
        factor_seconds=factor_done-bank_done,solve_verification_seconds=solve_done-factor_done,
        operator_dtype="float64",native_dense_dtype="float32",projector_symmetrized=False,
        raw_skew_H=_skew(operator.matrix),probe_rows=[],mode_rows=[],group_rows=[],
        reconstruction_rows=[],counterfactual_rows=[],native_block_diagnostics=[])
    result.update(EXECUTION_POLICY)
    solve['residual'].pop('passed',None)
    if not bool(torch.isfinite(y).all()):raise ValueError('NONFINITE_HISTORY_RESPONSE')
    gslice=slices["geometry"]
    probe_rows=fixed_probe_scores(projector,kbank[:,gslice],y[:,gslice])
    for row,case_id in zip(probe_rows,bank["geometry"]["ids"]):
        score=row["raw_score"]
        denominator=1.+score
        row.update(history_batch=history_batch,case_id=case_id,
            singleton_fitting_gain=score/denominator if denominator!=0 else None,
            singleton_gain_zero_denominator=denominator==0,
            history_solve_max_column_residual=solve["residual"]["max_relative"])
    result["probe_rows"]=probe_rows
    # Each100-request native block gets its OWN S/B even though the LU/RHS bank
    # is shared. Probe512 is never passed through a512-request native solver.
    for batch in NATIVE_ENTRY:
        name=f"B{batch:03d}";sl=slices[name];k=kbank[:,sl]
        s,b=factor_from_response(k,y[:,sl])
        result["native_block_diagnostics"].append(dict(batch=batch,requests=k.shape[1],
            raw_S_skew=_skew(s),B_norm=float(b.norm()),native_blocks_merged=False))
        if history_batch in PILOTS and batch in (1,2):
            if batch in residuals:
                controls=counterfactual_summary(k,residuals[batch].to(device,dtype=torch.float64),b,w0_norm)
                for row in controls:row.update(history_batch=history_batch,target_batch=batch,
                    residual_reference_entry=NATIVE_ENTRY[batch],status="PASS")
                result["counterfactual_rows"]+=controls
            else:
                result["counterfactual_rows"].append(dict(history_batch=history_batch,target_batch=batch,
                    status="BLOCKED",reason="TARGET_OR_ENTRY_H_COMPONENT_UNAVAILABLE",factual_edit_arm=False))
        if TARGET_NATIVE.get(history_batch) != batch:continue
        if batch not in residuals:
            result["reconstruction_rows"].append(dict(target_batch=batch,entry_batch=history_batch,
                status="BLOCKED",reason="TARGET_OR_ENTRY_H_COMPONENT_UNAVAILABLE"))
            continue
        r=residuals[batch].to(device,dtype=torch.float64)
        factor_delta=r@b.T
        reconstructed_kind="ACTUAL_ENDPOINT_REFERENCED_B1_UNCERTIFIED" if batch==1 else "RECONSTRUCTED_NATIVE_WRITE"
        row=dict(target_batch=batch,entry_batch=history_batch,provenance=reconstructed_kind,
            factor_delta_norm=float(factor_delta.norm()),
            relative_w0_norm=float(factor_delta.norm())/w0_norm if w0_norm else None,
            history_solve_max_column_residual=solve["residual"]["max_relative"],
            factor_dtype="float64",dense_dtype="float32",status="PASS")
        row["raw_key_singular_values"]=torch.linalg.svdvals(k).cpu().tolist()
        row["projected_key_singular_values"]=torch.linalg.svdvals(projector@k).cpu().tolist()
        row["raw_S_singular_values"]=torch.linalg.svdvals(s).cpu().tolist()
        transfer_operator=fitting_transfer(torch.eye(s.shape[0],dtype=s.dtype,device=s.device),s)
        row["fitting_transfer_singular_values"]=torch.linalg.svdvals(transfer_operator).cpu().tolist()
        row["key_spectrum_is_not_writer_gain"]=True
        response=factor_delta@k
        target_error=norm_discrepancy(response-r,r,1e-3)
        row.update(realized_target_error_relative=target_error["max_relative"],
                   realized_target_error_absolute=target_error["max_absolute"],
                   realized_target_error_is_performance_gate=False)
        transfer=fitting_transfer(r,s)
        row["raw_transfer_formula_absolute_error"]=float((transfer-response).norm())
        reference=None
        row.update(EXECUTION_POLICY)
        row['dense_factor_validation']='SKIPPED_USER_DIRECTED'
        if batch==1 and actual_b1 is not None:
            reference=actual_b1.to(device,dtype=torch.float64)
            row['actual_factor_absolute_error']=float((reference-factor_delta).norm())
        elif batch==1:
            row.update(status="BLOCKED",reason="ACTUAL_B1_DELTA_MISSING")
        # A failed anchor withholds its gain attribution but leaves the already
        # valid history/probe results intact (dependency-local failure).
        if row["status"]=="PASS":
            decomposition=native_mode_decomposition(b,r,reference)
            for mode in decomposition.pop("modes"):
                mode.update(target_batch=batch,entry_batch=history_batch,provenance=reconstructed_kind)
                result["mode_rows"].append(mode)
            for group in decomposition.pop("groups"):
                group.update(target_batch=batch,entry_batch=history_batch,provenance=reconstructed_kind)
                result["group_rows"].append(group)
            row["decomposition"]=decomposition
        result["reconstruction_rows"].append(row)
    sync(device)
    result.update(Y=y.detach().cpu(),bank_order=list(slices),seconds=time.perf_counter()-start,
                  downstream_algebra_and_verification_seconds=time.perf_counter()-solve_done,
                  factorization_count=1,new_z_optimization_calls=0,checkpoint_saved=False)
    return result


def _csv_rows(rows: list[dict]) -> list[dict]:
    return [{k:json.dumps(v,sort_keys=True,separators=(",",":")) if isinstance(v,(list,dict)) else v
             for k,v in row.items()} for row in rows]


def run_lane(histories: list[int], bank_path: str | Path, gate_path: str | Path,
             output: str | Path, pilot_receipts: list[str | Path], *, device_name: str = "cuda") -> dict:
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();stage="DEPENDENCIES";done=[]
    try:
        gate=read(gate_path);pilot_records=[read(p) for p in pilot_receipts]
        dependencies=validate_dependencies(histories,gate,pilot_records)
        source_sha=sha256(ATTEMPT/"inputs/source-map.json")
        if gate.get("source_map_sha256") != source_sha:raise ValueError("GATE_SOURCE_MAP_IDENTITY_MISMATCH")
        bank_path=Path(bank_path);bank_sha=sha256(bank_path)
        bank=torch.load(bank_path,map_location="cpu",weights_only=True)
        bank_validation=validate_bank(bank)
        bank_receipt=read(bank_path.parent/"C02.json")
        if bank.get("status") != "PASS" or bank_receipt.get("status") != "PASS":
            raise ValueError("C02_KEY_CAPTURE_PREREQUISITE_NOT_PASS")
        if bank_receipt["bank"]["sha256"] != bank_sha:
            raise ValueError("C02_KEY_BANK_IDENTITY_MISMATCH")
        if bank.get("gate_sha256") != sha256(gate_path):
            raise ValueError("BANK_PARENT_GATE_MISMATCH")
        write_json(output/"input-lock.json",dict(dependencies=dependencies,bank_validation=bank_validation,
            gate=dict(path=str(gate_path),sha256=sha256(gate_path)),
            bank=dict(path=str(bank_path),bytes=bank_path.stat().st_size,sha256=bank_sha),
            source_map_sha256=source_sha,history_analysis_dtype="float64",native_dense_dtype="float32",
            near_degenerate_diagnostic_gap=1e-3,checkpoint_saved=False,
            seed=20260920,permutations=20,lambda_write=1.,
            pilot_receipts=[dict(path=str(p),sha256=sha256(p)) for p in pilot_receipts]))
        stage="INPUT_LOAD"
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        device=torch.device(device_name)
        if device.type!="cuda":raise ValueError("PRODUCTION_OPERATOR_LANE_EXPECTS_ALLOCATED_CUDA_DEVICE")
        w0=load_w0(CONTRACT["paths"]["model_snapshot"]);w0_norm=float(w0.double().norm())
        p_all=torch.load(CONTRACT["paths"]["projector"],weights_only=True,map_location="cpu",mmap=True)
        p=p_all[0]
        if tensor_sha(p.unsqueeze(0))!=CONTRACT["identity"]["projector_selected_sha256"]:raise ValueError("P4_IDENTITY_MISMATCH")
        mapping=source_map();residuals={};residual_receipts={};target_receipts={}
        from .model_runtime import checkpoint
        for batch,entry in NATIVE_ENTRY.items():
            capture=bank["native"][f"B{batch:03d}"]
            try:
                z,target_receipts[batch]=source_targets(batch,capture,mapping)
                entry_weight=w0 if entry==0 else checkpoint(entry)["weights"][WEIGHT]
                residuals[batch],residual_receipts[batch]=residual_from_capture(z,capture,entry_weight,w0,entry)
                residual_receipts[batch]["status"]="PASS"
            except (ValueError,FileNotFoundError,KeyError) as exc:
                residual_receipts[batch]=dict(status="BLOCKED",reason=repr(exc),entry_batch=entry,
                    independent_history_operator_still_permitted=True)
        write_json(output/"demand-input-receipt.json",dict(targets=target_receipts,residuals=residual_receipts))
        cp1=checkpoint(1);actual_b1=cp1["weights"][WEIGHT].double()-w0.double()
        del cp1
        p32=p.to(device);p64=p32.double()
        for history_batch in histories:
            stage=f"HISTORY_{history_batch:03d}";folder=output/f"M{history_batch:03d}"
            folder.mkdir(exist_ok=False)
            obj=None if history_batch==0 else checkpoint(history_batch)
            m32=torch.zeros_like(p32) if obj is None else obj["cache_c"][0].to(device)
            m64=m32.double()
            history_start=time.perf_counter()
            try:
                result=compute_history(p64,m64,bank,residuals,history_batch,actual_b1=actual_b1,
                    original_dense_inputs=dict(P=p32,M=m32),w0_norm=w0_norm)
            except (RuntimeError,ValueError) as exc:
                # Numerical failure is scoped to this history. Independent
                # histories remain eligible; an OOM is technical, not repair by
                # changed dtype/threshold. CUDA device-fault errors still bubble.
                if "device-side assert" in str(exc) or "illegal memory access" in str(exc):raise
                result=dict(history_batch=history_batch,status="FAILED",seconds=time.perf_counter()-history_start,
                    status_scope="HISTORY_OPERATOR_NOT_ALL_NATIVE_DEMAND_CLAIMS",
                    error=repr(exc),traceback=traceback.format_exc(),
                    failure_type="TECHNICAL_OOM" if "out of memory" in str(exc).lower() else "NUMERICAL_OR_TECHNICAL_OPERATOR_FAILURE",
                    probe_rows=[],mode_rows=[],group_rows=[],reconstruction_rows=[],counterfactual_rows=[])
            # Exact-native FP32 anchor claim additionally needs a physical h
            # reference at B90. No target or forward regeneration is done here.
            for row in result["reconstruction_rows"]:
                batch=row["target_batch"]
                row["demand_binding"]=residual_receipts[batch]
                if batch==91 and not residual_receipts[batch].get("original_fp32_entry_capture",False):
                    row.update(status="BLOCKED",reason="B91_PHYSICAL_ENTRY_H_NOT_CAPTURED")
                    result["mode_rows"]=[r for r in result["mode_rows"] if r["target_batch"]!=91]
                    result["group_rows"]=[r for r in result["group_rows"] if r["target_batch"]!=91]
            result["native_demand_statuses"]=[dict(target_batch=r["target_batch"],status=r["status"],reason=r.get("reason"))
                                               for r in result["reconstruction_rows"]]
            result["counterfactual_status"]="PASS" if all(r.get("status")=="PASS" for r in result["counterfactual_rows"]) else "BLOCKED"
            cache=None
            cache_start=time.perf_counter()
            if "Y" in result:
                yy=result.pop("Y")
                cache=save_analysis_tensor(folder/"key-response-cache.pt",dict(Y=yy,
                    history_batch=history_batch,bank_order=result.pop("bank_order"),bank_sha256=bank_sha,
                    source_dtype="float32",analysis_dtype="float64"))
                del yy
            result["cache_io_seconds"]=time.perf_counter()-cache_start
            for filename,key in (("fixed_probe_history.csv","probe_rows"),("native_write_modes.csv","mode_rows"),
                                 ("near_degenerate_groups.csv","group_rows"),("reconstruction.csv","reconstruction_rows"),
                                 ("counterfactuals.csv","counterfactual_rows")):
                write_csv(folder/filename,_csv_rows(result.pop(key)))
            result.update(cache=cache,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                members=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(folder.iterdir()) if p.is_file()])
            write_json(folder/"terminal.json",result);done.append(result)
            print(json.dumps(dict(event="HISTORY_TERMINAL",history=history_batch,status=result["status"],seconds=result["seconds"])),flush=True)
            del m32,m64,obj;gc.collect();torch.cuda.empty_cache()
        terminal=dict(status="PASS" if all(x["status"]=="PASS" for x in done) else "FAILED",
            histories=done,elapsed_seconds=time.perf_counter()-start,
            history_factorizations_verified=sum(x.get("factorization_count",0) for x in done),
            attempted_histories=len(done),failed_history_factorization_count="NOT_INFERRED",
            slurm_job=os.environ.get("SLURM_JOB_ID"),gpu=torch.cuda.get_device_name(device),
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            source_map_sha256=source_sha,checkpoint_saved=False,z_optimizations=0,model_loads=0,
            new_edit_chains=0,unperformed_histories=[h for h in histories if h not in {x["history_batch"] for x in done}])
        terminal.update(EXECUTION_POLICY)
        write_json(output/"terminal.json",terminal)
        return terminal
    except BaseException as exc:
        write_json(output/"failure.json",dict(status="FAILED_TECHNICAL",stage=stage,error=repr(exc),
            traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start,
            completed_histories=[r["history_batch"] for r in done],checkpoint_saved=False,
            numerical_failures_are_not_capacity_failure=True))
        raise


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--histories",required=True)
    parser.add_argument("--bank",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--gate",required=True);parser.add_argument("--pilot-receipts",nargs="*",default=[])
    args=parser.parse_args()
    run_lane([int(x) for x in args.histories.split(",")],args.bank,args.gate,args.output,args.pilot_receipts)


if __name__=="__main__":main()
