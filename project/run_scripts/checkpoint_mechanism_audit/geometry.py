"""Read-only actual W/M geometry, shared-vector stored-history sketches.

CLI input JSON contains w0_snapshot, checkpoints[{batch,path}], and optional
commit_root. It creates scalar CSV/JSON only: never W/M/delta checkpoints.
Source files must have been bound by the parent's component input seal first.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import stat
import time
from pathlib import Path

import torch


WEIGHT_NAME = "model.layers.4.mlp.down_proj.weight"
W0_SHA256 = "9421d3f6dfae10b4663c07a41696f5c47e299e1f15d1728e162482f0245eb851"
CHECKPOINT_BATCHES = (1,5,10,20,30,40,50,60,70,80,90,100)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    # Exact original helper tensor_sha: dtype/shape prefix then C-order bytes.
    digest=hashlib.sha256(str((str(value.dtype),list(value.shape))).encode())
    digest.update(memoryview(value.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def file_identity(path: str | Path, expected_sha256: str | None = None,
                  expected_bytes: int | None = None) -> dict:
    path=Path(path);before=path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError(f"CHECKPOINT_NOT_REGULAR_NONSYMLINK:{path}")
    start=time.perf_counter();digest=hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda:handle.read(8<<20),b""):digest.update(data)
    after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError(f"INPUT_CHANGED_DURING_HASH:{path}")
    sha=digest.hexdigest()
    if expected_sha256 and sha != expected_sha256:raise ValueError(f"INPUT_FILE_SHA_MISMATCH:{path}")
    if expected_bytes is not None and before.st_size != expected_bytes:raise ValueError(f"INPUT_SIZE_MISMATCH:{path}")
    return dict(path=str(path),bytes=before.st_size,sha256=sha,mtime_ns=before.st_mtime_ns,
                device=before.st_dev,inode=before.st_ino,hash_seconds=time.perf_counter()-start,
                unchanged_during_hash=True,full_sha256=True)


def actual_difference(b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    if b.shape != a.shape:
        raise ValueError("weight shape mismatch")
    return b.double()-a.double()


def angle(a: torch.Tensor, b: torch.Tensor) -> dict:
    aa, bb = a.double().reshape(-1), b.double().reshape(-1)
    an, bn = torch.linalg.vector_norm(aa), torch.linalg.vector_norm(bb)
    if bool(an == 0) or bool(bn == 0):
        return dict(cosine=None, degrees=None, status="ZERO_NORM")
    cosine = float(torch.dot(aa, bb)/(an*bn))
    # Clamp only acos's representational domain, preserving raw cosine evidence.
    return dict(cosine=cosine, degrees=math.degrees(math.acos(min(1.,max(-1.,cosine)))),
                status="FINITE", acos_roundoff_clamp=not (-1 <= cosine <= 1))


def shared_rademacher(out_dim: int = 4096, probes: int = 256,
                      seed: int = 20260920) -> torch.Tensor:
    if probes != 256:
        raise ValueError("final sketch requires fixed256 probes; 64/128 diagnostics only")
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randint(0,2,(out_dim,probes),generator=g,dtype=torch.int8).double()*2-1)


def history_summary(m: torch.Tensor, *, requests: int | None = None,
                    chunk_rows: int = 512, seed: int = 20260920,
                    probe_count: int = 16) -> dict:
    """Chunked FP64 norms/skew and descriptive (not PSD-certified) probes."""
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("history shape must be square")
    start=time.perf_counter()
    d=m.shape[0]
    diag=m.diagonal().double()
    squared=skew_squared=0.
    finite=True
    g=torch.Generator(device="cpu").manual_seed(seed)
    probes=(torch.randint(0,2,(d,probe_count),generator=g,dtype=torch.int8).double()*2-1)/math.sqrt(d)
    probes=probes.to(m.device)
    quadratics=torch.zeros(probe_count,dtype=torch.float64,device=m.device)
    for i in range(0,d,chunk_rows):
        sl=slice(i,min(i+chunk_rows,d))
        block=m[sl,:].double()
        finite=finite and bool(torch.isfinite(block).all())
        squared+=float(block.square().sum())
        skew_squared+=float((block-m[:,sl].T.double()).square().sum())
        quadratics+=(probes[sl]*(block@probes)).sum(0)
    norm=math.sqrt(squared)
    trace=float(diag.sum())
    return dict(finite=finite, trace=trace, frobenius=norm,
        diagonal_min=float(diag.min()),diagonal_max=float(diag.max()),
        diagonal_mean=float(diag.mean()),diagonal_std=float(diag.std(unbiased=False)),
        symmetry_absolute=math.sqrt(skew_squared),
        symmetry_relative=math.sqrt(skew_squared)/norm if norm else None,
        trace_per_request=trace/requests if requests else None,
        quadratic_probes=quadratics.cpu().tolist(),minimum_quadratic_probe=float(quadratics.min()),
        negative_quadratic_probes=int((quadratics<0).sum()),probe_count=probe_count,
        probes_are_psd_or_rank_certificate=False,precision="float64",seconds=time.perf_counter()-start)


def stored_action_sketch(delta: torch.Tensor, m: torch.Tensor, z: torch.Tensor,
                         *, requests: int, chunk_rows: int = 512) -> dict:
    """J=mean(z.T D M D.T z), all256 shared output-space vectors."""
    if z.shape != (delta.shape[0],256) or delta.shape[1] != m.shape[0]:
        raise ValueError("invalid shared256 sketch shapes")
    start=time.perf_counter()
    dd=delta.double()
    q=dd.T@z.to(device=dd.device,dtype=torch.float64)
    values=torch.zeros(256,dtype=torch.float64,device=dd.device)
    for i in range(0,m.shape[0],chunk_rows):
        sl=slice(i,min(i+chunk_rows,m.shape[0]))
        values+=(q[sl]*(m[sl].to(dd.device,dtype=torch.float64)@q)).sum(0)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("NONFINITE_STORED_ACTION")
    dn2=float(dd.square().sum())
    trace=float(m.diagonal().double().sum())
    diagnostics=[]
    for count in (64,128,256):
        v=values[:count]
        mean=float(v.mean());se=float(v.std(unbiased=True)/math.sqrt(count))
        half=1.96*se
        near_zero=abs(mean)<=half
        diagnostics.append(dict(probes=count,mean=mean,mc_se=se,mc_halfwidth=half,
            relative_halfwidth=half/abs(mean) if mean>0 and not near_zero else None,
            relative_indicator_at_most_point1=(half/abs(mean)<=.1) if mean>0 and not near_zero else None,
            negative_mean=mean<0,exact_zero_mean=mean==0,
            near_zero_within_mc_halfwidth=near_zero,
            negative_probe_values=int((v<0).sum()),early_stop=False))
    final=diagnostics[-1]
    return dict(**final,J_per_request=final["mean"]/requests if requests else None,
        J_per_trace=final["mean"]/trace if trace else None,
        alignment_dimensionless=m.shape[0]*final["mean"]/(dn2*trace) if dn2 and trace else None,
        delta_squared_norm=dn2,history_trace=trace,
        normalization_zero_flags=dict(requests=requests==0,trace=trace==0,delta=dn2==0),
        diagnostics=diagnostics,vector_sha256=tensor_sha256(z),
        shared_vectors_required=True,adaptive_stopping=False,
        quantity="stored_history_quadratic_not_exact_key_energy",precision="float64",
        mc_precision_is_preservation_threshold=False,seconds=time.perf_counter()-start)


def activation_bookkeeping(entry_delta: torch.Tensor, interval_delta: torch.Tensor,
                           keys: torch.Tensor) -> dict:
    """Exact FP64 E K/D K energy identity; caller labels mean versus all-token."""
    b=entry_delta.double()@keys.double()
    v=interval_delta.double()@keys.double()
    base=b.square().sum(0);velocity=v.square().sum(0)
    cross=2*(b*v).sum(0)
    observed=(b+v).square().sum(0)-base
    return dict(entry_response_energy=base.cpu().tolist(),interval_response_energy=velocity.cpu().tolist(),
        cross_term=cross.cpu().tolist(),net_energy_change=observed.cpu().tolist(),
        maximum_closure_error=float((observed-cross-velocity).abs().max()),precision="float64")


def load_checkpoint(path: str | Path) -> tuple[torch.Tensor,torch.Tensor,dict]:
    payload=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    if set(payload) != {"weights","cache_c","metadata"}:
        raise ValueError(f"checkpoint top-level schema mismatch: {sorted(payload)}")
    if set(payload["weights"]) != {WEIGHT_NAME}:
        raise ValueError("checkpoint must contain exact singleton L4 weight inventory")
    w=payload["weights"][WEIGHT_NAME]
    cache=payload["cache_c"]
    if w.shape != (4096,14336) or w.dtype != torch.float32 or cache.shape != (1,14336,14336) or cache.dtype != torch.float32:
        raise ValueError("unexpected checkpoint tensor shape/dtype")
    return w,cache[0],payload["metadata"]


def load_w0(snapshot: str | Path) -> torch.Tensor:
    from safetensors import safe_open
    snapshot=Path(snapshot)
    index=json.loads((snapshot/"model.safetensors.index.json").read_text())
    with safe_open(snapshot/index["weight_map"][WEIGHT_NAME],framework="pt",device="cpu") as handle:
        raw=handle.get_tensor(WEIGHT_NAME)
    if raw.dtype != torch.bfloat16 or raw.shape != (4096,14336):
        raise ValueError("unexpected W0 safetensors dtype/shape")
    w0=raw.float()
    if tensor_sha256(w0) != W0_SHA256:
        raise ValueError("W0_FP32_TENSOR_IDENTITY_MISMATCH")
    return w0


def _flatten(row: dict, prefix: str = "") -> dict:
    result={}
    for key,value in row.items():
        k=prefix+key
        if isinstance(value,dict):result.update(_flatten(value,k+"_"))
        elif isinstance(value,list):result[k]=json.dumps(value,separators=(",",":"))
        else:result[k]=value
    return result


def run_geometry(input_map: dict, output_dir: str | Path) -> dict:
    """One pass through 12 states; computes all11 fixed256 nonzero-M intervals."""
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    with (output_dir/"geometry-input-map.json").open("x") as handle:json.dump(input_map,handle,indent=2)
    checkpoints=sorted(input_map["checkpoints"],key=lambda x:x["batch"])
    if tuple(x["batch"] for x in checkpoints) != CHECKPOINT_BATCHES:
        raise ValueError("checkpoint batch inventory differs from locked12")
    w0=load_w0(input_map["w0_snapshot"])
    z=shared_rademacher()
    w0norm=float(w0.double().norm())
    previous_w,previous_m,previous_e,previous_d,previous_batch=None,None,None,None,0
    rows=[dict(batch=0,requests=0,weight_sha256=tensor_sha256(w0),weight_norm=w0norm,
               cumulative_delta_norm=0.,relative_w0_norm=0.,history_trace=0.,history_frobenius=0.,
               history_status="COLD_M0_ZERO_NOT_LOADED_FROM_EDITED_CHECKPOINT")]
    start=time.perf_counter();input_identities=[]
    commit_root=Path(input_map["commit_root"]) if input_map.get("commit_root") else None
    commit_paths=input_map.get("commit_paths",{})
    def read_commit(b):
        path=Path(commit_paths[str(b)]) if str(b) in commit_paths else commit_root/f"B{b:03d}"/"commit.json"
        return json.loads(path.read_text())
    for item in checkpoints:
        batch=item["batch"]
        identity=file_identity(item["path"],item.get("sha256"),item.get("bytes"))
        input_identities.append(identity)
        w,m,metadata=load_checkpoint(item["path"])
        if not bool(torch.isfinite(w).all()):raise ValueError("NONFINITE_WEIGHT")
        weight_hash=tensor_sha256(w);cache_hash=tensor_sha256(m.unsqueeze(0))
        if metadata["batch"] != batch or metadata["state"] != dict(weights={WEIGHT_NAME:weight_hash},cache=cache_hash):
            raise ValueError("ACTUAL_CHECKPOINT_METADATA_STATE_IDENTITY_MISMATCH")
        if commit_root or commit_paths:
            endpoint_commit=read_commit(batch)
            if endpoint_commit["endpoint"] != metadata["state"]:raise ValueError("COMMIT_CHECKPOINT_STATE_IDENTITY_MISMATCH")
        e=actual_difference(w,w0)
        row=dict(batch=batch,requests=batch*100,checkpoint_path=str(item["path"]),
                 checkpoint_expected_sha256=item.get("sha256"),weight_sha256=weight_hash,
                 history_sha256=cache_hash,history_hash_shape="[1,14336,14336] original cache_c",
                 weight_norm=float(w.double().norm()),
                 cumulative_delta_norm=float(e.norm()),relative_w0_norm=float(e.norm())/w0norm,
                 history=history_summary(m,requests=batch*100),weight_difference_precision="float64")
        if previous_w is not None:
            d=actual_difference(w,previous_w)
            row.update(previous_batch=previous_batch,interval_delta_norm=float(d.norm()),
                       cumulative_angle=angle(previous_e,e),interval_vs_entry_angle=angle(d,previous_e),
                       adjacent_interval_angle=angle(d,previous_d) if previous_d is not None else {"status":"NO_PREVIOUS_INTERVAL"})
            dm=m.double()-previous_m.double()
            row["history_increment"]=history_summary(dm,requests=(batch-previous_batch)*100)
            del dm
            row["stored_action"]=stored_action_sketch(d,previous_m,z,requests=previous_batch*100)
            if commit_root or commit_paths:
                norms=[]
                for b in range(previous_batch+1,batch+1):
                    commit=read_commit(b)
                    norms.append(float(commit["layer_updates"][WEIGHT_NAME]["squared_norm"]))
                row["batch_squared_norm_sum"]=sum(norms)
                row["interval_update_total_cross_term"]=float(d.square().sum())-sum(norms)
                row["cross_term_status"]="ACTUAL_INTERVAL_MINUS_RECORDED_PER_BATCH_ACTUAL_SQUARED_NORMS"
            else:row["cross_term_status"]="BLOCKED_COMMIT_ROOT_NOT_AVAILABLE"
            previous_d=d
        else:
            row.update(previous_batch=0,interval_delta_norm=float(e.norm()),
                       stored_action_status="M0_ZERO_NOT_ONE_OF11_NONZERO_HISTORY_INTERVALS")
        rows.append(_flatten(row))
        with (output_dir/f"B{batch:03d}-geometry.json").open("x") as handle:json.dump(row,handle,indent=2,allow_nan=False)
        previous_w,previous_m,previous_e,previous_batch=w,m,e,batch
        print(json.dumps(dict(event="GEOMETRY_STATE_COMPLETE",batch=batch,elapsed=time.perf_counter()-start)),flush=True)
    names=list(dict.fromkeys(k for row in rows for k in row))
    with (output_dir/"checkpoint_geometry.csv").open("x",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=names);writer.writeheader();writer.writerows(rows)
    with (output_dir/"checkpoint-input-verification.json").open("x") as handle:json.dump(input_identities,handle,indent=2)
    receipt=dict(status="PASS",checkpoint_states=12,w0_states=1,geometry_rows=len(rows),history_intervals=11,shared_sketch_sha256=tensor_sha256(z),
        seed=20260920,final_sketch_vectors=256,precision="float64",threads=torch.get_num_threads(),
        seconds=time.perf_counter()-start,checkpoint_saved=False,new_weight_history_delta_files=0,
        input_hash_validation="ALL12_CURRENT_FULL_FILE_SHA256_MATCH_RECORDED_MANIFEST_AND_METADATA_COMMIT_TENSOR_IDENTITIES",
        checkpoint_file_bytes=sum(x["bytes"] for x in input_identities),
        hash_seconds=sum(x["hash_seconds"] for x in input_identities),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        random_probes_are_psd_certificate=False)
    with (output_dir/"geometry-receipt.json").open("x") as handle:json.dump(receipt,handle,indent=2)
    return receipt


def main() -> None:
    parser=argparse.ArgumentParser();choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--input-map");choice.add_argument("--contract")
    parser.add_argument("--source-map");parser.add_argument("--checkpoint-inventory")
    parser.add_argument("--output-dir",required=True);parser.add_argument("--threads",type=int,default=8)
    args=parser.parse_args()
    if args.threads < 1 or args.threads > 8:raise ValueError("CPU thread count must be1..8")
    torch.set_num_threads(args.threads)
    if args.input_map:
        input_map=json.loads(Path(args.input_map).read_text())
    else:
        if not args.source_map or not args.checkpoint_inventory:parser.error("contract mode requires source-map and checkpoint-inventory")
        contract=json.loads(Path(args.contract).read_text())
        source_map=json.loads(Path(args.source_map).read_text())
        inventory=json.loads(Path(args.checkpoint_inventory).read_text())
        source_root=contract["paths"]["server4_companion_root"]
        commits={str(b):source_map[f"{source_root}/B{b:03d}/commit.json"] for b in contract["inputs"]["current_batches"]}
        input_map=dict(w0_snapshot=contract["paths"]["model_snapshot"],commit_paths=commits,
            checkpoints=[dict(batch=x["indexed_batch"],path=x["path"],bytes=x["bytes"],sha256=x["historical_sha256"]) for x in inventory["paths"]],
            source_map_identity=file_identity(args.source_map),contract_identity=file_identity(args.contract),
            checkpoint_inventory_identity=file_identity(args.checkpoint_inventory))
    run_geometry(input_map,args.output_dir)


if __name__ == "__main__":main()
