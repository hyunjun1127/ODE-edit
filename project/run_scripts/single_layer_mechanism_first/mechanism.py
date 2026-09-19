"""Post-selection native-writer algebra, spectra and bounded interventions.

No model forward, target fitting, candidate selection or history append occurs
here. Torch device placement is caller-owned. Model-sized tensors/factors in
the returned objects are local artifacts; ``receipt``/``mode_rows`` are compact.
The actual raw-P writer and the orthogonal-projector ideal are kept distinct.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Iterable, Mapping

import torch

SEED = 20260919
IDEAL_CG_RELATIVE_TOLERANCE = 1e-10
IDEAL_CG_MAX_ITERATIONS = 2000


class MechanismTechnicalError(ValueError):
    pass


def _tensor(value: Any, name: str, *, ndim: int = 2) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim != ndim or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise MechanismTechnicalError(f"{name}: finite floating rank-{ndim} tensor required")
    return value.detach()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_binding(value: torch.Tensor) -> dict:
    """One boundary SHA, row-streamed; never called inside document loops."""
    value = _tensor(value, "tensor identity")
    header = {"shape": list(value.shape), "dtype": str(value.dtype)}
    digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode()+b"\n")
    for start in range(0,value.shape[0],128):
        digest.update(value[start:start+128].contiguous().cpu().numpy().tobytes())
    return {**header,"sha256_header_and_bytes":digest.hexdigest(),
            "bytes":value.numel()*value.element_size(),"device_at_binding":str(value.device)}


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.double()))


def _relative(error: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = _norm(reference)
    numerator = _norm(error)
    return numerator/denominator if denominator else (0.0 if numerator == 0 else float("inf"))


def extract_native_capture(result: Mapping[str,Any]) -> dict:
    """Decode source-exact NativeSingletonFitter capture without new z calls.

    compute_ks and module output captures are [requests,width]; compute_z is
    one [out] tensor per request. The native repeat_interleave is preserved.
    """
    captures = result["captures"]
    if len(captures.get("compute_ks",[])) != 1 or len(captures.get("get_module_input_output_at_words",[])) != 1:
        raise MechanismTechnicalError("expected one native key and current-output capture")
    zs = captures.get("compute_z",[])
    if not zs:
        raise MechanismTechnicalError("native z capture absent")
    z = torch.stack([_tensor(x,"native z",ndim=1) for x in zs],dim=1)
    k = _tensor(captures["compute_ks"][0],"native keys").T.contiguous()
    current = _tensor(captures["get_module_input_output_at_words"][0],"native current outputs").T.contiguous()
    if z.dtype != torch.float32 or current.dtype != torch.float32 or k.dtype != torch.float32:
        raise MechanismTechnicalError("native capture FP32 required")
    if current.shape != z.shape or k.shape[1] % z.shape[1]:
        raise MechanismTechnicalError("native capture coordinate/cardinality mismatch")
    repeat = k.shape[1]//z.shape[1]
    # Subtract BEFORE repeat and preserve FP32 native arithmetic.
    r = (z-current).repeat_interleave(repeat,dim=1)
    return {"K":k,"R":r,"z":z,"current_output":current,
            "receipt":{"source_capture_calls":{k:len(v) for k,v in captures.items()},
                       "request_count":len(zs),"key_columns":k.shape[1],"repeat_factor":repeat,
                       "coordinates":"K:input_by_native_mean_key;R:output_by_request",
                       "residual_route":"FP32(z-current_output).repeat_interleave",
                       "new_z_calls":0,"source":result.get("receipt",{}).get("source")}}


@dataclass
class LowRankComponent:
    left: torch.Tensor
    right: torch.Tensor
    receipt: dict

    def action(self, keys: torch.Tensor) -> torch.Tensor:
        """One global weight component applied to any input, not input patching."""
        if keys.ndim != 2 or keys.shape[0] != self.right.shape[0]:
            raise MechanismTechnicalError("component/key coordinate mismatch")
        k = keys.to(device=self.left.device,dtype=torch.float64)
        return self.left@(self.right.T@k)

    def materialize(self) -> torch.Tensor:
        return self.left@self.right.T


@dataclass
class WriterDiagnostics:
    receipt: dict
    mode_rows: list[dict]
    mode_factors: list[LowRankComponent]
    actual_delta: torch.Tensor
    algebra_map: torch.Tensor
    ideal_map: torch.Tensor | None


def workspace_estimate(input_dimension: int, output_dimension: int,
                       requests: int, allowed_dimension: int) -> dict:
    n,m,b,q = input_dimension,output_dimension,requests,allowed_dimension
    return {"status":"COMPONENT_PLAN_NOT_MEASURED_PEAK_OR_TOTAL_UPPER_BOUND",
            "native_system_FP32_bytes":4*n*n,"native_system_and_factor_workspace_additional":True,
            "raw_projector_and_history_FP32_bytes":8*n*n,
            "dense_RHS_optional_FP32_bytes":4*n*m,
            "actual_delta_FP64_bytes":8*n*m,"allowed_basis_FP64_bytes":8*n*q,
            "entry_and_native_FP32_bytes":8*n*m,
            "algebra_delta_and_permutation_control_FP64_bytes":16*n*m,
            "nonzero_history_FP64_operator_additional_bytes":8*n*n,
            "thin_native_map_FP32_bytes":4*n*b,"thin_ideal_map_FP64_bytes":8*n*b,
            "ideal_CG_five_thin_buffers_FP64_bytes":5*8*q*b,
            "request_Gram_FP64_bytes":8*b*b,
            "full_svd":False,"dense_reference_covariance":False,
            "caller_model_and_other_live_buffers_additional":True}


def _ideal_inverse_action(x: torch.Tensor, u: torch.Tensor, history: torch.Tensor,
                          l2: float, *, history_keys: torch.Tensor | None = None) -> tuple[torch.Tensor,dict]:
    """C^-1 X using the actual stored Chist operator, no q-by-q matrix.

    Optional exact retained keys are only a preconditioner. They never replace
    FP32 history's actual accumulated entries in the operator or residual.
    """
    if not bool(torch.count_nonzero(history)):
        return x/l2,{"status":"ZERO_HISTORY_EXACT_SCALAR_INVERSE","iterations":0,
                     "relative_residual_per_rhs":[0.0]*x.shape[1],"history_factor_used_as_operator":False}
    history64=history.double()
    def operator(v):
        # Cast one existing history matrix; do not construct U.T Chist U.
        return l2*v+u.T@(history64@(u@v))
    preconditioner_name = "RIDGE_DIAGONAL"
    if history_keys is None:
        def precondition(v):return v/l2
    else:
        hk = _tensor(history_keys,"history keys").to(device=x.device,dtype=torch.float64)
        if hk.shape[0] != u.shape[0]:
            raise MechanismTechnicalError("history key coordinates")
        h = u.T@hk
        small = torch.eye(h.shape[1],device=x.device,dtype=torch.float64)+h.T@h/l2
        factor = torch.linalg.cholesky(small)
        def precondition(v):
            return v/l2-h@torch.cholesky_solve(h.T@v,factor)/(l2*l2)
        preconditioner_name = "RETAINED_KEY_WOODBURY_PRECONDITIONER_ONLY"
    result = torch.zeros_like(x)
    residual = x.clone()
    z = precondition(residual)
    direction = z.clone()
    rz = (residual*z).sum(0)
    norm_rhs = torch.linalg.vector_norm(x,dim=0)
    active = norm_rhs>0
    iterations = torch.zeros(x.shape[1],dtype=torch.int64,device=x.device)
    for count in range(1,IDEAL_CG_MAX_ITERATIONS+1):
        if not bool(active.any()):break
        ad = operator(direction)
        denominator = (direction*ad).sum(0)
        if bool((denominator[active]<=0).any()) or not bool(torch.isfinite(denominator).all()):
            raise MechanismTechnicalError("ideal metric CG nonpositive/nonfinite curvature")
        alpha = torch.zeros_like(rz);alpha[active]=rz[active]/denominator[active]
        result += direction*alpha
        residual -= ad*alpha
        iterations[active]=count
        measured = torch.linalg.vector_norm(residual,dim=0)
        next_active = measured>IDEAL_CG_RELATIVE_TOLERANCE*norm_rhs
        next_active &= norm_rhs>0
        new_z=precondition(residual)
        new_rz=(residual*new_z).sum(0)
        beta=torch.zeros_like(rz)
        beta[next_active]=new_rz[next_active]/rz[next_active]
        direction=new_z+direction*beta
        direction[:,~next_active]=0
        rz=new_rz;active=next_active
    exact_residual = operator(result)-x
    relative = torch.linalg.vector_norm(exact_residual,dim=0)/norm_rhs.clamp_min(torch.finfo(torch.float64).tiny)
    if not bool(torch.isfinite(result).all()) or bool((relative>IDEAL_CG_RELATIVE_TOLERANCE).any()):
        raise MechanismTechnicalError("ideal metric CG residual unresolved; spectrum not established")
    return result,{"status":"ACTUAL_HISTORY_MATRIX_FREE_SOLVE_RESIDUAL_VERIFIED",
                   "iterations_per_rhs":iterations.cpu().tolist(),"iterations":int(iterations.max()),
                   "relative_residual_per_rhs":relative.cpu().tolist(),
                   "relative_residual_ceiling":IDEAL_CG_RELATIVE_TOLERANCE,
                   "preconditioner":preconditioner_name,"history_factor_used_as_operator":False,
                   "dense_reduced_C_created":False}


def analyze_writer(entry_weight: Any, native_weight: Any, keys: Any, residuals: Any,
                   projector: Any, history: Any, *, l2: float = 1.0,
                   allowed_basis: Any | None = None, history_keys: Any | None = None,
                   native_update: Any | None = None, replay_dense: bool = False,
                   source_identity: Mapping[str,Any] | None = None,
                   device: str | torch.device | None = None) -> WriterDiagnostics:
    """Source-order raw-P map plus actual FP32 endpoint and ideal-metric audit.

    ``native_update`` may be a preserved pre-add source-exact dense solve result;
    it is NOT reconstructed from WN-entry. If absent, ``replay_dense`` explicitly
    authorizes one diagnostic dense RHS solve, counted separately from fitting.
    """
    inputs={name:_tensor(value,name) for name,value in
            (("entry",entry_weight),("native",native_weight),("K",keys),("R",residuals),("P_raw",projector),("C_hist",history))}
    if any(t.dtype!=torch.float32 for t in inputs.values()):
        raise MechanismTechnicalError("native writer inputs must retain FP32 bytes")
    w,wn,k,r,p,ch=(inputs[name] for name in ("entry","native","K","R","P_raw","C_hist"))
    m,n=w.shape;b=k.shape[1]
    if wn.shape!=w.shape or k.shape[0]!=n or r.shape!=(m,b) or p.shape!=(n,n) or ch.shape!=(n,n) or b==0:
        raise MechanismTechnicalError("writer coordinate/cardinality mismatch")
    if l2!=1.0:
        raise MechanismTechnicalError("contract native L2 must be exactly 1")
    target_device=torch.device(device) if device is not None else k.device
    start=time.monotonic();_sync(target_device)
    timing={};bindings={name:tensor_binding(t) for name,t in inputs.items()}
    timing["boundary_hash_seconds"]=time.monotonic()-start
    w,wn,k,r,p,ch=(inputs[name].to(target_device) for name in ("entry","native","K","R","P_raw","C_hist"))
    actual=wn.double()-w.double()
    begin=time.monotonic()
    # Exact BLUE expression/order, not a symmetrized or inverse-substituted writer.
    system=p@(k@k.T+ch)+l2*torch.eye(n,dtype=torch.float32,device=target_device)
    pk=p@k
    if not bool(torch.isfinite(system).all()):raise MechanismTechnicalError("nonfinite native system")
    mapped=torch.linalg.solve(system,pk)
    _sync(target_device);timing["native_system_and_thin_map_solve_seconds"]=time.monotonic()-begin
    algebra=r.double()@mapped.double().T
    source_order={"operator":"P_raw @ (K @ K.T + C_hist) + L2 * eye_FP32",
                  "native_rhs":"(P_raw @ K) @ R.T",
                  "map_rhs":"P_raw @ K","source_dtype":"float32",
                  "map_diagnostic_solve_calls":1,"native_fit_or_z_calls":0,
                  "map_relative_solve_residual":_relative(system.double()@mapped.double()-pk.double(),pk.double()),
                  "algebra_vs_actual_delta_relative_error":_relative(algebra-actual,actual),
                  "algebra_vs_actual_delta_max_abs":float((algebra-actual).abs().max()),
                  "algebra_association_not_bitexact_native_rhs":True}
    if native_update is not None and replay_dense:
        raise MechanismTechnicalError("choose preserved native update OR diagnostic replay")
    update=None
    begin=time.monotonic()
    if native_update is not None:
        update=_tensor(native_update,"preserved native update").to(target_device)
        if update.dtype!=torch.float32 or update.shape!=w.shape:
            raise MechanismTechnicalError("native update shape/dtype")
        source_order["dense_source_update_origin"]="CALLER_PRESERVED_SOLVE_OUTPUT"
        source_order["diagnostic_dense_rhs_solve_calls"]=0
    elif replay_dense:
        update=torch.linalg.solve(system,pk@r.T).T
        source_order["dense_source_update_origin"]="NEW_DIAGNOSTIC_SOURCE_ORDER_SOLVE"
        source_order["diagnostic_dense_rhs_solve_calls"]=1
    else:
        source_order["dense_source_update_origin"]="NOT_AVAILABLE"
        source_order["diagnostic_dense_rhs_solve_calls"]=0
    if update is not None:
        replay_endpoint=w+update
        source_order.update(replay_endpoint_equal=torch.equal(replay_endpoint,wn),
                            replay_endpoint_max_abs=float((replay_endpoint-wn).abs().max()),
                            source_update_vs_actual_delta_relative_error=_relative(update.double()-actual,actual),
                            dense_relative_solve_residual=_relative(system.double()@update.T.double()-(pk@r.T).double(),(pk@r.T).double()),
                            source_update_binding=tensor_binding(update))
    _sync(target_device);timing["optional_dense_replay_and_audit_seconds"]=time.monotonic()-begin
    actual_realized=actual@k.double()
    receipt={"schema_version":1,"source_identity":dict(source_identity or {}),
             "source_identity_status":"CALLER_BOUND_REQUIRES_EXTERNAL_SOURCE_SEAL" if source_identity else "NOT_BOUND",
             "input_bindings":bindings,
             "source_order":source_order,"native_actual_delta_norm":_norm(actual),
             "native_weight_norm":_norm(wn),"entry_weight_norm":_norm(w),
             "actual_delta_to_entry_weight_norm":_norm(actual)/_norm(w) if _norm(w) else None,
             "actual_current_realization_relative_error":_relative(actual_realized-r.double(),r.double()),
             "native_mean_key_columns":b,"correction_full_token_keys_are_different_artifact":True,
             "history_nonzero":bool(torch.count_nonzero(ch)),"model_forwards":0,"new_z_calls":0,
             "standalone_method_time_inclusion":False,"observer_diagnostic_cost_separate":True}
    ideal_map=None;mode_rows=[];modes=[]
    begin=time.monotonic()
    if allowed_basis is not None:
        u=_tensor(allowed_basis,"allowed basis").to(device=target_device,dtype=torch.float64)
        if u.shape[0]!=n:raise MechanismTechnicalError("allowed basis coordinate mismatch")
        x=u.T@k.double()
        y,cg=_ideal_inverse_action(x,u,ch,l2,history_keys=history_keys)
        gram=x.T@y
        asymmetry=_relative(gram-gram.T,gram)
        # Eigh requires symmetric input; averaging occurs only in the FP64
        # ideal analysis Gram, never in the native writer operator.
        symmetry_error=float((gram-gram.T).abs().max())
        if asymmetry>1e-8:
            raise MechanismTechnicalError("ideal request Gram symmetry unresolved")
        eigen,v=torch.linalg.eigh((gram+gram.T)*.5)
        order=torch.argsort(eigen,descending=True);eigen,v=eigen[order],v[:,order]
        spectral_roundoff=128*torch.finfo(torch.float64).eps*max(1,b)*max(1.,float(eigen.abs().max()))
        if bool((eigen < -spectral_roundoff).any()):raise MechanismTechnicalError("ideal metric Gram not PSD")
        eigen=eigen.clamp_min(0)
        ideal_map=u@(y@v/(1+eigen)[None,:])@v.T
        leakage_sq=0.
        for start in range(0,m,128):
            block=actual[start:start+128]
            leak=block-(block@u)@u.T
            leakage_sq+=float((leak*leak).sum())
        receipt["allowed_leakage_relative"]=leakage_sq**.5/_norm(actual) if _norm(actual) else 0.
        receipt["ideal_metric"]={"definition":"C=L2 I+U.T C_hist U; Gram=X.T C^-1 X",
                                  "C_inverse_action":cg,"allowed_dimension":u.shape[1],
                                  "request_Gram_dimension":b,"dense_full_SVD":False,
                                  "Gram_asymmetry_relative":asymmetry,"Gram_asymmetry_max_abs":symmetry_error,
                                  "Gram_symmetrization":"ANALYSIS_ONLY_FP64_ROUNDOFF",
                                  "eigenvalue_negative_roundoff_ceiling":spectral_roundoff,
                                  "ideal_map_vs_actual_raw_map_relative_error":_relative(ideal_map-mapped.double(),mapped.double()),
                                  "U_orthonormality_provenance":"CALLER_PRIOR_VERIFIED_ALLOWED_BASIS;NO_DENSE_U_GRAM_RECHECK",
                                  "raw_P_not_replaced_in_writer":True}
        for index in range(b):
            value=float(eigen[index]);sigma=value**.5
            left=r.double()@v[:,index:index+1]
            right=mapped.double()@v[:,index:index+1]
            ideal_right=ideal_map@v[:,index:index+1]
            loading=float((left*left).sum())
            cost=loading*float((right*right).sum())
            row={"mode":index,"sigma":sigma,"target_loading_squared":loading,
                 "ridge_gain_sigma_over_one_plus_sigma_squared":sigma/(1+value),
                 "realized_gain_sigma_squared_over_one_plus_sigma_squared":value/(1+value),
                 "inverse_sigma_interpolation_quantity":1/sigma if sigma>0 else None,
                 "inverse_sigma_is_native_gain":False,
                 "raw_map_mode_weight_norm_squared":cost,
                 "ideal_mode_weight_norm_squared":loading*float((ideal_right*ideal_right).sum()),
                 "raw_vs_ideal_right_relative_error":_relative(right-ideal_right,ideal_right),
                 "rank":int(cost>0)}
            mode_rows.append(row)
            modes.append(LowRankComponent(left.cpu(),right.cpu(),row.copy()))
        # Sum of raw map modes equals R B.T, not FP32 endpoint-add rounding.
        receipt["mode_sum_target"]="R @ solve(M,P_raw K).T; actual FP32 discrepancy reported separately"
        receipt["workspace_estimate"]=workspace_estimate(n,m,b,u.shape[1])
    else:
        receipt["ideal_metric"]={"status":"NOT_AVAILABLE_ALLOWED_BASIS_MISSING"}
        receipt["allowed_leakage_relative"]=None
        receipt["workspace_estimate"]=workspace_estimate(n,m,b,0)
    _sync(target_device);timing["ideal_metric_spectrum_modes_seconds"]=time.monotonic()-begin
    begin=time.monotonic()
    permutation=torch.randperm(b,generator=torch.Generator(device="cpu").manual_seed(SEED)).to(target_device)
    control=r.double()[:,permutation]@mapped.double().T
    receipt["R_column_permutation_control"]={"seed":SEED,"permutation":permutation.cpu().tolist(),
        "fixed_K_and_map":True,"new_model_or_edit_arm":False,
        "original_R_norm":_norm(r),"permuted_R_norm":_norm(r[:,permutation]),
        "original_algebra_delta_norm":_norm(algebra),"permuted_algebra_delta_norm":_norm(control),
        "permuted_vs_original_delta_norm":_norm(control-algebra)}
    _sync(target_device);timing["fixed_K_permutation_control_seconds"]=time.monotonic()-begin
    begin=time.monotonic()
    actual_cpu=actual.cpu();map_cpu=mapped.detach().cpu()
    ideal_cpu=ideal_map.detach().cpu() if ideal_map is not None else None
    _sync(target_device);timing["return_tensor_D2H_seconds"]=time.monotonic()-begin
    timing["total_inclusive_seconds"]=time.monotonic()-start
    receipt["timing"]=timing
    receipt["timer_nesting"]="total_inclusive contains every phase; never sum total with phase timers"
    return WriterDiagnostics(receipt,mode_rows,modes,actual_cpu,map_cpu,ideal_cpu)


def stream_reference_action(entry_minus_W0: Any, native_minus_entry: Any,
                            document_keys: Iterable[Any], *, expected_documents: int=512,
                            native_gradient_factors: Iterable[Any] | None=None,
                            document_ids: Iterable[str] | None=None,
                            device: str | torch.device | None=None) -> dict:
    """Exact activation energy identity, all valid input keys per document.

    Optional factors A[out,tokens] report <A,Delta K>, the signed derivative
    of the caller's bound scalar; they do not imply finite margin improvement.
    """
    e,d=_tensor(entry_minus_W0,"E"),_tensor(native_minus_entry,"Delta")
    if e.shape!=d.shape or expected_documents<1:raise MechanismTechnicalError("action shape/count")
    target=torch.device(device) if device is not None else e.device
    e,d=e.to(device=target,dtype=torch.float64),d.to(device=target,dtype=torch.float64)
    ids=list(document_ids) if document_ids is not None else None
    if ids is not None and (len(ids)!=expected_documents or len(set(ids))!=len(ids)):
        raise MechanismTechnicalError("reference action ID cardinality/duplicate")
    factors=iter(native_gradient_factors) if native_gradient_factors is not None else None
    rows=[];start=time.monotonic()
    for index,raw in enumerate(document_keys):
        if index>=expected_documents:raise MechanismTechnicalError("excess reference action documents")
        k=_tensor(raw,"reference full-input keys").to(device=target,dtype=torch.float64)
        if k.shape[0]!=e.shape[1] or k.shape[1]<1:raise MechanismTechnicalError("reference action key shape")
        b,v=e@k,d@k
        old=float((b*b).sum());step=float((v*v).sum());cross=2*float((b*v).sum())
        after=float(((b+v)*(b+v)).sum())
        row={"document_index":index,"valid_input_tokens":k.shape[1],"entry_energy":old,
             "native_step_energy":step,"twice_entry_step_inner":cross,
             "native_net_energy":after,"net_minus_entry":after-old,
             "identity_rhs":cross+step,"identity_absolute_error":abs((after-old)-(cross+step)),
             "identity_relative_error":abs((after-old)-(cross+step))/max(1.,abs(after),abs(old),abs(cross),step)}
        if ids is not None:row["document_id"]=ids[index]
        if factors is not None:
            try:a=_tensor(next(factors),"reference scalar activation gradient").to(device=target,dtype=torch.float64)
            except StopIteration as exc:raise MechanismTechnicalError("missing reference factors") from exc
            if a.shape!=v.shape:raise MechanismTechnicalError("reference factor shape")
            row["signed_native_scalar_derivative"]=float((a*v).sum())
        rows.append(row)
    if len(rows)!=expected_documents:raise MechanismTechnicalError("missing reference action documents")
    if factors is not None:
        try:next(factors)
        except StopIteration:pass
        else:raise MechanismTechnicalError("excess reference factors")
    _sync(target)
    names=("entry_energy","native_step_energy","twice_entry_step_inner","native_net_energy","net_minus_entry")
    return {"documents":len(rows),"valid_input_tokens":sum(row["valid_input_tokens"] for row in rows),
            "document_order_binding":"CALLER_SEALED_IDS_NO_PAYLOAD_REHASH" if ids is not None else "NOT_BOUND_BY_THIS_FUNCTION",
            "document_order_sha256":hashlib.sha256(json.dumps(ids,separators=(",", ":")).encode()).hexdigest() if ids is not None else None,
            "rows":rows,"document_normalized_means":{name:sum(row[name]/row["valid_input_tokens"] for row in rows)/len(rows) for name in names},
            "identity_max_relative_error":max(row["identity_relative_error"] for row in rows),
            "dense_reference_covariance_created":False,"model_forwards":0,
            "seconds":time.monotonic()-start,"activation_identity_not_locality_claim":True}


def global_interventions(diagnostics: WriterDiagnostics, *, selection_seal: str,
                          native_parity_confirmed: bool,
                          allowed_basis: Any | None=None) -> tuple[list[LowRankComponent],dict]:
    """Two declared source modes + norm/rank-matched deterministic controls.

    Selection: largest target loading, then largest raw-map write cost among
    remaining nonzero modes; ties use ascending mode index. No panel outcomes
    enter this selection. Parent applies each same global component to every
    post-selection reference/N/Current panel input and keeps costs separate.
    """
    if not selection_seal or not native_parity_confirmed:
        raise MechanismTechnicalError("sealed selection and established native parity required")
    valid=[row for row in diagnostics.mode_rows if row["rank"] and row["raw_map_mode_weight_norm_squared"]>0]
    selected=[]
    if valid:
        selected.append(min(valid,key=lambda x:(-x["target_loading_squared"],x["mode"]))["mode"])
        remaining=[row for row in valid if row["mode"]!=selected[0]]
        if remaining:selected.append(min(remaining,key=lambda x:(-x["raw_map_mode_weight_norm_squared"],x["mode"]))["mode"])
    out=[];generator=torch.Generator(device="cpu").manual_seed(SEED)
    u=_tensor(allowed_basis,"control allowed basis").cpu().double() if allowed_basis is not None else None
    for number,index in enumerate(selected):
        mode=diagnostics.mode_factors[index]
        out.append(LowRankComponent(mode.left.clone(),mode.right.clone(),
            {**mode.receipt,"intervention_id":f"source_mode_{index}","kind":"POSTSEAL_GLOBAL_SOURCE_COMPONENT",
             "selection_seal":selection_seal,"subtraction":"H_native - Delta_component K_input",
             "selection_criterion":"largest_target_loading" if number==0 else "largest_remaining_mode_write_cost"}))
        left=torch.randn((mode.left.shape[0],1),generator=generator,dtype=torch.float64)
        if u is None:
            right=torch.randn((mode.right.shape[0],1),generator=generator,dtype=torch.float64)
        else:
            right=u@torch.randn((u.shape[1],1),generator=generator,dtype=torch.float64)
        norm=_norm(mode.left)*_norm(mode.right)
        if _norm(left)==0 or _norm(right)==0:
            raise MechanismTechnicalError("matched random control has zero norm")
        left=left/_norm(left);right=right*norm/_norm(right)
        out.append(LowRankComponent(left,right,{"intervention_id":f"random_match_mode_{index}",
            "kind":"POSTSEAL_GLOBAL_MATCHED_RANDOM_CONTROL","matched_source_mode":index,
            "seed":SEED,"rank":1,"weight_norm":norm,"selection_seal":selection_seal,
            "allowed_basis_used":u is not None,"panel_outcomes_used_for_selection":False}))
    return out,{"selection_seal":selection_seal,"interventions":len(out),"maximum":4,
                 "selected_mode_indices":selected,"native_parity_confirmed":True,
                 "postselection_only":True,"method_candidate_or_commit":False,
                 "insufficient_nonzero_modes":len(selected)<2}
