"""In-memory instrumentation of the frozen non-BLUE native AlphaEdit function.

The original function still computes targets, keys, residual divisors, physical
writes and final timestamp-history appends.  Only its module-global references
are temporarily wrapped; no shared source file or global torch function changes.
This wrapper is single-thread/non-reentrant by construction.  Caller owns entry
snapshots and physical branch restoration; no W/M/RNG checkpoint is written.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import marshal
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from .common import digest, file_sha, rng_get, rng_preserved, rng_set, save, tensor_file, tensor_sha
from .interventions import InterventionError, history_operand


LAYERS = (4, 5, 6, 7, 8)
_INSTRUMENT_LOCK = threading.Lock()


class NativeWriterError(RuntimeError):
    pass


def _sync(value: torch.Tensor | None = None) -> None:
    if value is not None and value.device.type == "cuda":
        torch.cuda.synchronize(value.device)


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _finite(value: torch.Tensor, name: str) -> None:
    if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
        raise NativeWriterError(f"NONFINITE_OR_NON_FP32:{name}")


def _rng_sha(value: Any) -> str:
    # Hash values, not pickle/storage pointers: cloned identical states bind.
    import numpy as np
    def canonical(item):
        if isinstance(item, torch.Tensor):
            return {"tensor_sha256": tensor_sha(item)}
        if isinstance(item, np.ndarray):
            return {"ndarray_sha256": hashlib.sha256(item.tobytes(order="C")).hexdigest(),
                    "shape": list(item.shape), "dtype": str(item.dtype)}
        if isinstance(item, (tuple, list)):
            return [canonical(x) for x in item]
        if isinstance(item, dict):
            return {str(k): canonical(v) for k, v in item.items()}
        if isinstance(item, np.generic):
            return item.item()
        return item
    return digest(canonical(value))


def _request_binding(requests: Sequence[Mapping[str, Any]]) -> tuple[list[dict], str]:
    normalized = copy.deepcopy(list(requests))
    for row in normalized:
        target = row["target_new"]["str"]
        if not target:
            raise NativeWriterError("EMPTY_NATIVE_TARGET")
        if target[0] != " ":
            row["target_new"]["str"] = " " + target
    ids = [row["case_id"] for row in normalized]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise NativeWriterError("NATIVE_BATCH_REQUIRES_100_UNIQUE_ORDERED_REQUESTS")
    return normalized, digest(normalized)


class _Delegating:
    def __init__(self, underlying: Any, **overrides: Any):
        self._underlying = underlying
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._underlying, name)


class NativeWriter:
    """One call = one restored own-entry native/fork batch, five history appends.

    ``current(layer, rt)`` must capture all locked H512 keys at that branch's
    current prefix.  ``stamp`` maps physical layers to validated [d_in,512]
    timestamp keys.  ``bank_ids`` is the ordered locked H512 identity.  Supplying
    those arguments attests timestamp parity was already established by E0;
    an explicit ``rt.timestamp_keys_validated is False`` blocks subtraction.

    Callback is ``stage_callback(stage, rt, info)`` for entry, z, W4..W8, history.
    It runs with RNG preserved and instrumentation bypassed.  Any temporary
    model/history mutations inside it MUST be restored by the caller.
    """
    def __init__(self, rt: Any):
        self.rt = rt

    def write(self, requests: Sequence[Mapping[str, Any]], out: str | Path, *,
              branch: str = "NATIVE", shared_z: Mapping[str, Any] | None = None,
              stamp: Mapping[int, torch.Tensor] | None = None,
              current: Callable[[int, Any], torch.Tensor] | None = None,
              bank_ids: Sequence[Any] | None = None,
              stage_callback: Callable[[str, Any, Mapping[str, Any]], None] | None = None
              ) -> dict[str, Any]:
        if not _INSTRUMENT_LOCK.acquire(blocking=False):
            raise NativeWriterError("NATIVE_INSTRUMENTATION_CONCURRENT_OR_REENTRANT")
        try:
            return self._write(requests, out, branch=branch, shared_z=shared_z,
                               stamp=stamp, current=current, bank_ids=bank_ids,
                               stage_callback=stage_callback)
        finally:
            _INSTRUMENT_LOCK.release()

    def _write(self, requests, out, *, branch, shared_z, stamp, current, bank_ids, stage_callback):
        rt = self.rt
        native = rt.native
        original = native.apply_AlphaEdit_to_model
        if original.__globals__ is not vars(native):
            raise NativeWriterError("NATIVE_FUNCTION_GLOBALS_NOT_MODULE_BOUND")
        if branch not in {"NATIVE", "SHAM", "H5", "H6", "H56", "MASS56"}:
            raise NativeWriterError("UNAPPROVED_HISTORY_BRANCH")
        if list(rt.hp.layers) != list(LAYERS) or rt.hp.blue is not False or rt.hp.L2 != 10:
            raise NativeWriterError("ORIGINAL_NONBLUE_L4_L8_L2_10_REQUIRED")
        if rt.M is None or rt.M.ndim != 3 or rt.P.shape != rt.M.shape or rt.M.shape[0] != 5:
            raise NativeWriterError("NATIVE_M_P_SHAPE_INVALID")
        if rt.M.dtype != torch.float32 or rt.P.dtype != torch.float32:
            raise NativeWriterError("NATIVE_M_P_NOT_FP32")
        if native.CONTEXT_TEMPLATES_CACHE != rt.contexts:
            raise NativeWriterError("NATIVE_CONTEXT_CACHE_MISMATCH")
        normalized, requests_sha = _request_binding(requests)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            raise NativeWriterError("OUTPUT_ATTEMPT_MUST_BE_EMPTY_CREATE_ONCE")
        entry_rng = rng_get()
        entry_signature = rt.signature()
        binding = {
            "entry": entry_signature, "requests_sha256": requests_sha,
            "hparams_sha256": digest(vars(rt.hp)), "contexts_sha256": digest(rt.contexts),
            "native_code_sha256": hashlib.sha256(marshal.dumps(original.__code__)).hexdigest(),
            "entry_rng_sha256": _rng_sha(entry_rng), "z_layer": 8,
        }
        binding_sha = digest(binding)
        if shared_z is not None:
            if shared_z.get("binding_sha256") != binding_sha:
                raise NativeWriterError("SHARED_Z_ENTRY_REQUEST_CONFIG_SOURCE_RNG_MISMATCH")
            if shared_z.get("case_ids") != [r["case_id"] for r in normalized]:
                raise NativeWriterError("SHARED_Z_CASE_ORDER_MISMATCH")
            shared_targets = shared_z.get("targets")
            if (not isinstance(shared_targets, torch.Tensor) or shared_targets.ndim != 2
                    or shared_targets.shape[1] != 100 or "rng_after_z" not in shared_z):
                raise NativeWriterError("SHARED_Z_INCOMPLETE")
        originals = {name: getattr(native, name) for name in
                     ("torch", "compute_z", "compute_ks", "get_module_input_output_at_words")}
        original_solve = originals["torch"].linalg.solve
        original_get = originals["get_module_input_output_at_words"]
        original_ks = originals["compute_ks"]
        original_z = originals["compute_z"]
        stages: list[dict[str, Any]] = []
        factors: dict[int, dict[str, Any]] = {}
        final_keys: dict[int, torch.Tensor] = {}
        history_receipts: list[dict[str, Any]] = []
        target_values: list[torch.Tensor] = []
        target_receipts: list[dict[str, Any]] = []
        counters = {"target_new": 0, "target_reused": 0, "native_solves": 0,
                    "diagnostic_solves": 0, "write_key_calls": 0,
                    "history_key_calls": 0, "history_appends": 0,
                    "final_l8_residual_calls": 0}
        events: list[dict[str, Any]] = []
        phase = "entry"
        bypass = False
        pending: dict[str, Any] | None = None
        pending_history: int | None = None
        z_bundle: dict[str, Any] | None = None
        final_l8: dict[str, Any] | None = None
        t_program = time.monotonic()

        def timed(kind, start, **metadata):
            value = {"kind": kind, "seconds": time.monotonic() - start, **metadata}
            events.append(value)
            return value["seconds"]

        def notify(stage: str, **info):
            nonlocal bypass, phase
            phase = f"stage:{stage}"
            row = {"stage": stage, "elapsed_seconds": time.monotonic() - t_program, **info}
            stages.append(row)
            save(out / "stages" / f"{len(stages):02d}-{stage}.json", row)
            if stage_callback is not None:
                before = time.monotonic()
                bypass = True
                try:
                    with rng_preserved():
                        stage_callback(stage, rt, {**row, "factors": factors, "z": z_bundle})
                finally:
                    bypass = False
                    timed("stage_observer", before, stage=stage)
                rt.assert_nonselected()

        def finish_z():
            nonlocal z_bundle, phase
            if z_bundle is not None:
                return
            if len(target_values) != 100:
                raise NativeWriterError("Z_PHASE_NOT_EXACTLY_100_REQUESTS")
            phase = "seal-z"
            if shared_z is not None:
                rng_set(shared_z["rng_after_z"])
            after_z_rng = rng_get()
            z_bundle = dict(binding_sha256=binding_sha, binding=binding,
                            case_ids=[r["case_id"] for r in normalized],
                            targets=torch.stack(target_values, dim=1), rng_after_z=after_z_rng)
            if shared_z is None:
                artifact = tensor_file(out / "native-z.pt", {
                    "case_ids": z_bundle["case_ids"], "targets": z_bundle["targets"],
                    "binding_sha256": binding_sha,
                })
            else:
                artifact = save(out / "native-z-reuse.json", {
                    "binding_sha256": binding_sha, "source_artifact": shared_z.get("artifact"),
                    "source_case_ids": z_bundle["case_ids"], "target_new": 0, "target_reused": 100,
                })
            z_bundle["artifact"] = artifact
            notify("z", binding_sha256=binding_sha, target_new=counters["target_new"],
                   target_reused=counters["target_reused"], artifact=artifact)

        def wrapped_z(model, tok, request, hp, layer, contexts):
            nonlocal phase
            if bypass:
                return original_z(model, tok, request, hp, layer, contexts)
            phase = "native-target"
            index = len(target_values)
            if index >= 100 or layer != 8 or digest(request) != digest(normalized[index]):
                raise NativeWriterError("NATIVE_TARGET_CALL_ORDER_OR_LAYER_MISMATCH")
            t = time.monotonic()
            if shared_z is None:
                value = original_z(model, tok, request, hp, layer, contexts)
                counters["target_new"] += 1
            else:
                value = shared_z["targets"][:, index].to(rt.weights[8].device).clone()
                counters["target_reused"] += 1
            _sync(value)
            seconds = timed("native_target" if shared_z is None else "native_target_reuse",
                            t, request_index=index, case_id=request["case_id"])
            _finite(value, "native_z")
            copied = _cpu(value)
            target_values.append(copied)
            row = {"index": index, "case_id": request["case_id"], "seconds": seconds,
                   "sha256": tensor_sha(copied), "new_fit": shared_z is None}
            if shared_z is None:
                row["artifact"] = tensor_file(out / "targets" / f"{index:03d}.pt", {
                    "target": copied, "case_id": request["case_id"], "binding_sha256": binding_sha,
                })
            target_receipts.append(row)
            return value

        def flush_history():
            nonlocal pending_history
            if pending_history is None:
                return
            layer = pending_history
            row = history_receipts[-1]
            row["post_M_sha256"] = tensor_sha(rt.M[layer - 4])
            row["append_completed"] = True
            row["validation"] = "ORIGINAL_SOURCE_ONCE_PATH_AND_KEY_CALL_COUNT_PLUS_M_HASHES"
            row["independent_full_M_reconstruction"] = "NOT_PERFORMED"
            counters["history_appends"] += 1
            pending_history = None

        def flush_write():
            nonlocal pending, phase
            if pending is None:
                return
            work = pending
            pending = None
            layer = work["layer"]
            factor = factors[layer]
            post = _cpu(rt.weights[layer])
            _finite(post, "post_write_weight")
            factor["post_weight_sha256"] = tensor_sha(post)
            actual = post - factor["pre_weight"]
            factor["actual_delta_norm"] = float(torch.linalg.vector_norm(actual))
            factor["actual_minus_solve_delta_norm"] = float(
                torch.linalg.vector_norm(actual - factor["delta"]))
            factor["physical_write_applied"] = True
            del post, actual
            phase = "diagnostic-response-factor"
            t = time.monotonic()
            try:
                C = original_solve(work["A"], work["PK"])
                _sync(C)
                _finite(C, "diagnostic_C")
                counters["diagnostic_solves"] += 1
                C_cpu = _cpu(C)
                G = factor["K"].double().T @ C_cpu.double()
                identity = torch.eye(G.shape[0], dtype=torch.float64)
                complement = identity - G
                H = torch.linalg.solve(complement.T, G.T).T
                if not bool(torch.isfinite(H).all()):
                    raise NativeWriterError("NONFINITE_DIAGNOSTIC_H")
                response_gain = C_cpu.double().T @ factor["K"].double()
                reconstructed = factor["R"] @ C_cpu.T
                error = reconstructed - factor["delta"]
                denom = float(torch.linalg.vector_norm(factor["delta"].double()))
                factor.update(C=C_cpu, G=G, H=H, response_gain=response_gain,
                              C_sha256=tensor_sha(C_cpu),
                              reconstruction_max_abs=float(error.abs().max()),
                              reconstruction_relative_frobenius=(float(torch.linalg.vector_norm(error.double())) / denom if denom else None),
                              G_minus_GT_frobenius=float(torch.linalg.vector_norm(G-G.T)),
                              H_minus_HT_frobenius=float(torch.linalg.vector_norm(H-H.T)),
                              condition_I_minus_G=float(torch.linalg.cond(complement)),
                              ideal_orthogonal_P_identity="NOT_ESTABLISHED",
                              diagnostic_factor_status="COMPLETED")
            except Exception as exc:
                factor["diagnostic_factor_status"] = "FAILED"
                save(out / f"L{layer}-diagnostic-failure.json", {
                    "exception_type": type(exc).__name__, "message": str(exc),
                    "physical_write_applied": True, "native_solve_succeeded": True,
                    "layer": layer, "native_delta_artifact": factor["native_artifact"],
                })
                raise NativeWriterError("DIAGNOSTIC_RESPONSE_FACTOR_FAILED") from exc
            finally:
                factor["diagnostic_seconds"] = timed("diagnostic_response_factor", t, layer=layer)
                work.clear()
            factor["artifact"] = tensor_file(out / f"L{layer}-diagnostic-factors.pt", {
                "C": factor["C"], "G": factor["G"], "H": factor["H"],
                "response_gain": factor["response_gain"], "physical_layer": layer,
            })
            compact = {key: value for key, value in factor.items() if not isinstance(value, torch.Tensor)}
            save(out / f"L{layer}-factor-receipt.json", compact)
            notify(f"W{layer}", physical_layer=layer, factor_receipt=compact,
                   post_weight_sha256=factor["post_weight_sha256"])

        def capture_final_residual():
            nonlocal phase, final_l8, bypass
            if final_l8 is not None:
                return
            phase = "final-L8-residual"
            t = time.monotonic()
            bypass = True
            try:
                with rng_preserved(), torch.no_grad():
                    got = original_get(rt.model, rt.tok, 8,
                        context_templates=[r["prompt"] for r in normalized],
                        words=[r["subject"] for r in normalized],
                        module_template=rt.hp.layer_module_tmp,
                        fact_token_strategy=rt.hp.fact_token)[1].T
                _sync(got)
                _finite(got, "final_l8_output")
                output = _cpu(got)
                zs = torch.stack(target_values, dim=1)
                residual = zs - output
                final_l8 = {"output": output, "residual": residual,
                            "residual_sha256": tensor_sha(residual),
                            "mean_column_norm": float(torch.linalg.vector_norm(residual, dim=0).mean())}
                final_l8["artifact"] = tensor_file(out / "final-l8-residual.pt", {
                    "output": output, "residual": residual,
                    "case_ids": [r["case_id"] for r in normalized],
                })
                counters["final_l8_residual_calls"] += 1
            finally:
                bypass = False
                timed("final_l8_residual_observer", t)

        def wrapped_ks(model, tok, request_rows, hp, layer, contexts):
            nonlocal phase, pending_history
            if bypass:
                return original_ks(model, tok, request_rows, hp, layer, contexts)
            finish_z()
            flush_write()
            flush_history()
            if counters["native_solves"] < 5:
                index = counters["write_key_calls"]
                if index >= 5 or layer != LAYERS[index] or index != counters["native_solves"]:
                    raise NativeWriterError("NATIVE_WRITE_KEY_SEQUENCE_INVALID")
                phase = f"native-key-L{layer}"
                t = time.monotonic()
                result = original_ks(model, tok, request_rows, hp, layer, contexts)
                _sync(result)
                _finite(result, "native_key")
                counters["write_key_calls"] += 1
                timed("native_key", t, layer=layer)
                return result
            capture_final_residual()
            index = counters["history_key_calls"]
            if index >= 5 or layer != LAYERS[index]:
                raise NativeWriterError("HISTORY_KEY_SEQUENCE_NOT_ONCE_PER_LAYER")
            phase = f"history-key-L{layer}"
            t = time.monotonic()
            before_sha = tensor_sha(rt.M[index])
            result = original_ks(model, tok, request_rows, hp, layer, contexts)
            _sync(result)
            _finite(result, "history_key")
            final_keys[layer] = _cpu(result.T)
            artifact = tensor_file(out / f"L{layer}-history-keys.pt", {
                "K": final_keys[layer], "case_ids": [r["case_id"] for r in normalized],
                "physical_layer": layer,
            })
            history_receipts.append(dict(layer=layer, pre_M_sha256=before_sha,
                                         key_sha256=tensor_sha(final_keys[layer]),
                                         columns=result.shape[0], artifact=artifact,
                                         append_completed=False))
            counters["history_key_calls"] += 1
            pending_history = layer
            timed("native_history_key_and_instrumentation", t, layer=layer)
            return result

        def wrapped_get(*args, **kwargs):
            if bypass:
                return original_get(*args, **kwargs)
            t = time.monotonic()
            result = original_get(*args, **kwargs)
            _sync(result[1])
            timed("native_l8_residual_forward", t, pending_layer=LAYERS[min(counters["native_solves"], 4)])
            return result

        def wrapped_solve(A, B, *args, **kwargs):
            nonlocal pending, phase, bypass
            frame = inspect.currentframe().f_back
            if frame.f_code is not original.__code__:
                return original_solve(A, B, *args, **kwargs)
            if args or kwargs:
                raise NativeWriterError("UNEXPECTED_NATIVE_SOLVE_ARGUMENTS")
            index = counters["native_solves"]
            if index >= 5 or pending is not None:
                raise NativeWriterError("NATIVE_SOLVE_COUNT_INVALID")
            loc = frame.f_locals
            layer = loc["layer"]
            if layer != LAYERS[index] or loc["i"] != index:
                raise NativeWriterError("NATIVE_SOLVE_LAYER_INDEX_MISMATCH")
            K, R = loc["layer_ks"], loc["resid"]
            _finite(K, "native_solve_K")
            _finite(R, "native_solve_R")
            if K.shape[1] != 100 or R.shape[1] != 100:
                raise NativeWriterError("NATIVE_SOLVE_NOT_COMPLETE_B100")
            phase = f"solve-L{layer}"
            operand_receipt = dict(branch=branch, layer=layer, applied=False,
                                   persistent_history_mutated=False)
            applies = branch == "SHAM" or (branch == "H5" and layer == 5) or (
                branch == "H6" and layer == 6) or (branch in {"H56", "MASS56"} and layer in (5, 6))
            matrix = A
            t_operand = time.monotonic()
            # This P copy also provides the independent response-factor RHS.
            projector = rt.P[index].to(A.device)
            PK = projector @ K
            if applies:
                if stamp is None or bank_ids is None or layer not in stamp:
                    raise NativeWriterError("VALIDATED_H512_TIMESTAMP_BANK_REQUIRED")
                if getattr(rt, "timestamp_keys_validated", True) is not True:
                    raise NativeWriterError("TIMESTAMP_PARITY_NOT_ESTABLISHED")
                stamped = stamp[layer].to(A.device)
                if branch == "SHAM":
                    refreshed = stamped
                else:
                    if not callable(current):
                        raise NativeWriterError("CURRENT_HISTORY_REQUIRES_FRESH_PREFIX_CALLBACK")
                    bypass = True
                    try:
                        with rng_preserved():
                            refreshed = current(layer, rt).to(A.device)
                    finally:
                        bypass = False
                effective = history_operand(rt.M[index].to(A.device), stamped, refreshed,
                    branch=branch, layer=layer, bank_ids=bank_ids,
                    expected_bank_ids=bank_ids, timestamp_validated=True)
                operand_receipt = dict(effective.receipt)
                # Original parenthesization, replacing only the approved M operand.
                matrix = projector @ (K @ K.T + effective.M) + rt.hp.L2 * torch.eye(
                    K.shape[0], dtype=torch.float32, device=A.device)
                del effective, stamped, refreshed
            _sync(matrix)
            timed("history_operand_and_diagnostic_PK", t_operand, layer=layer,
                  branch=branch, applied=applies)
            _finite(matrix, "native_A")
            _finite(B, "native_B")
            pre = _cpu(rt.weights[layer])
            t = time.monotonic()
            result = original_solve(matrix, B)
            _sync(result)
            seconds = timed("native_direct_solve", t, layer=layer)
            _finite(result, "native_solve_result")
            counters["native_solves"] += 1
            # Production Llama shape is non-square; use the original shape helper.
            delta = native.upd_matrix_match_shape(result, rt.weights[layer].shape)
            factor = dict(K=_cpu(K), R=_cpu(R), delta=_cpu(delta),
                          pre_weight=pre, l8_pre=_cpu(loc["cur_zs"]),
                          divisor=5-index, repeat_factor=int(loc["repeat_factor"]),
                          layer=layer, operand=operand_receipt, native_solve_seconds=seconds,
                          pre_weight_sha256=tensor_sha(pre), physical_write_applied=False,
                          raw_A_shape=list(matrix.shape), raw_B_shape=list(B.shape),
                          raw_A_stored=False, residual_source="ORIGINAL_NATIVE_FRAME_RESID")
            factor.update(K_sha256=tensor_sha(factor["K"]), R_sha256=tensor_sha(factor["R"]),
                          delta_sha256=tensor_sha(factor["delta"]))
            factor["native_artifact"] = tensor_file(out / f"L{layer}-native-factors.pt", {
                "K": factor["K"], "R": factor["R"], "delta": factor["delta"],
                "l8_pre": factor["l8_pre"], "divisor": 5-index,
                "physical_layer": layer, "request_ids": [r["case_id"] for r in normalized],
            })
            factors[layer] = factor
            pending = dict(layer=layer, A=matrix, PK=PK)
            return result

        patches_installed = False
        try:
            save(out / "entry-binding.json", {"binding": binding, "binding_sha256": binding_sha,
                                               "branch": branch, "shared_z": shared_z is not None})
            notify("entry", binding_sha256=binding_sha, entry_signature=entry_signature)
            native.compute_z = wrapped_z
            native.compute_ks = wrapped_ks
            native.get_module_input_output_at_words = wrapped_get
            native.torch = _Delegating(originals["torch"], linalg=_Delegating(
                originals["torch"].linalg, solve=wrapped_solve))
            patches_installed = True
            phase = "original-apply"
            returned = original(rt.model, rt.tok, list(requests), rt.hp,
                                cache_template=None, cache_c=rt.M, P=rt.P)
            flush_write()
            flush_history()
            if returned[0] is not rt.model or returned[1] is not rt.M:
                raise NativeWriterError("NATIVE_MODEL_OR_HISTORY_OBJECT_REPLACED")
            expected = {"native_solves": 5, "diagnostic_solves": 5,
                        "write_key_calls": 5, "history_key_calls": 5,
                        "history_appends": 5, "final_l8_residual_calls": 1}
            if any(counters[key] != value for key, value in expected.items()):
                raise NativeWriterError("NATIVE_OPERATION_CARDINALITY_FAILED")
            rt.assert_nonselected()
            notify("history", history=history_receipts, counters=dict(counters))
            terminal = dict(status="COMPLETED", branch=branch, counters=counters,
                stages=stages, history=history_receipts, events=events,
                program_seconds=time.monotonic()-t_program, target_receipts=target_receipts,
                final_l8={k: v for k, v in final_l8.items() if not isinstance(v, torch.Tensor)},
                binding_sha256=binding_sha, persistent_M_refreshed=False,
                new_W_M_RNG_checkpoint_saved=False, caller_restore_required_between_branches=True,
                original_function_called=True, numerical_model_parity="CALLER_E0_REQUIRED")
            terminal_artifact = save(out / "terminal.json", terminal)
            receipt_artifact = save(out / "receipt.json", terminal)
            return dict(z=z_bundle, stages=stages, factors=factors, receipts=terminal,
                        finalkeys=final_keys, final_l8=final_l8,
                        terminal_artifact=terminal_artifact, receipt_artifact=receipt_artifact)
        except Exception as exc:
            failure = dict(status="TECHNICAL_FAILED", branch=branch, phase=phase,
                           exception_type=type(exc).__name__, message=str(exc),
                           cause_type=type(exc.__cause__).__name__ if exc.__cause__ else None,
                           cause_message=str(exc.__cause__) if exc.__cause__ else None,
                           counters=counters, events=events, completed_stages=[x["stage"] for x in stages],
                           physical_write_layers=[l for l, f in factors.items() if f["physical_write_applied"]],
                           history=history_receipts, program_seconds=time.monotonic()-t_program,
                           caller_restore_required=True, automatic_fallback=False)
            try:
                save(out / "failure.json", failure)
            except Exception as secondary:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"failure receipt write failed: {type(secondary).__name__}: {secondary}")
            raise
        finally:
            if patches_installed:
                for name, value in originals.items():
                    setattr(native, name, value)


def write(rt, requests, out, *, branch="NATIVE", shared_z=None, stamp=None,
          current=None, bank_ids=None, stage_callback=None):
    return NativeWriter(rt).write(requests, out, branch=branch, shared_z=shared_z,
                                  stamp=stamp, current=current, bank_ids=bank_ids,
                                  stage_callback=stage_callback)
