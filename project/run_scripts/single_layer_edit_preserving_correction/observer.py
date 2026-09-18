"""Selection-sealed ENFC observer; historical canonical token semantics unchanged.

Only this module accesses official P/N.  It imports no correction controller,
native editor or selection code.  Returned ``raw``/generation rows contain local
prompt/token payload: the caller must retain them locally, never publish them to
Git.  This module never opens files.  Existing canonical observation modules
must first be bound with baseline_mechanism_first.evaluation.bind_evaluation_sources.
No source binding or CPU test is an actual Llama numerical PASS.
"""
from contextlib import contextmanager
import copy
import math
import random
import time

import numpy as np
import torch

from .alltoken import WEIGHT, model_guard
from .common import digest, tensor_sha

LAYOUT = "HISTORICAL_MICROBATCH16_MANUAL_LEFT_PADDING_NO_POSITION_OVERRIDE"
KINDS = ("rewrite_target_new", "rewrite_target_true", "rephrase_target_new",
         "rephrase_target_true", "locality_target_true", "locality_target_new")


class ObserverBoundary(RuntimeError):
    pass


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def require_selection_seal(seal, *, endpoint_sha256, case_ids):
    """Fail before construction/tokenization of any official P/N rows."""
    if not isinstance(seal, dict) or seal.get("status") != "SELECTION_SEALED":
        raise ObserverBoundary("EXPLICIT_SELECTION_SEAL_REQUIRED")
    if not seal.get("episode_id") or not seal.get("endpoint_id"):
        raise ObserverBoundary("SELECTION_EPISODE_ENDPOINT_REQUIRED")
    if not _hex(seal.get("selection_ledger_sha256")):
        raise ObserverBoundary("SELECTION_LEDGER_SHA_REQUIRED")
    if seal.get("endpoint_weight_sha256") != endpoint_sha256:
        raise ObserverBoundary("SELECTION_ENDPOINT_BYTES_MISMATCH")
    if seal.get("request_order_sha256") != digest(case_ids):
        raise ObserverBoundary("SELECTION_REQUEST_ORDER_MISMATCH")
    return copy.deepcopy(seal)


def _row_key(row):
    return (int(row["case_id"]), row["kind"], int(row["prompt_index"]),
            row["prompt"], row["target"], tuple(row["target_token_ids"]))


def _rng_capture(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state(device).cpu().clone() if device.type == "cuda" else None)


def _rng_equal(a, b):
    numpy_equal = (a["numpy"][0] == b["numpy"][0] and np.array_equal(a["numpy"][1], b["numpy"][1])
                   and a["numpy"][2:] == b["numpy"][2:])
    cuda_equal = ((a["cuda"] is None and b["cuda"] is None) or
                  (a["cuda"] is not None and b["cuda"] is not None and torch.equal(a["cuda"], b["cuda"])))
    return a["python"] == b["python"] and numpy_equal and torch.equal(a["torch"], b["torch"]) and cuda_equal


def _rng_restore(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def strict_summary(metrics, case_ids):
    """NLL-pair metrics, TF strict and two-P/joint remain distinct."""
    by_kind = {kind: {} for kind in ("RS", "PS", "NS")}
    for kind in by_kind:
        for row in metrics[kind]["rows"]:
            by_kind[kind].setdefault(row["case_id"], []).append(row)
    rows = []
    for case in case_ids:
        r, p = by_kind["RS"][case], by_kind["PS"][case]
        if len(r) != 1 or len(p) != 2:
            raise ObserverBoundary("STRICT_SUMMARY_REQUIRES_ONE_R_TWO_P")
        rows.append(dict(case_id=case, rewrite_strict=bool(r[0]["new_strict"]),
                         two_P_strict=all(x["new_strict"] for x in p),
                         R_two_P_strict=bool(r[0]["new_strict"] and all(x["new_strict"] for x in p)),
                         R_two_P_NLL_joint=bool(r[0]["success"] and all(x["success"] for x in p))))
    fields = ("rewrite_strict", "two_P_strict", "R_two_P_strict", "R_two_P_NLL_joint")
    return dict(denominator=len(case_ids), rows=rows,
                **{key: sum(row[key] for row in rows) for key in fields})


def w0_correct_retention(w0_metrics, selected_metrics):
    """Canonical NS, conditioned on the SAME prompt/target identity at W0."""
    before, after = w0_metrics["NS"]["rows"], selected_metrics["NS"]["rows"]
    if len(before) != len(after):
        raise ObserverBoundary("W0_NS_CARDINALITY_MISMATCH")
    rows = []
    for old, new in zip(before, after, strict=True):
        if (old["identity"], old["case_id"], old["prompt_index"]) != (
                new["identity"], new["case_id"], new["prompt_index"]):
            raise ObserverBoundary("W0_NS_EXACT_IDENTITY_MISMATCH")
        rows.append(dict(identity=old["identity"], case_id=old["case_id"],
                         prompt_index=old["prompt_index"], W0_success=bool(old["success"]),
                         selected_success=bool(new["success"]),
                         retained=bool(old["success"] and new["success"]),
                         lost=bool(old["success"] and not new["success"]),
                         gained=bool(not old["success"] and new["success"])))
    denominator = sum(row["W0_success"] for row in rows)
    retained = sum(row["retained"] for row in rows)
    return dict(denominator=denominator, numerator=retained,
                rate=retained/denominator if denominator else None,
                lost=sum(row["lost"] for row in rows), gained=sum(row["gained"] for row in rows),
                all_requested_N=len(rows), rows=rows)


class CanonicalObserver:
    """Post-seal canonical evaluation at actual selected physical L4 bytes.

    ``runtime_identity`` is the parent's locked model/tokenizer/library/kernel
    compatibility digest, not an inference from endpoint filename.  The caller
    binds the historical evaluator first.  ``reuse`` may contain partial raw
    kinds; only exact complete historical MB16 groups are reused, so missing
    work retains the original padding/reduction layout rather than repartitioning.
    """
    def __init__(self, model, tok, *, runtime_identity):
        from project.run_scripts.baseline_mechanism_first.evaluation import _binding
        if not _hex(runtime_identity):
            raise ObserverBoundary("LOCKED_RUNTIME_IDENTITY_SHA_REQUIRED")
        self.bindings = _binding()
        self.model, self.tok = model, tok
        self.runtime_identity = runtime_identity
        params = dict(model.named_parameters())
        if WEIGHT not in params:
            raise ObserverBoundary("PHYSICAL_L4_PARAMETER_REQUIRED")
        self.parameter = params[WEIGHT]
        self.device = self.parameter.device
        if model.training or any(p.requires_grad or p.dtype != torch.float32 for p in model.parameters()):
            raise ObserverBoundary("FROZEN_FP32_EVAL_MODEL_REQUIRED")
        if tok.padding_side != "right":
            raise ObserverBoundary("SHARED_CANONICAL_TOKENIZER_MUST_REMAIN_RIGHT")
        self.base_guard = model_guard(model)
        self.source_sha256s = sorted(v["sha256"] for v in self.bindings["receipt"].values())
        self.work = dict(observer_calls=0, model_forward_calls=0, model_input_tokens=0,
                         attention_mask_valid_entries=0, canonical_pair_rows_computed=0,
                         canonical_pair_rows_reused=0, canonical_microbatches_computed=0,
                         greedy_requests_computed=0, greedy_requests_reused=0,
                         generated_tokens=0, backward_calls=0, native_fit_calls=0,
                         history_appends=0, physical_installs=0, exact_restores=0,
                         observer_seconds=0., copy_restore_seconds=0.)
        self.events = []

    @contextmanager
    def _physical(self, weight):
        before = model_guard(self.model)
        rng = _rng_capture(self.device)
        saved = self.parameter.detach().cpu().clone()
        if weight.dtype != torch.float32 or tuple(weight.shape) != tuple(self.parameter.shape):
            raise ObserverBoundary("ENDPOINT_FP32_FULL_L4_SHAPE")
        if not torch.isfinite(weight).all():
            raise FloatingPointError("NONFINITE_OBSERVER_ENDPOINT")
        changed_rng = False
        try:
            started = time.perf_counter()
            with torch.no_grad():
                self.parameter.copy_(weight.to(self.device))
            if not torch.equal(self.parameter.detach().cpu(), weight.detach().cpu()):
                raise ObserverBoundary("ENDPOINT_PHYSICAL_INSTALL_NOT_EXACT")
            self.work["copy_restore_seconds"] += time.perf_counter()-started
            self.work["physical_installs"] += 1
            with torch.no_grad():
                yield
        finally:
            changed_rng = not _rng_equal(rng, _rng_capture(self.device))
            _rng_restore(rng, self.device)
            started = time.perf_counter()
            with torch.no_grad():
                self.parameter.copy_(saved.to(self.device))
            if not torch.equal(self.parameter.detach().cpu(), saved):
                raise ObserverBoundary("OBSERVER_EXACT_ROLLBACK_FAILED")
            self.work["copy_restore_seconds"] += time.perf_counter()-started
            self.work["exact_restores"] += 1
            after = model_guard(self.model)
            nonselected = lambda state: (tuple(row for row in state[0] if row[1] != WEIGHT), state[1])
            if nonselected(before) != nonselected(after):
                raise ObserverBoundary("OBSERVER_NONSELECTED_PARAMETER_BUFFER_MODE_HOOK_MUTATED")
            if dict(self.model.named_parameters()).get(WEIGHT) is not self.parameter:
                raise ObserverBoundary("OBSERVER_SELECTED_PARAMETER_OBJECT_REPLACED")
            if any(p.grad is not None or p.requires_grad for p in self.model.parameters()):
                raise ObserverBoundary("OBSERVER_MODEL_GRAD_MUTATION")
            if self.tok.padding_side != "right":
                raise ObserverBoundary("OBSERVER_TOKENIZER_PADDING_MUTATION")
            if changed_rng:
                raise ObserverBoundary("OBSERVER_RNG_MUTATED_AND_RESTORED")

    def _count_forward(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        self.work["model_forward_calls"] += 1
        if ids is not None:
            self.work["model_input_tokens"] += ids.numel()
        mask = kwargs.get("attention_mask")
        if mask is not None:
            self.work["attention_mask_valid_entries"] += int(mask.sum())

    def _pairs_and_identity(self, records):
        pairs = self.bindings["evaluator"].counterfact_pairs(records)
        pairs["locality_target_new"] = self.bindings["locality"].counterfact_locality_target_new_pairs(records)
        expected = {}
        token_inputs = []
        contracts = self.bindings["contracts"]
        for kind in KINDS:
            expected[kind] = []
            for pair in pairs[kind]:
                prompt_ids = contracts.prompt_token_ids(self.tok, pair.prompt)
                target_ids = contracts.target_token_ids(self.tok, pair.target)
                key = (pair.case_id, pair.kind, pair.prompt_index, pair.prompt, pair.target, tuple(target_ids))
                expected[kind].append(key)
                token_inputs.append([list(key[:-1]), prompt_ids, target_ids])
        return pairs, expected, digest(token_inputs)

    @staticmethod
    def _validate_row(row, expected):
        if _row_key(row) != expected:
            raise ObserverBoundary("RAW_PAIR_EXACT_IDENTITY_MISMATCH")
        if not math.isfinite(float(row["nll"])):
            raise FloatingPointError("NONFINITE_OBSERVER_NLL")
        correct = row["token_correct"]
        if len(correct) != len(row["target_token_ids"]) or any(type(v) is not bool for v in correct):
            raise ObserverBoundary("RAW_TOKEN_CORRECT_CARDINALITY")
        if bool(row["all_tokens_correct"]) != all(correct):
            raise ObserverBoundary("RAW_TF_STRICT_INCONSISTENT")

    def _reuse_index(self, reuse, compatibility, expected):
        if reuse is None:
            return {}, {}
        if reuse.get("compatibility") != compatibility:
            raise ObserverBoundary("REUSE_RUNTIME_ENDPOINT_TOKEN_OR_SOURCE_MISMATCH")
        reused = {}
        allowed = {key for values in expected.values() for key in values}
        for rows in reuse.get("raw", {}).values():
            for row in rows:
                key = _row_key(row)
                if key not in allowed:
                    raise ObserverBoundary("REUSE_UNEXPECTED_ROW_IDENTITY")
                if key in reused:
                    raise ObserverBoundary("REUSE_DUPLICATE_ROW_IDENTITY")
                self._validate_row(row, key)
                reused[key] = copy.deepcopy(row)
        greedy = {row["case_id"]: copy.deepcopy(row) for row in reuse.get("generation", [])}
        if len(greedy) != len(reuse.get("generation", [])):
            raise ObserverBoundary("REUSE_DUPLICATE_GREEDY_IDENTITY")
        return reused, greedy

    def _greedy(self, records, reused):
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.model.config.eos_token_id
        eos_ids = [int(eos)] if isinstance(eos, int) else list(eos or [])
        if not eos_ids:
            raise ObserverBoundary("ORIGINAL_MODEL_EOS_IDS_REQUIRED")
        original_config = copy.deepcopy(self.model.generation_config.to_dict())
        rows = []
        contracts = self.bindings["contracts"]
        for record in records:
            rewrite = record["requested_rewrite"]
            prompt = rewrite["prompt"].format(rewrite["subject"])
            prompt_ids = contracts.prompt_token_ids(self.tok, prompt)
            target_ids = contracts.target_token_ids(self.tok, rewrite["target_new"]["str"])
            identity = digest([record["case_id"], prompt_ids, target_ids, eos_ids, 32, False, 1])
            if record["case_id"] in reused:
                row = reused[record["case_id"]]
                if row.get("identity") != identity:
                    raise ObserverBoundary("REUSE_GREEDY_EXACT_IDENTITY_MISMATCH")
                rows.append(row)
                self.work["greedy_requests_reused"] += 1
                continue
            inputs = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            generated = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                       do_sample=False, num_beams=1, max_new_tokens=32,
                       eos_token_id=eos_ids, pad_token_id=self.tok.pad_token_id,
                       use_cache=True, return_dict_in_generate=False)
            if generated.ndim != 2 or generated.shape[0] != 1 or not torch.equal(generated[0, :len(prompt_ids)], inputs[0]):
                raise ObserverBoundary("GENERATION_PROMPT_PREFIX_CHANGED")
            new = generated[0, len(prompt_ids):].detach().cpu().tolist()
            if len(new) > 32:
                raise ObserverBoundary("GENERATION_MAX32_VIOLATED")
            row = dict(case_id=record["case_id"], identity=identity, prompt_token_ids=prompt_ids,
                       target_token_ids=target_ids, generated_token_ids=new,
                       generated_length=len(new), target_length=len(target_ids),
                       original_eos_ids=eos_ids, stopped_on_original_eos=bool(new and new[-1] in eos_ids),
                       reached_max_new_tokens=len(new) == 32,
                       target_prefix_match=len(new) >= len(target_ids) and new[:len(target_ids)] == target_ids,
                       target_over_32_censored=len(target_ids) > 32,
                       partial_observed_prefix_match=new[:min(len(new), len(target_ids))] == target_ids[:min(len(new), len(target_ids))],
                       denominator_retained=True, do_sample=False, max_new_tokens=32)
            rows.append(row)
            self.work["greedy_requests_computed"] += 1
            self.work["generated_tokens"] += len(new)
        if self.model.generation_config.to_dict() != original_config:
            raise ObserverBoundary("ORIGINAL_GENERATION_CONFIG_MUTATED")
        return rows

    def observe(self, records, endpoint_weight, *, selection_seal, w0_result=None,
                reuse=None, greedy=True, save_callback=None):
        """Return local raw payload + canonical aggregates, only after seal.

        The optional save_callback(result) runs after model/RNG restoration;
        its path/atomic/create-once policy is the caller's responsibility.
        It must not pass observer results back to an online selector.
        """
        records = list(records)
        case_ids = [int(record["case_id"]) for record in records]
        if not case_ids or len(case_ids) != len(set(case_ids)):
            raise ObserverBoundary("NONEMPTY_UNIQUE_REQUEST_INVENTORY_REQUIRED")
        seal = require_selection_seal(selection_seal, endpoint_sha256=tensor_sha(endpoint_weight), case_ids=case_ids)
        current_guard = model_guard(self.model)
        nonselected = lambda state: (tuple(row for row in state[0] if row[1] != WEIGHT), state[1])
        if nonselected(current_guard) != nonselected(self.base_guard):
            raise ObserverBoundary("OBSERVER_BASE_NONSELECTED_STATE_CHANGED")
        if dict(self.model.named_parameters()).get(WEIGHT) is not self.parameter:
            raise ObserverBoundary("OBSERVER_SELECTED_PARAMETER_OBJECT_REPLACED")
        # Do not even inspect official prompt arrays before the seal above.
        if any(len(r["paraphrase_prompts"]) != 2 or len(r["neighborhood_prompts"]) != 10 for r in records):
            raise ObserverBoundary("CANONICAL_R1_P2_N10_CARDINALITY")
        pairs, expected, token_identity = self._pairs_and_identity(records)
        compatibility = dict(endpoint_weight_sha256=seal["endpoint_weight_sha256"],
                             request_order_sha256=digest(case_ids), input_token_identity=token_identity,
                             runtime_identity=self.runtime_identity,
                             source_sha256s=self.source_sha256s, evaluator_layout=LAYOUT)
        reused, reused_greedy = self._reuse_index(reuse, compatibility, expected)
        before_work = dict(self.work)
        raw = {}
        started = time.perf_counter()
        # The counter hook is installed outside the guarded region and removed
        # after rollback, so the observer's own bookkeeping is not a mutation.
        hook = self.model.register_forward_pre_hook(self._count_forward, with_kwargs=True)
        try:
            with self._physical(endpoint_weight):
                for kind in KINDS:
                    rows = []
                    for offset in range(0, len(pairs[kind]), 16):
                        group = pairs[kind][offset:offset+16]
                        identities = expected[kind][offset:offset+16]
                        if all(key in reused for key in identities):
                            values = [reused[key] for key in identities]
                            self.work["canonical_pair_rows_reused"] += len(values)
                        else:
                            values = self.bindings["evaluator"].evaluate_pairs(
                                self.model, self.tok, group, device=self.device, microbatch_size=16)
                            self.work["canonical_pair_rows_computed"] += len(values)
                            self.work["canonical_microbatches_computed"] += 1
                        if len(values) != len(identities):
                            raise ObserverBoundary("CANONICAL_PAIR_ROW_CARDINALITY")
                        for row, identity in zip(values, identities, strict=True):
                            self._validate_row(row, identity)
                        rows.extend(values)
                    raw[kind] = rows
                generation = self._greedy(records, reused_greedy) if greedy else []
        finally:
            hook.remove()
        self.work["observer_calls"] += 1
        self.work["observer_seconds"] += time.perf_counter()-started
        metrics = self.bindings["historical"].reduce(raw)
        for tag, multiple in (("RS", 1), ("PS", 2), ("NS", 10)):
            if metrics[tag]["denominator"] != len(records)*multiple:
                raise ObserverBoundary("CANONICAL_REDUCER_CARDINALITY")
        result = dict(selection_seal=seal, selection_seal_verified_before_P_N_access=True,
                      compatibility=compatibility, requests=len(records), request_order=digest(case_ids),
                      raw=raw, metrics=metrics, strict=strict_summary(metrics, case_ids),
                      generation=generation, generation_status="RECORDED" if greedy else "NOT_REQUESTED",
                      generation_denominator=len(case_ids) if greedy else 0,
                      generation_censored=sum(row["target_over_32_censored"] for row in generation),
                      work={key:self.work[key]-before_work[key] for key in self.work},
                      source_binding=self.bindings["receipt"], evaluator_controller_influence=0,
                      physical_endpoint_bytes_installed=True, entry_selected_weight_restored_exact=True,
                      nonselected_guard="PARAMETER_BUFFER_POINTER_VERSION_HOOK_MODE; NOT_FULL_BYTE_REHASH",
                      RNG_unchanged_and_restored=True, actual_Llama_T_PASS_not_claimed=True,
                      raw_payload_local_only=True)
        if w0_result is None:
            result["W0_correct_NS"] = dict(status="NOT_AVAILABLE")
        else:
            c0 = w0_result["compatibility"]
            compare_keys = ("request_order_sha256", "input_token_identity", "runtime_identity", "source_sha256s", "evaluator_layout")
            if any(c0.get(key) != compatibility[key] for key in compare_keys):
                raise ObserverBoundary("W0_OBSERVER_COMPATIBILITY_MISMATCH")
            result["W0_correct_NS"] = w0_correct_retention(w0_result["metrics"], metrics)
        self.events.append({k:result[k] for k in ("selection_seal", "work", "requests", "compatibility")})
        if save_callback is not None:
            save_callback(result)
        return result
