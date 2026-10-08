"""Existing first300 B3 CF raw versus independent original-native observation.

Imports only the published ``official`` reference at actual invocation. Draft
worktree sources are never copied/imported/executed. Raw tokens/case identities
stay in the returned local-only report; ``validate_report`` returns compact CPU
evidence, not full-2K parity or a scientific performance promotion.
"""
import importlib
import json
import math
from pathlib import Path
import re
import time
from functools import wraps

from official.evaluation.reduce import counterfact
from official.experiments import checkpoint
from official.experiments.prepare import digest
from .common import rng_digest


SCHEMA = "official-server1-native-CF-parity-report-v1"
REFERENCE_SOURCE_SHA256 = "25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145"
REFERENCE_UPSTREAM_COMMIT = "b84624f44dfe8fc6cd9e41df916c44124a0c46dc"
REFERENCE_BYTES = 7941
SCOPE = "ACTUAL_MATCHED_SUBSET"
REQUESTS = 300
OCCURRENCES = list(range(1, REQUESTS + 1))
METHODS = ("FT", "MEMIT", "MEMIT_FE", "ALPHAEDIT", "SPHERE")


def validate_state(value, *, method=None):
    if value.get("method") not in ("ALPHAEDIT", "SPHERE"):
        from official.evaluation.w0_reference import _native_state
        return _native_state(value, method=method)
    require(method is None or value["method"] == method, "PROJECTED_STATE_METHOD")
    require(set(value) == {"method", "successful_calls", "cache_c", "selected_weights", "contexts_sha256", "identity_sha256"}
            and value["successful_calls"] == 3
            and value["identity_sha256"] == digest({k:v for k,v in value.items() if k != "identity_sha256"}),
            "PROJECTED_STATE_DIGEST_CURSOR")
    require(set(value["cache_c"]) == {str(i) for i in range(4,9)}
            and set(value["selected_weights"]) == {f"model.layers.{i}.mlp.down_proj.weight" for i in range(4,9)},
            "PROJECTED_STATE_LAYER_MAPPING")
    for name, shape in (("cache_c", [14336,14336]), ("selected_weights", [4096,14336])):
        for row in value[name].values():
            require(set(row) == {"sha256","shape","dtype"} and row["shape"] == shape
                    and row["dtype"] == "torch.float32" and re.fullmatch(r"[a-f0-9]{64}", row["sha256"]),
                    "PROJECTED_STATE_TENSOR_IDENTITY")
    require(re.fullmatch(r"[a-f0-9]{64}", value["contexts_sha256"]), "PROJECTED_STATE_CONTEXT")
PLAN = dict(schema="official-server1-native-CF-parity-plan-v1", completed_batch=3,
    stage="continuous_B3", first_occurrences=OCCURRENCES, requests=REQUESTS,
    evidence_scope=SCOPE, native_reference_upstream_sha256=REFERENCE_SOURCE_SHA256,
    native_reference_upstream_commit=REFERENCE_UPSTREAM_COMMIT,
    nll_abs_nats=1e-4, nll_relative=1e-5, aggregate_abs=1e-10, strict_bool="EXACT",
    canonical_batch_size=16, canonical_observation="EXISTING_B3_FIRST300",
    reference_observation="INDEPENDENT_ORIGINAL_FORWARD_SAME300",
    additional_canonical_forward_calls=0,
    additional_fit_calls=0, generation_calls=0, quality_promotion=False,
    tolerance_override=False, full_2k_parity="NOT_OBSERVED")
EXTERNAL_FIELDS = {"model", "model_revision", "tokenizer_sha256", "assets_sha256",
    "stream_sha256", "code_commit", "official_tree", "runtime", "precision", "raw_local_only"}
GROUPS = ("rewrite", "paraphrase", "neighborhood")


class NativeParityError(ValueError):
    pass


def _typed(function):
    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except NativeParityError:
            raise
        except (KeyError,TypeError,ValueError,AttributeError) as error:
            raise NativeParityError(function.__name__ + ".INVALID_PROOF:" + str(error)) from error
    return call


def require(value, code):
    if not value:
        raise NativeParityError(code)


def clone(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (ValueError, TypeError) as error:
        raise NativeParityError("NATIVE_PARITY_JSON_FINITE_IDENTITY") from error


def validate_plan(plan):
    require(type(plan) is dict and digest(plan) == digest(PLAN), "NATIVE_PARITY_PLAN_CHANGED_OR_NOT_FROZEN")
    return digest(plan)


def _published_reference():
    """Only the same official tree used by this runner may supply the oracle."""
    try:
        module = importlib.import_module("official.evaluation.cf_native_reference")
    except ImportError as error:
        raise NativeParityError("PUBLISHED_NATIVE_CF_REFERENCE_UNAVAILABLE") from error
    expected = Path(__file__).resolve().parents[2] / "evaluation" / "cf_native_reference.py"
    require(Path(module.__file__).resolve() == expected and expected.is_file()
            and not expected.is_symlink(), "NATIVE_PARITY_REFERENCE_OUTSIDE_FROZEN_OFFICIAL")
    require(module.SOURCE_SHA256 == REFERENCE_SOURCE_SHA256 and module.SOURCE_BYTES == REFERENCE_BYTES
            and module.UPSTREAM_COMMIT == REFERENCE_UPSTREAM_COMMIT and module.MATCHED_SCOPE == SCOPE,
            "NATIVE_PARITY_REFERENCE_SOURCE_OR_SCOPE_CHANGED")
    require(module.NLL_ABS_TOL_NATS == PLAN["nll_abs_nats"]
            and module.NLL_REL_TOL == PLAN["nll_relative"]
            and module.AGGREGATE_ABS_TOL == PLAN["aggregate_abs"]
            and callable(module.compare_native_counterfact), "NATIVE_PARITY_REFERENCE_API_OR_TOLERANCE_CHANGED")
    return module


def _caller_profile(model, tokenizer, engine):
    import torch
    require(engine.method in METHODS and getattr(engine, "source_verified", False) is True,
            "NATIVE_PARITY_FROZEN_SOURCE_OR_METHOD_REQUIRED")
    floating = [value for value in list(model.parameters()) + list(model.buffers()) if value.is_floating_point()]
    require(bool(floating) and all(value.dtype == torch.float32 and value.device.type == "cuda" for value in floating),
            "NATIVE_PARITY_ACTUAL_FP32_CUDA_REQUIRED")
    require(all(module.training is False for module in model.modules()), "NATIVE_PARITY_CALLER_EVAL_REQUIRED")
    require(model.config.use_cache is False and model.config._attn_implementation == "eager"
            and torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False
            and torch.is_autocast_enabled() is False, "NATIVE_PARITY_EAGER_NO_CACHE_TF32_AUTOCAST_OFF")
    require(tokenizer.padding_side == "right" and type(tokenizer.pad_token_id) is int
            and tokenizer.pad_token_id >= 0, "NATIVE_PARITY_NATIVE_RIGHT_PADDING")


def _external_identity(external):
    require(type(external) is dict and set(external) == EXTERNAL_FIELDS
            and external["model"] == "llama3" and external["raw_local_only"] is True,
            "NATIVE_PARITY_EXTERNAL_IDENTITY_FIELDS")
    for key in ("model_revision", "tokenizer_sha256", "assets_sha256", "stream_sha256", "code_commit", "official_tree"):
        require(type(external[key]) is str and re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", external[key]),
                "NATIVE_PARITY_EXTERNAL_SOURCE_SHA")
    return clone(external)


def _state(engine):
    value = clone(engine.state_identity())
    require(value.get("method") == engine.method and type(value.get("successful_calls")) is int
            and value["successful_calls"] == 3,
            "NATIVE_PARITY_ONLY_COMPLETED_B3")
    validate_state(value, method=engine.method)
    return value


def qualify(model, tokenizer, records, external, engine, plan, canonical_result):
    """Reuse unchanged existing B3 first300 raw; independently observe native300.

    Caller freezes literal PLAN in its scientific config before execution and
    invokes this only in continuous B3. ``engine.source_verified`` is supplied
    by the runner only after its reviewed main/archive membership verification.
    Caller supplies the exact in-memory result of its existing B3 first300
    factual observation, immediately at that same state. Its external identity,
    raw, and identity hash are never relabelled. Separate reference-binding
    locks certify this current B3 state without adding fields to canonical raw.
    There is no canonical evaluator/model call in this seam.
    """
    plan_sha = validate_plan(plan)
    _caller_profile(model, tokenizer, engine)
    module = _published_reference()
    external = _external_identity(external)
    records = list(records)
    selected = records[:REQUESTS]
    require(len(selected) == REQUESTS and [row.get("occurrence_index") for row in selected] == OCCURRENCES
            and all(type(row.get("case_id")) is int for row in selected), "NATIVE_PARITY_LOCKED_FIRST300_REQUIRED")
    model_identity = clone(engine.asset_manifest["model"]["identity"])
    tokenizer_sha = engine.asset_manifest["model"]["tokenizer_sha256"]
    require(model_identity["revision"] == external["model_revision"] and tokenizer_sha == external["tokenizer_sha256"],
            "NATIVE_PARITY_MODEL_TOKENIZER_BINDING")
    state_before, records_sha = _state(engine), digest(selected)
    binding = dict(model_identity=model_identity,
                   tokenizer_identity=dict(sha256=tokenizer_sha), state_identity=state_before)
    canonical = canonical_result
    _canonical(canonical, external, selected)
    canonical_sha = digest(canonical)
    locked_cohort = [dict(case_id=row["case_id"], occurrence_index=row["occurrence_index"]) for row in selected]
    rng_before, started = rng_digest(checkpoint.rng_snapshot()), time.monotonic()
    caught, value = None, None
    try:
        comparison = module.compare_native_counterfact(model, tokenizer, selected, canonical,
            identity=external, evidence_scope=SCOPE, locked_cohort=locked_cohort,
            state_callback=engine.state_identity, **binding)
        value = dict(schema=SCHEMA, plan_sha256=plan_sha, external_identity=external,
            reference_binding=binding, canonical=canonical, comparison=comparison,
            canonical_payload_sha256=canonical_sha,
            canonical_payload_unchanged=digest(canonical) == canonical_sha,
            canonical_observation=PLAN["canonical_observation"], additional_canonical_forward_calls=0,
            records_sha256=records_sha, state_before=state_before, state_after=_state(engine),
            RNG_restored=rng_digest(checkpoint.rng_snapshot()) == rng_before,
            input_records_unchanged=digest(selected) == records_sha, raw_local_only=True,
            performance_gate=False, additional_fit_calls=0, generation_calls=0,
            seconds=time.monotonic() - started)
        validate_report(value, external, plan)
        return value
    except BaseException as error:
        caught = error
        # Preserve the expensive independent observation on a failed gate. The
        # caller may write this exact local-only report into its typed failure
        # receipt; it must not upload it, change thresholds, or rerun to PASS.
        if value is not None:
            error.local_report = value
        raise
    finally:
        try:
            unchanged = (_state(engine) == state_before and rng_digest(checkpoint.rng_snapshot()) == rng_before
                         and digest(selected) == records_sha and digest(canonical) == canonical_sha)
        except BaseException:
            unchanged = False
        if not unchanged:
            if caught is None:
                raise NativeParityError("NATIVE_PARITY_FINAL_STATE_RNG_INPUT_MUTATED")
            if hasattr(caught, "add_note"):
                caught.add_note("NATIVE_PARITY_FINAL_STATE_RNG_INPUT_MUTATED")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _work(value, candidate_count, *, native):
    require(type(value) is dict and value.get("candidate_sequences") == candidate_count
            and type(value.get("forward_calls")) is int
            and value["forward_calls"] == (REQUESTS if native else math.ceil(candidate_count / 16)),
            "NATIVE_PARITY_INDEPENDENT_FORWARD_WORK_COUNT")
    for key in ("physical_input_tokens", "padded_input_tokens", "target_tokens"):
        require(type(value.get(key)) is int and value[key] > 0, "NATIVE_PARITY_PHYSICAL_TOKEN_WORK")
    require(value["padded_input_tokens"] >= value["physical_input_tokens"]
            and _finite(value.get("seconds")) and value["seconds"] > 0, "NATIVE_PARITY_WORK_FINITE")


@_typed
def _canonical(canonical, external, selected=None):
    """Check original existing raw before allowing the independent forwards."""
    require(type(canonical) is dict and type(canonical.get("identity")) is dict,
            "NATIVE_PARITY_EXISTING_CANONICAL_REQUIRED")
    from official.evaluation.w0_reference import _no_test_evidence
    _no_test_evidence(canonical)
    ci = canonical["identity"]
    require(ci.get("external_identity") == external and ci.get("ordered_occurrences") == OCCURRENCES
            and ci.get("schema") == "official-factual-causal-v1" and ci.get("dataset") == "cf"
            and ci.get("tokenization") == "NATIVE_CF_VERIFIED_BOUNDARY_ZSRE_EXACT_TOKEN_PREFIX_NO_TARGET_BOS"
            and ci.get("padding") == "RIGHT_EXPLICIT_ATTENTION_MASK" and ci.get("use_cache") is False
            and canonical.get("identity_sha256") == digest(ci)
            and all(canonical.get(key) is True for key in ("raw_local_only", "model_no_mutation", "RNG_restored")),
            "NATIVE_PARITY_CANONICAL_IDENTITY_OR_GUARDS")
    cases = canonical.get("cases")
    require(type(cases) is list and len(cases) == REQUESTS
            and all(type(row.get("case_id")) is int for row in cases)
            and [row.get("occurrence_index") for row in cases] == OCCURRENCES,
            "NATIVE_PARITY_EXISTING_FIRST300_RAW_REQUIRED")
    if selected is not None:
        require([(row["case_id"], row["occurrence_index"]) for row in cases]
                == [(row["case_id"], row["occurrence_index"]) for row in selected],
                "NATIVE_PARITY_EXISTING_CANONICAL_COHORT_MISMATCH")
        from official.evaluation.w0_reference import _dataset_queries, _signatures
        _dataset_queries(_signatures(cases,"cf"), selected, "cf")
    # Factual raw must retain the measured work rather than invent new calls.
    candidates = 0
    for row in cases:
        for kind in GROUPS:
            probabilities = row.get(kind + "_prompts_probs")
            correct = row.get(kind + "_prompts_correct")
            require(type(probabilities) is list and len(probabilities) > 0
                    and type(correct) is list and len(correct) == len(probabilities)
                    and all(type(value) is bool for value in correct), "NATIVE_PARITY_EXISTING_CANONICAL_ROWS")
            for probability in probabilities:
                require(type(probability) is dict and all(_finite(probability.get(target))
                        and probability[target] >= 0 for target in ("target_new", "target_true")),
                        "NATIVE_PARITY_EXISTING_CANONICAL_NLL")
            candidates += len(probabilities) * 2
    _work(canonical.get("work"), candidates, native=False)
    from official.evaluation.factual import validate_retained_observation
    from official.evaluation.w0_reference import _signatures
    validate_retained_observation(canonical,"cf",input_signature=_signatures(cases,"cf"),batch_size=16)
    return ci


@_typed
def validate_report(value, expected_external, plan):
    """Independent CPU acceptance of stored source/scope/raw/guards/work only."""
    plan_sha = validate_plan(plan)
    from official.evaluation.w0_reference import _no_test_evidence
    _no_test_evidence(value)
    expected_external = _external_identity(expected_external)
    require(type(value) is dict and value.get("schema") == SCHEMA and value.get("plan_sha256") == plan_sha
            and value.get("external_identity") == expected_external, "NATIVE_PARITY_REPORT_SOURCE_PLAN_IDENTITY")
    require(value.get("raw_local_only") is True and value.get("RNG_restored") is True
            and value.get("input_records_unchanged") is True and value.get("performance_gate") is False
            and type(value.get("additional_fit_calls")) is int and value["additional_fit_calls"] == 0
            and type(value.get("generation_calls")) is int and value["generation_calls"] == 0
            and type(value.get("additional_canonical_forward_calls")) is int
            and value["additional_canonical_forward_calls"] == 0
            and value.get("canonical_observation") == PLAN["canonical_observation"]
            and value.get("canonical_payload_unchanged") is True,
            "NATIVE_PARITY_REPORT_OBSERVATION_GUARDS")
    binding, canonical, comparison = (value[name] for name in ("reference_binding", "canonical", "comparison"))
    validate_state(value.get("state_before"))
    require(type(binding) is dict and set(binding) == {"model_identity", "tokenizer_identity", "state_identity"}
            and binding.get("state_identity") == value.get("state_before") == value.get("state_after")
            and binding["state_identity"].get("successful_calls") == 3
            and binding["state_identity"].get("method") in METHODS
            and binding.get("model_identity", {}).get("revision") == expected_external["model_revision"]
            and binding.get("tokenizer_identity") == dict(sha256=expected_external["tokenizer_sha256"]),
            "NATIVE_PARITY_REPORT_MODEL_TOKEN_STATE")
    require(comparison.get("schema") == "official-cf-original-native-reference-v1"
            and comparison.get("status") == "PASS" and comparison.get("evidence_scope") == SCOPE
            and comparison.get("evidence") == {"TEST_ONLY_CPU_FIXTURE": "NOT_OBSERVED",
                "ACTUAL_GPU_SMOKE": "NOT_OBSERVED", SCOPE: "PASS", "ACTUAL_FULL_2K": "NOT_OBSERVED"}
            and comparison.get("mismatches") == [] and comparison.get("display_mismatches") == []
            and comparison.get("raw_local_only") is True and comparison.get("checkpoint_saved") is False
            and comparison.get("scientific_performance_promotion") is False, "NATIVE_PARITY_ACTUAL_MATCHED_PASS_REQUIRED")
    require(comparison.get("tolerances") == dict(nll_abs_nats=1e-4, nll_relative_to_native=1e-5,
            strict_booleans="EXACT", aggregate_abs=1e-10), "NATIVE_PARITY_RECEIPT_TOLERANCES")
    ci = _canonical(canonical, expected_external)
    require(comparison.get("canonical_identity_sha256") == canonical["identity_sha256"]
            and value.get("canonical_payload_sha256") == digest(canonical),
            "NATIVE_PARITY_EXISTING_CANONICAL_PAYLOAD_CHANGED")
    native, ni = comparison["native"], comparison["native"]["identity"]
    require(native.get("schema") == "official-cf-original-native-reference-v1"
            and native.get("status") == "OBSERVED_UNCOMPARED"
            and native.get("evidence") == {"TEST_ONLY_CPU_FIXTURE":"NOT_OBSERVED","ACTUAL_GPU_SMOKE":"NOT_OBSERVED",
                                          SCOPE:"NOT_OBSERVED","ACTUAL_FULL_2K":"NOT_OBSERVED"}
            and ni.get("schema") == "official-cf-original-native-reference-v1"
            and ni.get("external_identity") == expected_external and ni.get("reference_binding") == binding
            and ni.get("source_sha256") == REFERENCE_SOURCE_SHA256
            and ni.get("source_commit") == REFERENCE_UPSTREAM_COMMIT and ni.get("source_bytes") == REFERENCE_BYTES
            and ni.get("original_function") == "test_batch_prediction" and ni.get("evidence_scope") == SCOPE
            and ni.get("ordered_occurrences") == OCCURRENCES and ni.get("cohort_sha256") == ci.get("cohort_sha256")
            and ni.get("native_device") == "cuda" and ni.get("padding") == "NATIVE_RIGHT"
            and ni.get("model_dtype") == "FP32" and ni.get("use_cache") is False
            and native.get("identity_sha256") == digest(ni)
            and native.get("canonical_payload_sha256") == value["canonical_payload_sha256"]
            and all(native.get(key) is True for key in ("raw_local_only", "model_no_mutation", "RNG_restored"))
            and native.get("checkpoint_saved") is False, "NATIVE_PARITY_ORIGINAL_SOURCE_IDENTITY_OR_GUARDS")
    left, right = canonical["cases"], native["cases"]
    from official.evaluation.factual import validate_retained_observation, retained_query_work
    validate_retained_observation(canonical,"cf",input_signature=native["case_signatures"],batch_size=16)
    for endpoint, per_case in ((canonical,False),(native,True)):
        derived = retained_query_work(native["case_signatures"],"cf",batch_size=16,per_case=per_case)
        require(all(type(endpoint["work"].get(key)) is int and endpoint["work"][key] == count
                    for key,count in derived.items()), "NATIVE_PARITY_OBSERVATION_DERIVED_WORK")
    require(type(left) is list and type(right) is list and len(left) == len(right) == REQUESTS,
            "NATIVE_PARITY_EXACT_FIRST300_RAW")
    require(ni.get("locked_cohort_sha256") == digest([dict(case_id=row["case_id"], occurrence_index=row["occurrence_index"])
            for row in left]) and digest(native["case_signatures"]) == ci.get("cohort_sha256"),
            "NATIVE_PARITY_NATIVE_LOCKED_COHORT_SIGNATURE")
    candidates = 0
    for ordinal, (a, b) in enumerate(zip(left, right), 1):
        require(a["case_id"] == b["case_id"] and a["occurrence_index"] == b["occurrence_index"] == ordinal,
                "NATIVE_PARITY_RAW_COHORT_IDENTITY")
        for kind in GROUPS:
            ar, br = a[kind + "_prompts_probs"], b[kind + "_prompts_probs"]
            ac, bc = a[kind + "_prompts_correct"], b[kind + "_prompts_correct"]
            require(type(ar) is list and type(br) is list and len(ar) == len(br) > 0
                    and type(ac) is list and type(bc) is list and ac == bc
                    and len(ac) == len(ar) and len(bc) == len(br)
                    and all(type(x) is bool for x in ac + bc), "NATIVE_PARITY_STRICT_RAW_BOOLEAN")
            for canonical_row, native_row in zip(ar, br):
                for target in ("target_new", "target_true"):
                    x, y = canonical_row[target], native_row[target]
                    require(_finite(x) and _finite(y) and x >= 0 and y >= 0
                            and abs(x - y) <= 1e-4 + 1e-5 * abs(y), "NATIVE_PARITY_RAW_NLL_MISMATCH")
                preference = lambda row: row["target_true"] < row["target_new"] if kind == "neighborhood" else row["target_new"] < row["target_true"]
                require(preference(canonical_row) == preference(native_row), "NATIVE_PARITY_RAW_STRICT_PREFERENCE")
            candidates += len(ar) * 2
    for endpoint in (canonical, native):
        reduced = counterfact(endpoint["cases"])
        for key, expected in reduced.items():
            require(_finite(endpoint["summary"].get(key)) and abs(endpoint["summary"][key] - expected) <= 1e-10,
                    "NATIVE_PARITY_REQUEST_MACRO_REDUCTION")
    _work(canonical["work"], candidates, native=False)
    _work(native["work"], candidates, native=True)
    require(comparison["work"] == native["work"] and all(native["work"][key] == canonical["work"][key]
            for key in ("candidate_sequences", "physical_input_tokens", "target_tokens")),
            "NATIVE_PARITY_INDEPENDENT_SAME_TOKEN_WORK")
    return dict(schema="official-server1-native-CF-parity-CPU-validation-v1", status="PASS", evidence_scope=SCOPE,
        requests=REQUESTS, completed_batch=3, plan_sha256=plan_sha,
        upstream_source_sha256=REFERENCE_SOURCE_SHA256, original_native_forward_calls=REQUESTS,
        existing_canonical_forward_calls=canonical["work"]["forward_calls"],
        additional_canonical_forward_calls=0, canonical_observation=PLAN["canonical_observation"],
        candidate_sequences=candidates,
        full_2k_parity="NOT_OBSERVED", performance_gate=False, additional_fit_calls=0, generation_calls=0,
        raw_local_only=True, case_ids_tokens_prompts_omitted=True)
