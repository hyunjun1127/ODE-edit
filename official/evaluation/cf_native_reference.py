"""Hash-pinned, unchanged AlphaEdit CounterFact factual oracle.

This is an observation/comparison tool, not a production evaluator or fit path.
Only ``test_batch_prediction`` is compiled from the original source. Native
logits always come from its own model call; canonical observations are used
only as the other side of the comparison. All returned raw data is local-only.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import time
import typing

import numpy as np
from scipy.stats import hmean
import torch

from .factual import FactualError, SCHEMA as CANONICAL_SCHEMA, TOKENIZATION, _observation


SCHEMA = "official-cf-original-native-reference-v1"
UPSTREAM_COMMIT = "b84624f44dfe8fc6cd9e41df916c44124a0c46dc"
SOURCE_SHA256 = "25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145"
SOURCE_BYTES = 7941
SOURCE_PATH = Path(__file__).with_name("reference") / "alphaedit_eval_utils_counterfact.py.txt"
NLL_ABS_TOL_NATS, NLL_REL_TOL, AGGREGATE_ABS_TOL = 1e-4, 1e-5, 1e-10
CPU_SCOPE = "TEST_ONLY_CPU_FIXTURE"
SMOKE_SCOPE = "ACTUAL_GPU_SMOKE"
FULL_SCOPE = "ACTUAL_FULL_2K"
MATCHED_SCOPE = "ACTUAL_MATCHED_SUBSET"
GROUPS = ("rewrite", "paraphrase", "neighborhood")
LABELS = ("Efficacy", "Generalization", "Specificity")


class CFNativeReferenceError(ValueError):
    """A qualification/identity failure; never a zero score or relaxed gate."""


def _require(condition, code):
    if not condition:
        raise CFNativeReferenceError(code)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _clone(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise CFNativeReferenceError("CF_NATIVE_NON_JSON_OR_NONFINITE_INPUT") from exc


def _checked_digest(value):
    try:
        return _digest(value)
    except (TypeError, ValueError) as exc:
        raise CFNativeReferenceError("CF_NATIVE_NON_JSON_OR_NONFINITE_INPUT") from exc


def load_original_counterfact(source_path=SOURCE_PATH):
    """Compile only the byte-verified original function, with no AST edits.

    The full source is never imported: its generation, sklearn, NLTK, dsets,
    transformers and util imports consequently never run. No device rewrite is
    applied to the function, including in CPU tests.
    """
    path = Path(source_path)
    _require(path.is_file() and not path.is_symlink(), "CF_NATIVE_SOURCE_REGULAR_FILE")
    source = path.read_bytes()
    _require(len(source) == SOURCE_BYTES and hashlib.sha256(source).hexdigest() == SOURCE_SHA256,
             "CF_NATIVE_SOURCE_SHA256")
    tree = ast.parse(source, filename=str(path))
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "test_batch_prediction"]
    _require(len(matches) == 1 and not matches[0].decorator_list, "CF_NATIVE_SOURCE_FUNCTION")
    module = ast.Module(body=[matches[0]], type_ignores=[])
    namespace = {"typing": typing, "np": np, "torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["test_batch_prediction"]


def _ids(tok, text, *, special=None):
    _require(isinstance(text, str) and bool(text), "CF_NATIVE_EMPTY_TEXT")
    options = {} if special is None else {"add_special_tokens": special, "truncation": False}
    ids = tok(text, **options)["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    _require(type(ids) is list and bool(ids)
             and all(type(token) is int and token >= 0 for token in ids), "CF_NATIVE_TOKEN_IDS")
    return ids


def _tokenizer_signature(tok):
    fields = ("name_or_path", "padding_side", "truncation_side", "pad_token_id",
              "bos_token_id", "eos_token_id", "add_bos_token", "add_eos_token",
              "model_max_length", "init_kwargs", "special_tokens_map")
    values = {key: str(getattr(tok, key, None)) for key in fields}
    if callable(getattr(tok, "get_vocab", None)):
        values["vocab"] = tok.get_vocab()
    backend = getattr(tok, "backend_tokenizer", None)
    if backend is not None and callable(getattr(backend, "to_str", None)):
        values["backend"] = backend.to_str()
    return _digest(values)


def _prepare(model, tok, records):
    """Independently bind native arguments and canonical token-query identity."""
    llama = "llama" in str(getattr(model.config, "_name_or_path", "")).lower()
    prepared, signatures, seen = [], [], set()
    for case_index, record in enumerate(records):
        _require(type(record) is dict, "CF_NATIVE_RECORD_SCHEMA")
        case_id = record.get("case_id")
        ordinal = record.get("occurrence_index", record.get("ordered_occurrence"))
        _require(type(case_id) is int and type(ordinal) is int and ordinal > 0,
                 "CF_NATIVE_CASE_OCCURRENCE")
        _require(ordinal not in seen, "CF_NATIVE_DUPLICATE_OCCURRENCE")
        seen.add(ordinal)
        rw = record.get("requested_rewrite")
        _require(type(rw) is dict and isinstance(rw.get("prompt"), str) and rw["prompt"].count("{}") == 1
                 and isinstance(rw.get("subject"), str) and bool(rw["subject"]), "CF_NATIVE_REWRITE_SCHEMA")
        _require(all(type(rw.get(key)) is dict and isinstance(rw[key].get("str"), str)
                     and bool(rw[key]["str"]) for key in ("target_new", "target_true")), "CF_NATIVE_EMPTY_TARGET")
        groups = [[rw["prompt"].format(rw["subject"])], record.get("paraphrase_prompts"),
                  record.get("neighborhood_prompts")]
        _require(all(type(prompts) is list and bool(prompts) for prompts in groups),
                 "CF_NATIVE_MISSING_PROMPTS")
        targets = [rw["target_new"]["str"], rw["target_true"]["str"]]
        _require(all(isinstance(target, str) and bool(target) for target in targets), "CF_NATIVE_EMPTY_TARGET")
        suffixes = [_ids(tok, " " + target, special=False) for target in targets]
        for target, suffix in zip(targets, suffixes):
            native = _ids(tok, " " + target)
            if llama:
                bos = getattr(tok, "bos_token_id", None)
                _require(type(bos) is int and native == [bos] + suffix,
                         "CF_NATIVE_LLAMA_GENUINE_TARGET_BOS_REQUIRED")
            else:
                _require(native == suffix, "CF_NATIVE_NON_LLAMA_TARGET_BOS_FORBIDDEN")
        queries, prefixes, desired = [], [], []
        for kind, prompts in zip(GROUPS, groups):
            for prompt_index, prompt in enumerate(prompts):
                prefix = _ids(tok, prompt)
                if llama:
                    _require(prefix == [tok.bos_token_id] + _ids(tok, prompt, special=False),
                             "CF_NATIVE_LLAMA_GENUINE_PREFIX_BOS_REQUIRED")
                prefixes.append(prompt)
                desired.append(1 if kind == "neighborhood" else 0)
                for target_kind, target, suffix in zip(("new", "true"), targets, suffixes):
                    combined = _ids(tok, prompt + " " + target)
                    _require(combined == prefix + suffix, "CF_NATIVE_TARGET_CONCAT_BOUNDARY")
                    queries.append(dict(kind=kind, prompt_index=prompt_index, prompt=prompt,
                        target_kind=target_kind, target=target, input_token_ids=combined,
                        target_start=len(prefix), target_token_ids=suffix))
        batched_prefixes = tok(prefixes)["input_ids"]
        _require(batched_prefixes == [_ids(tok, prompt) for prompt in prefixes],
                 "CF_NATIVE_BATCH_PREFIX_IDENTITY")
        limits = [value for owner, names in ((model.config, ("max_position_embeddings", "n_positions",
                  "max_sequence_length")), (tok, ("model_max_length",))) for name in names
                  for value in [getattr(owner, name, None)] if type(value) is int and 0 < value < 10**9]
        _require(not limits or max(len(query["input_token_ids"]) for query in queries) <= min(limits),
                 "CF_NATIVE_CONTEXT_LENGTH_EXCEEDED")
        signatures.append(dict(case_id=case_id, occurrence_index=ordinal, queries=queries))
        prepared.append(dict(case_id=case_id, occurrence_index=ordinal, case_index=case_index,
            groups=groups, prefixes=prefixes, desired=desired, targets=targets, queries=queries))
    _require(bool(prepared), "CF_NATIVE_EMPTY_COHORT")
    return prepared, signatures


class _TestOnlyCudaBatch(dict):
    """TEST ONLY: preserve literal native .to('cuda'), transport to CPU instead."""
    def to(self, device):
        _require(device == "cuda", "CF_NATIVE_TEST_ONLY_TRANSPORT")
        return {key: value.to("cpu") for key, value in self.items()}


class _TestOnlyTokenizer:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, *args, **kwargs):
        result = self.tok(*args, **kwargs)
        return _TestOnlyCudaBatch(result) if kwargs.get("return_tensors") == "pt" else result


class _NativeCallModel:
    """Check actual native inputs/output and count work, never score logits."""
    def __init__(self, model, queries, work, test_only_cpu):
        self.model, self.config, self.queries = model, model.config, queries
        self.work, self.test_only_cpu = work, test_only_cpu

    def __call__(self, **batch):
        ids, mask = batch["input_ids"], batch.get("attention_mask")
        expected = [query["input_token_ids"] for query in self.queries]
        width, pad = max(map(len, expected)), self.work["pad_token_id"]
        _require(ids.shape == (len(expected), width) and mask is not None and mask.shape == ids.shape,
                 "CF_NATIVE_PADDED_BATCH_SHAPE")
        _require(ids.device.type == ("cpu" if self.test_only_cpu else "cuda"),
                 "CF_NATIVE_ACTUAL_INPUT_DEVICE")
        _require(ids.detach().cpu().tolist() == [row + [pad] * (width-len(row)) for row in expected]
                 and mask.detach().cpu().tolist() == [[1]*len(row)+[0]*(width-len(row)) for row in expected],
                 "CF_NATIVE_PADDED_TOKEN_IDENTITY")
        before = (ids._version, mask._version)
        self.work["forward_calls"] += 1
        self.work["candidate_sequences"] += len(expected)
        self.work["physical_input_tokens"] += sum(map(len, expected))
        self.work["padded_input_tokens"] += ids.numel()
        self.work["target_tokens"] += sum(len(query["target_token_ids"]) for query in self.queries)
        output = self.model(**batch)
        _require((ids._version, mask._version) == before, "CF_NATIVE_MODEL_INPUT_MUTATED")
        _require(isinstance(output.logits, torch.Tensor) and output.logits.dtype == torch.float32
                 and output.logits.shape[:2] == ids.shape and output.logits.ndim == 3,
                 "CF_NATIVE_FP32_LOGITS_REQUIRED")
        _require(all(max(query["target_token_ids"]) < output.logits.shape[-1] for query in self.queries),
                 "CF_NATIVE_TARGET_OUT_OF_VOCAB")
        return output


def _aggregate(cases):
    """Independent original NumPy request macro; display rounding is separate."""
    summary, display = {}, {"success": {}, "strict_accuracy": {}}
    for kind, label in zip(GROUPS, LABELS):
        rates = [np.mean([row["target_true"] < row["target_new"] if kind == "neighborhood"
                        else row["target_new"] < row["target_true"]
                        for row in case[kind + "_prompts_probs"]]) for case in cases]
        accuracy = [np.mean(case[kind + "_prompts_correct"]) for case in cases]
        summary[label] = float(np.mean(rates) * 100)
        display["success"][label] = [float(np.around(value * 100, 2))
                                     for value in (np.mean(rates), np.std(rates))]
        display["strict_accuracy"][label] = [float(np.around(value * 100, 2))
                                             for value in (np.mean(accuracy), np.std(accuracy))]
    summary["Score"] = float(hmean([summary[label] for label in LABELS]))
    display["Score"] = float(hmean([display["success"][label][0] for label in LABELS]))
    summary.update(Score_AlphaEdit_display=display["Score"], requests=len(cases))
    return summary, display


def _qualification(scope, status):
    evidence = {name: "NOT_OBSERVED" for name in (CPU_SCOPE, SMOKE_SCOPE, MATCHED_SCOPE, FULL_SCOPE)}
    if scope in evidence and status in ("PASS", "CPU_FIXTURE_PASS", "MISMATCH"):
        evidence[scope] = "PASS" if status in ("PASS", "CPU_FIXTURE_PASS") else "FAIL"
    return evidence


def _profile(model, tok, identity, binding, scope, locked_cohort, records, test_only_cpu, state_callback):
    _require(scope in (CPU_SCOPE, SMOKE_SCOPE, MATCHED_SCOPE, FULL_SCOPE), "CF_NATIVE_EVIDENCE_SCOPE")
    _require(test_only_cpu == (scope == CPU_SCOPE), "CF_NATIVE_TEST_ONLY_SCOPE_REQUIRED")
    _require(type(identity) is dict and all(binding.get(key) for key in
        ("model_identity", "tokenizer_identity", "state_identity")), "CF_NATIVE_IDENTITY_LOCKS_REQUIRED")
    _require(getattr(tok, "padding_side", None) == "right", "CF_NATIVE_NATIVE_RIGHT_PADDING_REQUIRED")
    _require(type(getattr(tok, "pad_token_id", None)) is int and tok.pad_token_id >= 0,
             "CF_NATIVE_PAD_TOKEN_REQUIRED")
    _require(hasattr(model, "config") and isinstance(getattr(model.config, "_name_or_path", None), str),
             "CF_NATIVE_MODEL_NAME_REQUIRED")
    floating = [tensor for tensor in list(model.parameters()) + list(model.buffers())
                if tensor.is_floating_point()]
    _require(bool(floating) and all(tensor.dtype == torch.float32 for tensor in floating),
             "CF_NATIVE_FP32_MODEL_REQUIRED")
    _require(getattr(model.config, "use_cache", None) is False, "CF_NATIVE_CALLER_USE_CACHE_FALSE_REQUIRED")
    if test_only_cpu:
        _require(all(tensor.device.type == "cpu" for tensor in floating), "CF_NATIVE_TEST_ONLY_CPU_MODEL")
    else:
        _require(all(not module.training for module in model.modules()), "CF_NATIVE_CALLER_EVAL_REQUIRED")
        _require(all(tensor.device.type == "cuda" for tensor in floating), "CF_NATIVE_REAL_CUDA_REQUIRED")
        _require(callable(state_callback), "CF_NATIVE_ACTUAL_STATE_CALLBACK_REQUIRED")
        _require(not torch.is_autocast_enabled(), "CF_NATIVE_CALLER_NO_AUTOCAST_REQUIRED")
        _require(getattr(model.config, "_attn_implementation", None) == "eager"
                 and torch.backends.cuda.matmul.allow_tf32 is False
                 and torch.backends.cudnn.allow_tf32 is False, "CF_NATIVE_CALLER_EAGER_TF32_OFF_REQUIRED")
        _require(type(locked_cohort) is list and bool(locked_cohort), "CF_NATIVE_LOCKED_COHORT_REQUIRED")
        expected = [dict(case_id=row["case_id"], occurrence_index=row.get(
            "occurrence_index", row.get("ordered_occurrence"))) for row in records]
        _require(locked_cohort[:len(records)] == expected
                 and [row["occurrence_index"] for row in expected] == list(range(1, len(records)+1)),
                 "CF_NATIVE_LOCKED_FIRST_COHORT_IDENTITY")
        if scope == SMOKE_SCOPE:
            _require(len(records) == 4, "CF_NATIVE_SMOKE_LOCKED_FIRST4_REQUIRED")
        elif scope == FULL_SCOPE:
            method = identity.get("method")
            if isinstance(binding["model_identity"], dict):
                method = binding["model_identity"].get("method", method)
            _require(len(records) == 2000 and "llama" in model.config._name_or_path.lower()
                     and method in ("ALPHAEDIT", "AlphaEdit"), "CF_NATIVE_LLAMA_ALPHAEDIT_FULL2K_REQUIRED")


def _check_canonical(canonical, identity, signatures):
    _require(type(canonical) is dict, "CF_NATIVE_CANONICAL_MAPPING")
    ci = canonical.get("identity", {})
    _require(type(ci) is dict and ci.get("schema") == CANONICAL_SCHEMA and ci.get("dataset") == "cf"
             and ci.get("tokenization") == TOKENIZATION and ci.get("use_cache") is False
             and ci.get("padding") == "RIGHT_EXPLICIT_ATTENTION_MASK", "CF_NATIVE_CANONICAL_PROFILE")
    _require(canonical.get("identity_sha256") == _digest(ci), "CF_NATIVE_CANONICAL_IDENTITY_HASH")
    _require(ci.get("external_identity") == identity, "CF_NATIVE_CANONICAL_MODEL_TOKEN_STATE_IDENTITY")
    _require(ci.get("cohort_sha256") == _digest(signatures)
             and ci.get("ordered_occurrences") == [row["occurrence_index"] for row in signatures],
             "CF_NATIVE_CANONICAL_COHORT_TOKEN_IDENTITY")
    _require(all(canonical.get(key) is True for key in ("raw_local_only", "model_no_mutation", "RNG_restored")),
             "CF_NATIVE_CANONICAL_OBSERVATION_GUARDS")
    cases = canonical.get("cases")
    _require(type(cases) is list and len(cases) == len(signatures), "CF_NATIVE_CANONICAL_CASES")
    _require(type(canonical.get("summary")) is dict, "CF_NATIVE_CANONICAL_SUMMARY")
    for case_index, (case, signature) in enumerate(zip(cases, signatures)):
        _require(type(case) is dict and all(case.get(key) == signature[key] for key in ("case_id", "occurrence_index")),
                 "CF_NATIVE_CANONICAL_CASE_IDENTITY")
        for kind in GROUPS:
            queries = [query for query in signature["queries"] if query["kind"] == kind]
            observations = case.get(kind + "_observations", [])
            probs, correct = case.get(kind + "_prompts_probs", []), case.get(kind + "_prompts_correct", [])
            _require(all(type(value) is list for value in (observations, probs, correct))
                     and len(observations) == len(probs) == len(correct) == len(queries)//2,
                     "CF_NATIVE_CANONICAL_ROW_COUNT")
            for index, observation in enumerate(observations):
                _require(type(observation) is dict and type(probs[index]) is dict,
                         "CF_NATIVE_CANONICAL_ROW_SCHEMA")
                for offset, name in enumerate(("target_new", "target_true")):
                    row, expected = observation.get(name), queries[index*2 + offset]
                    _require(type(row) is dict and all(row.get(key) == value for key, value in expected.items())
                             and row.get("case_index") == case_index
                             and row.get("occurrence_index") == case["occurrence_index"],
                             "CF_NATIVE_CANONICAL_RAW_TOKEN_ROW_IDENTITY")
                    value = probs[index].get(name)
                    _require(type(value) in (int, float) and math.isfinite(value)
                             and value == row.get("mean_nll"), "CF_NATIVE_CANONICAL_NLL_ROW")
                desired = observation["target_true" if kind == "neighborhood" else "target_new"]
                _require(type(correct[index]) is bool and correct[index] == desired.get("strict_correct"),
                         "CF_NATIVE_CANONICAL_STRICT_ROW")


def _run(model, tok, records, *, identity, evidence_scope, locked_cohort,
         state_callback, source_path, test_only_cpu, work, canonical=None,
         model_identity=None, tokenizer_identity=None, state_identity=None):
    original = load_original_counterfact(source_path)
    identity = _clone(identity)
    _require(type(identity) is dict, "CF_NATIVE_EXTERNAL_IDENTITY_MAPPING")
    binding = _clone(dict(model_identity=model_identity if model_identity is not None else identity.get("model_identity"),
        tokenizer_identity=tokenizer_identity if tokenizer_identity is not None else identity.get("tokenizer_identity"),
        state_identity=state_identity if state_identity is not None else identity.get("state_identity")))
    _profile(model, tok, identity, binding, evidence_scope, locked_cohort, records, test_only_cpu, state_callback)
    input_before, tok_before = _checked_digest(records), _tokenizer_signature(tok)
    canonical_before = _checked_digest(canonical) if canonical is not None else None
    start = time.monotonic()
    with _observation(model):
        state_before = _clone(state_callback()) if state_callback is not None else None
        _require(state_callback is None or state_before == binding["state_identity"],
                 "CF_NATIVE_ACTUAL_STATE_IDENTITY")
        prepared, signatures = _prepare(model, tok, records)
        if canonical is not None:
            _check_canonical(canonical, identity, signatures)
        cases = []
        native_tok = _TestOnlyTokenizer(tok) if test_only_cpu else tok
        work["pad_token_id"] = tok.pad_token_id
        for item in prepared:
            proxy = _NativeCallModel(model, item["queries"], work, test_only_cpu)
            device_type = "cpu" if test_only_cpu else "cuda"
            with torch.autocast(device_type=device_type, enabled=False):
                probs, correct = original(proxy, native_tok, item["prefixes"], item["desired"], *item["targets"])
            _require(len(probs) == len(correct) == len(item["prefixes"])
                     and all(type(value) is bool for value in correct)
                     and all(type(row[key]) is float and math.isfinite(row[key])
                             for row in probs for key in ("target_new", "target_true")),
                     "CF_NATIVE_ORIGINAL_RESULT_SHAPE_OR_NUMERIC")
            case, offset = dict(case_id=item["case_id"], occurrence_index=item["occurrence_index"]), 0
            for kind, prompts in zip(GROUPS, item["groups"]):
                end = offset + len(prompts)
                case[kind + "_prompts_probs"], case[kind + "_prompts_correct"] = probs[offset:end], correct[offset:end]
                offset = end
            cases.append(case)
        _require(_digest(records) == input_before, "CF_NATIVE_RECORD_INPUT_MUTATED")
        _require(canonical is None or _checked_digest(canonical) == canonical_before,
                 "CF_NATIVE_CANONICAL_INPUT_MUTATED")
        _require(_tokenizer_signature(tok) == tok_before, "CF_NATIVE_TOKENIZER_MUTATED")
        _require(state_callback is None or _clone(state_callback()) == state_before,
                 "CF_NATIVE_EXTERNAL_STATE_MUTATED")
    work["seconds"] = time.monotonic() - start
    summary, display = _aggregate(cases)
    native_identity = dict(schema=SCHEMA, external_identity=identity, reference_binding=binding, source_commit=UPSTREAM_COMMIT,
        source_sha256=SOURCE_SHA256, source_bytes=SOURCE_BYTES, original_function="test_batch_prediction",
        evidence_scope=evidence_scope, cohort_sha256=_digest(signatures),
        locked_cohort_sha256=_digest(locked_cohort) if locked_cohort is not None else None,
        tokenizer_signature_sha256=tok_before, native_model_name=model.config._name_or_path,
        ordered_occurrences=[row["occurrence_index"] for row in signatures],
        native_device="TEST_ONLY_CPU_CUDA_TRANSPORT_BRIDGE" if test_only_cpu else "cuda",
        padding="NATIVE_RIGHT", use_cache=False, model_dtype="FP32")
    return dict(schema=SCHEMA, status="OBSERVED_UNCOMPARED", identity=native_identity,
        identity_sha256=_digest(native_identity), summary=summary, original_display=display,
        cases=cases, case_signatures=signatures, work=work, evidence=_qualification(evidence_scope, ""),
        raw_local_only=True, model_no_mutation=True, RNG_restored=True, checkpoint_saved=False,
        canonical_payload_sha256=canonical_before)


def _work():
    return dict(forward_calls=0, candidate_sequences=0, physical_input_tokens=0,
                padded_input_tokens=0, target_tokens=0, seconds=0.0)


def _not_qualified(scope, exc, work):
    return dict(schema=SCHEMA, status="NOT_QUALIFIED", reasons=[dict(type="QUALIFICATION", code=str(exc))],
        evidence_scope=scope, evidence=_qualification(scope, ""), work=work, raw_local_only=True,
        checkpoint_saved=False)


def evaluate_native_counterfact(model, tokenizer, records, *, identity,
        evidence_scope=SMOKE_SCOPE, locked_cohort=None, state_callback=None,
        source_path=SOURCE_PATH, test_only_cpu=False, model_identity=None,
        tokenizer_identity=None, state_identity=None):
    """Observe the original rows, without declaring a canonical parity PASS.

    Actual callers supply eval/FP32/CUDA, config.use_cache=False, native right
    padding, immutable cohort locks and a matching live state callback. CPU
    fixtures must explicitly set BOTH test_only_cpu=True and CPU_SCOPE.
    Known qualification failures return NOT_QUALIFIED; unexpected model errors
    propagate after the existing observation guard restores modes and RNG.
    """
    work = _work()
    started = time.monotonic()
    try:
        return _run(model, tokenizer, list(records), identity=identity, evidence_scope=evidence_scope,
            locked_cohort=locked_cohort, state_callback=state_callback, source_path=source_path,
            test_only_cpu=test_only_cpu, work=work, model_identity=model_identity,
            tokenizer_identity=tokenizer_identity, state_identity=state_identity)
    except (CFNativeReferenceError, FactualError) as exc:
        work["seconds"] = time.monotonic() - started
        return _not_qualified(evidence_scope, exc, work)


def compare_native_counterfact(model, tokenizer, records, canonical_result, *, identity,
        evidence_scope=SMOKE_SCOPE, locked_cohort=None, state_callback=None,
        source_path=SOURCE_PATH, test_only_cpu=False, model_identity=None,
        tokenizer_identity=None, state_identity=None):
    """Reuse exact canonical raw, but ALWAYS run independent original forwards.

    NLL: abs <= 1e-4 + 1e-5*abs(native), in nats. Desired strict argmax and NLL
    preference booleans: exact, including near ties. Raw request macros and
    original NumPy-rounded display: separately abs <= 1e-10. No tolerance
    overrides, scientific performance promotion, fitting or retry fallback.
    """
    work = _work()
    started = time.monotonic()
    try:
        native = _run(model, tokenizer, list(records), identity=identity, evidence_scope=evidence_scope,
            locked_cohort=locked_cohort, state_callback=state_callback, source_path=source_path,
            test_only_cpu=test_only_cpu, work=work, canonical=canonical_result,
            model_identity=model_identity, tokenizer_identity=tokenizer_identity, state_identity=state_identity)
    except (CFNativeReferenceError, FactualError) as exc:
        work["seconds"] = time.monotonic() - started
        return _not_qualified(evidence_scope, exc, work)
    mismatches = []
    for actual, expected in zip(native["cases"], canonical_result["cases"]):
        for kind in GROUPS:
            for index, (a, b) in enumerate(zip(actual[kind + "_prompts_probs"], expected[kind + "_prompts_probs"])):
                location = dict(case_id=actual["case_id"], occurrence_index=actual["occurrence_index"],
                                kind=kind, prompt_index=index)
                for target in ("target_new", "target_true"):
                    delta = abs(a[target]-b[target])
                    limit = NLL_ABS_TOL_NATS + NLL_REL_TOL * abs(a[target])
                    if delta > limit:
                        mismatches.append(dict(type="NLL", **location, target=target,
                            native=a[target], canonical=b[target], abs_difference_nats=delta, allowed_nats=limit))
                preference = lambda row: row["target_true"] < row["target_new"] if kind == "neighborhood" else row["target_new"] < row["target_true"]
                if preference(a) != preference(b):
                    mismatches.append(dict(type="STRICT_NLL_PREFERENCE", **location,
                                           native=preference(a), canonical=preference(b)))
                a_correct, b_correct = actual[kind + "_prompts_correct"][index], expected[kind + "_prompts_correct"][index]
                if a_correct != b_correct:
                    mismatches.append(dict(type="STRICT_DESIRED_ARGMAX", **location,
                                           native=a_correct, canonical=b_correct))
    for label in (*LABELS, "Score", "requests"):
        value = canonical_result.get("summary", {}).get(label)
        mismatch = (type(value) is not int or value != native["summary"][label]) if label == "requests" else (
            type(value) not in (int, float) or not math.isfinite(value) or abs(native["summary"][label]-value) > AGGREGATE_ABS_TOL)
        if mismatch:
            mismatches.append(dict(type="REQUEST_MACRO", metric=label, native=native["summary"][label], canonical=value))
    _, canonical_display = _aggregate(canonical_result["cases"])
    display_mismatches = []
    if native["original_display"] != canonical_display:
        display_mismatches.append(dict(type="ORIGINAL_DISPLAY_ROWS", native=native["original_display"], canonical=canonical_display))
    value = canonical_result.get("summary", {}).get("Score_AlphaEdit_display")
    if type(value) not in (int, float) or not math.isfinite(value) or abs(native["original_display"]["Score"]-value) > AGGREGATE_ABS_TOL:
        display_mismatches.append(dict(type="ORIGINAL_DISPLAY_SCORE", native=native["original_display"]["Score"], canonical=value))
    status = "MISMATCH" if mismatches or display_mismatches else "CPU_FIXTURE_PASS" if test_only_cpu else "PASS"
    return dict(schema=SCHEMA, status=status, evidence_scope=evidence_scope,
        evidence=_qualification(evidence_scope, status), mismatches=mismatches,
        display_mismatches=display_mismatches, tolerances=dict(nll_abs_nats=NLL_ABS_TOL_NATS,
            nll_relative_to_native=NLL_REL_TOL, strict_booleans="EXACT", aggregate_abs=AGGREGATE_ABS_TOL),
        native=native, canonical_identity_sha256=canonical_result["identity_sha256"], work=work,
        raw_local_only=True, checkpoint_saved=False, scientific_performance_promotion=False)
