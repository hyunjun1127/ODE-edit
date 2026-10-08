"""Independent CPU verification of official factual raw and native commit ledgers.

No model is loaded/called and no tensor is deserialized. Token planning reuses
the one canonical official evaluator; numerical/count reduction is independently
recomputed from the stored token observations. Returned metadata is compact,
while prompts, per-case IDs and tokens stay in the supplied local raw only.
"""
from pathlib import Path
import math
import re

import numpy as np

from official.baselines.registry import call_options, hparams, requests as native_requests
from official.evaluation.factual import (SCHEMA, TOKENIZATION, _digest, _plan,
                                          _reference_lookup)
from official.evaluation.reduce import counterfact, zsre
from official.experiments.prepare import digest, file_sha

from .common import read, verify


class AuditError(ValueError):
    pass


def require(value, code):
    if not value:
        raise AuditError(code)


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _candidate(row, planned):
    """Validate the whole serialized query and its token numerators, not means."""
    require(type(row) is dict and all(row.get(key) == value for key, value in planned.items()),
            "AUDIT_FACTUAL_QUERY_TOKEN_IDENTITY")
    gold = planned["target_token_ids"]
    predictions, correct, nll = (row.get(name) for name in
                                 ("predicted_token_ids", "token_correct", "nll_by_token"))
    require(type(predictions) is list and len(predictions) == len(gold)
            and all(type(token) is int and token >= 0 for token in predictions),
            "AUDIT_FACTUAL_PREDICTION_SHAPE")
    expected_correct = [actual == target for actual, target in zip(predictions, gold)]
    require(type(correct) is list and all(type(value) is bool for value in correct)
            and correct == expected_correct, "AUDIT_FACTUAL_TOKEN_NUMERATOR")
    require(type(row.get("token_count")) is int and row["token_count"] == len(gold)
            and type(row.get("token_correct_count")) is int
            and row["token_correct_count"] == sum(expected_correct)
            and type(row.get("strict_correct")) is bool
            and row["strict_correct"] == all(expected_correct), "AUDIT_FACTUAL_TOKEN_DENOMINATOR")
    require(type(nll) is list and len(nll) == len(gold)
            and all(_finite_number(value) and value >= 0 for value in nll),
            "AUDIT_FACTUAL_NLL_TOKEN_FINITE_COUNT")
    total = np.float32(0)
    for value in nll:
        total = np.float32(total + np.float32(value))
    expected_mean = float(np.float32(total / np.float32(len(nll))))
    require(_finite_number(row.get("mean_nll")) and row["mean_nll"] == expected_mean,
            "AUDIT_FACTUAL_NATIVE_FP32_NLL_MEAN")
    return dict(tokens=len(gold), correct=sum(expected_correct),
                strict=all(expected_correct), mean_nll=expected_mean)


def _accuracy(values):
    require(bool(values), "AUDIT_EMPTY_ACCURACY_GROUP")
    total = sum(row["tokens"] for row in values)
    correct = sum(row["correct"] for row in values)
    return dict(prompt_count=len(values), token_count=total, token_correct_count=correct,
        token_acc_pct=100 * correct / total,
        prompt_acc_pct=100 * math.fsum(row["correct"] / row["tokens"] for row in values) / len(values),
        strict_acc_pct=100 * sum(row["strict"] for row in values) / len(values))


def _scientific(value):
    return {key: item for key, item in value.items() if key != "work"}


def audit_factual(endpoint, records, dataset, tokenizer, external, w0_reference=None):
    """Reject wrong occurrence/prompt/token/source identity before CPU reduction."""
    records = list(records)
    expected_cases, queries, signatures = _plan(records, dataset, tokenizer)
    require(type(endpoint) is dict and endpoint.get("raw_local_only") is True
            and endpoint.get("model_no_mutation") is True and endpoint.get("RNG_restored") is True,
            "AUDIT_FACTUAL_OBSERVATION_FLAGS")
    cases = endpoint.get("cases")
    require(type(cases) is list and len(cases) == len(expected_cases)
            and [(row.get("case_id"), row.get("occurrence_index")) for row in cases] ==
                [(row["case_id"], row["occurrence_index"]) for row in expected_cases],
            "AUDIT_FACTUAL_ORDERED_COHORT")
    reference_lookup = None
    reference_sha = None
    if dataset == "zsre" and w0_reference is not None:
        reference_lookup = _reference_lookup(w0_reference, signatures)
        require(w0_reference["identity"].get("external_identity", {}) == external,
                "AUDIT_W0_EXTERNAL_IDENTITY")
        # The builder cannot include its own reference SHA in the already-made
        # W0 endpoint. This one exception requires exact self-observation bytes.
        is_self = (endpoint["identity"].get("W0_reference_sha256") is None
                   and _scientific(endpoint) == _scientific(w0_reference["evaluation"]))
        reference_sha = None if is_self else w0_reference["identity_sha256"]
    expected_identity = dict(schema=SCHEMA, dataset=dataset, tokenization=TOKENIZATION,
        cohort_sha256=_digest(signatures),
        ordered_occurrences=[row["occurrence_index"] for row in expected_cases],
        external_identity=external, padding="RIGHT_EXPLICIT_ATTENTION_MASK", use_cache=False,
        W0_reference_sha256=reference_sha)
    require(endpoint.get("identity") == expected_identity
            and endpoint.get("identity_sha256") == _digest(expected_identity),
            "AUDIT_FACTUAL_ENDPOINT_IDENTITY")
    planned = {(query["case_index"], query["kind"], query["prompt_index"], query["target_kind"]): query
               for query in queries}
    groups = {kind: [] for kind in ("rewrite", "paraphrase", "neighborhood")}
    audited_candidates = 0
    for index, case in enumerate(cases):
        for kind, values in groups.items():
            expected_prompts = len([query for query in signatures[index]["queries"]
                                    if query["kind"] == kind and query["target_kind"] in ("new", "loc_ans")])
            observations = case.get(kind + "_observations")
            require(type(observations) is list and len(observations) == expected_prompts,
                    "AUDIT_FACTUAL_PROMPT_DENOMINATOR")
            if dataset == "cf":
                probs, strict = [], []
                for prompt_index, observation in enumerate(observations):
                    new_query = planned[(index, kind, prompt_index, "new")]
                    true_query = planned[(index, kind, prompt_index, "true")]
                    require(observation.get("prompt_index") == prompt_index
                            and observation.get("prompt") == new_query["prompt"],
                            "AUDIT_CF_PROMPT_INDEX")
                    new = _candidate(observation.get("target_new"), new_query)
                    true = _candidate(observation.get("target_true"), true_query)
                    desired = true if kind == "neighborhood" else new
                    require(observation.get("desired_target") ==
                                ("true" if kind == "neighborhood" else "new"), "AUDIT_CF_DESIRED_TARGET")
                    require(observation.get("margin_true_minus_new") == true["mean_nll"] - new["mean_nll"],
                            "AUDIT_CF_NLL_MARGIN")
                    probs.append(dict(target_new=new["mean_nll"], target_true=true["mean_nll"]))
                    strict.append(desired["strict"])
                    values.append(desired)
                    audited_candidates += 2
                require(case.get(kind + "_prompts_probs") == probs
                        and case.get(kind + "_prompts_correct") == strict,
                        "AUDIT_CF_PROMPT_RAW_SUMMARY")
            else:
                correct, agreement = [], []
                target_kind = "loc_ans" if kind == "neighborhood" else "new"
                for prompt_index, observation in enumerate(observations):
                    query = planned[(index, kind, prompt_index, target_kind)]
                    value = _candidate(observation, query)
                    values.append(value)
                    correct.extend(observation["token_correct"])
                    audited_candidates += 1
                    if kind == "neighborhood" and reference_lookup is not None:
                        previous = reference_lookup[case["occurrence_index"]]["predictions"][prompt_index]
                        agreement.extend(actual == original for actual, original in
                                         zip(observation["predicted_token_ids"], previous))
                require(case.get(kind + "_prompts_correct") == correct, "AUDIT_ZSRE_TOKEN_CORRECT_RAW")
                if kind == "neighborhood":
                    expected_agreement = agreement if reference_lookup is not None else None
                    require(case.get("neighborhood_W0_agreement") == expected_agreement,
                            "AUDIT_ZSRE_W0_PREDICTION_AGREEMENT")
    require(audited_candidates == len(queries), "AUDIT_FACTUAL_CANDIDATE_COVERAGE")
    accuracy = {kind: _accuracy(values) for kind, values in groups.items()}
    require(endpoint.get("accuracy") == accuracy, "AUDIT_FACTUAL_ACCURACY_REDUCTION")
    if dataset == "cf":
        summary = counterfact(cases)
    elif reference_lookup is not None:
        summary = zsre(cases)
    else:
        summary = dict(requests=len(cases), Specificity_availability="NOT_MEASURED_W0_REFERENCE_REQUIRED")
        for kind, label in (("rewrite", "Efficacy"), ("paraphrase", "Generalization"),
                            ("neighborhood", "Specificity_loc_ans")):
            rates = [math.fsum(map(float, case[kind + "_prompts_correct"])) /
                     len(case[kind + "_prompts_correct"]) for case in cases]
            summary[label] = 100 * math.fsum(rates) / len(rates)
    require(endpoint.get("summary") == summary, "AUDIT_FACTUAL_REQUEST_MACRO_REDUCTION")
    if "work" in endpoint:
        work = endpoint["work"]
        require(work.get("candidate_sequences") == len(queries)
                and work.get("target_tokens") == sum(len(q["target_token_ids"]) for q in queries)
                and work.get("physical_input_tokens") == sum(len(q["input_token_ids"]) for q in queries),
                "AUDIT_FACTUAL_LOGICAL_WORK_COUNTS")
        require(_finite_number(work.get("seconds")) and work["seconds"] >= 0
                and type(work.get("forward_calls")) is int
                and 0 < work["forward_calls"] <= len(queries)
                and type(work.get("padded_input_tokens")) is int
                and work["padded_input_tokens"] >= work["physical_input_tokens"],
                "AUDIT_FACTUAL_PHYSICAL_WORK_METADATA")
    return dict(schema="official-factual-CPU-audit-v1", dataset=dataset,
        requests=len(cases), candidate_sequences=audited_candidates,
        prompt_pairs=sum(row["prompt_count"] for row in accuracy.values()),
        groups=accuracy, endpoint_identity_sha256=endpoint["identity_sha256"],
        ordered_stream_and_token_identity=True, independent_raw_reduction=True,
        actual_model_forward_calls=0, GPU_parity_claim=False)


def audit_commits(folder, method, identity, expected20=20, *, records=None):
    """Check the exact ledger and current payload; old checkpoints stay deleted."""
    require(method in ("FT", "MEMIT", "MEMIT_FE") and type(expected20) is int
            and 1 <= expected20 <= 20, "AUDIT_NATIVE_METHOD_OR_BATCH_SCOPE")
    folder = Path(folder)
    names = {f"batch-{batch:02d}.json" for batch in range(1, expected20 + 1)}
    observed = {path.name for path in (folder / "commits").glob("batch-*.json")}
    require(observed == names, "AUDIT_EXACT_COMMIT_LEDGER_NAMES")
    hp = hparams(method, "llama3")
    weights = {hp.rewrite_module_tmp.format(layer) + ("" if method == "FT" else ".weight")
               for layer in hp.layers}
    paths = {"FT": "baselines/easyedit/models/ft/ft_main.py",
             "MEMIT": "baselines/sphere/memit/memit_main.py",
             "MEMIT_FE": "baselines/easyedit/models/memit_FE/memit_FE_main.py"}
    native_sha = file_sha(Path(__file__).resolve().parents[2] / paths[method])
    if records is not None:
        records = list(records)
        require(len(records) >= expected20 * 100, "AUDIT_REQUEST_STREAM_LENGTH")
    previous_post, last_pointer = None, None
    for batch in range(1, expected20 + 1):
        path = folder / "commits" / f"batch-{batch:02d}.json"
        require(path.is_file() and not path.is_symlink(), "AUDIT_COMMIT_REGULAR_FILE")
        receipt = read(path)
        require(receipt.get("batch") == batch and receipt.get("identity") == identity
                and receipt.get("actual_applied_requests") == batch * 100,
                "AUDIT_COMMIT_BATCH_IDENTITY_REQUESTS")
        cursor, pointer = receipt.get("cursor", {}), receipt.get("checkpoint", {})
        require(cursor.get("completed_batch") == batch and pointer.get("batch") == batch
                and pointer.get("identity_sha256") == digest(identity)
                and pointer.get("final_W20") is (batch == 20)
                and _sha(pointer.get("sha256"))
                and pointer.get("file") == f"batch-{batch:02d}-{pointer['sha256'][:16]}.pt",
                "AUDIT_COMMIT_CURSOR_CHECKPOINT")
        edit = cursor.get("edit", {})
        unsigned = {key: value for key, value in edit.items() if key != "identity_sha256"}
        require(edit.get("schema") == "official-server1-native-apply-v1"
                and edit.get("identity_sha256") == digest(unsigned)
                and edit.get("method") == method and edit.get("batch") == batch
                and edit.get("requests") == 100 and _sha(edit.get("request_sha256"))
                and edit.get("native_source_sha256") == native_sha
                and edit.get("call_options") == call_options(method, "llama3"),
                "AUDIT_NATIVE_APPLY_SOURCE_REQUESTS_IDENTITY")
        if records is not None:
            expected_requests = native_requests(records[(batch - 1) * 100:batch * 100], method, "llama3")
            require(edit["request_sha256"] == digest(expected_requests), "AUDIT_NATIVE_REQUEST_STREAM_DIGEST")
        require(edit.get("same_model") is True
                and edit.get("nonselected_identity_version_unchanged") is True
                and edit.get("selected_FP32_finite") is True
                and edit.get("native_history") is False and edit.get("cache_c") == {}
                and edit.get("quality_gate") is False
                and _sha(edit.get("contexts_sha256")), "AUDIT_NATIVE_STATE_GUARDS")
        entry, post = edit.get("entry_selected_sha256", {}), edit.get("post_selected_sha256", {})
        require(set(entry) == set(post) == weights and all(_sha(value) for value in list(entry.values()) +
                                                        list(post.values())), "AUDIT_NATIVE_SELECTED_HASH_SCHEMA")
        require(previous_post is None or entry == previous_post, "AUDIT_NATIVE_ENTRY_POST_WEIGHT_LINK")
        previous_post = post
        if batch in (5, 10, 15, 20):
            require(cursor.get("evaluation") == "COMPLETE" and "factual" in cursor,
                    "AUDIT_SCHEDULED_FACTUAL_CURSOR")
            verify(cursor["factual"])
        if "generation" in cursor:
            require(batch == 20, "AUDIT_GENERATION_W20_ONLY")
            verify(cursor["generation"])
        last_pointer = pointer
    actual_pointer = read(folder / "checkpoint" / "latest.json")
    require(actual_pointer == last_pointer, "AUDIT_FINAL_CHECKPOINT_LEDGER_POINTER")
    payload = folder / "checkpoint" / actual_pointer["file"]
    require(payload.is_file() and not payload.is_symlink()
            and file_sha(payload) == actual_pointer["sha256"], "AUDIT_FINAL_CHECKPOINT_PAYLOAD_SHA")
    require({path.name for path in (folder / "checkpoint").glob("batch-*.pt")} == {payload.name},
            "AUDIT_KEEP_LATEST_ONE_CHECKPOINT")
    return dict(schema="official-native-commit-CPU-audit-v1", method=method,
        actual_native_apply_receipts=expected20, actual_applied_requests=100 * expected20,
        selected_weight_hash_chain=True, native_source_sha256=native_sha,
        native_request_digest_checked_against_stream=records is not None,
        final_checkpoint_payload_SHA_verified=True, historical_checkpoint_payloads_reloaded=False,
        actual_model_forward_calls=0, GPU_resume_parity_claim=False)
