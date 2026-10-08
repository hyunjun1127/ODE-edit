"""Borrow immutable cold W0 inputs without relabelling their producer.

Execution commit/tree/path/job/hardware identities are recorded separately.
Reuse requires equality of explicit consumed computational content, never full
execution identity equality. This reader performs no transfer or model forward.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import math
from functools import wraps


FINGERPRINT_SCHEMA = "official-W0-computational-fingerprint-v1"
READY_SCHEMA = "official-shared-cold-W0-READY-v1"
ZSRE_BINDING_SCHEMA = "official-borrowed-zsre-W0-reference-v1"
COMPONENTS = ("cf_factual", "cf_generation", "zsre_reference")
VERSIONS = ("python", "torch", "transformers", "numpy", "scipy", "sklearn", "nltk")
GENERATION_SOURCES = ("assets.py", "metrics.py", "native_generator.py", "native_observer.py",
                      "native_profile.py", "generator.py", "observer.py", "common.py",
                      "compatibility.py", "progress.py")
REFERENCE_FILES = ("attribute_snippets.json", "idf.npy", "tfidf_vocab.json")


class ReferenceInputError(ValueError):
    """A typed prerequisite failure, not authorization to duplicate W0."""
    def __init__(self, detail, *, missing=False):
        self.code = "REFERENCE_INPUT_MISSING" if missing else "REFERENCE_INPUT_NOT_COMPATIBLE"
        self.detail = detail
        super().__init__(self.code + ": " + str(detail))


def _require(condition, detail):
    if not condition:
        raise ReferenceInputError(detail)


def _typed(function):
    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ReferenceInputError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError, OSError) as error:
            raise ReferenceInputError(function.__name__ + ".MALFORMED_INPUT:" + str(error)) from error
    return call


def _json(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False))
    except (ValueError, TypeError) as error:
        raise ReferenceInputError("NONFINITE_OR_NONJSON_IDENTITY") from error


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(value, field):
    _require(type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value), field)


def _keys(value, keys, field):
    _require(type(value) is dict and set(value) == set(keys), field + ".FIELDS")


def _content_files(value, field):
    _require(type(value) is dict and bool(value), field + ".EMPTY")
    for name, row in value.items():
        _require(type(name) is str and name and not name.startswith("/") and ".." not in Path(name).parts,
                 field + ".LOGICAL_NAME")
        _keys(row, ("bytes", "sha256"), field + ".MEMBER")
        _require(type(row["bytes"]) is int and row["bytes"] > 0, field + ".BYTES")
        _sha(row["sha256"], field + ".SHA256")


@_typed
def computational_fingerprint(content):
    """Validate explicit consumed content; paths/full-tree/hardware are excluded.

    Dataset ``ordered_queries_sha256`` is factual._digest(_plan(...)[2]); its
    construction uses tokenization only, no new model observation. File maps
    use stable logical member names and exactly {bytes, sha256} values.
    """
    value = _json(content)
    _keys(value, ("model", "tokenizer", "datasets", "sources", "runtime", "generation", "references"), "CONTENT")
    model = value["model"]
    _keys(model, ("model_id", "revision", "config_sha256", "weights"), "MODEL")
    _require(type(model["model_id"]) is str and bool(model["model_id"])
             and type(model["revision"]) is str and bool(re.fullmatch(r"[a-f0-9]{40}", model["revision"])), "MODEL.REVISION")
    _sha(model["config_sha256"], "MODEL.CONFIG_SHA")
    _content_files(model["weights"], "MODEL.WEIGHTS")
    tokenizer = value["tokenizer"]
    _keys(tokenizer, ("files", "settings"), "TOKENIZER")
    _content_files(tokenizer["files"], "TOKENIZER.FILES")
    settings = tokenizer["settings"]
    _keys(settings, ("prompt_add_special_tokens", "target_add_special_tokens", "padding", "target_prefix", "lookup"), "TOKENIZER.SETTINGS")
    _require(type(settings["prompt_add_special_tokens"]) is bool
             and settings["target_add_special_tokens"] is False
             and all(type(settings[key]) is str and bool(settings[key]) for key in ("padding", "target_prefix", "lookup")),
             "TOKENIZER.SETTINGS_VALUES")
    _keys(value["datasets"], ("cf", "zsre"), "DATASETS")
    for dataset, row in value["datasets"].items():
        _keys(row, ("stream_sha256", "ordered_occurrences_sha256", "ordered_queries_sha256", "requests"), "DATASETS." + dataset)
        for key in ("stream_sha256", "ordered_occurrences_sha256", "ordered_queries_sha256"):
            _sha(row[key], "DATASETS." + dataset + "." + key)
        _require(type(row["requests"]) is int and row["requests"] == 2000, "DATASETS.FIRST2000")
        _require(row["ordered_occurrences_sha256"] == _digest(list(range(1, 2001))), "DATASETS.ORDERED_OCCURRENCES")
    source = value["sources"]
    _keys(source, ("factual", "reduce", "w0_reference", "generation"), "SOURCES")
    _sha(source["factual"], "SOURCES.FACTUAL")
    _sha(source["reduce"], "SOURCES.REDUCE")
    _sha(source["w0_reference"], "SOURCES.W0_REFERENCE")
    _require(type(source["generation"]) is dict and set(GENERATION_SOURCES) <= set(source["generation"]), "SOURCES.GENERATION_CLOSURE")
    for name, sha in source["generation"].items():
        _require(type(name) is str and name.endswith(".py") and not name.startswith("/")
                 and ".." not in Path(name).parts, "SOURCES.GENERATION_LOGICAL_NAME")
        _sha(sha, "SOURCES.GENERATION_SHA")
    runtime = value["runtime"]
    _keys(runtime, ("versions", "numeric"), "RUNTIME")
    _keys(runtime["versions"], VERSIONS, "RUNTIME.VERSIONS")
    _require(all(type(item) is str and bool(item) for item in runtime["versions"].values()), "RUNTIME.VERSION_MISSING")
    _require(_digest(runtime["numeric"]) == _digest(dict(weights_dtype="float32", attention="eager", matmul_tf32=False,
                 cudnn_tf32=False, autocast=False, factual_use_cache=False, generation_use_cache=True)), "RUNTIME.NUMERIC_SETTINGS")
    generation = value["generation"]
    _keys(generation, ("profile", "seed", "case_batching", "sampling_scope", "top_k", "max_total_tokens",
                       "n_gen_per_prompt", "EOS_stop", "decode"), "GENERATION")
    from .generation.native_profile import PROFILE, SOURCE
    _require(_digest(generation) == _digest(dict(profile=PROFILE, seed=20261007, case_batching="ALL_GENERATION_PROMPTS",
                 sampling_scope="ENDPOINT_GLOBAL_BATCH_STREAM", top_k=5, max_total_tokens=100,
                 n_gen_per_prompt=1, EOS_stop=False, decode=SOURCE["decode"])), "GENERATION.PROFILE_SETTINGS")
    references = value["references"]
    _keys(references, ("identity_sha256", "files", "nltk", "vectorizer"), "REFERENCES")
    _sha(references["identity_sha256"], "REFERENCES.IDENTITY_SHA")
    _require(set(references["files"]) == set(REFERENCE_FILES), "REFERENCES.FILES")
    _content_files(references["files"], "REFERENCES.FILES")
    _keys(references["nltk"], ("resources", "sources"), "REFERENCES.NLTK")
    _content_files(references["nltk"]["resources"], "REFERENCES.NLTK_RESOURCES")
    _content_files(references["nltk"]["sources"], "REFERENCES.NLTK_SOURCES")
    _require(type(references["vectorizer"]) is str and bool(references["vectorizer"]), "REFERENCES.VECTORIZER")
    return dict(schema=FINGERPRINT_SCHEMA, content=value, sha256=_digest(value))


def _fingerprint(value):
    _keys(value, ("schema", "content", "sha256"), "FINGERPRINT")
    _require(value == computational_fingerprint(value["content"]), "FINGERPRINT.DIGEST_OR_SCHEMA")
    return deepcopy(value)


def _compatible(producer, consumer):
    left, right = _fingerprint(producer), _fingerprint(consumer)
    if left != right:
        fields = [key for key in left["content"] if left["content"][key] != right["content"][key]]
        raise ReferenceInputError("COMPUTATIONAL_FIELDS_MISMATCH:" + ",".join(fields))


def _execution(value, label):
    result = _json(value)
    _require(type(result) is dict and bool(result), label + ".EXECUTION_IDENTITY_MISSING")
    return result


@_typed
def member(path):
    path = Path(path).absolute()
    if not path.is_file():
        raise ReferenceInputError(str(path), missing=True)
    _require(not path.is_symlink(), "REFERENCE_MEMBER_SYMLINK")
    before = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    after = path.stat()
    _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "REFERENCE_CHANGED_DURING_HASH")
    return dict(path=str(path), bytes=after.st_size, sha256=sha.hexdigest())


def _member(row, overrides=None, name=None):
    _keys(row, ("path", "bytes", "sha256"), "REFERENCE_MEMBER")
    _sha(row["sha256"], "REFERENCE_MEMBER.SHA")
    _require(type(row["bytes"]) is int and row["bytes"] >= 0, "REFERENCE_MEMBER.BYTES")
    override = (overrides or {}).get(name, (overrides or {}).get(row["path"], row["path"]))
    actual = member(override)
    _require(actual["bytes"] == row["bytes"] and actual["sha256"] == row["sha256"], "REFERENCE_MEMBER_BYTES_SHA_CHANGED")
    return Path(actual["path"]), dict(producer_member=deepcopy(row), consumed_member=actual,
                                      path_differs=actual["path"] != row["path"])


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError as error:
        raise ReferenceInputError(str(path), missing=True) from error
    except (OSError, ValueError) as error:
        raise ReferenceInputError("REFERENCE_JSON:" + str(path)) from error


def _read_verified(row, overrides=None, name=None):
    path, binding = _member(row, overrides, name)
    value = _read(path)
    _require(member(path) == binding["consumed_member"], "REFERENCE_CHANGED_DURING_READ")
    return value, binding


def _tokenizer_sha(fingerprint):
    return _digest({name: row["sha256"] for name, row in fingerprint["content"]["tokenizer"]["files"].items()})


def _factual_external(external, dataset, fingerprint, execution=None):
    content = fingerprint["content"]
    _require(type(external) is dict and external.get("model_revision") == content["model"]["revision"]
             and external.get("tokenizer_sha256") == _tokenizer_sha(fingerprint)
             and external.get("stream_sha256") == content["datasets"][dataset]["stream_sha256"]
             and external.get("precision") == "FP32_EAGER_TF32_OFF_NO_AUTOCAST"
             and external.get("runtime") == {name: content["runtime"]["versions"][name]
                                             for name in ("torch", "transformers", "numpy")},
             dataset + ".ACTUAL_MODEL_TOKEN_STREAM_RUNTIME_MISMATCH")
    if execution is not None:
        _require(external.get("code_commit") == execution["source"]["main_commit"]
                 and external.get("official_tree") == execution["source"]["official_tree"],
                 dataset + ".ACTUAL_PRODUCER_EXECUTION_SOURCE_MISMATCH")


def _signatures(cases, dataset):
    """Recover the original token plan from measured rows, not claimed labels."""
    keys = ("kind", "prompt_index", "prompt", "target_kind", "target", "input_token_ids", "target_start", "target_token_ids")
    result = []
    for case in cases:
        queries = []
        for kind in ("rewrite", "paraphrase", "neighborhood"):
            observations = case[kind + "_observations"]
            _require(bool(observations), "FACTUAL.OBSERVATION_GROUP_EMPTY")
            for row in observations:
                candidates = (row["target_new"], row["target_true"]) if dataset == "cf" else (row,)
                for candidate in candidates:
                    _require(candidate["kind"] == kind and candidate["occurrence_index"] == case["occurrence_index"],
                             "FACTUAL.OBSERVATION_OCCURRENCE_KIND")
                    tokens = candidate["target_token_ids"]
                    _require(type(tokens) is list and bool(tokens) and candidate["token_count"] == len(tokens)
                             and len(candidate["predicted_token_ids"]) == len(tokens)
                             and len(candidate["token_correct"]) == len(tokens)
                             and candidate["token_correct_count"] == sum(candidate["token_correct"])
                             and candidate["strict_correct"] is all(candidate["token_correct"]),
                             "FACTUAL.TOKEN_DENOMINATOR")
                    _require(all(type(token) is int and token >= 0 for token in tokens + candidate["input_token_ids"] + candidate["predicted_token_ids"])
                             and candidate["input_token_ids"][candidate["target_start"]:] == tokens
                             and candidate["token_correct"] == [a == b for a,b in zip(candidate["predicted_token_ids"],tokens)]
                             and len(candidate["nll_by_token"]) == len(tokens)
                             and all(type(nll) in (float,int) and math.isfinite(nll) and nll >= 0 for nll in candidate["nll_by_token"]),
                             "FACTUAL.TOKEN_CONTENT_OR_NLL_ROWS")
                    import numpy as np
                    total = np.float32(0)
                    for nll in candidate["nll_by_token"]:
                        total = np.float32(total + np.float32(nll))
                    _require(candidate["mean_nll"] == float(np.float32(total / np.float32(len(tokens)))), "FACTUAL.NLL_MEAN_ROW_BINDING")
                    queries.append({key: candidate[key] for key in keys})
        result.append(dict(case_id=case["case_id"], occurrence_index=case["occurrence_index"], queries=queries))
    return result


def _cold_component(value, dataset, fingerprint, producer_execution=None):
    from .factual import SCHEMA, TOKENIZATION
    endpoint = value["evaluation"] if dataset == "zsre" else value
    identity = endpoint["identity"]
    _require(identity.get("schema") == SCHEMA and identity.get("dataset") == dataset
             and identity.get("tokenization") == TOKENIZATION
             and identity.get("ordered_occurrences") == list(range(1, 2001))
             and identity.get("cohort_sha256") == fingerprint["content"]["datasets"][dataset]["ordered_queries_sha256"]
             and endpoint.get("identity_sha256") == _digest(identity)
             and endpoint.get("model_no_mutation") is True and endpoint.get("RNG_restored") is True,
             dataset + ".FACTUAL_IDENTITY_OR_ORDER")
    _factual_external(identity["external_identity"], dataset, fingerprint, producer_execution)
    _require(identity.get("padding") == "RIGHT_EXPLICIT_ATTENTION_MASK" and identity.get("use_cache") is False,
             dataset + ".ACTUAL_NUMERIC_SETTINGS_MISMATCH")
    _require(len(endpoint.get("cases", [])) == 2000 and endpoint.get("summary", {}).get("requests") == 2000
             and [row.get("occurrence_index") for row in endpoint["cases"]] == list(range(1, 2001)), dataset + ".FACTUAL_INCOMPLETE")
    _require(_digest(_signatures(endpoint["cases"], dataset)) == identity["cohort_sha256"], dataset + ".ACTUAL_TOKEN_QUERIES_CHANGED")
    from .reduce import counterfact, zsre
    _require(endpoint["summary"] == (counterfact(endpoint["cases"]) if dataset == "cf" else zsre(endpoint["cases"])),
             dataset + ".INDEPENDENT_REDUCER_COUNTS_VALUES")
    from .factual import _accuracy
    for kind in ("rewrite", "paraphrase", "neighborhood"):
        desired = []
        for case in endpoint["cases"]:
            observations = case[kind + "_observations"]
            if dataset == "cf":
                _require(case[kind + "_prompts_probs"] == [dict(target_new=row["target_new"]["mean_nll"],
                    target_true=row["target_true"]["mean_nll"]) for row in observations], "CF.NLL_PROBABILITY_OBSERVATION_BINDING")
                rows = [row["target_true" if kind == "neighborhood" else "target_new"] for row in observations]
                expected = [row["strict_correct"] for row in rows]
            else:
                rows = observations
                expected = [correct for row in rows for correct in row["token_correct"]]
            _require(case[kind + "_prompts_correct"] == expected, "FACTUAL.CORRECT_OBSERVATION_BINDING")
            desired.extend(rows)
        _require(endpoint["accuracy"][kind] == _accuracy(desired), "FACTUAL.ACCURACY_TOKEN_DENOMINATOR_BINDING")
    if dataset == "zsre":
        _zsre_source(value, identity["external_identity"])
        for source, measured in zip(value["cases"], endpoint["cases"]):
            _require(source["case_id"] == measured["case_id"]
                     and source["predictions"] == [row["predicted_token_ids"] for row in measured["neighborhood_observations"]]
                     and measured["neighborhood_W0_agreement"] == [True for group in source["predictions"] for _ in group],
                     "ZSRE.PREDICTIONS_ACTUAL_MEASUREMENT_BINDING")
    return endpoint


def _zsre_source(reference, producer_external_identity):
    from .factual import W0_SCHEMA, TOKENIZATION, _reference_lookup
    _require(type(reference) is dict and reference.get("schema") == W0_SCHEMA, "ZSRE.SCHEMA")
    identity = reference.get("identity", {})
    _require(reference.get("identity_sha256") == _digest(identity)
             and reference.get("payload_sha256") == _digest({key: value for key, value in reference.items() if key != "payload_sha256"}),
             "ZSRE.ORIGINAL_PAYLOAD_HASH")
    _require(identity.get("external_identity") == producer_external_identity
             and identity.get("tokenization") == TOKENIZATION and identity.get("state") == "W0_COLD_BASE_MODEL"
             and identity.get("ordered_occurrences") == list(range(1, 2001)), "ZSRE.ORIGINAL_PRODUCER_COLD_IDENTITY")
    rows = reference.get("cases", [])
    _require(len(rows) == 2000 and [row.get("occurrence_index") for row in rows] == list(range(1, 2001)), "ZSRE.SOURCE_ROWS_INCOMPLETE")
    # Reuse the original strict source-row/prediction validator, without a forward.
    try:
        _reference_lookup(reference, [dict(case_id=row["case_id"], occurrence_index=row["occurrence_index"],
                                         queries=row["queries"]) for row in rows])
    except (ValueError, KeyError, TypeError) as error:
        raise ReferenceInputError("ZSRE.SOURCE_ROW_VALIDATION:" + str(error)) from error


@dataclass(frozen=True)
class PortableZSREReference:
    """Explicit reviewed view; original ``reference`` is never rewritten."""
    reference: dict
    binding: dict


@_typed
def bind_zsre_reference(reference, *, producer_external_identity, consumer_external_identity,
                        producer_fingerprint, consumer_fingerprint,
                        producer_execution_identity, consumer_execution_identity):
    """Bind source W0 predictions to a distinct actual consumer execution."""
    _compatible(producer_fingerprint, consumer_fingerprint)
    producer_external = _execution(producer_external_identity, "PRODUCER_EXTERNAL")
    consumer_external = _execution(consumer_external_identity, "CONSUMER_EXTERNAL")
    _zsre_source(reference, producer_external)
    _factual_external(producer_external, "zsre", producer_fingerprint, producer_execution_identity)
    _factual_external(consumer_external, "zsre", consumer_fingerprint, consumer_execution_identity)
    _cold_component(reference, "zsre", producer_fingerprint, producer_execution_identity)
    _require(reference["identity"]["cohort_sha256"] == producer_fingerprint["content"]["datasets"]["zsre"]["ordered_queries_sha256"],
             "ZSRE.FINGERPRINT_QUERY_IDENTITY")
    binding = dict(schema=ZSRE_BINDING_SCHEMA, producer_external_identity=producer_external,
                   consumer_external_identity=consumer_external,
                   producer_execution_identity=_execution(producer_execution_identity, "PRODUCER"),
                   consumer_execution_identity=_execution(consumer_execution_identity, "CONSUMER"),
                   producer_fingerprint=_fingerprint(producer_fingerprint), consumer_fingerprint=_fingerprint(consumer_fingerprint),
                   original_reference_identity_sha256=reference["identity_sha256"],
                   original_reference_payload_sha256=reference["payload_sha256"],
                   borrowed_reference=True, producer_raw_relabelled=False,
                   cross_hardware_bitwise_equivalence_claimed=False)
    binding["binding_sha256"] = _digest(binding)
    return PortableZSREReference(deepcopy(reference), binding)


@_typed
def validate_zsre_reference(view, *, consumer_external_identity, signatures):
    """Factual integration seam: validate view BEFORE any consumer model forward.

    The plain-reference exact-external-identity guard stays unchanged. For an
    explicit PortableZSREReference only, factual.evaluate calls this function,
    uses returned original source for _assemble_zsre, and records ``binding``
    alongside its ACTUAL consumer external_identity.
    """
    _require(isinstance(view, PortableZSREReference), "ZSRE.EXPLICIT_PORTABLE_VIEW_REQUIRED")
    binding = view.binding
    _require(binding.get("binding_sha256") == _digest({key: value for key, value in binding.items() if key != "binding_sha256"})
             and binding.get("schema") == ZSRE_BINDING_SCHEMA, "ZSRE.BINDING_HASH")
    _compatible(binding["producer_fingerprint"], binding["consumer_fingerprint"])
    _require(binding["consumer_external_identity"] == _json(consumer_external_identity), "ZSRE.ACTUAL_CONSUMER_IDENTITY_CHANGED")
    _factual_external(binding["consumer_external_identity"], "zsre", binding["consumer_fingerprint"], binding["consumer_execution_identity"])
    _zsre_source(view.reference, binding["producer_external_identity"])
    _require(view.reference["identity_sha256"] == binding["original_reference_identity_sha256"]
             and view.reference["payload_sha256"] == binding["original_reference_payload_sha256"], "ZSRE.ORIGINAL_REFERENCE_CHANGED")
    _cold_component(view.reference, "zsre", binding["producer_fingerprint"], binding["producer_execution_identity"])
    from .factual import _reference_lookup
    try:
        _reference_lookup(view.reference, signatures)
    except (ValueError, KeyError, TypeError) as error:
        raise ReferenceInputError("ZSRE.CONSUMER_TOKEN_PREFIX_IDENTITY:" + str(error)) from error
    return deepcopy(view.reference), deepcopy(binding)


def _component_validation(value, fingerprint):
    _keys(value, COMPONENTS, "COMPONENT_VALIDATION")
    for name, row in value.items():
        dataset = "zsre" if name == "zsre_reference" else "cf"
        _require(type(row) is dict and row.get("actual_complete") is True
                 and type(row.get("actual_model_edits")) is int and row["actual_model_edits"] == 0
                 and row.get("state") == "W0_COLD_BASE_MODEL" and row.get("requests") == 2000
                 and row.get("ordered_queries_sha256") == fingerprint["content"]["datasets"][dataset]["ordered_queries_sha256"],
                 name + ".ACTUAL_COLD_COMPLETION_REQUIRED")
        qualification = row.get("native_qualification_state")
        _require(type(qualification) is dict and qualification.get("status") in ("PASS", "VALIDATED")
                 and type(qualification.get("evidence_sha256")) is str
                 and type(qualification.get("evidence_members")) is dict
                 and set(qualification["evidence_members"]) == {"FT", "MEMIT", "MEMIT_FE"}, name + ".NATIVE_QUALIFICATION_NOT_AVAILABLE")
        _sha(qualification["evidence_sha256"], name + ".QUALIFICATION_EVIDENCE_SHA")
        _require(qualification["evidence_sha256"] == _digest(qualification["evidence_members"]), name + ".QUALIFICATION_EVIDENCE_HASH")


def _sources(value, fingerprint, *, overrides=None):
    source = fingerprint["content"]["sources"]
    expected = {"factual.py":source["factual"], "reduce.py":source["reduce"], "w0_reference.py":source["w0_reference"]}
    expected.update({"generation/" + name: checksum for name, checksum in source["generation"].items()})
    _keys(value, expected, "ACTUAL_SOURCE_MEMBERS")
    consumed = {}
    for name, checksum in expected.items():
        _require(value[name]["sha256"] == checksum, "ACTUAL_CONSUMED_SOURCE_SHA_NOT_FINGERPRINT:" + name)
        _, consumed["source." + name] = _member(value[name], overrides, "source." + name)
    return consumed


def _datasets(value, fingerprint, *, overrides=None):
    _keys(value, ("cf", "zsre"), "ACTUAL_DATASET_MEMBERS")
    records, consumed = {}, {}
    for name, row in value.items():
        _require(row["sha256"] == fingerprint["content"]["datasets"][name]["stream_sha256"], "ACTUAL_DATASET_SOURCE_SHA_NOT_FINGERPRINT")
        records[name], consumed["dataset." + name] = _read_verified(row, overrides, "dataset." + name)
        _require(type(records[name]) is list and len(records[name]) == 2000
                 and [record["occurrence_index"] for record in records[name]] == list(range(1,2001)), "ACTUAL_DATASET_ORDERED_FIRST2000")
    return records, consumed


def _case_order(endpoint, records, dataset):
    cases = endpoint["evaluation"]["cases"] if dataset == "zsre" else endpoint["cases"]
    _require([(row["case_id"],row["occurrence_index"]) for row in cases] ==
             [(row["case_id"],row["occurrence_index"]) for row in records], "FACTUAL.ACTUAL_DATASET_CASE_ORDER")


@_typed
def _native_parity_proof(report, measured, state):
    """Replay the frozen first300 receipt contract, never a model observation.

    This validates stored original-source rows/tolerances/work and canonical
    reducers. It is not a new GPU test or a full2K parity certification.
    """
    source_sha = "25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145"
    source_commit = "b84624f44dfe8fc6cd9e41df916c44124a0c46dc"
    occurrences = list(range(1,301))
    scope = "ACTUAL_MATCHED_SUBSET"
    plan = dict(schema="official-server1-native-CF-parity-plan-v1", completed_batch=3, stage="continuous_B3",
        first_occurrences=occurrences, requests=300, evidence_scope=scope, native_reference_upstream_sha256=source_sha,
        native_reference_upstream_commit=source_commit, nll_abs_nats=1e-4, nll_relative=1e-5, aggregate_abs=1e-10,
        strict_bool="EXACT", canonical_batch_size=16, canonical_observation="EXISTING_B3_FIRST300",
        reference_observation="INDEPENDENT_ORIGINAL_FORWARD_SAME300", additional_canonical_forward_calls=0,
        additional_fit_calls=0,generation_calls=0,quality_promotion=False,tolerance_override=False,full_2k_parity="NOT_OBSERVED")
    _require(report.get("schema") == "official-server1-native-CF-parity-report-v1" and report.get("plan_sha256") == _digest(plan)
             and all(report.get(key) is True for key in ("raw_local_only", "RNG_restored", "input_records_unchanged", "canonical_payload_unchanged"))
             and report.get("performance_gate") is False
             and all(type(report.get(key)) is int and report[key] == 0 for key in ("additional_fit_calls", "generation_calls", "additional_canonical_forward_calls"))
             and report.get("canonical_observation") == plan["canonical_observation"], "QUALIFICATION.NATIVE_REPORT_PLAN_GUARDS")
    canonical, comparison, binding = report["canonical"], report["comparison"], report["reference_binding"]
    external = canonical["identity"]["external_identity"]
    _require(type(external) is dict and set(external) == {"model", "model_revision", "tokenizer_sha256", "assets_sha256",
                  "stream_sha256", "code_commit", "official_tree", "runtime", "precision", "raw_local_only"}
             and external["model"] == "llama3" and external["raw_local_only"] is True
             and all(type(external[key]) is str and re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}",external[key])
                     for key in ("model_revision", "tokenizer_sha256", "assets_sha256", "stream_sha256", "code_commit", "official_tree"))
             and report.get("external_identity") == external
             and {key:value for key,value in canonical.items() if key != "work"} == {key:value for key,value in measured.items() if key != "work"}
             and report.get("canonical_payload_sha256") == _digest(canonical)
             and report.get("state_before") == report.get("state_after") == state == binding.get("state_identity")
             and state.get("successful_calls") == 3 and state.get("method") in ("FT","MEMIT","MEMIT_FE")
             and binding.get("model_identity", {}).get("revision") == external["model_revision"]
             and binding.get("tokenizer_identity") == dict(sha256=external["tokenizer_sha256"]), "QUALIFICATION.NATIVE_REPORT_STATE_IDENTITY")
    _require(comparison.get("schema") == "official-cf-original-native-reference-v1"
             and comparison.get("status") == "PASS" and comparison.get("evidence_scope") == scope
             and comparison.get("evidence") == {"TEST_ONLY_CPU_FIXTURE":"NOT_OBSERVED", "ACTUAL_GPU_SMOKE":"NOT_OBSERVED",
                 scope:"PASS", "ACTUAL_FULL_2K":"NOT_OBSERVED"}
             and comparison.get("mismatches") == [] and comparison.get("display_mismatches") == []
             and comparison.get("raw_local_only") is True and comparison.get("checkpoint_saved") is False
             and comparison.get("scientific_performance_promotion") is False
             and comparison.get("tolerances") == dict(nll_abs_nats=1e-4,nll_relative_to_native=1e-5,strict_booleans="EXACT",aggregate_abs=1e-10)
             and comparison.get("canonical_identity_sha256") == canonical["identity_sha256"], "QUALIFICATION.NATIVE_FIXED_TOLERANCE_SCOPE")
    native = comparison["native"]
    ci, ni = canonical["identity"], native["identity"]
    _require(ci.get("schema") == "official-factual-causal-v1" and ci.get("dataset") == "cf"
             and ci.get("tokenization") == "NATIVE_CF_VERIFIED_BOUNDARY_ZSRE_EXACT_TOKEN_PREFIX_NO_TARGET_BOS"
             and ci.get("padding") == "RIGHT_EXPLICIT_ATTENTION_MASK" and ci.get("use_cache") is False
             and ci.get("ordered_occurrences") == occurrences and canonical.get("identity_sha256") == _digest(ci)
             and all(canonical.get(key) is True for key in ("raw_local_only", "model_no_mutation", "RNG_restored"))
             and native.get("schema") == "official-cf-original-native-reference-v1" and native.get("status") == "OBSERVED_UNCOMPARED"
             and ni.get("schema") == "official-cf-original-native-reference-v1" and ni.get("external_identity") == external
             and ni.get("reference_binding") == binding and ni.get("source_sha256") == source_sha and ni.get("source_commit") == source_commit
             and ni.get("source_bytes") == 7941 and ni.get("original_function") == "test_batch_prediction" and ni.get("evidence_scope") == scope
             and ni.get("ordered_occurrences") == occurrences and ni.get("cohort_sha256") == ci["cohort_sha256"]
             and ni.get("native_device") == "cuda" and ni.get("padding") == "NATIVE_RIGHT" and ni.get("model_dtype") == "FP32"
             and ni.get("use_cache") is False and native.get("identity_sha256") == _digest(ni)
             and native.get("canonical_payload_sha256") == report["canonical_payload_sha256"]
             and all(native.get(key) is True for key in ("raw_local_only", "model_no_mutation", "RNG_restored"))
             and native.get("checkpoint_saved") is False, "QUALIFICATION.NATIVE_ORIGINAL_SOURCE_AND_WORK_SCOPE")
    left, right = canonical["cases"], native["cases"]
    _require(type(left) is list and type(right) is list and len(left) == len(right) == 300
             and _digest(native["case_signatures"]) == ci["cohort_sha256"]
             and ni.get("locked_cohort_sha256") == _digest([dict(case_id=row["case_id"],occurrence_index=row["occurrence_index"]) for row in left]),
             "QUALIFICATION.NATIVE_EXACT_FIRST300_TOKEN_COHORT")
    candidates = 0
    for ordinal, (a,b) in enumerate(zip(left,right),1):
        _require(a["case_id"] == b["case_id"] and a["occurrence_index"] == b["occurrence_index"] == ordinal, "QUALIFICATION.NATIVE_CASE_ORDER")
        for kind in ("rewrite", "paraphrase", "neighborhood"):
            ar, br = a[kind+"_prompts_probs"], b[kind+"_prompts_probs"]
            _require(type(ar) is list and type(br) is list and len(ar) == len(br) > 0
                     and type(a[kind+"_prompts_correct"]) is list and type(b[kind+"_prompts_correct"]) is list
                     and a[kind+"_prompts_correct"] == b[kind+"_prompts_correct"]
                     and len(a[kind+"_prompts_correct"]) == len(ar)
                     and len(b[kind+"_prompts_correct"]) == len(br)
                     and all(type(bit) is bool for bits in (a[kind+"_prompts_correct"],b[kind+"_prompts_correct"]) for bit in bits),
                     "QUALIFICATION.NATIVE_EXACT_DESIRED_ARGMAX")
            for x,y in zip(ar,br):
                for target in ("target_new","target_true"):
                    _require(type(x[target]) in (float,int) and type(y[target]) in (float,int)
                             and math.isfinite(x[target]) and math.isfinite(y[target]) and x[target] >= 0 and y[target] >= 0
                             and abs(x[target]-y[target]) <= 1e-4+1e-5*abs(y[target]), "QUALIFICATION.NATIVE_NLL_MISMATCH")
                success = lambda row: row["target_true"] < row["target_new"] if kind == "neighborhood" else row["target_new"] < row["target_true"]
                _require(success(x) == success(y), "QUALIFICATION.NATIVE_STRICT_PREFERENCE_MISMATCH")
            candidates += 2*len(ar)
    from .reduce import counterfact
    for endpoint, actual_native in ((canonical,False),(native,True)):
        reduced = counterfact(endpoint["cases"])
        _require(all(type(endpoint["summary"].get(key)) in (float,int) and math.isfinite(endpoint["summary"][key])
                     and abs(endpoint["summary"][key]-value) <= 1e-10 for key,value in reduced.items()), "QUALIFICATION.NATIVE_CANONICAL_REDUCTION")
        work = endpoint["work"]
        _require(work.get("candidate_sequences") == candidates and type(work.get("forward_calls")) is int
                 and work["forward_calls"] == (300 if actual_native else math.ceil(candidates/16))
                 and all(type(work.get(key)) is int and work[key] > 0 for key in ("physical_input_tokens", "padded_input_tokens", "target_tokens"))
                 and work["padded_input_tokens"] >= work["physical_input_tokens"]
                 and type(work.get("seconds")) in (float,int) and math.isfinite(work["seconds"]) and work["seconds"] >= 0,
                 "QUALIFICATION.NATIVE_MEASURED_FORWARD_TOKEN_WORK")
    _require(comparison.get("work") == native["work"]
             and all(native["work"][key] == canonical["work"][key] for key in ("candidate_sequences","physical_input_tokens","target_tokens")),
             "QUALIFICATION.NATIVE_INDEPENDENT_SAME_INPUT_WORK")
    return dict(schema="official-server1-native-CF-parity-CPU-validation-v1",status="PASS",evidence_scope=scope,
        requests=300,completed_batch=3,plan_sha256=_digest(plan),upstream_source_sha256=source_sha,original_native_forward_calls=300,
        existing_canonical_forward_calls=canonical["work"]["forward_calls"],additional_canonical_forward_calls=0,
        canonical_observation=plan["canonical_observation"],candidate_sequences=candidates,full_2k_parity="NOT_OBSERVED",
        performance_gate=False,additional_fit_calls=0,generation_calls=0,raw_local_only=True,case_ids_tokens_prompts_omitted=True)


def _qualification(validation, fingerprint, producer_execution, nested, *, overrides=None, collect=False):
    """Verify source proof bytes; does NOT claim independent GPU certification."""
    evidence = validation["cf_factual"]["native_qualification_state"]["evidence_members"]
    _require(all(row["native_qualification_state"]["evidence_members"] == evidence for row in validation.values()),
             "QUALIFICATION.COMPONENT_PROOF_MISMATCH")
    consumed = {}
    expected_source = producer_execution["source"]
    for method, row in evidence.items():
        name = "qualification." + method
        original = row if collect else nested.get(name)
        _require(original == row, "QUALIFICATION.ORIGINAL_PROOF_MEMBER_CHANGED")
        value, receipt = _read_verified(original, overrides, name)
        identity = value.get("identity", {})
        _require(value.get("schema") == "official-server1-native-resume-READY-v1"
                 and value.get("method") == method and value.get("passed") is True
                 and value.get("actual_native_B3_and_B2_resume") is True
                 and type(value.get("actual_fit_calls")) is int and value["actual_fit_calls"] == 6
                 and type(value.get("generation_calls")) is int and value["generation_calls"] == 0
                 and identity.get("model_revision") == fingerprint["content"]["model"]["revision"]
                 and identity.get("tokenizer_sha256") == _tokenizer_sha(fingerprint)
                 and identity.get("code_commit") == expected_source["main_commit"]
                 and identity.get("official_tree_sha256") == expected_source["official_tree"],
                 "QUALIFICATION.ACTUAL_NATIVE_READY_IDENTITY")
        if collect:
            nested[name] = deepcopy(original)
        consumed[name] = receipt
        stages, proofs = {}, {}
        for key, stage_name, batches, calls in (("continuous", "continuous", 3, 3),
                ("stopped", "stop", 2, 2), ("resumed", "resume", 3, 1)):
            stage_key = name + "." + key
            original_stage = value[key] if collect else nested.get(stage_key)
            _require(original_stage == value[key], "QUALIFICATION.STAGE_MEMBER_CHANGED")
            stage, stage_receipt = _read_verified(original_stage, overrides, stage_key)
            _require(stage.get("schema") == "official-server1-native-qualification-stage-v1"
                     and stage.get("stage") == stage_name and stage.get("completed_batch") == batches
                     and stage.get("actual_native_fit_calls") == calls
                     and stage.get("GPU_actual") is True and stage.get("actual_model_loaded") is True
                     and stage.get("identity") == identity and stage.get("method") == method,
                     "QUALIFICATION.ACTUAL_STAGE_IDENTITY")
            if collect:
                nested[stage_key] = deepcopy(original_stage)
            consumed[stage_key] = stage_receipt
            stages[key] = stage
            if key in ("continuous", "resumed"):
                _require(stage.get("factual_member") is not None, "QUALIFICATION.ACTUAL_B3_FACTUAL_PROOF_REQUIRED")
            if key == "continuous":
                _require(stage.get("native_parity_member") is not None, "QUALIFICATION.ACTUAL_NATIVE_MATCHED_PROOF_REQUIRED")
            # Preserve/verify raw measured B3 and independent native parity proofs.
            for proof in ("factual_member", "native_parity_member"):
                if stage.get(proof) is not None:
                    proof_name = stage_key + "." + proof
                    original_proof = stage[proof] if collect else nested.get(proof_name)
                    _require(original_proof == stage[proof], "QUALIFICATION.MEASUREMENT_MEMBER_CHANGED")
                    proof_value, proof_receipt = _read_verified(original_proof, overrides, proof_name)
                    if collect:
                        nested[proof_name] = deepcopy(original_proof)
                    consumed[proof_name] = proof_receipt
                    proofs[(key, proof)] = proof_value
        left, right = stages["continuous"], stages["resumed"]
        _require(all(left[key] == right[key] for key in ("identity", "method", "completed_batch", "selected_state",
                    "contexts_sha256", "RNG_sha256", "checkpoint_RNG_sha256")), "QUALIFICATION.ACTUAL_RESUME_STATE_RNG_MISMATCH")
        left_fact, right_fact = proofs[("continuous", "factual_member")], proofs[("resumed", "factual_member")]
        _require(all(left_fact[key] == right_fact[key] for key in ("identity", "identity_sha256", "summary", "accuracy", "cases")),
                 "QUALIFICATION.ACTUAL_RESUME_FACTUAL_MISMATCH")
        _require(value.get("checks") == dict(selected_weights="EXACT_SHA256", contexts="EXACT", RNG="EXACT",
                    factual="EXACT_RAW_VALUES_AND_TOKEN_IDENTITY", no_tolerance_relaxation=True), "QUALIFICATION.EXACT_CHECK_CONTRACT")
        parity = proofs[("continuous", "native_parity_member")]
        comparison = parity.get("comparison", {})
        _require(value.get("native_parity") == _native_parity_proof(parity,left_fact,left["selected_state"]),
                 "QUALIFICATION.BOUND_NATIVE_MATCHED_RECEIPT")
    return consumed


def _generation_runtime(value, fingerprint, identity, producer_execution=None):
    from .generation.common import digest as generation_digest
    from .generation.native_profile import ROUTE, SOURCE
    runtime = value["identity"]
    content = fingerprint["content"]
    model = runtime.get("model_identity", {})
    settings = content["generation"]
    _require(value.get("identity_sha256") == generation_digest(runtime) == identity.get("runtime")
             and runtime.get("model_identity", {}).get("revision") == content["model"]["revision"]
             and model.get("tokenizer_sha256") == _tokenizer_sha(fingerprint)
             and runtime.get("reference_assets_sha256") == content["references"]["identity_sha256"]
             and runtime.get("route") == ROUTE and runtime.get("original_generator") == SOURCE
             and runtime.get("eval_seed") == settings["seed"]
             and all(runtime.get(key) == settings[key] for key in ("profile", "case_batching", "sampling_scope",
                 "n_gen_per_prompt", "top_k", "max_total_tokens", "EOS_stop")),
             "GENERATION.ACTUAL_MODEL_REFERENCE_PROFILE_RUNTIME_MISMATCH")
    if producer_execution is not None:
        _require(runtime.get("generation_source_sha") == dict(code_commit=producer_execution["source"]["main_commit"],
                  official_tree=producer_execution["source"]["official_tree"]), "GENERATION.ACTUAL_PRODUCER_EXECUTION_SOURCE_MISMATCH")
    return runtime


def _generation(value, fingerprint, nested_members, runtime_member, cf_records, *, overrides=None, collect=False, producer_execution=None):
    from .generation.common import digest as generation_digest
    from .generation.native_observer import verify_native_raw
    from .generation.metrics import reduce_cases
    from .generation.observer import _record_identity
    identity = value.get("identity", {})
    _require(value.get("identity_sha256") == generation_digest(identity) and identity.get("endpoint") == "W0"
             and identity.get("ordered_occurrences") == list(range(1, 2001))
             and value.get("RNG_restored") is True and value.get("observer_no_mutation") is True
             and value.get("raw_local_only") is True and len(value.get("rows", [])) == 2000
             and [row.get("occurrence") for row in value["rows"]] == list(range(1, 2001)), "GENERATION.INCOMPLETE_OR_NOT_W0")
    runtime_value, runtime_receipt = _read_verified(runtime_member, overrides, "cf_generation.observer_identity")
    runtime = _generation_runtime(runtime_value, fingerprint, identity, producer_execution)
    consumed = {"cf_generation.observer_identity": runtime_receipt}
    record_identities, observation_identities = [], []
    work = dict(physical_forward_calls=0, prefill_query_tokens=0, decode_query_tokens=0)
    for row in value["rows"]:
        name = "cf_generation.raw." + str(row["occurrence"])
        original = row.get("provenance", {}).get("raw_member") if collect else nested_members.get(name)
        _require(original is not None, "GENERATION.RAW_MEMBER_MISSING")
        _require(original == row["provenance"]["raw_member"] and original["path"] == row["observation_path"]
                 and row["provenance"]["runtime_sha256"] == identity["runtime"]
                 and row["provenance"]["generation_source_sha"] == runtime["generation_source_sha"]
                 and row["provenance"]["route"] == runtime["route"], "GENERATION.ORIGINAL_SOURCE_BYTE_PROVENANCE")
        raw, binding = _read_verified(original, overrides, name)
        try:
            verify_native_raw(raw, expected_runtime=identity["runtime"], expected_stream=identity["sampling_stream_sha256"])
        except (ValueError, RuntimeError, KeyError, TypeError) as error:
            raise ReferenceInputError("GENERATION.ORIGINAL_RAW_VALIDATION:" + str(error)) from error
        _require(raw["identity_sha256"] == row["identity_sha256"] and raw["payload_sha256"] == row["payload_sha256"]
                 and raw["occurrence"] == row["occurrence"] and raw["case_id"] == row["case_id"]
                 and raw["metrics"] == row["metrics"], "GENERATION.ORIGINAL_ROW_BINDING")
        state = raw["identity"]["state_identity"]
        _require(type(state.get("actual_model_edits")) is int and state["actual_model_edits"] == 0
                 and state.get("base_model", {}).get("revision") == fingerprint["content"]["model"]["revision"]
                 and state.get("base_model", {}).get("model_id") == fingerprint["content"]["model"]["model_id"]
                 and generation_digest(state) == identity["state_sha256"], "GENERATION.ACTUAL_COLD_MODEL_STATE")
        record_identities.append(raw["identity"]["record_identity"])
        observation_identities.append(raw["identity_sha256"])
        for key in work:
            work[key] += sum(observation[key] for observation in raw["observations"])
        if collect:
            nested_members[name] = original
        consumed[name] = binding
    _require(identity["observation_identities"] == observation_identities
             and record_identities == [_record_identity(record, index) for index,record in enumerate(cf_records,1)]
             and identity["sampling_stream_sha256"] == generation_digest(dict(runtime=identity["runtime"],
                 eval_seed=fingerprint["content"]["generation"]["seed"], ordered_record_identities=record_identities))
             and value["summary"] == reduce_cases(value["rows"]), "GENERATION.ORDER_STREAM_OR_REDUCER_CHANGED")
    execution_member = value.get("native_execution_member")
    _require(execution_member is not None, "GENERATION.NATIVE_EXECUTION_MEMBER_REQUIRED")
    name = "cf_generation.native_execution"
    original = execution_member if collect else nested_members.get(name)
    _require(original == execution_member, "GENERATION.NATIVE_EXECUTION_MEMBER_CHANGED")
    execution, binding = _read_verified(original, overrides, name)
    _require(execution.get("identity") == identity and execution.get("identity_sha256") == value["identity_sha256"]
             and execution.get("native_execution_complete") is True and execution.get("RNG_restored") is True
             and execution.get("observer_no_mutation") is True
             and execution.get("profile") == runtime["profile"] and execution.get("route") == runtime["route"]
             and execution.get("qualification_performed") is False and execution.get("no_fallback") is True
             and all(execution.get(key) == count for key, count in work.items()), "GENERATION.EXECUTION_INCOMPLETE")
    if collect:
        nested_members[name] = deepcopy(original)
    consumed[name] = binding
    return consumed


@_typed
def make_ready(*, producer_execution_identity, fingerprint, members, component_validation, generation_runtime_member, source_members, dataset_members):
    """Return metadata for an atomic immutable READY; never write or infer READY.

    Caller writes this via its create-once atomic writer only AFTER actual raw
    observations/qualification. Keep the classic runner READY separately.
    """
    fingerprint = _fingerprint(fingerprint)
    _sources(source_members, fingerprint)
    records, _ = _datasets(dataset_members, fingerprint)
    _keys(members, COMPONENTS, "READY_COMPONENT_MEMBERS")
    _component_validation(component_validation, fingerprint)
    values, component_members = {}, {}
    for name, row in members.items():
        values[name], _ = _read_verified(row)
        component_members[name] = deepcopy(row)
    execution = _execution(producer_execution_identity, "PRODUCER")
    _cold_component(values["cf_factual"], "cf", fingerprint, execution)
    _cold_component(values["zsre_reference"], "zsre", fingerprint, execution)
    _case_order(values["cf_factual"], records["cf"], "cf")
    _case_order(values["zsre_reference"], records["zsre"], "zsre")
    nested = {}
    _qualification(component_validation, fingerprint, execution, nested, collect=True)
    _generation(values["cf_generation"], fingerprint, nested, generation_runtime_member, records["cf"], collect=True, producer_execution=execution)
    result = dict(schema=READY_SCHEMA, status="READY", actual_complete=True, actual_model_edits=0,
                  producer_execution_identity=execution,
                  computational_fingerprint=fingerprint, members=component_members,
                  generation_raw_members=nested, generation_runtime_member=deepcopy(generation_runtime_member),
                  source_members=deepcopy(source_members),
                  dataset_members=deepcopy(dataset_members),
                  component_validation=_json(component_validation),
                  transfer_performed=False, producer_raw_relabelled=False,
                  cross_hardware_bitwise_equivalence_claimed=False)
    result["ready_sha256"] = _digest(result)
    return result


@dataclass(frozen=True)
class BorrowedW0:
    ready: dict
    consumer_execution_identity: dict
    consumer_fingerprint: dict
    members: dict
    values: dict
    binding: dict

    def zsre(self, *, consumer_external_identity):
        _require(self.binding["binding_sha256"] == _digest({key: value for key, value in self.binding.items() if key != "binding_sha256"})
                 and self.ready["ready_sha256"] == self.binding["original_ready_sha256"]
                 and self.ready["ready_sha256"] == _digest({key: value for key, value in self.ready.items() if key != "ready_sha256"})
                 and self.consumer_execution_identity == self.binding["consumer_execution_identity"], "BORROWED_VIEW_MUTATED")
        source = self.values["zsre_reference"]
        return bind_zsre_reference(source,
            producer_external_identity=source["identity"]["external_identity"],
            consumer_external_identity=consumer_external_identity,
            producer_fingerprint=self.ready["computational_fingerprint"], consumer_fingerprint=self.consumer_fingerprint,
            producer_execution_identity=self.ready["producer_execution_identity"],
            consumer_execution_identity=self.consumer_execution_identity)


@_typed
def read_ready(path, *, consumer_execution_identity, consumer_fingerprint, member_paths=None):
    """Verify all consumed raw bytes; explicit local overrides permit prior transfers.

    No files are transferred/created. ``member_paths`` maps logical member keys
    or original absolute paths to existing files; original READY/raw remains
    unchanged. Missing/partial/mismatched inputs do not expose a borrowed view.
    """
    ready_member = member(path)
    ready, _ = _read_verified(ready_member)
    _require(ready.get("schema") == READY_SCHEMA and ready.get("status") == "READY"
             and ready.get("actual_complete") is True and type(ready.get("actual_model_edits")) is int
             and ready["actual_model_edits"] == 0
             and ready.get("ready_sha256") == _digest({key: value for key, value in ready.items() if key != "ready_sha256"}), "READY.COLD_COMPLETE_IDENTITY")
    _compatible(ready["computational_fingerprint"], consumer_fingerprint)
    _component_validation(ready["component_validation"], ready["computational_fingerprint"])
    source_consumed = _sources(ready["source_members"], ready["computational_fingerprint"], overrides=member_paths)
    records, dataset_consumed = _datasets(ready["dataset_members"], ready["computational_fingerprint"], overrides=member_paths)
    _keys(ready["members"], COMPONENTS, "READY_COMPONENT_MEMBERS")
    values, consumed = {}, {}
    for name, row in ready["members"].items():
        values[name], consumed[name] = _read_verified(row, member_paths, name)
    _cold_component(values["cf_factual"], "cf", consumer_fingerprint, ready["producer_execution_identity"])
    _cold_component(values["zsre_reference"], "zsre", consumer_fingerprint, ready["producer_execution_identity"])
    _case_order(values["cf_factual"], records["cf"], "cf")
    _case_order(values["zsre_reference"], records["zsre"], "zsre")
    consumed.update(_qualification(ready["component_validation"], consumer_fingerprint,
                                  ready["producer_execution_identity"], ready["generation_raw_members"], overrides=member_paths))
    consumed.update(_generation(values["cf_generation"], consumer_fingerprint,
                                ready["generation_raw_members"], ready["generation_runtime_member"], records["cf"], overrides=member_paths,
                                producer_execution=ready["producer_execution_identity"]))
    _require(set(ready["generation_raw_members"]) == set(consumed) - set(COMPONENTS) - {"cf_generation.observer_identity"},
             "READY.UNCONSUMED_OR_MISSING_RAW_PROOF_MEMBERS")
    consumed.update(source_consumed)
    consumed.update(dataset_consumed)
    execution = _execution(consumer_execution_identity, "CONSUMER")
    binding = dict(schema="official-borrowed-W0-binding-v1", original_READY_member=ready_member,
                   original_ready_sha256=ready["ready_sha256"],
                   producer_execution_identity=deepcopy(ready["producer_execution_identity"]),
                   consumer_execution_identity=execution, computational_fingerprint_sha256=consumer_fingerprint["sha256"],
                   execution_identities_equal=ready["producer_execution_identity"] == execution,
                   consumed_members=consumed, producer_raw_relabelled=False, transfer_performed=False,
                   cross_hardware_bitwise_equivalence_claimed=False)
    binding["binding_sha256"] = _digest(binding)
    return BorrowedW0(deepcopy(ready), execution, _fingerprint(consumer_fingerprint), consumed, values, binding)
