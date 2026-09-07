"""Outcome-independent S cohort seal using exact Git-only exclusion inventories.

No tokenizer/model/results/live-server reads. Publication contains IDs, hashes,
and prompt counts only; original sample/prompt bytes remain in the pinned pool.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


NAMESPACE = "ODEEDIT-LLAMA-DIAG-SWEEP-20260907-V1"
DATASET_SHA256 = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
DATASET_BYTES = 45108470
PUBLICATION_REF = "0d0a0131e4a6a2a645dfa6530377d420a084d136"
MAIN_REF = "765ca516d8b836e6672429b68d9e36d7a587cb0b"
REQUIRED_INVENTORIES = frozenset(("pilot38", "completed_seq1000", "live_orb_reserved", "live_l8_reserved"))
SEQ_PATH = "experiment-reports/servers/server2/alpha-native-response-v31-sequential-routing-2026-09-06-v1/main-four-terminal-v1/sample.lock.json"


class SampleBoundary(RuntimeError):
    """Typed missing inventory, source identity, schema, or sample-pool hold."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class GitInventorySpec:
    label: str
    commit: str
    path: str
    sha256: str
    expected_count: int
    expected_root: str
    stream_schema: bool = False


@dataclass(frozen=True, slots=True)
class ExclusionInventory:
    label: str
    case_ids: tuple[int, ...]
    identity: Mapping[str, Any]


def default_inventory_specs() -> tuple[GitInventorySpec, ...]:
    seq = (PUBLICATION_REF, SEQ_PATH,
           "105750c7951164790e0fecced4bf5e08535f94e438bb1fbd3104ad9d77dba5e8", 1000,
           "40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd")
    return (
        GitInventorySpec("pilot38", MAIN_REF,
            "experiment-reports/servers/server1/native-response-v31-b10-warm-pilot-2026-09-06-v1/primary/sample.lock.json",
            "db2a863b4caad2d73e5485d194e4ab0d25f6625fe847d69851ca02dd31d9ba20", 38,
            "f1deb7fee7a7c0e7f8acf3691dd3fbab1ec2981be4746b61d2f9c2e170b70cf1"),
        GitInventorySpec("completed_seq1000", *seq),
        GitInventorySpec("live_orb_reserved", "8610faf0e114059a5e08116f1164baf31c573e80",
            "project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json",
            "550600af120070594078b73a5da7d066ad6287d8a3224c55f61babca592e0586", 1000,
            "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a", True),
        # The pinned SH4 takeover source explicitly consumes this identical seal.
        GitInventorySpec("live_l8_reserved", *seq),
    )


def _git_blob(repo: Path, commit: str, path: str, expected_sha: str) -> bytes:
    try:
        mode = subprocess.check_output(["git", "-C", str(repo), "ls-tree", commit, "--", path], text=True)
        if not mode.startswith("100644 blob "):
            raise SampleBoundary("EXCLUSION_GIT_MEMBER_NOT_REGULAR: " + path)
        data = subprocess.check_output(["git", "-C", str(repo), "show", commit + ":" + path], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise SampleBoundary("EXCLUSION_INVENTORY_UNAVAILABLE: " + path) from exc
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise SampleBoundary("EXCLUSION_GIT_SHA_MISMATCH: " + path)
    return data


def load_git_inventories(repo: Path) -> tuple[ExclusionInventory, ...]:
    inventories = []
    for spec in default_inventory_specs():
        data = _git_blob(repo, spec.commit, spec.path, spec.sha256)
        value = json.loads(data)
        records = value["requests" if spec.stream_schema else "records"]
        if spec.stream_schema:
            body = dict(value)
            stored_root = body.pop("root_digest")
            actual_root = canonical_hash(body)
        else:
            stored_root = value["ordered_root"]
            actual_root = canonical_hash(records)
        ids = tuple(int(row["case_id"]) for row in records)
        if (stored_root != spec.expected_root or actual_root != stored_root
                or len(ids) != spec.expected_count or len(set(ids)) != len(ids)):
            raise SampleBoundary("EXCLUSION_INVENTORY_ROOT_OR_COUNT: " + spec.label)
        identity = dict(commit=spec.commit, path=spec.path, sha256=spec.sha256,
                        bytes=len(data), ordered_root=actual_root, count=len(ids), scope="GIT_SEALED_ONLY")
        if spec.label == "live_l8_reserved":
            source_path = "project/run_scripts/alpha_native_response_ode_v31_sequential/server4_takeover.py"
            source_sha = "f550b1c8790d3f17f40dd451d2c3c3182c527078dbb6e3498251287389691f1c"
            _git_blob(repo, "44602a1a80554c67da0ef9646b43d843104785f2", source_path, source_sha)
            identity["binding_source"] = dict(commit="44602a1a80554c67da0ef9646b43d843104785f2",
                                               path=source_path, sha256=source_sha,
                                               meaning="published_inputs sample.lock -> identical SAMPLE_ROOT")
        elif spec.label == "live_orb_reserved":
            source_path = "project/run_scripts/ordered_response_barrier_ode/sequential_runtime.py"
            source_sha = "58591182167c388fefe83ae84757701612ff307c38aa967120d047863d4060d9"
            _git_blob(repo, spec.commit, source_path, source_sha)
            identity["binding_source"] = dict(commit=spec.commit, path=source_path,
                                               sha256=source_sha,
                                               meaning="sequential runtime imports STREAM_ROOT/ORDER_ROOT and _load_stream")
        inventories.append(ExclusionInventory(spec.label, ids, identity))
    return tuple(inventories)


def _record_schema(record: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
    try:
        case = int(record["case_id"])
        rw = record["requested_rewrite"]
        if isinstance(record["case_id"], bool) or not isinstance(rw, Mapping):
            raise TypeError
        for key in ("prompt", "subject"):
            if not isinstance(rw[key], str) or not rw[key]:
                raise TypeError
        rw["prompt"].format(rw["subject"])
        for key in ("target_new", "target_true"):
            if not isinstance(rw[key]["str"], str) or not rw[key]["str"]:
                raise TypeError
        for key in ("paraphrase_prompts", "neighborhood_prompts"):
            if (not isinstance(record[key], list) or not record[key]
                    or any(not isinstance(p, str) or not p for p in record[key])):
                raise TypeError
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise SampleBoundary("EVALUATOR_INPUT_SCHEMA_BOUNDARY") from exc
    return case, rw


def evaluation_inventory(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Seal actual prompt counts, never filter or duplicate to force 200/1000."""
    rows = []
    prompt_identities = []
    for record in records:
        case, rw = _record_schema(record)
        counts = dict(RS=1, PS=len(record["paraphrase_prompts"]), NS=len(record["neighborhood_prompts"]))
        rows.append(dict(case_id=case, **counts))
        for kind, prompts in (("rewrite", [rw["prompt"].format(rw["subject"])]),
                              ("rephrase", record["paraphrase_prompts"]),
                              ("locality", record["neighborhood_prompts"])):
            for ordinal, prompt in enumerate(prompts):
                prompt_identities.append(dict(case_id=case, kind=kind, prompt_index=ordinal,
                    input_sha256=canonical_hash([prompt, rw["target_new"]["str"], rw["target_true"]["str"]])))
    return dict(per_request=rows, request_count=len(rows),
                denominators={key: sum(row[key] for row in rows) for key in ("RS", "PS", "NS")},
                paired_new_true_rows=True, input_order_sha256=canonical_hash(prompt_identities),
                prompt_count_selection_influence=0, controller_influence_count=0)


def select_cohorts(records: Sequence[Mapping[str, Any]], inventories: Sequence[ExclusionInventory],
                   *, dataset_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Select first 100 + next 300 by prescribed hash, independent of D assets."""
    labels = [row.label for row in inventories]
    if set(labels) != REQUIRED_INVENTORIES or len(set(labels)) != len(labels):
        raise SampleBoundary("EXCLUSION_INVENTORY_UNAVAILABLE_OR_DUPLICATE")
    excluded = set().union(*(set(row.case_ids) for row in inventories))
    indexed = {}
    for record in records:
        case, _rw = _record_schema(record)
        if case in indexed:
            raise SampleBoundary("DATASET_DUPLICATE_CASE_ID")
        indexed[case] = record
    if not excluded <= set(indexed):
        raise SampleBoundary("EXCLUSION_CASE_NOT_IN_PINNED_DATASET")
    ranked = sorted((hashlib.sha256((NAMESPACE + "|" + str(case)).encode()).hexdigest(), case)
                    for case in indexed if case not in excluded)
    if len(ranked) < 400:
        raise SampleBoundary("SAMPLE_POOL_INSUFFICIENT")
    selected = []
    for ordinal, (rank, case) in enumerate(ranked[:400]):
        raw = indexed[case]
        selected.append(dict(ordinal=ordinal, fixture="S_DEV" if ordinal < 100 else "S_AUDIT_RESERVED",
            audit_batch=None if ordinal < 100 else (ordinal - 100) // 100,
            fixture_ordinal=ordinal if ordinal < 100 else ordinal - 100,
            case_id=case, rank_sha256=rank, raw_record_sha256=canonical_hash(raw),
            request_sha256=canonical_hash(raw["requested_rewrite"])))
    cohorts = {}
    for name, group in (("S_DEV", selected[:100]), ("S_AUDIT_RESERVED", selected[100:])):
        cohorts[name] = dict(records=group, ordered_root=canonical_hash(group),
            request_order_sha256=canonical_hash([row["request_sha256"] for row in group]),
            evaluation=evaluation_inventory([indexed[row["case_id"]] for row in group]))
    manifest = dict(schema="alpha-jv-diagnosis.s-samples.v1", namespace=NAMESPACE,
        selection="SHA256(namespace|canonical_case_id), ascending hash then integer case ID",
        models=["llama3-8b-inst", "qwen2.5-7b-inst"], model_payload_duplication=0,
        dataset=dict(dataset_identity), cohorts=cohorts, selected_count=400,
        eligible_count=len(ranked), excluded_unique_count=len(excluded),
        exclusions=[dict(label=row.label, case_ids=list(row.case_ids), identity=dict(row.identity),
                         selected_overlap_count=0) for row in inventories],
        cohort_overlap_count=0, outcome_selection_count=0, target_length_selection_count=0,
        residual_selection_count=0, source_target_success_selection_count=0,
        D_restore_dependency=False, audit_execution="RESERVED_NOT_OPENED_FOR_OUTCOMES",
        scientific_promotion=False)
    manifest["manifest_identity"] = canonical_hash(manifest)
    return manifest


def prepare_sample_manifest(repo: Path, dataset: Path) -> dict[str, Any]:
    """Read approved canonical pool locally; exact exclusions from immutable Git."""
    if dataset.is_symlink() or not dataset.is_file():
        raise SampleBoundary("DATASET_NOT_REGULAR")
    payload = dataset.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != DATASET_SHA256 or len(payload) != DATASET_BYTES:
        raise SampleBoundary("DATASET_IDENTITY_BOUNDARY")
    result = select_cohorts(json.loads(payload), load_git_inventories(repo),
                            dataset_identity=dict(path=str(dataset.absolute()), sha256=digest, bytes=len(payload)))
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != digest:
        raise SampleBoundary("DATASET_CHANGED_DURING_SELECTION")
    return result
