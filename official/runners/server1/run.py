"""Actual Llama3 official native runs, batch checkpoint/resume and CPU collection.

Qualification is the user-requested full-BS100 B1..B3 continuous versus B2 stop
and fresh-process B3 resume, not a small editing pilot. It executes inside a
registered GPU job. Control inspection never claims its future result is PASS.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from official.experiments import checkpoint
from official.experiments.prepare import digest, write_new
from official.evaluation.factual import evaluate, build_zsre_w0_reference
from official.evaluation.reduce import counterfact, zsre
from .common import (METHODS, MILESTONES, Tracking, bindings, factual_payload,
                     generation_payload, immutable_observation, load_model, local_output, member, read,
                     require, restore_checkpoint, rng_digest, rng_content, seed_edit,
                     source_binding, validate_config, verify)


def factual(model, tokenizer, records, dataset, external, reference=None, tracker=None, edits=0):
    def progress(value):
        if tracker:
            # Existing public scalar keys, no new schema invented by this runner.
            tracker.log({"step": value["completed_candidate_sequences"], "edits": edits})
    return evaluate(model, tokenizer, records, dataset, identity=external,
                    w0_reference=reference, batch_size=16, device="cuda:0", progress=progress)


def checkpoint_save(output, batch, engine, identity, cursor):
    return checkpoint.save(output / "checkpoint", batch=batch,
        weights=engine.selected_weights, cache_c={}, contexts=engine.contexts(),
        evaluation_cursor=cursor, identity=identity, method=engine.method, evaluation_complete=True)


def stage(args, config, lock, output):
    """One subprocess, genuine cold load or pinned checkpoint restore."""
    from .native import NativeEngine
    assets, records, identity, external = bindings(config, lock)
    require(args.dataset == "cf" and args.method in METHODS, "CF_NATIVE_B3_QUALIFICATION_SCOPE")
    model, tokenizer = load_model(assets)
    engine = NativeEngine(model, tokenizer, args.method, assets, source_verified=True)
    seed_edit()
    start = 0
    if args.qualification_stage == "resume":
        payload = checkpoint.load(output / "checkpoint", identity)
        require(payload["batch"] == 2, "QUALIFICATION_RESUME_MUST_BE_B2")
        start = restore_checkpoint(model, engine, payload, identity)
    else:
        require(not (output / "checkpoint" / "latest.json").exists(), "QUALIFICATION_COLD_CHECKPOINT_EXISTS")
        checkpoint_save(output, 0, engine, identity, dict(completed_batch=0, phase="QUALIFICATION_COLD_W0"))
    stop = 2 if args.qualification_stage == "stop" else 3
    started = time.monotonic()
    receipts = []
    for batch in range(start + 1, stop + 1):
        receipts.append(engine.apply(records[(batch - 1) * 100:batch * 100]))
        cursor = dict(completed_batch=batch, phase="QUALIFICATION_NOT_PRODUCTION_ENDPOINT")
        endpoint = None
        if batch == 3:
            endpoint = factual(model, tokenizer, records[:300], "cf", external)
            immutable_observation(output / "B3-factual-local.json", endpoint)
            # Preserve the original deterministic observation separately. The
            # immutable proof retains actual work for CPU receipt revalidation.
            write_new(output / "B3-factual-proof-local.json", endpoint)
            cursor["factual_member"] = member(output / "B3-factual-proof-local.json")
            if args.qualification_stage == "continuous":
                from .native_parity import qualify
                try:
                    oracle = qualify(model, tokenizer, records, external, engine,
                                     config["qualification_plan"]["native_reference_matched_B3"], endpoint)
                except BaseException as error:
                    if getattr(error, "local_report", None) is not None:
                        write_new(output / "native-CF-parity-rejected-local.json", error.local_report)
                    raise
                write_new(output / "native-CF-parity-first300-local.json", oracle)
                cursor["native_parity_member"] = member(output / "native-CF-parity-first300-local.json")
        checkpoint_save(output, batch, engine, identity, cursor)
    state = engine.state_identity()
    payload = checkpoint.load(output / "checkpoint", identity)
    from .w0_binding import execution_identity
    value = dict(schema="official-server1-native-qualification-stage-v1",
        stage=args.qualification_stage, completed_batch=stop, selected_state=state,
        contexts_sha256=digest(engine.contexts()), RNG_sha256=rng_digest(checkpoint.rng_snapshot()),
        checkpoint_RNG_sha256=rng_digest(payload["rng"]), method=args.method,
        context_content=engine.contexts(), RNG_content=rng_content(checkpoint.rng_snapshot()),
        checkpoint_RNG_content=rng_content(payload["rng"]),
        execution_identity=execution_identity(config,lock,assets,output=output,role="QUALIFICATION"),
        identity=identity, actual_model_loaded=True, actual_native_fit_calls=len(receipts),
        edit_receipts=receipts, seconds=time.monotonic() - started,
        factual_member=member(output / "B3-factual-proof-local.json") if stop == 3 else None,
        native_parity_member=member(output / "native-CF-parity-first300-local.json")
            if args.qualification_stage == "continuous" else None,
        GPU_actual=True, performance_gate=False, generation_calls=0)
    write_new(output / ("stage-stop.json" if stop == 2 else "stage-complete.json"), value)


def compare_qualification(continuous, resumed):
    for key in ("identity", "method", "completed_batch", "selected_state",
                "contexts_sha256", "RNG_sha256", "checkpoint_RNG_sha256"):
        require(continuous[key] == resumed[key], "ACTUAL_NATIVE_RESUME_MISMATCH_" + key.upper())
    require(continuous["completed_batch"] == 3, "ACTUAL_NATIVE_B3_MISSING")
    left, right = (read(verify(row["factual_member"])) for row in (continuous, resumed))
    for key in ("identity", "identity_sha256", "summary", "accuracy", "cases"):
        require(left[key] == right[key], "ACTUAL_FACTUAL_RESUME_MISMATCH_" + key.upper())
    return dict(selected_weights="EXACT_SHA256", contexts="EXACT", RNG="EXACT",
                factual="EXACT_RAW_VALUES_AND_TOKEN_IDENTITY", no_tolerance_relaxation=True)


def qualification(args, config, lock, output, tracker):
    started = time.monotonic()
    write_new(output / "qualification-plan.json", config["qualification_plan"])
    for stage_name, folder in (("continuous", output / "continuous"),
                               ("stop", output / "split"), ("resume", output / "split")):
        tracker.log({"phase_id": ("continuous", "stop", "resume").index(stage_name) + 1})
        argv = [sys.executable, "-B", "-u", "-m", "official.runners.server1.run",
                "--mode", "qualification_stage", "--qualification-stage", stage_name,
                "--method", args.method, "--dataset", "cf", "--config", str(args.config),
                "--output", str(folder), "--source-lock", str(args.source_lock)]
        # Same Slurm allocation, independent process/native globals, no new job.
        result = subprocess.run(argv, check=False)
        require(result.returncode == 0, "ACTUAL_NATIVE_QUALIFICATION_STAGE_FAILED_" + stage_name.upper())
    continuous = read(output / "continuous" / "stage-complete.json")
    resumed = read(output / "split" / "stage-complete.json")
    checks = compare_qualification(continuous, resumed)
    oracle = verify_native_parity_stage(continuous, config["qualification_plan"])
    value = dict(schema="official-server1-native-resume-READY-v1", method=args.method,
                 identity=continuous["identity"], qualification_plan_sha256=digest(config["qualification_plan"]),
                 checks=checks, continuous=member(output / "continuous" / "stage-complete.json"),
                 stopped=member(output / "split" / "stage-stop.json"),
                 resumed=member(output / "split" / "stage-complete.json"),
                 native_parity=oracle,
                 actual_native_B3_and_B2_resume=True, passed=True,
                 actual_fit_calls=6, generation_calls=0, seconds=time.monotonic() - started)
    write_new(output / "READY.json", value)
    tracker.log({"setup_ok": 1, "step": 6, "time/elapsed_seconds": value["seconds"]})


def verify_native_parity_stage(continuous, plan):
    from .native_parity import validate_report
    report = read(verify(continuous["native_parity_member"]))
    endpoint = read(verify(continuous["factual_member"]))
    require(report.get("state_after") == continuous["selected_state"]
            and report.get("reference_binding", {}).get("state_identity", {}).get("method") == continuous["method"],
            "NATIVE_PARITY_COMPLETED_B3_STATE_IDENTITY")
    require(report.get("canonical") == endpoint
            and report.get("canonical_payload_sha256") == digest(endpoint),
            "NATIVE_PARITY_CANONICAL_COMPLETED_B3_MEMBER")
    return validate_report(report, endpoint["identity"]["external_identity"], plan["native_reference_matched_B3"])


def verify_qualifications(config, identity):
    expected_identity = dict(identity)
    if config.get("dataset") == "zsre":
        # The independently observed qualification is always CF first300, even
        # when a later science caller consumes the separate zsRE stream.
        # Bind its CF input to the actual pinned bundle/assets, not to an
        # untrusted report's self-declared stream or the zsRE endpoint identity.
        bundle = read(verify(config["stream_bundle_member"]))
        assets = read(verify(config["assets_member"]))
        cf_member = member(verify(bundle["datasets"]["cf"]["stream"]))
        require(cf_member["sha256"] == assets["streams"]["cf"]["stream_sha256"],
                "QUALIFICATION_ACTUAL_CF_STREAM_ASSET_IDENTITY")
        expected_identity["stream_sha256"] = cf_member["sha256"]
    for method in METHODS:
        value = read(config["qualification_outputs"][method] + "/READY.json")
        verify_qualification(value, method, expected_identity, config.get("qualification_plan"))


def verify_qualification(value, method, identity=None, plan=None):
    require(value.get("schema") == "official-server1-native-resume-READY-v1"
            and value.get("method") == method and value.get("passed") is True
            and value.get("actual_native_B3_and_B2_resume") is True
            and value.get("actual_fit_calls") == 6 and value.get("generation_calls") == 0,
            "ACTUAL_NATIVE_QUALIFICATION_READY_REQUIRED")
    if identity is not None:
        for key in ("code_commit", "official_tree_sha256", "model_revision", "tokenizer_sha256", "assets_sha256", "stream_sha256"):
            require(value["identity"][key] == identity[key], "QUALIFICATION_SOURCE_ASSET_IDENTITY")
    if plan is not None:
        require(value["qualification_plan_sha256"] == digest(plan), "QUALIFICATION_PLAN_IDENTITY")
    stages = {key: read(verify(value[key])) for key in ("continuous", "stopped", "resumed")}
    for key, stage_name, batches, calls in (("continuous", "continuous", 3, 3),
            ("stopped", "stop", 2, 2), ("resumed", "resume", 3, 1)):
        stage_value = stages[key]
        require(stage_value.get("schema") == "official-server1-native-qualification-stage-v1"
                and stage_value.get("stage") == stage_name and stage_value.get("completed_batch") == batches
                and stage_value.get("actual_native_fit_calls") == calls
                and stage_value.get("GPU_actual") is True and stage_value.get("actual_model_loaded") is True
                and stage_value.get("identity") == value["identity"] and stage_value.get("method") == method,
                "QUALIFICATION_ACTUAL_STAGE_IDENTITY")
    require(value["checks"] == compare_qualification(stages["continuous"], stages["resumed"]),
            "QUALIFICATION_ACTUAL_CHECKS")
    require(plan is not None and value.get("native_parity") == verify_native_parity_stage(stages["continuous"], plan),
            "QUALIFICATION_ACTUAL_NATIVE_MATCHED_B3_PARITY")


def generation_observer(model, tokenizer, assets, lock, output, engine=None, tracker=None, *, endpoint="W20"):
    from official.evaluation.generation.assets import load_assets
    from official.evaluation.generation.native_observer import NativeGenerationObserver
    from official.evaluation.generation.native_profile import PROFILE
    references = load_assets(assets["generation_reference"]["manifest"]["path"])
    config = dict(model_identity=dict(model="llama3", revision=assets["model"]["identity"]["revision"],
                    tokenizer_sha256=assets["model"]["tokenizer_sha256"]),
                  profile=PROFILE, eval_seed=20261007,
                  generation_source_sha=dict(code_commit=lock["source"]["main_commit"],
                                             official_tree=lock["source"]["official_tree"]))
    def native_state():
        return engine.contexts() if engine else dict(cold_base=True)
    def progress(payload):
        if tracker:
            from official.tracking import official_generation_progress
            tracker.log(official_generation_progress(payload, endpoint=endpoint))
    return NativeGenerationObserver(model, tokenizer, references, config, output,
        state_callback=native_state, progress_callback=progress)


def base_w0(args, config, lock, output, tracker):
    assets, cf_records, identity, external = bindings(config, lock)
    verify_qualifications(config, identity)
    model, tokenizer = load_model(assets)
    seed_edit()
    cf = factual(model, tokenizer, cf_records, "cf", external, tracker=tracker)
    immutable_observation(output / "cf-factual-local.json", cf)
    write_new(output / "cf-factual-proof-local.json", cf)
    observer = generation_observer(model, tokenizer, assets, lock, output / "cf-generation", tracker=tracker, endpoint="W0")
    generation = observer.observe(cf_records, "W0", cohort="first2000", state_identity=dict(
        base_model=assets["model"]["identity"], assets_sha256=assets["assets_sha256"], actual_model_edits=0))
    immutable_observation(output / "cf-generation-local.json", generation)
    full_W0 = factual_payload(cf, "W0_first2000", 0)
    full_W0.update(generation_payload(generation, 0, prefix="W0_first2000"))
    tracker.log(full_W0)
    # Load the same model's independently bound zsRE stream, no repeated CF W0.
    bundle = read(verify(config["stream_bundle_member"]))
    zsre_path = verify(bundle["datasets"]["zsre"]["stream"])
    require(member(zsre_path)["sha256"] == assets["streams"]["zsre"]["stream_sha256"],
            "ZSRE_W0_STREAM_ASSET_IDENTITY")
    zsre_records = read(zsre_path)
    require(len(zsre_records) == 2000 and [row["occurrence_index"] for row in zsre_records]
            == list(range(1, 2001)), "ZSRE_W0_ORDERED_2000")
    zsre_external = dict(external, stream_sha256=assets["streams"]["zsre"]["stream_sha256"])
    reference = build_zsre_w0_reference(model, tokenizer, zsre_records, identity=zsre_external,
                                         batch_size=16, device="cuda:0")
    write_new(output / "zsre-reference-local.json", reference)
    from .w0_binding import build_fingerprint, consumed_source_members, execution_identity
    from official.evaluation.w0_reference import make_ready
    fingerprint = build_fingerprint(assets, tokenizer, dict(cf=cf_records, zsre=zsre_records))
    qualification_members = {method:member(Path(config["qualification_outputs"][method]) / "READY.json")
                             for method in METHODS}
    components = dict(cf_factual=member(output / "cf-factual-proof-local.json"),
                      cf_generation=member(output / "cf-generation-local.json"),
                      zsre_reference=member(output / "zsre-reference-local.json"))
    validation = {name:dict(actual_complete=True, actual_model_edits=0, state="W0_COLD_BASE_MODEL",
        requests=2000, ordered_queries_sha256=fingerprint["content"]["datasets"][
            "zsre" if name == "zsre_reference" else "cf"]["ordered_queries_sha256"],
        native_qualification_state=dict(status="PASS", evidence_members=qualification_members,
                                        evidence_sha256=digest(qualification_members))) for name in components}
    portable = make_ready(producer_execution_identity=execution_identity(config, lock, assets, output=output),
        fingerprint=fingerprint, members=components, component_validation=validation,
        generation_runtime_member=member(output / "cf-generation" / "observer-identity.json"),
        source_members=consumed_source_members(),
        dataset_members={name:bundle["datasets"][name]["stream"] for name in ("cf","zsre")},
        generation_assets=dict(manifest=assets["generation_reference"]["manifest"]))
    write_new(output / "PORTABLE_READY.json", portable)
    # CF W0 run's config/namespace cannot carry zsRE scores. The separately
    # bound zsRE chains log the shared exact token reference in their own runs.
    ready = dict(schema="official-server1-base-W0-READY-v1", actual_model_edits=0,
        model_identity=assets["model"]["identity"], source=lock["source"], assets_sha256=assets["assets_sha256"],
        cf_external_identity=external, zsre_external_identity=zsre_external,
        cf_factual=member(output / "cf-factual-proof-local.json"),
        cf_generation=member(output / "cf-generation-local.json"),
        zsre_reference=member(output / "zsre-reference-local.json"),
        portable_reference=member(output / "PORTABLE_READY.json"),
        CF_W0_observed_once=True, shared_across_methods=True, actual_complete=True)
    write_new(output / "READY.json", ready)


def read_w0(config, lock, assets):
    ready = read(Path(config["base_W0_output"]) / "READY.json")
    require(ready.get("schema") == "official-server1-base-W0-READY-v1" and ready.get("actual_complete") is True
            and type(ready.get("actual_model_edits")) is int and ready["actual_model_edits"] == 0
            and ready.get("CF_W0_observed_once") is True and ready.get("shared_across_methods") is True
            and ready["source"] == lock["source"] and ready["assets_sha256"] == assets["assets_sha256"]
            and ready["model_identity"] == assets["model"]["identity"], "COMMON_W0_READY_IDENTITY")
    for key in ("cf_factual", "cf_generation", "zsre_reference"):
        verify(ready[key])
    from official.evaluation.factual import _digest as factual_digest
    base_external = dict(model="llama3", model_revision=assets["model"]["identity"]["revision"],
        tokenizer_sha256=assets["model"]["tokenizer_sha256"], assets_sha256=assets["assets_sha256"],
        code_commit=lock["source"]["main_commit"], official_tree=lock["source"]["official_tree"],
        runtime=assets["runtime"]["dependency_versions"],
        precision="FP32_EAGER_TF32_OFF_NO_AUTOCAST", raw_local_only=True)
    for dataset, ready_key in (("cf", "cf_factual"), ("zsre", "zsre_reference")):
        external = dict(base_external, stream_sha256=assets["streams"][dataset]["stream_sha256"])
        require(ready[dataset + "_external_identity"] == external, "COMMON_W0_EXTERNAL_IDENTITY")
        value = read(ready[ready_key]["path"])
        endpoint = value["evaluation"] if dataset == "zsre" else value
        require(endpoint["identity"]["external_identity"] == external
                and endpoint["identity"]["dataset"] == dataset
                and endpoint["identity"]["ordered_occurrences"] == list(range(1, 2001))
                and endpoint["identity_sha256"] == factual_digest(endpoint["identity"])
                and endpoint.get("model_no_mutation") is True and endpoint.get("RNG_restored") is True
                and len(endpoint["cases"]) == 2000, "COMMON_W0_FACTUAL_IDENTITY_COUNT")
        if dataset == "zsre":
            require(value["identity"]["external_identity"] == external
                    and value["payload_sha256"] == factual_digest(
                        {key: item for key, item in value.items() if key != "payload_sha256"}),
                    "COMMON_W0_ZSRE_REFERENCE_PAYLOAD")
    portable_path = verify(ready["portable_reference"])
    require(portable_path == Path(config["base_W0_output"]) / "PORTABLE_READY.json",
            "COMMON_W0_PORTABLE_REFERENCE_OWN_MEMBER")
    return ready


def factual_equivalent(left, right):
    require(left["summary"] == right["summary"] and left["cases"] == right["cases"]
            and left["accuracy"] == right["accuracy"], "SMOKE_NATIVE_FACTUAL_MISMATCH")


def recover_committed_ledger(output, payload, identity):
    """Close only the checkpoint-pointer→ledger crash window, without refitting.

    The already SHA-verified local checkpoint is authoritative for this one
    latest batch. Older missing ledger rows cannot be reconstructed from it.
    Existing rows are immutable and must equal its exact cursor/identity.
    """
    batch = payload["batch"]
    require(payload["identity"] == identity, "RESUME_LEDGER_CHECKPOINT_IDENTITY")
    if batch == 0:
        return
    cursor = payload["evaluation_cursor"]
    pointer = read(output / "checkpoint" / "latest.json")
    require(cursor.get("completed_batch") == batch and pointer.get("batch") == batch
            and pointer.get("identity_sha256") == digest(identity)
            and cursor.get("edit", {}).get("batch") == batch,
            "RESUME_LEDGER_CURSOR_IDENTITY")
    for prior in range(1, batch):
        prior_path = output / "commits" / f"batch-{prior:02d}.json"
        require(prior_path.is_file(),
                "RESUME_OLDER_LEDGER_MISSING_CANNOT_RECONSTRUCT")
        old = read(prior_path)
        require(old.get("identity") == identity and old.get("batch") == prior
                and old.get("actual_applied_requests") == 100 * prior
                and old.get("cursor", {}).get("completed_batch") == prior,
                "RESUME_OLDER_LEDGER_IDENTITY_CONFLICT")
    for key in ("factual", "generation"):
        if key in cursor:
            verify(cursor[key])
    expected = dict(batch=batch, cursor=cursor, checkpoint=pointer,
                    actual_applied_requests=100 * batch, identity=identity)
    path = output / "commits" / f"batch-{batch:02d}.json"
    if path.exists():
        require(read(path) == expected, "RESUME_EXISTING_LEDGER_CONFLICT")
    else:
        write_new(path, expected)


def chain(args, config, lock, output, tracker):
    from .native import NativeEngine
    assets, records, identity, external = bindings(config, lock)
    verify_qualifications(config, identity)
    ready = read_w0(config, lock, assets)
    if args.dataset == "zsre" and args.mode == "chain":
        smoke = read(Path(config["zsre_smoke_output"]) / "READY.json")
        require(smoke.get("actual_batch100_completed") is True and smoke["source"] == lock["source"]
                and smoke["assets_sha256"] == assets["assets_sha256"], "ZSRE_ACTUAL_BATCH100_SMOKE_REQUIRED")
    reference = read(ready["zsre_reference"]["path"]) if args.dataset == "zsre" else None
    model, tokenizer = load_model(assets)
    engine = NativeEngine(model, tokenizer, args.method, assets, source_verified=True)
    seed_edit()
    start, cursor = 0, dict(completed_batch=0, W0_ready_member=member(Path(config["base_W0_output"]) / "READY.json"))
    if args.resume:
        payload = checkpoint.load(output / "checkpoint", identity)
        start = restore_checkpoint(model, engine, payload, identity)
        cursor = payload["evaluation_cursor"]
        require(cursor["completed_batch"] == start, "RESUME_EVALUATION_CURSOR")
        recover_committed_ledger(output, payload, identity)
    else:
        require(not (output / "checkpoint" / "latest.json").exists(), "EXISTING_CHECKPOINT_USE_EXPLICIT_RESUME")
        checkpoint_save(output, 0, engine, identity, cursor)
        W0_endpoint = reference["evaluation"] if reference else read(ready["cf_factual"]["path"])
        tracker.log(factual_payload(W0_endpoint, "W0_first2000", 0))
    stop = 1 if args.mode == "smoke" else 20
    require(start < stop, "CHAIN_ALREADY_COMPLETED_NO_DUPLICATE_EXECUTION")
    observer = None
    for batch in range(start + 1, stop + 1):
        edit = engine.apply(records[(batch - 1) * 100:batch * 100])
        cursor = dict(completed_batch=batch, edit=edit, evaluation="NOT_SCHEDULED_THIS_BATCH")
        if batch in MILESTONES or args.mode == "smoke":
            endpoint = factual(model, tokenizer, records[:100 * batch], args.dataset,
                               external, reference=reference, tracker=tracker, edits=100 * batch)
            endpoint_path = output / "factual" / f"batch-{batch:02d}.json"
            immutable_observation(endpoint_path, endpoint)
            cursor.update(evaluation="COMPLETE", factual=member(endpoint_path))
            if args.mode != "smoke" and not (batch == 20 and args.dataset == "cf"):
                tracker.log(factual_payload(endpoint, "all_seen/post", 100 * batch))
        if batch == 20 and args.dataset == "cf":
            observer = generation_observer(model, tokenizer, assets, lock, output / "generation-W20",
                                           engine=engine, tracker=tracker)
            observed = observer.observe(records, "W20", cohort="first2000", state_identity=dict(
                method=args.method, model=assets["model"]["identity"], selected_state=engine.state_identity(),
                source=lock["source"], actual_model_edits=2000))
            path = output / "generation-W20-local.json"
            immutable_observation(path, observed)
            cursor["generation"] = member(path)
            full_W20 = factual_payload(endpoint, "all_seen/post", 2000)
            full_W20.update(generation_payload(observed, 2000))
            tracker.log(full_W20)
        # No checkpoint is committed before all scheduled observations finish.
        # A W20 generation exception leaves the pinned B19 pointer intact.
        pointer = checkpoint_save(output, batch, engine, identity, cursor)
        write_new(output / "commits" / f"batch-{batch:02d}.json", dict(batch=batch, cursor=cursor,
                  checkpoint=pointer, actual_applied_requests=100 * batch, identity=identity))
        # Do not replace the measured eval row in the bounded SDK readback
        # with a final generic commit/status row that happens to carry edits.
        tracker.log({"batch": batch, "step": batch, "status_code": 1})
    if args.mode == "smoke":
        write_new(output / "READY.json", dict(schema="official-server1-zsre-smoke-v1", source=lock["source"],
            assets_sha256=assets["assets_sha256"], actual_batch100_completed=True,
            requests=100, native_method=args.method, factual=cursor["factual"], performance_gate=False))
    else:
        write_new(output / "COMPLETE.json", dict(schema="official-server1-chain-complete-v1", method=args.method,
            dataset=args.dataset, identity=identity, actual_native_apply_calls=20, actual_applied_requests=2000,
            final_cursor=cursor, final_checkpoint=member(output / "checkpoint" / "latest.json"),
            checkpoint_W20_preserved=True, scientific_complete=True, model_forward_not_mock=True))


def _collect_local_path(path, parent=None):
    """Read-only scope check: never create missing science output directories."""
    from .common import LOCAL_ROOT
    value = Path(path)
    require(value.is_absolute() and value.is_relative_to(LOCAL_ROOT), "COLLECTOR_OWN_LOCAL_SCOPE")
    if parent is not None:
        require(value.is_relative_to(Path(parent)), "COLLECTOR_ENDPOINT_OUTPUT_SCOPE")
    for ancestor in (value, *value.parents):
        if ancestor == LOCAL_ROOT.parent:
            break
        require(not ancestor.is_symlink(), "COLLECTOR_LOCAL_SYMLINK")
    require(value == value.resolve(), "COLLECTOR_LOCAL_NONCANONICAL_PATH")
    return value


def _collect_generation(path, records, tokenizer, assets, lock, endpoint, *, method=None, final_edit=None,
                        references=None, folder=None):
    """Re-score original native text on CPU with its actual fixed references."""
    from official.evaluation.generation.assets import load_assets
    from official.evaluation.generation.common import digest as generation_digest
    from official.evaluation.generation.native_observer import read_observed
    from official.evaluation.generation.native_profile import PROFILE, ROUTE, runtime_identity
    from official.evaluation.generation.observer import _record_identity
    records = list(records)
    require(len(records) == 2000 and [r["occurrence_index"] for r in records] == list(range(1, 2001)),
            "COLLECTOR_GENERATION_EXACT_FIRST2000")
    verify(assets["generation_reference"]["manifest"])
    references = references or load_assets(assets["generation_reference"]["manifest"]["path"])
    require(references.sha == assets["generation_reference"]["identity_sha256"],
            "COLLECTOR_GENERATION_REFERENCE_IDENTITY")
    generation_config = dict(model_identity=dict(model="llama3",
        revision=assets["model"]["identity"]["revision"],
        tokenizer_sha256=assets["model"]["tokenizer_sha256"]), profile=PROFILE, eval_seed=20261007,
        generation_source_sha=dict(code_commit=lock["source"]["main_commit"],
                                   official_tree=lock["source"]["official_tree"]))
    expected_runtime = generation_digest(runtime_identity(generation_config, references.sha))
    saved = read(_collect_local_path(path, folder))
    raw_path = _collect_local_path(saved["rows_path"], folder)
    raw_endpoint = read(raw_path)
    require("native_execution_member" in raw_endpoint and not any(key in raw_endpoint for key in
            ("parent_endpoint_member", "compatibility_member", "qualification_receipt_member")),
            "COLLECTOR_FULL_NATIVE_ENDPOINT_REQUIRED")
    _collect_local_path(raw_endpoint["native_execution_member"]["path"], folder)
    for row in raw_endpoint["rows"]:
        _collect_local_path(row["observation_path"], folder)
        if "provenance" in row:
            _collect_local_path(row["provenance"]["raw_member"]["path"], folder)
    parent = read_observed(raw_path, expected_runtime=expected_runtime, assets=references)
    scientific = lambda value: {k: v for k, v in value.items() if k not in ("work", "rows_path")}
    require(scientific(parent) == scientific(saved), "COLLECTOR_GENERATION_LOCAL_RAW_IDENTITY")
    require(parent["identity"]["endpoint"] == endpoint and parent["identity"]["cohort"] == "first2000"
            and parent["identity"]["ordered_occurrences"] == list(range(1, 2001))
            and len(parent["rows"]) == 2000 and parent["summary"]["planned_count"] == 2000
            and "native_execution_member" in parent and "parent_endpoint_member" not in parent,
            "COLLECTOR_GENERATION_ENDPOINT_COHORT")
    execution = read(verify(parent["native_execution_member"]))
    require(execution["profile"] == PROFILE and execution["route"] == ROUTE,
            "COLLECTOR_NATIVE_GENERATION_PROFILE")
    expected_state = dict(base_model=assets["model"]["identity"], assets_sha256=assets["assets_sha256"],
                          actual_model_edits=0)
    observed_prompts = 0
    for record, row in zip(records, parent["rows"]):
        raw = read(_collect_local_path(row["observation_path"], folder))
        require(raw["identity"]["record_identity"] == _record_identity(record, record["occurrence_index"]),
                "COLLECTOR_GENERATION_ORDERED_PROMPT_TARGET_IDENTITY")
        state = raw["identity"]["state_identity"]
        if endpoint == "W0":
            require(state == expected_state, "COLLECTOR_GENERATION_COLD_W0_STATE")
        else:
            require(state.get("method") == method and state.get("model") == assets["model"]["identity"]
                    and state.get("source") == lock["source"]
                    and type(state.get("actual_model_edits")) is int and state["actual_model_edits"] == 2000,
                    "COLLECTOR_GENERATION_EDITED_W20_STATE")
            selected = state.get("selected_state", {})
            unsigned = {k: v for k, v in selected.items() if k != "identity_sha256"}
            hashes = {name: value.get("sha256") for name, value in selected.get("selected_weights", {}).items()}
            require(final_edit is not None and selected.get("method") == method
                    and selected.get("successful_calls") == 20 and selected.get("cache_c") == {}
                    and selected.get("identity_sha256") == digest(unsigned)
                    and hashes == final_edit["post_selected_sha256"]
                    and selected.get("contexts_sha256") == final_edit["contexts_sha256"]
                    and all(value.get("dtype") == "torch.float32" for value in selected["selected_weights"].values()),
                    "COLLECTOR_GENERATION_NATIVE_COMMIT_STATE")
        for observation in raw["observations"]:
            planned_tokens = tokenizer(observation["prompt"])["input_ids"]
            require(observation["input_token_ids"] == planned_tokens,
                    "COLLECTOR_GENERATION_NATIVE_INPUT_TOKENS")
        observed_prompts += len(raw["observations"])
    require(parent["summary"]["generation_prompt_count"] == observed_prompts,
            "COLLECTOR_GENERATION_PROMPT_DENOMINATOR")
    return dict(endpoint=endpoint, requests=2000, prompts=observed_prompts,
                runtime_sha256=expected_runtime, endpoint_identity_sha256=parent["identity_sha256"],
                reference_assets_sha256=references.sha, native_profile=PROFILE,
                independent_text_metric_reduction=True, actual_model_forward_calls=0)


def _collect_profile(profile, profile_config, lock, folder, assets, records, identity, external, tokenizer,
                     references_cache):
    from .audit import audit_commits, audit_factual
    mode, method, dataset = (profile[key] for key in ("mode", "method", "dataset"))
    terminal = folder / ("COMPLETE.json" if mode == "chain" else "READY.json")
    receipt = read(terminal)
    if mode == "qualification":
        verify_qualification(receipt, method, identity, profile_config["qualification_plan"])
        for name in ("continuous", "resumed"):
            stage_value = read(verify(receipt[name]))
            audit_factual(read(verify(stage_value["factual_member"])), records[:300], "cf", tokenizer, external)
        return dict(verification="ACTUAL_QUALIFICATION_RECEIPTS_VERIFIED", actual_fit_calls=6,
                    actual_native_B3_and_B2_resume=True, performance_gate=False)
    if mode == "base_w0":
        ready = read_w0(dict(profile_config, base_W0_output=str(folder)), lock, assets)
        cf_audit = audit_factual(read(verify(ready["cf_factual"])), records, "cf", tokenizer, external)
        bundle = read(verify(profile_config["stream_bundle_member"]))
        zsre_path = verify(bundle["datasets"]["zsre"]["stream"])
        require(member(zsre_path)["sha256"] == assets["streams"]["zsre"]["stream_sha256"],
                "COLLECTOR_W0_ZSRE_STREAM_IDENTITY")
        zsre_records, reference = read(zsre_path), read(verify(ready["zsre_reference"]))
        require(len(zsre_records) == 2000 and [r["occurrence_index"] for r in zsre_records] == list(range(1, 2001)),
                "COLLECTOR_W0_ZSRE_ORDERED_2000")
        zsre_external = dict(external, stream_sha256=assets["streams"]["zsre"]["stream_sha256"])
        zsre_audit = audit_factual(reference["evaluation"], zsre_records, "zsre", tokenizer,
                                   zsre_external, reference)
        reference_assets = references_cache[assets["assets_sha256"]]
        generation_audit = _collect_generation(verify(ready["cf_generation"]), records, tokenizer, assets,
            lock, "W0", references=reference_assets, folder=folder)
        from .w0_binding import build_fingerprint, execution_identity
        from official.evaluation.w0_reference import read_ready
        borrowed = read_ready(verify(ready["portable_reference"]),
            consumer_execution_identity=execution_identity(profile_config,lock,assets,output=folder,role="CPU_REDUCER"),
            consumer_fingerprint=build_fingerprint(assets, tokenizer, dict(cf=records,zsre=zsre_records)),
            generation_assets=dict(manifest=assets["generation_reference"]["manifest"]))
        return dict(verification="ACTUAL_COLD_W0_RAW_VERIFIED", actual_model_edits=0,
                    cf_factual=cf_audit, zsre_reference=zsre_audit, generation=generation_audit,
                    portable_reference=ready["portable_reference"],
                    portable_binding_sha256=borrowed.binding["binding_sha256"])
    require(mode in ("chain", "smoke"), "COLLECTOR_UNSUPPORTED_ACTUAL_PROFILE")
    ready = read_w0(profile_config, lock, assets)
    reference = read(verify(ready["zsre_reference"])) if dataset == "zsre" else None
    expected_batches = 1 if mode == "smoke" else 20
    if mode == "smoke":
        require(dataset == "zsre" and receipt.get("schema") == "official-server1-zsre-smoke-v1"
                and receipt.get("source") == lock["source"] and receipt.get("assets_sha256") == assets["assets_sha256"]
                and receipt.get("actual_batch100_completed") is True and type(receipt.get("requests")) is int
                and receipt["requests"] == 100 and receipt.get("native_method") == method
                and receipt.get("performance_gate") is False, "COLLECTOR_ACTUAL_ZSRE_SMOKE100_REQUIRED")
    else:
        require(receipt.get("schema") == "official-server1-chain-complete-v1" and receipt.get("method") == method
                and receipt.get("dataset") == dataset and receipt.get("identity") == identity
                and type(receipt.get("actual_native_apply_calls")) is int and receipt["actual_native_apply_calls"] == 20
                and type(receipt.get("actual_applied_requests")) is int and receipt["actual_applied_requests"] == 2000
                and receipt.get("checkpoint_W20_preserved") is True and receipt.get("scientific_complete") is True
                and receipt.get("model_forward_not_mock") is True, "COLLECTOR_CHAIN_ACTUAL_SOURCE_CONFIG_COUNTS")
    ledger = audit_commits(folder, method, identity, expected20=expected_batches, records=records)
    batches = (1,) if mode == "smoke" else MILESTONES
    factual_checks = []
    for batch in batches:
        commit = read(folder / "commits" / f"batch-{batch:02d}.json")
        factual_member = commit["cursor"]["factual"]
        require(Path(factual_member["path"]) == folder / "factual" / f"batch-{batch:02d}.json",
                "COLLECTOR_FACTUAL_CURSOR_PATH")
        endpoint = read(verify(factual_member))
        factual_checks.append(audit_factual(endpoint, records[:100 * batch], dataset, tokenizer, external, reference))
    last = read(folder / "commits" / f"batch-{expected_batches:02d}.json")
    require(last["cursor"]["completed_batch"] == expected_batches, "COLLECTOR_TERMINAL_CURSOR_BATCH")
    if mode == "smoke":
        require(receipt["factual"] == last["cursor"]["factual"], "COLLECTOR_SMOKE_FACTUAL_MEMBER")
    else:
        require(receipt["final_cursor"] == last["cursor"]
                and receipt["final_checkpoint"] == member(folder / "checkpoint" / "latest.json"),
                "COLLECTOR_COMPLETE_LEDGER_CURSOR_POINTER")
    result = dict(verification="ACTUAL_LOCAL_RAW_AND_LEDGER_VERIFIED", native_commits=ledger,
                  factual=factual_checks)
    if mode == "chain" and dataset == "cf":
        generation_member = last["cursor"].get("generation")
        require(generation_member is not None and Path(generation_member["path"]) == folder / "generation-W20-local.json",
                "COLLECTOR_W20_GENERATION_CURSOR")
        result["generation"] = _collect_generation(verify(generation_member), records, tokenizer, assets, lock,
            "W20", method=method, final_edit=last["cursor"]["edit"],
            references=references_cache[assets["assets_sha256"]], folder=folder)
    return result


def collect(args, config, lock, output):
    """Audit actual local receipts without model, SDK, Slurm or remote inference."""
    import re
    from transformers import AutoTokenizer
    from official.evaluation.generation.assets import load_assets
    manifest = read(args.job_manifest)
    require(manifest.get("schema") == "official-server1-job-manifest-v1"
            and manifest.get("source") == lock["source"] and manifest.get("plan_sha256") == lock["plan_sha256"]
            and read(verify(manifest["execution_lock"])) == lock, "COLLECTOR_SOURCE_EXECUTION_LOCK_IDENTITY")
    plan = read(verify(lock["plan"]))
    require(digest(plan) == lock["plan_sha256"] and plan["source"] == lock["source"]
            and manifest["profiles"] == lock["profiles"] == plan["jobs"], "COLLECTOR_IMMUTABLE_PLAN_PROFILES")
    profiles, rows = manifest["profiles"], []
    keys = [profile["key"] for profile in profiles]
    require(len(set(keys)) == len(keys) and set(manifest["jobs"]) == set(keys)
            and all(type(job) is str and re.fullmatch(r"\d+(?:_\d+)?", job) for job in manifest["jobs"].values())
            and len(set(manifest["jobs"].values())) == len(keys), "COLLECTOR_ACTUAL_JOB_PROFILE_BINDING")
    collectors = [profile for profile in profiles if profile["mode"] == "collect"]
    require(len(collectors) == 1 and Path(collectors[0]["output"]) == output
            and read(verify(collectors[0]["config"])) == config,
            "COLLECTOR_OWN_PROFILE_CONFIG_OUTPUT")
    tokens, references_cache = {}, {}
    for profile in profiles:
        if profile["mode"] == "collect":
            continue
        require(profile["mode"] in ("qualification", "base_w0", "chain", "smoke")
                and profile["config"] in lock["job_configs"], "COLLECTOR_LOCKED_PROFILE_CONFIG_REQUIRED")
        profile_config = validate_config(read(verify(profile["config"])))
        require(profile_config["dataset"] == profile["dataset"]
                and (profile["method"] is None or profile_config["method"] == profile["method"]),
                "COLLECTOR_PROFILE_CONFIG_CONFLICT")
        folder = _collect_local_path(profile["output"])
        row = dict(key=profile["key"], job_id=manifest["jobs"][profile["key"]], method=profile["method"],
                   dataset=profile["dataset"], mode=profile["mode"], source=lock["source"],
                   config_sha256=profile_config["config_sha256"])
        terminal = folder / ("COMPLETE.json" if profile["mode"] == "chain" else "READY.json")
        row.update(unverified_present_commit_files=len(list((folder / "commits").glob("batch-*.json"))),
                   local_failure_receipt_count=len(list((folder / "failures").glob("*.json"))))
        if not terminal.exists():
            row.update(status="NOT_OBSERVED_COMPLETE", actual_completion=False)
        else:
            try:
                assets, records, identity, external = bindings(profile_config, lock)
                token_key = (assets["model"]["tokenizer_path"], assets["model"]["tokenizer_sha256"])
                if token_key not in tokens:
                    tokenizer = AutoTokenizer.from_pretrained(token_key[0], local_files_only=True)
                    require(tokenizer.eos_token_id is not None, "COLLECTOR_PINNED_TOKENIZER_EOS")
                    tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "right"
                    tokens[token_key] = tokenizer
                if (profile["mode"] == "base_w0" or profile["mode"] == "chain" and profile["dataset"] == "cf"):
                    if assets["assets_sha256"] not in references_cache:
                        verify(assets["generation_reference"]["manifest"])
                        references_cache[assets["assets_sha256"]] = load_assets(
                            assets["generation_reference"]["manifest"]["path"])
                require(terminal.is_file() and not terminal.is_symlink(), "COLLECTOR_REGULAR_TERMINAL_RECEIPT")
                checks = _collect_profile(profile, profile_config, lock, folder, assets, records, identity,
                                          external, tokens[token_key], references_cache)
                row.update(status="VERIFIED_COMPLETE_LOCAL_RECEIPTS", actual_completion=True,
                           receipt=member(terminal), CPU_audit=checks)
            except Exception as error:
                code = str(error)
                row.update(status="VALIDATION_FAILED", actual_completion=False, error_type=type(error).__name__,
                    error_code=code if re.fullmatch(r"[A-Z0-9_]{1,160}", code) else "COLLECTOR_LOCAL_AUDIT_REJECTED")
        rows.append(row)
    expected_chains = sum(profile["mode"] == "chain" for profile in profiles)
    verified = sum(row["mode"] == "chain" and row["actual_completion"] for row in rows)
    from .costs import collect_costs
    accounting = collect_costs(manifest)
    write_new(output / "allocated-cost-snapshot.json", accounting)
    write_new(output / "summary.json", dict(schema="official-server1-collector-v2", rows=rows,
        source=lock["source"], plan_sha256=lock["plan_sha256"], purpose=plan["purpose"],
        actual_complete_chains=verified, expected_chains=expected_chains,
        coverage="COMPLETE" if all(row["actual_completion"] for row in rows) else "PARTIAL_OR_NOT_OBSERVED",
        actual_allocated_cost=member(output / "allocated-cost-snapshot.json"),
        accounting_snapshot_is_not_scientific_completion=True,
        missing_not_zero=True, actual_model_forward_calls=0, model_loaded=False,
        GPU_execution_independently_certified=False, remote_WB_delivery_not_inferred_from_CPU=True, GPU=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("qualification", "qualification_stage", "base_w0", "chain", "smoke", "collect"), required=True)
    parser.add_argument("--qualification-stage", choices=("continuous", "stop", "resume"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--dataset", choices=("cf", "zsre"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--job-manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = validate_config(read(args.config))
    require(args.dataset == config["dataset"] and
            (args.method is None or args.method == config["method"]), "RUNNER_PROFILE_CONFIG_CONFLICT")
    lock = source_binding(args.source_lock, args.config)
    output = local_output(args.output)
    if args.mode == "collect":
        require(args.job_manifest is not None, "COLLECTOR_ACTUAL_JOB_MANIFEST_REQUIRED")
        collect(args, config, lock, output)
        return
    if args.mode == "qualification_stage":
        require(args.qualification_stage is not None, "QUALIFICATION_STAGE_REQUIRED")
        stage(args, config, lock, output)
        return
    assets, records, identity, external = bindings(config, lock)
    tracker = None
    try:
        tracker = Tracking(config, output, identity, mode=args.mode, method=args.method, dataset=args.dataset)
        if args.mode == "qualification":
            qualification(args, config, lock, output, tracker)
        elif args.mode == "base_w0":
            base_w0(args, config, lock, output, tracker)
        else:
            chain(args, config, lock, output, tracker)
    except BaseException as error:
        # Error type/code only; native local stdout/raw is not uploaded/published.
        write_new(output / "failures" / (uuid.uuid4().hex + ".json"), dict(error_type=type(error).__name__, error=str(error),
            mode=args.mode, method=args.method, dataset=args.dataset, identity=identity,
            old_source_raw_preserved=True, automatic_retry=False, checkpoint_latest_preserved=True))
        if tracker:
            try:
                tracker.finish(exit_code=1)
            except BaseException as transport_error:
                if hasattr(error, "add_note"):
                    error.add_note("Transport finish also failed: " + type(transport_error).__name__)
        raise
    else:
        if tracker:
            tracker.finish(exit_code=0)


if __name__ == "__main__":
    main()
