"""Content-only shared W0 fingerprint; execution/path/hardware stay separate."""
import importlib.metadata
from pathlib import Path
import platform

from official.evaluation.factual import _digest, _plan
from official.evaluation.generation.native_profile import PROFILE, SOURCE
from official.experiments.prepare import file_sha
from .common import member, read, require


GENERATION_CLOSURE = ("__init__.py", "assets.py", "metrics.py", "native_generator.py",
    "native_observer.py", "native_profile.py", "generator.py", "observer.py", "common.py",
    "compatibility.py", "progress.py")


def consumed_source_members():
    root = Path(__file__).resolve().parents[2] / "evaluation"
    names = ("factual.py", "reduce.py", "w0_reference.py")
    return {**{name:member(root / name) for name in names},
            **{"generation/" + name:member(root / "generation" / name) for name in GENERATION_CLOSURE}}


def content_member(value):
    require(type(value.get("bytes")) is int and value["bytes"] > 0 and value.get("sha256"),
            "SHARED_W0_CONSUMED_CONTENT_SHA_REQUIRED")
    return dict(bytes=value["bytes"], sha256=value["sha256"])


def build_fingerprint(assets, tokenizer, records_by_dataset):
    """Token planning only: zero model forward, generation, transfer or fitting."""
    from official.evaluation.w0_reference import computational_fingerprint
    import nltk
    root = Path(__file__).resolve().parents[2] / "evaluation"
    datasets = {}
    for name in ("cf", "zsre"):
        records = records_by_dataset[name]
        require(len(records) == 2000 and [r["occurrence_index"] for r in records] == list(range(1, 2001)),
                "SHARED_W0_EXACT_ORDERED_STREAM")
        signatures = _plan(records, name, tokenizer)[2]
        datasets[name] = dict(stream_sha256=assets["streams"][name]["stream_sha256"],
            ordered_occurrences_sha256=_digest(list(range(1, 2001))),
            ordered_queries_sha256=_digest(signatures), requests=2000)
    require(tokenizer.padding_side == "right", "SHARED_W0_RIGHT_PADDING_REQUIRED")
    resources, sources = {}, {}
    ref = assets["generation_reference"]
    resource_root = Path(ref["nltk_data_root"]).resolve()
    nltk_root = Path(nltk.__file__).resolve().parent
    for value in ref["tokenizer_resources_and_sources"]:
        path = Path(value["path"]).resolve()
        if path.is_relative_to(resource_root):
            resources[str(path.relative_to(resource_root))] = content_member(value)
        else:
            require(path.is_relative_to(nltk_root), "SHARED_W0_NLTK_CONSUMED_SOURCE_SCOPE")
            sources[str(path.relative_to(nltk_root))] = content_member(value)
    reference_manifest = read(ref["manifest"]["path"])
    content = dict(
        model=dict(model_id=assets["model"]["identity"]["model_id"],
            revision=assets["model"]["identity"]["revision"],
            config_sha256=assets["model"]["config"]["sha256"],
            weights={Path(v["path"]).name: content_member(v) for v in assets["model"]["weights"]}),
        tokenizer=dict(files={name: content_member(v) for name,v in assets["model"]["tokenizer_files"].items()},
            settings=dict(prompt_add_special_tokens=True, target_add_special_tokens=False,
                          padding="right", target_prefix=" ", lookup="NATIVE_CF_VERIFIED_BOUNDARY_ZSRE_EXACT_TOKEN_PREFIX_NO_TARGET_BOS")),
        datasets=datasets,
        sources=dict(factual=file_sha(root / "factual.py"), reduce=file_sha(root / "reduce.py"),
            w0_reference=file_sha(root / "w0_reference.py"),
            generation={name:file_sha(root / "generation" / name) for name in GENERATION_CLOSURE}),
        runtime=dict(versions=dict(python=platform.python_version(),
            **{name:importlib.metadata.version(package) for name,package in
               (("torch","torch"),("transformers","transformers"),("numpy","numpy"),
                ("scipy","scipy"),("sklearn","scikit-learn"),("nltk","nltk"))}),
            numeric=dict(weights_dtype="float32", attention="eager", matmul_tf32=False,
                cudnn_tf32=False, autocast=False, factual_use_cache=False, generation_use_cache=True)),
        generation=dict(profile=PROFILE, seed=20261007, case_batching="ALL_GENERATION_PROMPTS",
            sampling_scope="ENDPOINT_GLOBAL_BATCH_STREAM", top_k=5, max_total_tokens=100,
            n_gen_per_prompt=1, EOS_stop=False, decode=SOURCE["decode"]),
        references=dict(identity_sha256=ref["identity_sha256"],
            files={name:content_member(v) for name,v in ref["files"].items()},
            nltk=dict(resources=resources, sources=sources),
            vectorizer=reference_manifest["fixed_vectorizer"]["class_name"] + ":PUBLIC_IDF_SETTER_NO_FIT"))
    return computational_fingerprint(content)


def execution_identity(config, lock, assets, *, output, role="GPU_PRODUCER"):
    """Actual producer/consumer execution, deliberately not fingerprint equality."""
    import os
    import torch
    require(role in ("GPU_PRODUCER","GPU_CONSUMER","QUALIFICATION","CPU_REDUCER"), "SHARED_W0_EXECUTION_ROLE")
    identity = dict(server="server1", role=role, source=lock["source"], config_sha256=config["config_sha256"],
        assets_manifest=config["assets_member"], runtime=assets["runtime"],
        input_stream_bundle=config["stream_bundle_member"], output=str(Path(output).absolute()),
        base_W0_input=(None if config.get("projected_CF_addition") is True else config["base_W0_output"]),
        hardware=dict(device="CPU" if role == "CPU_REDUCER" else torch.cuda.get_device_name(0),
            capability=[] if role == "CPU_REDUCER" else list(torch.cuda.get_device_capability(0)), cross_hardware_bitwise_claim=False))
    # Whitelist metadata only. No entire environment or credential is read.
    identity["slurm"] = {name:os.environ[name] for name in
        ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_STEP_ID")
        if name in os.environ and not (name == "SLURM_STEP_ID" and os.environ[name] == "")}
    require(bool(identity["slurm"].get("SLURM_JOB_ID")), "SHARED_W0_ACTUAL_SLURM_ID_REQUIRED")
    # Reuse the official identity validator; do not fork logger semantics.
    from official.tracking.schema import SLURM_ENV, job_identity
    raw = {key:identity["slurm"][name] for key,name in SLURM_ENV.items()
           if name in identity["slurm"] and not (key == "step_id" and identity["slurm"][name] == "")}
    raw.update(execution_backend="slurm", identity_source="SLURM_ENV")
    raw["job_display_id"] = (raw["array_job_id"] + "_" + raw.get("array_task_id", "")
                             if "array_job_id" in raw else raw["job_id"])
    job_identity(raw)
    return identity
