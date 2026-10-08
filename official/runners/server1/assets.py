"""Server1's read-only, fail-closed Llama asset binding.

EasyEdit is an asset directory, never an algorithm import. Large model hashes
are optional and their absence is explicit; C0 validation streams the stored
uncentered sum without constructing a model, covariance, or projector.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import zipfile

from official.experiments.prepare import ROOT, digest, file_sha, load_plan, normalize_records, write_new


SCHEMA = "official-server1-assets-v1"
PROJECT = Path("/mnt/raid5/janghj/ODE-edit")
EASYEDIT = Path("/mnt/raid5/janghj/EasyEdit")
MODEL_REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
MODEL_SNAPSHOT = Path("/mnt/raid5/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots") / MODEL_REVISION
DEFAULT_REFERENCE = PROJECT / "local/baseline-generation-eval-assets/20261007/reference-ready-r1/manifest.json"
DEFAULT_OUTPUT = PROJECT / "local/official-baselines/server1/assets-r1.json"


class AssetBindingError(RuntimeError):
    def __init__(self, code, path=None, detail=None):
        self.code, self.path, self.detail = code, str(path) if path is not None else None, detail
        super().__init__(code + (": " + self.path if self.path else ""))

    def receipt(self):
        return dict(code=self.code, path=self.path, detail=self.detail,
                    status="BLOCKED_ASSET_IDENTITY", GPU_qualification=False)


def require(condition, code, path=None, detail=None):
    if not condition:
        raise AssetBindingError(code, path, detail)


def json_read(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as error:
        raise AssetBindingError("ASSET_JSON_UNAVAILABLE", path, type(error).__name__) from error


def member(path, *, hash_bytes=True, expected_sha=None, expected_bytes=None, allow_symlink=False):
    """Bind a single explicit path, never walk a directory or follow unknown links."""
    path = Path(path).absolute()
    require(path.is_file(), "ASSET_MISSING", path)
    require(allow_symlink or not path.is_symlink(), "ASSET_UNEXPECTED_SYMLINK", path)
    before = path.stat()
    require(expected_bytes is None or before.st_size == expected_bytes, "ASSET_SIZE_MISMATCH", path)
    actual_sha = file_sha(path) if hash_bytes else None
    after = path.stat()
    key = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    require(key(before) == key(after), "ASSET_CHANGED_DURING_BINDING", path)
    if expected_sha is not None and hash_bytes:
        require(actual_sha == expected_sha, "ASSET_SHA_MISMATCH", path)
    return dict(path=str(path), real_path=str(path.resolve()), bytes=after.st_size,
                allocated_bytes=after.st_blocks * 512, device=after.st_dev, inode=after.st_ino,
                nlink=after.st_nlink, mtime_ns=after.st_mtime_ns, ctime_ns=after.st_ctime_ns,
                sha256=actual_sha, expected_sha256=expected_sha,
                verification="FRESH_SHA256" if hash_bytes else "CURRENT_STAT_ONLY_NOT_SHA_VERIFIED")


def validate_c0(path, width, *, expected_sample_size=100000):
    """Stream native NPZ sum/count with allow_pickle=False, <=8MiB workspace.

    The native serialization calls its SUM ``mom2.mom2``; ``moment()`` divides
    this sum by the actual masked vector count. No SVD/PSD claim is made.
    """
    import numpy as np

    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == {"mom2.constructor", "mom2.count", "mom2.mom2", "sample_size"},
                    "C0_NATIVE_SCHEMA", path)
            count = archive["mom2.count"]
            samples = archive["sample_size"]
            require(count.ndim == samples.ndim == 0 and count.dtype.kind in "iu"
                    and samples.dtype.kind in "iu", "C0_INTEGER_SCALARS", path)
            count, samples = int(count), int(samples)
            require(count > 0 and samples == expected_sample_size, "C0_COUNT_OR_SAMPLE_SIZE", path)
            constructor = str(archive["mom2.constructor"].item())
            require(constructor.endswith(".SecondMoment()"), "C0_CONSTRUCTOR", path)
        finite, minimum, maximum, elements = True, float("inf"), float("-inf"), 0
        diagonal_minimum = float("inf")
        with zipfile.ZipFile(path) as archive:
            require(len(archive.namelist()) == len(set(archive.namelist())), "C0_DUPLICATE_NPZ_MEMBER", path)
            info = archive.getinfo("mom2.mom2.npy")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), "C0_NPY_VERSION", path)
                header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)(stream)
                shape, fortran_order, dtype = header
                require(shape == (width, width) and dtype == np.dtype("float32") and not fortran_order,
                        "C0_SHAPE_DTYPE_OR_LAYOUT", path,
                        dict(shape=list(shape), dtype=str(dtype), fortran_order=fortran_order))
                while True:
                    payload = stream.read(8 << 20)
                    if not payload:
                        break
                    require(len(payload) % dtype.itemsize == 0, "C0_TRUNCATED_ARRAY", path)
                    block = np.frombuffer(payload, dtype=dtype)
                    finite = finite and bool(np.isfinite(block).all())
                    minimum, maximum = min(minimum, float(block.min())), max(maximum, float(block.max()))
                    first_diag = (-elements) % (width + 1)
                    if first_diag < len(block):
                        diagonal_minimum = min(diagonal_minimum, float(block[first_diag::width + 1].min()))
                    elements += len(block)
        require(elements == width * width and finite, "C0_NONFINITE_OR_TRUNCATED", path)
        require(diagonal_minimum >= 0, "C0_NEGATIVE_DIAGONAL", path)
    except AssetBindingError:
        raise
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise AssetBindingError("C0_SAFE_NPZ_READ_FAILED", path, type(error).__name__) from error
    return dict(shape=[width, width], dtype="float32", stored_field="mom2.mom2",
                stored_value="UNCENTERED_SECOND_MOMENT_SUM", normalization="sum / masked_token_vector_count",
                masked_token_vector_count=count, sample_documents=samples,
                finite=True, diagonal_nonnegative=True, min_stored_sum=minimum,
                max_stored_sum=maximum, min_stored_diagonal=diagonal_minimum,
                constructor=constructor, validation="STREAMED_FULL_SUM_FINITE_SHAPE_COUNT",
                PSD_eigenvalidation=False, centered_covariance=False, recomputed=False)


def bind_stream(path, dataset):
    lock = json_read(ROOT / "hparams" / f"{dataset}-stream.lock.json")
    source = member(path, expected_sha=lock["source_sha256"])
    records = normalize_records(json_read(path), dataset)
    # write_new's exact bytes are the public normalized-stream byte contract.
    normalized = (json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    stream_sha = hashlib.sha256(normalized).hexdigest()
    ordered_sha = digest([record["case_id"] for record in records])
    require(stream_sha == lock["stream_sha256"] and ordered_sha == lock["ordered_case_ids_sha256"],
            "STREAM_NORMALIZATION_OR_ORDER_MISMATCH", path)
    for batch in lock["batches"]:
        selected = records[batch["start_inclusive"]:batch["end_exclusive"]]
        require(len(selected) == 100 and digest([row["case_id"] for row in selected]) == batch["ordered_case_ids_sha256"],
                "STREAM_BATCH_LOCK_MISMATCH", path)
    return dict(source=source, dataset=dataset, requests=len(records), batch_size=100, batches=20,
                stream_sha256=stream_sha, ordered_case_ids_sha256=ordered_sha,
                stream_lock=member(ROOT / "hparams" / f"{dataset}-stream.lock.json"),
                order=lock["order"], target_mapping=lock["target_mapping"], raw_in_Git=False)


def bind_model(snapshot, *, hash_model=False):
    contract, _ = load_plan()
    identity = contract["models"]["llama3"]
    snapshot = Path(snapshot).absolute()
    require(snapshot.name == identity["revision"], "MODEL_REVISION_PATH_MISMATCH", snapshot)
    config = json_read(snapshot / "config.json")
    require((config.get("model_type"), config.get("hidden_size"), config.get("intermediate_size"),
             config.get("num_hidden_layers")) == ("llama", 4096, 14336, 32), "MODEL_CONFIG_SHAPE", snapshot)
    expected_tokenizer = json_read(ROOT / "hparams/tokenizers.lock.json")["audits"]["llama3-cf"]
    tokenizer_members = {name: member(snapshot / name, expected_sha=value, allow_symlink=True)
                         for name, value in expected_tokenizer["tokenizer_files_sha256"].items()}
    observed = {name: row["sha256"] for name, row in tokenizer_members.items()}
    require(digest(observed) == expected_tokenizer["tokenizer_sha256"], "TOKENIZER_LOCK_MISMATCH", snapshot)
    index = json_read(snapshot / "model.safetensors.index.json")
    shard_names = sorted(set(index["weight_map"].values()))
    require(shard_names and all(re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name) for name in shard_names),
            "MODEL_SHARD_INDEX", snapshot)
    shards = []
    for name in shard_names:
        target = (snapshot / name).resolve()
        # Known HF snapshot links resolve only to the same model repository's blobs.
        blob_root = snapshot.parent.parent / "blobs"
        require(target.parent == blob_root and re.fullmatch(r"[a-f0-9]{64}", target.name),
                "MODEL_SHARD_UNEXPECTED_TARGET", target)
        row = member(snapshot / name, hash_bytes=hash_model, expected_sha=target.name, allow_symlink=True)
        row.update(hf_content_address_sha256=target.name,
                   current_payload_SHA_verified=hash_model,
                   provenance="PINNED_HF_SNAPSHOT_CONTENT_ADDRESSED_BLOB_NOT_UPSTREAM_MODEL_EQUIVALENCE")
        shards.append(row)
    return dict(model="llama3", identity=identity, snapshot=str(snapshot), tokenizer_path=str(snapshot),
                config=member(snapshot / "config.json", allow_symlink=True),
                index=member(snapshot / "model.safetensors.index.json", allow_symlink=True),
                weights=shards, tokenizer_files=tokenizer_members,
                tokenizer_sha256=expected_tokenizer["tokenizer_sha256"],
                actual_model_loaded=False, GPU_qualification=False,
                payload_verification="FRESH_FULL_SHA256" if hash_model else "PINNED_CONTENT_ID_AND_FRESH_STAT_ONLY")


def bind_generation(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json_read(manifest_path)
    ready = json_read(manifest_path.parent / "READY.json")
    lock = json_read(ROOT / "hparams/generation.lock.json")
    expected_identity = lock["reference_identity_sha256"]
    require(manifest.get("identity_sha256") == ready.get("identity_sha256") == expected_identity
            and manifest.get("status") == ready.get("status") == "READY", "REFERENCE_READY_IDENTITY", manifest_path)
    manifest_member = member(manifest_path, expected_sha=ready["manifest"]["sha256"],
                             expected_bytes=ready["manifest"]["bytes"])
    refs = {}
    for name, expected in lock["reference_files"].items():
        original = manifest["files"][name]
        require(original["sha256"] == expected["sha256"] and original["bytes"] == expected["bytes"],
                "REFERENCE_LOCK_CHANGED", original["path"])
        refs[name] = member(original["path"], expected_sha=expected["sha256"], expected_bytes=expected["bytes"])
    tokenizer = manifest["tokenizer"]
    tokenizer_members = [member(row["path"], expected_sha=row["sha256"], expected_bytes=row["bytes"])
                         for row in tokenizer["required_resources"] + tokenizer["source_members"]]
    versions = {name: importlib.metadata.version(dist) for name, dist in
                (("numpy", "numpy"), ("scipy", "scipy"), ("sklearn", "scikit-learn"), ("nltk", "nltk"))}
    require(versions == manifest["versions"], "GENERATION_RUNTIME_VERSION_MISMATCH", manifest_path,
            dict(expected=manifest["versions"], actual=versions))
    return dict(manifest=manifest_member, READY=member(manifest_path.parent / "READY.json"),
                manifest_path=str(manifest_path), identity_sha256=expected_identity,
                files=refs, tokenizer_resources_and_sources=tokenizer_members, versions=versions,
                nltk_data_root=str(Path(tokenizer["required_resources"][0]["path"]).parents[3]),
                reference_refit=False, downloads=False, observed_model_generation=False)


def build_manifest(*, easyedit_root=EASYEDIT, model_snapshot=MODEL_SNAPSHOT,
                   cf_source=PROJECT / "local/datasets/counterfact-fixed-10k-v1/counterfact.json",
                   zsre_source=EASYEDIT / "data/zsre/zsre_mend_eval.json",
                   generation_reference_manifest=DEFAULT_REFERENCE, validate_stats=True, hash_model=False):
    """Return a fresh actual manifest; ``ready_to_run`` never certifies GPU parity."""
    contract, _ = load_plan()
    easyedit_root = Path(easyedit_root).absolute()
    require(easyedit_root.is_dir(), "EASYEDIT_ASSET_ROOT_MISSING", easyedit_root)
    model = bind_model(model_snapshot, hash_model=hash_model)
    stats_root = easyedit_root / "examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats"
    C0 = {}
    for layer in contract["models"]["llama3"]["layers"]:
        module = f"model.layers.{layer}.mlp.down_proj"
        path = stats_root / f"{module}_float32_mom2_100000.npz"
        row = member(path)
        row.update(layer=layer, module=module, native_cache_name=path.name, logical_model_basename="Meta-Llama-3-8B-Instruct")
        if validate_stats:
            row["tensor_validation"] = validate_c0(path, 14336)
        else:
            row["tensor_validation"] = dict(validation="NOT_PERFORMED", GPU_qualification=False)
        C0[str(layer)] = row
    if validate_stats:
        require(len({row["tensor_validation"]["masked_token_vector_count"] for row in C0.values()}) == 1,
                "C0_LAYERS_MASKED_COUNT_MISMATCH", stats_root)
    P_path = easyedit_root / "examples/null_space_project_Meta-Llama-3-8B-Instruct.pt"
    projector = dict(path=str(P_path), exists=P_path.is_file(), required_for_assigned_methods=False,
                     protection="PROTECTED_EXISTING_ASSET_NOT_LOADED_OR_RECOMPUTED",
                     physical_layers=[4, 5, 6, 7, 8], shape_not_reverified=True)
    if P_path.is_file():
        projector["member"] = member(P_path, hash_bytes=False)
    source_files = ["SOURCES.json", "hparams/contract.json", "hparams/sources.lock.json",
                    "hparams/profiles.json", "hparams/tokenizers.lock.json", "hparams/generation.lock.json",
                    "baselines/registry.py", "baselines/sphere/util/runningstats.py",
                    "baselines/easyedit/util/runningstats.py"]
    source_files += [f"hparams/{method}/llama3.json" for method in ("FT", "MEMIT", "MEMIT_FE")]
    result = dict(schema=SCHEMA, server="server1", assignment=dict(model="llama3", methods=["FT", "MEMIT", "MEMIT_FE"],
                  datasets=["cf", "zsre"]), easyedit_root=str(easyedit_root), model=model,
                  streams={"cf": bind_stream(cf_source, "cf"), "zsre": bind_stream(zsre_source, "zsre")},
                  C0=C0, stats_root=str(stats_root.parent.parent), protected_projector=projector,
                  generation_reference=bind_generation(generation_reference_manifest),
                  runtime=dict(python=str(Path(sys.executable).absolute()), python_version=platform.python_version(),
                      dependency_versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}),
                  official_source_members={name: member(ROOT / name) for name in source_files},
                  identity_policy="EXACT_OFFICIAL_LOCKS_NO_ASSET_RECOMPUTE_OR_DOWNLOAD",
                  GPU_qualification=False, model_forward_calls=0, EasyEdit_code_imported=False,
                  copied_assets=False, created_checkpoint=False, ready_to_run=False,
                  remaining=["exact main commit/official tree freeze", "native GPU factual/parity/resume qualification"])
    result["assets_sha256"] = digest(result)
    return result


def verify_manifest(manifest, *, rehash_large=False, verify_preparation_source=True):
    """Check a sealed asset manifest without rebinding historical source provenance.

    The default also rechecks the original preparation source paths. A frozen
    runner may pass False only AFTER its independent source binding verifies
    every member in its immutable execution archive. That narrowly excludes
    ``official_source_members``: their old hashes remain immutable provenance,
    not a claim that patched/frozen execution code has identical source bytes.
    Model, C0, reference, tokenizer, and dataset/stream locks remain checked.
    """
    require(type(verify_preparation_source) is bool, "ASSET_SOURCE_VERIFICATION_FLAG_TYPE")
    require(manifest.get("schema") == SCHEMA and manifest.get("server") == "server1", "ASSET_MANIFEST_SCOPE")
    unsigned = dict(manifest)
    expected = unsigned.pop("assets_sha256")
    require(digest(unsigned) == expected, "ASSET_MANIFEST_DIGEST")

    def visit(value):
        if isinstance(value, dict):
            if "path" in value and "bytes" in value and "verification" in value:
                path = Path(value["path"])
                actual = member(path, hash_bytes=rehash_large or value["bytes"] < (32 << 20), allow_symlink=True)
                for key in ("real_path", "bytes", "device", "inode", "mtime_ns", "ctime_ns"):
                    require(actual[key] == value[key], "ASSET_CURRENT_STAT_CHANGED", path)
                if actual["sha256"] is not None and value.get("sha256") is not None:
                    require(actual["sha256"] == value["sha256"], "ASSET_CURRENT_SHA_CHANGED", path)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    for key, value in manifest.items():
        if key == "official_source_members" and not verify_preparation_source:
            continue
        visit(value)
    return dict(status="ASSET_MANIFEST_CURRENT_STAT_VERIFIED", assets_sha256=expected,
                large_bytes_rehashed=rehash_large, GPU_qualification=False,
                preparation_source_paths_verified=verify_preparation_source,
                historical_preparation_source_members_preserved=True,
                actual_execution_source_equivalence_claimed=False,
                separately_verified_frozen_source_required=not verify_preparation_source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paths", type=Path, help="Explicit small path override JSON (no directory scan)")
    parser.add_argument("--hash-model", action="store_true")
    parser.add_argument("--metadata-only", action="store_true", help="No C0 content certification")
    args = parser.parse_args()
    try:
        paths = json_read(args.paths) if args.paths else {}
        allowed = {"easyedit_root", "model_snapshot", "cf_source", "zsre_source", "generation_reference_manifest"}
        require(set(paths) <= allowed, "ASSET_PATH_OVERRIDE_KEYS")
        result = build_manifest(**paths, validate_stats=not args.metadata_only, hash_model=args.hash_model)
        write_new(args.output, result)
        print(json.dumps(dict(status="ACTUAL_ASSET_BINDING_PREPARED_NOT_GPU_QUALIFIED", path=str(args.output),
                             assets_sha256=result["assets_sha256"], model_payload=result["model"]["payload_verification"],
                             C0_layers=list(result["C0"]), ready_to_run=False), indent=2))
    except AssetBindingError as error:
        write_new(args.output.with_suffix(".error.json"), error.receipt())
        print(json.dumps(error.receipt(), indent=2))
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
