"""Bind rent assets, streams, tokenizer and frozen source into one manifest per model.

Runs in a CPU-only Pod from the frozen source directory (``freeze.py``). Every
large file is hashed and compared with the reference locks; nothing is
downloaded, copied or recomputed.
"""
import argparse
import os
from pathlib import Path
import platform

from official.experiments.prepare import ROOT, digest, file_sha, prepare_stream, read, write_new
from official.runners.rent.freeze import tree
from official.runners.rent.native import model_spec

INSTRUCTION = "USER-OFFICIAL-BASELINES-20261008-R1"
WORKSPACE = Path("/workspace")
HUB = WORKSPACE / "cache/huggingface/hub"
EASYEDIT = WORKSPACE / "data/EasyEdit"
SOURCES = dict(cf=WORKSPACE / "data/datasets/counterfact-fixed-10k-v1/counterfact.json",
               zsre=EASYEDIT / "data/zsre/zsre_mend_eval.json")
WANDB_ENV = WORKSPACE / "code/ODE-edit/servers/local/wandb.env"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "vocab.json", "merges.txt", "added_tokens.json")
RUN_ROOT = Path(__file__).resolve().parents[4]


def require(value, code):
    if not value:
        raise ValueError(code)


def source(run_root):
    value = read(run_root / "source.json")
    digest_, members = tree(run_root / "source")
    require(digest_ == value["official_tree_sha256"] and members == value["members"]
            and Path(__file__).resolve().is_relative_to((run_root / "source").resolve()),
            "FROZEN_SOURCE_IDENTITY")
    return value


def streams(run_root):
    folder, result = run_root / "streams", {}
    for dataset, path in SOURCES.items():
        lock = folder / f"{dataset}-stream.lock.json"
        if not lock.exists():
            prepare_stream(path, dataset, folder)
        receipt, expected = read(lock), read(ROOT / "hparams" / f"{dataset}-stream.lock.json")
        require(receipt == expected, "STREAM_LOCK_MISMATCH: " + dataset)
        stream = folder / f"{dataset}-stream.json"
        require(file_sha(stream) == expected["stream_sha256"], "STREAM_BYTES_CHANGED: " + dataset)
        result[dataset] = dict(path=str(stream), lock=receipt)
    return result


def model_files(snapshot):
    files = {}
    for path in sorted(snapshot.iterdir()):
        real = path.resolve()
        sha = file_sha(real)
        # Hub LFS blobs are named by their SHA256; small files by git SHA1.
        require(len(real.name) != 64 or real.name == sha, "HF_LFS_BLOB_SHA_MISMATCH: " + path.name)
        files[path.name] = dict(bytes=real.stat().st_size, sha256=sha)
    return files


def assets(model):
    import numpy as np
    import torch
    from official.baselines import registry
    spec = model_spec(model)
    reference = read(Path(__file__).with_name("assets.lock.json"))["models"][model]
    snapshot = HUB / ("models--" + spec["model_id"].replace("/", "--")) / "snapshots" / spec["revision"]
    files = model_files(snapshot)
    identity = {name: files[name]["sha256"] for name in TOKENIZER_FILES if name in files}
    tokenizer_sha256 = digest(identity)
    locks = read(ROOT / "hparams/tokenizers.lock.json")["audits"]
    require(all(locks[f"{model}-{d}"]["tokenizer_sha256"] == tokenizer_sha256 for d in ("cf", "zsre")),
            "TOKENIZER_LOCK_MISMATCH")
    stats_root = EASYEDIT / "examples/data/stats" / spec["basename"] / "wikipedia_stats"
    module = registry.hparams("MEMIT", model).rewrite_module_tmp
    members = {}
    for layer in spec["layers"]:
        path = stats_root / f"{module.format(layer)}_float32_mom2_100000.npz"
        sha = file_sha(path)
        require(sha == reference[str(path.relative_to(EASYEDIT))], f"C0_REFERENCE_SHA: {layer}")
        with np.load(path, allow_pickle=False) as archive:
            count, samples, total = int(archive["mom2.count"]), int(archive["sample_size"]), archive["mom2.mom2"]
            require(total.dtype == np.float32 and total.shape == (spec["key_width"],) * 2
                    and samples == 100000 and count > 0, f"C0_PAYLOAD: {layer}")
        members[f"C0_L{layer}"] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha,
                                       count=count, sample_size=samples, shape=list(total.shape))
    path = EASYEDIT / "examples" / f"null_space_project_{spec['basename']}.pt"
    sha = file_sha(path)
    require(sha == reference[str(path.relative_to(EASYEDIT))], "PROJECTOR_REFERENCE_SHA")
    projector = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    shape = [len(spec["layers"]), spec["key_width"], spec["key_width"]]
    require(projector.dtype == torch.float32 and list(projector.shape) == shape, "PROJECTOR_SHAPE_DTYPE")
    members["projector"] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha, shape=shape,
        slot_layers=list(spec["layers"]),
        threshold=registry.hparams("ALPHAEDIT", model).nullspace_threshold,
        threshold_provenance="DECLARED_BY_PINNED_HPARAMS; BYTES_MATCH_SERVER1_REFERENCE")
    return dict(model_snapshot=str(snapshot), model_revision=spec["revision"], model_files=files,
                tokenizer_sha256=tokenizer_sha256, stats_root=str(stats_root), assets=members,
                assets_identity_sha256=digest({k: v["sha256"] for k, v in members.items()}))


def prepare(model, run_root=RUN_ROOT):
    import torch
    import transformers
    frozen = source(run_root)
    manifest = dict(schema="official-rent-execution-v1", instruction_id=INSTRUCTION, server="rent",
        model=model, run_root=str(run_root), code_commit=frozen["code_commit"],
        official_tree_sha256=frozen["official_tree_sha256"], on_main=frozen["on_main"],
        streams=streams(run_root), runtime=dict(python=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__),
        tracking=dict(namespace="official.tracking", env_file=str(WANDB_ENV),
            source_sha256=file_sha(ROOT / "tracking/__init__.py")),
        generation_schedule="DEFERRED_CHECKPOINT_EVALUATION",
        prepared_by_job=os.environ.get("ODEEDIT_RENT_JOB_ID"), **assets(model))
    path = run_root / f"manifest-{model}.json"
    write_new(path, manifest)
    return dict(manifest=str(path), sha256=file_sha(path), model=model, on_main=frozen["on_main"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("llama3", "qwen25", "gptj"), required=True)
    args = parser.parse_args()
    for model in args.models:
        print(prepare(model), flush=True)


if __name__ == "__main__":
    main()
