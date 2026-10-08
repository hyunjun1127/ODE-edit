"""Inspect existing EasyEdit assets without importing or modifying EasyEdit code."""
import argparse
import json
from pathlib import Path

from official.experiments.prepare import ROOT, file_sha, load_plan, write_new


def inventory(easyedit_root, model, *, reference_manifest=None, hash_large=False):
    contract, _ = load_plan()
    identity = contract["models"][model]
    root = Path(easyedit_root).resolve()
    basename = identity["model_id"].rsplit("/", 1)[1]
    stats_root = root / "examples/data/stats" / basename / "wikipedia_stats"
    module = "transformer.h.{}.mlp.fc_out" if model == "gptj" else "model.layers.{}.mlp.down_proj"
    paths = {f"C0_L{layer}": stats_root / (module.format(layer) + "_float32_mom2_100000.npz")
             for layer in identity["layers"]}
    paths["projector"] = root / "examples" / f"null_space_project_{basename}.pt"
    result = dict(model=model, identity=identity, easyedit_root=str(root), assets={},
                  EasyEdit_code_imported=False, copied_assets=False, model_forward_calls=0)
    for key, path in paths.items():
        member = dict(path=str(path), exists=path.is_file())
        if path.is_file():
            member.update(bytes=path.stat().st_size,
                          sha256=file_sha(path) if hash_large else None,
                          verification="SHA256" if hash_large else "EXISTENCE_ONLY")
        result["assets"][key] = member
    if reference_manifest:
        manifest_path = Path(reference_manifest)
        manifest = json.loads(manifest_path.read_text())
        expected = json.loads((ROOT / "hparams/generation.lock.json").read_text())
        if manifest["identity_sha256"] != expected["reference_identity_sha256"]:
            raise ValueError("REFERENCE_ASSET_IDENTITY_CHANGED")
        refs = {}
        for name, locked in expected["reference_files"].items():
            actual = manifest["files"][name]
            path = Path(actual["path"])
            if actual["sha256"] != locked["sha256"] or actual["bytes"] != locked["bytes"]:
                raise ValueError("REFERENCE_ASSET_LOCK_MISMATCH")
            if not path.is_file() or path.stat().st_size != locked["bytes"]:
                raise ValueError("REFERENCE_ASSET_MISSING")
            if hash_large and file_sha(path) != locked["sha256"]:
                raise ValueError("REFERENCE_ASSET_BYTES_CHANGED")
            refs[name] = dict(actual, verification="SHA256" if hash_large else "SEALED_MANIFEST_PLUS_SIZE")
        result["generation_reference"] = dict(manifest_path=str(manifest_path),
            manifest_sha256=file_sha(manifest_path), files=refs,
            identity_sha256=manifest["identity_sha256"])
    result["ready_to_submit"] = False
    result["remaining"] = ["model/tokenizer binding", "C0/projector provenance and tensor validation",
                           "native edit/evaluator/resume GPU smoke"]
    return result


def main(server=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--easyedit-root", type=Path, required=True)
    parser.add_argument("--model", choices=("llama3", "qwen25", "gptj"), required=True)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--hash-large", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.easyedit_root, args.model,
                       reference_manifest=args.reference_manifest, hash_large=args.hash_large)
    result["server"] = server
    write_new(args.output, result)
    print(json.dumps(dict(server=server, model=args.model,
                         asset_files=len(result["assets"]),
                         available=sum(x["exists"] for x in result["assets"].values()),
                         ready_to_submit=False), indent=2))


if __name__ == "__main__":
    main()
