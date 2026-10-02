"""CPU tokenizer/arithmetic audit of v4 input work; never loads model weights.

The current worktree's preparation function and all twenty sealed v3 packing
identities are checked. Only counts, identities and source provenance are
saved; no prompt text, token arrays, model state, inference or GPU timings.

This audits the frozen BS100 x 20 Llama/CounterFact instance; it is not a
general method runner. On another host supply --model, --dataset, --contexts
and a fresh --output path. The optional original snapshot adds a second
packing identity binding when present; the repository's twenty sealed v3
identities are mandatory on every host.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
METHOD = HERE.parent
ROOT = HERE.parents[3]
DEFAULT_PRIOR_CONFIG = Path("/mnt/raid5/janghj/.codex/worktrees/odeedit-jlz-strength-review-20261002-v1/local/jlz-strength-review/snapshot/config.json")


def member(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def padded_slots(lengths, size):
    return sum(len(lengths[i:i + size]) * max(lengths[i:i + size])
               for i in range(0, len(lengths), size))


def build(*, model_path=None, dataset_path=None, contexts_path=None,
          prior_config_path=DEFAULT_PRIOR_CONFIG):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(ROOT))
    import torch
    import transformers
    from transformers import AutoTokenizer

    prompts = importlib.import_module("project.run_scripts.jlz_pilot.prompts")
    expected_source = ROOT / "project/run_scripts/jlz_pilot/prompts.py"
    assert Path(prompts.__file__).resolve() == expected_source.resolve(), "CURRENT_WORKTREE_IMPORT_REQUIRED"
    torch.set_num_threads(1)
    input_path = METHOD / "inputs/exact-native-inputs.json"
    inputs = json.loads(input_path.read_text())
    prior_path = ROOT / "plans/global/2026-10-02-jlz-shared-subject-v3/compute/native-prefix-counts.json"
    prior = json.loads(prior_path.read_text())
    config_path = Path(prior_config_path) if prior_config_path is not None else None
    config = json.loads(config_path.read_text()) if config_path is not None and config_path.is_file() else None
    if model_path is not None:
        model = Path(model_path)
    elif config is not None:
        model = Path(config["model"])
        if not model.is_dir():
            model = Path(str(model).replace("/data/janghj/", "/mnt/raid5/janghj/", 1))
    else:
        raise ValueError("--model is required when the optional prior snapshot config is unavailable")
    stream = Path(dataset_path) if dataset_path is not None else Path(inputs["dataset"]["local_path"])
    context_path = Path(contexts_path) if contexts_path is not None else Path(inputs["contexts"]["local_path"])
    assert member(stream)["sha256"] == inputs["dataset"]["sha256"] == prior["stream_sha256"]
    assert member(context_path)["sha256"] == inputs["contexts"]["sha256"] == prior["context_sha256"]
    assert member(expected_source)["sha256"] == prior["preparation_source_sha256"]
    records = json.loads(stream.read_text())
    contexts = json.loads(context_path.read_text())
    assert digest([r["case_id"] for r in records[:2000]]) == inputs["ordered_case_ids_sha256"]
    assert list(map(len, contexts)) == [1, 5]
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tok.padding_side = "right"
    tok.pad_token = tok.eos_token
    mc = json.loads((model / "config.json").read_text())
    width, hidden, layers = mc["intermediate_size"], mc["hidden_size"], 5
    kv_heads = mc["num_key_value_heads"]
    head_dim = hidden // mc["num_attention_heads"]
    suffix_layers = mc["num_hidden_layers"] - 5
    kv_bytes_per_token = suffix_layers * 2 * kv_heads * head_dim * 4
    batches = []
    for batch in range(1, 21):
        subset = records[(batch - 1) * 100:batch * 100]
        requests = [r["requested_rewrite"] | {"case_id": r["case_id"]} for r in subset]
        sp = prompts.prepare(tok, requests, contexts, "cpu")
        old = next(r for r in prior["batches"] if r["batch"] == batch)
        assert sp["identity"] == old["sealed_token_identity"], "REPOSITORY_PACKING_IDENTITY"
        if config is not None:
            sealed = next(r for r in config["packing"] if r["phase"] == "main" and r["batch"] == batch)
            assert sp["identity"] == sealed["identity"], "OPTIONAL_SNAPSHOT_PACKING_IDENTITY"
        lengths = [int(n) for n in sp["tokens"]["attention_mask"].sum(1).tolist()]
        starts = sp["lookup"]
        assert len(lengths) == 700 and all(0 <= s < n for s, n in zip(starts, lengths))
        assert sp["tokens"]["input_ids"].device.type == "cpu"
        matches = 0
        for key_row, rewrite_row in enumerate(sp["rw_rows"]):
            start, key_start = starts[rewrite_row], sp["key_lookup"][key_row]
            positions = sp["targets"][rewrite_row].ne(-100).nonzero().flatten().tolist()
            assert positions and min(positions) >= start
            equal = (start == key_start and
                     torch.equal(sp["tokens"]["input_ids"][rewrite_row, :start + 1],
                                 sp["key_tokens"]["input_ids"][key_row, :key_start + 1]))
            assert equal, "TEACHER_FORCED_VS_TARGET_FREE_SUBJECT_PREFIX_MISMATCH"
            matches += int(equal)
        assert matches == len(sp["rw_rows"]) == 600
        prefix = sum(starts)
        valid = sum(lengths)
        dynamic = valid - prefix
        kl_future = sum(lengths[row] - starts[row] - 1 for row in sp["kl_rows"])
        target_positions = int(sp["targets"].ne(-100).sum())
        stable_order = sorted(range(len(lengths)), key=lambda r: (lengths[r], r))
        counts = {"batch": batch, "rows": 700, "valid_tokens": valid,
                  "cached_prefix_tokens": prefix, "dynamic_suffix_tokens_including_subject": dynamic,
                  "KL_future_tokens": kl_future, "dynamic_suffix_after_KL_future_prune": dynamic - kl_future,
                  "rewrite_prediction_positions": target_positions, "KL_prediction_positions": len(sp["kl_rows"]),
                  "selected_head_positions": target_positions + len(sp["kl_rows"]),
                  "global_padding_slots": sp["tokens"]["input_ids"].numel(),
                  "original_order_MB4_crop_slots": padded_slots(lengths, 4),
                  "stable_length_MB4_crop_slots": padded_slots([lengths[r] for r in stable_order], 4),
                  "target_free_subject_prefix_matches": matches,
                  "full_unpadded_causal_attention_pairs": sum(n * (n + 1) // 2 for n in lengths),
                  "suffix_unpadded_causal_attention_pairs": sum((n * (n + 1) - s * (s + 1)) // 2
                                                                for n, s in zip(lengths, starts)),
                  "prefix_KV_fp32_bytes": prefix * kv_bytes_per_token,
                  "L4_suffix_fp32_bytes": dynamic * hidden * 4,
                  "sealed_token_identity": sp["identity"]}
        assert (valid, prefix, dynamic) == (old["valid_tokens"], old["cached_prefix_tokens"], old["dynamic_suffix_tokens"])
        batches.append(counts)
    additive = [k for k in batches[0] if k not in ("batch", "sealed_token_identity")]
    aggregate = {key: sum(row[key] for row in batches) for key in additive}
    aggregate.update(dynamic_suffix_fraction=aggregate["dynamic_suffix_tokens_including_subject"] / aggregate["valid_tokens"],
                     selected_head_fraction=aggregate["selected_head_positions"] / aggregate["valid_tokens"],
                     max_prefix_KV_fp32_GiB=max(r["prefix_KV_fp32_bytes"] for r in batches) / 2 ** 30)
    # These arithmetic counts are leading-term proxies, never GPU timing.
    batch_size = 100
    geometry = {"I": width, "O": hidden, "B": batch_size, "layers": layers,
                "large_matrix_elements": width * width,
                "Cholesky_leading_FLOPs_per_layer": width ** 3 / 3,
                "two_triangular_solves_leading_FLOPs_per_layer": 2 * width * width * batch_size,
                "small_system_shape": [batch_size, batch_size],
                "one_dense_FP64_factor_GiB": width * width * 8 / 2 ** 30,
                "five_dense_FP64_factors_GiB": layers * width * width * 8 / 2 ** 30,
                "policy_gradient_matmul_FLOPs_per_candidate": layers * 2 * hidden * batch_size * batch_size,
                "delta_grad_two_Adam_moments_FP32_MiB": layers * hidden * batch_size * 4 * 4 / 2 ** 20,
                "baseline_suffix_layers_L9_to_L31": mc["num_hidden_layers"] - 9,
                "v4_suffix_layers_L5_to_L31": suffix_layers,
                "layer_count_relative_increment": suffix_layers / (mc["num_hidden_layers"] - 9) - 1,
                "FLOP_convention": "multiply plus add = 2 FLOPs; leading-term algebra, not elapsed time"}
    token_names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                   "tokenizer.model", "vocab.json", "merges.txt")
    tokenizer_members = [{"logical_name": name, **member(model / name)}
                         for name in token_names if (model / name).is_file()]
    assert tokenizer_members and (model / "tokenizer.json").is_file()
    tokenizer_digest = digest([{k: v for k, v in item.items() if k != "path"}
                               for item in tokenizer_members])
    return {"status": "PASS_CPU_TOKENIZER_AND_ARITHMETIC_ONLY", "script": member(__file__),
            "input_manifest": member(input_path), "preparation_source": member(expected_source),
            "prior_count_evidence": member(prior_path),
            "prior_config": member(config_path) if config is not None else None,
            "stream": member(stream), "contexts": member(context_path),
            "model_directory": str(model.resolve()), "model_config": member(model / "config.json"),
            "tokenizer": {"files": tokenizer_members, "files_identity_sha256": tokenizer_digest,
                          "class": type(tok).__name__, "padding_side": tok.padding_side,
                          "pad_token_id": tok.pad_token_id, "bos_token_id": tok.bos_token_id,
                          "eos_token_id": tok.eos_token_id, "python": sys.version.split()[0],
                          "torch": torch.__version__, "transformers": transformers.__version__},
            "all_twenty_prior_packing_identities_match": True,
            "all_twenty_optional_snapshot_packing_identities_match": True if config is not None else None,
            "optional_snapshot_binding": "PRESENT_AND_VERIFIED" if config is not None else "NOT_AVAILABLE_REPOSITORY_IDENTITIES_VERIFIED",
            "model_weights_loaded": False,
            "model_inference_calls": 0, "GPU_calls": 0, "scope": "first2000/BS100x20; one logical candidate at each entry",
            "cached_KV_layout": {"layers": suffix_layers, "kv_heads": kv_heads, "head_dim": head_dim,
                                 "dtype": "float32", "bytes_per_prefix_token": kv_bytes_per_token},
            "batches": batches, "aggregate": aggregate, "geometry_arithmetic": geometry,
            "limitations": ["Token/position identity does not certify actual hidden/key/teacher/gradient parity.",
                            "KV memory omits padding, allocator, other activations and temporary copies.",
                            "Only positions strictly before subject are cacheable above L4; suffix includes subject.",
                            "KL future pruning retains the full original logical input identity and changes only causal work.",
                            "Attention and layer counts are proxies, not measured full-model FLOPs or speedup.",
                            "Twenty entries are input counts; no edited model state has been executed."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "native-work-counts.json")
    parser.add_argument("--model", type=Path, help="Local tokenizer/model-config directory; model weights are never loaded")
    parser.add_argument("--dataset", type=Path, help="Local dataset file with the exact input-manifest SHA")
    parser.add_argument("--contexts", type=Path, help="Local context file with the exact input-manifest SHA")
    parser.add_argument("--prior-snapshot-config", type=Path, default=DEFAULT_PRIOR_CONFIG,
                        help="Optional second packing identity binding; an absent file is recorded, not required")
    args = parser.parse_args()
    result = build(model_path=args.model, dataset_path=args.dataset, contexts_path=args.contexts,
                   prior_config_path=args.prior_snapshot_config)
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output.exists():
        assert args.output.read_text() == encoded, "REFUSE_TO_OVERWRITE_DIFFERENT_AUDIT"
    else:
        args.output.write_text(encoded)
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "aggregate": result["aggregate"], "geometry_arithmetic": result["geometry_arithmetic"]}, indent=2))


if __name__ == "__main__":
    main()
