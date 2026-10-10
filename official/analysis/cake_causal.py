"""CAKE Appendix E/E.1: known-fact selection and single-MLP AIE at W0.

This is a Qwen2.5 port of the paper protocol, not a claim that the paper
reported Qwen2.5 settings. No editing, history writer, gradients or checkpoints.
Raw prompts and per-noise probabilities remain in the local output directory.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def validate(config):
    expected = dict(schema="cake-paper-aie-v1", known_facts=1000,
                    noise_samples=10, restore_token="subject_last", restore_window=1,
                    corrupt_tokens="all_subject_tokens",
                    noise_rule="3_times_subject_embedding_std")
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("PAPER_PROTOCOL_MISMATCH")
    if config["layers"] != [4, 5, 6, 7, 8]:
        raise ValueError("QWEN_BASELINE_CRITICAL_SET_MISMATCH")
    if config["model"] != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError("QWEN25_PORT_ONLY")
    if (config["restore_module"] != "model.layers.{}.mlp"
            or config["embedding_module"] != "model.embed_tokens"):
        raise ValueError("MLP_OUTPUT_PATCH_REQUIRED")
    return config


def object_position(continuation, object_text):
    """First correct object mention must precede any other capitalized word.

    Match the object with word boundaries, then trace its first token. Requiring
    the complete object mention prevents e.g. United States matching United
    Kingdom merely because their first subword token is shared.
    """
    match = re.search(r"(?<!\w)" + re.escape(object_text) + r"(?!\w)", continuation)
    if match is None:
        return None
    earlier = re.finditer(r"\b[^\W\d_]\w*", continuation[:match.start()])
    if any(m.group()[0].isupper() for m in earlier):
        return None
    return match.start()


def selection_finished(continuation, object_text):
    if object_position(continuation, object_text) is not None:
        return True
    for match in re.finditer(r"\b[^\W\d_]\w*", continuation):
        if match.group()[0].isupper():
            # Wait if the object name is still being completed token by token.
            partial = continuation[match.start():]
            return not object_text.startswith(partial)
    return False


def known_record(tokenizer, record, prompt_ids, continuation_ids):
    request = record["requested_rewrite"]
    object_text = request["target_true"]["str"]
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False)
    prompt = decode(prompt_ids)
    full_ids = prompt_ids + continuation_ids
    full = decode(full_ids)
    if not full.startswith(prompt):
        return None, "NONCONCATENATIVE_DECODE"
    position = object_position(full[len(prompt):], object_text)
    if position is None:
        return None, "OBJECT_NOT_FIRST_CAPITALIZED_MENTION"
    position += len(prompt)
    # Retain the actual generated token prefix; do not decode/re-tokenize the
    # answer into a different next-token target (especially a trailing space).
    for index in range(len(prompt_ids), len(full_ids)):
        prefix, through = decode(full_ids[:index]), decode(full_ids[:index + 1])
        if len(prefix) <= position < len(through):
            encoded = tokenizer(prefix, add_special_tokens=False,
                                return_offsets_mapping=True)
            if encoded["input_ids"] != full_ids[:index]:
                return None, "PREFIX_TOKENIZATION_MISMATCH"
            subject = request["subject"]
            start = prefix.find(subject)
            if start < 0:
                return None, "SUBJECT_MISSING"
            end = start + len(subject)
            spans = [i for i, (a, b) in enumerate(encoded["offset_mapping"])
                     if b > start and a < end and b > a]
            if not spans or spans != list(range(spans[0], spans[-1] + 1)):
                return None, "SUBJECT_SPAN_MISMATCH"
            return dict(case_id=record["case_id"], subject=subject,
                        object_text=object_text, prompt=prefix,
                        input_ids=full_ids[:index], object_token_id=full_ids[index],
                        subject_range=[spans[0], spans[-1] + 1]), "ELIGIBLE"
    return None, "OBJECT_TOKEN_NOT_FOUND"


def collect_knowns(model, tokenizer, records, config, output, progress):
    from transformers import StoppingCriteria, StoppingCriteriaList

    options = config["engineering"]
    candidates, counts = [], Counter()
    device = next(model.parameters()).device
    batch_size = options["generation_batch_size"]
    with (output / "selection.jsonl").open("x") as journal:
        for offset in range(0, len(records), batch_size):
            batch = records[offset:offset + batch_size]
            prompts = [r["requested_rewrite"]["prompt"].format(
                r["requested_rewrite"]["subject"]) for r in batch]
            inputs = tokenizer(prompts, padding=True, add_special_tokens=False,
                               return_tensors="pt").to(device)
            width = inputs["input_ids"].shape[1]
            objects = [r["requested_rewrite"]["target_true"]["str"] for r in batch]

            class CompletedMention(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    text = tokenizer.batch_decode(input_ids[:, width:],
                        skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    return torch.tensor([selection_finished(t, o) for t, o in zip(text, objects)],
                                        dtype=torch.bool, device=input_ids.device)

            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, num_beams=1,
                    max_new_tokens=options["max_new_tokens"], use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=StoppingCriteriaList([CompletedMention()]))
            rows = generated[:, width:].tolist()
            for i, (record, row) in enumerate(zip(batch, rows)):
                prompt_ids = inputs["input_ids"][i][inputs["attention_mask"][i].bool()].tolist()
                candidate, status = known_record(tokenizer, record, prompt_ids, row)
                counts[status] += 1
                if candidate is not None:
                    candidates.append(candidate)
                journal.write(json.dumps(dict(case_id=record["case_id"], status=status,
                    generated_token_ids=row, known=candidate), ensure_ascii=False) + "\n")
            journal.flush()
            progress(dict(phase_id=1, **{"causal/candidates_scanned": offset + len(batch),
                                        "causal/eligible_facts": len(candidates)}))
    if len(candidates) < config["known_facts"]:
        write(output / "selection-summary.json", dict(counts=dict(counts), eligible=len(candidates)))
        raise ValueError("FEWER_THAN_1000_BASE_KNOWN_FACTS")
    selected = random.Random(options["sampling_seed"]).sample(candidates, config["known_facts"])
    write(output / "knowns.json", dict(selected=selected, eligible=len(candidates),
        scanned=len(records), counts=dict(counts), sampling_seed=options["sampling_seed"]))
    return selected


def subject_embedding_std(model, tokenizer, knowns):
    """Same subject-only embeddings/std() as CAKE collect_embedding_std.

    Reading the embedding module directly yields the same values as hooking
    its output during full subject-only forwards; no Transformer is needed.
    """
    embedding = model.get_input_embeddings()
    pieces = []
    with torch.inference_mode():
        for row in knowns:
            ids = tokenizer.encode(row["subject"], add_special_tokens=False)
            if not ids:
                raise ValueError("EMPTY_SUBJECT")
            token_ids = torch.tensor(ids, device=embedding.weight.device)
            pieces.append(embedding(token_ids).float())
        values = torch.cat(pieces)
        std = values.std().item()
    if not math.isfinite(std) or std <= 0:
        raise ValueError("INVALID_EMBEDDING_STD")
    return std, values.numel()


@contextmanager
def intervention(model, embedding_name, module_name, subject_range, noise):
    handles = []
    start, end = subject_range

    def corrupt(module, args, output):
        value = output.clone()
        value[1:, start:end] += noise.to(device=value.device, dtype=value.dtype)
        return value

    def restore(module, args, output):
        if not torch.is_tensor(output):
            raise ValueError("MLP_OUTPUT_MUST_BE_TENSOR")
        value = output.clone()
        value[1:, end - 1] = value[0, end - 1]
        return value

    try:
        handles.append(model.get_submodule(embedding_name).register_forward_hook(corrupt))
        if module_name is not None:
            handles.append(model.get_submodule(module_name).register_forward_hook(restore))
        yield
    finally:
        for handle in handles:
            handle.remove()


def trace_case(model, row, config, noise_scale):
    samples = config["noise_samples"]
    device = next(model.parameters()).device
    ids = torch.tensor([row["input_ids"]] * (samples + 1), device=device)
    mask = torch.ones_like(ids)
    start, end = row["subject_range"]
    if not (0 <= start < end <= ids.shape[1]):
        raise ValueError("SUBJECT_RANGE_INVALID")
    width = model.get_input_embeddings().weight.shape[1]
    seed_material = f'{config["engineering"]["noise_seed"]}:{row["case_id"]}'
    seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
    noise = noise_scale * np.random.RandomState(seed).randn(samples, end - start, width)
    noise = torch.from_numpy(noise).to(device=device, dtype=torch.float32)
    result = dict(case_id=row["case_id"], noise_seed=seed,
                  object_token_id=row["object_token_id"], restored={})
    for layer in [None] + config["layers"]:
        module = None if layer is None else config["restore_module"].format(layer)
        with torch.inference_mode(), intervention(model, config["embedding_module"],
                                                   module, (start, end), noise):
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1]
            if logits[0].argmax().item() != row["object_token_id"]:
                raise ValueError("SELECTED_FACT_IS_NOT_BASE_KNOWN")
            probs = logits.float().softmax(-1)[:, row["object_token_id"]].tolist()
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in probs):
            raise ValueError("NONFINITE_OR_INVALID_PROBABILITY")
        if layer is None:
            result.update(clean_probability=probs[0], corrupted=probs[1:])
        else:
            if probs[0] != result["clean_probability"]:
                raise ValueError("CLEAN_REFERENCE_CHANGED")
            result["restored"][str(layer)] = probs[1:]
    result["indirect_effect"] = {
        str(layer): math.fsum(a - b for a, b in zip(result["restored"][str(layer)],
                                                   result["corrupted"])) / samples
        for layer in config["layers"]}
    return result


def aggregate(rows, config):
    if len(rows) != config["known_facts"] or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("EXACT_UNIQUE_COHORT_REQUIRED")
    layers, samples = config["layers"], config["noise_samples"]
    scores = {}
    for layer in layers:
        effects = []
        for row in rows:
            low, high = row["corrupted"], row["restored"][str(layer)]
            if len(low) != samples or len(high) != samples:
                raise ValueError("EXACT_NOISE_SAMPLE_COUNT_REQUIRED")
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in low + high):
                raise ValueError("INVALID_PROBABILITY")
            effects.append(math.fsum(a - b for a, b in zip(high, low)) / samples)
        scores[str(layer)] = math.fsum(effects) / len(rows)
    exponentials = [math.exp((scores[str(l)] - max(scores.values())) / config["temperature"])
                    for l in layers]
    return dict(physical_layer_scores=scores,
                causal_scores={str(i): scores[str(l)] for i, l in enumerate(layers)},
                layer_weights={str(l): w / sum(exponentials) for l, w in zip(layers, exponentials)},
                layers=layers, temperature=config["temperature"], requests=len(rows),
                noise_samples=samples, score_definition=config["score"])


def parameter_signature(model):
    return {k: (v.data_ptr(), v._version) for k, v in model.named_parameters()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--counterfact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--wandb-env", required=True)
    parser.add_argument("--server", required=True)
    args = parser.parse_args()
    config = validate(json.loads(Path(args.config).read_text()))
    model_path = Path(args.model).resolve()
    if model_path.name != config["model_revision"]:
        raise ValueError("BASE_MODEL_REVISION_MISMATCH")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started, tracker = time.monotonic(), None
    from official.tracking import init
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        tracker = init(env_file=args.wandb_env, spool=output / "tracking",
            config=dict(server=args.server, task_id="cake-qwen25-causal-score-20261011",
                        arm="qwen25-cake-causal-score", attempt=output.name,
                        source_sha=args.source_sha, config_sha=sha(args.config)))

        def progress(values):
            values = dict(values, **{"time/elapsed_seconds": time.monotonic() - started})
            tracker.log(values)
            write(output / "progress.json", values)

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_grad_enabled(False)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
            torch_dtype=torch.float32, attn_implementation="eager").to("cuda:0").eval()
        model.requires_grad_(False)
        before = parameter_signature(model)
        records = json.loads(Path(args.counterfact).read_text())
        if len(records) != 21919 or len({r["case_id"] for r in records}) != len(records):
            raise ValueError("ORIGINAL_COUNTERFACT_21919_REQUIRED")
        write(output / "identity.json", dict(source_sha=args.source_sha, config=config,
            config_sha256=sha(args.config), counterfact_sha256=sha(args.counterfact),
            model_snapshot=str(model_path), model_config_sha256=sha(model_path / "config.json"),
            model_index_sha256=sha(model_path / "model.safetensors.index.json"),
            tokenizer_sha256=sha(model_path / "tokenizer.json"), torch=torch.__version__,
            GPU=torch.cuda.get_device_name(0), W0=True, editing=False, checkpoint_saved=False))
        selected = collect_knowns(model, tokenizer, records, config, output, progress)
        std, elements = subject_embedding_std(model, tokenizer, selected)
        scale = 3 * std
        write(output / "noise.json", dict(subject_embedding_std=std, noise_scale=scale,
            elements=elements, correction=1, model_specific_paper_constant=False,
            rule=config["noise_rule"], knowns_sha256=sha(output / "knowns.json")))
        results = []
        with (output / "traces.jsonl").open("x") as journal:
            for row in selected:
                result = trace_case(model, row, config, scale)
                results.append(result)
                journal.write(json.dumps(result, allow_nan=False) + "\n")
                journal.flush()
                progress(dict(phase_id=2, **{"causal/traced_facts": len(results),
                                            "causal/noise_scale": scale}))
        if parameter_signature(model) != before:
            raise ValueError("MODEL_PARAMETERS_MUTATED")
        score = aggregate(results, config)
        score.update(status="COMPLETED", model=config["model"], model_revision=config["model_revision"],
            source_sha=args.source_sha, noise_scale=scale, config_sha256=sha(args.config),
            knowns_sha256=sha(output / "knowns.json"), traces_sha256=sha(output / "traces.jsonl"),
            parameters_unchanged=True, checkpoint_saved=False)
        write(output / "causal-scores.json", score)
        for layer in config["layers"]:
            progress(dict(phase_id=3, **{"causal/layer": layer,
                "causal/AIE": score["physical_layer_scores"][str(layer)],
                "causal/layer_weight": score["layer_weights"][str(layer)]}))
        write(output / "COMPLETE.json", dict(status="COMPLETED", result_sha256=sha(output / "causal-scores.json"),
            model_updates=0, requests=len(results), elapsed_seconds=time.monotonic() - started))
    except Exception as error:
        write(output / "FAILED.json", dict(status="FAILED", error_type=type(error).__name__,
            error=str(error), elapsed_seconds=time.monotonic() - started))
        if tracker is not None:
            tracker.finish(exit_code=1)
        raise
    else:
        tracker.finish()


if __name__ == "__main__":
    main()
