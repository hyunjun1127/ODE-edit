"""CPU integration evidence for JLZ on a tiny random Transformers Llama.

This is an independent architecture fixture, not a production/native-text parity
test or a convergence experiment. Run with the existing EasyEdit virtualenv.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn.functional as F
import transformers
from transformers import LlamaConfig, LlamaForCausalLM


LAYERS = (4, 5, 6, 7, 8)
B, C, WIDTH, INTERMEDIATE = 2, 6, 32, 48


def max_error(a, b):
    return float((a.detach().double() - b.detach().double()).abs().max())


def detach_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(detach_tree(x) for x in value)
    if isinstance(value, list):
        return [detach_tree(x) for x in value]
    if isinstance(value, dict):
        return {k: detach_tree(v) for k, v in value.items()}
    return value


def fixtures():
    tokens, labels, subjects, requests, kinds = [], [], [], [], []
    for request, target in enumerate(((11, 12), (17,))):
        for context in range(C):
            prefix = [1] + ([23 + context] if context else [])
            subject = len(prefix)
            row = prefix + [5 + request, 9] + list(target[:-1])
            target_labels = [-100] * len(row)
            for i, token in enumerate(target):
                target_labels[len(prefix) + 1 + i] = token
            tokens.append(row)
            labels.append(target_labels)
            subjects.append(subject)
            requests.append(request)
            kinds.append("rewrite")
        tokens.append([1, 5 + request, 20])
        labels.append([-100] * 3)
        subjects.append(1)
        requests.append(request)
        kinds.append("kl")
    length = max(map(len, tokens))
    ids = torch.tensor([row + [0] * (length - len(row)) for row in tokens])
    mask = torch.tensor([[1] * len(row) + [0] * (length - len(row)) for row in tokens])
    labels = torch.tensor([row + [-100] * (length - len(row)) for row in labels])
    return dict(input_ids=ids, attention_mask=mask), labels, subjects, requests, kinds


class PrefixComplete(Exception):
    pass


def capture_prefix(model, tokens):
    """Cache actual mask/RoPE and inputs immediately before edited layer L4."""
    cache = {}

    def before(_module, args, kwargs):
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache", False):
            raise AssertionError("Mutable KV cache is forbidden")
        cache["args"] = detach_tree(args)
        cache["kwargs"] = detach_tree(kwargs)
        raise PrefixComplete()

    handle = model.model.layers[LAYERS[0]].register_forward_pre_hook(before, with_kwargs=True)
    try:
        with torch.no_grad():
            try:
                model(**tokens, use_cache=False)
            except PrefixComplete:
                pass
    finally:
        handle.remove()
    if not cache:
        raise AssertionError("Prefix cache was not captured")
    return cache


def replay(model, cache):
    args, kwargs = cache["args"], cache["kwargs"]
    hidden = args[0] if args else kwargs["hidden_states"]
    for layer in model.model.layers[LAYERS[0]:]:
        local = dict(kwargs)
        if args:
            output = layer(hidden, *args[1:], **local)
        else:
            local["hidden_states"] = hidden
            output = layer(**local)
        hidden = output[0] if isinstance(output, (tuple, list)) else output
    return model.lm_head(model.model.norm(hidden))


@contextmanager
def functional_weights(model, effective):
    handles = []
    for layer in LAYERS:
        def replace(module, args, _output, index=layer):
            return F.linear(args[0], effective[index], module.bias)
        handles.append(model.model.layers[layer].mlp.down_proj.register_forward_hook(replace))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def capture_entry(model, tokens, subjects):
    keys, anchors, handles = {}, {}, []
    rows = torch.arange(len(subjects))
    cols = torch.tensor(subjects)
    for layer in LAYERS:
        def key_hook(_module, args, index=layer):
            keys[index] = args[0][rows, cols].detach().clone()
        def anchor_hook(_module, _args, output, index=layer):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            anchors[index] = hidden[rows, cols].detach().clone()
        handles.append(model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(key_hook))
        handles.append(model.model.layers[layer].register_forward_hook(anchor_hook))
    try:
        with torch.no_grad():
            logits = model(**tokens, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()
    averaged = {layer: torch.stack([keys[layer][r * (C + 1):r * (C + 1) + C].mean(0)
                                   for r in range(B)], dim=1) for layer in LAYERS}
    canonical = {layer: torch.stack([anchors[layer][r * (C + 1)] for r in range(B)], dim=1)
                 for layer in LAYERS}
    return logits, averaged, canonical


def run():
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261001)
    torch.use_deterministic_algorithms(True)
    config = LlamaConfig(vocab_size=64, hidden_size=WIDTH, intermediate_size=INTERMEDIATE,
                         num_hidden_layers=10, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64, attention_dropout=0.0, use_cache=False)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    entry_w = {layer: model.model.layers[layer].mlp.down_proj.weight.detach().clone() for layer in LAYERS}
    tokens, labels, subjects, requests, kinds = fixtures()
    entry_logits, keys, anchors = capture_entry(model, tokens, subjects)
    kl_rows = [r * (C + 1) + C for r in range(B)]
    teacher = entry_logits[kl_rows, torch.tensor(subjects)[kl_rows]].log_softmax(-1).detach()
    c0 = {layer: torch.eye(INTERMEDIATE, dtype=torch.float64) * 1e-4 for layer in LAYERS}
    history = {layer: torch.eye(INTERMEDIATE, dtype=torch.float64) * 1e-5 for layer in LAYERS}
    c0_snapshot = {layer: value.clone() for layer, value in c0.items()}
    history_before = {layer: value.clone() for layer, value in history.items()}
    adj, solve_errors = {}, {}
    for layer in LAYERS:
        key = keys[layer].double()
        system = c0[layer] + history[layer] + key @ key.T
        adj[layer] = torch.linalg.solve(system, key)
        solve_errors[layer] = max_error(system @ adj[layer], key)
    radius = {layer: 0.75 * anchors[layer].norm(dim=0) for layer in LAYERS}
    native_c = {layer: 0.5 / anchors[layer].norm(dim=0).square() for layer in LAYERS}
    fixture_c = {layer: 1e-4 / anchors[layer].norm(dim=0).square() for layer in LAYERS}
    rows_full = list(range(len(requests)))
    micro_rows = [rows_full[start:start + 3] for start in range(0, len(requests), 3)]
    full_cache = capture_prefix(model, tokens)
    micro_cache = [capture_prefix(model, {name: value[rows] for name, value in tokens.items()})
                   for rows in micro_rows]
    active = torch.ones(B, dtype=torch.bool)

    def loss(logits, rows):
        lp = logits.log_softmax(-1)
        objective = logits.sum() * 0.0
        for local, row in enumerate(rows):
            request = requests[row]
            if not active[request]:
                continue
            if kinds[row] == "rewrite":
                use = labels[row] != -100
                objective = objective - lp[local, use, labels[row, use]].mean() / C
            else:
                # Native compute_z direction: KL(p_current || p_entry).
                objective = objective + 0.0625 * F.kl_div(
                    teacher[request], lp[local, subjects[row]], log_target=True, reduction="sum")
        return objective

    entry_nll = []
    for request in range(B):
        values = []
        for row in range(request * (C + 1), request * (C + 1) + C):
            use = labels[row] != -100
            values.append(-entry_logits[row].log_softmax(-1)[use, labels[row, use]].mean())
        entry_nll.append(float(torch.stack(values).mean()))
    active = torch.tensor(entry_nll) >= 0.05

    def effective_weights(residual):
        return {layer: entry_w[layer] + (residual[layer].double() @ adj[layer].T).float()
                for layer in LAYERS}

    def oracle(residual, route="suffix"):
        variables = {layer: residual[layer].detach().clone().requires_grad_(True) for layer in LAYERS}
        if route == "micro":
            # Backpropagate and release one prompt chunk at a time. Rebuilding
            # the small W_eff graph avoids retaining suffix activation graphs.
            total_value = 0.0
            total_gradient = {layer: torch.zeros_like(variables[layer]) for layer in LAYERS}
            outputs = []
            for rows, cache in zip(micro_rows, micro_cache):
                effective = effective_weights(variables)
                with functional_weights(model, effective):
                    logits = replay(model, cache)
                    value = loss(logits, rows)
                gradients = torch.autograd.grad(value, tuple(variables.values()))
                total_value += float(value.detach())
                for layer, gradient in zip(LAYERS, gradients):
                    total_gradient[layer].add_(gradient.detach())
                outputs.append(logits.detach())
            return (total_value, total_gradient, torch.cat(outputs),
                    {layer: weight.detach().clone() for layer, weight in effective.items()})
        effective = effective_weights(variables)
        with functional_weights(model, effective):
            logits = (model(**tokens, use_cache=False).logits if route == "full"
                      else replay(model, full_cache))
            value = loss(logits, rows_full)
        gradients = torch.autograd.grad(value, tuple(variables.values()))
        return (float(value.detach()), dict(zip(LAYERS, (g.detach() for g in gradients))),
                logits.detach(), {layer: weight.detach().clone() for layer, weight in effective.items()})

    def prox(residual, step, coefficients):
        result = {}
        for layer in LAYERS:
            norm = residual[layer].norm(dim=0)
            new_norm = torch.minimum(radius[layer], (norm - step * coefficients[layer]).clamp_min(0))
            result[layer] = residual[layer] * (new_norm / norm.clamp_min(torch.finfo(norm.dtype).tiny))
            result[layer][:, ~active] = 0
        return result

    def decay(residual, coefficients):
        return float(sum((coefficients[layer] * residual[layer].norm(dim=0)).sum() for layer in LAYERS))

    zero = {layer: torch.zeros(WIDTH, B) for layer in LAYERS}
    probe = {layer: torch.randn(WIDTH, B) for layer in LAYERS}
    probe = {layer: value / value.norm(dim=0) * radius[layer] * 0.2 for layer, value in probe.items()}
    full = oracle(probe, "full")
    suffix = oracle(probe, "suffix")
    micro = oracle(probe, "micro")
    parity = dict(
        prefix_full_logits_max_abs=max_error(full[2], suffix[2]),
        prefix_full_loss_abs=abs(full[0] - suffix[0]),
        prefix_full_gradient_max_abs=max(max_error(full[1][l], suffix[1][l]) for l in LAYERS),
        micro_full_logits_max_abs=max_error(full[2], micro[2]),
        micro_full_loss_abs=abs(full[0] - micro[0]),
        micro_full_gradient_max_abs=max(max_error(full[1][l], micro[1][l]) for l in LAYERS))
    zero_eval = oracle(zero)
    step = 0.01
    native_step = prox({l: -step * zero_eval[1][l] for l in LAYERS}, step, native_c)
    native_nonzero = sum(int((native_step[l].norm(dim=0) > 0).sum()) for l in LAYERS)
    native_ratio = max(float((zero_eval[1][l].norm(dim=0) / native_c[l]).max()) for l in LAYERS)
    residual, trajectory = zero, []
    for iteration in range(4):
        value, gradient, _, _ = oracle(residual)
        trajectory.append(dict(iteration=iteration, smooth=value,
                               objective=value + decay(residual, fixture_c),
                               gradient_norm=float(sum(g.double().square().sum() for g in gradient.values()).sqrt())))
        residual = prox({l: residual[l] - step * gradient[l] for l in LAYERS}, step, fixture_c)
    final_value, final_gradient, fit_logits, committed_weights = oracle(residual)
    trajectory.append(dict(iteration=4, smooth=final_value,
                           objective=final_value + decay(residual, fixture_c),
                           gradient_norm=float(sum(g.double().square().sum() for g in final_gradient.values()).sqrt())))
    oversize = {l: probe[l] / probe[l].norm(dim=0) * radius[l] * 10 for l in LAYERS}
    clipped = prox(oversize, step, fixture_c)
    clamp_error = max(max_error(clipped[l].norm(dim=0), radius[l]) for l in LAYERS)
    feasible_excess = max(float((residual[l].norm(dim=0) - radius[l]).clamp_min(0).max()) for l in LAYERS)
    before_commit_unchanged = all(torch.equal(p, original[name]) for name, p in model.named_parameters())
    # Commit the exact FP32 W_eff evaluated above; do not reconstruct it here.
    with torch.no_grad():
        for layer in LAYERS:
            model.model.layers[layer].mlp.down_proj.weight.copy_(committed_weights[layer])
    post_logits, post_keys, _ = capture_entry(model, tokens, subjects)
    append_counts = {layer: 0 for layer in LAYERS}
    for layer in LAYERS:
        key = post_keys[layer].double()
        history[layer].add_(key @ key.T)
        append_counts[layer] += 1
    history_error = max(max_error(history[l], history_before[l] + post_keys[l].double() @ post_keys[l].double().T)
                        for l in LAYERS)
    edit_names = {f"model.layers.{l}.mlp.down_proj.weight" for l in LAYERS}
    fixed_unchanged = all(torch.equal(p, original[name]) for name, p in model.named_parameters()
                          if name not in edit_names)
    commit_error = max_error(fit_logits, post_logits)
    low_key_error = max_error(keys[LAYERS[0]], post_keys[LAYERS[0]])
    native_kl = F.kl_div(teacher, fit_logits[kl_rows, torch.tensor(subjects)[kl_rows]].log_softmax(-1),
                         log_target=True, reduction="none").sum(-1)
    current_lp = fit_logits[kl_rows, torch.tensor(subjects)[kl_rows]].log_softmax(-1)
    explicit_kl = (current_lp.exp() * (current_lp - teacher)).sum(-1)
    nonzero_blocks = sum(int((residual[l].norm(dim=0) > 0).sum()) for l in LAYERS)
    edited_weight_change = {l: max_error(model.model.layers[l].mlp.down_proj.weight, entry_w[l])
                            for l in LAYERS}
    assertions = {
        "finite_value_gradient_logits": all(torch.isfinite(t).all().item() for t in
                                               [fit_logits, *final_gradient.values()]) and torch.isfinite(torch.tensor(final_value)).item(),
        "finite_residual_solve_weights_history": all(torch.isfinite(t).all().item() for t in
                                                        [*residual.values(), *adj.values(), *committed_weights.values(),
                                                         *history.values(), *c0.values(), *model.parameters()]),
        "nonzero_residual_path_exercised": nonzero_blocks > 0,
        "nonzero_weight_commit_exercised": max(edited_weight_change.values()) > 0,
        "fp64_linear_solve": max(solve_errors.values()) < 1e-12,
        "prefix_full_logits": parity["prefix_full_logits_max_abs"] < 2e-6,
        "prefix_full_loss_gradient": parity["prefix_full_loss_abs"] < 2e-6 and parity["prefix_full_gradient_max_abs"] < 2e-5,
        "micro_full_logits": parity["micro_full_logits_max_abs"] < 2e-6,
        "micro_full_loss_gradient": parity["micro_full_loss_abs"] < 4e-6 and parity["micro_full_gradient_max_abs"] < 2e-5,
        "model_unchanged_before_commit": before_commit_unchanged,
        "exact_evaluated_weights_committed": all(torch.equal(model.model.layers[l].mlp.down_proj.weight, committed_weights[l]) for l in LAYERS),
        "commit_logits_parity": commit_error < 2e-6,
        "history_append_once": all(count == 1 for count in append_counts.values()) and history_error == 0.0,
        "lowest_layer_keys_unchanged": low_key_error == 0.0,
        "fixed_weights_unchanged": fixed_unchanged,
        "c0_unchanged": all(torch.equal(c0[l], c0_snapshot[l]) for l in LAYERS),
        "clamp_probe_reaches_boundary": clamp_error < 2e-7,
        "returned_residual_feasible": feasible_excess < 2e-7,
        "native_kl_direction": max_error(native_kl, explicit_kl) < 1e-7,
        "all_hooks_removed": all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()),
    }
    result = dict(
        status="PASS" if all(assertions.values()) else "FAIL", assertions=assertions,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(device="cpu", threads=torch.get_num_threads(), torch=torch.__version__,
                     transformers=transformers.__version__, seconds=time.monotonic() - started),
        fixture=dict(seed=20261001, model="random LlamaForCausalLM", hidden=WIDTH, intermediate=INTERMEDIATE,
                     vocab=64, layers=10, edit_layers_zero_based=list(LAYERS), requests=B, rewrite_contexts=C,
                     kl_prompts_per_request=1, total_prompts=len(requests), prompt_microbatch_size=3,
                     prompt_microbatch_backward="Accumulate R gradients per chunk; release each suffix activation graph.",
                     sequence_length=tokens["input_ids"].shape[1], target_token_counts=[2, 1],
                     weight_dtype="float32", solve_dtype="float64", effective_weight="W_entry + (R.double() @ adj.T).float()",
                     c0="1e-4 I synthetic", entry_history="1e-5 I synthetic", lambda_value=1.0,
                     kl_factor=0.0625, kl_direction="KL(p_current || p_entry)", clamp=0.75),
        entry_nll=entry_nll, active_requests=active.tolist(),
        native_weight_decay_probe=dict(weight_decay=0.5, gradient_to_threshold_max=native_ratio,
                                      nonzero_blocks_after_one_prox=native_nonzero,
                                      interpretation="Native decay may correctly select zero on random weights; no performance inference."),
        optimization=dict(status="FIXED_FOUR_STEPS_NOT_CONVERGENCE", fixture_weight_decay=1e-4,
                          reason="Exercise nonzero effective-weight and commit paths with random untrained anchors.",
                          step_size=step, trajectory=trajectory,
                          nonzero_blocks=nonzero_blocks),
        parity=parity, commit_logits_max_abs=commit_error, lowest_layer_key_max_abs=low_key_error,
        edited_weight_max_abs_change=max(edited_weight_change.values()),
        per_layer_edited_weight_max_abs_change=edited_weight_change,
        per_layer_key_change={l: max_error(keys[l], post_keys[l]) for l in LAYERS},
        history_append_max_abs=history_error, history_append_counts=append_counts,
        linear_solve_max_abs=max(solve_errors.values()), clamp_boundary_max_abs=clamp_error,
        feasibility_max_excess=feasible_excess,
        not_tested=["production pinned Transformers 4.44.2", "native text/tokenizer/context parity",
                    "pretrained knowledge editing", "production memit_hj adapter", "solver convergence/KKT",
                    "full-scale runtime or GPU memory", "sequential multi-batch editing"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("local/jlz-smoke-20261001/tiny-llama.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(status=result["status"], seconds=result["runtime"]["seconds"],
                         assertions=result["assertions"], output=str(args.output)), ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
