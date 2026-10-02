"""CPU FP64 checks for v4 compute reuse, not real Llama/FP32 validation.

Checks cached Cholesky/Woodbury geometry; real causal-attention prefix K/V
reuse and KL future pruning; pooled selected heads; and a sequential actual
weight writer with reused layer states and post-write history keys. No GPU,
download, model checkpoint, quality assertion, or timing estimate is used.
"""
from pathlib import Path
import hashlib
import json
import math
import torch
import torch.nn.functional as F


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    rng = torch.Generator(device="cpu").manual_seed(20261004)
    rand = lambda *s: torch.randn(*s, generator=rng, device="cpu")
    H, I, heads, head_dim, depth, B, contexts, vocab = 8, 12, 2, 4, 10, 2, 6, 13
    edits = list(range(4, 9))
    down = [rand(H, I) * .12 for _ in range(depth)]
    up = [rand(I, H) * .14 for _ in range(depth)]
    query = [rand(H, H) * .13 for _ in range(depth)]
    key = [rand(H, H) * .13 for _ in range(depth)]
    value = [rand(H, H) * .13 for _ in range(depth)]
    output = [rand(H, H) * .13 for _ in range(depth)]
    head = rand(vocab, H) * .2
    inputs, requests, subject, kl_flags = [], [], [], []
    for request in range(B):
        for context in range(contexts + 1):
            length = 5 + (request + context) % 3
            inputs.append(rand(length, H) * .6)
            requests.append(request)
            subject.append(1 + (context % 2))
            kl_flags.append(context == contexts)
    subject[0] = 0
    subject[contexts + 1] = len(inputs[contexts + 1]) - 1
    kl = torch.tensor(kl_flags)
    target = torch.tensor([(2 + 3 * r) % vocab for r in requests])
    norm = lambda x: x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

    def before_down(x, layer, cached_prefix=None, start=0):
        normalized = norm(x)
        q = F.linear(normalized, query[layer]).reshape(-1, heads, head_dim).transpose(0, 1)
        k = F.linear(normalized, key[layer]).reshape(-1, heads, head_dim).transpose(0, 1)
        v = F.linear(normalized, value[layer]).reshape(-1, heads, head_dim).transpose(0, 1)
        if cached_prefix is not None:
            k = torch.cat([cached_prefix[0], k], dim=1)
            v = torch.cat([cached_prefix[1], v], dim=1)
        allowed = torch.arange(k.shape[1])[None, :] <= (start + torch.arange(q.shape[1]))[:, None]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
        attention = scores.masked_fill(~allowed[None, :, :], -torch.inf).softmax(-1) @ v
        joined = attention.transpose(0, 1).reshape(-1, H)
        residual = x + F.linear(joined, output[layer])
        write_input = torch.tanh(F.linear(norm(residual), up[layer]))
        return residual, write_input, (k, v)

    def full_row(row, weights, deltas=None, stop=None, prune_kl=False):
        x = inputs[row]
        if prune_kl and kl_flags[row]:
            x = x[:subject[row] + 1]
        history_keys, kvs, states = {}, {}, {}
        for layer in range(depth):
            residual, write_input, kv = before_down(x, layer)
            if layer in edits:
                history_keys[layer] = write_input[subject[row]]
            x = residual + F.linear(write_input, weights[layer])
            if deltas is not None and layer in edits:
                mask = F.one_hot(torch.tensor(subject[row]), len(x)).to(x.dtype)[:, None]
                x = x + mask * deltas[layer - 4][:, requests[row]]
            kvs[layer], states[layer] = kv, x
            if stop is not None and layer == stop:
                break
        return x, history_keys, kvs, states

    with torch.no_grad():
        entry = [full_row(row, down) for row in range(len(inputs))]
        entry_selected = [r[0][subject[row] if kl_flags[row] else -1] for row, r in enumerate(entry)]
        teacher = F.linear(norm(torch.stack(entry_selected)), head).log_softmax(-1)
        anchor_sq = []
        for layer in edits:
            anchor_sq.append(torch.stack([entry[r * (contexts + 1)][3][layer][subject[r * (contexts + 1)]].square().sum()
                                          for r in range(B)]))
        radii = [.75 * v.sqrt() for v in anchor_sq]
        point = []
        for radius in radii:
            d = rand(H, B)
            point.append(d * (.25 * radius / d.norm(dim=0))[None, :])

    def cached_row(row, deltas, prune_kl):
        s = subject[row]
        end = s + 1 if prune_kl and kl_flags[row] else len(inputs[row])
        x = entry[row][3][4][s:end].clone()
        mask = F.one_hot(torch.tensor(0), len(x)).to(x.dtype)[:, None]
        x = x + mask * deltas[0][:, requests[row]]
        for layer in range(5, depth):
            full_k, full_v = entry[row][2][layer]
            prefix = (full_k[:, :s], full_v[:, :s])
            residual, write_input, _ = before_down(x, layer, prefix, s)
            x = residual + F.linear(write_input, down[layer])
            if layer in edits:
                x = x + mask * deltas[layer - 4][:, requests[row]]
        return x[0 if kl_flags[row] else -1]

    def objective(deltas, route, pooled_head):
        selected = []
        for row in range(len(inputs)):
            if route.startswith("cached"):
                selected.append(cached_row(row, deltas, route == "cached_pruned"))
            else:
                x = full_row(row, down, deltas, prune_kl=route == "full_pruned")[0]
                selected.append(x[subject[row] if kl_flags[row] else -1])
        if pooled_head:
            logits = F.linear(norm(torch.stack(selected)), head)
        else:
            logits = torch.stack([F.linear(norm(x), head) for x in selected])
        logp = logits.log_softmax(-1)
        nll = -logp[~kl, target[~kl]].sum() / contexts
        native_kl = (logp[kl].exp() * (logp[kl] - teacher[kl])).sum()
        regularizer = sum((.5 * d.norm(dim=0) / scale).sum() for d, scale in zip(deltas, anchor_sq))
        return nll + .0625 * native_kl + regularizer, logits

    def evaluate(route, pooled):
        ds = [d.clone().requires_grad_(True) for d in point]
        loss, logits = objective(ds, route, pooled)
        gradients = torch.autograd.grad(loss, ds)
        return float(loss.detach()), logits.detach(), [g.detach() for g in gradients]

    pack = lambda gs: torch.cat([g.flatten() for g in gs])
    reference = evaluate("full", False)
    reuse_checks = []
    for route, pooled in [("full", True), ("full_pruned", True), ("cached", True), ("cached_pruned", True)]:
        loss, logits, gradient = evaluate(route, pooled)
        diff = pack(gradient) - pack(reference[2])
        item = dict(route=route, pooled_head=pooled, loss_abs_error=abs(loss - reference[0]),
                    logits_max_abs_error=float((logits - reference[1]).abs().max()),
                    gradient_relative_error=float(diff.norm() / pack(reference[2]).norm()),
                    per_layer_gradient_abs_error=[float((g - r).norm()) for g, r in zip(gradient, reference[2])])
        assert max(item["loss_abs_error"], item["logits_max_abs_error"], item["gradient_relative_error"]) < 1e-10
        reuse_checks.append(item)

    # Reuse one Cholesky factor of A for both entry and changed current K.
    a_raw = rand(I, I)
    A = torch.eye(I) + a_raw.T @ a_raw
    factor = torch.linalg.cholesky(A)
    K0 = rand(I, B) * .3
    D = rand(H, B) * .2
    geometry_checks = []
    for label, K in (("entry", K0), ("changed_current_key", K0 + .17 * rand(I, B))):
        direct = torch.linalg.solve(A + K @ K.T, K)
        Tsolve = torch.cholesky_solve(K, factor)
        S = torch.eye(B) + K.T @ Tsolve
        woodbury = torch.linalg.solve(S.T, Tsolve.T).T
        q_direct, q_reuse = K.T @ direct, K.T @ woodbury
        value_grad = []
        for Q in (q_direct, q_reuse):
            d = D.clone().requires_grad_(True)
            val = .5 * ((d @ (torch.eye(B) - Q)) * d).sum()
            grad = torch.autograd.grad(val, d)[0]
            value_grad.append((float(val.detach()), grad))
        item = dict(key_state=label, P_max_abs_error=float((direct - woodbury).abs().max()),
                    Q_max_abs_error=float((q_direct - q_reuse).abs().max()),
                    value_abs_error=abs(value_grad[0][0] - value_grad[1][0]),
                    gradient_max_abs_error=float((value_grad[0][1] - value_grad[1][1]).abs().max()))
        assert max(v for k, v in item.items() if k != "key_state") < 1e-11
        geometry_checks.append(item)

    # Generic geometry must not assume B < input dimension. The matrix-free
    # E action and output tiles retain coupling across ALL request columns.
    generic_geometry = []
    for name, din, dout, batch in (("B1", 7, 5, 1), ("B_lt_din", 7, 3, 4),
                                   ("B_eq_din", 4, 6, 4), ("B_gt_din", 3, 5, 7)):
        raw = rand(din, din)
        a = torch.eye(din) + raw @ raw.T
        k, d = rand(din, batch) * .4, rand(dout, batch) * .3
        system = a + k @ k.T
        direct = torch.linalg.solve(system, k)
        t = torch.cholesky_solve(k, torch.linalg.cholesky(a))
        dual = torch.linalg.solve((torch.eye(batch) + k.T @ t).T, t.T).T
        e = torch.eye(batch) - k.T @ direct
        x = d.T
        action = x - k.T @ torch.linalg.solve(system, k @ x)
        expected_gradient = d @ e
        value_explicit = .5 * ((d @ e) * d).sum()
        value_action = .5 * (x * action).sum()
        d_leaf = d.clone().requires_grad_(True)
        value_ad = .5 * ((d_leaf @ e) * d_leaf).sum()
        gradient_ad = torch.autograd.grad(value_ad, d_leaf)[0]
        # Tile output dimensions, accumulate KX over the entire B dimension,
        # solve only after that accumulation, then form each output tile.
        tiled_gradient = torch.empty_like(d)
        system_factor = torch.linalg.cholesky(system)
        max_rhs_error = 0.
        for out_start in range(0, dout, 2):
            out_stop = min(out_start + 2, dout)
            xtile = x[:, out_start:out_stop]
            rhs = torch.zeros(din, out_stop - out_start)
            for request_start in range(0, batch, 2):
                request_stop = min(request_start + 2, batch)
                rhs += k[:, request_start:request_stop] @ xtile[request_start:request_stop]
            max_rhs_error = max(max_rhs_error, float((rhs - k @ xtile).abs().max()))
            solved = torch.cholesky_solve(rhs, system_factor)
            for request_start in range(0, batch, 2):
                request_stop = min(request_start + 2, batch)
                etile = xtile[request_start:request_stop] - k[:, request_start:request_stop].T @ solved
                tiled_gradient[out_start:out_stop, request_start:request_stop] = etile.T
        item = dict(case=name, input_dim=din, output_dim=dout, B=batch,
                    P_direct_dual_max_error=float((direct - dual).abs().max()),
                    E_action_max_error=float((action - e @ x).abs().max()),
                    gradient_action_max_error=float((action.T - expected_gradient).abs().max()),
                    gradient_autograd_max_error=float((gradient_ad - expected_gradient).abs().max()),
                    value_action_error=float((value_action - value_explicit).abs()),
                    KX_full_B_accumulation_error=max_rhs_error,
                    output_tiled_gradient_max_error=float((tiled_gradient - expected_gradient).abs().max()),
                    output_tiled_value_error=float((.5 * (d * tiled_gradient).sum() - value_explicit).abs()),
                    request_tile_size=2, output_tile_size=2,
                    every_output_tile_uses_all_B_columns=True)
        assert max(v for key_name, v in item.items() if key_name.endswith("error")) < 1e-11
        generic_geometry.append(item)
    partial_batch_schedule = [min(128, 2000 - start) for start in range(0, 2000, 128)]
    assert partial_batch_schedule == [128] * 15 + [80]
    assert sum(partial_batch_schedule) == 2000

    # Materialized all-token writes: compare repeated prefix recapture with
    # a one-pass layer stream. Both use changed current keys before each write.
    initial_H = {layer: .1 * torch.eye(I) for layer in edits}
    As = {}
    for layer in edits:
        raw = rand(I, I)
        As[layer] = torch.eye(I) + raw.T @ raw + initial_H[layer]

    def pool(per_row):
        return torch.stack([.5 * per_row[r * (contexts + 1)] +
                            .1 * torch.stack(per_row[r * (contexts + 1) + 1:r * (contexts + 1) + contexts]).sum(0)
                            for r in range(B)], dim=1)

    def write_weight(layer, K):
        P = torch.linalg.solve(As[layer] + K @ K.T, K)
        return down[layer] + point[layer - 4] @ P.T

    with torch.no_grad():
        repeated = [w.clone() for w in down]
        repeated_keys = {}
        for layer in edits:
            keys = [full_row(row, repeated, stop=layer)[1][layer] for row in range(len(inputs))]
            repeated_keys[layer] = pool(keys)
            repeated[layer] = write_weight(layer, repeated_keys[layer])
        repeated_final = [full_row(row, repeated) for row in range(len(inputs))]
        repeated_post = {layer: pool([r[1][layer] for r in repeated_final]) for layer in edits}
        stream_weights = [w.clone() for w in down]
        states = [x.clone() for x in inputs]
        stream_keys = {}
        for layer in range(depth):
            partial = [before_down(x, layer) for x in states]
            if layer in edits:
                stream_keys[layer] = pool([p[1][subject[row]] for row, p in enumerate(partial)])
                stream_weights[layer] = write_weight(layer, stream_keys[layer])
            # Exactly one modified down_proj per layer/token; do not replace
            # its result with a virtual subject delta.
            states = [p[0] + F.linear(p[1], stream_weights[layer]) for p in partial]
        stream_post_full = [full_row(row, stream_weights) for row in range(len(inputs))]
        stream_post = {layer: pool([r[1][layer] for r in stream_post_full]) for layer in edits}
        writer_checks = []
        for layer in edits:
            entry_K = pool([r[1][layer] for r in entry])
            repeated_history = initial_H[layer] + repeated_post[layer] @ repeated_post[layer].T
            reused_history = initial_H[layer] + stream_keys[layer] @ stream_keys[layer].T
            item = dict(layer=layer, current_key_route_error=float((stream_keys[layer] - repeated_keys[layer]).abs().max()),
                        materialized_weight_error=float((stream_weights[layer] - repeated[layer]).abs().max()),
                        prewrite_postall_key_error=float((stream_keys[layer] - stream_post[layer]).abs().max()),
                        reused_vs_remeasured_H_error=float((reused_history - repeated_history).abs().max()),
                        current_vs_entry_key_change=float((stream_keys[layer] - entry_K).norm()),
                        history_appends_per_route=1)
            assert max(v for k, v in item.items() if k.endswith("error")) < 1e-11
            assert item["current_vs_entry_key_change"] == 0 if layer == 4 else item["current_vs_entry_key_change"] > 1e-8
            writer_checks.append(item)
        final_state_error = max(float((x - r[0]).abs().max()) for x, r in zip(states, repeated_final))
        assert final_state_error < 1e-11

    script = Path(__file__).resolve()
    result = dict(status="PASS_CPU_FP64_TINY_ATTENTION_NOT_LLAMA_FP32_CERTIFICATION",
                  script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(), torch_version=str(torch.__version__),
                  device="cpu", dtype="float64", dimensions=dict(hidden=H, intermediate=I, heads=heads,
                      head_dim=head_dim, model_layers=depth, edit_layers=edits, requests=B, rewrite_contexts=contexts,
                      KL_rows_per_request=1, rows=len(inputs)),
                  cholesky_woodbury=dict(large_A_factorizations=1, key_states=geometry_checks),
                  generic_geometry=generic_geometry,
                  partial_batch_schedule=dict(requests=2000, requested_batch_size=128,
                      actual_batch_sizes=partial_batch_schedule, full_batches=15, final_batch_size=80,
                      request_coverage=2000, number_of_batches=16),
                  delta_forward_loss_gradient=reuse_checks,
                  coverage=dict(subject_at_position_zero=True, subject_at_final_position=True,
                      KL_future_tokens_removed=sum(len(inputs[r]) - subject[r] - 1 for r in range(len(inputs)) if kl_flags[r]),
                      later_layer_suffix_recomputed=True, pooled_head_full_vocabulary=True,
                      all_five_deltas_jointly_differentiated=True),
                  writer=dict(layer_checks=writer_checks, final_alltoken_hidden_max_abs_error=final_state_error,
                      lower_layer_writes_reflected_in_current_keys=True, actual_materialized_weights=True,
                      history_updated_after_all_writes_only=True),
                  limitations=["CPU FP64 tiny attention model; real Llama rotary/GQA/mask/cache APIs and FP32 kernels remain unvalidated.",
                      "Generic SPD geometry and schedule checks do not certify arbitrary production models, datasets, or benchmark integrations.",
                      "Prefix K/V remains visible to every dynamic suffix query; no prior context is deleted.",
                      "KL future pruning is justified by causal output dependence, not prompt removal.",
                      "No GPU timing, speedup, model quality or production implementation claim."])
    destination = script.with_name("compute-reuse-check.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], output=str(destination),
                         maximum_gradient_relative_error=max(r["gradient_relative_error"] for r in reuse_checks),
                         final_writer_hidden_error=final_state_error), indent=2))


if __name__ == "__main__":
    main()
