"""Connect the CPU reference pipeline to a synthetic nonlinear causal suffix.

No model download, native compute_z, GPU job, or knowledge-editing evaluation.
The synthetic contexts and labels test wiring, not native tokenizer semantics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from .flow_core import QuadraticGeometry, integrate, native_map, preservation_gram
from .oracle import CachedSuffixOracle, make_affine_cache
from .transaction import (
    CommitBudgetError, CommitParityError, InMemoryBatchTransaction,
)


DEFAULT_CONTRACT = (Path(__file__).resolve().parents[3] / "plans" / "global" /
                    "2026-09-16-single-layer-zflow-contract-v1.json")


def run_demo(contract: dict, seed: int = 31) -> dict:
    """Return a JSON-safe receipt, retaining resource/no-update/commit failures."""
    settings = dict(contract["integrator"])
    price = float(contract["objective"]["price"])
    mode, budget = settings["barrier_mode"], settings["budget"]
    if price <= 0:
        raise ValueError("this working pipeline requires positive price")
    if mode == "off" and budget is not None:
        raise ValueError("off mode requires budget=null in the pipeline contract")
    if mode != "off" and budget is None:
        raise ValueError("a constrained mode requires an explicit positive budget")
    if contract["runtime"]["native_z_warm_start"]:
        raise ValueError("native-z warm starts are not part of this pipeline")

    started = time.perf_counter()
    # A private generator keeps caller RNG state unchanged.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = torch.float64

    def randn(*shape):
        return torch.randn(*shape, dtype=dtype, generator=generator)

    requests, contexts, tokens, din, dout, vocab = 3, 2, 5, 8, 6, 11
    sequences = requests * contexts
    weight = (0.15 * randn(dout, din)).float()
    entry = weight.clone()
    history_keys = 0.15 * randn(din, 4)
    history = (history_keys @ history_keys.T).float()
    history_entry = history.clone()
    all_keys = randn(sequences, tokens, din)
    # Two synthetic contexts/request; this is not the native context extractor.
    keys = all_keys[:, 1].reshape(requests, contexts, din).mean(1).T.contiguous()
    ridge = float(contract["native_writer"]["ridge"])
    writer = native_map(keys, torch.eye(din, dtype=dtype), history.double(), ridge)
    geom = QuadraticGeometry(
        preservation_gram(writer, history.double(), ridge), **contract["geometry"])
    residual = 0.2 * randn(sequences, tokens, dout)
    h_entry = residual + all_keys @ entry.double().T
    head = randn(dout, vocab) / dout**0.5
    causal_mask = torch.ones((tokens, tokens), dtype=torch.bool).triu(1)

    def suffix(hidden):
        scores = hidden @ hidden.transpose(-1, -2) / dout**0.5
        attention = scores.masked_fill(causal_mask, -torch.inf).softmax(-1)
        return torch.tanh(hidden + attention @ hidden) @ head

    with torch.no_grad():
        baseline_logits = suffix(h_entry)
    # New labels differ from entry top-1 at two synthetic prediction positions.
    labels = (baseline_logits[:, -2:].argmax(-1) + 1) % vocab
    teacher = baseline_logits[:, 0].log_softmax(-1).detach()
    caches = []
    for start, stop in ((0, 1), (1, 4), (4, 6)):
        count = stop-start
        edit_positions = torch.tensor(
            [(i, t) for i in range(count) for t in (tokens-2, tokens-1)],
            dtype=torch.long)
        caches.append(make_affine_cache(
            h_entry[start:stop], all_keys[start:stop], writer,
            edit_positions=edit_positions,
            edit_labels=labels[start:stop].reshape(-1),
            edit_weights=torch.full((count*2,), 1/(sequences*2), dtype=dtype),
            kl_positions=torch.tensor([(i, 0) for i in range(count)], dtype=torch.long),
            teacher_log_probs=teacher[start:stop],
            kl_weights=torch.full((count,), 1/sequences, dtype=dtype),
        ))
    preparation_seconds = time.perf_counter()-started
    work = {"logical_oracle_calls": 0, "suffix_forward_microbatches": 0,
            "suffix_forward_tokens": 0, "suffix_backward_microbatches": 0,
            "suffix_backward_tokens": 0}

    def counted_suffix(hidden):
        work["suffix_forward_microbatches"] += 1
        work["suffix_forward_tokens"] += hidden.shape[0]*hidden.shape[1]
        # This callback is only used by the full F+B reference oracle.
        work["suffix_backward_microbatches"] += 1
        work["suffix_backward_tokens"] += hidden.shape[0]*hidden.shape[1]
        return suffix(hidden)

    cached_oracle = CachedSuffixOracle(
        counted_suffix, caches, beta=contract["objective"]["beta_essence"])
    initial_loss = None

    def oracle(x):
        nonlocal initial_loss
        value, gradient = cached_oracle(x)
        work["logical_oracle_calls"] += 1
        if initial_loss is None:
            initial_loss = value
        return value, gradient

    ledger = {}
    transaction = InMemoryBatchTransaction("synthetic-batch-001", weight, history, ledger)
    solve_started = time.perf_counter()
    result = integrate(oracle, torch.zeros((dout, requests), dtype=dtype), geom,
                       price=price, **settings)
    solve_seconds = time.perf_counter()-solve_started
    assert result.oracle_calls == 1+result.accepted_steps+result.rejected_steps
    assert work["logical_oracle_calls"] == result.oracle_calls
    assert torch.equal(weight, entry) and torch.equal(history, history_entry)

    parity = {"checked": False, "full_write_forward_tokens": 0,
              "cached_terminal_forward_tokens": 0}
    receipt = None
    commit_status = "NO_ACCEPTED_UPDATE"
    commit_error = None
    commit_started = time.perf_counter()
    if result.accepted_steps:
        # Compare the original accepted FP64 cache prediction against actual
        # rounded FP32 weight materialization, not a re-anchored cached target.
        with torch.no_grad():
            expected = suffix(h_entry + (all_keys @ writer.T) @ result.x.T)
        parity["cached_terminal_forward_tokens"] = sequences*tokens

        def parity_checker(candidate, actual_delta):
            del actual_delta
            with torch.no_grad():
                actual = suffix(residual + all_keys @ candidate.double().T)
                difference = (actual-expected).abs()
                parity.update(checked=True, full_write_forward_tokens=sequences*tokens,
                              maximum_absolute_logit_error=float(difference.max()),
                              atol=2e-6, rtol=2e-6)
                return bool(torch.allclose(actual, expected, atol=2e-6, rtol=2e-6))
        try:
            kwargs = dict(cost_budget=budget, lambda_w=ridge, request_count=requests)
            receipt = transaction.commit(result.x.float(), writer.float(), keys.float(),
                                         parity_checker, **kwargs)
            replay = transaction.commit(result.x.float(), writer.float(), keys.float(),
                                        parity_checker, **kwargs)
            assert replay.replayed
            commit_status = "COMMITTED"
        except CommitParityError as error:
            commit_status, commit_error = "COMMIT_PARITY_FAIL", str(error)
        except CommitBudgetError as error:
            commit_status, commit_error = "COMMIT_COST_FAIL", str(error)
    if receipt is None:
        assert torch.equal(weight, entry) and torch.equal(history, history_entry)
    else:
        torch.testing.assert_close(history, history_entry + keys.float() @ keys.float().T,
                                   rtol=0, atol=0)

    def finite_trace(row):
        return {key: (None if isinstance(value, float) and not torch.isfinite(
            torch.tensor(value, dtype=dtype)) else value) for key, value in row.items()}

    return {
        "scope": "synthetic_cpu_wiring_check_not_knowledge_editing_performance",
        "seed": seed,
        "settings": contract,
        "settings_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
        "torch_version": str(torch.__version__),
        "geometry_dtype": "float64", "committed_weight_dtype": "float32",
        "requests": requests, "contexts_per_request": contexts,
        "tokens_per_sequence": tokens,
        "initial_L": initial_loss,
        "solver_status": result.status,
        "commit_status": commit_status,
        "commit_error": commit_error,
        "terminal": result.terminal_stats,
        "oracle_calls": result.oracle_calls,
        "accepted_steps": result.accepted_steps,
        "rejected_steps": result.rejected_steps,
        "work": work,
        "preparation": {"native_z_fits": 0, "writer_factorizations": 1,
                        "eigendecompositions": 1,
                        "teacher_forward_sequences": sequences,
                        "teacher_forward_tokens": sequences*tokens},
        "parity": parity,
        "commit_receipt": None if receipt is None else asdict(receipt),
        "duplicate_commit_was_noop": receipt is not None,
        "timings_seconds": {"preparation": preparation_seconds, "flow": solve_seconds,
                            "terminal_and_commit": time.perf_counter()-commit_started},
        "trace": [finite_trace(row) for row in result.trace],
        "llama_adapter_tested": False,
        "durable_transaction_tested": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run_demo(json.loads(args.contract.read_text()))
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print(json.dumps({key: result[key] for key in (
            "scope", "solver_status", "commit_status", "oracle_calls",
            "accepted_steps", "rejected_steps")}, ensure_ascii=False))
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
