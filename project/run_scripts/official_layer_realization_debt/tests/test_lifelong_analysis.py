from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt

from project.run_scripts.official_layer_realization_debt.lifelong_analysis import write_csv_once
from project.run_scripts.official_layer_realization_debt.lifelong_analysis_figures import (
    FigureTables,
    load_figure_tables,
    run_cli,
    weight_magnitude_figure,
)
from project.run_scripts.official_layer_realization_debt.lifelong_analysis_io import (
    ArmSpec,
    InputLock,
    canonical_hash,
    extend_hash_chain,
    validate_arm,
)
from project.run_scripts.official_layer_realization_debt.lifelong_analysis_tables import (
    _attach_checkpoint_action_realization,
    _paired,
    summary,
)


MODELS = ("llama3-8b-inst", "qwen2.5-7b-inst")
METHODS = ("memit", "alphaedit")
LAYERS = (4, 5, 6, 7, 8)
CHECKPOINTS = (0, 1000, 1500, 2000, 3000, 5000, 7500, 10000)


def _write600(path: Path, payload: object | bytes) -> str:
    raw = payload if isinstance(payload, bytes) else (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


class SummaryAndPairingTest(unittest.TestCase):
    def test_summary_has_requested_statistics(self) -> None:
        value = summary([1, 2, 3, 4])
        self.assertEqual(value["n"], 4)
        self.assertEqual(value["median"], 2.5)
        self.assertEqual(value["iqr"], 1.5)
        self.assertEqual(value["max"], 4)

    def test_pairing_is_hash_matched_and_has_deterministic_ci(self) -> None:
        rows = []
        for arm, shift in (("a", 2.0), ("b", 0.0)):
            for index in range(10):
                rows.append({"arm": arm, "request": f"r{index}", "metric": index + shift})
        first = _paired(rows, lambda row: row["arm"] == "a", lambda row: row["arm"] == "b", ("request",), ("metric",), "a-b")
        second = _paired(rows, lambda row: row["arm"] == "a", lambda row: row["arm"] == "b", ("request",), ("metric",), "a-b")
        self.assertEqual(first, second)
        self.assertEqual(first[0]["paired_n"], 10)
        self.assertEqual(first[0]["mean"], 2.0)
        self.assertEqual(first[0]["paired_mean_bootstrap95_low"], 2.0)

    def test_checkpoint_action_realization_uses_one_shared_batch_weight_profile(self) -> None:
        requests = [{"request_sha256": "a"}, {"request_sha256": "b"}]
        layers = []
        for request in ("a", "b"):
            for layer in LAYERS:
                layers.append(
                    {
                        "request_sha256": request,
                        "layer": layer,
                        "rho": 1.0,
                        "allocation_norm_over_R1": 0.2,
                    }
                )
        weights = [
            {"layer": layer, "update_magnitude_share": share}
            for layer, share in zip(LAYERS, (0.1, 0.2, 0.3, 0.2, 0.2), strict=True)
        ]
        _attach_checkpoint_action_realization(requests, layers, weights)
        for row in requests:
            self.assertAlmostEqual(row["D_TV"], 0.1)
        self.assertEqual([row["negative_progress_count"] for row in requests], [0, 0])
        self.assertTrue(all(row["positive_progress_total"] == 1.0 for row in requests))


class HashChainValidationTest(unittest.TestCase):
    def test_two_batch_chain_and_state_bytes_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cell"
            stream_path = Path(temporary) / "stream.json"
            stream_sha = _write600(stream_path, {"seal": True})
            model, method, campaign = "llama3-8b-inst", "memit", "test"
            initial = {f"w{layer}": "0" * 63 + str(layer - 4) for layer in LAYERS}
            chain = canonical_hash({"campaign_id": campaign, "genesis": True})
            checkpoint_refs = []
            journal_refs = []

            def checkpoint(batch: int) -> None:
                nonlocal chain
                state_path = root / "checkpoints" / f"state-{batch * 100:05d}.pt"
                state_sha = _write600(state_path, f"state-{batch}".encode())
                state = {"path": str(state_path), "sha256": state_sha, "bytes": state_path.stat().st_size, "batch_index": batch, "accepted_edit_count": batch * 100, "journal_chain_root": chain}
                cp_observer = {
                    "status": "TERMINAL_OBSERVATION_VALID",
                    "request_count": 100,
                    "direct_z_compute_count": 100,
                    "direct_z_recompute_count": 0,
                    "direct_z_shared_replay_count": 0,
                    "layer_loop_pass_through_call_count": 5,
                    "layer_loop_observation_copy_count": 5,
                    "terminal_post_L8_forward_count": 1,
                    "decision_or_update_tensor_mutation_count": 0,
                    "raw_prompt_logit_publish_count": 0,
                    "residual_debt": {
                        "nonfinite_count": 0,
                        "maximum_recurrence_closure_relative_error": 0.0,
                        "records": [{}] * 100,
                    },
                    "weight_action": {"layers": [{}] * 5},
                }
                primary = {
                    "ordered": {
                        "layer_realization_observer": cp_observer,
                        "official_call_audit": {
                            "compute_ks_call_count": 5,
                            "torch_linalg_solve_call_count": 5,
                            "module_compute_ks_identity_restored": True,
                            "torch_linalg_solve_identity_restored": True,
                            "output_or_decision_mutation_count": 0,
                        },
                    },
                    "same_entry": {
                        "activation_forward_count": 6,
                        "additional_compute_z": 0,
                        "additional_key_compute": 0,
                        "additional_solve": 0,
                        "layers": [{}] * 5,
                        "restore": {"weights": {"exact": True}, "cache": {"exact": True}},
                    },
                    "completion_geometry": {
                        "A0": 1.0,
                        "V_to_go": 1.0,
                        "Vbar": 1.0,
                        "minimum_completion_budget_h_normalized": 0.0,
                        "unreachable_fraction": 0.0,
                        "full_parameter_space_claim": False,
                        "waypoints": [
                            {"waypoint": value, "remaining_layer_count": 5 - value}
                            for value in range(6)
                        ],
                    },
                }
                probe = {
                    "sentinel_request_count": 100,
                    "direct_z_recompute_count": 0,
                    "direct_z_optimizer_count": 100,
                    "shared_z_fork_replay_count": 0,
                    "probe_contamination": {
                        "weight_restore_exact": True,
                        "cache_restore_exact": True,
                        "accepted_history_append_count": 0,
                        "decision_influence_count": 0,
                    },
                    "primary": primary,
                    "reset_cache_fork": None,
                }
                def panel(count: int, *, low_cost: bool = False) -> dict[str, object]:
                    records = []
                    for index in range(count):
                        measured = lambda n: {"count": n, "nll_mean": 0.0, "margin_mean": 0.0, "strict_count": 0}
                        records.append(
                            {
                                "request_sha256": f"request-{batch}-{index}",
                                "case_identity_sha256": f"case-{index}",
                                "raw_prompt_logit_generation_publish_count": 0,
                                "rewrite_target_new": measured(1),
                                "rewrite_target_true": measured(1),
                                "rephrase_target_new": ({"status": "NOT_RECORDED_LOW_COST_BATCH"} if low_cost else measured(2)),
                                "locality_target_true": ({"status": "NOT_RECORDED_LOW_COST_BATCH"} if low_cost else measured(10)),
                            }
                        )
                    return {"request_count": count, "controller_or_selection_influence_count": 0, "records": records}

                functional = (
                    {"sentinel_pre_edit": panel(100)}
                    if batch == 0
                    else {
                        "controller_or_selection_influence_count": 0,
                        "current": panel(100),
                        "retention": {
                            "earliest": panel(32, low_cost=True),
                            "recent": panel(32, low_cost=True),
                            "hash_stratified": panel(64, low_cost=True),
                        },
                    }
                )
                payload = {"model": model, "method": method, "batch_index": batch, "accepted_edit_count": batch * 100, "journal_chain_root": chain, "state": state, "probe": probe, "functional": functional, "scientific_promotion": False}
                path = root / "checkpoints" / f"checkpoint-{batch * 100:05d}.json"
                digest = _write600(path, payload)
                chain = extend_hash_chain(chain, digest, 10_000 + batch)
                checkpoint_refs.append({"batch_index": batch, "accepted_edit_count": batch * 100, "path": str(path), "sha256": digest, "state": state, "chain_after": chain})

            checkpoint(0)
            previous = initial
            for batch in (1, 2):
                commit = {key: hashlib.sha256(f"{key}-{batch}".encode()).hexdigest() for key in initial}
                debt_record = {"request_sha256": "r", "case_identity_sha256": "c", "q": [1, .8, .6, .4, .2, 0], "q_L8_gt_0_2": False, "d_parallel": 0.0, "d_perp": 0.0, "recurrence_closure_relative_error": 0.0, "layers": []}
                observer = {"status": "TERMINAL_OBSERVATION_VALID", "request_count": 100, "direct_z_compute_count": 100, "direct_z_recompute_count": 0, "layer_loop_pass_through_call_count": 5, "layer_loop_observation_copy_count": 5, "terminal_post_L8_forward_count": 1, "decision_or_update_tensor_mutation_count": 0, "raw_prompt_logit_publish_count": 0, "module_global_identity_restored": True, "residual_debt": {"nonfinite_count": 0, "maximum_recurrence_closure_relative_error": 0.0, "records": [debt_record] * 100}, "weight_action": {"scalar_reduction_dtype": "float64", "share_sum": 1.0, "update_magnitude_share_sum": 1.0, "layers": [{"layer": layer} for layer in LAYERS]}}
                cache_sha = "c" * 64
                payload = {"status": "B100_ATOMIC_TERMINAL_VALID", "model": model, "method": method, "batch_index": batch, "request_count": 100, "accepted_edit_start": (batch - 1) * 100, "accepted_edit_end": batch * 100, "apply": {"layer_realization_observer": observer, "official_call_audit": {"compute_ks_call_count": 5, "torch_linalg_solve_call_count": 5, "module_compute_ks_identity_restored": True, "torch_linalg_solve_identity_restored": True, "output_or_decision_mutation_count": 0}}, "endpoint": {"records": [{}] * 100}, "action_realization": {"requests": [{}] * 100}, "weight_continuity": {"entry_sha256": previous, "commit_sha256": commit, "entry_matches_previous_commit": True, "official_original_copy_matches_entry": True}, "cache_continuity": {"entry_sha256": cache_sha, "exit_sha256": cache_sha, "entry_width": 0, "exit_width": 0, "append_width": 0, "consume_width": 0, "checks": {"entry_reuse_after_first": True, "historical_decision_state_false": True, "request_history_width_zero": True, "silent_reset_zero": True, "static_covariance_identity_link": True}}, "first_valid_gate": {"status": "FIRST_B100_OBSERVER_ON_OFF_PARITY_PASS"} if batch == 1 else None}
                path = root / "journals" / f"batch-{batch:03d}.json"
                digest = _write600(path, payload)
                chain = extend_hash_chain(chain, digest, batch)
                journal_refs.append({"batch_index": batch, "path": str(path), "sha256": digest, "request_count": 100, "chain_after": chain, "weight_commit_sha256": commit, "cache_exit_sha256": cache_sha})
                previous = commit
                if batch == 2:
                    checkpoint(2)
            result = {"status": "LIFELONG_10K_TERMINAL_VALID", "campaign_id": campaign, "model": model, "method": method, "source": {"head": "h", "tree": "t", "tracked_clean": True}, "stream": {"path": str(stream_path), "sha256": stream_sha, "root": "sr", "order": "or", "sentinel_order": "so", "sample_duplication_count": 0}, "model_binding": {"full_fp32": True, "padding_gate": {"status": "PASS", "padding_side": "left", "batch_reorder_identity": True, "padding_length_identity": True}}, "initial_w0": {"sha256": initial, "dtypes": {key: "torch.float32" for key in initial}}, "terminal_w0_restore": {"exact": True, "sha256": initial, "dtypes": {key: "torch.float32" for key in initial}}, "terminal_cache_restore": {"exact": True}, "terminal_committed_weight_sha256": previous, "valid_batch_denominator": 2, "valid_request_denominator": 200, "checkpoint_denominator": 2, "nonfinite_count": 0, "rollback_violation_count": 0, "target_recomputation_count": 0, "scientific_promotion": False, "journals": journal_refs, "checkpoints": checkpoint_refs, "journal_chain_root": chain}
            result["identity_sha256"] = canonical_hash(result)
            result_path = root / "result.json"
            _write600(result_path, result)
            spec = ArmSpec(model, method, "1", result_path, "h", "t")
            lock = InputLock((spec,), "sr", "or", "so", stream_sha, Path(temporary), "eh", "et", "ev", (0, 2))
            validated = validate_arm(spec, lock, expected_batches=2, expected_requests=200, expected_checkpoints=2)
            self.assertEqual(len(validated.journals), 2)
            self.assertEqual(len(validated.raw_members), 7)
            (root / "journals" / "batch-002.json").write_text("tampered")
            with self.assertRaisesRegex(Exception, "SHA differs"):
                validate_arm(spec, lock, expected_batches=2, expected_requests=200, expected_checkpoints=2)


def _figure_rows() -> FigureTables:
    cp_request, cp_layer, cp_weight, production_weight, functional, compute = [], [], [], [], [], []
    for model in MODELS:
        for method in METHODS:
            for accepted in CHECKPOINTS:
                for request in range(2):
                    cp_request.append({"model": model, "method": method, "accepted_edit_count": accepted, "fork": "ACCUMULATED_OR_STATIC", "q_pre_L4": 1, "q_pre_L5": .8, "q_pre_L6": .6, "q_pre_L7": .4, "q_pre_L8": .3, "q_post_L8": .1, "q_pre_L8_gt_0_2": 1, "d_parallel": .1, "d_perp": .2})
                    for layer in LAYERS:
                        cp_layer.append({"model": model, "method": method, "accepted_edit_count": accepted, "fork": "ACCUMULATED_OR_STATIC", "layer": layer, "rho": .9, "tau": .1})
                for layer in LAYERS:
                    cp_weight.append({"model": model, "method": method, "accepted_edit_count": accepted, "fork": "ACCUMULATED_OR_STATIC", "layer": layer, "frobenius_magnitude": 1})
            for batch in range(1, 101):
                for layer in LAYERS:
                    production_weight.append({"model": model, "method": method, "batch_index": batch, "layer": layer, "frobenius_magnitude": float(layer + batch / 100)})
            for accepted in CHECKPOINTS[1:]:
                for metric in ("rewrite_target_new", "rephrase_target_new", "locality_target_true"):
                    for request in range(2):
                        functional.append({"model": model, "method": method, "accepted_edit_count": accepted, "cohort": "current", "metric": metric, "availability": "RECORDED", "strict_rate": request / 2})
                functional.append({"model": model, "method": method, "accepted_edit_count": accepted, "cohort": "retention_earliest", "metric": "rephrase_target_new", "availability": "NOT_RECORDED_LOW_COST_BATCH", "strict_rate": "NOT_RECORDED_SCHEMA"})
            compute.append({"model": model, "method": method, "production_edit_core_wall_seconds_sum": 3600, "production_observer_wall_seconds_sum": 100, "cell_wall_seconds": 4000})
    return FigureTables(cp_request, cp_layer, cp_weight, production_weight, functional, compute)


class FigureTest(unittest.TestCase):
    def test_weight_title_panel_order_and_no_reference_line_or_bars_text(self) -> None:
        figure = weight_magnitude_figure(_figure_rows())
        self.assertEqual(figure._suptitle.get_text(), "Layer-wise Update Magnitude")
        expected = ["Llama-3-8B-Instruct / Official MEMIT", "Llama-3-8B-Instruct / Official AlphaEdit", "Qwen2.5-7B-Instruct / Official MEMIT", "Qwen2.5-7B-Instruct / Official AlphaEdit"]
        self.assertEqual([axis.get_title() for axis in figure.axes[:4]], expected)
        # IQR cap markers are allowed; an ideal/equal-allocation reference
        # would be a visible Line2D stroke and is forbidden on this panel.
        self.assertTrue(
            all(
                all(line.get_linestyle() in {"None", "none", "", " "} for line in axis.lines)
                for axis in figure.axes[:4]
            )
        )
        self.assertNotIn("bars:", " ".join(text.get_text().lower() for axis in figure.axes for text in axis.texts))
        plt.close(figure)

    def test_plot_cli_consumes_sealed_tables_and_emits_eleven_figures(self) -> None:
        tables = _figure_rows()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = (("checkpoint-request-complete.csv.gz", tables.checkpoint_request, True), ("checkpoint-layer-complete.csv.gz", tables.checkpoint_layer, True), ("checkpoint-weight-complete.csv.gz", tables.checkpoint_weight, True), ("production-weight-batch-unit-complete.csv.gz", tables.production_weight, True), ("functional-endpoint-complete.csv.gz", tables.functional, True), ("compute-accounting.csv", tables.compute, False))
            for name, rows, compressed in inputs:
                write_csv_once(root / name, rows, compressed=compressed)
            seal = {
                "schema": "test",
                "tables": [
                    {"path": name, "rows": len(rows), "bytes": (root / name).stat().st_size, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
                    for name, rows, _ in inputs
                ],
            }
            seal["identity_sha256"] = canonical_hash(seal)
            (root / "derived-table-seal.json").write_text(json.dumps(seal))
            loaded = load_figure_tables(root)
            self.assertEqual(len(loaded.production_weight), 2000)
            self.assertTrue(
                any(row["strict_rate"] == "NOT_RECORDED_SCHEMA" for row in loaded.functional)
            )
            receipt = run_cli(root, root)
            self.assertEqual(len(receipt["figures"]), 11)
            self.assertEqual(receipt["input_row_counts"]["production_weight"], 2000)
            self.assertEqual(receipt["panel_order"][0], ["llama3-8b-inst", "memit"])
            self.assertTrue(all((root / row["path"]).is_file() for row in receipt["figures"]))
            second = root / "second-render"
            second.mkdir()
            second_receipt = run_cli(root, second)
            self.assertEqual(
                {row["path"]: row["sha256"] for row in receipt["figures"]},
                {row["path"]: row["sha256"] for row in second_receipt["figures"]},
            )


if __name__ == "__main__":
    unittest.main()
