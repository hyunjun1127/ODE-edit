"""Synthetic CPU fixtures only; no access to running or historical raw roots."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from .analysis_s import (AnalysisBoundary, MISSING, SUCCESS, analyze_cell, build_package,
                         load_frozen, paired, plot_pngs, report_ko, runtime_integrity)
from .publication import digest, member, verify_package, write_once
from .sweep import MODELS, endpoints, endpoint_binding, paths


def evaluation(records, shift=0.):
    out = {}
    for kind, count in (("rewrite", 1), ("rephrase", 2), ("locality", 10)):
        for target in ("new", "true"):
            label = kind + "_target_" + target
            rows = []
            token = 2 if target == "new" else 3
            for i, record in enumerate(records):
                for prompt in range(count):
                    # First case ties exactly; second fails preference; others hit.
                    new = 1. + shift + (i % 3) * .1 + prompt * .001
                    true = new if i == 0 else new - .2 if i == 1 else new + .4
                    correct = i % 2 == 0 and prompt % 2 == 0
                    rows.append(dict(case_id=record["case_id"], kind=label, prompt_index=prompt,
                        prompt=f"SYNTHETIC_PRIVATE_PROMPT_{kind}_{i}_{prompt}", target="new" if token == 2 else "true",
                        target_token_ids=[token, token], nll=new if target == "new" else true,
                        token_predictions=[token, token] if correct else [token, 4],
                        token_correct=[True, correct], all_tokens_correct=correct))
            out[label] = rows
    return out


def fixture(model=MODELS[0], count=3):
    records = [dict(case_id=i + 1, request_sha256=digest([i]), fixture="S_DEV") for i in range(count)]
    cohort = dict(records=records, request_order_sha256=digest([r["request_sha256"] for r in records]),
        ordered_root=digest(records), evaluation={"denominators": {"RS": count, "PS": 2 * count, "NS": 10 * count}})
    source = dict(head="a" * 40, tree="b" * 40)
    z, w, m = "c" * 64, "d" * 64, "e" * 64
    all_specs = {s.candidate_id: s for s in endpoints()}
    trajectories, endpoint_results = [], {}
    for path in paths():
        nodes = []
        for n in range(path.config.N):
            active = [4, 5, 6, 7, 8]
            # Squared step and net differ, deliberately not additive norms.
            physical = [dict(layer=l, actual_step_DeltaW_squared=(l - 3) * .001,
                actual_net_DeltaW_squared=(l - 3) * .001 * (n + 1)**2, actual_nonzero=l) for l in active]
            actions = [dict(layer=l, direction_raw_action=2., direction_frobenius_squared=.2,
                raw_native_velocity_action=.02, normalized_native_velocity_action=.01,
                history_velocity_action=.012, L2_velocity_action=.008, frobenius_velocity_squared=.002) for l in active]
            node = dict(node=n, **path.config.receipt(), t=(n + 1) * path.config.h,
                V_before=.5, V_after=.5 - .01 * (n + 1), V0=.5, V_ratio=1 - .02 * (n + 1),
                E=.05 * (n + 1) * path.config.h, native_path_length_normalized=.3 * (n + 1),
                qN=.05, qN_ref=2., gain=.2, active_layers=active,
                c=[.1] * 5, g=[.4] * 5, q_layers=[1.] * 5,
                raw_physical_coefficients=[.1] * 5, actual_step_coefficients=[path.config.h * .1] * 5,
                predicted_target_contribution=[.04] * 5, full_H=[[1. if i == j else 0. for j in range(5)] for i in range(5)],
                G=[[1. if i == j else 0. for j in range(5)] for i in range(5)],
                source_entry_sha256=w, fixed_z_sha256=z, actual_physical=physical, layer_actions=actions,
                KKT_stationarity=0., KKT_complementarity=0., model_error_normalized=.002,
                inner_weight_mutation_count=0, inner_history_append_count=0, controller_heldout_access_count=0,
                raw_residual_norms_by_request=[.1] * count,
                training_semantic_observation=dict(request_strict=[n == 1] * count, all_strict=n == 1,
                    strict_event_count=count if n == 1 else 0, event_count=count,
                    request_strict_count=count if n == 1 else 0, strict_tie_count=0),
                compute_seconds=dict(native_dictionary=.1, main_jvp=.2, primary_NNLS=.001),
                shadows={"single_layer": []})
            nodes.append(node)
        observations = {}
        for label in path.endpoint_ids:
            spec = all_specs[label]
            binding = endpoint_binding(spec, path)
            ep = dict(status="TERMINAL_VALID", evaluation=evaluation(records, .01),
                **path.config.endpoint_clock(spec.completed_nodes), candidate_id=label,
                history_append_count=0 if spec.derived_observation_only else 1,
                selected_weight_endpoint_sha256=digest(label), terminal_net_frobenius_squared=.3)
            observations[label] = ep
            endpoint_results[label] = dict(observation=ep, binding=binding)
        result = dict(status="TERMINAL_VALID", nodes=nodes, endpoints=observations,
            main_jvp_count=5 * path.config.N, dictionary_build_count=path.config.N,
            solve_count=5 * path.config.N, total_write_and_endpoint_seconds=float(path.config.N),
            native_net_raw=.04, native_net_normalized=.02, jvp_ledger={"jvp_call_count": 5 * path.config.N})
        trajectories.append(dict(path_id=path.path_id, result=result))
    result = dict(status=SUCCESS, model=model, fixed_z_bundle_sha256=z,
        entry_W_sha256=w, entry_M_sha256=m, entry_restore=True,
        entry_evaluation=evaluation(records),
        Official_endpoint=dict(status="TERMINAL_VALID", evaluation=evaluation(records),
            selected_weight_endpoint_sha256=digest("official"), history_append_count=1),
        endpoints=endpoint_results, trajectories=trajectories, first_fidelity={"status": "PASS"})
    terminal = dict(status=SUCCESS, model_alias=model, source_head=source["head"], source_tree=source["tree"],
        JV_endpoints=9, actual_paths=7, completed_nodes=34, official_endpoints=1,
        requests_per_endpoint=count, fixed_z_capture_count=1, fixed_z_recompute_count=0,
        qref_capture_count=1, audit_outcomes_opened=0, source_science_change_count=0, tolerance_change_count=0,
        W0_restore=True, M_restore=True, entry_W_sha256=w, entry_M_sha256=m,
        S_DEV_order_sha256=cohort["request_order_sha256"], completed=["O_NATIVE", *all_specs])
    return terminal, result, cohort, source


def frozen_files(base, *, fail_qwen=False):
    root = base / "raw"
    root.mkdir()
    seal = dict(root=str(root), frozen=True, model_or_gpu_actions=0, cells={})
    cohort = source = None
    for cell, model in enumerate(MODELS):
        terminal, result, cohort, source = fixture(model, 100)
        directory = root / f"cell-{cell}"
        directory.mkdir()
        if cell == 1 and fail_qwen:
            state = dict(status="TECHNICAL_FIDELITY_EXCEPTION", stage="Fidelity", completed=[],
                exception_type="RuntimeError", original_exception_receipt={}, entry_restore={"W0_bytes": True, "M_bytes": True})
            marker = "failure-boundary.json"
        else:
            state, marker = terminal, "terminal.json"
            write_once(directory / "sweep-result.json", result, root=root)
            write_once(directory / "compute-accounting.json", dict(model_load_seconds=2., model_forward_invocations=500,
                process_residency_seconds=44., scheduler_GPU_seconds="AUTHORITATIVE_SACCT_RECONCILIATION_AFTER_EXIT"), root=root)
            write_once(directory / "runtime.lock.json", runtime_fixture(model, source, cohort), root=root)
        write_once(directory / marker, state, root=root)
        seal["cells"][str(cell)] = dict(marker=marker, sha256=member(directory / marker)["sha256"])
    sample = dict(cohorts={"S_DEV": cohort}, audit_outcomes_opened=0)
    sample["manifest_identity"] = digest(sample)
    write_once(root / "sample.lock.json", sample, root=root)
    write_once(root / "source.lock.json", source, root=root)
    seal_path = base / "frozen.json"
    write_once(seal_path, seal, root=base)
    return root, seal_path


def runtime_fixture(model, source, cohort):
    return dict(model_alias=model, family="AlphaEdit", source=source,
        model_storage_forward_dtype="float32", controller_dtype="float64", autocast=False, tf32=False,
        independent_cold_S_DEV=True, D_restore_dependency=False,
        sample_root=cohort["ordered_root"], request_order_sha256=cohort["request_order_sha256"],
        dtype={"parameter_dtype_counts": {"torch.float32": 12}}, official={"tracked_clean": True, "head": "1" * 40})


class ScientificReductionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture()
        self.tables = analyze_cell(*self.fixture)

    def test_exact_nine_endpoints_seven_paths_34_nodes(self):
        self.assertEqual(len(self.tables["main"]), 11)
        self.assertEqual(len(self.tables["clocks"]), 9)
        self.assertEqual(len(self.tables["nodes"]), 34)
        self.assertEqual(len(self.tables["layers"]), 170)
        self.assertEqual(len([r for r in self.tables["compute"] if r["scope"] == "ACTUAL_PATH"]), 7)

    def test_tie_failure_and_ns_direction_exact(self):
        row = self.tables["main"][0]
        self.assertEqual((row["RS_n"], row["RS_d"]), (1, 3))
        self.assertEqual((row["PS_n"], row["PS_d"]), (2, 6))
        self.assertEqual((row["NS_n"], row["NS_d"]), (10, 30))
        self.assertEqual(row["rephrase_strict_request_accuracy_n"], 0)
        self.assertGreater(row["rephrase_accuracy_n"], row["rephrase_strict_n"])

    def test_prefix_clock_and_cost_not_parent_duplicate(self):
        row = next(r for r in self.tables["main"] if r["candidate"] == "JV-BASE")
        self.assertEqual((row["effective_T"], row["effective_N"], row["parent_T"], row["parent_N"]), (2., 4, 4., 8))
        self.assertEqual(row["E_T"], self.fixture[1]["trajectories"][0]["result"]["nodes"][3]["E"])
        self.assertEqual(row["native_net_raw"], MISSING)
        self.assertEqual(row["endpoint_cost"], "SHARED_PARENT_PATH_NOT_ALLOCATED")

    def test_paired_direction_new_true_and_request_counts(self):
        rows = paired(self.tables["prompts"])
        rs = next(r for r in rows if r["candidate"] == "JV-BASE" and r["metric"] == "RS" and r["reference"] == "O_NATIVE" and r["field"] == "target_new_nll")
        self.assertEqual((rs["n"], rs["worse"]), (3, 3))
        self.assertAlmostEqual(rs["mean"], .01)
        ns = next(r for r in rows if r["candidate"] == "JV-BASE" and r["metric"] == "NS" and r["reference"] == "O_NATIVE" and r["field"] == "target_new_nll")
        self.assertEqual(ns["better"], 30)
        request = paired(self.tables["requests"], request_level=True)
        self.assertTrue(all(r["n"] == 3 for r in request))

    def test_actual_net_not_sum_of_step_magnitudes(self):
        rows = [r for r in self.tables["nodes"] if r["path_id"] == "S-PATH-T4"]
        self.assertNotEqual(rows[-1]["actual_net_DeltaW_squared"], sum(r["actual_step_DeltaW_squared"] for r in rows))
        layers = [r for r in self.tables["layers"] if r["path_id"] == "S-PATH-T4" and r["node"] == 0]
        self.assertAlmostEqual(sum(r["actual_step_energy_share"] for r in layers), 1.)

    def test_first_hit_observation_not_entry_inference(self):
        row = self.tables["first_hit"][0]
        self.assertEqual(row["first_observed_hit_node"], 2)
        self.assertTrue(row["transient_hit_terminal_miss"])
        self.assertEqual(row["entry_already_hit"], MISSING)
        self.assertFalse(row["first_hit_controls_dynamics"])

    def test_missing_node_metric_stays_missing(self):
        self.assertEqual(self.tables["nodes"][0]["finite_step_dissipation_defect"], MISSING)
        self.assertEqual(self.tables["main"][1]["E_T"], "NOT_APPLICABLE_NO_EULER_TRAJECTORY")

    def test_declared_science_not_performance_filter(self):
        values = copy.deepcopy(self.fixture)
        for path in values[1]["trajectories"]:
            for node in path["result"]["nodes"]:
                node["V_ratio"] = 4.
                node["V_after"] = 2.
        result = analyze_cell(*values)
        self.assertTrue(all(row["V_ratio"] == 4. for row in result["main"] if row["candidate"].startswith("JV-")))

    def test_clock_restore_source_and_z_negative(self):
        for mutate in (
            lambda x: x[0].update(W0_restore=False),
            lambda x: x[0].update(source_tree="wrong"),
            lambda x: x[1]["trajectories"][0]["result"]["nodes"][0].update(fixed_z_sha256="wrong"),
            lambda x: x[1]["endpoints"]["JV-BASE"]["binding"].update(effective_T=4.),
        ):
            values = copy.deepcopy(self.fixture); mutate(values)
            with self.assertRaises(AnalysisBoundary):
                analyze_cell(*values)

    def test_nonfinite_missing_endpoint_and_pair_join_fail_closed(self):
        values = copy.deepcopy(self.fixture)
        values[1]["trajectories"][0]["result"]["nodes"][0]["qN"] = float("nan")
        with self.assertRaisesRegex(AnalysisBoundary, "NONFINITE"):
            analyze_cell(*values)
        values = copy.deepcopy(self.fixture); del values[1]["endpoints"]["JV-LAM-1"]
        with self.assertRaisesRegex(AnalysisBoundary, "NINE_ENDPOINT"):
            analyze_cell(*values)
        rows = [r for r in self.tables["prompts"] if r["candidate"] != "O_NATIVE"]
        with self.assertRaisesRegex(AnalysisBoundary, "MISSING_PAIRED"):
            paired(rows)

    def test_nested_component_not_summed_into_total(self):
        tables = analyze_cell(*self.fixture, compute=dict(process_residency_seconds=40., model_load_seconds=5.),
            components=[dict(accounting=dict(component="fixed_target_and_entry", wall_seconds=15., includes_nested_components=True)),
                        dict(accounting=dict(component="first_fidelity", wall_seconds=10.))])
        process = next(r for r in tables["compute"] if r["scope"] == "PROCESS_TOTAL")
        self.assertEqual(process["wall_seconds"], 40.)
        self.assertFalse(process["sum_nested_components"])

    def test_actual_official_metric_and_endpoint_layer_only(self):
        values = copy.deepcopy(self.fixture)
        values[1]["Official_endpoint"].update(terminal_net_frobenius_squared=.2,
            terminal_net_frobenius_squared_by_weight={"model.layers.4.mlp.down_proj.weight": .2})
        tables = analyze_cell(*values, components=[dict(accounting={"component": "O_NATIVE"},
            payload={"actual_physical": {"native_net_raw": .3, "native_net_normalized": .15}})])
        self.assertEqual(tables["endpoint_layers"][0]["energy_share"], 1.)
        self.assertEqual(tables["endpoint_layers"][0]["layer"], 4)
        self.assertEqual(tables["main"][1]["native_net_raw"], .3)
        self.assertFalse(any(r["path_id"] == "O_NATIVE" for r in tables["nodes"]))

    def test_runtime_fp32_and_controller_fp64_binding(self):
        _, _, cohort, source = self.fixture
        runtime = runtime_fixture(MODELS[0], source, cohort)
        self.assertEqual(runtime_integrity(runtime, MODELS[0], source, cohort)["status"], "RECORDED_RUNTIME_BINDING_PASS")
        runtime["dtype"]["parameter_dtype_counts"]["torch.bfloat16"] = 1
        with self.assertRaisesRegex(AnalysisBoundary, "PARAMETER_DTYPE"):
            runtime_integrity(runtime, MODELS[0], source, cohort)

    def test_png_bytes_reproduced_from_same_tables(self):
        a = plot_pngs(self.tables); b = plot_pngs(self.tables)
        self.assertEqual(a, b)
        self.assertTrue(all(v.startswith(b"\x89PNG") for v in a.values()))


class FrozenPublicationTests(unittest.TestCase):
    def test_detail_tables_keep_prefix_cost_and_distinct_quantiles(self):
        from .analysis_details import augment_tables
        tables = analyze_cell(*fixture())
        augment_tables(tables)
        self.assertEqual(len(tables['paired_headline']), 18)
        self.assertEqual(len(tables['kernel_cost']), 7)
        self.assertEqual(len(tables['endpoint_mechanics']), 10)
        self.assertTrue(all(r['distribution_tail_delta_is_not_paired_delta_quantile'] for r in tables['paired_headline']))
        self.assertEqual(tables['budget'], [])
        self.assertEqual(tables['first_hit_summary'][0]['transient_hit_terminal_miss'], 3)

    def test_accounting_identity_and_unfinished_job_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _, original = frozen_files(base)
            seal = json.loads(original.read_text())
            account = dict(jobs=[dict(state='RUNNING', live_state='RUNNING')])
            account['identity'] = digest(account)
            ledger = base/'account.json'
            write_once(ledger, account, root=base)
            seal['accounting'] = member(ledger)
            updated = base/'account-seal.json'
            write_once(updated, seal, root=base)
            with self.assertRaisesRegex(AnalysisBoundary, 'SCHEDULER_NOT_TERMINAL'):
                load_frozen(updated)

    def test_frozen_two_model_package_and_raw_firewall(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, seal = frozen_files(base)
            before = member(root / "cell-0/sweep-result.json")
            out = base / "reports/s-terminal-v1"
            verification = build_package(seal, out, analysis_source={"head": "f" * 40, "tree": "0" * 40}, approved_base=base / "reports")
            self.assertEqual(verification["status"], "PACKAGE_REHASH_PASS")
            self.assertEqual(before, member(root / "cell-0/sweep-result.json"))
            receipt = json.loads((out / "rooted-receipt.json").read_text())
            self.assertEqual((receipt["JV_endpoint_count"], receipt["evaluated_observation_count"]), (18, 22))
            self.assertEqual(receipt["scope"], "FROZEN_S_EXPERIMENT_FACTUAL_ANALYSIS")
            self.assertFalse(receipt["raw_reconstruction_verified"])
            self.assertEqual(receipt["table_rows"]["nodes"], 68)
            self.assertEqual(receipt["table_rows"]["layers"], 340)
            for path in out.glob("*.csv"):
                self.assertNotIn("SYNTHETIC_PRIVATE_PROMPT", path.read_text())
                self.assertNotIn("target_token_ids", path.read_text())
            with self.assertRaisesRegex(AnalysisBoundary, "CREATE_ONCE"):
                build_package(seal, out, analysis_source={}, approved_base=base / "reports")
            self.assertEqual(verify_package(out)["receipt_identity"], receipt["identity"])

    def test_partial_failure_not_denominator_or_imputed(self):
        with tempfile.TemporaryDirectory() as directory:
            root, seal = frozen_files(Path(directory), fail_qwen=True)
            package = load_frozen(seal)
            self.assertEqual(package["complete_models"], 1)
            self.assertEqual(len(package["tables"]["main"]), 11)
            failed = package["tables"]["status"][1]
            self.assertEqual(failed["scientific_JV_endpoint_denominator"], 0)
            self.assertFalse(failed["terminal_valid"])
            self.assertIn("완료 모델 1/2", report_ko(package))

    def test_live_unpinned_changed_marker_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, seal_path = frozen_files(Path(directory), fail_qwen=True)
            seal = json.loads(seal_path.read_text())
            seal["frozen"] = False
            other = Path(directory) / "unfrozen.json"
            write_once(other, seal, root=Path(directory))
            with self.assertRaisesRegex(AnalysisBoundary, "FROZEN_INPUT"):
                load_frozen(other)
            seal["frozen"] = True; seal["cells"]["0"]["sha256"] = "0" * 64
            bad = Path(directory) / "wrong-marker.json"
            write_once(bad, seal, root=Path(directory))
            with self.assertRaisesRegex(AnalysisBoundary, "MARKER_SHA"):
                load_frozen(bad)
            link = Path(directory) / "linked.json"
            link.symlink_to(seal_path)
            with self.assertRaisesRegex(ValueError, "SYMLINK"):
                load_frozen(link)


if __name__ == "__main__":
    unittest.main()
