from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from project.run_scripts.ode_edit_motivation import mv2_refresh_analysis as analysis


def _sha(value: int) -> str:
    return f"{value:064x}"


def _case(
    index: int,
    *,
    direction: float = 0.10,
    coefficient: float = 0.05,
    partial: float = 0.10,
    no_op: float = 0.0,
    sham: float = 0.0,
) -> dict:
    c_budget = 2.0
    c_progress = partial + 0.05
    b_progress = c_progress + coefficient
    a_progress = b_progress + direction
    arms = {
        "no_op_replay": {
            "progress": no_op,
            "c_energy": 0.0,
            "nfe": 1,
            "success": True,
        },
        "h0_sham": {
            "progress": sham,
            "c_energy": 0.0,
            "nfe": 1,
            "success": True,
        },
        "partial_joint": {
            "progress": partial,
            "c_energy": c_budget,
            "nfe": 1,
            "success": True,
        },
        "refreshed_direction_refreshed_coefficient": {
            "progress": a_progress,
            "c_energy": c_budget,
            "nfe": 1,
            "success": True,
        },
        "fixed_direction_refreshed_coefficient": {
            "progress": b_progress,
            "c_energy": c_budget,
            "nfe": 1,
            "success": True,
        },
        "fixed_direction_fixed_coefficient": {
            "progress": c_progress,
            "c_energy": c_budget,
            "nfe": 1,
            "success": True,
        },
    }
    return {
        "model_alias": "llama3-8b-inst",
        "case_id": index,
        "request_id": _sha(index + 501),
        "pass": True,
        "technical": {
            "exact_panel": True,
            "lineage_exact": True,
            "matched_second_c": True,
            "rollback_exact": True,
            "firewall_pass": True,
            "receipt_before_outcome": True,
        },
        "branch_order": list(analysis.ARM_ORDER),
        "arms": arms,
        "budgets": {"second_step_c_energy": c_budget},
        "lineage": {
            "origin_state_id": _sha(index + 201),
            "w1_state_id": _sha(index + 301),
            "lineage_id": _sha(index + 401),
            "target_token_sha256": _sha(index + 1),
            "direct_z_tensor_sha256": _sha(index + 101),
            "context_id": _sha(999),
        },
        "compute": {
            "controlled_nfe": 43,
            "proposal_build_count": 4,
            "probe_panel_count": 3,
            "wall_seconds": 1.25,
        },
    }


def _records(**kwargs) -> list[dict]:
    return [_case(index, **kwargs) for index in range(12)]


class MV2RefreshAnalysisTests(unittest.TestCase):
    def test_known_answer_direction_refresh_is_clear(self) -> None:
        summary = analysis.analyze_records(
            _records(), model_alias="llama3-8b-inst"
        )

        self.assertEqual(summary["scope"], "single_model")
        self.assertIs(summary["technical_validity"]["pass"], True)
        self.assertEqual(summary["replay_sham_envelope"]["value"], 1e-4)
        self.assertEqual(
            summary["fixed_contract"]["practical_effect_floor"], 1e-4
        )
        self.assertAlmostEqual(
            summary["effects"]["direction_refresh"]["mean"], 0.10
        )
        self.assertAlmostEqual(
            summary["effects"]["coefficient_refresh"]["mean"], 0.05
        )
        self.assertAlmostEqual(
            summary["effects"]["total_refresh"]["mean"], 0.15
        )
        self.assertEqual(
            summary["effects"]["direction_refresh"]["positive_count"], 12
        )
        self.assertAlmostEqual(
            summary["refresh_opportunity_oracle"]["mean"], 0.15
        )
        self.assertAlmostEqual(
            summary["generic_continuation_gain"]["mean"], 0.20
        )
        self.assertEqual(
            summary["decision"]["verdict"], "DIRECTION_REFRESH_CLEAR"
        )
        self.assertEqual(summary["compute"]["controlled_nfe_total"], 516)
        self.assertEqual(
            summary["compute"]["arms"][
                "refreshed_direction_refreshed_coefficient"
            ]["nfe_total"],
            12,
        )
        self.assertEqual(
            summary["compute"]["w1_policy_incremental_cost_per_case"][
                "fixed_direction_fixed_coefficient"
            ]["probe_nfe"],
            0,
        )

    def test_failed_case_stays_in_denominator_and_contributes_zero(self) -> None:
        records = _records()
        records[0]["pass"] = False
        records[0]["arms"]["refreshed_direction_refreshed_coefficient"][
            "progress"
        ] = 10_000.0

        summary = analysis.analyze_records(records)

        effect = summary["effects"]["direction_refresh"]
        self.assertEqual(effect["n"], 12)
        self.assertAlmostEqual(effect["mean"], 11 * 0.10 / 12)
        self.assertEqual(
            summary["technical_validity"]["failed_case_count_in_denominator"], 1
        )
        self.assertIs(summary["case_effects"][0]["analysis_success"], False)
        self.assertEqual(
            summary["case_effects"][0]["direction_refresh_effect"], 0.0
        )
        self.assertEqual(
            summary["case_effects"][0]["total_refresh_effect"], 0.0
        )

    def test_successful_case_with_failed_arm_is_technical_block(self) -> None:
        records = _records()
        action_id = "refreshed_direction_refreshed_coefficient"
        records[0]["arms"][action_id]["success"] = False

        summary = analysis.analyze_records(records)

        self.assertEqual(
            summary["decision"]["verdict"],
            "BLOCK_TECHNICAL_INVALID",
        )
        self.assertEqual(
            summary["technical_validity"]["failed_arm_counts"][action_id],
            1,
        )

    def test_successful_zero_second_c_budget_is_technical_block(self) -> None:
        records = _records()
        records[0]["budgets"]["second_step_c_energy"] = 0.0
        for action_id in analysis.ARM_ORDER[2:]:
            records[0]["arms"][action_id]["c_energy"] = 0.0

        summary = analysis.analyze_case_records(records)

        self.assertEqual(
            summary["decision"]["verdict"],
            "BLOCK_TECHNICAL_INVALID",
        )
        self.assertTrue(
            any(
                "positive-C" in failure
                for failure in summary["technical_validity"]["failures"]
            )
        )

    def test_exact_six_arm_panel_and_order_are_required(self) -> None:
        for mutation in ("missing_arm", "wrong_order"):
            with self.subTest(mutation=mutation):
                records = _records()
                arms = records[0]["arms"]
                if mutation == "missing_arm":
                    arms.pop("h0_sham")
                else:
                    records[0]["branch_order"] = list(
                        reversed(analysis.ARM_ORDER)
                    )

                with self.assertRaisesRegex(
                    analysis.MV2AnalysisError, "exact six-arm"
                ):
                    analysis.analyze_records(records)

    def test_lineage_rollback_firewall_receipt_fail_closed(self) -> None:
        cases = (
            ("lineage_exact", "lineage_exact=false"),
            ("rollback_exact", "rollback_exact=false"),
            ("firewall_pass", "firewall_pass=false"),
            ("receipt_before_outcome", "receipt_before_outcome=false"),
        )
        for technical_key, reason_fragment in cases:
            with self.subTest(technical_key=technical_key):
                records = _records()
                records[0]["technical"][technical_key] = False

                summary = analysis.analyze_records(records)

                self.assertIs(summary["technical_validity"]["pass"], False)
                self.assertEqual(
                    summary["decision"]["verdict"], "BLOCK_TECHNICAL_INVALID"
                )
                self.assertTrue(
                    any(
                        reason_fragment in reason
                        for reason in summary["technical_validity"]["failures"]
                    )
                )

    def test_measured_matched_c_mismatch_blocks_even_if_flag_claims_pass(
        self,
    ) -> None:
        records = _records()
        records[0]["arms"]["fixed_direction_fixed_coefficient"][
            "c_energy"
        ] = 2.01

        summary = analysis.analyze_records(records)

        self.assertIs(summary["technical_validity"]["pass"], False)
        self.assertEqual(
            summary["decision"]["verdict"], "BLOCK_TECHNICAL_INVALID"
        )
        self.assertTrue(
            any(
                "fixed_direction_fixed_coefficient.c_energy" in reason
                for reason in summary["technical_validity"]["failures"]
            )
        )

    def test_noncanonical_lineage_hash_is_rejected(self) -> None:
        for field in ("target_token_sha256", "direct_z_tensor_sha256"):
            with self.subTest(field=field):
                records = _records()
                records[0]["lineage"][field] = "not-a-sha256"

                with self.assertRaisesRegex(
                    analysis.MV2AnalysisError, "SHA256"
                ):
                    analysis.analyze_records(records)

    def test_nonfinite_progress_is_rejected_even_on_failed_arm(self) -> None:
        for bad_value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad_value=bad_value):
                records = _records()
                arm = records[0]["arms"]["fixed_direction_fixed_coefficient"]
                arm["success"] = False
                arm["progress"] = bad_value

                with self.assertRaisesRegex(
                    analysis.MV2AnalysisError, "non-finite"
                ):
                    analysis.analyze_records(records)

    def test_bootstrap_is_deterministic_and_fixed(self) -> None:
        records = _records()
        for index, record in enumerate(records):
            record["arms"]["refreshed_direction_refreshed_coefficient"][
                "progress"
            ] += (index - 5) * 0.01

        first = analysis.analyze_records(copy.deepcopy(records))
        second = analysis.analyze_records(copy.deepcopy(records))

        first_bootstrap = first["effects"]["direction_refresh"]["bootstrap"]
        second_bootstrap = second["effects"]["direction_refresh"]["bootstrap"]
        self.assertEqual(first_bootstrap, second_bootstrap)
        self.assertEqual(first_bootstrap["seed"], 20260801)
        self.assertEqual(first_bootstrap["samples"], 4000)

    def test_coefficient_only_positive_selects_fixed_direction_pivot(self) -> None:
        summary = analysis.analyze_records(
            _records(direction=0.0, coefficient=0.10)
        )

        self.assertIs(summary["decision"]["direction_clear"], False)
        self.assertIs(summary["decision"]["coefficient_clear"], True)
        self.assertEqual(
            summary["decision"]["verdict"],
            "PIVOT_FIXED_DIRECTION_DYNAMIC_COEFFICIENT",
        )

    def test_immaterial_positive_signs_do_not_clear_practical_floor(self) -> None:
        summary = analysis.analyze_records(
            _records(direction=5e-5, coefficient=0.0)
        )

        self.assertEqual(summary["replay_sham_envelope"]["value"], 1e-4)
        self.assertEqual(
            summary["effects"]["direction_refresh"]["positive_count"], 12
        )
        self.assertIs(summary["decision"]["direction_clear"], False)

    def test_both_null_and_no_refresh_oracle_select_static_routing(
        self,
    ) -> None:
        records = _records(direction=0.0, coefficient=0.0, partial=0.15)
        for record in records:
            for arm_id in (
                "refreshed_direction_refreshed_coefficient",
                "fixed_direction_refreshed_coefficient",
                "fixed_direction_fixed_coefficient",
            ):
                record["arms"][arm_id]["progress"] = 0.15
        summary = analysis.analyze_records(records)

        self.assertAlmostEqual(
            summary["refresh_opportunity_oracle"]["mean"], 0.0
        )
        self.assertIs(summary["decision"]["direction_kill"], True)
        self.assertIs(summary["decision"]["coefficient_null"], True)
        self.assertEqual(
            summary["decision"]["verdict"], "PIVOT_STATIC_ROUTING"
        )

    def test_generic_continuation_gain_does_not_rescue_refresh_kill(self) -> None:
        records = _records(direction=0.0, coefficient=0.0, partial=0.10)
        for record in records:
            for arm_id in (
                "refreshed_direction_refreshed_coefficient",
                "fixed_direction_refreshed_coefficient",
                "fixed_direction_fixed_coefficient",
            ):
                record["arms"][arm_id]["progress"] = 0.20

        summary = analysis.analyze_records(records)

        self.assertAlmostEqual(
            summary["generic_continuation_gain"]["mean"], 0.10
        )
        self.assertAlmostEqual(
            summary["refresh_opportunity_oracle"]["mean"], 0.0
        )
        self.assertIs(summary["decision"]["direction_kill"], True)
        self.assertEqual(
            summary["decision"]["verdict"], "PIVOT_STATIC_ROUTING"
        )

    def test_direction_mechanism_does_not_advance_when_total_is_negative(
        self,
    ) -> None:
        summary = analysis.analyze_records(
            _records(direction=0.10, coefficient=-0.20)
        )

        self.assertIs(summary["decision"]["direction_clear"], True)
        self.assertIs(summary["decision"]["total_clear"], False)
        self.assertIs(summary["decision"]["total_nonnegative"], False)
        self.assertAlmostEqual(
            summary["effects"]["total_refresh"]["mean"], -0.10
        )
        self.assertEqual(
            summary["decision"]["verdict"],
            "DIRECTION_MECHANISM_ONLY_REDESIGN_COEFFICIENT",
        )

    def test_cli_has_no_retuning_knobs(self) -> None:
        for forbidden_option in (
            "--bootstrap-seed",
            "--bootstrap-samples",
            "--cases",
        ):
            with self.subTest(forbidden_option=forbidden_option):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    temporary_path = Path(temporary_directory)
                    input_path = temporary_path / "input.jsonl"
                    input_path.write_text(
                        "\n".join(json.dumps(record) for record in _records())
                        + "\n",
                        encoding="utf-8",
                    )
                    argv = [
                        "--input-jsonl",
                        str(input_path),
                        "--model-alias",
                        "llama3-8b-inst",
                        "--output-json",
                        str(temporary_path / "summary.json"),
                        "--output-report",
                        str(temporary_path / "report.md"),
                        forbidden_option,
                        "1",
                    ]
                    stderr = io.StringIO()
                    with mock.patch("sys.stderr", stderr):
                        with self.assertRaises(SystemExit):
                            analysis._build_parser().parse_args(argv)
                    self.assertIn("unrecognized arguments", stderr.getvalue())

    def test_runner_sanitized_stream_is_unwrapped_with_exact_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "analysis_cases.jsonl"
            rows = [
                {
                    "schema_version": analysis.RUN_STREAM_SCHEMA,
                    "run_id": "mv2-test",
                    "sequence": index,
                    "recorded_at": "2026-08-01T00:00:00+00:00",
                    "event": analysis.RUN_STREAM_EVENT,
                    "payload": record,
                }
                for index, record in enumerate(_records())
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            loaded = analysis.load_jsonl(path)
            self.assertEqual(loaded, _records())

            rows[3]["sequence"] = 9
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(analysis.MV2AnalysisError):
                analysis.load_jsonl(path)

    def test_cli_writes_korean_report_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input.jsonl"
            output_json = temporary_path / "summary.json"
            output_report = temporary_path / "report.md"
            input_path.write_text(
                "\n".join(json.dumps(record) for record in _records()) + "\n",
                encoding="utf-8",
            )

            exit_code = analysis.main(
                [
                    "--input-jsonl",
                    str(input_path),
                    "--model-alias",
                    "llama3-8b-inst",
                    "--output-json",
                    str(output_json),
                    "--output-report",
                    str(output_report),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                json.loads(output_json.read_text(encoding="utf-8"))[
                    "model_alias"
                ],
                "llama3-8b-inst",
            )
            report = output_report.read_text(encoding="utf-8")
            self.assertIn("단일 모델", report)
            self.assertIn("기술 gate", report)
            self.assertIn("method 기대효과 proxy", report)


if __name__ == "__main__":
    unittest.main()
