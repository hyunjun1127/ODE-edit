from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
import tempfile
import unittest

from project.run_scripts.ode_edit_method.contracts import (
    Arm,
    EventReading,
    MethodContractError,
    StepRecord,
)
from project.run_scripts.ode_edit_method.lock import load_lock
from project.run_scripts.ode_edit_method.oracle_absolute_lock import (
    load_oracle_absolute_lock,
)
from project.run_scripts.ode_edit_method.oracle_absolute_event import (
    OracleAbsoluteMeanMarginCalibration,
    OracleAbsoluteMeanMarginTarget,
)
from project.run_scripts.ode_edit_method.v4_macrostep_retry import (
    V4_MACROSTEP_LOCK_PATH,
    load_v4_macrostep_lock,
    v4_controller_config,
)
from project.run_scripts.session02_v4_macrostep_retry_p1 import (
    _calibration_accounting,
    _load_v3_native_baseline,
    _retry_diagnostics,
    dry_plan,
    run,
)


def _step(
    *,
    coefficients: tuple[float, ...],
    requested: float,
    retry_index: int,
    retry_scale: float,
    radius: float,
    accepted: bool,
) -> StepRecord:
    norm = sum(value * value for value in coefficients) ** 0.5
    return StepRecord(
        arm=Arm.FULL_ODE_EDIT,
        position=0,
        layer=None,
        snapshot_id="same-state",
        direction_ids=("d0", "d1"),
        solver_coefficients=coefficients,
        applied_coefficients=coefficients,
        accepted=accepted,
        reason="accepted" if accepted else "nonpositive-actual-progress",
        hard_phi_before=1.0,
        hard_phi_after=0.5,
        smooth_phi_before=1.0,
        smooth_phi_after=0.5,
        trust_ratio=1.0 if accepted else -1.0,
        radius=radius,
        radius_cap=2.0,
        requested_progress=requested,
        predicted_progress=requested,
        coefficient_l2=norm,
        retry_index=retry_index,
        retry_scale=retry_scale,
    )


class V4MacrostepRetryTests(unittest.TestCase):
    def test_real_calibration_shape_uses_serialized_backward_accounting(self) -> None:
        target = OracleAbsoluteMeanMarginTarget.calibrate(
            entry_new=(-2.0,),
            entry_old=(-1.0,),
            oracle_new=(-0.5,),
            oracle_old=(-1.5,),
            epsilon=1e-6,
        )
        entry = EventReading(1.0, 1.0, (-1.0,), nfe=2)
        oracle = EventReading(-1.0, -1.0, (1.0,), nfe=2)
        calibration = OracleAbsoluteMeanMarginCalibration(
            target=target,
            entry_reading=entry,
            oracle_shadow_reading=oracle,
            entry_forward_count=2,
            oracle_forward_count=2,
            hook_call_count=2,
            layer_name="layer",
            lookup_positions=(0,),
            h0_dtype="torch.float32",
            delta_dtype="torch.float32",
            delta_l2=1.0,
            parameter_guard_exact=True,
            rng_guard_exact=True,
        )
        self.assertFalse(hasattr(calibration, "extra_backward"))
        metadata = calibration.to_dict()
        self.assertEqual(
            _calibration_accounting(metadata),
            {
                "calibration_total_model_forwards": 4,
                "oracle_extra_model_forwards": 2,
                "oracle_backward_count": 0,
            },
        )

        missing = dict(metadata)
        missing.pop("extra_backward")
        invalid_rows = (missing,) + tuple(
            {**metadata, "extra_backward": value}
            for value in (1, "0", True, float("nan"))
        )
        for row in invalid_rows:
            with self.subTest(extra_backward=row.get("extra_backward", "missing")):
                with self.assertRaises(RuntimeError):
                    _calibration_accounting(row)

    def test_lock_changes_only_three_controller_constants(self) -> None:
        base = load_lock()
        base.pop("proposal_id")
        v4 = load_v4_macrostep_lock()
        config = v4_controller_config(base, v4)
        self.assertEqual(config.h0_fraction, 0.5)
        self.assertEqual(config.h_max_fraction, 1.0)
        self.assertEqual(config.kappa, 1.25)
        self.assertEqual(config.beta, 0.8)
        self.assertEqual(config.gamma_down, 0.5)
        self.assertEqual(config.gamma_up, 1.5)
        self.assertEqual(config.eta_reject, 0.1)
        self.assertEqual(config.eta_expand, 0.75)
        self.assertEqual(config.s_max, 6)
        self.assertEqual(config.max_rejections_per_state, 4)

    def test_v3_event_and_common_dry_plan_are_frozen(self) -> None:
        v3 = load_oracle_absolute_lock()
        v4 = load_v4_macrostep_lock()
        self.assertEqual(v4["v3_frozen"]["proposal_id"], v3["proposal_id"])
        self.assertEqual(v4["v3_frozen"]["lock_sha256"], v3["lock_sha256"])
        self.assertEqual(v3["event"]["rho"], 0.5)
        self.assertEqual(v3["event"]["required_mean_margin"], 0.0)
        self.assertFalse(v3["event"]["q_margin_decision"])
        self.assertFalse(v3["event"]["per_context_decision"])
        plan = dry_plan(v4)
        self.assertFalse(plan["submission_authorized"])
        self.assertEqual(plan["pair_gpu"], 2)
        self.assertEqual(plan["server1_project_gpu_cap"], 3)
        self.assertEqual(plan["arms"], ["full-ode-edit"])
        self.assertEqual(
            [job["model_alias"] for job in plan["jobs"]],
            ["llama3-8b-inst", "qwen2.5-7b-inst"],
        )
        for job in plan["jobs"]:
            self.assertIn(v4["proposal_id"][:8], job["output_root"])
            root = Path(job["output_root"])
            if root.exists():
                terminal = json.loads(
                    (root / "terminal_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(terminal["status"], "COMPLETE")

    def test_native_baselines_hash_validate_without_runtime_branch(self) -> None:
        lock = load_v4_macrostep_lock()
        missing = [
            Path.cwd() / str(spec["root"])
            for spec in lock["v3_native_baselines"].values()
            if not (Path.cwd() / str(spec["root"])).is_dir()
        ]
        if missing:
            self.skipTest(
                "validated local V3 Native artifacts are not present in this worktree"
            )
        for alias in ("llama3-8b-inst", "qwen2.5-7b-inst"):
            rows = _load_v3_native_baseline(Path.cwd(), alias, lock)
            self.assertEqual(tuple(rows), ("2022", "12498", "20964", "768"))
            self.assertTrue(all(row["status"] == "event_hit" for row in rows.values()))
        source = inspect.getsource(run)
        self.assertNotIn("if args.model_alias", source)
        self.assertNotIn("elif args.model_alias", source)
        self.assertNotIn("legacy_replay", source)

    def test_lock_rejects_event_or_model_specific_mutation(self) -> None:
        payload = json.loads(V4_MACROSTEP_LOCK_PATH.read_text(encoding="utf-8"))
        for mutate in (
            lambda row: row["v3_frozen"].__setitem__("rho", 0.75),
            lambda row: row["selection"].__setitem__(
                "model_specific_policy", True
            ),
            lambda row: row["controller_overrides"].__setitem__(
                "h0_fraction", 0.25
            ),
        ):
            changed = json.loads(json.dumps(payload))
            mutate(changed)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "v4.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(MethodContractError):
                    load_v4_macrostep_lock(path)

    def test_retry_diagnostics_require_unique_scaled_candidate(self) -> None:
        base = load_lock()
        base.pop("proposal_id")
        config = v4_controller_config(base, load_v4_macrostep_lock())
        first = _step(
            coefficients=(0.4, 0.4),
            requested=0.8,
            retry_index=0,
            retry_scale=1.0,
            radius=1.0,
            accepted=False,
        )
        retry = _step(
            coefficients=(0.1, 0.1),
            requested=0.2,
            retry_index=1,
            retry_scale=0.5,
            radius=0.5,
            accepted=True,
        )
        result = type("Result", (), {"steps": (first, retry)})()
        diagnostic = _retry_diagnostics(result, config, n_field=1)
        self.assertEqual(diagnostic["retry_pair_count"], 1)
        self.assertTrue(diagnostic["candidate_fingerprints_unique"])

        replay = replace(
            retry,
            solver_coefficients=first.solver_coefficients,
            applied_coefficients=first.applied_coefficients,
            requested_progress=first.requested_progress,
            predicted_progress=first.predicted_progress,
            coefficient_l2=first.coefficient_l2,
        )
        bad = type("Result", (), {"steps": (first, replay)})()
        with self.assertRaisesRegex(RuntimeError, "not unique"):
            _retry_diagnostics(bad, config, n_field=1)


if __name__ == "__main__":
    unittest.main()
