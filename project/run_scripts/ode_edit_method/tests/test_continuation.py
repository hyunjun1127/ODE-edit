from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from project.run_scripts.ode_edit_motivation.contracts import ExpectedFileIdentity
from project.run_scripts.ode_edit_motivation.hooks import tensor_sha256
from project.run_scripts.ode_edit_method.continuation import (
    BASE_S_MAX,
    CANONICAL_CASE_ORDER,
    CONTINUATION_S_MAX,
    EXPECTED_AFFECTED_CASES,
    PinnedDirectZEasyEditBackend,
    R2_EXECUTION_HEAD,
    R2_PROPOSAL_ID,
    R2SourceSpec,
    assert_prefix_match,
    canonical_six_round_prefix,
    continuation_config,
    continuation_hit_round,
    dispatch_eligibility,
    file_sha256,
    promotion_decision,
    validate_r2_sources,
)
from project.run_scripts.ode_edit_method.contracts import MethodContractError
from project.run_scripts.ode_edit_method.lock import controller_config, load_lock


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _prefix_steps(case_id: str) -> list[dict[str, object]]:
    return [
        {
            "arm": "full-ode-edit",
            "position": position,
            "layer": None,
            "snapshot_id": f"snapshot-{case_id}-{position}",
            "direction_ids": [
                f"direction-{case_id}-{position}-{layer}" for layer in range(2)
            ],
            "solver_coefficients": [0.01 + position, 0.02 + position],
            "applied_coefficients": [0.01 + position, 0.02 + position],
            "hard_phi_before": float(10 - position),
            "hard_phi_after": float(9 - position),
            "smooth_phi_before": float(9.5 - position),
            "smooth_phi_after": float(8.5 - position),
            "accepted": True,
            "reason": "accepted",
            "trust_ratio": 0.5,
        }
        for position in range(BASE_S_MAX)
    ]


def _make_source(
    parent: Path,
    model_alias: str,
    *,
    nonzero_omega: bool = False,
    different_w0: bool = False,
    raw_prompt: bool = False,
) -> R2SourceSpec:
    root = parent / model_alias
    (root / "direct_z").mkdir(parents=True)
    affected = set(EXPECTED_AFFECTED_CASES[model_alias])
    controller: list[dict[str, object]] = []
    compute: list[dict[str, object]] = []
    mechanism: list[dict[str, object]] = []
    files: dict[str, dict[str, object]] = {}

    for arm in (
        "native-memit",
        "static-synchronous",
        "one-refresh",
        "full-ode-edit",
    ):
        for order_position, case_id in enumerate(CANONICAL_CASE_ORDER):
            is_affected = arm == "full-ode-edit" and case_id in affected
            status = "resolution_cap_unresolved" if is_affected else "event_hit"
            if arm == "full-ode-edit" and case_id == "12498":
                status = "trust_rejection_limit"
            request_state = f"request-state-{model_alias}-{case_id}"
            target_state = f"w0-{case_id}" if different_w0 else "w0"
            steps = _prefix_steps(case_id) if is_affected else []
            hashes: dict[str, object] = {"proposal_id": R2_PROPOSAL_ID}
            if is_affected:
                relative = (
                    f"direct_z/{model_alias}-full-ode-edit-position-{order_position}-"
                    f"case-{case_id}.pt"
                )
                direct_z_path = root / relative
                direct_z_path.write_bytes(f"direct-z-{model_alias}-{case_id}".encode())
                direct_identity = {
                    "sha256": file_sha256(direct_z_path),
                    "size": direct_z_path.stat().st_size,
                }
                files[relative] = direct_identity
                hashes["direct_z"] = {
                    "artifact_sha256": direct_identity["sha256"],
                    "artifact_size": direct_identity["size"],
                    "tensor_sha256": hashlib.sha256(
                        f"tensor-{model_alias}-{case_id}".encode()
                    ).hexdigest(),
                    "source_state_id": request_state,
                }
            omega_value = 1.0 if nonzero_omega and is_affected else 0.0
            row: dict[str, object] = {
                "arm": arm,
                "case_id": case_id,
                "order_position": order_position,
                "status": status,
                "pre_edit_state_id": request_state,
                "post_edit_state_id": request_state,
                "pre_edit_target_weight_state_id": target_state,
                "post_edit_target_weight_state_id": target_state,
                "result": {
                    "status": status,
                    "omega_appended": False if is_affected else True,
                    "direct_z_compute_count": 1,
                    "failure_type": (
                        "resolution_cap_unresolved" if is_affected else None
                    ),
                    "terminal_state_id": request_state,
                    "steps": steps,
                },
                "sequential_state": {
                    "history_length_before": 0,
                    "history_length_after": 0,
                    "accepted_terminal_count": 0,
                    "current_edit_terminal_appended": False,
                    "Omega_before": {"0": omega_value},
                    "Omega_after": {"0": omega_value},
                },
                "Omega": {
                    "before": {"0": omega_value},
                    "after": {"0": omega_value},
                    "receipts_through_current_edit": [],
                },
                "hashes": hashes,
            }
            if raw_prompt and is_affected:
                row["prompt"] = "forbidden raw rewrite prompt"
            controller.append(row)
            compute.append(
                {
                    "arm": arm,
                    "case_id": case_id,
                    "order_position": order_position,
                    "status": status,
                    "K_acc": BASE_S_MAX if is_affected else 0,
                    "N_reject": 0,
                    "N_z": 1,
                    "N_write": BASE_S_MAX if is_affected else 0,
                    "N_trial": BASE_S_MAX if is_affected else 0,
                    "N_field": BASE_S_MAX if is_affected else 0,
                    "N_bw": BASE_S_MAX if is_affected else 0,
                    "N_eval": 0,
                }
            )
            mechanism.append(
                {"arm": arm, "case_id": case_id, "order_position": order_position}
            )

    _jsonl(root / "controller_steps.jsonl", controller)
    _jsonl(root / "compute.jsonl", compute)
    _jsonl(root / "mechanism.jsonl", mechanism)
    (root / "evaluation.jsonl").write_bytes(b"")
    _json(
        root / "manifest.json",
        {
            "status": "COMPLETE_TECHNICAL_MECHANISM_ONLY",
            "git": {"commit": R2_EXECUTION_HEAD, "proposal_id": R2_PROPOSAL_ID},
            "model": {
                "model_alias": model_alias,
                "checkpoint_original_dtype": "torch.bfloat16",
                "observed_parameter_dtype": "torch.bfloat16",
                "config_torch_dtype": "torch.bfloat16",
            },
            "selection": {"case_ids": list(CANONICAL_CASE_ORDER)},
            "expected_run_count": 16,
            "artifact_firewall": {
                "evaluation_executed": False,
                "generation_executed": False,
            },
            "policy": {
                "controller": {"s_max": BASE_S_MAX},
                "event_backend": "two-separate-teacher-forced-forwards",
                "event_nfe": 2,
                "trial_backend": "quantized-full-linear-commit-emulator",
                "model_specific_rescue": False,
            },
        },
    )
    _json(
        root / "summary.json",
        {
            "status": "COMPLETE_TECHNICAL_MECHANISM_ONLY_NO_EVALUATION",
            "record_count": 16,
            "evaluation_count": 0,
            "scientific_outcome_count": 0,
        },
    )
    for relative in (
        "manifest.json",
        "controller_steps.jsonl",
        "compute.jsonl",
        "mechanism.jsonl",
        "evaluation.jsonl",
        "summary.json",
    ):
        path = root / relative
        files[relative] = {"sha256": file_sha256(path), "size": path.stat().st_size}
    _json(
        root / "terminal_manifest.json",
        {
            "schema_version": "ode-edit-session02-terminal-manifest/v1",
            "status": "COMPLETE",
            "files": files,
        },
    )
    return R2SourceSpec(model_alias, root, file_sha256(root / "terminal_manifest.json"))


class ContinuationContractTests(unittest.TestCase):
    def _pair(
        self, root: Path, **llama_kwargs: object
    ) -> tuple[R2SourceSpec, R2SourceSpec]:
        return (
            _make_source(root, "llama3-8b-inst", **llama_kwargs),
            _make_source(root, "qwen2.5-7b-inst"),
        )

    def test_source_eligibility_is_exact_and_qwen_is_mutation_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = validate_r2_sources(root, self._pair(root))
            self.assertEqual(selection.pooled_affected_count, 3)
            self.assertEqual(
                tuple(item.case_id for item in selection.for_alias("llama3-8b-inst")),
                ("2022", "20964", "768"),
            )
            self.assertEqual(selection.for_alias("qwen2.5-7b-inst"), ())
            output = root / "must-not-exist"
            self.assertEqual(
                dispatch_eligibility(selection, "qwen2.5-7b-inst", output_root=output),
                (),
            )
            self.assertFalse(output.exists())

    def test_source_guards_zero_omega_same_w0_hash_and_raw_firewall(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(MethodContractError, "zero"):
                validate_r2_sources(root, self._pair(root, nonzero_omega=True))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(MethodContractError, "same W0"):
                validate_r2_sources(root, self._pair(root, different_w0=True))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(MethodContractError, "forbidden raw"):
                validate_r2_sources(root, self._pair(root, raw_prompt=True))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = self._pair(root)
            target = next((specs[0].root / "direct_z").glob("*.pt"))
            target.write_bytes(target.read_bytes() + b"tamper")
            with self.assertRaisesRegex(MethodContractError, "identity differs"):
                validate_r2_sources(root, specs)

    def test_prefix_fingerprint_is_raw_free_timer_free_and_fail_closed(self) -> None:
        source_steps = _prefix_steps("2022")
        expected = canonical_six_round_prefix(source_steps)
        observed = [dict(step, timer_seconds=999.0) for step in source_steps]
        self.assertEqual(assert_prefix_match(expected, observed), hashlib.sha256(
            json.dumps(
                expected,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest())
        self.assertNotIn("timer_seconds", json.dumps(expected))
        for field, replacement in (
            ("direction_ids", ["wrong-a", "wrong-b"]),
            ("solver_coefficients", [8.0, 9.0]),
            ("hard_phi_after", -1.0),
            ("accepted", False),
            ("reason", "trust-ratio-reject"),
        ):
            changed = [dict(step) for step in source_steps]
            changed[0][field] = replacement
            with self.subTest(field=field), self.assertRaises(MethodContractError):
                assert_prefix_match(expected, changed)

    def test_only_smax_changes_and_promotion_rule_is_prelocked(self) -> None:
        raw = load_lock()
        raw.pop("proposal_id")
        base = controller_config(raw)
        extended = continuation_config(base)
        self.assertEqual(base.s_max, 6)
        self.assertEqual(extended.s_max, 8)
        changed = {
            key
            for key in base.to_dict()
            if base.to_dict()[key] != extended.to_dict()[key]
        }
        self.assertEqual(changed, {"s_max"})
        self.assertEqual(
            promotion_decision((7, 8, None))["common_future_s_max_candidate"],
            8,
        )
        self.assertEqual(
            promotion_decision((7, None, None))["common_future_s_max_candidate"],
            6,
        )

    def test_only_round_seven_or_eight_hit_is_promotion_evidence(self) -> None:
        seven = _prefix_steps("2022") + [
            {**_prefix_steps("2022")[0], "position": 6, "accepted": True}
        ]
        self.assertEqual(
            continuation_hit_round({"status": "event_hit", "steps": seven}),
            7,
        )
        unresolved = {"status": "resolution_cap_unresolved", "steps": seven}
        self.assertIsNone(continuation_hit_round(unresolved))
        with self.assertRaisesRegex(MethodContractError, "round 7 or 8"):
            continuation_hit_round(
                {"status": "event_hit", "steps": _prefix_steps("2022")}
            )

    def test_pinned_backend_passes_identity_and_never_requests_recompute(self) -> None:
        values = torch.tensor([[1.25], [-0.75]], dtype=torch.float32)
        value_sha = tensor_sha256(values)
        identity = ExpectedFileIdentity(sha256="a" * 64, size=123)
        request = object()
        result = SimpleNamespace(
            values=values,
            tensor_sha256=value_sha,
            source_state_id="entry-state",
        )

        class Bridge:
            def __init__(self) -> None:
                self.kwargs: dict[str, object] | None = None

            def load_or_compute_direct_z(
                self, *args: object, **kwargs: object
            ) -> object:
                self.kwargs = kwargs
                return result

        bridge = Bridge()
        backend = object.__new__(PinnedDirectZEasyEditBackend)
        backend.request = request
        backend._direct_z = None
        backend.bridge = bridge
        backend.model = object()
        backend.tokenizer = object()
        backend.motivation_request = object()
        backend.hparams = object()
        backend.contexts = object()
        backend.runtime = SimpleNamespace(spec=SimpleNamespace(snapshot_name="model"))
        backend.direct_z_cache_root = Path("/read-only/source")
        backend.direct_z_cache_path = Path("/read-only/source/direct-z.pt")
        backend._expected_direct_z_identity = identity
        backend._expected_direct_z_tensor_sha256 = value_sha
        backend.current_state_id = lambda: "entry-state"
        self.assertIs(backend.compute_direct_z(request), result)
        self.assertIs(bridge.kwargs["expected_identity"], identity)
        self.assertNotIn("compute", bridge.kwargs)
        self.assertEqual(backend._direct_z_tensor_sha256, value_sha)

    def test_runner_and_template_have_no_submit_or_qwen_gpu_path(self) -> None:
        from project.run_scripts import session02_p1_smax8_continuation as runner

        source = inspect.getsource(runner)
        self.assertNotIn("sbatch(", source)
        self.assertIn("zero_eligibility_no_output_mutation", source)
        loop = source[source.index("for request, trajectory") :]
        restore = loop.index("baseline.restore(runtime.model)")
        new_ledger = loop.index("ledger = OmegaLedger(denominators)")
        execute = loop.index("FiveArmRunner(extended_config, ledger).run")
        terminal_restore = loop.index("baseline.restore(runtime.model)", execute)
        self.assertLess(restore, new_ledger)
        self.assertLess(new_ledger, execute)
        self.assertLess(execute, terminal_restore)
        template = Path(
            "project/run_scripts/session02_p1_smax8_continuation.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:1", template)
        self.assertIn("#SBATCH --time=04:00:00", template)
        self.assertIn('test "${MODEL_ALIAS}" = "llama3-8b-inst"', template)
        self.assertNotIn("qwen2.5-7b-inst", template)


if __name__ == "__main__":
    unittest.main()
