from __future__ import annotations

import ast
import inspect
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from project.run_scripts import (
    session05_ode_bf_canonical_alpha_posfield_dry_plan as dry,
    session05_ode_bf_submit_canonical_alpha_posfield as submit,
)
from project.run_scripts.ode_bf import (
    canonical_alpha_posfield as alpha,
    common_coldcoord_fixed_e8_runtime as runtime,
    p1_adaptive_runtime,
    p1_backend,
    p1_runtime,
)
from project.run_scripts.ode_bf.contracts import (
    ODEBFContractError,
    ODEBFStateError,
    canonical_hash,
)
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.functional import WaypointFactor
from project.run_scripts.ode_bf.p1_canonical_alpha_posfield_panel import (
    CANONICAL_ALPHA_CASE_ROOT,
    CANONICAL_ALPHA_NUMERICAL_LOCK_FILE,
    expected_canonical_alpha_result_name,
    load_and_validate_canonical_alpha_lock,
)
from project.run_scripts.ode_bf.p1_common_coldcoord_fixed_e8_panel import (
    COMMON_COLD_CASE_SEAL_FILE,
    common_cold_schedule,
    verify_common_cold_case_seal,
)
from project.run_scripts.ode_bf.sampling import load_p1_sampling_seal
from project.run_scripts.ode_bf.tests.test_cold_start_target import (
    _ColdLayerOverlayModel,
    _cold_semantic_field,
)


ROOT = Path(__file__).resolve().parents[4]
LOCKS = ROOT / "project/run_scripts/ode_bf/locks"


def _canonical_field():
    field, solve, risk = _cold_semantic_field(wall_seconds=0.25)
    layers = []
    for ordinal, layer in enumerate(field.layers):
        divisor = len(field.layers) - ordinal
        factor = replace(layer.factor, left=layer.residual.clone())
        layers.append(
            replace(
                layer,
                residual_definition=(
                    p1_backend.CANONICAL_ORDERED_ALPHA_REMAINING_RESIDUAL_V1
                ),
                residual_divisor=divisor,
                factor=factor,
            )
        )
    construction = {
        "policy_identity": (
            p1_backend.CANONICAL_ORDERED_ALPHA_REMAINING_RESIDUAL_V1
        ),
        "persistent_state_exact_restoration": True,
        "e8_transition_count": 0,
        "tau_advance_count": 0,
        "endpoint_capacity_influence_count": 0,
        "scientific_commit_count": 0,
    }
    construction["identity_sha256"] = canonical_hash(construction)
    return replace(
        field,
        layers=tuple(layers),
        construction_receipt=construction,
        identity_sha256=canonical_hash(
            {
                "layers": [item.raw_free_payload() for item in layers],
                "construction": construction["identity_sha256"],
            }
        ),
    ), solve, risk


def _overlay_outputs(
    field,
    target: torch.Tensor,
    coefficients: torch.Tensor,
    hidden: dict[int, torch.Tensor],
):
    model = _ColdLayerOverlayModel()
    with alpha.CanonicalOrderedTargetFieldOverlay(
        model, field, coefficients, target
    ):
        return {
            layer.layer: model.layers[str(layer.layer)](hidden[layer.layer])
            for layer in field.layers
        }


class CanonicalAlphaPosfieldTests(unittest.TestCase):
    def test_layer_local_anchor_divisor_gradient_and_isolation(self) -> None:
        field, _, _ = _canonical_field()
        generator = torch.Generator().manual_seed(8111)
        hidden = {
            layer.layer: torch.randn((3, 11), generator=generator)
            for layer in field.layers
        }
        coefficients = torch.tensor((0.125, 0.10, 0.075, 0.05, 0.025))
        target = field.target_state.clone().requires_grad_(True)
        outputs = _overlay_outputs(field, target, coefficients, hidden)
        for ordinal, layer in enumerate(field.layers):
            expected = (
                (hidden[layer.layer] @ layer.q) @ layer.residual.T
            ) * coefficients[ordinal]
            self.assertTrue(torch.allclose(outputs[layer.layer], expected))
        scalar = sum(value.square().sum() for value in outputs.values())
        gradient = torch.autograd.grad(scalar, target)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(torch.linalg.vector_norm(gradient)), 0.0)

        changed_layer = 2
        changed_residual = field.layers[changed_layer].residual.clone()
        changed_residual[0, 0] += 1.0
        changed_factor = replace(
            field.layers[changed_layer].factor, left=changed_residual.clone()
        )
        changed_layers = list(field.layers)
        changed_layers[changed_layer] = replace(
            changed_layers[changed_layer],
            residual=changed_residual,
            factor=changed_factor,
        )
        changed = replace(field, layers=tuple(changed_layers))
        changed_outputs = _overlay_outputs(
            changed,
            field.target_state.clone().requires_grad_(True),
            coefficients,
            hidden,
        )
        for ordinal, layer in enumerate(field.layers):
            equal = torch.equal(outputs[layer.layer], changed_outputs[layer.layer])
            self.assertEqual(equal, ordinal != changed_layer)

        malformed = replace(field, layers=tuple(reversed(field.layers)))
        with self.assertRaises(ODEBFContractError):
            with alpha.CanonicalOrderedTargetFieldOverlay(
                _ColdLayerOverlayModel(),
                malformed,
                coefficients,
                field.target_state.clone().requires_grad_(True),
            ):
                pass

    def test_canonical_field_receipt_separates_ordered_residuals(self) -> None:
        field, solve, risk = _canonical_field()
        receipt = runtime._common_field_semantic_receipt(
            field,
            actuator=alpha.AlphaActuator.CANONICAL_ORDERED_AE,
            solve_history=solve,
            risk_history=risk,
        )
        scientific = receipt["scientific_content"]
        self.assertEqual(
            scientific["residual_policy"],
            p1_backend.CANONICAL_ORDERED_ALPHA_REMAINING_RESIDUAL_V1,
        )
        self.assertEqual(
            [item["residual_divisor"] for item in scientific["layers"]],
            [5, 4, 3, 2, 1],
        )
        self.assertEqual(scientific["key_hash_unique_count"], 5)
        self.assertEqual(scientific["q_hash_unique_count"], 5)
        self.assertEqual(scientific["writer_identity_unique_count"], 5)
        changed = replace(
            field,
            layers=(replace(field.layers[0], residual_divisor=4), *field.layers[1:]),
        )
        with self.assertRaises(ODEBFContractError):
            runtime._common_field_semantic_receipt(
                changed,
                actuator=alpha.AlphaActuator.CANONICAL_ORDERED_AE,
                solve_history=solve,
                risk_history=risk,
            )

    def test_shadow_policy_is_explicit_pure_and_not_an_e8_transition(self) -> None:
        signature = inspect.signature(p1_backend.build_p1_canonical_ordered_field)
        self.assertNotIn("shared_terminal_residual", signature.parameters)
        source = inspect.getsource(p1_backend.build_p1_canonical_ordered_field)
        self.assertIn("remaining_layer_count", source)
        self.assertIn("_ordered_shadow_factor_map", source)
        self.assertIn("persistent_state_exact_restoration", source)
        self.assertIn('"e8_transition_count": 0', source)
        self.assertIn('"scientific_commit_count": 0', source)
        self.assertIn("pre_shadow_effective_bf16_state_sha256", source)
        self.assertIn("post_layer_shadow_effective_bf16_state_sha256", source)
        self.assertIn("full_shadow_effective_weight_sha256", source)
        self.assertIn("canonical_terminal_activation_capture_count", source)
        self.assertIn(
            "ORDERED_PROPOSAL_STATE_GEOMETRY_NOT_NATIVE_DIRECT_Z", source
        )
        self.assertNotIn("compute_z(", source)
        self.assertNotIn("compute_z(", source)
        self.assertIn('"native_or_direct_z_access_count": 0', source)
        self.assertNotIn("shared_terminal_residual", source)
        self.assertIs(p1_backend.ODEBFStateError, ODEBFStateError)
        self.assertIn("layers != (4, 5, 6, 7, 8)", source)
        self.assertIn("projector.dtype is not torch.float32", source)
        legacy_source = inspect.getsource(p1_backend.build_p1_dynamic_field)
        self.assertNotIn(
            "CANONICAL_ORDERED_ALPHA_REMAINING_RESIDUAL_V1", legacy_source
        )
        self.assertIsNone(
            inspect.signature(p1_runtime.run_p1)
            .parameters["canonical_alpha_posfield_actuator"]
            .default
        )

    def test_shadow_effective_bf16_hash_is_pure_and_factor_sensitive(self) -> None:
        generator = torch.Generator().manual_seed(319)
        parameter = torch.nn.Parameter(
            torch.randn((12, 13), generator=generator).to(torch.bfloat16),
            requires_grad=False,
        )
        before = tensor_sha256(parameter)
        pointer = parameter.data_ptr()
        version = parameter._version
        factor = WaypointFactor(
            "layers.4.weight",
            4,
            0,
            0,
            0,
            1.0,
            torch.randn((12, 10), generator=generator),
            torch.randn((13, 10), generator=generator),
        )
        empty_sha = p1_backend._ordered_effective_weight_sha256(parameter, ())
        factor_sha = p1_backend._ordered_effective_weight_sha256(
            parameter, (factor,)
        )
        self.assertEqual(empty_sha, before)
        self.assertNotEqual(factor_sha, before)
        self.assertEqual(tensor_sha256(parameter), before)
        self.assertEqual(parameter.data_ptr(), pointer)
        self.assertEqual(parameter._version, version)

    def test_shadow_purity_firewall_raises_typed_state_error(self) -> None:
        common = {
            "before_bytes": {"w": "a" * 64},
            "after_bytes": {"w": "a" * 64},
            "before_pointers": {"w": 11},
            "after_pointers": {"w": 11},
            "before_versions": {"w": 3},
            "after_versions": {"w": 3},
            "before_rng": "b" * 64,
            "after_rng": "b" * 64,
            "before_target": "c" * 64,
            "after_target": "c" * 64,
        }
        p1_backend._assert_canonical_ordered_shadow_purity(**common)
        changed = dict(common)
        changed["after_versions"] = {"w": 4}
        with self.assertRaises(ODEBFStateError):
            p1_backend._assert_canonical_ordered_shadow_purity(**changed)

    def test_zero_positive_field_is_persisted_as_arm_local_prefix(self) -> None:
        build_source = inspect.getsource(runtime._build_common_field)
        run_source = inspect.getsource(runtime._run_common_arm)
        self.assertLess(
            build_source.index("recorder.field(failure_payload)"),
            build_source.index("raise ZeroPositiveDirection"),
        )
        self.assertIn("except ZeroPositiveDirection as stop", run_source)
        self.assertIn("trajectory_complete=False", run_source)
        self.assertIn('"valid_prefix_preserved": True', run_source)
        self.assertIn('"termination_label": "ZERO_POSITIVE_DIRECTION"', run_source)
        self.assertNotIn("BOOTSTRAP_READINESS_FAILED: p_max", build_source)
        self.assertIn(
            "routing.mode is FixedE8StepMode.ZERO_WRITE_TARGET_RECOVERY",
            build_source,
        )
        self.assertIn("zero-positive routing mode/p_max differs", build_source)
        signature = inspect.signature(runtime._run_common_arm)
        self.assertIsNone(signature.parameters["result_variant_label"].default)
        self.assertIn(
            "variant_label = arm.value if result_variant_label is None",
            run_source,
        )

    def test_current_control_reproduction_is_exact_and_fail_closed(self) -> None:
        expected = {
            "accepted_snapshot_count": 2,
            "field_semantic_sha256": ["a" * 64, "b" * 64],
            "snapshot_sha256": ["c" * 64, "d" * 64],
            "source_manifest_sha256": "e" * 64,
            "source_terminal_sha256": "f" * 64,
            "terminal_status": "COMMON_COLD_FIXED_E8_TAU_COMPLETE",
        }
        rollout = SimpleNamespace(
            variant_label=alpha.AlphaActuator.CURRENT_SHARED_AE.value,
            status=expected["terminal_status"],
            snapshots=[
                SimpleNamespace(
                    routing_payload={"field_semantic_sha256": field},
                    snapshot_sha256=snapshot,
                )
                for field, snapshot in zip(
                    expected["field_semantic_sha256"],
                    expected["snapshot_sha256"],
                    strict=True,
                )
            ],
        )
        receipt = runtime._r10_current_reproduction_gate(rollout, expected)
        self.assertTrue(receipt["semantic_prefix_exact"])
        rollout.snapshots[1].snapshot_sha256 = "0" * 64
        with self.assertRaises(ODEBFContractError):
            runtime._r10_current_reproduction_gate(rollout, expected)

    def test_postfreeze_extra_baseline_is_opt_in_and_controller_firewalled(self) -> None:
        signature = inspect.signature(p1_adaptive_runtime._postfreeze_stepwise_panel)
        self.assertIsNone(
            signature.parameters["additional_candidate_baselines"].default
        )
        source = inspect.getsource(p1_adaptive_runtime._postfreeze_stepwise_panel)
        self.assertLess(
            source.index("freeze_root_sha256 = write_once"),
            source.index("load_counterfact_cases_after_freeze"),
        )
        self.assertIn("controller_heldout_access_count", source)
        self.assertNotIn("model.generate", source)

    def test_lock_dry_plan_and_four_job_cap(self) -> None:
        seal = verify_common_cold_case_seal(
            json.loads((LOCKS / COMMON_COLD_CASE_SEAL_FILE).read_text(encoding="utf-8"))
        )
        self.assertEqual(seal["root_digest"], CANONICAL_ALPHA_CASE_ROOT)
        base_schedule = load_p1_sampling_seal(
            LOCKS / "p1r2_p_population_seal.json",
            stream_path=LOCKS / "p1r2_seqb10_stream_seal.json",
        )
        schedule = common_cold_schedule(base_schedule)
        population = json.loads(
            (LOCKS / "p1r2_p_population_seal.json").read_text(encoding="utf-8")
        )
        for actuator in alpha.AlphaActuator:
            value, _ = load_and_validate_canonical_alpha_lock(
                LOCKS / CANONICAL_ALPHA_NUMERICAL_LOCK_FILE,
                controller_identity_sha256=dry.numerical_controller_identity(),
                case_root_digest=seal["root_digest"],
                population_root_digest=population["root_digest"],
                schedule=schedule,
                actuator=actuator,
            )
            self.assertEqual(value["joint_grid_count"], 8)
            self.assertEqual(value["joint_h"], 0.125)
            self.assertEqual(
                set(value["current_shared_r10_reproduction_by_alias"]),
                {"llama3-8b-inst", "qwen2.5-7b-inst"},
            )
        plan = dry.build_plan("1" * 40, repository_root=ROOT)
        self.assertEqual(plan["server1_project_gpu_cap"], 4)
        self.assertEqual(plan["new_job_gpu"], 4)
        self.assertEqual(len(plan["jobs"]), 4)
        self.assertEqual({job["server"] for job in plan["jobs"]}, {"server1"})
        self.assertEqual({job["slurm_node"] for job in plan["jobs"]}, {"devbox"})
        self.assertEqual(
            len({job["result_name"] for job in plan["jobs"]}), 4
        )
        self.assertEqual(
            expected_canonical_alpha_result_name(
                "llama3-8b-inst", alpha.AlphaActuator.CURRENT_SHARED_AE
            ),
            "s05-canonical-alpha-posfield-p1r11-llama3-8b-inst-current_shared_ae-v1",
        )
        submit_source = inspect.getsource(submit.main)
        self.assertNotIn("--state-root", submit_source)
        self.assertIn('state_root = REPO_ROOT / "local/odebf/state"', submit_source)
        self.assertEqual(
            submit._local_gpu_cap_gate(),
            {
                "server": "server1",
                "slurm_node": "devbox",
                "project_gpu_cap": 4,
                "canonical_tracked_record_commit": "fd49e3a",
            },
        )
        manifest_gate_source = inspect.getsource(submit._source_manifest_gate)
        self.assertIn('"diff",', manifest_gate_source)
        self.assertIn('"--name-only",', manifest_gate_source)
        self.assertIn("manifest_relative not in changed_paths", manifest_gate_source)
        self.assertIn(
            "observed != [path for path in changed_paths if path != manifest_relative]",
            manifest_gate_source,
        )

    def test_source_ast_has_no_adaptive_or_first_hit_control(self) -> None:
        tree = ast.parse(inspect.getsource(alpha))
        names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        self.assertNotIn("AdaptiveTauClock", names)
        self.assertNotIn("direct_z", names)
        self.assertNotIn("model_generate", names)


if __name__ == "__main__":
    unittest.main()
