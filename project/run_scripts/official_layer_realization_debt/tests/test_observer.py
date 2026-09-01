from __future__ import annotations

import inspect

import pytest
import torch

from project.run_scripts.ode_bf.p1r52_official_sequential_baselines import run_official_memit_apply
from project.run_scripts.ode_bf.scalable_batched_native import run_official_native_apply
from project.run_scripts.official_layer_realization_debt.contracts import Method, ObservationBoundary
from project.run_scripts.official_layer_realization_debt.metrics import residual_debt_metrics, weight_action_energy
from project.run_scripts.official_layer_realization_debt.observer import OfficialCallAudit, OfficialLayerObserver


class DummyModule:
    def __init__(self, *, alpha: bool = False):
        self.alpha = alpha

    def compute_z(self, *_args, **_kwargs):
        return torch.tensor([1.0, 2.0], dtype=torch.float32)

    def get_module_input_output_at_words(self, *_args, **_kwargs):
        out = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        return (out + 1, out) if self.alpha else out

    def compute_ks(self, value):
        return value + 1


def test_ideal_schedule_recurrence_and_geometry() -> None:
    z = [torch.tensor([1.0, 0.0])]
    pre = [torch.tensor([[value, 0.0]]) for value in (0.0, 0.2, 0.4, 0.6, 0.8)]
    terminal = torch.tensor([[1.0, 0.0]])
    receipt = residual_debt_metrics(
        z_rows=z,
        pre_layer_rows=pre,
        terminal_rows=terminal,
        request_sha256=["a" * 64],
    )
    row = receipt["records"][0]
    assert row["q"] == pytest.approx([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    assert [value["rho"] for value in row["layers"]] == pytest.approx([1.0] * 5)
    assert [value["tau"] for value in row["layers"]] == pytest.approx([0.0] * 5)
    assert row["recurrence_closure_relative_error"] < 1e-12


def test_zero_initial_residual_is_typed_boundary() -> None:
    with pytest.raises(ObservationBoundary, match="ZERO_INITIAL_RESIDUAL"):
        residual_debt_metrics(
            z_rows=[torch.zeros(2)],
            pre_layer_rows=[torch.zeros(1, 2) for _ in range(5)],
            terminal_rows=torch.zeros(1, 2),
            request_sha256=["b" * 64],
        )


@pytest.mark.parametrize("method,alpha", [(Method.MEMIT, False), (Method.ALPHAEDIT, True)])
def test_observer_return_identity_shape_and_restore(method: Method, alpha: bool) -> None:
    module = DummyModule(alpha=alpha)
    original_z = module.compute_z
    original_activation = module.get_module_input_output_at_words
    observer = OfficialLayerObserver(method=method, request_sha256=["c" * 64], capture_layers=True)
    with observer.observe(module):
        z = module.compute_z(None)
        outputs = [module.get_module_input_output_at_words(None, None) for _ in range(5)]
        observer.capture_terminal(None, None)
    assert torch.equal(z, torch.tensor([1.0, 2.0]))
    assert len(outputs) == 5
    assert module.compute_z == original_z
    assert module.get_module_input_output_at_words == original_activation


def test_observer_exception_restores_identity() -> None:
    module = DummyModule()
    original_z = module.compute_z
    observer = OfficialLayerObserver(method=Method.MEMIT, request_sha256=["d" * 64], capture_layers=False)
    with pytest.raises(RuntimeError):
        with observer.observe(module):
            raise RuntimeError("fixture")
    assert module.compute_z == original_z


def test_parity_control_copies_no_layer_or_terminal() -> None:
    module = DummyModule()
    observer = OfficialLayerObserver(method=Method.MEMIT, request_sha256=["e" * 64], capture_layers=False)
    with observer.observe(module):
        module.compute_z(None)
        for _ in range(5):
            module.get_module_input_output_at_words(None, None)
        observer.capture_terminal(None, None)
    payload = observer.payload(module)
    assert payload["layer_loop_pass_through_call_count"] == 5
    assert payload["layer_loop_observation_copy_count"] == 0
    assert payload["terminal_post_L8_forward_count"] == 0


def test_call_audit_counts_and_restores() -> None:
    module = DummyModule()
    original_ks = module.compute_ks
    original_solve = torch.linalg.solve
    audit = OfficialCallAudit()
    with audit.observe(module):
        module.compute_ks(torch.tensor(1))
        torch.linalg.solve(torch.eye(2), torch.ones(2))
    payload = audit.payload(module)
    assert payload["compute_ks_call_count"] == 1
    assert payload["torch_linalg_solve_call_count"] == 1
    assert module.compute_ks == original_ks
    assert torch.linalg.solve is original_solve


def test_weight_energy_fp64_and_share() -> None:
    touched = {f"w{layer}": torch.nn.Parameter(torch.ones(2, 2) * layer) for layer in range(5)}
    originals = {name: value.detach().clone() - 1 for name, value in touched.items()}
    receipt = weight_action_energy(touched, originals)
    assert receipt["scalar_reduction_dtype"] == "float64"
    assert receipt["total_frobenius_energy"] == pytest.approx(20.0)
    assert receipt["share_sum"] == pytest.approx(1.0)


def test_wrapper_interfaces_are_optional_and_default_none() -> None:
    for function in (run_official_memit_apply, run_official_native_apply):
        signature = inspect.signature(function)
        assert signature.parameters["observer"].default is None
        assert signature.parameters["call_audit"].default is None
