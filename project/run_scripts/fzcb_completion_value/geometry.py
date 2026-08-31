"""MEMIT residual writer geometry and full-model matrix-free control map."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch.func import functional_call, jvp, vjp

from project.run_scripts.fixed_z_nonuniqueness.official import official_model_name_binding

from .contracts import NumericalLock, ScientificBoundary
from .hashing import canonical_hash, tensor_sha256


@contextmanager
def _forward_ad_attention(model: torch.nn.Module):
    """Temporarily use public eager attention, restoring the exact backend."""
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "_attn_implementation_internal"):
        yield False
        return
    original = config._attn_implementation
    if original == "eager":
        yield False
        return
    config._attn_implementation_internal = "eager"
    try:
        if config._attn_implementation != "eager":
            raise ScientificBoundary("forward-AD eager attention activation failed")
        yield True
    finally:
        config._attn_implementation_internal = original
        if config._attn_implementation != original:
            raise ScientificBoundary("forward-AD attention backend restore failed")


@dataclass(frozen=True, slots=True)
class GeometryReceipt:
    weight_names: tuple[str, ...]
    right_factor_sha256: tuple[str, ...]
    h_scales: tuple[float, ...]
    metric_epsilons: tuple[float, ...]
    key_sha256: tuple[str, ...]
    writer_factor_solve_count: int
    identity: str


class FullModelControlOperator:
    """A=C H^-1/2 using only weight-factor JVP and activation VJP."""

    def __init__(
        self, model: Any, batch: dict[str, torch.Tensor], z_layer: int,
        weight_names: tuple[str, ...], right_factors: tuple[torch.Tensor, ...],
        h_sqrt_blocks: tuple[float, ...], last_direct_gain: float,
    ) -> None:
        self.model = model
        self.batch = batch
        self.z_layer = int(z_layer)
        self.weight_names = weight_names
        self.right_factors = right_factors
        self.h_sqrt_blocks = h_sqrt_blocks
        self.last_direct_gain = float(last_direct_gain)
        params = dict(model.named_parameters())
        self.primals = tuple(params[name] for name in weight_names)
        self.block_widths = tuple(int(weight.shape[0] if weight.shape[1] == right.numel() else weight.shape[1]) for weight, right in zip(self.primals, right_factors, strict=True))
        self.offsets = [0]
        for width in self.block_widths:
            self.offsets.append(self.offsets[-1] + width)
        self.coefficient_dimension = self.offsets[-1]
        self.output_dimension = int(model.config.hidden_size)
        self.jvp_count = 0
        self.vjp_count = 0
        self.forward_count = 0
        self.gram_floor = (self.last_direct_gain / float(self.h_sqrt_blocks[-1])) ** 2
        if not math.isfinite(self.gram_floor) or self.gram_floor <= 0:
            raise ScientificBoundary("last-layer direct range authority is absent")

    def phi(self) -> torch.Tensor:
        value = self._functional_phi(*self.primals)
        self.forward_count += 1
        return value.detach()

    def split(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if value.dtype != torch.float32 or value.ndim != 1 or value.numel() != self.coefficient_dimension:
            raise ScientificBoundary("coefficient vector shape/dtype mismatch")
        return tuple(value[self.offsets[index] : self.offsets[index + 1]] for index in range(len(self.block_widths)))

    def coefficient_to_tangents(self, coefficient: torch.Tensor) -> tuple[torch.Tensor, ...]:
        blocks = self.split(coefficient)
        result = []
        for block, right, weight in zip(blocks, self.right_factors, self.primals, strict=True):
            update = torch.outer(block, right)
            if update.shape != weight.shape:
                if update.T.shape != weight.shape:
                    raise ScientificBoundary("MEMIT tangent orientation mismatch")
                update = update.T
            result.append(update.to(weight))
        return tuple(result)

    def _clean_batch(self) -> dict[str, torch.Tensor]:
        return {key: value for key, value in self.batch.items() if key != "fzcb_lookup_index"}

    def _functional_phi(self, *weights: torch.Tensor) -> torch.Tensor:
        override = dict(zip(self.weight_names, weights, strict=True))
        output = functional_call(
            self.model, override, (), {
                **self._clean_batch(), "output_hidden_states": True,
                "use_cache": False, "return_dict": True,
            }, strict=False,
        )
        lookup = int(self.batch["fzcb_lookup_index"].item())
        return output.hidden_states[self.z_layer + 1][0, lookup, :]

    def apply(self, whitened: torch.Tensor) -> torch.Tensor:
        blocks = self.split(whitened)
        coefficient = torch.cat([
            block / float(scale) for block, scale in zip(blocks, self.h_sqrt_blocks, strict=True)
        ])
        tangents = self.coefficient_to_tangents(coefficient)
        with _forward_ad_attention(self.model):
            _, tangent = jvp(self._functional_phi, self.primals, tangents)
        self.jvp_count += 1
        return tangent.detach().to(dtype=torch.float32)

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        if value.dtype != torch.float32 or value.shape != (self.output_dimension,):
            raise ScientificBoundary("activation cotangent mismatch")
        with _forward_ad_attention(self.model):
            _, pullback = vjp(self._functional_phi, *self.primals)
            gradients = pullback(value)
        blocks = []
        for gradient, right, weight, scale in zip(gradients, self.right_factors, self.primals, self.h_sqrt_blocks, strict=True):
            if weight.shape[1] == right.numel():
                raw = gradient @ right
            elif weight.shape[0] == right.numel():
                raw = gradient.T @ right
            else:
                raise ScientificBoundary("MEMIT adjoint orientation mismatch")
            blocks.append(raw / float(scale))
        self.vjp_count += 1
        return torch.cat(blocks).detach().to(dtype=torch.float32)

    def h_sqrt_vector(self) -> torch.Tensor:
        device = self.primals[0].device
        return torch.cat([
            torch.full((width,), float(scale), dtype=torch.float32, device=device)
            for width, scale in zip(self.block_widths, self.h_sqrt_blocks, strict=True)
        ])

    def output_preconditioner(self, value: torch.Tensor) -> torch.Tensor:
        return value / float(self.gram_floor)

    def raw_coefficient_tangents(self, coefficient: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.coefficient_to_tangents(coefficient)

    def receipt(self) -> dict[str, Any]:
        return {
            "coefficient_dimension": self.coefficient_dimension,
            "output_dimension": self.output_dimension,
            "jvp_count": self.jvp_count, "vjp_count": self.vjp_count,
            "explicit_kronecker_count": 0, "dense_inverse_count": 0,
        }


def canonical_batch(tokenizer: Any, row: dict[str, Any], hparams: Any, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from easyeditor.models.memit.compute_z import find_fact_lookup_idx

    request = row["requested_rewrite"]
    prompt = request["prompt"].format(request["subject"])
    tokenizer.padding_side = "right"
    batch = tokenizer(prompt, return_tensors="pt", padding=False)
    attention = batch["attention_mask"]
    position = attention.long().cumsum(-1) - 1
    position.masked_fill_(attention == 0, 0)
    lookup = find_fact_lookup_idx(
        request["prompt"], request["subject"], tokenizer, hparams.fact_token, verbose=False
    )
    semantic = int(attention[0].sum().item()) + lookup if lookup < 0 else lookup
    if semantic < 0 or semantic >= int(attention[0].sum().item()):
        raise ScientificBoundary("canonical semantic lookup index invalid")
    result = {key: value.to(device) for key, value in batch.items()}
    result["position_ids"] = position.to(device)
    result["fzcb_lookup_index"] = torch.tensor(semantic, device=device)
    return result, {
        "prompt": prompt, "lookup_index": semantic,
        "input_ids_sha256": tensor_sha256(result["input_ids"]),
        "attention_mask_sha256": tensor_sha256(result["attention_mask"]),
        "position_ids_sha256": tensor_sha256(result["position_ids"]),
    }


class MEMITGeometryFactory:
    """Freeze M0/covariance factorizations; refresh only current MEMIT keys."""

    def __init__(self, model: Any, tokenizer: Any, row: dict[str, Any], hparams: Any, batch: dict[str, torch.Tensor]) -> None:
        from easyeditor.models.memit import memit_main

        self.model, self.tokenizer, self.row, self.hparams, self.batch = model, tokenizer, row, hparams, batch
        self.context_templates = memit_main.get_context_templates(model, tokenizer)
        self.names = tuple(f"{hparams.rewrite_module_tmp.format(layer)}.weight" for layer in hparams.layers)
        covariances = []
        factors = []
        epsilons = []
        with official_model_name_binding(model, str(hparams.model_name)):
            for layer in hparams.layers:
                covariance = memit_main.get_cov(
                    model, tokenizer, hparams.rewrite_module_tmp.format(layer),
                    hparams.mom2_dataset, hparams.mom2_n_samples,
                    hparams.mom2_dtype, hparams=hparams,
                ).float()
                factor, info = torch.linalg.cholesky_ex(covariance)
                if int(info.max().item()) != 0:
                    raise ScientificBoundary("native covariance is not Cholesky-feasible without ridge")
                mean_diag = float(torch.diagonal(covariance).mean().item())
                epsilon = float(256.0 * torch.finfo(torch.float32).eps * mean_diag)
                covariances.append(covariance)
                factors.append(factor)
                epsilons.append(epsilon)
        self.covariances = tuple(covariances)
        self.cholesky_factors = tuple(factors)
        self.epsilons = tuple(epsilons)
        self.factorization_count = len(self.names)

    def build(self) -> tuple[FullModelControlOperator, GeometryReceipt]:
        from easyeditor.models.memit import memit_main

        rights = []
        h_scales = []
        keys_sha = []
        with official_model_name_binding(self.model, str(self.hparams.model_name)):
            for layer, covariance, factor, epsilon in zip(
                self.hparams.layers, self.covariances, self.cholesky_factors, self.epsilons, strict=True
            ):
                keys = memit_main.compute_ks(self.model, self.tokenizer, [{
                    "case_id": str(self.row["case_id"]),
                    "prompt": self.row["requested_rewrite"]["prompt"],
                    "subject": self.row["requested_rewrite"]["subject"],
                    "target_new": self.row["requested_rewrite"]["target_new"]["str"],
                    "target_true": self.row["requested_rewrite"]["target_true"]["str"],
                }], self.hparams, layer, self.context_templates).T.float()
                if keys.shape[1] != 1:
                    raise ScientificBoundary("B1 MEMIT key width differs")
                base_solve = torch.cholesky_solve(keys, factor)
                denominator = float(self.hparams.mom2_update_weight) + torch.dot(keys.reshape(-1), base_solve.reshape(-1))
                right = (base_solve / denominator).reshape(-1)
                metric_action = torch.dot(right, covariance @ right) + epsilon * torch.dot(right, right)
                if not bool(torch.isfinite(metric_action)) or float(metric_action) <= 0:
                    raise ScientificBoundary("MEMIT H block is non-positive")
                rights.append(right.detach())
                h_scales.append(math.sqrt(float(metric_action.item())))
                keys_sha.append(tensor_sha256(keys))
        canonical_input = memit_main.get_module_input_output_at_words(
            self.model, self.tokenizer, int(self.hparams.layers[-1]),
            context_templates=[self.row["requested_rewrite"]["prompt"]],
            words=[self.row["requested_rewrite"]["subject"]],
            module_template=self.hparams.rewrite_module_tmp,
            fact_token_strategy=self.hparams.fact_token, track="in",
        ).reshape(-1).to(rights[-1])
        last_direct_gain = float(torch.dot(rights[-1], canonical_input).item())
        payload = {
            "weight_names": list(self.names),
            "right_factor_sha256": [tensor_sha256(value) for value in rights],
            "h_scales": h_scales, "metric_epsilons": list(self.epsilons), "key_sha256": keys_sha,
            "writer_factor_solve_count": len(self.names), "covariance_factorization_count": self.factorization_count,
            "last_direct_gain": last_direct_gain,
            "dense_inverse_count": 0, "explicit_kronecker_count": 0,
        }
        receipt = GeometryReceipt(
            self.names, tuple(payload["right_factor_sha256"]), tuple(h_scales),
            self.epsilons, tuple(keys_sha), len(self.names), canonical_hash(payload),
        )
        return FullModelControlOperator(
            self.model, self.batch, int(self.hparams.layers[-1]), self.names,
            tuple(rights), tuple(h_scales), last_direct_gain
        ), receipt
