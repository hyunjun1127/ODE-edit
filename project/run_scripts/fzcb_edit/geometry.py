"""Gauge-free MEMIT action basis and full-model matrix-free control map."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.func import functional_call, jvp, vjp

from project.run_scripts.fixed_z_nonuniqueness.official import official_model_name_binding
from project.run_scripts.fixed_z_nonuniqueness.padding import semantic_position_ids, semantic_token_columns

from .contracts import NumericalLock, ScientificBoundary
from .hashing import canonical_hash, tensor_sha256


@contextmanager
def _forward_ad_attention(model: torch.nn.Module):
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
            raise ScientificBoundary("forward-AD attention restore failed")


@dataclass(frozen=True, slots=True)
class LayerBasisReceipt:
    weight_name: str
    key_sha256: str
    native_right_sha256: str
    whitened_right_sha256: str
    request_width: int
    reduced_rank: int
    removed_gauge_dimension: int
    metric_epsilon: float
    gram_eigen_min: float
    gram_eigen_max: float


def gauge_free_whitened_right(
    native_right: torch.Tensor, covariance: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Remove pure coefficient gauge and whiten the fixed M0 action.

    For ``delta_W = R @ native_right``, the request-side metric Gram is
    ``native_right (C+eps I) native_right.T``.  Its positive eigenspace is the
    quotient by ``ker(T)``; the returned right factor gives identity action in
    reduced coordinates.
    """

    if native_right.ndim != 2 or covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ScientificBoundary("writer/covariance rank mismatch")
    if native_right.shape[1] != covariance.shape[0] or epsilon <= 0:
        raise ScientificBoundary("writer/covariance dimension or precision floor mismatch")
    gram = native_right @ covariance @ native_right.T + epsilon * (native_right @ native_right.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    largest = float(eigenvalues[-1].item())
    threshold = max(gram.shape) * torch.finfo(torch.float32).eps * largest
    keep = eigenvalues > threshold
    rank = int(keep.sum().item())
    if rank <= 0:
        raise ScientificBoundary("MEMIT writer basis has no metric-positive authority")
    values = eigenvalues[keep]
    vectors = eigenvectors[:, keep]
    whitened = (values.rsqrt().unsqueeze(1) * vectors.T) @ native_right
    identity = whitened @ covariance @ whitened.T + epsilon * (whitened @ whitened.T)
    error = float(torch.linalg.matrix_norm(identity - torch.eye(rank, device=identity.device)).item())
    return whitened.detach(), {
        "request_width": int(native_right.shape[0]),
        "reduced_rank": rank,
        "removed_gauge_dimension": int(native_right.shape[0] - rank),
        "gram_eigen_min": float(values.min().item()),
        "gram_eigen_max": float(values.max().item()),
        "whitening_identity_error": error,
    }


def registered_batch(
    tokenizer: Any, rows: list[dict[str, Any]], hparams: Any, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from easyeditor.models.memit.compute_z import find_fact_lookup_idx

    prompts = [row["requested_rewrite"]["prompt"].format(row["requested_rewrite"]["subject"]) for row in rows]
    previous = tokenizer.padding_side
    try:
        tokenizer.padding_side = "right"
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
    finally:
        tokenizer.padding_side = previous
    positions = semantic_position_ids(tokenized["attention_mask"])
    real_columns = semantic_token_columns(tokenized["attention_mask"])
    lookups = []
    for row, columns in zip(rows, real_columns, strict=True):
        request = row["requested_rewrite"]
        semantic = find_fact_lookup_idx(
            request["prompt"], request["subject"], tokenizer, hparams.fact_token, verbose=False
        )
        if semantic < 0:
            semantic += int(columns.numel())
        if semantic < 0 or semantic >= int(columns.numel()):
            raise ScientificBoundary("registered semantic lookup index invalid")
        lookups.append(int(columns[semantic].item()))
    batch = {key: value.to(device) for key, value in tokenized.items()}
    batch["position_ids"] = positions.to(device)
    batch["fzcb_lookup_indices"] = torch.tensor(lookups, device=device, dtype=torch.long)
    receipt = {
        "prompt_count": len(prompts),
        "prompt_sha256": canonical_hash(prompts),
        "lookup_indices": lookups,
        "input_ids_sha256": tensor_sha256(batch["input_ids"]),
        "attention_mask_sha256": tensor_sha256(batch["attention_mask"]),
        "position_ids_sha256": tensor_sha256(batch["position_ids"]),
    }
    return batch, receipt


class FullModelControlOperator:
    """Metric-whitened A=C H^-1/2 without dense Jacobian/Kronecker objects."""

    def __init__(
        self,
        model: Any,
        batch: dict[str, torch.Tensor],
        z_layer: int,
        weight_names: tuple[str, ...],
        whitened_right_factors: tuple[torch.Tensor, ...],
        direct_gram_floor: float,
        *,
        primals_override: tuple[torch.Tensor, ...] | None = None,
        linearization_state: str = "CURRENT",
    ) -> None:
        self.model = model
        self.batch = batch
        self.z_layer = int(z_layer)
        self.weight_names = weight_names
        self.right_factors = whitened_right_factors
        self.direct_gram_floor = float(direct_gram_floor)
        parameters = dict(model.named_parameters())
        self.primals = (
            tuple(parameters[name] for name in weight_names)
            if primals_override is None
            else tuple(value.detach().clone() for value in primals_override)
        )
        self.linearization_state = linearization_state
        self.block_shapes = []
        self.offsets = [0]
        for weight, right in zip(self.primals, self.right_factors, strict=True):
            if weight.shape[1] == right.shape[1]:
                shape = (int(weight.shape[0]), int(right.shape[0]))
            elif weight.shape[0] == right.shape[1]:
                shape = (int(weight.shape[1]), int(right.shape[0]))
            else:
                raise ScientificBoundary("MEMIT factor/weight orientation mismatch")
            self.block_shapes.append(shape)
            self.offsets.append(self.offsets[-1] + shape[0] * shape[1])
        self.coefficient_dimension = self.offsets[-1]
        self.output_dimension = int(batch["input_ids"].shape[0] * model.config.hidden_size)
        self.jvp_count = 0
        self.vjp_count = 0
        self.forward_count = 0
        if not math.isfinite(self.direct_gram_floor) or self.direct_gram_floor <= 0:
            raise ScientificBoundary("last-layer direct range authority is absent")

    def _clean_batch(self) -> dict[str, torch.Tensor]:
        return {key: value for key, value in self.batch.items() if key != "fzcb_lookup_indices"}

    def _functional_phi(self, *weights: torch.Tensor) -> torch.Tensor:
        override = dict(zip(self.weight_names, weights, strict=True))
        output = functional_call(
            self.model, override, (), {
                **self._clean_batch(), "output_hidden_states": True,
                "use_cache": False, "return_dict": True,
            }, strict=False,
        )
        hidden = output.hidden_states[self.z_layer + 1]
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        values = hidden[rows, self.batch["fzcb_lookup_indices"], :]
        return values.reshape(-1)

    def phi(self) -> torch.Tensor:
        value = self._functional_phi(*self.primals)
        self.forward_count += 1
        return value.detach().float()

    def split(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if value.dtype != torch.float32 or value.ndim != 1 or value.numel() != self.coefficient_dimension:
            raise ScientificBoundary("whitened coefficient shape/dtype mismatch")
        return tuple(
            value[self.offsets[index] : self.offsets[index + 1]].reshape(self.block_shapes[index])
            for index in range(len(self.block_shapes))
        )

    def coefficient_to_tangents(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        updates = []
        for block, right, weight in zip(self.split(value), self.right_factors, self.primals, strict=True):
            update = block @ right
            if update.shape != weight.shape:
                if update.T.shape != weight.shape:
                    raise ScientificBoundary("MEMIT tangent orientation mismatch")
                update = update.T
            updates.append(update.to(weight))
        return tuple(updates)

    def layer_action(self, value: torch.Tensor, delta_s: float) -> tuple[float, ...]:
        if delta_s <= 0.0:
            raise ScientificBoundary("layer action requires positive progress")
        return tuple(
            0.5 * float(delta_s) * float(torch.dot(block.reshape(-1), block.reshape(-1)).item())
            for block in self.split(value)
        )

    def frozen_clone(self) -> "FullModelControlOperator":
        """Freeze T0,H0,J_Phi0,C0 by freezing functional primals and right factors."""

        return FullModelControlOperator(
            self.model,
            self.batch,
            self.z_layer,
            self.weight_names,
            tuple(value.detach().clone() for value in self.right_factors),
            self.direct_gram_floor,
            primals_override=tuple(value.detach().clone() for value in self.primals),
            linearization_state="FROZEN_ENTRY_T0_H0_JPHI0_C0",
        )

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        tangents = self.coefficient_to_tangents(value)
        with _forward_ad_attention(self.model):
            _, tangent = jvp(self._functional_phi, self.primals, tangents)
        self.jvp_count += 1
        return tangent.detach().float()

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        if value.dtype != torch.float32 or value.shape != (self.output_dimension,):
            raise ScientificBoundary("activation cotangent shape/dtype mismatch")
        with _forward_ad_attention(self.model):
            _, pullback = vjp(self._functional_phi, *self.primals)
            gradients = pullback(value)
        blocks = []
        for gradient, right, weight in zip(gradients, self.right_factors, self.primals, strict=True):
            if weight.shape[1] == right.shape[1]:
                block = gradient @ right.T
            elif weight.shape[0] == right.shape[1]:
                block = gradient.T @ right.T
            else:
                raise ScientificBoundary("MEMIT adjoint orientation mismatch")
            blocks.append(block.reshape(-1))
        self.vjp_count += 1
        return torch.cat(blocks).detach().float()

    def output_preconditioner(self, value: torch.Tensor) -> torch.Tensor:
        return value / self.direct_gram_floor

    def receipt(self) -> dict[str, Any]:
        return {
            "coefficient_dimension": self.coefficient_dimension,
            "output_dimension": self.output_dimension,
            "jvp_count": self.jvp_count,
            "vjp_count": self.vjp_count,
            "forward_count": self.forward_count,
            "direct_gram_floor": self.direct_gram_floor,
            "explicit_kronecker_count": 0,
            "dense_jacobian_count": 0,
            "linearization_state": self.linearization_state,
        }


class MEMITGeometryFactory:
    """Freeze original covariance metric; refresh only current MEMIT keys."""

    def __init__(
        self, model: Any, tokenizer: Any, rows: list[dict[str, Any]], requests: list[dict[str, Any]],
        hparams: Any, batch: dict[str, torch.Tensor]
    ) -> None:
        from easyeditor.models.memit import memit_main

        self.model = model
        self.tokenizer = tokenizer
        self.rows = rows
        self.requests = requests
        self.hparams = hparams
        self.batch = batch
        self.context_templates = memit_main.get_context_templates(model, tokenizer)
        self.weight_names = tuple(f"{hparams.rewrite_module_tmp.format(layer)}.weight" for layer in hparams.layers)
        covariances = []
        cholesky = []
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
                    raise ScientificBoundary("stock MEMIT covariance is not Cholesky-feasible")
                epsilon = float(
                    256.0 * torch.finfo(torch.float32).eps * float(torch.diagonal(covariance).mean().item())
                )
                covariances.append(covariance)
                cholesky.append(factor)
                epsilons.append(epsilon)
        self.covariances = tuple(covariances)
        self.cholesky = tuple(cholesky)
        self.epsilons = tuple(epsilons)

    def build(self) -> tuple[FullModelControlOperator, dict[str, Any]]:
        from easyeditor.models.memit import memit_main

        lock = NumericalLock()
        whitened_rights = []
        layer_receipts = []
        last_keys = None
        with official_model_name_binding(self.model, str(self.hparams.model_name)):
            for layer, name, covariance, factor, epsilon in zip(
                self.hparams.layers, self.weight_names, self.covariances, self.cholesky, self.epsilons, strict=True
            ):
                keys = memit_main.compute_ks(
                    self.model, self.tokenizer, self.requests, self.hparams, layer, self.context_templates
                ).T.float()
                if keys.shape[1] != len(self.requests):
                    raise ScientificBoundary("MEMIT key/request width mismatch")
                base = torch.cholesky_solve(keys, factor)
                small = (
                    float(self.hparams.mom2_update_weight) * torch.eye(keys.shape[1], device=keys.device, dtype=keys.dtype)
                    + keys.T @ base
                )
                native_right = torch.linalg.solve(small, base.T).detach()
                whitened, gauge = gauge_free_whitened_right(native_right, covariance, epsilon)
                rank = int(gauge["reduced_rank"])
                error = float(gauge["whitening_identity_error"])
                if error > 16.0 * lock.finite_difference_step:
                    raise ScientificBoundary(f"metric whitening identity failed: {error}")
                whitened_rights.append(whitened.detach())
                layer_receipts.append(asdict(LayerBasisReceipt(
                    weight_name=name,
                    key_sha256=tensor_sha256(keys),
                    native_right_sha256=tensor_sha256(native_right),
                    whitened_right_sha256=tensor_sha256(whitened),
                    request_width=int(gauge["request_width"]),
                    reduced_rank=rank,
                    removed_gauge_dimension=int(gauge["removed_gauge_dimension"]),
                    metric_epsilon=epsilon,
                    gram_eigen_min=float(gauge["gram_eigen_min"]),
                    gram_eigen_max=float(gauge["gram_eigen_max"]),
                )))
                last_keys = keys
        if last_keys is None:
            raise ScientificBoundary("MEMIT geometry has no editable layers")
        canonical_inputs = memit_main.get_module_input_output_at_words(
            self.model, self.tokenizer, int(self.hparams.layers[-1]),
            context_templates=[row["requested_rewrite"]["prompt"] for row in self.rows],
            words=[row["requested_rewrite"]["subject"] for row in self.rows],
            module_template=self.hparams.rewrite_module_tmp,
            fact_token_strategy=self.hparams.fact_token, track="in",
        ).T.float()
        direct = whitened_rights[-1] @ canonical_inputs
        singular = torch.linalg.svdvals(direct)
        gram_floor = float(singular.min().square().item())
        payload = {
            "layers": layer_receipts,
            "writer_factor_solve_count": len(layer_receipts),
            "covariance_factorization_count": len(self.cholesky),
            "direct_map_sha256": tensor_sha256(direct),
            "direct_gram_floor": gram_floor,
            "dense_inverse_count": 0,
            "explicit_kronecker_count": 0,
        }
        payload["identity"] = canonical_hash(payload)
        operator = FullModelControlOperator(
            self.model, self.batch, int(self.hparams.layers[-1]), self.weight_names,
            tuple(whitened_rights), gram_floor,
        )
        return operator, payload
