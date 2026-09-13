"""Pinned AlphaEdit writer geometry and exact cache snapshots for atomic B10."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from project.run_scripts.alphaedit_strength_neutral_barrier.official_state import prepare_official_state
from project.run_scripts.fixed_z_nonuniqueness.official import official_model_name_binding

from .contracts import NumericalLock, ScientificBoundary
from .geometry import (
    FullModelControlOperator,
    LayerBasisReceipt,
    gauge_free_whitened_right,
)
from .hashing import canonical_hash, tensor_sha256


class AlphaCacheSnapshot:
    """Exact-copy snapshot of stock AlphaEdit P/cache globals."""

    def __init__(self, projector: torch.Tensor, cache: torch.Tensor, *, p_loaded: bool, cache_new: bool) -> None:
        self.projector = projector.detach().clone().to("cpu", torch.float32)
        self.cache = cache.detach().clone().to("cpu", torch.float32)
        self.p_loaded = bool(p_loaded)
        self.cache_new = bool(cache_new)
        self.root = canonical_hash({
            "projector": tensor_sha256(self.projector),
            "cache": tensor_sha256(self.cache),
            "p_loaded": self.p_loaded,
            "cache_new": self.cache_new,
        })

    @classmethod
    def cold(cls, model: Any, tokenizer: Any, hparams: Any) -> "AlphaCacheSnapshot":
        from easyeditor.models.alphaedit import AlphaEdit_main as official

        prepare_official_state(model, tokenizer, hparams, reset_cache=True)
        return cls(
            official.P,
            official.cache_c,
            p_loaded=official.P_loaded,
            cache_new=official.cache_c_new,
        )

    def restore(self) -> dict[str, Any]:
        from easyeditor.models.alphaedit import AlphaEdit_main as official

        if not isinstance(getattr(official, "P", None), torch.Tensor):
            official.P = self.projector.detach().clone()
        else:
            official.P.copy_(self.projector)
        if not isinstance(getattr(official, "cache_c", None), torch.Tensor):
            official.cache_c = self.cache.detach().clone()
        else:
            official.cache_c.copy_(self.cache)
        official.P_loaded = self.p_loaded
        official.cache_c_new = self.cache_new
        observed = AlphaCacheSnapshot(
            official.P,
            official.cache_c,
            p_loaded=official.P_loaded,
            cache_new=official.cache_c_new,
        )
        if observed.root != self.root:
            raise ScientificBoundary("AlphaEdit projector/cache exact restore failed")
        return {"exact": True, "root": self.root}

    def receipt(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "projector_sha256": tensor_sha256(self.projector),
            "cache_sha256": tensor_sha256(self.cache),
            "projector_shape": list(self.projector.shape),
            "cache_shape": list(self.cache.shape),
            "dtype": "torch.float32",
        }


class AlphaEditGeometryFactory:
    """Freeze P/history/M0 and refresh current AlphaEdit keys only."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        rows: list[dict[str, Any]],
        requests: list[dict[str, Any]],
        hparams: Any,
        batch: dict[str, torch.Tensor],
        cache_snapshot: AlphaCacheSnapshot,
    ) -> None:
        from easyeditor.models.alphaedit import AlphaEdit_main as official

        self.model = model
        self.tokenizer = tokenizer
        self.rows = rows
        self.requests = requests
        self.hparams = hparams
        self.batch = batch
        self.context_templates = official.get_context_templates(model, tokenizer)
        self.weight_names = tuple(f"{hparams.rewrite_module_tmp.format(layer)}.weight" for layer in hparams.layers)
        self.projectors = tuple(cache_snapshot.projector[index].detach().clone() for index in range(len(hparams.layers)))
        self.history = tuple(cache_snapshot.cache[index].detach().clone() for index in range(len(hparams.layers)))
        covariances: list[torch.Tensor] = []
        epsilons: list[float] = []
        with official_model_name_binding(model, str(hparams.model_name)):
            for layer in hparams.layers:
                covariance = official.get_cov(
                    model,
                    tokenizer,
                    hparams.rewrite_module_tmp.format(layer),
                    hparams.mom2_dataset,
                    hparams.mom2_n_samples,
                    hparams.mom2_dtype,
                    hparams=hparams,
                ).float()
                epsilon = float(
                    256.0 * torch.finfo(torch.float32).eps
                    * float(torch.diagonal(covariance).mean().item())
                )
                covariances.append(covariance)
                epsilons.append(epsilon)
        self.covariances = tuple(covariances)
        self.epsilons = tuple(epsilons)

    def build(self) -> tuple[FullModelControlOperator, dict[str, Any]]:
        from easyeditor.models.alphaedit import AlphaEdit_main as official
        from easyeditor.models.alphaedit.compute_ks import compute_ks

        lock = NumericalLock()
        rights: list[torch.Tensor] = []
        receipts: list[dict[str, Any]] = []
        last_keys: torch.Tensor | None = None
        with official_model_name_binding(self.model, str(self.hparams.model_name)):
            for layer, name, projector_cpu, history_cpu, covariance, epsilon in zip(
                self.hparams.layers,
                self.weight_names,
                self.projectors,
                self.history,
                self.covariances,
                self.epsilons,
                strict=True,
            ):
                keys = compute_ks(
                    self.model,
                    self.tokenizer,
                    self.requests,
                    self.hparams,
                    layer,
                    self.context_templates,
                ).T.float()
                projector = projector_cpu.to(keys.device)
                history = history_cpu.to(keys.device)
                identity = torch.eye(keys.shape[0], dtype=torch.float32, device=keys.device)
                solve_matrix = projector @ (keys @ keys.T + history) + float(self.hparams.L2) * identity
                rhs = projector @ keys
                writer_map = torch.linalg.solve(solve_matrix, rhs)
                solve_residual = solve_matrix @ writer_map - rhs
                denominator = (
                    torch.linalg.matrix_norm(solve_matrix) * torch.linalg.matrix_norm(writer_map)
                    + torch.linalg.matrix_norm(rhs)
                ).clamp_min(torch.finfo(torch.float32).tiny)
                backward = float((torch.linalg.matrix_norm(solve_residual) / denominator).item())
                native_right = writer_map.T.detach()
                whitened, gauge = gauge_free_whitened_right(native_right, covariance, epsilon)
                error = float(gauge["whitening_identity_error"])
                if error > 16.0 * lock.finite_difference_step:
                    raise ScientificBoundary(f"AlphaEdit metric whitening identity failed: {error}")
                rights.append(whitened)
                row = asdict(LayerBasisReceipt(
                    weight_name=name,
                    key_sha256=tensor_sha256(keys),
                    native_right_sha256=tensor_sha256(native_right),
                    whitened_right_sha256=tensor_sha256(whitened),
                    request_width=int(gauge["request_width"]),
                    reduced_rank=int(gauge["reduced_rank"]),
                    removed_gauge_dimension=int(gauge["removed_gauge_dimension"]),
                    metric_epsilon=epsilon,
                    gram_eigen_min=float(gauge["gram_eigen_min"]),
                    gram_eigen_max=float(gauge["gram_eigen_max"]),
                ))
                row.update({
                    "writer": "PINNED_STOCK_ALPHAEDIT_HARD_PROJECTED_BASIS",
                    "projector_sha256": tensor_sha256(projector_cpu),
                    "history_cache_sha256": tensor_sha256(history_cpu),
                    "native_solve_backward_error": backward,
                    "native_solve_matrix_rank": int(torch.linalg.matrix_rank(solve_matrix).item()),
                })
                receipts.append(row)
                last_keys = keys
        if last_keys is None:
            raise ScientificBoundary("AlphaEdit geometry has no editable layers")
        canonical_inputs = official.get_module_input_output_at_words(
            self.model,
            self.tokenizer,
            int(self.hparams.layers[-1]),
            context_templates=[row["requested_rewrite"]["prompt"] for row in self.rows],
            words=[row["requested_rewrite"]["subject"] for row in self.rows],
            module_template=self.hparams.rewrite_module_tmp,
            fact_token_strategy=self.hparams.fact_token,
            track="in",
        ).T.float()
        direct = rights[-1] @ canonical_inputs
        singular = torch.linalg.svdvals(direct)
        gram_floor = float(singular.min().square().item())
        payload = {
            "method": "alphaedit",
            "layers": receipts,
            "writer_factor_solve_count": len(receipts),
            "covariance_load_count": len(self.covariances),
            "projector_refresh_count": 0,
            "history_cache_refresh_count": 0,
            "direct_map_sha256": tensor_sha256(direct),
            "direct_gram_floor": gram_floor,
            "dense_inverse_count": 0,
            "explicit_kronecker_count": 0,
        }
        payload["identity"] = canonical_hash(payload)
        return FullModelControlOperator(
            self.model,
            self.batch,
            int(self.hparams.layers[-1]),
            self.weight_names,
            tuple(rights),
            gram_floor,
        ), payload
