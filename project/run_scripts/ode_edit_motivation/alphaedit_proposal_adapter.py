"""ODE-side proposal adapter for pinned isolated-first-edit AlphaEdit geometry.

The adapter reuses the verified EasyEdit activation/key extraction bridge but
never imports or invokes AlphaEdit's mutating entrypoint.  Projectors are read
from :class:`AlphaEditProjectorBank`, ``cache_c`` is exactly zero for every
atomic case, and no state is accumulated across cases.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .alphaedit_factors import (
    GENUINE_ISOLATED_ALPHAEDIT_SOLVER_PREFIX,
    POSTHOC_ALPHAEDIT_PROJECTOR_TOKEN,
    UNPROJECTED_ISOLATED_ALPHAEDIT_SOLVER_PREFIX,
    AlphaEditRightLeak,
    alphaedit_factor_right_leak,
    make_isolated_alphaedit_proposal,
    make_posthoc_alphaedit_proposal,
    make_unprojected_isolated_alphaedit_proposal,
)
from .alphaedit_reference import AlphaEditSolverConfig
from .contracts import (
    ContractError,
    EditRequest,
    LowRankFactor,
    MemitFactorProposal,
    ProposalSemantics,
    SnapshotManifest,
)
from .direct_z import FrozenDirectZ
from .easyedit_bridge import EasyEditBridge
from .frozen_target_lineage import FrozenTargetLineage, proposal_direction_hash
from .hooks import TemporaryLowRankApplication, capture_snapshot
from .projector_adapter import AlphaEditProjectorBank


ALPHA_SOLVE_RESIDUAL_TOLERANCE = 5e-4


@dataclass(frozen=True, slots=True)
class AlphaProposalBuild:
    """One proposal plus scalar-only construction diagnostics."""

    proposal: MemitFactorProposal
    solve_residual_by_layer: Mapping[int, float]
    construction: str

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, MemitFactorProposal):
            raise ContractError("Alpha proposal build requires a factor proposal")
        if not self.solve_residual_by_layer:
            raise ContractError("Alpha proposal build has no layer residuals")
        for layer, value in self.solve_residual_by_layer.items():
            if (
                isinstance(layer, bool)
                or not isinstance(layer, int)
                or not isinstance(value, float)
                or not math.isfinite(value)
                or value < 0.0
                or value > ALPHA_SOLVE_RESIDUAL_TOLERANCE
            ):
                raise ContractError("Alpha proposal solve residual failed its tolerance")
        if self.construction not in {
            "genuine-p-inside-solve",
            "posthoc-unprojected-alpha-base-at-p",
        }:
            raise ContractError("unknown Alpha proposal construction")

    @property
    def max_solve_residual(self) -> float:
        return max(self.solve_residual_by_layer.values())


@dataclass(frozen=True, slots=True)
class AlphaProposalLeak:
    """Aggregate right-projector leak over block-separated layer weights."""

    update_frobenius: float
    leak_frobenius: float
    leak_ratio: float
    per_layer_ratio: Mapping[int, float]

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.update_frobenius)
            or not math.isfinite(self.leak_frobenius)
            or not math.isfinite(self.leak_ratio)
            or self.update_frobenius <= 0.0
            or self.leak_frobenius < 0.0
            or self.leak_ratio < 0.0
        ):
            raise ContractError("Alpha proposal leak summary is invalid")
        if not self.per_layer_ratio or any(
            not math.isfinite(value) or value < 0.0
            for value in self.per_layer_ratio.values()
        ):
            raise ContractError("Alpha proposal per-layer leak is invalid")


def _snapshot_parameter_hashes(snapshot: SnapshotManifest) -> dict[str, str]:
    return {record.name: record.sha256 for record in snapshot.parameters}


def _snapshot_parameter_layout(
    snapshot: SnapshotManifest,
) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        record.name: (tuple(record.shape), record.dtype)
        for record in snapshot.parameters
    }


@dataclass(frozen=True, slots=True)
class _OrderedAlphaLineageHop:
    """One exact temporary layer-write transition under the W0 target."""

    parent_snapshot_id: str
    parent_state_id: str
    child_snapshot_id: str
    child_state_id: str
    action_hash: str
    child_parameter_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _OrderedAlphaTargetLineage:
    """Bounded local ancestry for ordered Alpha target reuse.

    ``FrozenTargetLineage`` intentionally permits only the preregistered MV-2
    and quarter-step envelopes.  An ordered Alpha solve is neither envelope,
    so it must not counterfeit one of those hop labels or scales.  This local
    receipt instead chains every active layer application while retaining the
    canonical frozen-target object as the sole authority for the W0 target
    bytes and token identity.
    """

    target_origin: FrozenTargetLineage
    entry_snapshot: SnapshotManifest
    terminal_snapshot: SnapshotManifest
    hops: tuple[_OrderedAlphaLineageHop, ...] = ()

    @classmethod
    def start(
        cls,
        *,
        target_origin: FrozenTargetLineage,
        entry_snapshot: SnapshotManifest,
        direct_z: FrozenDirectZ,
        target_token_ids: torch.Tensor,
    ) -> "_OrderedAlphaTargetLineage":
        target_origin.assert_authorizes(
            direct_z=direct_z,
            snapshot=entry_snapshot,
            target_token_ids=target_token_ids,
        )
        return cls(
            target_origin=target_origin,
            entry_snapshot=entry_snapshot,
            terminal_snapshot=entry_snapshot,
        )

    def _assert_terminal_snapshot(self, snapshot: SnapshotManifest) -> None:
        terminal = self.terminal_snapshot
        if (
            snapshot.snapshot_id != terminal.snapshot_id
            or snapshot.state_id != terminal.state_id
            or snapshot.model_id != terminal.model_id
            or snapshot.context_id != terminal.context_id
            or snapshot.request_ids != terminal.request_ids
            or snapshot.hparams_sha256 != terminal.hparams_sha256
            or _snapshot_parameter_hashes(snapshot)
            != _snapshot_parameter_hashes(terminal)
        ):
            raise ContractError(
                "ordered Alpha snapshot is outside its verified target lineage"
            )

    def assert_authorizes(
        self,
        *,
        direct_z: FrozenDirectZ,
        snapshot: SnapshotManifest,
        target_token_ids: torch.Tensor,
    ) -> None:
        # The canonical authority continues to validate the immutable W0
        # artifact and tokens.  The local chain separately proves that the
        # requested descendant is reached only by the observed Alpha writes.
        self.target_origin.assert_authorizes(
            direct_z=direct_z,
            snapshot=self.entry_snapshot,
            target_token_ids=target_token_ids,
        )
        self._assert_terminal_snapshot(snapshot)

    def derive_child(
        self,
        *,
        parent_snapshot: SnapshotManifest,
        child_snapshot: SnapshotManifest,
        application: TemporaryLowRankApplication,
        direct_z: FrozenDirectZ,
        target_token_ids: torch.Tensor,
    ) -> "_OrderedAlphaTargetLineage":
        """Derive and authorize one exact active layer-write descendant."""

        self._assert_terminal_snapshot(parent_snapshot)
        if type(application) is not TemporaryLowRankApplication:
            raise ContractError(
                "ordered Alpha lineage requires the canonical application"
            )
        if application.scale != 1.0:
            raise ContractError("ordered Alpha lineage requires application scale one")
        proposal = application.proposal
        if not isinstance(proposal, MemitFactorProposal):
            raise ContractError("ordered Alpha lineage requires a factor proposal")
        proposal_snapshot = proposal.snapshot
        if (
            proposal_snapshot.snapshot_id != parent_snapshot.snapshot_id
            or proposal_snapshot.state_id != parent_snapshot.state_id
            or proposal_snapshot.model_id != parent_snapshot.model_id
            or proposal_snapshot.context_id != parent_snapshot.context_id
            or proposal_snapshot.request_ids != parent_snapshot.request_ids
            or proposal_snapshot.hparams_sha256 != parent_snapshot.hparams_sha256
            or _snapshot_parameter_hashes(proposal_snapshot)
            != _snapshot_parameter_hashes(parent_snapshot)
        ):
            raise ContractError(
                "ordered Alpha proposal does not start at the lineage terminal"
            )
        if (
            child_snapshot.model_id != parent_snapshot.model_id
            or child_snapshot.context_id != parent_snapshot.context_id
            or child_snapshot.request_ids != parent_snapshot.request_ids
            or child_snapshot.hparams_sha256 != parent_snapshot.hparams_sha256
            or _snapshot_parameter_layout(child_snapshot)
            != _snapshot_parameter_layout(parent_snapshot)
        ):
            raise ContractError("ordered Alpha transition changed immutable metadata")

        parent_hashes = _snapshot_parameter_hashes(parent_snapshot)
        child_hashes = _snapshot_parameter_hashes(child_snapshot)
        factor_names = tuple(factor.weight_name for factor in proposal.factors)
        if (
            not factor_names
            or len(factor_names) != len(set(factor_names))
            or not set(factor_names).issubset(parent_hashes)
            or any(
                factor.expected_weight_sha256 != parent_hashes[factor.weight_name]
                for factor in proposal.factors
            )
        ):
            raise ContractError("ordered Alpha factors are outside the lineage parent")
        observed = application.applied_hashes
        if set(observed) != set(factor_names) or any(
            observed[name] != child_hashes[name] for name in factor_names
        ):
            raise ContractError("ordered Alpha application receipt is incomplete")
        if any(
            child_hashes[name] != parent_hashes[name]
            for name in set(parent_hashes) - set(factor_names)
        ):
            raise ContractError("an unobserved parameter changed in ordered Alpha")
        derived = _OrderedAlphaTargetLineage(
            target_origin=self.target_origin,
            entry_snapshot=self.entry_snapshot,
            terminal_snapshot=child_snapshot,
            hops=(
                *self.hops,
                _OrderedAlphaLineageHop(
                    parent_snapshot_id=parent_snapshot.snapshot_id,
                    parent_state_id=parent_snapshot.state_id,
                    child_snapshot_id=child_snapshot.snapshot_id,
                    child_state_id=child_snapshot.state_id,
                    action_hash=proposal_direction_hash(proposal),
                    child_parameter_hashes=tuple(sorted(child_hashes.items())),
                ),
            ),
        )
        # Revalidate the W0 artifact/tokens only after the exact child receipt
        # has been bound, so every subsequent residual computation has an
        # explicit descendant authorization.
        derived.assert_authorizes(
            direct_z=direct_z,
            snapshot=child_snapshot,
            target_token_ids=target_token_ids,
        )
        return derived


def _cpu_factor(factor: LowRankFactor, *, expected_sha256: str) -> LowRankFactor:
    return LowRankFactor(
        weight_name=factor.weight_name,
        left=factor.left.detach().cpu().float(),
        right=factor.right.detach().cpu().float(),
        expected_weight_sha256=expected_sha256,
        native_update_transposed=factor.native_update_transposed,
    )


def _combine_layer_factors(
    *,
    snapshot: SnapshotManifest,
    factors: Sequence[LowRankFactor],
    semantics: ProposalSemantics,
    solver_name: str,
    residual_denominator: int | None,
) -> MemitFactorProposal:
    return MemitFactorProposal(
        snapshot=snapshot,
        factors=tuple(factors),
        semantics=semantics,
        solver_name=solver_name,
        residual_denominator=residual_denominator,
    )


def _factor_adjusted_keys(factor: LowRankFactor) -> torch.Tensor:
    # The fixed Llama/Qwen down_proj weights are [d_out, d_in].  Fail closed
    # instead of silently assigning an input-space diagnostic to another
    # orientation.
    if not factor.native_update_transposed:
        raise ContractError("fixed AlphaEdit adapter requires transposed down_proj updates")
    return factor.right


def _solve_residual_ratio(
    *,
    keys: torch.Tensor,
    projected_keys: torch.Tensor,
    factor: LowRankFactor,
    l2: float,
) -> float:
    adjusted = _factor_adjusted_keys(factor).to(
        device=keys.device,
        dtype=torch.float32,
    )
    k = keys.detach().to(dtype=torch.float32)
    u = projected_keys.detach().to(dtype=torch.float32)
    residual = l2 * adjusted + u @ (k.transpose(0, 1) @ adjusted) - u
    denominator = torch.linalg.vector_norm(u)
    if not bool(torch.isfinite(residual).all()) or not bool(torch.isfinite(denominator)):
        raise ContractError("AlphaEdit linear-system residual is non-finite")
    denominator_value = float(denominator.detach().cpu())
    if denominator_value <= 0.0:
        raise ContractError("AlphaEdit projected key direction collapsed")
    ratio = float(torch.linalg.vector_norm(residual).detach().cpu()) / denominator_value
    if not math.isfinite(ratio) or ratio > ALPHA_SOLVE_RESIDUAL_TOLERANCE:
        raise ContractError("AlphaEdit linear-system residual exceeds tolerance")
    return max(0.0, ratio)


class AlphaEditProposalAdapter:
    """Build genuine or post-hoc Alpha proposals at verified model states."""

    def __init__(
        self,
        *,
        bridge: EasyEditBridge,
        config: AlphaEditSolverConfig,
        projector_bank: AlphaEditProjectorBank,
        model: torch.nn.Module,
        tokenizer: Any,
        request: EditRequest,
        memit_hparams: Any,
        contexts: Any,
        model_id: str,
        direct_z: FrozenDirectZ,
        target_token_ids: torch.Tensor,
        provenance_id: str,
    ) -> None:
        if not isinstance(bridge, EasyEditBridge):
            raise ContractError("Alpha adapter requires the verified EasyEdit bridge")
        if not isinstance(config, AlphaEditSolverConfig):
            raise ContractError("Alpha adapter requires a pinned solver config")
        if not isinstance(projector_bank, AlphaEditProjectorBank):
            raise ContractError("Alpha adapter requires a pinned projector bank")
        if not isinstance(request, EditRequest) or not isinstance(direct_z, FrozenDirectZ):
            raise ContractError("Alpha adapter request/target type is invalid")
        self.bridge = bridge
        self.config = config
        self.projector_bank = projector_bank
        self.model = model
        self.tokenizer = tokenizer
        self.request = request
        self.hparams = memit_hparams
        self.contexts = contexts
        self.model_id = model_id
        self.direct_z = direct_z
        self.target_token_ids = target_token_ids.detach().cpu().contiguous().clone()
        self.provenance_id = provenance_id
        self.weight_names = tuple(
            f"{config.rewrite_module_tmp.format(layer)}.weight"
            for layer in config.layers
        )
        if (
            tuple(int(layer) for layer in memit_hparams.layers) != config.layers
            or str(memit_hparams.fact_token) != config.fact_token
            or str(memit_hparams.rewrite_module_tmp) != config.rewrite_module_tmp
            or str(memit_hparams.layer_module_tmp) != config.layer_module_tmp
            or direct_z.z_layer != config.layers[-1]
            or direct_z.request_ids != (request.request_id,)
        ):
            raise ContractError("Alpha adapter target and shared solver semantics differ")
        self._last_builds: list[AlphaProposalBuild] = []

    @property
    def builds(self) -> tuple[AlphaProposalBuild, ...]:
        return tuple(self._last_builds)

    def _snapshot(self) -> SnapshotManifest:
        return capture_snapshot(
            self.model,
            model_id=self.model_id,
            requests=(self.request,),
            context_id=self.contexts.manifest_id,
            hparams=self.hparams,
            weight_names=self.weight_names,
            provenance_ids=(self.provenance_id,),
        )

    def _current_z(self) -> torch.Tensor:
        bindings = self.bridge.load()
        with torch.no_grad():
            value = bindings.compute_z.get_module_input_output_at_words(
                self.model,
                self.tokenizer,
                self.config.layers[-1],
                context_templates=[self.request.prompt],
                words=[self.request.subject],
                module_template=self.config.layer_module_tmp,
                fact_token_strategy=self.config.fact_token,
            )[1].transpose(0, 1)
        if (
            value.ndim != 2
            or value.shape[1] != 1
            or value.shape[0] != self.direct_z.values.shape[0]
            or not bool(torch.isfinite(value).all())
        ):
            raise ContractError("Alpha adapter current z has an invalid shape")
        return value.detach()

    def _keys(self, layer: int) -> torch.Tensor:
        bindings = self.bridge.load()
        with torch.no_grad():
            keys = bindings.compute_ks.compute_ks(
                self.model,
                self.tokenizer,
                [self.request.to_easyedit()],
                self.hparams,
                layer,
                self.contexts.to_easyedit(),
            ).transpose(0, 1)
        if (
            keys.ndim != 2
            or keys.shape[0] <= 0
            or keys.shape[1] <= 0
            or not bool(torch.isfinite(keys).all())
        ):
            raise ContractError("Alpha adapter keys have an invalid shape")
        return keys.detach()

    def _distributed_residual(
        self,
        *,
        current_z: torch.Tensor,
        key_count: int,
        denominator: int,
    ) -> torch.Tensor:
        if key_count % current_z.shape[1] != 0 or denominator <= 0:
            raise ContractError("Alpha adapter residual distribution is invalid")
        target = self.direct_z.values.to(
            device=current_z.device,
            dtype=current_z.dtype,
        )
        residual = target - current_z
        repeat = key_count // current_z.shape[1]
        return residual.repeat_interleave(repeat, dim=1) / float(denominator)

    def _one_layer(
        self,
        *,
        snapshot: SnapshotManifest,
        layer: int,
        denominator: int,
        construction: str,
        suffix: str,
        current_z: torch.Tensor | None = None,
    ) -> tuple[MemitFactorProposal, float]:
        keys = self._keys(layer)
        observed_z = self._current_z() if current_z is None else current_z
        residual = self._distributed_residual(
            current_z=observed_z,
            key_count=keys.shape[1],
            denominator=denominator,
        )
        weight_name = f"{self.config.rewrite_module_tmp.format(layer)}.weight"
        projector = self.projector_bank.layer_matrix(
            layer,
            device=keys.device,
            dtype=torch.float32,
        )
        try:
            if construction == "genuine-p-inside-solve":
                proposal = make_isolated_alphaedit_proposal(
                    snapshot=snapshot,
                    keys=keys,
                    residuals=residual,
                    projector=projector,
                    l2=self.config.l2,
                    weight_name=weight_name,
                    solver_suffix=suffix,
                    semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
                    residual_denominator=denominator,
                )
                projected_keys = projector @ keys.to(dtype=torch.float32)
            elif construction == "posthoc-unprojected-alpha-base-at-p":
                base = make_unprojected_isolated_alphaedit_proposal(
                    snapshot=snapshot,
                    keys=keys,
                    residuals=residual,
                    l2=self.config.l2,
                    weight_name=weight_name,
                    solver_suffix=f"{suffix}/p-is-identity-base",
                    semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
                    residual_denominator=denominator,
                )
                proposal = make_posthoc_alphaedit_proposal(
                    unprojected_alpha_base=base,
                    projector=projector,
                    solver_suffix=suffix,
                )
                projected_keys = keys.to(dtype=torch.float32)
            else:
                raise ContractError("unknown Alpha layer construction")
            solve_residual = _solve_residual_ratio(
                keys=keys,
                projected_keys=projected_keys,
                factor=(base if construction.startswith("posthoc") else proposal).factors[0],
                l2=self.config.l2,
            )
        finally:
            del projector
        return proposal, solve_residual

    def propose_synchronous(
        self,
        *,
        frozen_target_lineage: FrozenTargetLineage,
        construction: str = "genuine-p-inside-solve",
        solver_suffix: str = "synchronous",
    ) -> AlphaProposalBuild:
        snapshot = self._snapshot()
        frozen_target_lineage.assert_authorizes(
            direct_z=self.direct_z,
            snapshot=snapshot,
            target_token_ids=self.target_token_ids,
        )
        factors: list[LowRankFactor] = []
        residuals: dict[int, float] = {}
        denominator = len(self.config.layers)
        current_z = self._current_z()
        for layer in self.config.layers:
            one, residual = self._one_layer(
                snapshot=snapshot,
                layer=layer,
                denominator=denominator,
                construction=construction,
                suffix=f"{solver_suffix}/layer-{layer}",
                current_z=current_z,
            )
            factor = one.factors[0]
            factors.append(
                _cpu_factor(
                    factor,
                    expected_sha256=snapshot.parameter(factor.weight_name).sha256,
                )
            )
            residuals[layer] = residual
        prefix = (
            GENUINE_ISOLATED_ALPHAEDIT_SOLVER_PREFIX
            if construction == "genuine-p-inside-solve"
            else UNPROJECTED_ISOLATED_ALPHAEDIT_SOLVER_PREFIX
            + POSTHOC_ALPHAEDIT_PROJECTOR_TOKEN
            + "unprojected-alpha-base"
        )
        proposal = _combine_layer_factors(
            snapshot=snapshot,
            factors=factors,
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name=f"{prefix}/{solver_suffix}",
            residual_denominator=denominator,
        )
        build = AlphaProposalBuild(
            proposal=proposal,
            solve_residual_by_layer=residuals,
            construction=construction,
        )
        self._last_builds.append(build)
        return build

    def propose_ordered(
        self,
        *,
        origin_lineage: FrozenTargetLineage,
        construction: str,
        solver_suffix: str,
    ) -> AlphaProposalBuild:
        origin = self._snapshot()
        ordered_lineage = _OrderedAlphaTargetLineage.start(
            target_origin=origin_lineage,
            entry_snapshot=origin,
            direct_z=self.direct_z,
            target_token_ids=self.target_token_ids,
        )
        factors: list[LowRankFactor] = []
        residuals: dict[int, float] = {}
        try:
            with ExitStack() as stack:
                for index, layer in enumerate(self.config.layers):
                    current = self._snapshot()
                    ordered_lineage._assert_terminal_snapshot(current)
                    one, residual = self._one_layer(
                        snapshot=current,
                        layer=layer,
                        denominator=len(self.config.layers) - index,
                        construction=construction,
                        suffix=f"{solver_suffix}/layer-{layer}",
                    )
                    current_factor = one.factors[0]
                    factors.append(
                        _cpu_factor(
                            current_factor,
                            expected_sha256=origin.parameter(
                                current_factor.weight_name
                            ).sha256,
                        )
                    )
                    residuals[layer] = residual
                    application = stack.enter_context(
                        TemporaryLowRankApplication(self.model, one)
                    )
                    child = self._snapshot()
                    ordered_lineage = ordered_lineage.derive_child(
                        parent_snapshot=current,
                        child_snapshot=child,
                        application=application,
                        direct_z=self.direct_z,
                        target_token_ids=self.target_token_ids,
                    )
        finally:
            # Every nested application uses exact backups.  Re-capturing W0
            # proves the ordered construction did not leak a temporary write.
            restored = self._snapshot()
            if restored.state_id != origin.state_id:
                raise ContractError("ordered Alpha proposal did not restore W0")
        if construction == "genuine-p-inside-solve":
            prefix = GENUINE_ISOLATED_ALPHAEDIT_SOLVER_PREFIX
        elif construction == "posthoc-unprojected-alpha-base-at-p":
            prefix = (
                UNPROJECTED_ISOLATED_ALPHAEDIT_SOLVER_PREFIX
                + POSTHOC_ALPHAEDIT_PROJECTOR_TOKEN
                + "unprojected-alpha-base"
            )
        else:
            raise ContractError("unknown ordered Alpha construction")
        proposal = _combine_layer_factors(
            snapshot=origin,
            factors=factors,
            semantics=ProposalSemantics.ORDERED_GAUSS_SEIDEL,
            solver_name=f"{prefix}/ordered/{solver_suffix}",
            residual_denominator=None,
        )
        build = AlphaProposalBuild(
            proposal=proposal,
            solve_residual_by_layer=residuals,
            construction=construction,
        )
        self._last_builds.append(build)
        return build

    # Compatibility method used by the existing quarter-step refreshed path.
    def propose_synchronous_memit_factors(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        requests: Sequence[EditRequest],
        hparams: Any,
        contexts: Any,
        direct_z: FrozenDirectZ,
        covariance_caches: Sequence[Any],
        *,
        model_id: str,
        solver: Any | None = None,
        residual_denominator: int | None = None,
        frozen_target_lineage: FrozenTargetLineage | None = None,
    ) -> MemitFactorProposal:
        if (
            model is not self.model
            or tokenizer is not self.tokenizer
            or tuple(requests) != (self.request,)
            or hparams is not self.hparams
            or contexts is not self.contexts
            or direct_z is not self.direct_z
            or model_id != self.model_id
            or solver is not None
            or residual_denominator is not None
            or frozen_target_lineage is None
            or not covariance_caches
        ):
            raise ContractError("quarter-step Alpha adapter call differs from its lock")
        return self.propose_synchronous(
            frozen_target_lineage=frozen_target_lineage,
            construction="genuine-p-inside-solve",
            solver_suffix=f"refreshed-call-{len(self._last_builds) + 1}",
        ).proposal

    def proposal_right_leak(self, proposal: MemitFactorProposal) -> AlphaProposalLeak:
        if not isinstance(proposal, MemitFactorProposal):
            raise ContractError("right leak requires a factor proposal")
        by_weight = {factor.weight_name: factor for factor in proposal.factors}
        if not by_weight or not set(by_weight).issubset(self.weight_names):
            raise ContractError("right leak proposal weight set is invalid")
        update_sq = 0.0
        leak_sq = 0.0
        per_layer: dict[int, float] = {}
        for layer, weight_name in zip(self.config.layers, self.weight_names, strict=True):
            factor = by_weight.get(weight_name)
            if factor is None:
                continue
            if (
                float(torch.linalg.vector_norm(factor.left).detach().cpu()) == 0.0
                or float(torch.linalg.vector_norm(factor.right).detach().cpu()) == 0.0
            ):
                # Cone diagnostics may retain exact zero coefficient blocks.
                # They contribute neither update norm nor leak norm.
                per_layer[layer] = 0.0
                continue
            device = next(self.model.parameters()).device
            projector = self.projector_bank.layer_matrix(
                layer,
                device=device,
                dtype=torch.float32,
            )
            device_factor = LowRankFactor(
                weight_name=factor.weight_name,
                left=factor.left.to(device=device, dtype=torch.float32),
                right=factor.right.to(device=device, dtype=torch.float32),
                expected_weight_sha256=factor.expected_weight_sha256,
                native_update_transposed=factor.native_update_transposed,
            )
            try:
                leak: AlphaEditRightLeak = alphaedit_factor_right_leak(
                    device_factor,
                    projector,
                )
            finally:
                del projector, device_factor
            update_sq += leak.update_frobenius * leak.update_frobenius
            leak_sq += leak.right_leak_frobenius * leak.right_leak_frobenius
            per_layer[layer] = leak.right_leak_ratio
        update_norm = math.sqrt(update_sq)
        leak_norm = math.sqrt(leak_sq)
        if update_norm <= 0.0:
            raise ContractError("right leak proposal has zero norm")
        return AlphaProposalLeak(
            update_frobenius=update_norm,
            leak_frobenius=leak_norm,
            leak_ratio=leak_norm / update_norm,
            per_layer_ratio=per_layer,
        )
