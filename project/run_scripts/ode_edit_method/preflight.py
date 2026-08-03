"""Fail-closed static and loaded-runtime preflight for the concrete backend."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from project.run_scripts.ode_edit_motivation.contracts import ProvenanceManifest
from project.run_scripts.ode_edit_motivation.easyedit_bridge import (
    CovarianceCacheSpec,
    EasyEditBridge,
    _preflight_covariance_caches,
)
from project.run_scripts.ode_edit_motivation.gpu_runtime import FixedModelRuntime
from project.run_scripts.ode_edit_motivation.manifests import (
    COUNTERFACT_RELATIVE_PATH,
    _CASE_ID_PATTERN,
    _decode_case_id,
    _iter_top_level_json_objects,
    fixed_model_spec,
    preflight_fixed_artifacts,
)
from project.run_scripts.ode_edit_motivation.mv0_fidelity import (
    _bridge_pins,
    _covariance_specs,
    _freeze_contexts,
    _load_hparams,
    _load_verified_covariances,
)

from .contracts import MethodContractError
from .events import (
    ControllerRequest,
    build_allowed_contexts,
    build_teacher_batch,
    context_manifest_payload,
)


@dataclass(slots=True)
class PreparedConcreteEnvironment:
    easyedit_root: Path
    runtime: FixedModelRuntime
    bridge: EasyEditBridge
    hparams: Any
    contexts: Any
    covariance_specs: tuple[CovarianceCacheSpec, ...]
    covariance_contract: "CovarianceRuntimeContract"
    covariance_by_layer: dict[int, torch.Tensor]
    covariance_paths: tuple[str, ...]
    fixed_artifacts: ProvenanceManifest
    controller_requests: tuple[ControllerRequest, ...]
    tokenization_manifests: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class CovarianceRuntimeContract:
    """Full-hash-once covariance identity with an in-run stat guard.

    The full manifest is verified during setup and again at terminal cleanup.
    Proposal refreshes use immutable stat plus bounded byte-sample guards so
    controller timing is not dominated by repeatedly hashing the same
    multi-GiB moment files.
    """

    manifest: ProvenanceManifest
    cache_paths: tuple[tuple[int, str], ...]
    layer_names: tuple[str, ...]
    file_stats: tuple[tuple[str, int, int, int, int, int, str], ...]

    @staticmethod
    def _sample_sha256(path: Path, size: int) -> str:
        """Hash bounded edge/middle samples; terminal cleanup rehashes all bytes."""

        width = 64 * 1024
        offsets = sorted({0, max(0, size // 2 - width // 2), max(0, size - width)})
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for offset in offsets:
                handle.seek(offset)
                chunk = handle.read(min(width, size - offset))
                digest.update(offset.to_bytes(8, "big"))
                digest.update(len(chunk).to_bytes(8, "big"))
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def capture(
        cls,
        manifest: ProvenanceManifest,
        cache_path_by_layer: Mapping[int, str],
        layer_names: Sequence[str],
    ) -> "CovarianceRuntimeContract":
        stats = []
        for record in manifest.files:
            path = Path(record.path).resolve(strict=True)
            observed = path.stat()
            if not path.is_file() or observed.st_size != record.size:
                raise MethodContractError("verified covariance stat identity differs")
            stats.append(
                (
                    str(path),
                    int(observed.st_dev),
                    int(observed.st_ino),
                    int(observed.st_size),
                    int(observed.st_mtime_ns),
                    int(observed.st_ctime_ns),
                    cls._sample_sha256(path, int(observed.st_size)),
                )
            )
        return cls(
            manifest=manifest,
            cache_paths=tuple(
                sorted((int(layer), str(path)) for layer, path in cache_path_by_layer.items())
            ),
            layer_names=tuple(layer_names),
            file_stats=tuple(sorted(stats)),
        )

    @property
    def manifest_id(self) -> str:
        return self.manifest.manifest_id

    @property
    def cache_path_by_layer(self) -> dict[int, str]:
        return dict(self.cache_paths)

    def assert_current(self) -> None:
        for (
            path_text,
            device,
            inode,
            size,
            mtime_ns,
            ctime_ns,
            sample_sha256,
        ) in self.file_stats:
            path = Path(path_text)
            try:
                observed = path.stat()
                matches = (
                    path.is_file()
                    and int(observed.st_dev) == device
                    and int(observed.st_ino) == inode
                    and int(observed.st_size) == size
                    and int(observed.st_mtime_ns) == mtime_ns
                    and int(observed.st_ctime_ns) == ctime_ns
                    and self._sample_sha256(path, size) == sample_sha256
                )
            except OSError as exc:
                raise MethodContractError(
                    "covariance artifact became unreadable during controller run"
                ) from exc
            if not matches:
                raise MethodContractError(
                    "covariance artifact changed during controller run"
                )


def load_controller_requests(
    easyedit_root: str | Path,
    case_ids: Sequence[str],
) -> tuple[ControllerRequest, ...]:
    """Stream only selected rows and immediately drop evaluation fields."""

    ordered = tuple(str(case_id) for case_id in case_ids)
    if not ordered or len(set(ordered)) != len(ordered):
        raise MethodContractError("controller case IDs must be non-empty and unique")
    requested = set(ordered)
    source = Path(easyedit_root).resolve(strict=True) / COUNTERFACT_RELATIVE_PATH
    found: dict[str, ControllerRequest] = {}
    for blob in _iter_top_level_json_objects(source):
        matches = _CASE_ID_PATTERN.findall(blob)
        if len(matches) != 1:
            raise MethodContractError("CounterFact row case identity is ambiguous")
        case_id = _decode_case_id(matches[0])
        if case_id not in requested:
            continue
        try:
            row = json.loads(blob)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MethodContractError("selected CounterFact row is invalid JSON") from exc
        request = ControllerRequest.from_counterfact_row(row)
        if request.case_id != case_id:
            raise MethodContractError("selected CounterFact identity changed on projection")
        found[case_id] = request
        del row
        if len(found) == len(requested):
            break
    missing = requested - set(found)
    if missing:
        raise MethodContractError(f"CounterFact is missing cases: {sorted(missing)}")
    return tuple(found[case_id] for case_id in ordered)


def preflight_static_inputs(
    easyedit_root: str | Path,
    *,
    model_alias: str,
    case_ids: Sequence[str],
) -> tuple[ProvenanceManifest, tuple[ControllerRequest, ...]]:
    """Hash all fixed sources/artifacts before model load or controller action."""

    root = Path(easyedit_root).resolve(strict=True)
    fixed_model_spec(model_alias)
    manifest = preflight_fixed_artifacts(root, model_alias=model_alias)
    manifest.assert_current()
    return manifest, load_controller_requests(root, case_ids)


def assert_static_lock_identities(
    easyedit_root: str | Path,
    *,
    fixed_artifacts: ProvenanceManifest,
    model_lock: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
) -> None:
    """Bind tracked lock identities to the already-verified fixed manifest."""

    root = Path(easyedit_root).resolve(strict=True)
    records: dict[str, Any] = {}
    for record in fixed_artifacts.files:
        path = Path(record.path).resolve(strict=True)
        try:
            relative = str(path.relative_to(root))
        except ValueError as exc:
            raise MethodContractError("fixed artifact escaped the EasyEdit root") from exc
        records[relative] = record

    expected = (
        (
            str(selection_lock.get("dataset_relative_path", "")),
            selection_lock.get("dataset_sha256"),
            selection_lock.get("dataset_size_bytes"),
        ),
        (
            str(model_lock.get("hparams_relative_path", "")),
            model_lock.get("hparams_sha256"),
            model_lock.get("hparams_size_bytes"),
        ),
    )
    for relative, sha256, size in expected:
        record = records.get(relative)
        if (
            record is None
            or record.sha256 != sha256
            or record.size != size
        ):
            raise MethodContractError(
                f"tracked lock identity differs from fixed artifact: {relative}"
            )
    fixed_artifacts.assert_current()


def prepare_concrete_environment(
    easyedit_root: str | Path,
    *,
    runtime: FixedModelRuntime,
    fixed_artifacts: ProvenanceManifest,
    controller_requests: Sequence[ControllerRequest],
    expected_model_lock: Mapping[str, Any],
    seed: int,
) -> PreparedConcreteEnvironment:
    """Bind the loaded model to pinned hparams, contexts, and covariance bytes."""

    root = Path(easyedit_root).resolve(strict=True)
    spec = fixed_model_spec(runtime.spec.alias)
    if (
        runtime.spec != spec
        or runtime.observed_model_commit != spec.revision
        or runtime.observed_tokenizer_commit != spec.revision
    ):
        raise MethodContractError("loaded runtime differs from fixed model revision")
    if (
        expected_model_lock.get("repository_id") != spec.repository_id
        or expected_model_lock.get("revision") != spec.revision
        or expected_model_lock.get("tokenizer_revision") != spec.revision
        or expected_model_lock.get("hparams_relative_path") != spec.hparams_path
    ):
        raise MethodContractError("numerical lock fixed-model identity differs")
    bridge = EasyEditBridge(root, expected_files=_bridge_pins())
    source_manifest = bridge.preflight()
    bindings = bridge.load()
    hparams = _load_hparams(root, spec, bindings)
    contexts = _freeze_contexts(bridge, runtime, seed=seed)
    expected_context = expected_model_lock.get("context_manifest")
    if not isinstance(expected_context, Mapping):
        raise MethodContractError("lock context manifest is absent")
    if (
        contexts.manifest_id != expected_context.get("manifest_id")
        or [len(group) for group in contexts.templates]
        != expected_context.get("group_sizes")
    ):
        raise MethodContractError("fresh context manifest differs before action")

    token_manifests = []
    for request in controller_requests:
        contexts_rendered = build_allowed_contexts(request, contexts.templates)
        # Both suffixes are constructed before action; either mismatch fails.
        build_teacher_batch(runtime.tokenizer, contexts_rendered, request.target_new)
        build_teacher_batch(runtime.tokenizer, contexts_rendered, request.target_old)
        token_manifests.append(
            context_manifest_payload(
                request, contexts.templates, runtime.tokenizer
            )
        )

    covariance_specs = _covariance_specs(root, spec)
    covariance_manifest, cache_path_by_layer, layer_names = (
        _preflight_covariance_caches(
            runtime.model,
            hparams,
            spec.layers,
            covariance_specs,
        )
    )
    covariance_contract = CovarianceRuntimeContract.capture(
        covariance_manifest,
        cache_path_by_layer,
        layer_names,
    )
    covariance_by_layer, covariance_paths = _load_verified_covariances(
        root=root,
        runtime=runtime,
        specs=covariance_specs,
    )
    fixed_artifacts.assert_current()
    source_manifest.assert_current()
    return PreparedConcreteEnvironment(
        easyedit_root=root,
        runtime=runtime,
        bridge=bridge,
        hparams=hparams,
        contexts=contexts,
        covariance_specs=covariance_specs,
        covariance_contract=covariance_contract,
        covariance_by_layer=covariance_by_layer,
        covariance_paths=covariance_paths,
        fixed_artifacts=fixed_artifacts,
        controller_requests=tuple(controller_requests),
        tokenization_manifests=tuple(token_manifests),
    )
