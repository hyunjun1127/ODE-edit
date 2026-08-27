"""Narrow, provenance-checked bridge to approved EasyEdit source files.

The default import closure remains the historical MEMIT-only bridge.  A caller
that must execute the pinned Official AlphaEdit reference may explicitly opt in
to the complete four-file AlphaEdit closure; partial or caller-selected source
sets remain forbidden.
"""

from __future__ import annotations

import importlib
import importlib.abc
import random
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

import torch

from .contracts import (
    ContextManifest,
    ContractError,
    EditRequest,
    ExpectedFileIdentity,
    MemitFactorProposal,
    ProposalSemantics,
    ProvenanceManifest,
    SnapshotManifest,
    freeze_provenance,
    orient_easyedit_factor,
    preflight_pinned_files,
    sha256_bytes,
)
from .diagnostic_math import MemitSystemSolver, NativeMemitSolver
from .direct_z import DirectZCache, FrozenDirectZ
from .frozen_target_lineage import FrozenTargetLineage
from .hooks import (
    assert_snapshot_current,
    capture_snapshot,
    resolve_parameter,
)


APPROVED_EASYEDIT_FILES: tuple[str, ...] = (
    "easyeditor/models/memit/memit_main.py",
    "easyeditor/models/memit/compute_ks.py",
    "easyeditor/models/memit/compute_z.py",
    "easyeditor/models/memit/memit_hparams.py",
    "easyeditor/util/nethook.py",
    "easyeditor/util/generate.py",
    "easyeditor/util/logit_lens.py",
    "easyeditor/util/globals.py",
    "easyeditor/util/runningstats.py",
    "easyeditor/util/hparams.py",
    "easyeditor/models/rome/layer_stats.py",
    "easyeditor/models/rome/repr_tools.py",
    "easyeditor/models/rome/tok_dataset.py",
)

APPROVED_ALPHAEDIT_REFERENCE_FILES: tuple[str, ...] = (
    "easyeditor/models/alphaedit/AlphaEdit_hparams.py",
    "easyeditor/models/alphaedit/compute_z.py",
    "easyeditor/models/alphaedit/compute_ks.py",
    "easyeditor/models/alphaedit/AlphaEdit_main.py",
)

_EASYEDIT_GLOBAL_LOCK = threading.RLock()

_MINIMAL_PACKAGE_PATHS: Mapping[str, str] = {
    "easyeditor": "easyeditor",
    "easyeditor.models": "easyeditor/models",
    "easyeditor.models.memit": "easyeditor/models/memit",
    "easyeditor.models.rome": "easyeditor/models/rome",
    "easyeditor.util": "easyeditor/util",
}

_ALPHAEDIT_PACKAGE_PATHS: Mapping[str, str] = {
    "easyeditor.models.alphaedit": "easyeditor/models/alphaedit",
}

_APPROVED_MODULE_PATHS: Mapping[str, str] = {
    relative.removesuffix(".py").replace("/", "."): relative
    for relative in APPROVED_EASYEDIT_FILES
}

_ALPHAEDIT_REFERENCE_MODULE_PATHS: Mapping[str, str] = {
    relative.removesuffix(".py").replace("/", "."): relative
    for relative in APPROVED_ALPHAEDIT_REFERENCE_FILES
}


class _VerifiedSourceLoader(importlib.abc.Loader):
    """Compile only source bytes that match one frozen provenance record."""

    def __init__(
        self,
        module_name: str,
        path: Path,
        identity: ExpectedFileIdentity,
    ) -> None:
        self.module_name = module_name
        self.path = path
        self.identity = identity

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        del spec
        return None

    def exec_module(self, module: ModuleType) -> None:
        source = self.path.read_bytes()
        if (
            len(source) != self.identity.size
            or sha256_bytes(source) != self.identity.sha256
        ):
            raise ImportError(
                f"verified EasyEdit source changed before compile: {self.path}"
            )
        module.__file__ = str(self.path)
        module.__cached__ = None
        module.__package__ = self.module_name.rpartition(".")[0]
        code = compile(
            source,
            str(self.path),
            "exec",
            dont_inherit=True,
            optimize=-1,
        )
        exec(code, module.__dict__)


class _VerifiedEasyEditFinder(importlib.abc.MetaPathFinder):
    """Deny unapproved EasyEdit imports and bypass every bytecode cache."""

    def __init__(
        self,
        sources: Mapping[str, tuple[Path, ExpectedFileIdentity]],
    ) -> None:
        self.sources = dict(sources)

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        del path, target
        if fullname != "easyeditor" and not fullname.startswith("easyeditor."):
            return None
        source = self.sources.get(fullname)
        if source is None:
            raise ImportError(f"unapproved EasyEdit module import: {fullname}")
        source_path, identity = source
        loader = _VerifiedSourceLoader(fullname, source_path, identity)
        spec = ModuleSpec(fullname, loader, origin=str(source_path))
        spec.has_location = True
        spec.cached = None
        return spec


def _require_eval_model(model: torch.nn.Module) -> None:
    training_modules = [
        name or "<root>"
        for name, module in model.named_modules()
        if bool(module.training)
    ]
    if training_modules:
        raise ContractError(
            "EasyEdit diagnostic requires model.eval(); training modules include "
            + ", ".join(training_modules[:3])
        )


@contextmanager
def _preserve_model_runtime_state(model: torch.nn.Module) -> Iterator[None]:
    """Restore module mode, cache flag, and process RNGs after an EasyEdit call."""

    training_flags = tuple((module, bool(module.training)) for module in model.modules())
    python_rng_state = random.getstate()
    try:
        import numpy as np
    except ImportError:
        np = None
        numpy_rng_state = None
    else:
        numpy_rng_state = np.random.get_state()
    cpu_rng_state = torch.get_rng_state().clone()
    cuda_rng_states = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )
    config = getattr(model, "config", None)
    missing = object()
    use_cache = getattr(config, "use_cache", missing) if config is not None else missing
    try:
        yield
    finally:
        for module, training in training_flags:
            module.training = training
        if config is not None and use_cache is not missing:
            config.use_cache = use_cache
        random.setstate(python_rng_state)
        if np is not None and numpy_rng_state is not None:
            np.random.set_state(numpy_rng_state)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states:
            torch.cuda.set_rng_state_all(list(cuda_rng_states))


@dataclass(frozen=True, slots=True)
class EasyEditBindings:
    memit_main: ModuleType
    compute_ks: ModuleType
    compute_z: ModuleType
    memit_hparams: ModuleType
    nethook: ModuleType
    generate: ModuleType
    layer_stats: ModuleType
    repr_tools: ModuleType
    provenance: ProvenanceManifest


class CovarianceCacheMissError(RuntimeError):
    """A guarded covariance request would compute or use an unpinned cache."""


@dataclass(frozen=True, slots=True)
class CovarianceCacheSpec:
    """Pinned layer-stat cache consumed by one synchronous layer."""

    layer: int
    path: str
    identity: ExpectedFileIdentity

    def __post_init__(self) -> None:
        if isinstance(self.layer, bool) or not isinstance(self.layer, int):
            raise ContractError("covariance cache layer must be an integer")
        if not isinstance(self.path, str) or not self.path:
            raise ContractError("covariance cache path must not be empty")
        object.__setattr__(
            self,
            "identity",
            ExpectedFileIdentity.from_value(self.identity),
        )

    @classmethod
    def from_value(
        cls,
        value: "CovarianceCacheSpec | Mapping[str, Any]",
    ) -> "CovarianceCacheSpec":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or set(value) != {
            "layer",
            "path",
            "sha256",
            "size",
        }:
            raise ContractError(
                "covariance cache spec requires layer, path, sha256, and size"
            )
        return cls(
            layer=int(value["layer"]),
            path=str(value["path"]),
            identity=ExpectedFileIdentity(
                sha256=value["sha256"],
                size=value["size"],
            ),
        )


def _layer_stats_cache_path(
    model: torch.nn.Module,
    *,
    layer_name: str,
    stats_dir: str | Path,
    dataset_name: str,
    to_collect: Sequence[str],
    model_name: str | None,
    sample_size: int | None,
    precision: str | None,
    batch_tokens: int | None,
) -> Path:
    """Mirror the cache filename logic in the approved layer_stats.py."""

    if batch_tokens is not None:
        raise CovarianceCacheMissError(
            "guarded MEMIT covariance loads do not permit batch_tokens overrides"
        )
    resolved_precision = "float64" if precision is None else precision
    resolved_model_name = (
        model.config._name_or_path.rsplit("/")[-1]
        if model_name is None
        else model_name
    )
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    filename = (
        f"{layer_name}_{resolved_precision}_"
        f"{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    )
    return (
        Path(stats_dir)
        / resolved_model_name
        / f"{dataset_name}_stats"
        / filename
    ).expanduser().resolve(strict=False)


@contextmanager
def guard_precomputed_layer_stats(
    bindings: EasyEditBindings,
    allowed_paths: Sequence[str | Path],
) -> Iterator[None]:
    """Fail before layer_stats can enter dataset/recompute code."""

    resolved_allowed = {
        str(Path(path).expanduser().resolve(strict=True))
        for path in allowed_paths
    }
    original_layer_stats = bindings.memit_main.layer_stats
    original_load_dataset = bindings.layer_stats.load_dataset

    def guarded_layer_stats(
        model: torch.nn.Module,
        tokenizer: Any,
        layer_name: str,
        stats_dir: str | Path,
        dataset_name: str,
        to_collect: Sequence[str],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if args:
            raise CovarianceCacheMissError(
                "unexpected positional layer_stats options; refusing an unverified load"
            )
        if kwargs.get("force_recompute", False):
            raise CovarianceCacheMissError(
                "covariance force_recompute is forbidden in synchronous diagnostics"
            )
        expected_path = _layer_stats_cache_path(
            model,
            layer_name=layer_name,
            stats_dir=stats_dir,
            dataset_name=dataset_name,
            to_collect=to_collect,
            model_name=kwargs.get("model_name"),
            sample_size=kwargs.get("sample_size"),
            precision=kwargs.get("precision"),
            batch_tokens=kwargs.get("batch_tokens"),
        )
        resolved = str(expected_path)
        if resolved not in resolved_allowed:
            raise CovarianceCacheMissError(
                f"layer_stats resolved unpinned covariance cache: {resolved}"
            )
        if not expected_path.is_file():
            raise CovarianceCacheMissError(
                f"precomputed covariance cache disappeared: {resolved}"
            )
        return original_layer_stats(
            model,
            tokenizer,
            layer_name,
            stats_dir,
            dataset_name,
            to_collect,
            **kwargs,
        )

    def reject_dataset_load(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise CovarianceCacheMissError(
            "layer_stats attempted a dataset load; covariance cache miss is forbidden"
        )

    bindings.memit_main.layer_stats = guarded_layer_stats
    bindings.layer_stats.load_dataset = reject_dataset_load
    try:
        yield
    finally:
        bindings.memit_main.layer_stats = original_layer_stats
        bindings.layer_stats.load_dataset = original_load_dataset


@contextmanager
def _isolated_covariance_cache(
    memit_main: ModuleType,
    model: torch.nn.Module,
    layer_names: Sequence[str],
) -> Iterator[None]:
    """Force validated files to be read instead of stale global COV_CACHE."""

    model_name = model.config._name_or_path.replace("/", "_")
    sentinel = object()
    saved: dict[tuple[str, str], Any] = {}
    for layer_name in layer_names:
        key = (model_name, layer_name)
        saved[key] = memit_main.COV_CACHE.pop(key, sentinel)
    try:
        yield
    finally:
        for key, previous in saved.items():
            if previous is sentinel:
                memit_main.COV_CACHE.pop(key, None)
            else:
                memit_main.COV_CACHE[key] = previous


def _preflight_covariance_caches(
    model: torch.nn.Module,
    hparams: Any,
    layers: Sequence[int],
    covariance_caches: Sequence[CovarianceCacheSpec | Mapping[str, Any]],
) -> tuple[ProvenanceManifest, dict[int, str], tuple[str, ...]]:
    """Bind an exact, already-computed covariance file to every MEMIT layer."""

    specs = tuple(CovarianceCacheSpec.from_value(value) for value in covariance_caches)
    spec_layers = [spec.layer for spec in specs]
    if len(spec_layers) != len(set(spec_layers)) or set(spec_layers) != set(layers):
        raise ContractError(
            "covariance cache specs must contain exactly one entry per MEMIT layer"
        )
    covariance_manifest = preflight_pinned_files(
        {spec.path: spec.identity for spec in specs},
        label="precomputed MEMIT covariance",
    )
    cache_path_by_layer = {
        spec.layer: str(Path(spec.path).expanduser().resolve(strict=True))
        for spec in specs
    }
    try:
        rewrite_template = str(hparams.rewrite_module_tmp)
        stats_dir = hparams.stats_dir
        dataset_name = hparams.mom2_dataset
        sample_size = hparams.mom2_n_samples
        precision = hparams.mom2_dtype
    except AttributeError as exc:
        raise ContractError(
            "hparams must expose rewrite_module_tmp and MEMIT covariance fields"
        ) from exc
    layer_names = tuple(rewrite_template.format(layer) for layer in layers)
    for layer, layer_name in zip(layers, layer_names, strict=True):
        expected_path = _layer_stats_cache_path(
            model,
            layer_name=layer_name,
            stats_dir=stats_dir,
            dataset_name=dataset_name,
            to_collect=("mom2",),
            model_name=None,
            sample_size=sample_size,
            precision=precision,
            batch_tokens=None,
        )
        if str(expected_path) != cache_path_by_layer[layer]:
            raise CovarianceCacheMissError(
                f"layer {layer} pin resolves to {cache_path_by_layer[layer]}, "
                f"but EasyEdit will request {expected_path}"
            )
    return covariance_manifest, cache_path_by_layer, layer_names


class EasyEditBridge:
    """Lazy import bridge that rejects unapproved or changed source files."""

    def __init__(
        self,
        easyedit_root: str | Path,
        *,
        expected_files: Mapping[
            str, ExpectedFileIdentity | Mapping[str, Any]
        ] | None = None,
        include_alphaedit_reference: bool = False,
    ) -> None:
        self.root = Path(easyedit_root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise ContractError(f"EasyEdit root is not a directory: {self.root}")
        if type(include_alphaedit_reference) is not bool:
            raise ContractError("AlphaEdit bridge opt-in must be a boolean")
        self.include_alphaedit_reference = include_alphaedit_reference
        self._approved_files = APPROVED_EASYEDIT_FILES + (
            APPROVED_ALPHAEDIT_REFERENCE_FILES if include_alphaedit_reference else ()
        )
        self._module_paths = dict(_APPROVED_MODULE_PATHS)
        self._package_paths = dict(_MINIMAL_PACKAGE_PATHS)
        if include_alphaedit_reference:
            self._module_paths.update(_ALPHAEDIT_REFERENCE_MODULE_PATHS)
            self._package_paths.update(_ALPHAEDIT_PACKAGE_PATHS)
        self.expected_files = dict(expected_files) if expected_files is not None else None
        if self.expected_files is not None and set(self.expected_files) != set(
            self._approved_files
        ):
            missing = set(self._approved_files) - set(self.expected_files)
            extra = set(self.expected_files) - set(self._approved_files)
            raise ContractError(
                "pinned EasyEdit source set must exactly match the approved bridge files; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        self._provenance: ProvenanceManifest | None = None
        self._bindings: EasyEditBindings | None = None
        self._loaded_modules: dict[str, ModuleType] = {}
        self._import_finder: _VerifiedEasyEditFinder | None = None

    def preflight(self) -> ProvenanceManifest:
        if self.expected_files is None:
            paths = [self.root / relative for relative in self._approved_files]
            manifest = freeze_provenance(paths, label="EasyEdit verified source")
        else:
            manifest = preflight_pinned_files(
                self.expected_files,
                label="EasyEdit verified source",
                base_dir=self.root,
            )
        manifest.assert_current()
        self._provenance = manifest
        return manifest

    def _checked_import(self, module_name: str, relative_file: str) -> ModuleType:
        module = importlib.import_module(module_name)
        loaded_file = getattr(module, "__file__", None)
        if loaded_file is None:
            raise ImportError(f"{module_name} has no source file")
        actual = Path(loaded_file).resolve(strict=True)
        expected = (self.root / relative_file).resolve(strict=True)
        if actual != expected:
            raise ImportError(f"{module_name} resolved to {actual}, expected {expected}")
        return module

    def _install_minimal_namespace(self) -> tuple[str, ...]:
        """Install inert packages so EasyEdit's eager initializers never run."""

        conflicts = sorted(
            name
            for name in sys.modules
            if name == "easyeditor" or name.startswith("easyeditor.")
        )
        if conflicts:
            raise ImportError(
                "EasyEdit bridge requires a fresh process with no preloaded "
                f"easyeditor modules; found {conflicts[:5]}"
            )
        created: list[str] = []
        try:
            for name, relative in self._package_paths.items():
                package_path = (self.root / relative).resolve(strict=True)
                module = ModuleType(name)
                module.__package__ = name
                module.__path__ = [str(package_path)]
                module.__file__ = None
                spec = ModuleSpec(name, loader=None, is_package=True)
                spec.submodule_search_locations = [str(package_path)]
                module.__spec__ = spec
                module.__loader__ = None
                setattr(module, "__ode_edit_inert_namespace__", True)
                sys.modules[name] = module
                created.append(name)
                parent_name, _, child_name = name.rpartition(".")
                if parent_name:
                    setattr(sys.modules[parent_name], child_name, module)
        except BaseException:
            for name in reversed(created):
                sys.modules.pop(name, None)
            raise
        try:
            if self._provenance is None:
                raise ImportError("EasyEdit source provenance is not frozen")
            records = {
                str(Path(record.path).resolve(strict=True)): ExpectedFileIdentity(
                    sha256=record.sha256,
                    size=record.size,
                )
                for record in self._provenance.files
            }
            sources: dict[str, tuple[Path, ExpectedFileIdentity]] = {}
            for module_name, relative in self._module_paths.items():
                source_path = (self.root / relative).resolve(strict=True)
                identity = records.get(str(source_path))
                if identity is None:
                    raise ImportError(
                        f"approved EasyEdit module lacks frozen provenance: {module_name}"
                    )
                sources[module_name] = (source_path, identity)
            finder = _VerifiedEasyEditFinder(sources)
            self._import_finder = finder
            sys.meta_path.insert(0, finder)
            return tuple(created)
        except BaseException:
            self._remove_import_finder()
            self._clear_easyeditor_modules()
            raise

    def _assert_import_closure(self) -> None:
        """Require every loaded repo-local EasyEdit module to be approved."""

        if self._import_finder is None or self._import_finder not in sys.meta_path:
            raise ImportError("verified EasyEdit import guard is no longer installed")
        approved = {
            str((self.root / relative).resolve(strict=True)): relative
            for relative in self._approved_files
        }
        observed_files: set[str] = set()
        for name, module in tuple(sys.modules.items()):
            if name != "easyeditor" and not name.startswith("easyeditor."):
                continue
            loaded_file = getattr(module, "__file__", None)
            if loaded_file is None:
                if name not in self._package_paths or not bool(
                    getattr(module, "__ode_edit_inert_namespace__", False)
                ):
                    raise ImportError(
                        f"unexpected fileless EasyEdit module in import closure: {name}"
                    )
                continue
            actual = str(Path(loaded_file).resolve(strict=True))
            if actual not in approved:
                raise ImportError(
                    f"unapproved EasyEdit source entered import closure: "
                    f"{name} -> {actual}"
                )
            observed_files.add(actual)
        if observed_files != set(approved):
            missing = sorted(set(approved) - observed_files)
            raise ImportError(
                f"approved EasyEdit source was not loaded: {missing}"
            )
        for name, module in self._loaded_modules.items():
            if sys.modules.get(name) is not module:
                raise ImportError(
                    f"EasyEdit module identity changed after verified import: {name}"
                )

    @staticmethod
    def _clear_easyeditor_modules() -> None:
        for name in sorted(
            (
                name
                for name in sys.modules
                if name == "easyeditor" or name.startswith("easyeditor.")
            ),
            key=lambda value: (value.count("."), value),
            reverse=True,
        ):
            sys.modules.pop(name, None)

    def _remove_import_finder(self) -> None:
        if self._import_finder is not None:
            while self._import_finder in sys.meta_path:
                sys.meta_path.remove(self._import_finder)
            self._import_finder = None

    def load(self) -> EasyEditBindings:
        if self._bindings is not None:
            self._bindings.provenance.assert_current()
            self._assert_import_closure()
            return self._bindings
        provenance = self._provenance or self.preflight()
        provenance.assert_current()
        with _EASYEDIT_GLOBAL_LOCK:
            self._install_minimal_namespace()
            previous_dont_write_bytecode = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            try:
                memit_main = self._checked_import(
                    "easyeditor.models.memit.memit_main",
                    "easyeditor/models/memit/memit_main.py",
                )
                compute_ks = self._checked_import(
                    "easyeditor.models.memit.compute_ks",
                    "easyeditor/models/memit/compute_ks.py",
                )
                compute_z = self._checked_import(
                    "easyeditor.models.memit.compute_z",
                    "easyeditor/models/memit/compute_z.py",
                )
                memit_hparams = self._checked_import(
                    "easyeditor.models.memit.memit_hparams",
                    "easyeditor/models/memit/memit_hparams.py",
                )
                nethook = self._checked_import(
                    "easyeditor.util.nethook",
                    "easyeditor/util/nethook.py",
                )
                generate = self._checked_import(
                    "easyeditor.util.generate",
                    "easyeditor/util/generate.py",
                )
                self._checked_import(
                    "easyeditor.util.logit_lens",
                    "easyeditor/util/logit_lens.py",
                )
                self._checked_import(
                    "easyeditor.util.globals",
                    "easyeditor/util/globals.py",
                )
                self._checked_import(
                    "easyeditor.util.runningstats",
                    "easyeditor/util/runningstats.py",
                )
                self._checked_import(
                    "easyeditor.util.hparams",
                    "easyeditor/util/hparams.py",
                )
                layer_stats = self._checked_import(
                    "easyeditor.models.rome.layer_stats",
                    "easyeditor/models/rome/layer_stats.py",
                )
                repr_tools = self._checked_import(
                    "easyeditor.models.rome.repr_tools",
                    "easyeditor/models/rome/repr_tools.py",
                )
                tok_dataset = self._checked_import(
                    "easyeditor.models.rome.tok_dataset",
                    "easyeditor/models/rome/tok_dataset.py",
                )
                alphaedit_modules: tuple[ModuleType, ...] = ()
                if self.include_alphaedit_reference:
                    alphaedit_modules = tuple(
                        self._checked_import(module_name, relative)
                        for module_name, relative in (
                            _ALPHAEDIT_REFERENCE_MODULE_PATHS.items()
                        )
                    )
                logit_lens = sys.modules["easyeditor.util.logit_lens"]
                globals_module = sys.modules["easyeditor.util.globals"]
                runningstats = sys.modules["easyeditor.util.runningstats"]
                hparams_module = sys.modules["easyeditor.util.hparams"]
                self._loaded_modules = {
                    module.__name__: module
                    for module in (
                        memit_main,
                        compute_ks,
                        compute_z,
                        memit_hparams,
                        nethook,
                        generate,
                        logit_lens,
                        globals_module,
                        runningstats,
                        hparams_module,
                        layer_stats,
                        repr_tools,
                        tok_dataset,
                        *alphaedit_modules,
                    )
                }
                self._assert_import_closure()
                provenance.assert_current()
                self._bindings = EasyEditBindings(
                    memit_main=memit_main,
                    compute_ks=compute_ks,
                    compute_z=compute_z,
                    memit_hparams=memit_hparams,
                    nethook=nethook,
                    generate=generate,
                    layer_stats=layer_stats,
                    repr_tools=repr_tools,
                    provenance=provenance,
                )
                return self._bindings
            except BaseException:
                self._bindings = None
                self._loaded_modules.clear()
                self._clear_easyeditor_modules()
                self._remove_import_finder()
                raise
            finally:
                sys.dont_write_bytecode = previous_dont_write_bytecode

    def freeze_generated_contexts(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        source: str,
        fresh: bool = True,
    ) -> ContextManifest:
        """Generate once, deep-freeze, and restore EasyEdit's global cache."""

        _require_eval_model(model)
        bindings = self.load()
        with _EASYEDIT_GLOBAL_LOCK, _preserve_model_runtime_state(model):
            previous = bindings.memit_main.CONTEXT_TEMPLATES_CACHE
            try:
                if fresh:
                    bindings.memit_main.CONTEXT_TEMPLATES_CACHE = None
                templates = bindings.memit_main.get_context_templates(model, tokenizer)
                return ContextManifest.freeze(templates, source=source)
            finally:
                bindings.memit_main.CONTEXT_TEMPLATES_CACHE = previous

    def propose_ordered_memit_factors(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        requests: Sequence[EditRequest],
        hparams: Any,
        contexts: ContextManifest,
        *,
        model_id: str,
        direct_z: FrozenDirectZ,
        covariance_caches: Sequence[
            CovarianceCacheSpec | Mapping[str, Any]
        ],
    ) -> MemitFactorProposal:
        """Return EasyEdit's canonical ordered Gauss-Seidel factor trajectory.

        Contexts are injected from a frozen manifest.  All target weights are
        hashed before the call and must be bit-identical afterwards; on any
        exception they are restored from exact backups.  EasyEdit temporarily
        applies each preceding layer update while measuring later layers, so
        these factors do *not* share one unchanged model snapshot.  ``snapshot``
        records only the guarded entry/rollback state.
        """

        requests = tuple(requests)
        if not requests:
            raise ContractError("at least one request is required")
        _require_eval_model(model)
        bindings = self.load()
        try:
            layers = tuple(int(layer) for layer in hparams.layers)
            rewrite_template = str(hparams.rewrite_module_tmp)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ContractError("hparams must expose layers and rewrite_module_tmp") from exc
        if not layers:
            raise ContractError("MEMIT layers must not be empty")
        if len(layers) != len(set(layers)):
            raise ContractError("MEMIT layers must be unique")
        (
            covariance_manifest,
            cache_path_by_layer,
            layer_names,
        ) = _preflight_covariance_caches(
            model,
            hparams,
            layers,
            covariance_caches,
        )
        weight_names = tuple(
            f"{rewrite_template.format(layer)}.weight" for layer in layers
        )
        direct_z_source_snapshot = capture_snapshot(
            model,
            model_id=model_id,
            requests=requests,
            context_id=contexts.manifest_id,
            hparams=hparams,
            weight_names=weight_names,
            provenance_ids=(bindings.provenance.manifest_id,),
        )
        if direct_z.source_snapshot_id != direct_z_source_snapshot.snapshot_id:
            raise ContractError(
                "direct-z was not generated for this source/model entry snapshot"
            )
        if (
            direct_z.source_state_id != direct_z_source_snapshot.state_id
            or direct_z.z_layer != layers[-1]
            or direct_z.request_ids != direct_z_source_snapshot.request_ids
        ):
            raise ContractError("direct-z metadata does not match the ordered request")
        direct_z_manifest = ProvenanceManifest(
            label="frozen direct-z artifact",
            files=(direct_z.artifact,),
        )
        direct_z_manifest.assert_current()
        snapshot = SnapshotManifest(
            model_id=direct_z_source_snapshot.model_id,
            context_id=direct_z_source_snapshot.context_id,
            request_ids=direct_z_source_snapshot.request_ids,
            hparams_sha256=direct_z_source_snapshot.hparams_sha256,
            parameters=direct_z_source_snapshot.parameters,
            provenance_ids=tuple(
                sorted(
                    {
                        bindings.provenance.manifest_id,
                        covariance_manifest.manifest_id,
                        direct_z.artifact_id,
                    }
                )
            ),
        )
        backups = {
            name: resolve_parameter(model, name).detach().clone()
            for name in weight_names
        }
        raw_requests = [request.to_easyedit() for request in requests]

        with _EASYEDIT_GLOBAL_LOCK, _preserve_model_runtime_state(model):
            previous_contexts = bindings.memit_main.CONTEXT_TEMPLATES_CACHE
            requires_grad = [
                (parameter, parameter.requires_grad)
                for parameter in model.parameters()
            ]
            bindings.memit_main.CONTEXT_TEMPLATES_CACHE = contexts.to_easyedit()
            original_compute_z = bindings.memit_main.compute_z
            compute_z_call_index = 0

            def frozen_compute_z(
                current_model: torch.nn.Module,
                current_tokenizer: Any,
                request: Mapping[str, Any],
                current_hparams: Any,
                z_layer: int,
                context_templates: Sequence[Sequence[str]],
            ) -> torch.Tensor:
                nonlocal compute_z_call_index
                if (
                    current_model is not model
                    or current_tokenizer is not tokenizer
                    or current_hparams is not hparams
                    or int(z_layer) != layers[-1]
                    or context_templates != contexts.to_easyedit()
                ):
                    raise ContractError(
                        "EasyEdit requested direct-z outside the frozen ordered contract"
                    )
                if compute_z_call_index >= len(raw_requests):
                    raise ContractError("EasyEdit requested direct-z too many times")
                expected_request = raw_requests[compute_z_call_index]
                if dict(request) != expected_request:
                    raise ContractError(
                        "EasyEdit direct-z request order/content differs from the frozen manifest"
                    )
                column = direct_z.values[:, compute_z_call_index]
                compute_z_call_index += 1
                return column.to(device=next(model.parameters()).device)

            bindings.memit_main.compute_z = frozen_compute_z
            try:
                with guard_precomputed_layer_stats(
                    bindings,
                    tuple(cache_path_by_layer.values()),
                ), _isolated_covariance_cache(
                    bindings.memit_main,
                    model,
                    layer_names,
                ):
                    deltas = bindings.memit_main.execute_memit(
                        model,
                        tokenizer,
                        raw_requests,
                        hparams,
                        cache_template=None,
                    )
                if compute_z_call_index != len(raw_requests):
                    raise ContractError(
                        "EasyEdit did not consume every frozen direct-z column"
                    )
                assert_snapshot_current(model, snapshot)
                covariance_manifest.assert_current()
                direct_z_manifest.assert_current()
            except BaseException:
                with torch.no_grad():
                    for name, backup in backups.items():
                        resolve_parameter(model, name).copy_(backup)
                raise
            finally:
                bindings.memit_main.CONTEXT_TEMPLATES_CACHE = previous_contexts
                bindings.memit_main.compute_z = original_compute_z
                for parameter, original_flag in requires_grad:
                    parameter.requires_grad_(original_flag)

        expected_names = set(weight_names)
        if set(deltas) != expected_names:
            raise ContractError(
                "EasyEdit returned an unexpected factor set; "
                f"expected={sorted(expected_names)}, actual={sorted(deltas)}"
            )
        factors = []
        for name in weight_names:
            pair = deltas[name]
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ContractError(f"EasyEdit delta for {name} is not an (adj_k, resid) pair")
            record = snapshot.parameter(name)
            factors.append(
                orient_easyedit_factor(
                    pair[0],
                    pair[1],
                    weight_name=name,
                    weight_shape=record.shape,
                    expected_weight_sha256=record.sha256,
                )
            )
        self._bindings.provenance.assert_current()
        return MemitFactorProposal(
            snapshot=snapshot,
            factors=tuple(factors),
            semantics=ProposalSemantics.ORDERED_GAUSS_SEIDEL,
            solver_name="easyedit.execute_memit/ordered-gauss-seidel",
            residual_denominator=None,
        )

    def propose_memit_factors(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        requests: Sequence[EditRequest],
        hparams: Any,
        contexts: ContextManifest,
        *,
        model_id: str,
        direct_z: FrozenDirectZ,
        covariance_caches: Sequence[
            CovarianceCacheSpec | Mapping[str, Any]
        ],
    ) -> MemitFactorProposal:
        """Compatibility name for the canonical ordered Gauss-Seidel proposal."""

        return self.propose_ordered_memit_factors(
            model,
            tokenizer,
            requests,
            hparams,
            contexts,
            model_id=model_id,
            direct_z=direct_z,
            covariance_caches=covariance_caches,
        )

    def load_or_compute_direct_z(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        requests: Sequence[EditRequest],
        hparams: Any,
        contexts: ContextManifest,
        *,
        model_id: str,
        local_cache_root: str | Path,
        cache_path: str | Path,
        expected_identity: ExpectedFileIdentity | Mapping[str, Any] | None = None,
    ) -> FrozenDirectZ:
        """Load a pinned direct-z artifact or compute each request exactly once.

        A cache miss is allowed only at ``cache_path`` under
        ``local_cache_root``.  Existing artifacts require full SHA-256 and size
        pins before ``torch.load`` is called.
        """

        requests = tuple(requests)
        if not requests:
            raise ContractError("at least one request is required")
        _require_eval_model(model)
        bindings = self.load()
        layers = tuple(int(layer) for layer in hparams.layers)
        if not layers:
            raise ContractError("MEMIT layers must not be empty")
        if len(layers) != len(set(layers)):
            raise ContractError("MEMIT layers must be unique")
        weight_names = tuple(
            f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            for layer in layers
        )
        source_snapshot = capture_snapshot(
            model,
            model_id=model_id,
            requests=requests,
            context_id=contexts.manifest_id,
            hparams=hparams,
            weight_names=weight_names,
            provenance_ids=(bindings.provenance.manifest_id,),
        )
        z_layer = layers[-1]
        raw_requests = [request.to_easyedit() for request in requests]

        def compute_once() -> torch.Tensor:
            with _EASYEDIT_GLOBAL_LOCK, _preserve_model_runtime_state(model):
                requires_grad = [
                    (parameter, parameter.requires_grad)
                    for parameter in model.parameters()
                ]
                try:
                    assert_snapshot_current(model, source_snapshot)
                    values = [
                        bindings.compute_z.compute_z(
                            model,
                            tokenizer,
                            request,
                            hparams,
                            z_layer,
                            contexts.to_easyedit(),
                        )
                        for request in raw_requests
                    ]
                finally:
                    for parameter, original_flag in requires_grad:
                        parameter.requires_grad_(original_flag)
                assert_snapshot_current(model, source_snapshot)
            return torch.stack(values, dim=1)

        identity = (
            None
            if expected_identity is None
            else ExpectedFileIdentity.from_value(expected_identity)
        )
        result = DirectZCache(local_cache_root, cache_path).load_or_compute(
            source_snapshot=source_snapshot,
            z_layer=z_layer,
            compute=compute_once,
            expected_identity=identity,
        )
        bindings.provenance.assert_current()
        return result

    def propose_synchronous_memit_factors(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        requests: Sequence[EditRequest],
        hparams: Any,
        contexts: ContextManifest,
        direct_z: FrozenDirectZ,
        covariance_caches: Sequence[
            CovarianceCacheSpec | Mapping[str, Any]
        ],
        *,
        model_id: str,
        solver: MemitSystemSolver | None = None,
        residual_denominator: int | None = None,
        frozen_target_lineage: FrozenTargetLineage | None = None,
    ) -> MemitFactorProposal:
        """Measure all layer factors against one unchanged model snapshot.

        The default path retains the original exact-source binding used by
        MV-0/MV-1.  MV-2 may explicitly present a canonical
        :class:`FrozenTargetLineage` to reuse the *unchanged* W0 direct-z bytes
        at one verified descendant.  No implicit rebinding is permitted.
        """

        requests = tuple(requests)
        if not requests:
            raise ContractError("at least one request is required")
        _require_eval_model(model)
        bindings = self.load()
        layers = tuple(int(layer) for layer in hparams.layers)
        if not layers:
            raise ContractError("MEMIT layers must not be empty")
        if len(layers) != len(set(layers)):
            raise ContractError("MEMIT layers must be unique")
        denominator = len(layers) if residual_denominator is None else residual_denominator
        if (
            isinstance(denominator, bool)
            or not isinstance(denominator, int)
            or denominator <= 0
        ):
            raise ContractError("residual_denominator must be a positive integer")
        selected_solver = NativeMemitSolver() if solver is None else solver

        (
            covariance_manifest,
            cache_path_by_layer,
            layer_names,
        ) = _preflight_covariance_caches(
            model,
            hparams,
            layers,
            covariance_caches,
        )
        direct_z_manifest = ProvenanceManifest(
            label="frozen direct-z artifact",
            files=(direct_z.artifact,),
        )
        direct_z_manifest.assert_current()

        rewrite_template = str(hparams.rewrite_module_tmp)
        weight_names = tuple(
            f"{rewrite_template.format(layer)}.weight"
            for layer in layers
        )
        source_snapshot = capture_snapshot(
            model,
            model_id=model_id,
            requests=requests,
            context_id=contexts.manifest_id,
            hparams=hparams,
            weight_names=weight_names,
            provenance_ids=(bindings.provenance.manifest_id,),
        )
        if frozen_target_lineage is None:
            if direct_z.source_snapshot_id != source_snapshot.snapshot_id:
                raise ContractError(
                    "direct-z was not generated for this source/model entry snapshot"
                )
            if (
                direct_z.source_state_id != source_snapshot.state_id
                or direct_z.z_layer != layers[-1]
                or direct_z.request_ids != source_snapshot.request_ids
            ):
                raise ContractError(
                    "direct-z metadata does not match the synchronous request"
                )
        else:
            if type(frozen_target_lineage) is not FrozenTargetLineage:
                raise ContractError("frozen-target lineage runtime type is invalid")
            if (
                direct_z.z_layer != layers[-1]
                or direct_z.request_ids != source_snapshot.request_ids
            ):
                raise ContractError(
                    "lineage target metadata does not match the synchronous request"
                )
            frozen_target_lineage.assert_authorizes(
                direct_z=direct_z,
                snapshot=source_snapshot,
            )
        snapshot_provenance = {
            bindings.provenance.manifest_id,
            covariance_manifest.manifest_id,
            direct_z.artifact_id,
        }
        if frozen_target_lineage is not None:
            snapshot_provenance.add(frozen_target_lineage.lineage_id)
        snapshot = SnapshotManifest(
            model_id=source_snapshot.model_id,
            context_id=source_snapshot.context_id,
            request_ids=source_snapshot.request_ids,
            hparams_sha256=source_snapshot.hparams_sha256,
            parameters=source_snapshot.parameters,
            provenance_ids=tuple(sorted(snapshot_provenance)),
        )

        raw_requests = [request.to_easyedit() for request in requests]
        z_layer = layers[-1]
        factors = []
        # Proposal construction is inference-only.  Without this guard,
        # upstream representation helpers may retain full autograd graphs
        # until the per-layer solver loop exits, which needlessly amplifies
        # GPU memory in descendant-state refresh diagnostics.
        with (
            torch.no_grad(),
            _EASYEDIT_GLOBAL_LOCK,
            _preserve_model_runtime_state(model),
        ):
            assert_snapshot_current(model, snapshot)
            current_z = bindings.compute_z.get_module_input_output_at_words(
                model,
                tokenizer,
                z_layer,
                context_templates=[request["prompt"] for request in raw_requests],
                words=[request["subject"] for request in raw_requests],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
                track="out",
            ).T
            # EasyEdit keeps compute_z's output dtype and promotes current_z
            # during subtraction.  Do not downcast a float32 optimized target
            # merely because the model activation is fp16/bf16.
            direct_z_on_device = direct_z.values.to(device=current_z.device)
            common_targets = direct_z_on_device - current_z.to(
                dtype=direct_z_on_device.dtype
            )

            with guard_precomputed_layer_stats(
                bindings,
                tuple(cache_path_by_layer.values()),
            ), _isolated_covariance_cache(
                bindings.memit_main,
                model,
                layer_names,
            ):
                for layer in layers:
                    keys = bindings.compute_ks.compute_ks(
                        model,
                        tokenizer,
                        raw_requests,
                        hparams,
                        layer,
                        contexts.to_easyedit(),
                    ).T
                    if keys.shape[1] % common_targets.shape[1] != 0:
                        raise ContractError(
                            "MEMIT key count is not divisible by direct-z request count"
                        )
                    repeat_factor = keys.shape[1] // common_targets.shape[1]
                    targets = common_targets.repeat_interleave(
                        repeat_factor,
                        dim=1,
                    )
                    covariance = bindings.memit_main.get_cov(
                        model,
                        tokenizer,
                        rewrite_template.format(layer),
                        hparams.mom2_dataset,
                        hparams.mom2_n_samples,
                        hparams.mom2_dtype,
                        force_recompute=False,
                        hparams=hparams,
                    )
                    adjusted_keys = selected_solver.adjusted_keys(
                        covariance.double(),
                        keys.double(),
                        hparams.mom2_update_weight,
                    )
                    residuals = targets.double() / denominator
                    weight_name = f"{rewrite_template.format(layer)}.weight"
                    record = snapshot.parameter(weight_name)
                    factors.append(
                        orient_easyedit_factor(
                            adjusted_keys,
                            residuals,
                            weight_name=weight_name,
                            weight_shape=record.shape,
                            expected_weight_sha256=record.sha256,
                        )
                    )
            assert_snapshot_current(model, snapshot)
        covariance_manifest.assert_current()
        direct_z_manifest.assert_current()
        bindings.provenance.assert_current()
        return MemitFactorProposal(
            snapshot=snapshot,
            factors=tuple(factors),
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name=f"synchronous/{selected_solver.name}",
            residual_denominator=denominator,
        )
