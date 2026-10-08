"""GPT-J native-state connector for the pinned ``official`` distribution.

This file supplies existing assets and checkpoint state; algorithms and parsers
are obtained only from official.baselines.registry. It neither regenerates C0/P
nor implements a replacement fit, solver, or history update.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, fields
from importlib import import_module
from pathlib import Path

import torch

from official.baselines import registry
from official.experiments.prepare import file_sha


METHODS = ("FT", "MEMIT", "ALPHAEDIT", "ALPHAEDIT_BLUE", "MEMIT_FE", "SPHERE")
HISTORY_METHODS = ("ALPHAEDIT", "ALPHAEDIT_BLUE", "SPHERE")
CANONICAL_MODEL_NAME = "gpt-j-6b"
MODEL_REVISION = "47e169305d2e8376be1d31e765533382721b2cc1"
PHYSICAL_LAYERS = (3, 4, 5, 6, 7, 8)
_ABSENT = object()


def _member(manifest, key):
    try:
        member = manifest["assets"][key]
    except KeyError as error:
        raise ValueError("NATIVE_ASSET_MISSING: " + key) from error
    path = Path(member["path"])
    if not path.is_file() or path.stat().st_size != member["bytes"]:
        raise ValueError("NATIVE_ASSET_STAT_MISMATCH: " + key)
    stat = path.stat()
    for field, actual in (("inode", stat.st_ino), ("mtime_ns", stat.st_mtime_ns),
                          ("device", stat.st_dev)):
        if field in member and member[field] != actual:
            raise ValueError("NATIVE_ASSET_STAT_MISMATCH: " + key + ":" + field)
    # The owner asset manifest binds large bytes to a full-SHA receipt/current
    # stat. Rehashing every multi-GiB plane here would duplicate preflight I/O.
    if not member.get("sha256"):
        raise ValueError("NATIVE_ASSET_SHA_NOT_BOUND: " + key)
    return member, path


def selected_weights(model, hparams, method):
    parameters = dict(model.named_parameters())
    if method == "FT":
        # Exact native FT selection includes the fc_out bias as well as weight.
        names = [name for name in parameters for layer in hparams.layers
                 if hparams.rewrite_module_tmp.format(layer) in name]
    else:
        names = [hparams.rewrite_module_tmp.format(layer) + ".weight"
                 for layer in hparams.layers]
    if not names or len(set(names)) != len(names) or any(name not in parameters for name in names):
        raise ValueError("NATIVE_PARAMETER_MAPPING")
    result = {name: parameters[name] for name in names}
    if any(value.dtype != torch.float32 for value in result.values()):
        raise ValueError("NATIVE_FP32_WEIGHTS_REQUIRED")
    return result


def _check_model(model, manifest):
    config = model.config
    if (getattr(config, "model_type", None) != "gptj"
            or getattr(config, "n_layer", None) != 28
            or getattr(config, "n_embd", None) != 4096
            or getattr(config, "vocab_size", None) != 50400
            or (getattr(config, "n_inner", None) or 4 * config.n_embd) != 16384):
        raise ValueError("NATIVE_GPTJ_MODEL_STRUCTURE")
    revision = manifest.get("model_revision", manifest.get("revision"))
    if revision != MODEL_REVISION:
        raise ValueError("NATIVE_GPTJ_MODEL_REVISION")
    return 16384


def _state_signature(model, selected):
    return {name: (value.data_ptr(), value._version, tuple(value.shape), value.dtype)
            for name, value in model.named_parameters() if name not in selected}


class NativeEngine:
    """One independent cold/native trajectory; native globals are call-local.

    ``contexts()`` initializes the native templates once when cold. Resume must
    call ``restore`` before that method; it installs saved templates without a
    generation call. ``contexts(generate=False)`` is metadata-only.
    """

    def __init__(self, model, tok, method, manifest):
        if method not in METHODS:
            raise ValueError("UNSUPPORTED_OFFICIAL_GPTJ_METHOD")
        self.model, self.tok, self.method, self.manifest = model, tok, method, manifest
        self.key_width = _check_model(model, manifest)
        self.module, _, self.native_apply = registry.implementation(method, "gptj")
        if not self.module.__name__.startswith("official.baselines."):
            raise ValueError("EXTERNAL_ALGORITHM_IMPORT")
        self.hparams = registry.hparams(method, "gptj")
        # Device and existing asset paths are deployment overrides, never native
        # fit/solver hyperparameter overrides. Model naming controls the native
        # cache key/path, not which snapshot the caller actually loaded.
        original = asdict(self.hparams)
        allowed = {field.name for field in fields(type(self.hparams))}
        if "device" in allowed:
            self.hparams.device = 0
        if "stats_dir" in allowed and method in ("MEMIT", "MEMIT_FE", "ALPHAEDIT", "SPHERE"):
            self.hparams.stats_dir = str(Path(manifest["stats_root"]).parent.parent)
        if "P_loc" in allowed and method in HISTORY_METHODS:
            self.hparams.P_loc = str(_member(manifest, "projector")[1])
        self.hparams_overrides = {key: {"original": original[key], "actual": asdict(self.hparams)[key]}
                                 for key in original if original[key] != asdict(self.hparams)[key]}
        if set(self.hparams_overrides) - {"device", "stats_dir", "P_loc"}:
            raise ValueError("SCIENTIFIC_HPARAM_OVERRIDE")
        self._weights = selected_weights(model, self.hparams, method)
        if method != "FT" and any(tuple(value.shape) != (4096, self.key_width)
                                   for value in self._weights.values()):
            raise ValueError("NATIVE_GPTJ_FC_OUT_SHAPE")
        self._contexts = {} if method == "FT" else None
        self._history = None
        self._projector = None
        self._covariance = {}
        self._covariance_receipts = {}
        self.batch = 0
        if method in HISTORY_METHODS:
            self._load_projector()
            self._history = torch.zeros((len(self.hparams.layers), self.key_width, self.key_width),
                                        dtype=torch.float32, device="cpu")
        if method in ("MEMIT", "MEMIT_FE"):
            self._load_native_covariance()

    def _load_projector(self):
        member, path = _member(self.manifest, "projector")
        if member.get("slot_layers") != list(PHYSICAL_LAYERS) or member.get("threshold") != .02:
            raise ValueError("NATIVE_PROJECTOR_PROVENANCE")
        full = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if (not isinstance(full, torch.Tensor) or full.dtype != torch.float32
                or tuple(full.shape) != (6, self.key_width, self.key_width)):
            raise ValueError("NATIVE_PROJECTOR_SHAPE_DTYPE")
        if self.method == "ALPHAEDIT_BLUE":
            if self.hparams.layers != [3, 8]:
                raise ValueError("BLUE_LAYER_MAPPING")
            # A strided view keeps physical slots 0 and 5 without materializing
            # or changing the six-plane immutable source projector.
            self._projector = full[0:6:5]
            self.projector_slots = [0, 5]
        else:
            if self.hparams.layers != list(PHYSICAL_LAYERS):
                raise ValueError("NATIVE_SIX_LAYER_MAPPING")
            self._projector = full
            self.projector_slots = list(range(6))

    def _load_native_covariance(self):
        # Use the pinned runningstats loader and its FP32 mom2/count reduction,
        # then seed the native cache. This fail-closed asset connector never
        # invokes layer_stats, a dataset collector, or automatic C0 generation.
        runningstats = import_module("official.baselines.easyedit.util.runningstats")
        canonical = CANONICAL_MODEL_NAME.replace("/", "_")
        for layer in self.hparams.layers:
            member, path = _member(self.manifest, f"C0_L{layer}")
            module_name = self.hparams.rewrite_module_tmp.format(layer)
            expected = (Path(self.hparams.stats_dir) / CANONICAL_MODEL_NAME / "wikipedia_stats"
                        / (module_name + "_float32_mom2_100000.npz"))
            if path.resolve() != expected.resolve():
                raise ValueError("NATIVE_C0_CANONICAL_PATH")
            stat = runningstats.CombinedStat(mom2=runningstats.SecondMoment())
            stat.load(path)
            if stat.mom2.count != member.get("count") or stat.mom2.count != 54924275:
                raise ValueError("NATIVE_C0_TOKEN_COUNT")
            if stat.mom2.mom2.dtype != torch.float32:
                raise ValueError("NATIVE_C0_RAW_DTYPE")
            covariance = stat.mom2.moment().float().to("cpu")
            if tuple(covariance.shape) != (self.key_width, self.key_width):
                raise ValueError("NATIVE_C0_SHAPE")
            if not bool(torch.isfinite(covariance).all()):
                raise ValueError("NATIVE_C0_NONFINITE")
            self._covariance[(canonical, module_name)] = covariance
            self._covariance_receipts[str(layer)] = dict(path=str(path), sha256=member["sha256"],
                count=stat.mom2.count, reduction="official.SecondMoment.moment().float().cpu()",
                source_sha256=file_sha(Path(runningstats.__file__)))

    @contextmanager
    def _activated(self):
        names = ["CONTEXT_TEMPLATES_CACHE", "COV_CACHE", "STATS_DIR"]
        if self.method in ("ALPHAEDIT", "SPHERE"):
            names += ["P", "P_loaded", "cache_c", "cache_c_new"]
        previous = {name: getattr(self.module, name, _ABSENT) for name in names}
        previous_model_name = self.model.config._name_or_path
        self.model.config._name_or_path = CANONICAL_MODEL_NAME
        if self.method != "FT":
            self.module.CONTEXT_TEMPLATES_CACHE = self._contexts
            self.module.COV_CACHE = self._covariance
            self.module.STATS_DIR = Path(self.manifest["stats_root"]).parent.parent
        if self.method in ("ALPHAEDIT", "SPHERE"):
            # Both globals are explicit native run state. This also avoids the
            # pinned initializer's missing GPT-J branch without a solver fork.
            self.module.P, self.module.P_loaded = self._projector, True
            self.module.cache_c, self.module.cache_c_new = self._history, True
        try:
            yield
        finally:
            if self.method != "FT":
                self._contexts = self.module.CONTEXT_TEMPLATES_CACHE
                self._covariance = self.module.COV_CACHE
            if self.method in ("ALPHAEDIT", "SPHERE"):
                self._history = self.module.cache_c
            for name, old in previous.items():
                if old is _ABSENT:
                    if hasattr(self.module, name):
                        delattr(self.module, name)
                else:
                    setattr(self.module, name, old)
            self.model.config._name_or_path = previous_model_name

    def weights(self):
        return dict(self._weights)

    def history(self):
        if self._history is None:
            return {}
        if (self._history.dtype != torch.float32 or self._history.device.type != "cpu"
                or tuple(self._history.shape) != (len(self.hparams.layers), self.key_width, self.key_width)):
            raise ValueError("NATIVE_HISTORY_SHAPE_DTYPE")
        return {str(layer): self._history[i] for i, layer in enumerate(self.hparams.layers)}

    def contexts(self, generate=True):
        if self.method != "FT" and self._contexts is None and generate:
            with self._activated():
                self.module.get_context_templates(self.model, self.tok)
        return {"native_templates": deepcopy(self._contexts), "method": self.method}

    def state_identity(self):
        return dict(method=self.method, batch=self.batch,
                    layers=list(self.hparams.layers), projector_slots=getattr(self, "projector_slots", []),
                    weights={key: dict(shape=list(value.shape), dtype=str(value.dtype),
                                       pointer=value.data_ptr(), version=value._version)
                             for key, value in self._weights.items()},
                    history={key: dict(shape=list(value.shape), dtype=str(value.dtype),
                                       pointer=value.data_ptr(), version=value._version)
                             for key, value in self.history().items()},
                    contexts=self.contexts(generate=False))

    def apply(self, records, batch):
        if type(batch) is not int or batch != self.batch + 1 or not 1 <= batch <= 20:
            raise ValueError("NONCONTIGUOUS_NATIVE_BATCH")
        if len(records) != 100:
            raise ValueError("OFFICIAL_LOGICAL_BATCH_100_REQUIRED")
        native_requests = registry.requests(records, self.method, "gptj")
        options = registry.call_options(self.method, "gptj")
        if self.method == "ALPHAEDIT_BLUE":
            options.update(cache_c=self._history, P=self._projector)
        elif self.method != "FT":
            options["cache_template"] = None
        nonselected_before = _state_signature(self.model, self._weights)
        grad_before = {name: value.requires_grad for name, value in self.model.named_parameters()}
        training_before = {name: value.training for name, value in self.model.named_modules()}
        with self._activated():
            try:
                with torch.enable_grad():
                    returned = self.native_apply(self.model, self.tok, native_requests, self.hparams, **options)
                if not isinstance(returned, tuple) or len(returned) != 2 or returned[0] is not self.model:
                    raise ValueError("NATIVE_RETURN_SCHEMA_OR_MODEL_REPLACEMENT")
                if self.method == "ALPHAEDIT_BLUE":
                    self._history = returned[1]
            finally:
                for name, value in self.model.named_parameters():
                    value.requires_grad_(grad_before[name])
                for name, value in self.model.named_modules():
                    value.training = training_before[name]
        if _state_signature(self.model, self._weights) != nonselected_before:
            raise ValueError("NATIVE_NONSELECTED_PARAMETER_MUTATION")
        if any(not bool(torch.isfinite(value).all()) for value in self._weights.values()):
            raise ValueError("NATIVE_NONFINITE_WEIGHT")
        if any(not bool(torch.isfinite(value).all()) for value in self.history().values()):
            raise ValueError("NATIVE_NONFINITE_HISTORY")
        self.batch = batch
        return dict(method=self.method, batch=batch, requests=len(records),
                    native_source=str(Path(self.module.__file__).resolve()),
                    native_source_sha256=file_sha(Path(self.module.__file__)),
                    native_call_options={key: value for key, value in options.items()
                                         if key not in ("cache_c", "P")},
                    selected_parameters=list(self._weights), history_layers=list(self.history()),
                    history_appends_expected=len(self.hparams.layers) if self.method in HISTORY_METHODS else 0,
                    projector_slots=getattr(self, "projector_slots", []),
                    native_apply_status="COMPLETED_REQUIRES_SEPARATE_RESUME_PARITY")

    def restore(self, snapshot):
        if snapshot.get("method") != self.method or type(snapshot.get("batch")) is not int:
            raise ValueError("NATIVE_CHECKPOINT_METHOD_OR_BATCH")
        batch = snapshot["batch"]
        if not 0 <= batch <= 20 or set(snapshot["weights"]) != set(self._weights):
            raise ValueError("NATIVE_CHECKPOINT_WEIGHT_MAPPING")
        history = snapshot["cache_c"]
        expected_history = {str(layer) for layer in self.hparams.layers} if self.method in HISTORY_METHODS else set()
        if set(history) != expected_history:
            raise ValueError("NATIVE_CHECKPOINT_HISTORY_MAPPING")
        contexts = snapshot["contexts"]
        if contexts.get("method") != self.method:
            raise ValueError("NATIVE_CHECKPOINT_CONTEXT_METHOD")
        templates = contexts.get("native_templates")
        if self.method != "FT" and templates is None:
            raise ValueError("NATIVE_CHECKPOINT_CONTEXT_NOT_INITIALIZED")
        for name, destination in self._weights.items():
            value = snapshot["weights"][name]
            if (value.dtype != torch.float32 or value.shape != destination.shape
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("NATIVE_CHECKPOINT_WEIGHT_SHAPE_DTYPE_FINITE")
        for value in history.values():
            if (value.dtype != torch.float32 or tuple(value.shape) != (self.key_width, self.key_width)
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("NATIVE_CHECKPOINT_HISTORY_SHAPE_DTYPE_FINITE")
        with torch.no_grad():
            for name, destination in self._weights.items():
                destination.copy_(snapshot["weights"][name].to(destination.device))
        if history:
            # Copy into the already allocated run-owned native H; do not
            # synthesize a new history or use observations to reconstruct it.
            for i, layer in enumerate(self.hparams.layers):
                self._history[i].copy_(history[str(layer)])
        self._contexts = deepcopy(templates)
        self.batch = batch
        return dict(method=self.method, batch=batch, context_regenerated=False,
                    native_history_restored_layers=list(history))
