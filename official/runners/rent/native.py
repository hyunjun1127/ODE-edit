"""Model-generic native-state connector for rent (Llama3, Qwen2.5, GPT-J).

Adapted from server2's GPT-J connector: algorithms, parsers and call options
come only from ``official.baselines.registry``. This file supplies the verified
C0/P assets and the run-owned native state; it never regenerates C0/P and never
implements a replacement fit, solver or history update.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, fields
from importlib import import_module
from pathlib import Path

import torch

from official.baselines import registry
from official.experiments.prepare import file_sha, load_plan


METHODS = ("FT", "MEMIT", "ALPHAEDIT", "ALPHAEDIT_BLUE", "MEMIT_FE", "SPHERE")
HISTORY_METHODS = ("ALPHAEDIT", "ALPHAEDIT_BLUE", "SPHERE")
COVARIANCE_METHODS = ("MEMIT", "MEMIT_FE")
# AlphaEdit/SPHERE read the precomputed P; their stats path is still pinned.
STATS_METHODS = ("MEMIT", "MEMIT_FE", "ALPHAEDIT", "SPHERE")
# Structure checks only; revision, layers and widths come from contract.json.
STRUCTURE = {
    "llama3": dict(model_type="llama", family="llama", n_layer=32, vocab=128256),
    "qwen25": dict(model_type="qwen2", family="qwen2", n_layer=28, vocab=152064),
    "gptj": dict(model_type="gptj", family="gptj", n_layer=28, vocab=50400),
}
_ABSENT = object()


def model_spec(model):
    if model not in STRUCTURE:
        raise ValueError("UNSUPPORTED_RENT_MODEL")
    identity = load_plan()[0]["models"][model]
    return dict(STRUCTURE[model], model=model, model_id=identity["model_id"],
                basename=identity["model_id"].rsplit("/", 1)[1], revision=identity["revision"],
                layers=tuple(identity["layers"]), hidden=identity["hidden"],
                key_width=identity["intermediate"])


def _member(manifest, key):
    try:
        member = manifest["assets"][key]
    except KeyError as error:
        raise ValueError("NATIVE_ASSET_MISSING: " + key) from error
    path = Path(member["path"])
    if not path.is_file() or path.stat().st_size != member["bytes"]:
        raise ValueError("NATIVE_ASSET_STAT_MISMATCH: " + key)
    # The asset manifest binds every large file to a full SHA at preparation;
    # rehashing multi-GiB planes here would only duplicate that read.
    if not member.get("sha256"):
        raise ValueError("NATIVE_ASSET_SHA_NOT_BOUND: " + key)
    return member, path


def selected_weights(model, hparams, method):
    parameters = dict(model.named_parameters())
    if method == "FT":
        # Native FT selects by substring, e.g. GPT-J fc_out weight and bias.
        names = [name for name in parameters for layer in hparams.layers
                 if hparams.rewrite_module_tmp.format(layer) in name]
    else:
        names = [hparams.rewrite_module_tmp.format(layer) + ".weight" for layer in hparams.layers]
    if not names or len(set(names)) != len(names) or any(name not in parameters for name in names):
        raise ValueError("NATIVE_PARAMETER_MAPPING")
    result = {name: parameters[name] for name in names}
    if any(value.dtype != torch.float32 for value in result.values()):
        raise ValueError("NATIVE_FP32_WEIGHTS_REQUIRED")
    return result


def check_model(model, spec, manifest):
    config = model.config
    if getattr(config, "model_type", None) != spec["model_type"]:
        raise ValueError("NATIVE_MODEL_TYPE")
    if spec["model"] == "gptj":
        shape = (config.n_layer, config.n_embd, config.n_inner or 4 * config.n_embd)
    else:
        shape = (config.num_hidden_layers, config.hidden_size, config.intermediate_size)
    if shape != (spec["n_layer"], spec["hidden"], spec["key_width"]) or config.vocab_size != spec["vocab"]:
        raise ValueError("NATIVE_MODEL_STRUCTURE")
    if manifest.get("model_revision") != spec["revision"]:
        raise ValueError("NATIVE_MODEL_REVISION")


def projector_slots(physical, layers):
    """Rows of the immutable source P used by these hparams layers."""
    physical, layers = list(physical), list(layers)
    if any(layer not in physical for layer in layers) or len(set(layers)) != len(layers):
        raise ValueError("NATIVE_PROJECTOR_LAYER_MAPPING")
    return [physical.index(layer) for layer in layers]


def _state_signature(model, selected):
    return {name: (value.data_ptr(), value._version, tuple(value.shape), value.dtype)
            for name, value in model.named_parameters() if name not in selected}


class NativeEngine:
    """One independent cold/native trajectory; native globals are call-local.

    ``contexts()`` initializes the native templates once when cold. Resume must
    call ``restore`` before that method; it installs saved templates without a
    generation call. ``contexts(generate=False)`` is metadata-only.
    """

    def __init__(self, model, tok, method, manifest, hparams):
        if method not in METHODS:
            raise ValueError("UNSUPPORTED_OFFICIAL_METHOD")
        self.model, self.tok, self.method, self.manifest = model, tok, method, manifest
        self.spec = model_spec(manifest["model"])
        check_model(model, self.spec, manifest)
        self.key_width = self.spec["key_width"]
        self.module, _, self.native_apply = registry.implementation(method, self.spec["model"])
        if not self.module.__name__.startswith("official.baselines."):
            raise ValueError("EXTERNAL_ALGORITHM_IMPORT")
        self.hparams = hparams
        # Device and asset paths are deployment overrides, never native fit or
        # solver hyperparameters. The model name below only keys native caches.
        original = asdict(self.hparams)
        allowed = {field.name for field in fields(type(self.hparams))}
        if "device" in allowed:
            self.hparams.device = 0
        if "stats_dir" in allowed and method in STATS_METHODS:
            self.hparams.stats_dir = str(Path(manifest["stats_root"]).parent.parent)
        if "P_loc" in allowed and method in HISTORY_METHODS:
            self.hparams.P_loc = str(_member(manifest, "projector")[1])
        self.hparams_overrides = {key: {"original": original[key], "actual": asdict(self.hparams)[key]}
                                  for key in original if original[key] != asdict(self.hparams)[key]}
        if set(self.hparams_overrides) - {"device", "stats_dir", "P_loc"}:
            raise ValueError("SCIENTIFIC_HPARAM_OVERRIDE")
        self._weights = selected_weights(model, self.hparams, method)
        if method != "FT" and any(tuple(value.shape) != (self.spec["hidden"], self.key_width)
                                   for value in self._weights.values()):
            raise ValueError("NATIVE_REWRITE_WEIGHT_SHAPE")
        self._contexts = {} if method == "FT" else None
        self._history = None
        self._projector = None
        self._covariance = {}
        self.covariance_receipts = {}
        self.slots = []
        self.batch = 0
        if method in HISTORY_METHODS:
            self._load_projector()
            self._history = torch.zeros((len(self.hparams.layers), self.key_width, self.key_width),
                                        dtype=torch.float32, device="cpu")
        if method in COVARIANCE_METHODS:
            self._load_native_covariance()

    def _load_projector(self):
        member, path = _member(self.manifest, "projector")
        physical = list(self.spec["layers"])
        if member.get("slot_layers") != physical or member.get("threshold") != .02:
            raise ValueError("NATIVE_PROJECTOR_PROVENANCE")
        full = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if (not isinstance(full, torch.Tensor) or full.dtype != torch.float32
                or tuple(full.shape) != (len(physical), self.key_width, self.key_width)):
            raise ValueError("NATIVE_PROJECTOR_SHAPE_DTYPE")
        self.slots = projector_slots(physical, self.hparams.layers)
        if self.slots == list(range(len(physical))):
            self._projector = full
        else:
            # BLUE edits the first/last layer only; a strided view keeps those
            # source planes without materializing a modified projector copy.
            stride = self.slots[1] - self.slots[0] if len(self.slots) > 1 else 1
            if self.slots != list(range(self.slots[0], self.slots[-1] + 1, stride)):
                raise ValueError("NATIVE_PROJECTOR_SLOTS_NOT_STRIDED")
            self._projector = full[self.slots[0]:self.slots[-1] + 1:stride]

    def _load_native_covariance(self):
        # Pinned runningstats loader and its FP32 mom2/count reduction, then the
        # native cache is seeded. layer_stats is replaced by a fail-closed
        # sentinel during calls, so a cache miss can never recompute C0.
        runningstats = import_module("official.baselines.easyedit.util.runningstats")
        for layer in self.hparams.layers:
            member, path = _member(self.manifest, f"C0_L{layer}")
            module_name = self.hparams.rewrite_module_tmp.format(layer)
            expected = (Path(self.hparams.stats_dir) / self.spec["basename"] / "wikipedia_stats"
                        / (module_name + "_float32_mom2_100000.npz"))
            if path.resolve() != expected.resolve():
                raise ValueError("NATIVE_C0_CANONICAL_PATH")
            stat = runningstats.CombinedStat(mom2=runningstats.SecondMoment())
            stat.load(path)
            if stat.mom2.count != member.get("count") or not stat.mom2.count > 0:
                raise ValueError("NATIVE_C0_TOKEN_COUNT")
            if stat.mom2.mom2.dtype != torch.float32:
                raise ValueError("NATIVE_C0_RAW_DTYPE")
            covariance = stat.mom2.moment().float().to("cpu")
            if tuple(covariance.shape) != (self.key_width, self.key_width):
                raise ValueError("NATIVE_C0_SHAPE")
            if not bool(torch.isfinite(covariance).all()):
                raise ValueError("NATIVE_C0_NONFINITE")
            self._covariance[(self.spec["basename"], module_name)] = covariance
            self.covariance_receipts[str(layer)] = dict(path=str(path), sha256=member["sha256"],
                count=stat.mom2.count, reduction="official.SecondMoment.moment().float().cpu()",
                source_sha256=file_sha(Path(runningstats.__file__)))

    @contextmanager
    def _activated(self):
        names = ["CONTEXT_TEMPLATES_CACHE", "COV_CACHE", "STATS_DIR", "layer_stats"]
        if self.method in ("ALPHAEDIT", "SPHERE"):
            names += ["P", "P_loaded", "cache_c", "cache_c_new"]
        previous = {name: getattr(self.module, name, _ABSENT) for name in names}
        previous_model_name = self.model.config._name_or_path
        self.model.config._name_or_path = self.spec["basename"]
        if self.method != "FT":
            self.module.CONTEXT_TEMPLATES_CACHE = self._contexts
            self.module.COV_CACHE = self._covariance
            self.module.STATS_DIR = Path(self.manifest["stats_root"]).parent.parent
            if previous["layer_stats"] is not _ABSENT:
                def fail_closed_layer_stats(*args, **kwargs):
                    raise ValueError("NATIVE_C0_CACHE_MISS_RECOMPUTE_FORBIDDEN")
                self.module.layer_stats = fail_closed_layer_stats
        if self.method in ("ALPHAEDIT", "SPHERE"):
            # Explicit run-owned native globals; the pinned initializer would
            # otherwise compute a new P when P_loaded is false.
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
        return dict(method=self.method, batch=self.batch, layers=list(self.hparams.layers),
                    projector_slots=list(self.slots),
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
        native_requests = registry.requests(records, self.method, self.spec["model"])
        options = registry.call_options(self.method, self.spec["model"])
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
                    history = returned[1]
                    if isinstance(history, (tuple, list)) and len(history) == 1:
                        history = history[0]
                    self._history = history
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
                    projector_slots=list(self.slots),
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
            # Copy into the run-owned native H; never synthesize a history.
            for i, layer in enumerate(self.hparams.layers):
                self._history[i].copy_(history[str(layer)])
        self._contexts = deepcopy(templates)
        self.batch = batch
        return dict(method=self.method, batch=batch, context_regenerated=False,
                    native_history_restored_layers=list(history))
