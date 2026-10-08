# Provenance: server3 readonly source at fcca2007; server4 Qwen transfer binding.
"""Server 3 binding for the unmodified official native baseline implementations.

The native EasyEdit AlphaEdit/SPHERE implementations keep P and cache_c in
module globals; BLUE passes them explicitly.  This adapter makes those states
belong to one cold run and translates them to the common per-layer checkpoint
schema.  EasyEdit on this host supplies *assets only*, never algorithm imports.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from official.baselines import registry
from official.experiments.checkpoint import rng_restore
from official.experiments.prepare import file_sha


HISTORY_METHODS = frozenset(("ALPHAEDIT", "ALPHAEDIT_BLUE", "SPHERE"))
COVARIANCE_METHODS = frozenset(("MEMIT", "MEMIT_FE"))


def _asset_file(info, kind):
    if not isinstance(info, Mapping) or not info.get("path") or not info.get("sha256"):
        raise ValueError(f"{kind}_PROVENANCE_REQUIRED")
    path = Path(info["path"])
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{kind}_REGULAR_FILE_REQUIRED: {path}")
    if file_sha(path) != info["sha256"]:
        raise ValueError(f"{kind}_SHA_MISMATCH: {path}")
    return path


class NativeState:
    """One native method/model trajectory; construct anew for every cold run.

    ``assets`` uses the server readiness manifest's ``assets`` shape:
    ``projector={path,sha256,shape,dtype,layer_mapping}`` and
    ``covariance={physical_layer: {path,sha256,shape,dtype,npz_key}}``.
    The verified C0 is seeded into the *native* COV_CACHE before editing, so
    layer_stats cannot silently generate a replacement Wikipedia covariance.
    """

    def __init__(self, method, model, hparams, assets, *, require_cuda=True):
        if method not in ("FT", "MEMIT", "ALPHAEDIT", "ALPHAEDIT_BLUE", "MEMIT_FE", "SPHERE"):
            raise ValueError("UNSUPPORTED_NATIVE_METHOD")
        if not isinstance(assets, Mapping):
            raise ValueError("ASSET_MANIFEST_REQUIRED")
        self.method = method
        self.model = model
        self.hparams = hparams
        self.layers = tuple(int(layer) for layer in hparams.layers)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("INVALID_PHYSICAL_LAYERS")
        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if require_cuda and visible != 1:
            raise ValueError(f"ONE_VISIBLE_SLURM_GPU_REQUIRED: {visible}")
        nominal_device = getattr(hparams, "device", None)
        # Published profiles include physical host ordinals (FT=1, MEMIT=5).
        # A single Slurm GPU is always exposed as logical cuda:0.  This only
        # binds placement; the sealed hyperparameter JSON is never modified.
        if hasattr(hparams, "device"):
            hparams.device = 0
        self.placement = dict(nominal_device=nominal_device, effective_device=0,
                              visible_gpu_count=visible, runtime_override=nominal_device != 0)
        self.module, _, self.native_apply = registry.implementation(method, "qwen25")
        self._names = self._resolve_editable_parameters()
        if require_cuda and any(self.model.get_parameter(name).device != torch.device("cuda:0")
                                for name in self._names):
            raise ValueError("EDITABLE_WEIGHTS_NOT_ON_VISIBLE_GPU")
        self._width = self._input_width()
        self._projector = None
        self._cache = None

        # Cold-run globals must never inherit a prior chain's contexts or H.
        if hasattr(self.module, "CONTEXT_TEMPLATES_CACHE"):
            self.module.CONTEXT_TEMPLATES_CACHE = None
        if hasattr(self.module, "COV_CACHE"):
            self.module.COV_CACHE.clear()
        if method in ("ALPHAEDIT", "SPHERE"):
            self.module.P = None
            self.module.P_loaded = False
            self.module.cache_c = None
            self.module.cache_c_new = False

        if method in HISTORY_METHODS:
            self._load_projector(assets.get("projector"))
            self.restore_cache({})
        if method in COVARIANCE_METHODS:
            self._seed_covariance(assets.get("covariance"))

    def _resolve_editable_parameters(self):
        available = dict(self.model.named_parameters())
        result = []
        for layer in self.layers:
            stem = self.hparams.rewrite_module_tmp.format(layer)
            if self.method == "FT":
                matches = [name for name in available if stem in name]
                if len(matches) != 1 or matches[0] != stem:
                    raise ValueError(f"FT_WEIGHT_MAPPING_AMBIGUOUS: {stem}")
                name = stem
            else:
                name = stem + ".weight"
                if name not in available:
                    raise ValueError(f"EDITABLE_WEIGHT_MISSING: {name}")
            if available[name].dtype != torch.float32:
                raise ValueError(f"EDITABLE_WEIGHT_NOT_FP32: {name}")
            result.append(name)
        return tuple(result)

    def _input_width(self):
        widths = {int(self.model.get_parameter(name).shape[1]) for name in self._names}
        if len(widths) != 1:
            raise ValueError("EDITABLE_INPUT_WIDTH_MISMATCH")
        return widths.pop()

    def editable_parameter_names(self):
        return self._names

    def _load_projector(self, info):
        path = _asset_file(info, "PROJECTOR")
        if info.get("dtype") != "float32":
            raise ValueError("PROJECTOR_DTYPE_PROVENANCE_MISMATCH")
        mapping = info.get("layer_mapping")
        if not isinstance(mapping, Mapping):
            raise ValueError("PROJECTOR_LAYER_MAPPING_REQUIRED")
        source_shape = tuple(info.get("shape", ()))
        if len(source_shape) != 3 or source_shape[1:] != (self._width, self._width):
            raise ValueError("PROJECTOR_DECLARED_SHAPE_MISMATCH")
        indices = []
        for layer in self.layers:
            index = mapping.get(str(layer))
            if type(index) is not int or index < 0 or index >= source_shape[0]:
                raise ValueError(f"PROJECTOR_PHYSICAL_LAYER_MISSING: {layer}")
            indices.append(index)
        if len(set(indices)) != len(indices):
            raise ValueError("PROJECTOR_LAYER_MAPPING_DUPLICATE")
        projector = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(projector, torch.Tensor) or projector.dtype != torch.float32:
            raise ValueError("PROJECTOR_TENSOR_NOT_FP32")
        if tuple(projector.shape) != source_shape:
            raise ValueError("PROJECTOR_ACTUAL_SHAPE_MISMATCH")
        # BLUE uses L4/L8 from the same five-layer source P.  The regular
        # strided view avoids an unnecessary multi-GiB selected-P copy.
        if len(indices) == 1:
            selected = projector[indices[0]:indices[0] + 1]
        else:
            stride = indices[1] - indices[0]
            if stride > 0 and indices == list(range(indices[0], indices[0] + stride * len(indices), stride)):
                selected = projector[indices[0]:indices[-1] + 1:stride]
            else:
                selected = projector[indices]
        self._projector = selected
        if self.method in ("ALPHAEDIT", "SPHERE"):
            # Native code tests P_loc first and otherwise computes a new P.
            self.hparams.P_loc = str(path)
            self.module.P = selected
            self.module.P_loaded = True

    def _seed_covariance(self, entries):
        if not isinstance(entries, Mapping):
            raise ValueError("COVARIANCE_LAYER_MANIFEST_REQUIRED")
        model_leaf = self.hparams.model_name.rsplit("/", 1)[-1]
        model_key = self.model.config._name_or_path.replace("/", "_")
        roots = set()
        for layer in self.layers:
            info = entries.get(str(layer))
            path = _asset_file(info, f"COVARIANCE_L{layer}")
            roots.add(path.parents[2])
            layer_name = self.hparams.rewrite_module_tmp.format(layer)
            expected = (path.parents[2] / model_leaf / f"{self.hparams.mom2_dataset}_stats" /
                        f"{layer_name}_{self.hparams.mom2_dtype}_mom2_{self.hparams.mom2_n_samples}.npz")
            if path != expected or info.get("npz_key") != "mom2.mom2":
                raise ValueError(f"COVARIANCE_NATIVE_PATH_MISMATCH: {layer}")
            if info.get("dtype") != "float32" or tuple(info.get("shape", ())) != (self._width, self._width):
                raise ValueError(f"COVARIANCE_DECLARED_SHAPE_MISMATCH: {layer}")
            with np.load(path, allow_pickle=False) as archive:
                if int(archive["sample_size"]) != self.hparams.mom2_n_samples:
                    raise ValueError(f"COVARIANCE_SAMPLE_SIZE_MISMATCH: {layer}")
                count = int(archive["mom2.count"])
                if count <= 0:
                    raise ValueError(f"COVARIANCE_COUNT_INVALID: {layer}")
                summed = archive["mom2.mom2"]
                if summed.dtype != np.float32 or summed.shape != (self._width, self._width):
                    raise ValueError(f"COVARIANCE_ACTUAL_SHAPE_MISMATCH: {layer}")
                # Native SecondMoment.moment() is this same sum/count division.
                covariance = torch.from_numpy(summed).div(count)
            self.module.COV_CACHE[(model_key, layer_name)] = covariance
        if len(roots) != 1:
            raise ValueError("COVARIANCE_ROOT_MISMATCH")
        self.hparams.stats_dir = str(roots.pop())

    def _check_cache(self, cache):
        if not isinstance(cache, torch.Tensor) or cache.dtype != torch.float32 or cache.device.type != "cpu":
            raise ValueError("NATIVE_HISTORY_CPU_FP32_REQUIRED")
        if tuple(cache.shape) != (len(self.layers), self._width, self._width):
            raise ValueError("NATIVE_HISTORY_SHAPE_MISMATCH")
        return cache

    def restore_cache(self, cache_c):
        """Accept exactly the checkpoint's physical-layer keyed cache schema."""
        if not isinstance(cache_c, Mapping):
            raise ValueError("NATIVE_HISTORY_DICT_REQUIRED")
        if self.method not in HISTORY_METHODS:
            if cache_c:
                raise ValueError("HISTORY_FOR_STATELESS_METHOD")
            return
        if not cache_c:
            cache = torch.zeros((len(self.layers), self._width, self._width), dtype=torch.float32)
        else:
            expected = {str(layer) for layer in self.layers}
            if set(cache_c) != expected:
                raise ValueError("NATIVE_HISTORY_LAYER_SET_MISMATCH")
            ordered = []
            for layer in self.layers:
                tensor = cache_c[str(layer)]
                if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or
                        tensor.device.type != "cpu" or tuple(tensor.shape) != (self._width, self._width)):
                    raise ValueError(f"NATIVE_HISTORY_LAYER_SHAPE_MISMATCH: {layer}")
                ordered.append(tensor)
            cache = torch.stack(ordered)
        self._cache = self._check_cache(cache)
        if self.method in ("ALPHAEDIT", "SPHERE"):
            self.module.cache_c = self._cache
            self.module.cache_c_new = True

    def cache_for_checkpoint(self):
        if self.method not in HISTORY_METHODS:
            return {}
        cache = self.module.cache_c if self.method in ("ALPHAEDIT", "SPHERE") else self._cache
        self._check_cache(cache)
        return {str(layer): cache[i] for i, layer in enumerate(self.layers)}

    def context_snapshot(self):
        return deepcopy(getattr(self.module, "CONTEXT_TEMPLATES_CACHE", None))

    def restore_context(self, contexts):
        if hasattr(self.module, "CONTEXT_TEMPLATES_CACHE"):
            self.module.CONTEXT_TEMPLATES_CACHE = deepcopy(contexts)

    def apply(self, tokenizer, requests):
        if not requests:
            raise ValueError("EMPTY_NATIVE_BATCH")
        options = registry.call_options(self.method, "qwen25")
        if self.method == "ALPHAEDIT_BLUE":
            options.update(cache_c=self._check_cache(self._cache), P=self._projector)
        result = self.native_apply(self.model, tokenizer, requests, self.hparams, **options)
        if not isinstance(result, tuple) or len(result) != 2 or result[0] is not self.model:
            raise ValueError("NATIVE_APPLY_RETURN_SCHEMA")
        if self.method == "ALPHAEDIT_BLUE":
            returned = result[1]
            if isinstance(returned, (tuple, list)) and len(returned) == 1:
                returned = returned[0]
            self._cache = self._check_cache(returned)
        elif self.method in ("ALPHAEDIT", "SPHERE"):
            self._cache = self._check_cache(self.module.cache_c)
        return self.model

    def restore_from_checkpoint(self, payload):
        """Restore checkpointed editable W/native H/context/RNG to this model."""
        if payload.get("method") != self.method:
            raise ValueError("NATIVE_CHECKPOINT_METHOD_MISMATCH")
        weights = payload.get("weights")
        if not isinstance(weights, Mapping) or set(weights) != set(self._names):
            raise ValueError("NATIVE_CHECKPOINT_WEIGHT_SET_MISMATCH")
        for name in self._names:
            saved = weights[name]
            current = self.model.get_parameter(name)
            if (not isinstance(saved, torch.Tensor) or saved.dtype != torch.float32 or
                    tuple(saved.shape) != tuple(current.shape)):
                raise ValueError(f"NATIVE_CHECKPOINT_WEIGHT_SHAPE_MISMATCH: {name}")
        self.restore_cache(payload.get("cache_c", {}))
        self.restore_context(payload.get("contexts"))
        rng_restore(payload["rng"])
        with torch.no_grad():
            for name in self._names:
                current = self.model.get_parameter(name)
                current.copy_(weights[name].to(device=current.device, dtype=current.dtype))
        return payload["evaluation_cursor"]
