"""Thin server1 adapter for official Llama FT/MEMIT/MEMIT-FE.

No scientific implementation is imported from a local EasyEdit checkout. The
only runtime binding is native precomputed C0, native process-local context,
and technical guards. Checkpoint persistence belongs to the common runner.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import torch

from official.baselines import registry
from official.experiments.prepare import digest, file_sha
from official.runners.server1.assets import verify_manifest


METHODS = ("FT", "MEMIT", "MEMIT_FE", "ALPHAEDIT", "SPHERE")
HISTORY_METHODS = ("ALPHAEDIT", "SPHERE")
CONTEXT_SCHEMA = "official-server1-native-context-v1"


class NativeBindingError(RuntimeError):
    def __init__(self, code, detail=None):
        self.code, self.detail = code, detail
        super().__init__(code + (": " + str(detail) if detail is not None else ""))


def require(condition, code, detail=None):
    if not condition:
        raise NativeBindingError(code, detail)


def tensor_sha(tensor):
    """One selected FP32 byte stream; no weight snapshot is written to disk."""
    value = tensor.detach().to("cpu").contiguous().numpy()
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _parameter_metadata(model):
    return {name: dict(pointer=value.data_ptr(), object=id(value), version=value._version,
                       dtype=str(value.dtype), device=str(value.device), shape=list(value.shape))
            for name, value in model.named_parameters()}


def _json_contexts(value):
    if value is None:
        return None
    require(type(value) is list and all(type(group) is list and all(type(text) is str for text in group)
                                     for group in value), "NATIVE_CONTEXT_SCHEMA")
    return deepcopy(value)


class NativeEngine:
    """One independent cold trajectory, one method, unchanged registry apply.

    ``selected_weights`` contains the actual mutable FP32 Parameters. The
    second native return value is original weight copies, never history H.
    FT/MEMIT/MEMIT-FE therefore always have ``cache_c == {}``.
    """
    def __init__(self, model, tokenizer, method, asset_manifest, *, source_verified=False):
        require(method in METHODS, "SERVER1_METHOD_SCOPE", method)
        require(type(source_verified) is bool, "NATIVE_SOURCE_VERIFICATION_FLAG_TYPE")
        # This affirmative flag is supplied only after the parent runner's
        # immutable archive lock has verified ALL frozen source members.
        self.source_verified = source_verified
        self.asset_verification = verify_manifest(
            asset_manifest, verify_preparation_source=not source_verified)
        require(asset_manifest["assignment"]["model"] == "llama3"
                and method in asset_manifest["assignment"]["methods"], "ASSET_ASSIGNMENT_SCOPE")
        require(asset_manifest["model"]["payload_verification"] == "FRESH_FULL_SHA256",
                "MODEL_PAYLOAD_NOT_SHA_VERIFIED")
        self.model, self.tokenizer, self.method = model, tokenizer, method
        self.asset_manifest = asset_manifest
        self.module, _, self._native_apply = registry.implementation(method, "llama3")
        self.native_source_sha256 = file_sha(self.module.__file__)
        self.hparams = registry.hparams(method, "llama3")
        self.call_options = registry.call_options(method, "llama3")
        require(self.call_options.get("copy", False) is False, "NATIVE_COPY_DISABLED")
        if method == "MEMIT":
            require(self.call_options.get("beta_hse") == 0 and self.call_options.get("save_weights") is False,
                    "LLAMA_MEMIT_NATIVE_OPTIONS")
        self.cache_c = {}
        self.successful_calls = 0
        self._closed = False
        self._saved_layer_stats = None
        self._layer_stats_sentinel = None
        self._reset_cold_globals()
        parameters = dict(model.named_parameters())
        if method == "FT":
            # Exact native FT selection: substring of the rewrite template.
            self.selected_weights = {name: value for name, value in parameters.items()
                                     if any(self.hparams.rewrite_module_tmp.format(layer) in name
                                            for layer in self.hparams.layers)}
        else:
            names = [self.hparams.rewrite_module_tmp.format(layer) + ".weight" for layer in self.hparams.layers]
            require(all(name in parameters for name in names), "NATIVE_SELECTED_WEIGHT_MISSING", names)
            self.selected_weights = {name: parameters[name] for name in names}
        require(bool(self.selected_weights), "NATIVE_SELECTED_WEIGHTS_EMPTY")
        hidden = asset_manifest["model"]["identity"]["hidden"]
        width = asset_manifest["model"]["identity"]["intermediate"]
        for name, value in self.selected_weights.items():
            require(value.dtype == torch.float32 and list(value.shape) == [hidden, width],
                    "NATIVE_SELECTED_FP32_SHAPE", name)
        self._freeze()
        if method != "FT":
            self._bind_C0()
        if method in HISTORY_METHODS:
            self._bind_projector()
        self._last_hashes = self._selected_hashes()
        self._last_metadata = _parameter_metadata(model)

    def _reset_cold_globals(self):
        if self.method != "FT":
            require(hasattr(self.module, "COV_CACHE") and hasattr(self.module, "CONTEXT_TEMPLATES_CACHE"),
                    "NATIVE_REQUIRED_GLOBALS")
            self.module.COV_CACHE = {}
            self.module.CONTEXT_TEMPLATES_CACHE = None
        if self.method == "MEMIT":
            require(hasattr(self.module, "GLOBAL_EDIT_COUNT"), "NATIVE_EDIT_COUNTER_MISSING")
            self.module.GLOBAL_EDIT_COUNT = 0

    def _freeze(self):
        self.model.zero_grad(set_to_none=True)
        for value in self.model.parameters():
            value.requires_grad_(False)
        self.model.eval()

    def _bind_projector(self):
        row = self.asset_manifest["projector"]
        require(row["physical_layers"] == self.hparams.layers == [4, 5, 6, 7, 8]
                and row["threshold"] == self.hparams.nullspace_threshold == .02,
                "PROJECTOR_LAYER_THRESHOLD_BINDING")
        path = Path(row["path"])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == row["bytes"]
                and file_sha(path) == row["sha256"], "PROJECTOR_IMMUTABLE_SHA")
        packed = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        width = self.asset_manifest["model"]["identity"]["intermediate"]
        require(torch.is_tensor(packed) and packed.dtype == torch.float32
                and list(packed.shape) == [5, width, width], "PROJECTOR_SHAPE_DTYPE")
        for layer in packed:
            require(bool(torch.isfinite(layer).all()), "PROJECTOR_NONFINITE")
        self.hparams.P_loc = str(path)
        self.module.P, self.module.P_loaded = packed, True
        # Exact native cold state, initialized before the required W0 checkpoint.
        self.module.cache_c = torch.zeros_like(packed, device="cpu")
        self.module.cache_c_new = True
        self._refresh_history()

    def _refresh_history(self):
        if self.method in HISTORY_METHODS:
            self.cache_c = {str(layer): self.module.cache_c[i]
                            for i, layer in enumerate(self.hparams.layers)}

    def history_identity(self):
        return {key: dict(sha256=tensor_sha(value), shape=list(value.shape), dtype=str(value.dtype))
                for key, value in self.cache_c.items()}

    def restore_history(self, history):
        if self.method not in HISTORY_METHODS:
            require(not history, "NON_HISTORY_CHECKPOINT_REJECTED")
            return
        require(set(history) == set(self.cache_c), "RESTORE_NATIVE_HISTORY_LAYERS")
        for key, target in self.cache_c.items():
            value = history[key]
            require(value.dtype == target.dtype == torch.float32 and value.shape == target.shape
                    and bool(torch.isfinite(value).all()), "RESTORE_HISTORY_SHAPE_DTYPE_FINITE")
            target.copy_(value.to(target.device))

    def _bind_C0(self):
        name = self.model.config._name_or_path.replace("/", "_")
        require(type(name) is str and bool(name), "NATIVE_MODEL_CACHE_IDENTITY")
        widths = {value.shape[1] for value in self.selected_weights.values()}
        require(len(widths) == 1, "NATIVE_INPUT_WIDTHS")
        width = next(iter(widths))
        counts = set()
        for layer in self.hparams.layers:
            row = self.asset_manifest["C0"].get(str(layer))
            require(isinstance(row, dict), "C0_LAYER_BINDING_MISSING", layer)
            module = self.hparams.rewrite_module_tmp.format(layer)
            require(row["module"] == module and row["layer"] == layer, "C0_LAYER_MODULE_MISMATCH", layer)
            validation = row.get("tensor_validation", {})
            require(validation.get("shape") == [width, width] and validation.get("dtype") == "float32"
                    and validation.get("finite") is True
                    and validation.get("stored_value") == "UNCENTERED_SECOND_MOMENT_SUM"
                    and validation.get("sample_documents") == self.hparams.mom2_n_samples,
                    "C0_VALIDATED_NATIVE_SUM_REQUIRED", layer)
            path = Path(row["path"])
            require(path.is_file() and path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"],
                    "C0_SOURCE_SHA_CHANGED", layer)
            with np.load(path, allow_pickle=False) as archive:
                count = archive["mom2.count"]
                require(count.ndim == 0 and count.dtype.kind in "iu" and int(count) > 0,
                        "C0_NATIVE_MASKED_COUNT", layer)
                count = int(count)
                require(count == validation["masked_token_vector_count"]
                        and int(archive["sample_size"]) == self.hparams.mom2_n_samples,
                        "C0_LOCKED_COUNT_MISMATCH", layer)
                total = archive["mom2.mom2"]
                require(total.dtype == np.float32 and total.shape == (width, width),
                        "C0_NATIVE_PAYLOAD", layer)
                # Identical native SecondMoment.moment() CPU FP32 operation.
                moment = torch.from_numpy(total) / count
            require(moment.dtype == torch.float32 and bool(torch.isfinite(moment).all()), "C0_MOMENT_NONFINITE", layer)
            self.module.COV_CACHE[(name, module)] = moment
            counts.add(count)
        require(len(counts) == 1, "C0_SHARED_MASKED_COUNT")
        self._saved_layer_stats = self.module.layer_stats

        def fail_closed_layer_stats(*args, **kwargs):
            raise NativeBindingError("NATIVE_C0_CACHE_MISS_RECOMPUTE_FORBIDDEN")
        self._layer_stats_sentinel = fail_closed_layer_stats
        # This authorized cache-miss sentinel changes no get_cov/solver math.
        self.module.layer_stats = self._layer_stats_sentinel

    def contexts(self):
        result = dict(schema=CONTEXT_SCHEMA, method=self.method, module=self.module.__name__,
                      successful_calls=self.successful_calls,
                      CONTEXT_TEMPLATES_CACHE=_json_contexts(
                          getattr(self.module, "CONTEXT_TEMPLATES_CACHE", None)))
        if self.method == "MEMIT":
            result["GLOBAL_EDIT_COUNT"] = self.module.GLOBAL_EDIT_COUNT
        result["identity_sha256"] = digest(result)
        return result

    def restore_contexts(self, value):
        require(type(value) is dict, "RESTORE_CONTEXT_OBJECT")
        unsigned = dict(value)
        expected = unsigned.pop("identity_sha256", None)
        require(expected == digest(unsigned), "RESTORE_CONTEXT_DIGEST")
        require(value.get("schema") == CONTEXT_SCHEMA and value.get("method") == self.method
                and value.get("module") == self.module.__name__, "RESTORE_CONTEXT_SCOPE")
        keys = {"schema", "method", "module", "successful_calls", "CONTEXT_TEMPLATES_CACHE", "identity_sha256"}
        if self.method == "MEMIT":
            keys.add("GLOBAL_EDIT_COUNT")
        require(set(value) == keys, "RESTORE_CONTEXT_KEYS")
        calls = value.get("successful_calls")
        require(type(calls) is int and 0 <= calls <= 20, "RESTORE_CONTEXT_CURSOR")
        context = _json_contexts(value.get("CONTEXT_TEMPLATES_CACHE"))
        if self.method == "FT":
            require(context is None and "GLOBAL_EDIT_COUNT" not in value, "FT_HAS_NO_NATIVE_CONTEXT")
        else:
            require(calls == 0 or context is not None, "RESTORE_COMPLETED_NATIVE_CONTEXT_MISSING")
        if self.method == "MEMIT":
            count = value.get("GLOBAL_EDIT_COUNT")
            require(type(count) is int and count == calls, "RESTORE_NATIVE_EDIT_COUNTER")
        # Validate the whole context member before changing native module state.
        if self.method != "FT":
            self.module.CONTEXT_TEMPLATES_CACHE = context
        if self.method == "MEMIT":
            self.module.GLOBAL_EDIT_COUNT = count
        self.successful_calls = calls
        self._freeze()
        # The caller restores the checkpoint's selected FP32 weights first.
        self._last_hashes = self._selected_hashes()
        self._last_metadata = _parameter_metadata(self.model)

    def _selected_hashes(self):
        return {name: tensor_sha(value) for name, value in self.selected_weights.items()}

    def state_identity(self):
        hashes = self._selected_hashes()
        result = dict(method=self.method, successful_calls=self.successful_calls, cache_c=self.history_identity(),
                      selected_weights={name: dict(sha256=hashes[name], shape=list(value.shape), dtype=str(value.dtype))
                                        for name, value in self.selected_weights.items()},
                      contexts_sha256=self.contexts()["identity_sha256"])
        result["identity_sha256"] = digest(result)
        return result

    def _guard_after(self, before):
        after = _parameter_metadata(self.model)
        require(set(before) == set(after), "NATIVE_PARAMETER_NAMES_CHANGED")
        for name in before:
            old, new = before[name], after[name]
            for field in ("pointer", "object", "dtype", "device", "shape"):
                require(old[field] == new[field], "NATIVE_PARAMETER_IDENTITY_CHANGED", (name, field))
            if name not in self.selected_weights:
                require(old["version"] == new["version"], "NATIVE_NONSELECTED_VERSION_CHANGED", name)
        for name, value in self.selected_weights.items():
            require(bool(torch.isfinite(value).all()), "NATIVE_SELECTED_NONFINITE", name)
        require(not self.model.training and all(not value.requires_grad and value.grad is None
                                               for value in self.model.parameters()), "NATIVE_POST_FREEZE_STATE")
        return after

    def apply(self, records):
        require(not self._closed, "NATIVE_ENGINE_CLOSED")
        require(self.successful_calls < 20, "NATIVE_NO_BATCH21")
        require(type(records) is list and len(records) == 100, "NATIVE_BATCH100_REQUIRED")
        before = _parameter_metadata(self.model)
        # Checkpoint restore uses restore_contexts to update this binding first.
        require(before == self._last_metadata, "NATIVE_ENTRY_STATE_CHANGED")
        original_records = digest(records)
        history_before = self.history_identity()
        requests = registry.requests(records, self.method, "llama3")
        request_sha = digest(requests)
        caught = None
        try:
            returned = self._native_apply(self.model, self.tokenizer, requests, self.hparams, **self.call_options)
            require(type(returned) is tuple and len(returned) == 2 and returned[0] is self.model,
                    "NATIVE_SAME_CUMULATIVE_MODEL_REQUIRED")
            # Sphere MEMIT's weights_copy may be nonempty even return_orig=False.
            # It is not edit history and is intentionally not persisted as H.
            del returned
            self._refresh_history()
            require(all(bool(torch.isfinite(value).all()) for value in self.cache_c.values()),
                    "NATIVE_HISTORY_NONFINITE")
        except BaseException as error:
            caught = error
            raise
        finally:
            self._freeze()
            try:
                after = self._guard_after(before)
                require(digest(records) == original_records, "NATIVE_INPUT_RECORDS_MUTATED")
            except BaseException as guard_error:
                if caught is None:
                    raise
                if hasattr(caught, "add_note"):
                    caught.add_note("Native technical guard also failed: " + str(guard_error))
        self.successful_calls += 1
        require(self.method != "MEMIT" or self.module.GLOBAL_EDIT_COUNT == self.successful_calls,
                "NATIVE_EDIT_COUNTER_DRIFT")
        after_hashes = self._selected_hashes()
        result = dict(schema="official-server1-native-apply-v1", method=self.method,
                      batch=self.successful_calls, requests=100, request_sha256=request_sha,
                      entry_selected_sha256=dict(self._last_hashes), post_selected_sha256=after_hashes,
                      native_module=self.module.__name__, call_options=dict(self.call_options),
                      native_source_sha256=self.native_source_sha256,
                      frozen_source_verified_by_caller=self.source_verified,
                      asset_manifest_verification=dict(self.asset_verification),
                      same_model=True, nonselected_identity_version_unchanged=True,
                      selected_FP32_finite=True, native_history=self.method in HISTORY_METHODS,
                      cache_c=self.history_identity(),
                      contexts_sha256=self.contexts()["identity_sha256"],
                      quality_gate=False)
        if self.method in HISTORY_METHODS:
            result["entry_cache_c"] = history_before
        result["identity_sha256"] = digest(result)
        self._last_hashes, self._last_metadata = after_hashes, after
        return result

    def close(self):
        """Remove our sentinel only; never restore scientific weights or context."""
        if self._layer_stats_sentinel is not None and self.module.layer_stats is self._layer_stats_sentinel:
            self.module.layer_stats = self._saved_layer_stats
        self._closed = True
