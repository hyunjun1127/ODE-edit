"""Bounded BS4x2 named-baseline pilots on the shared, independently reset W0.

Production native equations/Adam/stopping/history remain in read-only source.
The only native compatibility adapter is the decoder Tensor/tuple container
view required by current transformers. No model/optimizer checkpoint is saved.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from contextlib import contextmanager

from .common import LAYERS, digest, member, require, write


FAMILIES = ("MEMIT-H", "BASE_MEMIT", "BASE_ALPHAEDIT")
CONTEXT_SHA = "33cec0eef9ec130f26c2f0e17f7c8be39e93c717ebb47eaa5e9f1c88b263524e"
NATIVE_DEFAULT = "/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/blue-source"
CONFIG_DEFAULT = "/data/janghj/ODE-edit/local/fixed10k-native-baselines/attempt-v1/configs"


class BaselineBlocked(RuntimeError):
    """No silent native substitution is permitted on source/asset mismatch."""


class _Proxy:
    def __init__(self, base, **changes):
        self.base, self.changes = base, changes

    def __getattr__(self, key):
        return self.changes[key] if key in self.changes else getattr(self.base, key)


def _native_container_callback(callback):
    """Adapt only the returned container, never hidden values or gradient path."""
    def wrapped(output, layer):
        import torch
        tensor_output = isinstance(output, torch.Tensor)
        result = callback((output,) if tensor_output else output, layer)
        if tensor_output:
            if not isinstance(result, (tuple, list)) or len(result) != 1:
                raise BaselineBlocked("NATIVE_BLOCK_CONTAINER_CHANGED")
            return result[0]
        return result
    return wrapped


class _TraceView:
    def __init__(self, trace):
        self.trace = trace

    def __getattr__(self, key):
        import torch
        value = getattr(self.trace, key)
        return (value,) if key == "output" and isinstance(value, torch.Tensor) else value


def _trace_dict_compat(original):
    class TraceDictCompat:
        def __init__(self, *args, **kwargs):
            if kwargs.get("edit_output") is not None:
                kwargs["edit_output"] = _native_container_callback(kwargs["edit_output"])
            self.inner = original(*args, **kwargs)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def __getitem__(self, key):
            return _TraceView(self.inner[key])
    return TraceDictCompat


def _tensor_sha(value):
    import torch
    value = value.detach().to("cpu").contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(tuple(value.shape)).encode())
    h.update(memoryview(value.view(torch.uint8).numpy()).cast("B"))
    return h.hexdigest()


def _weights(model):
    return {l: model.model.layers[l].mlp.down_proj.weight for l in LAYERS}


def _weight_hashes(weights):
    return {str(l): _tensor_sha(w) for l, w in weights.items()}


def _restore_w0(weights, w0):
    import torch
    with torch.no_grad():
        for l, weight in weights.items():
            original = w0.get(l, w0.get(f"model.layers.{l}.mlp.down_proj.weight"))
            require(original is not None and original.dtype == torch.float32,
                    "BASELINE_W0_SNAPSHOT_MISSING")
            require(original.shape == weight.shape, "BASELINE_W0_SHAPE")
            weight.copy_(original.to(weight.device))
            require(torch.equal(weight.detach().cpu(), original.cpu()), "BASELINE_W0_RESTORE")


def _rng_capture():
    import numpy as np
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def _rng_restore(state):
    import numpy as np
    import torch
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3]:
        torch.cuda.set_rng_state_all(state[3])


def _rng_hash(state):
    n = state[1]
    return digest({"python": state[0], "numpy": [n[0], n[1].tolist(), *n[2:]],
                   "torch_cpu": _tensor_sha(state[2]),
                   "torch_cuda": [_tensor_sha(v) for v in state[3]]})


def _synchronize():
    import torch
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _source_closure(root):
    """Inventory real imported files, never resolve a virtual name against cwd.

    PyTorch synthetic modules can advertise ``__file__='_ops.py'`` without a
    filesystem-backed spec. Native import runs with cwd=root, so resolving that
    name fabricates a nonexistent native file. Use an absolute __file__ or an
    absolute spec origin; unresolved virtual modules outside native namespaces
    are excluded. Missing/unresolvable required native sources still hard-fail.
    """
    root = Path(root).resolve()
    files = {}
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        raw = inspect.getattr_static(mod, "__file__", None)
        spec = inspect.getattr_static(mod, "__spec__", None)
        origin = inspect.getattr_static(spec, "origin", None) if spec is not None else None
        required = name.split(".")[0] in {"memit", "AlphaEdit", "rome", "util"}
        if isinstance(raw, str) and Path(raw).is_absolute():
            path = Path(raw)
        elif isinstance(origin, str) and Path(origin).is_absolute():
            path = Path(origin)
        else:
            if required:
                raise BaselineBlocked(f"BASELINE_REQUIRED_NATIVE_SOURCE_UNRESOLVED:{name}:{raw!r}")
            continue
        path = path.resolve()
        if required and not path.is_relative_to(root):
            raise BaselineBlocked(f"BASELINE_REQUIRED_NATIVE_SOURCE_OUTSIDE_ROOT:{name}:{path}")
        if path.suffix != ".py":
            continue
        if path.is_relative_to(root):
            if not path.is_file():
                raise FileNotFoundError(f"BASELINE_REAL_SCOPED_IMPORT_MISSING:{name}:{path}")
            files[str(path)] = {"module": name, **member(path)}
    return [files[k] for k in sorted(files)]


@contextmanager
def _import_native(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise BaselineBlocked(f"BASELINE_NATIVE_ROOT_MISSING:{root}")
    for name, mod in list(sys.modules.items()):
        if name.split(".")[0] not in {"memit", "AlphaEdit", "rome", "util"}:
            continue
        raw = getattr(mod, "__file__", None)
        if raw and not Path(raw).resolve().is_relative_to(root):
            raise BaselineBlocked(f"BASELINE_NATIVE_IMPORT_COLLISION:{name}:{raw}")
    old_path, old_cwd = list(sys.path), Path.cwd()
    try:
        sys.path.insert(0, str(root))
        os.chdir(root)  # Read-only native util.globals opens its own globals.yml.
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def _config(config):
    options = dict(config.get("baseline_pilot", config))
    defaults = {
        "native_root": NATIVE_DEFAULT,
        "stats_root": "/data/janghj/EasyEdit/examples/data/stats",
        "projector": "/data/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt",
        "projector_source_layers": list(LAYERS),
        "configs": {
            "BASE_ALPHAEDIT": str(Path(CONFIG_DEFAULT) / "AlphaEdit-native.json"),
            "BASE_MEMIT": str(Path(CONFIG_DEFAULT) / "MEMIT-native.json"),
            "MEMIT-H": str(Path(__file__).resolve().parents[1] / "memit_history_lifelong/hparams.json"),
        },
    }
    for key, value in defaults.items():
        options.setdefault(key, value)
    # Caller may restrict only when exact prior same8 evidence is separately bound.
    options.setdefault("families", list(FAMILIES))
    require(set(options["families"]) <= set(FAMILIES) and
            len(set(options["families"])) == len(options["families"]), "BASELINE_FAMILY_IDENTITY")
    for family in set(FAMILIES) - set(options["families"]):
        require(bool(options.get("reused_same8_receipts", {}).get(family)),
                "BASELINE_PILOT_OMISSION_WITHOUT_EXACT_SAME8_REUSE")
    return options


def _verify_hparams(hp, family):
    expected = {"layers": list(LAYERS), "blue": False, "v_num_grad_steps": 25,
                "v_lr": .1, "v_weight_decay": .5, "clamp_norm_factor": .75,
                "kl_factor": .0625, "v_loss_layer": 31, "fact_token": "subject_last",
                "mom2_update_weight": 15000, "mom2_n_samples": 100000,
                "mom2_dtype": "float32"}
    if family == "BASE_ALPHAEDIT":
        expected.update(L2=10, nullspace_threshold=.02)
    for key, value in expected.items():
        if getattr(hp, key, None) != value:
            raise BaselineBlocked(f"BASELINE_HPARAM_CHANGED:{family}:{key}")


def _stats_guard(original, options):
    def stats(*args, **kwargs):
        if kwargs.get("force_recompute", False):
            raise BaselineBlocked("BASELINE_C0_REGENERATION_FORBIDDEN")
        kwargs["model_name"] = "Meta-Llama-3-8B-Instruct"
        kwargs["download"] = False
        expected = (Path(args[3]) / kwargs["model_name"] / f"{args[4]}_stats" /
                    f"{args[2]}_{kwargs['precision']}_mom2_{kwargs['sample_size']}.npz")
        if not expected.is_file():
            raise BaselineBlocked(f"BASELINE_C0_MISSING:{expected}")
        return original(*args, **kwargs)
    return stats


@contextmanager
def _bound_native(family, options, model, tok, contexts):
    """Process-local metadata bindings, original apply functions remain intact."""
    import torch
    if family == "BASE_ALPHAEDIT":
        module_name, hp_name, hp_class = "AlphaEdit.AlphaEdit_main", "AlphaEdit.AlphaEdit_hparams", "AlphaEditHyperParams"
    else:
        module_name = "memit.memit_seq_main" if family == "MEMIT-H" else "memit.memit_main"
        hp_name, hp_class = "memit.memit_hparams", "MEMITHyperParams"
    try:
        module = importlib.import_module(module_name)
    except (ImportError, FileNotFoundError) as exc:
        raise BaselineBlocked(f"BASELINE_SOURCE_UNAVAILABLE:{module_name}:{exc}") from exc
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(Path(options["native_root"]).resolve()):
        raise BaselineBlocked(f"BASELINE_SOURCE_OUTSIDE_PINNED_ROOT:{module_path}")
    identity = member(module_path)
    expected_sha = options.get("source_sha256", {}).get(module_name)
    if expected_sha is not None and identity["sha256"] != expected_sha:
        raise BaselineBlocked(f"BASELINE_SOURCE_SHA_CHANGED:{module_name}")
    config_path = Path(options["configs"][family])
    hp = getattr(importlib.import_module(hp_name), hp_class).from_json(config_path)
    _verify_hparams(hp, family)
    saved = {k: getattr(module, k) for k in ("CONTEXT_TEMPLATES_CACHE", "COV_CACHE", "STATS_DIR", "layer_stats")}
    module.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(contexts)
    module.COV_CACHE = {}
    module.STATS_DIR = Path(options["stats_root"])
    module.layer_stats = _stats_guard(saved["layer_stats"], options)
    state = projector = None
    try:
        if family == "BASE_ALPHAEDIT":
            # One bounded selected projector copy; no implicit alternate P.
            allp = torch.load(options["projector"], map_location="cpu", weights_only=True, mmap=True)
            indices = [options["projector_source_layers"].index(l) for l in LAYERS]
            require(isinstance(allp, torch.Tensor) and allp.dtype == torch.float32,
                    "BASELINE_PROJECTOR_SCHEMA")
            require(allp.ndim == 3 and allp.shape[1:] == (14336, 14336), "BASELINE_PROJECTOR_SHAPE")
            projector = allp[indices].clone()
            del allp
            state = torch.zeros_like(projector)
        elif family == "MEMIT-H":
            state = torch.zeros((5, 14336, 14336), dtype=torch.float32, device="cpu")
        else:
            state = torch.empty(0, dtype=torch.float32)  # MEMIT has no history.

        def apply(requests):
            if family == "BASE_ALPHAEDIT":
                result, returned = module.apply_AlphaEdit_to_model(
                    model, tok, requests, hp, cache_template=None, cache_c=state, P=projector)
                require(result is model and returned is state, "BASELINE_ALPHA_RETURN_IDENTITY")
            elif family == "MEMIT-H":
                result, returned = module.apply_memit_seq_to_model(
                    model, tok, requests, hp, copy=False, return_orig_weights=False,
                    cache_template=None, cache_c=state)
                require(result is model and returned is state, "BASELINE_MEMITH_RETURN_IDENTITY")
            else:
                result, originals = module.apply_memit_to_model(
                    model, tok, requests, hp, copy=False, return_orig_weights=False, cache_template=None)
                require(result is model and originals == {}, "BASELINE_MEMIT_RETURN_IDENTITY")

        binding = {"family": family, "native": identity, "hparams": vars(hp),
                   "hparams_file": member(config_path), "context_identity": digest(contexts),
                   "history": "NONE" if family == "BASE_MEMIT" else "native post-all-five key append",
                   "projector_physical_layers": list(LAYERS) if projector is not None else [],
                   "projector_sha256": _tensor_sha(projector) if projector is not None else None,
                   "cache_template": None, "save_checkpoints": False,
                   "runtime_baseline_equivalence": "NOT_ASSUMED; current runtime separately recorded"}
        yield module, hp, state, apply, binding
    finally:
        for key, value in saved.items():
            setattr(module, key, value)


@contextmanager
def _observe_native(module, hp, family, model):
    """Counters/provenance only; original functions and optimizer are delegated."""
    import torch
    original_z, original_keys, original_torch = module.compute_z, module.compute_ks, module.torch
    zmodule = importlib.import_module(original_z.__module__)
    original_ztorch, original_nethook = zmodule.torch, zmodule.nethook
    receipt = {"z_calls": [], "keys": [], "solves": [], "target_forward_calls": 0,
               "target_adam_steps": 0, "target_seconds": 0., "key_seconds": 0.,
               "solve_seconds": 0., "tensor_tuple_adapter": True,
               "optimizer_and_stopping": "ORIGINAL_UNCHANGED"}
    active_z = [False]

    class CountAdam(original_ztorch.optim.Adam):
        def step(self, *args, **kwargs):
            receipt["target_adam_steps"] += 1
            return super().step(*args, **kwargs)

    def forward(_module, _args):
        if active_z[0]:
            receipt["target_forward_calls"] += 1

    def z(*args, **kwargs):
        start = time.monotonic()
        active_z[0] = True
        try:
            value = original_z(*args, **kwargs)
            _synchronize()
            require(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()), "BASELINE_Z_NONFINITE")
            receipt["z_calls"].append({"case_id": int(args[2]["case_id"]), "layer": int(args[4]),
                                        "sha256": _tensor_sha(value), "norm": float(value.norm())})
            return value
        finally:
            active_z[0] = False
            receipt["target_seconds"] += time.monotonic() - start

    def keys(*args, **kwargs):
        index = len(receipt["keys"])
        expected = 5 if family == "BASE_MEMIT" else 10
        require(index < expected and args[4] == hp.layers[index % 5], "BASELINE_KEY_ORDER")
        require(len(receipt["solves"]) == min(index, 5), "BASELINE_POST_KEY_BEFORE_ALL_WRITES")
        start = time.monotonic()
        value = original_keys(*args, **kwargs)
        _synchronize()
        require(bool(torch.isfinite(value).all()), "BASELINE_KEY_NONFINITE")
        receipt["key_seconds"] += time.monotonic() - start
        receipt["keys"].append({"layer": int(args[4]), "phase": "pre_layer" if index < 5 else "post_all_five",
                                  "shape": list(value.shape), "sha256": _tensor_sha(value)})
        return value

    def solve(a, b, *args, **kwargs):
        index = len(receipt["solves"])
        require(index < 5 and len(receipt["keys"]) == index + 1, "BASELINE_SOLVE_ORDER")
        require(a.dtype == b.dtype == (torch.float32 if family == "BASE_ALPHAEDIT" else torch.float64),
                "BASELINE_SOLVE_DTYPE")
        require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "BASELINE_SOLVE_INPUT_NONFINITE")
        start = time.monotonic()
        value = original_torch.linalg.solve(a, b, *args, **kwargs)
        _synchronize()
        require(bool(torch.isfinite(value).all()), "BASELINE_SOLVE_NONFINITE")
        receipt["solve_seconds"] += time.monotonic() - start
        receipt["solves"].append({"layer": hp.layers[index], "dtype": str(a.dtype),
                                    "shape": list(a.shape), "rhs_shape": list(b.shape)})
        return value

    module.compute_z, module.compute_ks = z, keys
    module.torch = _Proxy(original_torch, linalg=_Proxy(original_torch.linalg, solve=solve))
    zmodule.torch = _Proxy(original_ztorch, optim=_Proxy(original_ztorch.optim, Adam=CountAdam))
    zmodule.nethook = _Proxy(original_nethook, TraceDict=_trace_dict_compat(original_nethook.TraceDict))
    handle = model.register_forward_pre_hook(forward)
    try:
        yield receipt
    finally:
        handle.remove()
        module.compute_z, module.compute_ks, module.torch = original_z, original_keys, original_torch
        zmodule.torch, zmodule.nethook = original_ztorch, original_nethook


def run_baseline_pilots(model, tok, records8, contexts, config, out, w0, evaluate=None):
    """Run each named baseline's independent fresh-W0 BS4x2 and restore finally.

    evaluate(model, tok, records, family, batch_index) is optional and must be
    observation only; state/RNG/nonselected guards surround the callback.
    Native source unavailable or incompatible => precise failure, no substitute.
    """
    import torch
    import transformers
    options = _config(config)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    require(len(records8) == 8 and len({int(r["case_id"]) for r in records8}) == 8,
            "BASELINE_PILOT_FIRST8_IDENTITY")
    weights = _weights(model)
    require(all(w.dtype == torch.float32 and tuple(w.shape) == (4096, 14336) for w in weights.values()),
            "BASELINE_FP32_FIVE_WEIGHT_SHAPE")
    require(len(contexts) == 2 and list(map(len, contexts)) == [1, 5] and contexts[0] == ["{}"],
            "BASELINE_NATIVE_CONTEXT_GROUPS")
    if options.get("context_path"):
        require(member(options["context_path"])["sha256"] == options.get("context_sha256", CONTEXT_SHA),
                "BASELINE_CONTEXT_FILE_SHA")
        require(json.loads(Path(options["context_path"]).read_text()) == contexts, "BASELINE_CONTEXT_BYTES")
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    rng = _rng_capture()
    original_context = digest(contexts)
    selected_ids = {id(w) for w in weights.values()}
    nonselected = {name: (p.data_ptr(), p._version) for name, p in model.named_parameters() if id(p) not in selected_ids}
    start = time.monotonic()
    results = []
    stage = "PREPARE"
    try:
        with _import_native(options["native_root"]):
            for family in options["families"]:
                stage = family + "/RESTORE_W0"
                _restore_w0(weights, w0)
                _rng_restore(rng)
                root = out / family
                root.mkdir()
                with _bound_native(family, options, model, tok, contexts) as (module, hp, state, apply, binding):
                    write(root / "binding.json", {**binding, "torch": torch.__version__,
                        "transformers": transformers.__version__, "tokenizer_class": type(tok).__name__,
                        "tokenizer_add_bos": getattr(tok, "add_bos_token", None),
                        "tokenizer_padding_side": tok.padding_side, "context_source": options.get("context_path"),
                        "precision": "float32", "attention": getattr(model.config, "_attn_implementation", None),
                        "model_use_cache": getattr(model.config, "use_cache", None),
                        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                        "compatibility_adapter": {"function": "_trace_dict_compat", "change": "Tensor/tuple view only",
                            "native_source_files_modified": False, "prior_native_transformers": "4.44.2",
                            "current_transformers": transformers.__version__,
                            "numerical_equivalence_to_prior_run": "NOT_ESTABLISHED",
                            "CPU_container_output_gradient": "CHECK_SEPARATE_TEST_RECEIPT"},
                        "source_closure": _source_closure(options["native_root"]),
                        "first8_case_ids": [int(r["case_id"]) for r in records8],
                        "parent_rng_sha256": _rng_hash(rng), "W0": _weight_hashes(weights)})
                    prior = {"W": _weight_hashes(weights), "H": _tensor_sha(state)}
                    prior_rng = _rng_hash(_rng_capture())
                    family_rows = []
                    for bi in (1, 2):
                        stage = family + f"/B{bi}_NATIVE"
                        rows = records8[(bi - 1) * 4:bi * 4]
                        requests = [dict(copy.deepcopy(r["requested_rewrite"]), case_id=int(r["case_id"])) for r in rows]
                        entry = {"W": _weight_hashes(weights), "H": _tensor_sha(state)}
                        require(entry == prior, "BASELINE_OWN_ENTRY_CHAIN")
                        entry_rng = _rng_hash(_rng_capture())
                        require(entry_rng == prior_rng, "BASELINE_OWN_RNG_CHAIN")
                        native_start = time.monotonic()
                        with _observe_native(module, hp, family, model) as counters:
                            try:
                                apply(requests)
                            finally:
                                write(root / f"B{bi:03d}-native-counters.json", counters)
                        _synchronize()
                        native_seconds = time.monotonic() - native_start
                        require(len(counters["z_calls"]) == 4 and
                                [x["case_id"] for x in counters["z_calls"]] == [r["case_id"] for r in rows],
                                "BASELINE_NATIVE_TARGET_COVERAGE")
                        require(all(x["layer"] == 8 for x in counters["z_calls"]), "BASELINE_NATIVE_TARGET_LAYER")
                        require(len(counters["solves"]) == 5, "BASELINE_NATIVE_SOLVE_COUNT")
                        expected_keys = 5 if family == "BASE_MEMIT" else 10
                        require(len(counters["keys"]) == expected_keys, "BASELINE_NATIVE_KEY_COUNT")
                        require(all(bool(torch.isfinite(w).all()) for w in weights.values()) and
                                bool(torch.isfinite(state).all()), "BASELINE_NONFINITE_ENDPOINT")
                        endpoint = {"W": _weight_hashes(weights), "H": _tensor_sha(state)}
                        before_rng = _rng_hash(_rng_capture())
                        context_before = digest(module.CONTEXT_TEMPLATES_CACHE)
                        stage = family + f"/B{bi}_OBSERVER"
                        observation_start = time.monotonic()
                        observation = evaluate(model, tok, rows, family, bi) if evaluate else None
                        require(endpoint == {"W": _weight_hashes(weights), "H": _tensor_sha(state)},
                                "BASELINE_OBSERVER_STATE_MUTATION")
                        require(_rng_hash(_rng_capture()) == before_rng and
                                digest(module.CONTEXT_TEMPLATES_CACHE) == context_before == original_context,
                                "BASELINE_OBSERVER_AUXILIARY_MUTATION")
                        require(nonselected == {name: (p.data_ptr(), p._version) for name, p in model.named_parameters()
                                                if id(p) not in selected_ids}, "BASELINE_NONSELECTED_MUTATION")
                        row = {"status": "BATCH_COMMITTED", "batch": bi, "batch_size": 4,
                               "offered": bi * 4, "case_ids": [r["case_id"] for r in rows],
                               "entry": entry, "endpoint": endpoint, "native_seconds": native_seconds,
                               "entry_rng_sha256": entry_rng, "endpoint_rng_sha256": before_rng,
                               "native_timer_boundary": "native apply plus counter/hash/receipt overhead; components nested",
                               "observer_seconds": time.monotonic() - observation_start,
                               "observer_timer_boundary": "callback plus state/RNG/context validation; not pure evaluator",
                               "history_append_layers": 0 if family == "BASE_MEMIT" else 5,
                               "z_fits": 4, "solve_calls": 5,
                               "target_forward_calls": counters["target_forward_calls"],
                               "target_adam_steps": counters["target_adam_steps"],
                               "observer": observation if evaluate else "NOT_MEASURED_CALLBACK_ABSENT",
                               "observer_state_rng_context_unchanged": True,
                               "nonselected_guard": "POINTER_VERSION_NOT_FULL_BYTE_SCAN",
                               "save_checkpoints": False, "exact_resume": "NOT_AVAILABLE"}
                        write(root / f"B{bi:03d}-commit.json", row)
                        family_rows.append(row)
                        prior = endpoint
                        prior_rng = before_rng
                    results.append({"family": family, "status": "COMPLETED", "batches": 2,
                                    "requests": 8, "z_fits": 8, "solve_calls": 10,
                                    "history_append_layers": sum(r["history_append_layers"] for r in family_rows),
                                    "native_seconds": sum(r["native_seconds"] for r in family_rows),
                                    "observer_seconds": sum(r["observer_seconds"] for r in family_rows)})
                # All native C0/projector/history references from this family are released.
                del state, apply, module
                _restore_w0(weights, w0)
                _rng_restore(rng)
            stage = "COMPLETE"
    except BaseException as exc:
        write(out / "failure.json", {"status": "TECHNICAL_FAILURE", "stage": stage,
             "first_exception": repr(exc), "traceback": traceback.format_exc(),
             "completed_families": results, "seconds": time.monotonic() - start,
             "source_preserved": True, "save_checkpoints": False})
        raise
    finally:
        # Do not mask first failure with cleanup. Exact cleanup receipt is separate.
        already_failing = sys.exc_info()[0] is not None
        try:
            _restore_w0(weights, w0)
            _rng_restore(rng)
            for name, param in model.named_parameters():
                param.requires_grad_(flags[name])
            require(digest(contexts) == original_context, "BASELINE_CALLER_CONTEXT_MUTATION")
            write(out / "restore.json", {"W0_exact": True, "RNG_exact": True,
                  "requires_grad_restored": True, "contexts_unchanged": True,
                  "pilot_history": "DISCARDED_RAM_ONLY; next family constructs own zeros",
                  "save_checkpoints": False, "state": _weight_hashes(weights)})
        except BaseException as cleanup:
            write(out / "cleanup-failure.json", {"error": repr(cleanup), "stage": stage})
            if not already_failing:
                raise
    terminal = {"status": "COMPLETED", "families": results,
                "seconds": time.monotonic() - start, "new_native_z_fits": sum(r["z_fits"] for r in results),
                "reused_same8_receipts": options.get("reused_same8_receipts", {}),
                "scientific_quality_gate": False, "save_checkpoints": False,
                "exact_resume": "NOT_AVAILABLE", "W0_and_RNG_restored": True}
    write(out / "terminal.json", terminal)
    return terminal
