"""Direct, SHA-bound stock native Llama baseline adapters.

Only module import/path plumbing and observable call counters are wrapped.
No PRICE optimizer/writer, fabricated MEMIT history, fit caching, checkpoint or
durable tensor is introduced. PRUNE uses the explicitly approved cold-base fix.
"""
import copy
from dataclasses import asdict
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import math
from pathlib import Path
import sys
import time
import types
import uuid

import torch

from .native_binding import (METHODS, SPECS, SCIENTIFIC_FIELDS, asset_stat,
    digest, effective_source, require, source_files, verify)


class _NativeLoader(importlib.abc.Loader):
    def __init__(self, binding, namespace, relative):
        self.binding, self.namespace, self.relative = binding, namespace, relative

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        row = next(row for row in self.binding['files'] if row['relative'] == self.relative)
        path = verify(row)
        text = effective_source(self.relative, path.read_bytes(), self.namespace)
        module.__file__ = str(path)
        exec(compile(text, str(path), 'exec'), module.__dict__)


class _NativeFinder(importlib.abc.MetaPathFinder):
    def __init__(self, binding, namespace):
        self.binding, self.namespace = binding, namespace
        self.paths = {row['relative'] for row in binding['files'] if row['relative'].endswith('.py')}

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.namespace + '.'):
            return None
        relative = fullname[len(self.namespace) + 1:].replace('.', '/') + '.py'
        require(relative in self.paths, 'NATIVE_IMPORT_NOT_IN_EXACT_CLOSURE:' + relative)
        loader = _NativeLoader(self.binding, self.namespace, relative)
        return importlib.util.spec_from_loader(fullname, loader,
            origin=str(Path(self.binding['root']) / relative))


def load_native(binding):
    """CPU-safe import/parser check; never load models, covariance or P here."""
    method = binding['method']
    require(method in METHODS, 'NATIVE_METHOD')
    if binding.get('private_source_copy'):
        root = Path(binding['root']).resolve()
        approved = Path('/data/janghj/ODE-edit/local/llama3-baselines-fluency-consistency-2k').resolve()
        require(root.is_relative_to(approved) and root != approved
                and binding['origin_root'] == str(SPECS[method]['root']), 'NATIVE_PRIVATE_ROOT_SCOPE')
        origin = {row['relative']: row for row in binding['origin_files']}
        require(set(origin) == {row['relative'] for row in binding['files']}, 'NATIVE_PRIVATE_ORIGIN_CLOSURE')
        for row in binding['files']:
            prior = origin[row['relative']]
            require((row['bytes'], row['sha256']) == (prior['bytes'], prior['sha256'])
                    and Path(prior['path']).resolve() == SPECS[method]['root'] / row['relative'],
                    'NATIVE_PRIVATE_ORIGINAL_BYTES_IDENTITY')
        require((binding['hparams']['bytes'], binding['hparams']['sha256'])
                == (binding['origin_hparams']['bytes'], binding['origin_hparams']['sha256'])
                and Path(binding['origin_hparams']['path']).resolve() == Path(SPECS[method]['hparams']),
                'NATIVE_PRIVATE_HPARAMS_ORIGIN')
        require(Path(binding['hparams']['path']).resolve().is_relative_to(root), 'NATIVE_PRIVATE_HPARAMS_SCOPE')
    else:
        require(binding['root'] == str(SPECS[method]['root']), 'NATIVE_ORIGIN_ROOT')
    expected = set(source_files(method))
    if not binding['easyedit']:
        expected.add('globals.yml')
    require({row['relative'] for row in binding['files']} == expected, 'NATIVE_EXACT_SOURCE_CLOSURE')
    for row in binding['files']:
        require(verify(row) == Path(binding['root']) / row['relative'], 'NATIVE_ORIGINAL_PATH')
    namespace = binding['namespace'] + '_' + uuid.uuid4().hex
    packages = {''}
    for relative in source_files(method):
        pieces = relative.split('/')[:-1]
        for n in range(1, len(pieces) + 1):
            packages.add('.'.join(pieces[:n]))
    for relative in sorted(packages, key=len):
        name = namespace + ('.' + relative if relative else '')
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(Path(binding['root']).joinpath(*relative.split('.')))]
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
    finder = _NativeFinder(binding, namespace)
    sys.meta_path.insert(0, finder)
    try:
        module = importlib.import_module(namespace + '.' + binding['package'] + '.' + binding['main'])
        z = importlib.import_module(namespace + '.' + binding['package'] + '.compute_z')
    finally:
        sys.meta_path.remove(finder)
    actual = {Path(m.__file__).relative_to(binding['root']).as_posix()
        for name, m in sys.modules.items() if name.startswith(namespace + '.') and getattr(m, '__file__', None)}
    require(actual == {p for p in expected if p.endswith('.py')}, 'NATIVE_IMPORTED_CLOSURE_IDENTITY')
    cls = getattr(module, binding['parser'])
    hp_path = str(verify(binding['hparams']))
    hp = cls.from_hparams(hp_path) if binding['easyedit'] else cls.from_json(hp_path)
    actual_hp = asdict(hp)
    require({key: actual_hp[key] for key in binding['scientific_fields']}
            == binding['scientific_fields'], 'NATIVE_PARSED_PARAMETERS_CHANGED')
    require(hp.layers == binding['layers'], 'NATIVE_SELECTED_PHYSICAL_LAYERS')
    return types.SimpleNamespace(module=module, z_module=z, hp=hp,
        namespace=namespace, binding=binding, imported_files=sorted(actual))


def normalize_requests(records):
    """Schema conversion only; preserve occurrences and native leading-space rule."""
    requests = []
    for row in records:
        rewrite = copy.deepcopy(row.get('requested_rewrite', row))
        rewrite['case_id'] = row['case_id']
        target = rewrite['target_new']
        if isinstance(target, str):
            rewrite['target_new'] = {'str': target}
        require(type(rewrite['case_id']) is int and isinstance(rewrite['target_new'], dict)
            and type(rewrite['target_new'].get('str')) is str and rewrite['target_new']['str']
            and all(type(rewrite.get(k)) is str and rewrite[k] for k in ('prompt', 'subject')),
            'NATIVE_REQUEST_SCHEMA')
        require(rewrite['prompt'].count('{}') == 1, 'NATIVE_SUBJECT_PLACEHOLDER')
        requests.append(rewrite)
    require(len(requests) == 100, 'NATIVE_BS100_OCCURRENCES_NO_EXCLUSION')
    return requests


native_requests = normalize_requests


class _TupleTraceDict:
    """Compatibility view for original tuple-assuming Llama compute_z only.

    Transformers 4.57 returns a decoder Tensor, whereas the original native
    code expects (Tensor, ...). A pre-Trace hook boxes the existing Tensor;
    original Trace/edit functions act on that same Tensor; a post-Trace hook
    unboxes it before the physical decoder output is consumed. No new tensor,
    forward, arithmetic, detach or persistent model hook is introduced.
    """
    def __init__(self, native, allowed_layers, counters, *args, **kwargs):
        bound = inspect.signature(native.TraceDict).bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        model, layers = values['module'], list(dict.fromkeys(values['layers']))
        require(set(layers) <= set(allowed_layers), 'NATIVE_TUPLE_SHIM_DECODER_LAYERS_ONLY')
        require(not values['clone'] and not values['detach'] and not values['retain_grad']
                and not values['stop'], 'NATIVE_TUPLE_SHIM_EXACT_COMPUTE_Z_OPTIONS')
        self.native, self.counters = native, counters
        self.handles, self.tensor_outputs, self.inner = [], {}, None
        try:
            for layer in layers:
                module = native.get_module(model, layer)
                def box(module, inputs, output, layer=layer):
                    is_tensor = isinstance(output, torch.Tensor)
                    hidden = output if is_tensor else output[0] if isinstance(output, tuple) and output else None
                    require(isinstance(hidden, torch.Tensor) and hidden.ndim == 3,
                            'NATIVE_TUPLE_SHIM_DECODER_OUTPUT_LAYOUT')
                    self.tensor_outputs[layer] = is_tensor
                    if is_tensor:
                        self.counters['boxed_decoder_outputs'] += 1
                        return (output,)
                    self.counters['native_tuple_outputs'] += 1
                    return output
                self.handles.append(module.register_forward_hook(box))
            self.inner = native.TraceDict(*bound.args, **bound.kwargs)
            for layer in layers:
                module = native.get_module(model, layer)
                def unbox(module, inputs, output, layer=layer):
                    if self.tensor_outputs[layer]:
                        require(isinstance(output, tuple) and len(output) == 1
                                and isinstance(output[0], torch.Tensor), 'NATIVE_TUPLE_SHIM_UNBOX_IDENTITY')
                        self.counters['unboxed_decoder_outputs'] += 1
                        return output[0]
                    require(isinstance(output, tuple), 'NATIVE_TUPLE_SHIM_ORIGINAL_TUPLE_PRESERVED')
                    return output
                self.handles.append(module.register_forward_hook(unbox))
        except BaseException:
            self.close()
            raise
        self.counters['trace_calls'] += 1

    def __getitem__(self, key):
        return self.inner[key]

    def __getattr__(self, key):
        return getattr(self.inner, key)

    def __enter__(self):
        return self

    def close(self):
        if self.inner is not None:
            self.inner.close()
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []

    def __exit__(self, exc_type, value, traceback):
        self.close()
        return False


class _TupleNethook:
    """Only one foreign compute_z module sees this private compatibility view."""
    def __init__(self, original, allowed_layers):
        self.original, self.allowed_layers = original, allowed_layers
        self.counters = dict(trace_calls=0, boxed_decoder_outputs=0,
                             unboxed_decoder_outputs=0, native_tuple_outputs=0)

    def __getattr__(self, key):
        return getattr(self.original, key)

    def TraceDict(self, *args, **kwargs):
        return _TupleTraceDict(self.original, self.allowed_layers, self.counters, *args, **kwargs)


def native_terminal_compression(cold_weight, dense_weight):
    """Native PRUNE spectral formula, final=savedW0+compressed(dense-savedW0).

    Preserve the original cold CPU SVD and current update device SVD. Cold W0
    is RAM-only; it is never added to already-edited W20 a second time.
    """
    require(cold_weight.shape == dense_weight.shape
            and cold_weight.dtype == dense_weight.dtype == torch.float32,
            'PRUNE_TERMINAL_LAYOUT')
    require(torch.isfinite(cold_weight).all().item() and torch.isfinite(dense_weight).all().item(),
            'PRUNE_TERMINAL_INPUT_NONFINITE')
    with torch.no_grad():
        update = dense_weight - cold_weight.to(dense_weight.device)
        _, cold_singular, _ = torch.svd(cold_weight)
        max_sigma = cold_singular.max().item()
        u, singular, v = torch.svd(update)
        adjusted = torch.where(singular > max_sigma,
            torch.log(singular) - torch.log(torch.tensor(max_sigma, device=dense_weight.device)) + max_sigma,
            singular)
        compressed = torch.matmul(u, torch.matmul(torch.diag(adjusted), v.t()))
        final = cold_weight.to(dense_weight.device) + compressed
        require(torch.isfinite(final).all().item(), 'PRUNE_TERMINAL_RESULT_NONFINITE')
    return final, dict(explicit_repair='PRUNE_TERMINAL_BASE_FIX', native_svd_calls=2,
        max_sigma_cold=float(max_sigma), singular_values=int(singular.numel()),
        compressed_count=int((singular > max_sigma).sum().item()),
        dense_delta_norm=float(torch.linalg.vector_norm(update)),
        compressed_delta_norm=float(torch.linalg.vector_norm(compressed)),
        double_add=False, base='RAM_COLD_W0', checkpoint_saved=False)


class _LinalgCalls:
    def __init__(self, original, engine):
        self.original, self.engine = original, engine

    def __getattr__(self, name):
        return getattr(self.original, name)

    def solve(self, *args, **kwargs):
        result = self.engine._call('solves', self.original.solve, *args, **kwargs)
        require(torch.isfinite(result).all().item(), 'NATIVE_SOLVE_NONFINITE')
        return result


class _OptimCalls:
    def __init__(self, original, engine):
        self.original, self.engine = original, engine

    def __getattr__(self, name):
        return getattr(self.original, name)

    def Adam(self, *args, **kwargs):
        result = self.original.Adam(*args, **kwargs)
        original_step = result.step
        def step(*args, **kwargs):
            return self.engine._call('fit_updates', original_step, *args, **kwargs)
        result.step = step
        return result


class _TorchCalls:
    def __init__(self, original, engine, optimizer=False):
        self.original = original
        self.linalg = _LinalgCalls(original.linalg, engine)
        self.optim = _OptimCalls(original.optim, engine) if optimizer else original.optim

    def __getattr__(self, name):
        return getattr(self.original, name)


COUNTERS = ('native_z', 'write_keys', 'history_keys', 'solves', 'history_appends',
    'fit_forwards', 'fit_updates', 'public_applies')


class NativeEngine:
    def __init__(self, config, model, tokenizer, bundle, projector=None):
        self.config, self.model, self.tokenizer, self.bundle = config, model, tokenizer, bundle
        self.module, self.hp, self.binding = bundle.module, bundle.hp, bundle.binding
        self.method, self.layers = self.binding['method'], list(self.hp.layers)
        self.writer = {'MEMIT': 'memit', 'PRUNE': 'memit_prune', 'RECT': 'memit_rect',
            'ALPHAEDIT': 'alphaedit', 'ALPHAEDIT_BLUE': 'alphaedit_blue', 'CAKE': 'cake'}[self.method]
        self.next_batch, self.current, self.on_progress = 1, None, None
        self.cumulative = {key: 0 for key in COUNTERS}
        self.P = projector
        self.H = (torch.zeros(len(self.layers), 14336, 14336, dtype=torch.float32, device='cpu')
                  if self.method in ('CAKE', 'ALPHAEDIT_BLUE') else None)
        self.saved_cold_weights, self.prune_applied = {}, False
        self.context_receipt = dict(status='PRESERVED_NATIVE_INPUT_NOT_NEW_GENERATION',
            contexts_sha256=self.binding['context_reuse']['contexts']['sha256'],
            source=self.binding['context_reuse']['source'], generated_by_wrapper=False,
            no_new_fit=True, native_generator_bitwise_parity_claim=False)
        require(self.module.CONTEXT_TEMPLATES_CACHE is None and not self.module.COV_CACHE,
                'NATIVE_INITIAL_MODULE_STATE')
        self.original_z, self.original_ks = self.module.compute_z, self.module.compute_ks
        self.module.compute_z, self.module.compute_ks = self._compute_z, self._compute_ks
        self.module.torch = _TorchCalls(torch, self)
        self.bundle.z_module.torch = _TorchCalls(torch, self, optimizer=True)
        self.tuple_compatibility = None
        if not self.binding['easyedit']:
            allowed = [self.hp.layer_module_tmp.format(layer)
                       for layer in set(self.layers + [self.hp.v_loss_layer])]
            self.tuple_compatibility = _TupleNethook(self.bundle.z_module.nethook, allowed)
            # Shared namespace util.nethook, repr_tools and main remain original.
            self.bundle.z_module.nethook = self.tuple_compatibility
        self.native_apply = getattr(self.module, {
            'MEMIT': 'apply_memit_to_model', 'PRUNE': 'apply_memit_to_model',
            'RECT': 'apply_memit_rect_to_model', 'ALPHAEDIT': 'apply_AlphaEdit_to_model',
            'ALPHAEDIT_BLUE': 'apply_AlphaEdit_to_model', 'CAKE': 'apply_Cake_to_model'}[self.method])
        self._bind_stats_guard()
        if self.method == 'PRUNE':
            for layer in self.layers:
                name = self.hp.rewrite_module_tmp.format(layer) + '.weight'
                self.saved_cold_weights[name] = self.module.nethook.get_parameter(model, name).detach().cpu().clone()

    @property
    def progress(self):
        return self.on_progress

    @progress.setter
    def progress(self, callback):
        require(callback is None or callable(callback), 'NATIVE_PROGRESS_CALLBACK')
        self.on_progress = callback

    @property
    def counts(self):
        return dict(self.cumulative)

    def expected_counts(self):
        sites = len(self.layers)
        history = self.binding['native_has_history']
        return dict(native_z=100 * sites if self.method == 'ALPHAEDIT_BLUE' else 100,
            write_keys=sites, history_keys=sites if history else 0,
            solves=sites, history_appends=sites if history else 0)

    def _call(self, key, original, *args, **kwargs):
        require(self.current is not None, 'NATIVE_COUNTER_OUTSIDE_APPLY')
        self.current[key] += 1
        self.cumulative[key] += 1
        started = time.monotonic()
        try:
            return original(*args, **kwargs)
        finally:
            self.current['seconds'][key] += time.monotonic() - started

    def _emit(self, row):
        if self.on_progress is not None:
            try:
                self.on_progress(row)
            except Exception:
                self.current['logging_callback_errors'] += 1

    def _fit_forward(self, *unused):
        self.current['fit_forwards'] += 1
        self.cumulative['fit_forwards'] += 1
        self._emit(dict(batch=self.current['batch'], native_z_completed=self.cumulative['native_z'] - 1,
            fit_global_candidate=self.cumulative['fit_forwards'], fit_updates=self.cumulative['fit_updates']))

    def _compute_z(self, *args, **kwargs):
        observed, previous = {}, sys.getprofile()
        def trace(frame, event, arg):
            if event == 'return' and frame.f_code is self.original_z.__code__:
                observed.update({key: frame.f_locals[key] for key in
                    ('it', 'loss', 'nll_loss', 'kl_loss', 'weight_decay') if key in frame.f_locals})
            if previous is not None:
                previous(frame, event, arg)
        before = (self.current['fit_forwards'], self.current['fit_updates'])
        handle = self.model.register_forward_hook(self._fit_forward)
        sys.setprofile(trace)
        try:
            result = self._call('native_z', self.original_z, *args, **kwargs)
        finally:
            sys.setprofile(previous)
            handle.remove()
        require(isinstance(result, torch.Tensor) and tuple(result.shape) == (4096,)
                and result.dtype == torch.float32 and torch.isfinite(result).all().item(), 'NATIVE_TARGET_LAYOUT_NONFINITE')
        require('it' in observed, 'NATIVE_FIT_ITERATION_CAPTURE')
        evaluations = int(observed.pop('it')) + 1
        updates = self.current['fit_updates'] - before[1]
        require(1 <= evaluations <= self.hp.v_num_grad_steps
                and updates == evaluations - 1 and self.current['fit_forwards'] - before[0] == evaluations,
                'NATIVE_ORIGINAL_FIT_COUNTERS')
        row = dict(request_index=self.current['native_z'], evaluations=evaluations, Adam_updates=updates)
        row.update({k: float(v.detach()) for k, v in observed.items()})
        require(all(math.isfinite(v) for v in row.values()), 'NATIVE_FIT_NONFINITE')
        row['stop'] = 'TOTAL_LOSS_BELOW_005' if row['loss'] < .05 else 'BUDGET_EXHAUSTED'
        self.current['fit_trace'].append(row)
        self._emit(dict(batch=self.current['batch'], native_z_completed=self.cumulative['native_z'],
            fit_global_candidate=self.cumulative['fit_forwards'], fit_updates=self.cumulative['fit_updates'], **row))
        return result

    def _compute_ks(self, *args, **kwargs):
        ordinal = self.current['write_keys'] + self.current['history_keys']
        key = ('history_keys' if self.binding['native_has_history'] and ordinal >= len(self.layers) else 'write_keys')
        result = self._call(key, self.original_ks, *args, **kwargs)
        require(tuple(result.shape) == (100, 14336) and result.dtype == torch.float32
                and torch.isfinite(result).all().item(), 'NATIVE_FULL_B_KEY_LAYOUT_NONFINITE')
        return result

    def _bind_stats_guard(self):
        original = self.module.layer_stats
        signature = inspect.signature(original)
        def existing_only(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            args = bound.arguments
            require(not args.get('force_recompute', False) and args['to_collect'] == ['mom2']
                    and args['sample_size'] == self.hp.mom2_n_samples and args['precision'] == self.hp.mom2_dtype,
                    'NATIVE_C0_RECOMPUTE_NOT_AUTHORIZED')
            layer = args['layer_name']
            row = next((row for row in self.binding['stats']
                        if self.hp.rewrite_module_tmp.format(row['layer']) == layer), None)
            require(row is not None, 'NATIVE_C0_LAYER_BINDING')
            asset_stat(row)
            expected = Path(args['stats_dir']) / self.binding['stats_model_name'] / 'wikipedia_stats' / Path(row['path']).name
            require(expected.resolve() == Path(row['path']), 'NATIVE_C0_EXACT_CACHE_PATH')
            return original(*bound.args, **bound.kwargs)
        self.module.layer_stats = existing_only

    def contexts(self):
        return copy.deepcopy(self.module.CONTEXT_TEMPLATES_CACHE)

    contexts_snapshot = contexts
    context_snapshot = contexts
    snapshot_context = contexts
    context_state = contexts

    def restore_contexts(self, contexts):
        self.module.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(contexts)

    restore_context = restore_contexts
    restore_context_state = restore_contexts

    def prepare_contexts(self):
        require(self.next_batch == 1 and self.module.CONTEXT_TEMPLATES_CACHE is None,
                'NATIVE_CONTEXT_BIND_ONCE')
        path = verify(self.binding['context_reuse']['contexts'])
        import json
        contexts = json.loads(path.read_text())
        require(contexts[0] == ['{}'] and len(contexts) == 2 and len(contexts[1]) == 5,
                'NATIVE_CONTEXT_1_PLUS_5')
        self.module.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(contexts)
        return self.contexts()

    def history(self):
        if not self.binding['native_has_history']:
            return {}
        if self.method == 'ALPHAEDIT':
            if not getattr(self.module, 'cache_c_new', False):
                return {}
            history = self.module.cache_c
        else:
            history = self.H
        require(history.device.type == 'cpu' and history.dtype == torch.float32
                and tuple(history.shape) == (len(self.layers), 14336, 14336), 'NATIVE_HISTORY_LAYOUT')
        return {layer: history[index] for index, layer in enumerate(self.layers)}

    def restore_history(self, history):
        require(set(history) == set(self.history()), 'NATIVE_HISTORY_ROLLBACK_LAYER_SET')
        with torch.no_grad():
            for layer, value in self.history().items():
                value.copy_(history[layer])

    def reset_history_to_uninitialized(self):
        if self.method == 'ALPHAEDIT':
            self.module.cache_c, self.module.cache_c_new = None, False
        elif self.H is not None:
            self.H.zero_()

    def snapshot_ledger(self):
        """RAM-only logical state, retaining large immutable-cache references.

        History values themselves belong to the caller's W/H transaction.  The
        fixed C0/projector/cold-base references are not cloned or serialized.
        Incurred cumulative call/time counters deliberately are NOT rollback
        state: an unsuccessful batch still consumed that physical work.
        """
        def retain(value):
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, dict):
                return {key: retain(item) for key, item in value.items()}
            if isinstance(value, list):
                return [retain(item) for item in value]
            if isinstance(value, tuple):
                return tuple(retain(item) for item in value)
            return copy.deepcopy(value)
        names = ('COV_CACHE', 'cache_c', 'cache_c_new', 'P', 'P_loaded', 'P_loaded_from')
        attributes = {name: (hasattr(self.module, name),
            retain(getattr(self.module, name, None))) for name in names}
        protected = []
        def immutable(value):
            if isinstance(value, torch.Tensor):
                protected.append((value, value.data_ptr(), value._version,
                    tuple(value.shape), str(value.dtype), str(value.device)))
            elif isinstance(value, dict):
                for item in value.values():
                    immutable(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    immutable(item)
        immutable(getattr(self.module, 'COV_CACHE', {}))
        immutable(getattr(self.module, 'P', None))
        immutable(self.P)
        immutable(self.saved_cold_weights)
        return dict(schema='llama-native-logical-ledger-v1', owner=id(self),
            next_batch=self.next_batch, prune_applied=self.prune_applied,
            current=retain(self.current), module_attributes=attributes,
            history_storage=self.H, projector=self.P,
            saved_cold_weights=retain(self.saved_cold_weights),
            protected_tensor_versions=protected)

    def restore_ledger(self, snapshot):
        """Restore batch/PRUNE/cache ledger without erasing incurred counters."""
        require(snapshot['schema'] == 'llama-native-logical-ledger-v1'
                and snapshot['owner'] == id(self), 'NATIVE_LEDGER_SNAPSHOT_OWNER')
        protected_unchanged = all((value.data_ptr(), value._version, tuple(value.shape),
            str(value.dtype), str(value.device)) == (pointer, version, shape, dtype, device)
            for value, pointer, version, shape, dtype, device
            in snapshot['protected_tensor_versions'])
        def retain(value):
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, dict):
                return {key: retain(item) for key, item in value.items()}
            if isinstance(value, list):
                return [retain(item) for item in value]
            if isinstance(value, tuple):
                return tuple(retain(item) for item in value)
            return copy.deepcopy(value)
        for name, (present, value) in snapshot['module_attributes'].items():
            if present:
                setattr(self.module, name, retain(value))
            elif hasattr(self.module, name):
                delattr(self.module, name)
        self.next_batch, self.prune_applied = snapshot['next_batch'], snapshot['prune_applied']
        self.current = retain(snapshot['current'])
        self.H, self.P = snapshot['history_storage'], snapshot['projector']
        self.saved_cold_weights = retain(snapshot['saved_cold_weights'])
        # A protected input mutation is not repairable without an unapproved
        # huge clone. Still restore mutable W/H/ledger best-effort; the caller
        # must reject verified rollback when this is False.
        return protected_unchanged

    def ledger_identity(self, snapshot=None):
        """Small RAM metadata identity; mutable H bytes are checked separately."""
        value = self.snapshot_ledger() if snapshot is None else snapshot
        require(value['schema'] == 'llama-native-logical-ledger-v1'
                and value['owner'] == id(self), 'NATIVE_LEDGER_IDENTITY_OWNER')
        def identity(item):
            if isinstance(item, torch.Tensor):
                # cache_c/H contents are restored by NativeTransaction; their
                # copy_ version changes do not constitute a logical mismatch.
                return ('tensor-reference', id(item), item.data_ptr(), tuple(item.shape),
                        str(item.dtype), str(item.device))
            if isinstance(item, dict):
                return ('dict', tuple((key, identity(entry)) for key, entry in item.items()))
            if isinstance(item, (list, tuple)):
                return (type(item).__name__, tuple(identity(entry) for entry in item))
            return item
        return dict(next_batch=value['next_batch'], prune_applied=value['prune_applied'],
            current=identity(value['current']), module_attributes=identity(value['module_attributes']),
            history_storage=identity(value['history_storage']), projector=identity(value['projector']),
            saved_cold_weights=identity(value['saved_cold_weights']),
            protected_tensor_versions=tuple((id(item), pointer, version, shape, dtype, device)
                for item, pointer, version, shape, dtype, device in value['protected_tensor_versions']))

    def apply(self, records, batch_number):
        require(batch_number == self.next_batch and 1 <= batch_number <= 20, 'NATIVE_BATCH_SEQUENCE_NO_B21')
        require(self.module.CONTEXT_TEMPLATES_CACHE is not None, 'NATIVE_BOUND_CONTEXT_REQUIRED')
        requests = normalize_requests(records)
        if self.binding['easyedit']:
            requests = [dict(row, target_new=row['target_new']['str']) for row in requests]
        self.current = dict(batch=batch_number, logging_callback_errors=0, fit_trace=[],
            seconds={key: 0. for key in COUNTERS}, **{key: 0 for key in COUNTERS})
        previous = self.history()
        hist = None if not previous else next(iter(previous.values()))._base
        before = None if hist is None else (hist.data_ptr(), hist._version)
        context_before = digest(self.contexts())
        started = time.monotonic()
        kwargs = dict(cache_template=None)
        if self.method in ('CAKE', 'ALPHAEDIT_BLUE'):
            kwargs.update(cache_c=self.H, P=self.P)
        else:
            kwargs.update(copy=False, return_orig_weights=False)
            if self.method == 'ALPHAEDIT':
                kwargs['reset_cache'] = batch_number == 1
        try:
            result = self._call('public_applies', self.native_apply,
                self.model, self.tokenizer, requests, self.hp, **kwargs)
        finally:
            if self.binding['native_has_history']:
                now = self.history()
                current_hist = next(iter(now.values()))._base if now else None
                if current_hist is not None:
                    require(before is None or current_hist.data_ptr() == before[0], 'NATIVE_HISTORY_STORAGE_REPLACED')
                    delta = current_hist._version - (0 if before is None else before[1])
                    require(delta >= 0 and delta % 2 == 0, 'NATIVE_HISTORY_INDEXED_APPEND_VERSION')
                    self.current['history_appends'] = delta // 2
                    self.cumulative['history_appends'] += delta // 2
        require(isinstance(result, tuple) and len(result) == 2 and result[0] is self.model, 'NATIVE_SAME_MODEL_RETURN')
        if self.method in ('CAKE', 'ALPHAEDIT_BLUE'):
            require(result[1] is self.H, 'NATIVE_SAME_HISTORY_RETURN')
        else:
            require(isinstance(result[1], dict) and not result[1], 'NATIVE_NO_RETURNED_WEIGHT_PAYLOAD')
        sites = len(self.layers)
        require(self.current['native_z'] == (100 * sites if self.method == 'ALPHAEDIT_BLUE' else 100)
                and self.current['write_keys'] == sites and self.current['solves'] == sites
                and self.current['history_keys'] == (sites if self.binding['native_has_history'] else 0)
                and self.current['history_appends'] == (sites if self.binding['native_has_history'] else 0),
                'NATIVE_EXACT_FIT_WRITE_HISTORY_COUNTS')
        require(context_before == digest(self.contexts()), 'NATIVE_CONTEXT_CHANGED_DURING_APPLY')
        self.next_batch += 1
        return self.model, dict(status='NATIVE_APPLY_RETURNED', method=self.method, writer=self.writer,
            batch=batch_number, requests=100, request_identity=digest(requests), counts=copy.deepcopy(self.current),
            cumulative=self.counts, native_has_history=self.binding['native_has_history'],
            caller_history_appends=0, layers=self.layers, hparams=asdict(self.hp),
            native_source_commit=self.binding['source_commit'], same_model_returned=True,
            compatibility=dict(name='TASK_PRIVATE_DECODER_TENSOR_TUPLE_VIEW',
                native_numerical_AST_preserved=True, physical_output_unboxed=True,
                counters=copy.deepcopy(self.tuple_compatibility.counters)) if self.tuple_compatibility else None,
            context=copy.deepcopy(self.context_receipt), seconds=time.monotonic() - started,
            timing_policy='apply inclusive; native_z/key/solve nested not additive; no added timing synchronization',
            cache_template=None, checkpoint_saved=False, exact_resume='NOT_AVAILABLE')

    def terminal_prune(self):
        require(self.method == 'PRUNE' and self.next_batch == 21 and not self.prune_applied,
                'PRUNE_EXACT_TERMINAL_W20_ONCE')
        started, rows = time.monotonic(), []
        for name, cold in self.saved_cold_weights.items():
            weight = self.module.nethook.get_parameter(self.model, name)
            final, row = native_terminal_compression(cold, weight.detach())
            with torch.no_grad():
                weight.copy_(final)
            rows.append(dict(weight_name=name, **row))
        self.prune_applied = True
        return dict(status='PRUNE_TERMINAL_BASE_FIX_APPLIED', rows=rows,
            seconds=time.monotonic() - started, checkpoint_saved=False)


def prepare_native(c, model, tok, method, attempt=None):
    binding = c['native'][method]
    require(binding['method'] == method, 'NATIVE_METHOD_CONFIG')
    bundle = load_native(binding)
    hp = bundle.hp
    require(model.config.model_type == 'llama' and model.config.hidden_size == 4096
            and model.config.intermediate_size == 14336 and model.config.num_hidden_layers == 32,
            'NATIVE_LLAMA_MODEL_LAYOUT')
    for layer in hp.layers:
        module = bundle.module.nethook.get_module(model, hp.rewrite_module_tmp.format(layer))
        require(isinstance(module, torch.nn.Linear) and module.bias is None
                and tuple(module.weight.shape) == (4096, 14336) and module.weight.dtype == torch.float32,
                'NATIVE_LLAMA_DOWN_PROJ_LAYOUT')
    model.config._name_or_path = binding['stats_model_name']
    hp.device, hp.batch_size = 0, 100
    hp.stats_dir = binding['stats_root']
    bundle.module.STATS_DIR = Path(binding['stats_root'])
    for row in binding['stats']:
        asset_stat(row)
    projector = None
    if binding['native_has_history']:
        path = asset_stat(binding['projector'])['path']
        full = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        require(isinstance(full, torch.Tensor) and full.dtype == torch.float32
                and tuple(full.shape) == (5, 14336, 14336), 'NATIVE_PROJECTOR_LAYOUT')
        slots = binding['projector_slots']
        require(slots == [layer - 4 for layer in hp.layers], 'NATIVE_PROJECTOR_PHYSICAL_SLOTS')
        # Basic indexed views retain mmap storage; avoid a whole advanced-index copy.
        projector = full if slots == [0, 1, 2, 3, 4] else torch.stack([full[index] for index in slots])
        if method == 'ALPHAEDIT':
            hp.P_loc = path
            bundle.module.P, bundle.module.P_loaded, bundle.module.P_loaded_from = full, True, path
    return NativeEngine(c, model, tok, bundle, projector)
