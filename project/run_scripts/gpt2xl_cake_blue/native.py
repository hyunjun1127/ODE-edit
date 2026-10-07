"""Pinned native CAKE/AlphaEdit-BLUE, with no fitting/writer reimplementation.

The adopted source differs only in import/path plumbing. Package shells avoid
the unrelated original package __init__ side effects. Native context generation,
candidate optimization, dtype, solves, residuals and public writes remain native.
Scalar wrappers count existing work without extra forwards or timing sync.
Explicit finite safety guards do synchronize; that cost is not isolated timing.
"""
import copy
import difflib
import hashlib
import importlib
import importlib.machinery
import inspect
import json
import subprocess
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


NATIVE_SPECS = {
    'CAKE': dict(root='/mnt/raid5/janghj/CAKE',
                 commit='0b378234862bd76c69f58404ef84c27d5f4bf9ef',
                 namespace='_odeedit_cake_native', package='Cake', main='Cake_main',
                 hparams='hparams/Cake/gpt2-xl.json',
                 hparams_sha='44cc8de6e4f5978e1d3079fc98e93d75291ec57e86935aab418fe3bf676e1179',
                 hparams_class='CakeHyperParams', layers=[13, 14, 15, 16, 17]),
    'ALPHAEDIT_BLUE': dict(root='/mnt/raid5/janghj/BLUE',
                       commit='311b076a92e4ed0f14f5c8b4909732da781bc5f7',
                       namespace='_odeedit_blue_native', package='AlphaEdit', main='AlphaEdit_main',
                       hparams='hparams/AlphaEdit/gpt2-xl-blue.json',
                       hparams_sha='4514321a42ea06d95b9c41984886095c8b1235c32622d5f54b7a67e04a314818',
                       hparams_class='AlphaEditHyperParams', layers=[13, 17]),
}
SHARED_FILES = ('rome/layer_stats.py', 'rome/repr_tools.py', 'rome/tok_dataset.py',
                'util/generate.py', 'util/globals.py', 'util/hparams.py',
                'util/logit_lens.py', 'util/nethook.py', 'util/runningstats.py')
COUNTERS = ('native_z', 'write_keys', 'history_keys', 'solves', 'history_appends',
            'fit_forwards', 'fit_updates', 'public_applies')
COUNT_FIELDS = COUNTERS[:5]


def require(condition, label):
    if not condition:
        raise RuntimeError(label)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def member(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def verify(row):
    path = Path(row['path']).resolve()
    require(path.is_file() and path.stat().st_size == row['bytes']
            and sha(path) == row['sha256'], 'NATIVE_MEMBER_IDENTITY:' + str(path))
    return path


def source_files(arm):
    spec = NATIVE_SPECS[arm]
    package = spec['package']
    hparams = 'Cake_hparams' if arm == 'CAKE' else 'AlphaEdit_hparams'
    return (f'{package}/{spec["main"]}.py', f'{package}/compute_z.py',
            f'{package}/compute_ks.py', f'{package}/{hparams}.py',
            *SHARED_FILES, 'globals.yml', spec['hparams'])


def effective_source(relative, original, arm):
    """Exact compatibility diff; never alter numerical expressions."""
    if not relative.endswith('.py'):
        return original
    text = original.decode('utf-8')
    namespace = NATIVE_SPECS[arm]['namespace']
    for package in ('util', 'rome'):
        text = text.replace(f'from {package} ', f'from {namespace}.{package} ')
        text = text.replace(f'from {package}.', f'from {namespace}.{package}.')
    if relative == 'Cake/Cake_main.py':
        require(text.count('from notebooks.util import hparams\n') == 1,
                'CAKE_UNUSED_IMPORT_REPAIR_EXACT')
        text = text.replace('from notebooks.util import hparams\n', '')
    if relative == 'util/globals.py':
        require(text.count('with open("globals.yml", "r") as stream:') == 1,
                'NATIVE_GLOBALS_PATH_REPAIR_EXACT')
        text = text.replace('with open("globals.yml", "r") as stream:',
                            'with open(Path(__file__).resolve().parents[1] / "globals.yml", "r") as stream:')
        text = text.replace('    Path(z)\n',
                            '    (Path(z) if Path(z).is_absolute() else Path(__file__).resolve().parents[1] / z)\n')
    return text.encode('utf-8')


def adopt_native(destination):
    """CPU/source-only create-once adoption; originals remain read-only."""
    destination = Path(destination).resolve()
    require(not destination.exists(), 'NATIVE_ADOPTION_CREATE_ONCE')
    destination.mkdir(parents=True)
    result = {}
    for arm, spec in NATIVE_SPECS.items():
        root = Path(spec['root'])
        actual = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'],
                                check=True, capture_output=True, text=True).stdout
        require(actual == spec['commit'] and not status.strip(), 'NATIVE_ORIGINAL_CLEAN_COMMIT:' + arm)
        target = destination / arm
        target.mkdir()
        files = []
        for relative in source_files(arm):
            source = root / relative
            require(source.is_file() and not source.is_symlink(), 'NATIVE_SOURCE_REGULAR_FILE')
            original = source.read_bytes()
            effective = effective_source(relative, original, arm)
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('xb') as stream:
                stream.write(effective)
            diff = ''.join(difflib.unified_diff(original.decode().splitlines(True),
                                               effective.decode().splitlines(True),
                                               fromfile='original/' + relative,
                                               tofile='private/' + relative))
            files.append(dict(relative=relative, original=member(source),
                              effective=member(output), compatibility_diff=diff))
        hp = next(row for row in files if row['relative'] == spec['hparams'])
        require(hp['original']['sha256'] == spec['hparams_sha']
                and hp['effective']['sha256'] == spec['hparams_sha'], 'NATIVE_HPARAMS_EXACT_BYTES')
        result[arm] = dict(root=str(target), arm=arm, namespace=spec['namespace'],
                           original_root=str(root), original_commit=actual,
                           original_clean=True, files=files,
                           hparams=hp['effective'], package_shells=True,
                           original_package_initializers_executed=False)
    receipt = destination / 'adoption.json'
    with receipt.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


@dataclass
class NativeBundle:
    arm: str
    root: Path
    namespace: str
    module: object
    z_module: object
    runningstats: object
    binding: dict


def load_native(binding, arm):
    """Actual native modules under two non-colliding private namespaces."""
    spec = NATIVE_SPECS[arm]
    root = Path(binding['root']).resolve()
    expected = {row['relative']: row for row in binding['files']}
    require(set(expected) == set(source_files(arm)), 'NATIVE_EXACT_15_MEMBER_CLOSURE')
    require(binding['original_commit'] == spec['commit']
            and binding['namespace'] == spec['namespace'], 'NATIVE_SOURCE_ORIGIN_BINDING')
    for relative, row in expected.items():
        path = verify(row['effective'])
        require(path == root / relative, 'NATIVE_PRIVATE_ROOT_IDENTITY')
    prefix = spec['namespace']
    require(not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules),
            'NATIVE_NAMESPACE_ALREADY_LOADED')
    for relative in ('', spec['package'], 'rome', 'util'):
        name = prefix + ('.' + relative if relative else '')
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(root / relative) if relative else str(root)]
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
    main = importlib.import_module(prefix + '.' + spec['package'] + '.' + spec['main'])
    z = importlib.import_module(prefix + '.' + spec['package'] + '.compute_z')
    stats = importlib.import_module(prefix + '.util.runningstats')
    bundle = NativeBundle(arm, root, prefix, main, z, stats, binding)
    actual = closure(bundle)
    require({row['relative'] for row in actual} == {p for p in source_files(arm) if p.endswith('.py')},
            'NATIVE_IMPORTED_CLOSURE_EXACT')
    return bundle


def closure(bundle):
    result = []
    for name, module in sorted(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if name.startswith(bundle.namespace + '.') and file:
            path = Path(file).resolve()
            require(path.is_relative_to(bundle.root), 'NATIVE_IMPORT_ESCAPE')
            result.append(dict(module=name, relative=str(path.relative_to(bundle.root)), **member(path)))
    return result


def parse_hparams(bundle):
    spec = NATIVE_SPECS[bundle.arm]
    cls = getattr(bundle.module, spec['hparams_class'])
    hp = cls.from_json(str(verify(bundle.binding['hparams'])))
    require(hp.layers == spec['layers'] and hp.model_name == 'gpt2-xl'
            and (hp.v_lr, hp.v_num_grad_steps, hp.v_loss_layer, hp.v_weight_decay,
                 hp.clamp_norm_factor, hp.kl_factor) == (.5, 20, 47, .5, .75, .0625),
            'NATIVE_ACTUAL_PARSER_SCIENTIFIC_FIELDS')
    require(hp.fact_token == 'subject_last' and hp.mom2_dataset == 'wikipedia'
            and hp.mom2_n_samples == 100000 and hp.mom2_dtype == 'float32'
            and hp.rewrite_module_tmp == 'transformer.h.{}.mlp.c_proj'
            and hp.layer_module_tmp == 'transformer.h.{}'
            and hp.ln_f_module == 'transformer.ln_f' and hp.lm_head_module == 'lm_head',
            'NATIVE_GPT2_ADAPTER_FIELDS')
    if bundle.arm == 'CAKE':
        require(hp.L2 == 40 and hp.nullspace_threshold == .02 and hp.temperature == .1
                and list(sorted(int(k) for k in hp.causal_scores)) == list(range(5))
                and [hp.causal_scores[str(k)] for k in range(5)] ==
                [.0881231502, .0905135274, .0928233638, .0948726162, .0953092128],
                'CAKE_L2_CAUSAL_NATIVE_SLOTS')
    else:
        require(hp.blue is True and hp.L2 == 80 and hp.nullspace_threshold == .02,
                'BLUE_USER_ALPHAEDIT_BLUE_L2_80_NOT_MEMIT')
    return hp


def native_requests(records):
    requests = []
    for record in records:
        rewrite = record.get('requested_rewrite', record)
        request = copy.deepcopy(rewrite)
        request['case_id'] = record['case_id']
        require(type(request['case_id']) is int
                and isinstance(request.get('target_new'), dict)
                and type(request['target_new'].get('str')) is str
                and request['target_new']['str']
                and all(type(request.get(k)) is str and request[k] for k in ('prompt', 'subject')),
                'CAKE_BLUE_DICT_TARGET_REQUEST_SCHEMA')
        require(request['prompt'].count('{}') == 1 and request['subject'] in
                request['prompt'].format(request['subject']), 'NATIVE_SUBJECT_TEMPLATE')
        requests.append(request)
    require(len(requests) == 100, 'NATIVE_BS100_NO_EXCLUSION')
    return requests


normalize_requests = native_requests


class _LinalgCalls:
    def __init__(self, original, engine):
        self.original, self.engine = original, engine

    def __getattr__(self, name):
        return getattr(self.original, name)

    def solve(self, *args, **kwargs):
        return self.engine._call('solves', self.original.solve, *args, **kwargs)


class _OptimCalls:
    def __init__(self, original, engine):
        self.original, self.engine = original, engine

    def __getattr__(self, name):
        return getattr(self.original, name)

    def Adam(self, *args, **kwargs):
        opt = self.original.Adam(*args, **kwargs)
        step = opt.step

        def counted_step(*a, **k):
            return self.engine._call('fit_updates', step, *a, **k)

        opt.step = counted_step
        return opt


class _TorchCalls:
    def __init__(self, original, engine, optimizer=False):
        self.original = original
        self.linalg = _LinalgCalls(original.linalg, engine)
        self.optim = _OptimCalls(original.optim, engine) if optimizer else original.optim

    def __getattr__(self, name):
        return getattr(self.original, name)


class NativeEngine:
    """One native public apply per batch, same cumulative model, RAM-only H."""
    def __init__(self, bundle, hp, model, tokenizer, projector=None, history=None):
        self.bundle, self.module, self.hp = bundle, bundle.module, hp
        self.model, self.tokenizer, self.arm = model, tokenizer, bundle.arm
        self.writer = 'cake' if self.arm == 'CAKE' else 'alphaedit_blue'
        self.layers = NATIVE_SPECS[self.arm]['layers']
        self.P, self.H = projector, history
        require(self.H is not None and self.P is not None, 'NATIVE_HISTORY_PROJECTOR_ARM')
        self.next_batch, self.current, self.on_progress = 1, None, None
        self.cumulative = {key: 0 for key in COUNTERS}
        self.context_receipt = dict(status='GENERATE_INSIDE_FIRST_REAL_NATIVE_APPLY',
                                    EasyEdit_context_transplanted=False,
                                    source_generator='util/generate.py', generated_by_wrapper=False)
        require(self.module.CONTEXT_TEMPLATES_CACHE is None and not self.module.COV_CACHE,
                'NATIVE_GLOBAL_INITIAL_STATE')
        require(not getattr(self.module, '_ODEEDIT_COUNTERS_BOUND', False), 'NATIVE_ENGINE_ALREADY_BOUND')
        self.original_z, self.original_ks = self.module.compute_z, self.module.compute_ks
        self.native_apply = (self.module.apply_Cake_to_model if self.arm == 'CAKE'
                             else self.module.apply_AlphaEdit_to_model)
        self.module.compute_z, self.module.compute_ks = self._compute_z, self._compute_ks
        self.module.torch = _TorchCalls(torch, self)
        bundle.z_module.torch = _TorchCalls(torch, self, optimizer=True)
        self.module._ODEEDIT_COUNTERS_BOUND = True
        # Native CAKE repr_tools contains a skip-on-invalid branch. Preserve
        # its original calculation but fail closed if any requested row vanished.
        repr_tools = bundle.z_module.repr_tools
        original_reprs = repr_tools.get_reprs_at_idxs
        repr_signature = inspect.signature(original_reprs)

        def checked_reprs(*args, **kwargs):
            bound = repr_signature.bind(*args, **kwargs)
            expected_rows = len(bound.arguments['contexts'])
            require(len(bound.arguments['idxs']) == expected_rows, 'NATIVE_REPR_INPUT_CARDINALITY')
            result = original_reprs(*args, **kwargs)
            values = result if isinstance(result, tuple) else (result,)
            require(all(isinstance(value, torch.Tensor) and value.ndim == 2
                        and value.shape[0] == expected_rows
                        and torch.isfinite(value).all().item() for value in values),
                    'NATIVE_REPR_SILENT_DROP_OR_NONFINITE')
            return result

        repr_tools.get_reprs_at_idxs = checked_reprs

    @property
    def counts(self):
        return dict(self.cumulative)

    @property
    def progress(self):
        return self.on_progress

    @progress.setter
    def progress(self, callback):
        require(callback is None or callable(callback), 'NATIVE_PROGRESS_CALLBACK')
        self.on_progress = callback

    def _call(self, key, original, *args, **kwargs):
        require(self.current is not None, 'NATIVE_COUNTER_OUTSIDE_PUBLIC_APPLY')
        self.current[key] += 1
        self.cumulative[key] += 1
        started = time.monotonic()
        try:
            return original(*args, **kwargs)
        finally:
            self.current['seconds'][key] += time.monotonic() - started

    def _forward(self, *unused):
        self.current['fit_forwards'] += 1
        self.cumulative['fit_forwards'] += 1
        if self.on_progress is not None:
            try:
                self.on_progress(dict(batch=self.current['batch'],
                                      native_z=self.cumulative['native_z'],
                                      native_z_completed=self.cumulative['native_z'],
                                      fit_global_candidate=self.cumulative['fit_forwards'],
                                      fit_updates=self.cumulative['fit_updates']))
            except Exception:
                self.current['logging_callback_errors'] += 1

    def _compute_z(self, *args, **kwargs):
        handle = self.model.register_forward_hook(self._forward)
        try:
            result = self._call('native_z', self.original_z, *args, **kwargs)
            require(torch.isfinite(result).all().item(), 'NATIVE_TARGET_NONFINITE')
            return result
        finally:
            handle.remove()

    def _compute_ks(self, *args, **kwargs):
        require(self.current is not None, 'NATIVE_KEY_OUTSIDE_APPLY')
        total = self.current['write_keys'] + self.current['history_keys']
        key = 'history_keys' if total >= len(self.layers) else 'write_keys'
        result = self._call(key, self.original_ks, *args, **kwargs)
        require(tuple(result.shape) == (100, 6400) and result.dtype == torch.float32
                and torch.isfinite(result).all().item(), 'NATIVE_KEY_FULL_REQUEST_CARDINALITY_NONFINITE')
        return result

    def history(self):
        require(self.H.device.type == 'cpu' and self.H.dtype == torch.float32
                and tuple(self.H.shape) == (len(self.layers), 6400, 6400), 'NATIVE_PHYSICAL_LAYER_HISTORY_SCHEMA')
        return {layer: self.H[index] for index, layer in enumerate(self.layers)}

    def restore_history(self, entries):
        require(set(entries) == set(self.layers),
                'NATIVE_HISTORY_ROLLBACK_LAYER_IDENTITY')
        with torch.no_grad():
            for index, layer in enumerate(self.layers):
                self.H[index].copy_(entries[layer])

    def contexts(self):
        return copy.deepcopy(self.module.CONTEXT_TEMPLATES_CACHE)

    def contexts_snapshot(self):
        return self.contexts()

    context_snapshot = contexts_snapshot

    def restore_contexts(self, contexts):
        self.module.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(contexts)

    restore_context = restore_contexts

    def prepare_contexts(self):
        """Once in the first real cold native input preparation; no fitting.

        Calls the actual native getter/generator, not an EasyEdit transplant.
        The first public apply then finds precisely this native process cache.
        """
        require(self.next_batch == 1 and self.current is None,
                'NATIVE_CONTEXT_ONLY_FIRST_COLD_INPUT_PREPARATION')
        require(self.module.CONTEXT_TEMPLATES_CACHE is None, 'NATIVE_CONTEXT_PREPARE_ONCE')
        before = tuple((name, param.data_ptr(), param._version)
                       for name, param in self.model.named_parameters())
        hooks = {name: (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
                 for name, module in self.model.named_modules()}
        started = time.monotonic()
        contexts = self.module.get_context_templates(self.model, self.tokenizer)
        require(tuple((name, param.data_ptr(), param._version)
                      for name, param in self.model.named_parameters()) == before
                and {name: (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
                     for name, module in self.model.named_modules()} == hooks,
                'NATIVE_CONTEXT_MODEL_NONMUTATION')
        require(isinstance(contexts, list) and len(contexts) == 2
                and contexts[0] == ['{}'] and len(contexts[1]) == 5,
                'NATIVE_CONTEXT_PREPARED_1_PLUS5')
        self.context_receipt.update(status='GENERATED_IN_FIRST_COLD_NATIVE_INPUT_PREPARATION',
                                    contexts_sha256=digest(contexts),
                                    native_input_preparation_seconds=time.monotonic() - started,
                                    native_z_calls=0, public_apply_calls=0,
                                    context_method='actual native get_context_templates/generate_fast')
        return copy.deepcopy(contexts)

    def reset_history_to_uninitialized(self):
        self.H.zero_()

    def apply(self, records, batch):
        require(batch == self.next_batch and 1 <= batch <= 20, 'NATIVE_BATCH_SEQUENCE_NO21')
        requests = native_requests(records)
        self.current = dict(batch=batch, logging_callback_errors=0,
                            seconds={key: 0. for key in COUNTERS}, **{key: 0 for key in COUNTERS})
        started = time.monotonic()
        before = None if self.H is None else (self.H.data_ptr(), self.H._version)
        try:
            result = self._call('public_applies', self.native_apply, self.model, self.tokenizer,
                                requests, self.hp, cache_template=None, cache_c=self.H, P=self.P)
            require(isinstance(result, tuple) and len(result) == 2
                    and result[0] is self.model and result[1] is self.H,
                    'NATIVE_MODEL_RETURNED_HISTORY_IDENTITY')
        finally:
            if before is not None:
                require(self.H.data_ptr() == before[0], 'NATIVE_HISTORY_REPLACED')
                change = self.H._version - before[1]
                require(change >= 0 and change % 2 == 0, 'NATIVE_INDEXED_HISTORY_VERSION')
                self.current['history_appends'] += change // 2
                self.cumulative['history_appends'] += change // 2
        cake = self.arm == 'CAKE'
        require(self.current['native_z'] == (100 if cake else 200)
                and self.current['write_keys'] == (5 if cake else 2)
                and self.current['solves'] == (5 if cake else 2)
                and self.current['history_keys'] == (5 if cake else 2)
                and self.current['history_appends'] == (5 if cake else 2)
                and self.current['public_applies'] == 1, 'NATIVE_EXACT_PUBLIC_CALL_AND_HISTORY_COUNTS')
        if self.H is not None:
            require(torch.isfinite(self.H).all().item(), 'NATIVE_HISTORY_NONFINITE')
        contexts = self.module.CONTEXT_TEMPLATES_CACHE
        require(isinstance(contexts, list) and len(contexts) == 2 and contexts[0] == ['{}']
                and len(contexts[1]) == 5 and all(type(t) is str for group in contexts for t in group),
                'NATIVE_GENERATED_CONTEXT_1_PLUS_5')
        self.context_receipt.update(status='NATIVE_CONTEXT_CREATED_OR_CARRIED',
                                    contexts_sha256=digest(contexts), cold_creation_batch=1,
                                    no_cross_native_context_transplant=True)
        self.next_batch += 1
        receipt = dict(status='NATIVE_APPLY_RETURNED', arm=self.arm, writer=self.writer,
                       batch=batch, requests=100, request_identity=digest(requests),
                       counts=copy.deepcopy(self.current),
                       delta={key: self.current[key] for key in COUNT_FIELDS}, cumulative=self.counts,
                       same_model_returned=True, native_has_history=True,
                       caller_history_appends=0, native_z_disk_cache=False, cache_template=None,
                       returned_second_value_is_native_H=True,
                       context=copy.deepcopy(self.context_receipt), hparams=asdict(self.hp),
                       seconds=time.monotonic() - started,
                       timing_policy='public apply inclusive; z/key/solve/update nested, not additive',
                       solve_timer='CPU dispatch wall only; no added GPU synchronization',
                       checkpoint_saved=False, exact_resume='NOT_AVAILABLE')
        return self.model, receipt


def _bound_assets(config, arm):
    binding = config['native']
    if 'arms' in binding:
        return binding['arms'][arm]
    if arm in binding:
        return binding[arm]
    return binding


def prepare_native(config, model, tokenizer, arm, attempt=None, progress=None):
    binding = _bound_assets(config, arm)
    adoption = binding.get('bundle', config.get('native_bundle', {}).get(arm))
    require(adoption is not None, 'NATIVE_PRIVATE_ADOPTION_BINDING_REQUIRED')
    bundle = load_native(adoption, arm)
    hp = parse_hparams(bundle)
    require(model.config.model_type == 'gpt2' and model.config.n_positions == 1024
            and model.config.n_layer == 48 and model.config.n_embd == 1600,
            'NATIVE_GPT2_MODEL_CONFIG')
    require(model.get_input_embeddings().weight is model.get_output_embeddings().weight,
            'NATIVE_GPT2_TIED_EMBEDDING_HEAD')
    for layer in hp.layers:
        weight = bundle.module.nethook.get_parameter(model, hp.rewrite_module_tmp.format(layer) + '.weight')
        bias = bundle.module.nethook.get_parameter(model, hp.rewrite_module_tmp.format(layer) + '.bias')
        require(tuple(weight.shape) == (6400, 1600) and tuple(bias.shape) == (1600,)
                and weight.dtype == bias.dtype == torch.float32, 'NATIVE_CONV1D_STORED_LAYOUT_AND_BIAS')
    full_projector = torch.load(verify(binding['projector']), map_location='cpu', weights_only=True)
    require(isinstance(full_projector, torch.Tensor) and full_projector.dtype == torch.float32
            and tuple(full_projector.shape) == (5, 6400, 6400)
            and torch.isfinite(full_projector).all().item(), 'EXISTING_P_STACK_PHYSICAL_13_TO17')
    # The new user-selected AlphaEdit-BLUE consumes P in its own [13,17]
    # layer order. Reuse physical slots [0,4] in RAM; no SVD/new disk P.
    projector = full_projector if arm == 'CAKE' else full_projector[[0, 4], :, :].contiguous()
    if arm != 'CAKE':
        del full_projector
    history = torch.zeros((len(hp.layers), 6400, 6400), dtype=torch.float32, device='cpu')
    engine = NativeEngine(bundle, hp, model, tokenizer, projector, history)
    engine.progress = progress

    def blocked_stats(*args, **kwargs):
        raise RuntimeError('NATIVE_C0_CACHE_MISS_RECOMPUTE_FORBIDDEN')

    bundle.module.layer_stats = blocked_stats
    return engine
