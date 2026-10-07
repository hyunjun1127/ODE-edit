"""SHA-pinned GPT-J CAKE/AlphaEdit-BLUE source adapters, not scientific ports.

Only import/absolute-path plumbing is repaired in a private small closure.
Native parsers, generators, fitting, representation hooks, dtype, solves and
public writes remain original. Telemetry observes existing work; it never adds
a fitting or representation forward. H and the read-only mmap P are RAM-only.
"""
import copy
import difflib
import hashlib
import importlib
import importlib.machinery
import inspect
import json
import math
import random
import subprocess
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


LAYERS = [3, 4, 5, 6, 7, 8]
HIDDEN, INTERMEDIATE = 4096, 16384
NATIVE_SPECS = {
    'CAKE': dict(root='/mnt/raid5/janghj/CAKE',
                 commit='0b378234862bd76c69f58404ef84c27d5f4bf9ef',
                 namespace='_odeedit_gptj_cake_native', package='Cake', main='Cake_main',
                 main_sha='d1ae01c7f1d5f6f8af899a589b8e407171c1a5dbb69b8ad6ff08e39f02823e20',
                 hparams='hparams/Cake/EleutherAI_gpt-j-6B.json',
                 hparams_sha='a37d65b4f7b8c9a8ce439abadb4548010777ec0b6547d00d64cea8e12adc22d0',
                 hparams_class='CakeHyperParams', layers=LAYERS, P_slots=list(range(6))),
    'ALPHAEDIT_BLUE': dict(root='/mnt/raid5/janghj/BLUE',
                 commit='311b076a92e4ed0f14f5c8b4909732da781bc5f7',
                 namespace='_odeedit_gptj_blue_native', package='AlphaEdit', main='AlphaEdit_main',
                 main_sha='79da927aad5ab817fd008c5958768adcd00556989a8efbaa2c4bdc80d8fc842e',
                 hparams='hparams/AlphaEdit/EleutherAI_gpt-j-6B-blue.json',
                 hparams_sha='afc9ae1ffd7a6f9d1d6aa04bc2d19b11684bea69ebd51ef0d515d2b073d066e2',
                 hparams_class='AlphaEditHyperParams', layers=[3, 8], P_slots=[0, 5]),
}
SHARED_FILES = ('rome/layer_stats.py', 'rome/repr_tools.py', 'rome/tok_dataset.py',
                'util/generate.py', 'util/globals.py', 'util/hparams.py',
                'util/logit_lens.py', 'util/nethook.py', 'util/runningstats.py')
COUNT_FIELDS = ('native_z', 'write_keys', 'history_keys', 'solves', 'history_appends')
COUNTERS = (*COUNT_FIELDS, 'fit_forwards', 'fit_updates', 'public_applies',
            'repr_calls', 'repr_forwards', 'ln_1_extra_forwards', 'shape_matches')


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
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes']
            and sha(path) == row['sha256'], 'NATIVE_MEMBER_IDENTITY:' + str(path))
    return path


def _sealed_asset(row):
    """Prior exact SHA plus current stat, avoiding another multi-GiB fullhash."""
    path = Path(row['path'])
    require(path.is_file() and not path.is_symlink(), 'NATIVE_ASSET_REGULAR_FILE')
    stat = path.stat()
    require(all(key in row for key in ('sha256', 'bytes', 'inode', 'mtime_ns'))
            and len(row['sha256']) == 64
            and (stat.st_size, stat.st_ino, stat.st_mtime_ns)
            == (row['bytes'], row['inode'], row['mtime_ns']), 'NATIVE_ASSET_STAT_CHANGED')
    return path.resolve()


def source_files(arm):
    spec = NATIVE_SPECS[arm]
    hp = 'Cake_hparams' if arm == 'CAKE' else 'AlphaEdit_hparams'
    return (f'{spec["package"]}/{spec["main"]}.py', f'{spec["package"]}/compute_z.py',
            f'{spec["package"]}/compute_ks.py', f'{spec["package"]}/{hp}.py',
            *SHARED_FILES, 'globals.yml', spec['hparams'])


def effective_source(relative, original, arm):
    """Exact compatibility-only diff; all function/class ASTs remain native."""
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
        require(text.count('with open("globals.yml", "r") as stream:') == 1
                and text.count('    Path(z)\n') == 1, 'NATIVE_GLOBALS_PATH_REPAIR_EXACT')
        text = text.replace('with open("globals.yml", "r") as stream:',
                    'with open(Path(__file__).resolve().parents[1] / "globals.yml", "r") as stream:')
        text = text.replace('    Path(z)\n',
                    '    (Path(z) if Path(z).is_absolute() else Path(__file__).resolve().parents[1] / z)\n')
    return text.encode('utf-8')


def adopt_native(destination):
    """Create-once small source adoption only; originals and large assets read-only."""
    destination = Path(destination).resolve()
    require(not destination.exists(), 'NATIVE_ADOPTION_CREATE_ONCE')
    # Validate every origin before creating any private files.
    for arm, spec in NATIVE_SPECS.items():
        root = Path(spec['root'])
        require(root.is_dir(), 'NATIVE_SOURCE_ROOT_MISSING:' + str(root))
        actual = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'],
                                check=True, capture_output=True, text=True).stdout
        require(actual == spec['commit'] and not status.strip(),
                'NATIVE_ORIGINAL_CLEAN_COMMIT:' + arm)
        for relative in source_files(arm):
            source = root / relative
            require(source.is_file() and not source.is_symlink(),
                    'NATIVE_SOURCE_MISSING_OR_SYMLINK:' + str(source))
        require(sha(root / f'{spec["package"]}/{spec["main"]}.py') == spec['main_sha']
                and sha(root / spec['hparams']) == spec['hparams_sha'],
                'NATIVE_PINNED_MAIN_HPARAMS_SHA:' + arm)
    destination.mkdir(parents=True)
    result = {}
    for arm, spec in NATIVE_SPECS.items():
        root, target = Path(spec['root']), destination / arm
        target.mkdir()
        files = []
        for relative in source_files(arm):
            source = root / relative
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
        result[arm] = dict(root=str(target), arm=arm, namespace=spec['namespace'],
                          original_root=str(root), original_commit=spec['commit'],
                          original_clean=True, files=files, hparams=hp['effective'],
                          package_shells=True, original_package_initializers_executed=False)
    with (destination / 'adoption.json').open('x') as stream:
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
    spec = NATIVE_SPECS[arm]
    root = Path(binding['root']).resolve()
    expected = {row['relative']: row for row in binding['files']}
    require(set(expected) == set(source_files(arm)), 'NATIVE_EXACT_15_MEMBER_CLOSURE')
    require(binding['original_commit'] == spec['commit']
            and binding['namespace'] == spec['namespace'] and binding['arm'] == arm,
            'NATIVE_SOURCE_ORIGIN_BINDING')
    for relative, row in expected.items():
        path = verify(row['effective'])
        require(path == root / relative, 'NATIVE_PRIVATE_ROOT_IDENTITY')
        original = verify(row['original']).read_bytes()
        require(effective_source(relative, original, arm) == path.read_bytes(),
                'NATIVE_COMPATIBILITY_ONLY_EFFECTIVE_SOURCE')
    require(expected[f'{spec["package"]}/{spec["main"]}.py']['original']['sha256']
            == spec['main_sha'] and binding['hparams']['sha256'] == spec['hparams_sha'],
            'NATIVE_PINNED_ORIGIN_SHA')
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
    require({row['relative'] for row in closure(bundle)}
            == {p for p in source_files(arm) if p.endswith('.py')}, 'NATIVE_IMPORTED_CLOSURE_EXACT')
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
    require(hp.layers == spec['layers'] and hp.model_name == 'EleutherAI_gpt-j-6B'
            and (hp.v_lr, hp.v_num_grad_steps, hp.v_loss_layer, hp.v_weight_decay,
                 hp.clamp_norm_factor, hp.kl_factor) == (.5, 25, 27, .5, .75, .0625),
            'NATIVE_ACTUAL_PARSER_SCIENTIFIC_FIELDS')
    require(hp.fact_token == 'subject_last' and hp.layer_selection == 'all'
            and hp.mom2_adjustment is True and hp.mom2_update_weight == 15000
            and hp.mom2_dataset == 'wikipedia' and hp.mom2_n_samples == 100000
            and hp.mom2_dtype == 'float32' and hp.nullspace_threshold == .02
            and hp.rewrite_module_tmp == 'transformer.h.{}.mlp.fc_out'
            and hp.layer_module_tmp == 'transformer.h.{}'
            and hp.mlp_module_tmp == 'transformer.h.{}.mlp'
            and hp.attn_module_tmp == 'transformer.h.{}.attn'
            and hp.ln_f_module == 'transformer.ln_f' and hp.lm_head_module == 'lm_head',
            'NATIVE_GPTJ_ADAPTER_FIELDS')
    if bundle.arm == 'CAKE':
        require(hp.L2 == 30 and hp.temperature == .1
                and sorted(hp.causal_scores) == [str(i) for i in range(6)]
                and [hp.causal_scores[str(i)] for i in range(6)] ==
                [.3301409185, .3326811492, .3301301003, .3262162805, .3117056787, .3029971123],
                'CAKE_GPTJ_L2_30_CAUSAL_NATIVE_SLOTS')
    else:
        require(hp.blue is True and hp.L2 == 95, 'BLUE_GPTJ_L2_95_NOT_MEMIT')
    return hp


def native_requests(records):
    """Schema-only deep copy; native execute owns the leading-space rule."""
    requests = []
    for record in records:
        request = copy.deepcopy(record.get('requested_rewrite', record))
        request['case_id'] = record['case_id']
        require(type(request['case_id']) is int
                and isinstance(request.get('target_new'), dict)
                and type(request['target_new'].get('str')) is str
                and request['target_new']['str']
                and all(type(request.get(k)) is str and request[k] for k in ('prompt', 'subject')),
                'CAKE_BLUE_DICT_TARGET_REQUEST_SCHEMA')
        require(request['prompt'].count('{}') == 1, 'NATIVE_SUBJECT_TEMPLATE')
        requests.append(request)
    require(len(requests) == 100 and len({r['case_id'] for r in requests}) == 100,
            'NATIVE_BS100_OCCURRENCE_IDENTITY')
    return requests


normalize_requests = native_requests


def expected_counts(arm):
    layers = len(NATIVE_SPECS[arm]['layers'])
    return dict(native_z=100 if arm == 'CAKE' else 200, write_keys=layers,
                history_keys=layers, solves=layers, history_appends=layers)


class ProjectorSlots:
    """Read-only virtual indexing: BLUE local sites 0/1 use physical slots 0/5."""
    def __init__(self, full, slots):
        self.full, self.slots = full, tuple(slots)
        self.shape = (len(slots), *full.shape[1:])
        self.dtype = full.dtype
        self.device = full.device

    def __getitem__(self, index):
        index = (index,) if isinstance(index, int) else index
        require(isinstance(index, tuple) and type(index[0]) is int
                and 0 <= index[0] < len(self.slots), 'NATIVE_P_LOCAL_SLOT_INDEX')
        return self.full[(self.slots[index[0]], *index[1:])]


class _LinalgCalls:
    def __init__(self, original, engine):
        self.original, self.engine = original, engine

    def __getattr__(self, name):
        return getattr(self.original, name)

    def solve(self, *args, **kwargs):
        result = self.engine._call('solves', self.original.solve, *args, **kwargs)
        require(isinstance(result, torch.Tensor) and result.dtype == torch.float32
                and tuple(result.shape) == (INTERMEDIATE, HIDDEN)
                and torch.isfinite(result).all().item(), 'NATIVE_GPTJ_SOLVE_SHAPE_NONFINITE')
        return result


class _OptimCalls:
    def __init__(self, original, engine):
        self.original, self.engine = original, engine

    def __getattr__(self, name):
        return getattr(self.original, name)

    def Adam(self, *args, **kwargs):
        optimizer = self.original.Adam(*args, **kwargs)
        step = optimizer.step

        def counted_step(*a, **k):
            return self.engine._call('fit_updates', step, *a, **k)

        optimizer.step = counted_step
        return optimizer


class _TorchCalls:
    def __init__(self, original, engine, optimizer=False):
        self.original = original
        self.linalg = _LinalgCalls(original.linalg, engine)
        self.optim = _OptimCalls(original.optim, engine) if optimizer else original.optim

    def __getattr__(self, name):
        return getattr(self.original, name)


class NativeEngine:
    """Exactly one native public apply per batch; same model and independent H."""
    def __init__(self, bundle, hp, model, tokenizer, projector, history, config=None):
        self.bundle, self.module, self.hp = bundle, bundle.module, hp
        self.model, self.tokenizer, self.arm = model, tokenizer, bundle.arm
        self.writer = 'cake' if self.arm == 'CAKE' else 'alphaedit_blue'
        self.layers = NATIVE_SPECS[self.arm]['layers']
        self.P, self.H, self.config = projector, history, config or {}
        require(self.H is not None and self.P is not None, 'NATIVE_HISTORY_PROJECTOR_ARM')
        self.next_batch, self.current, self.on_progress = 1, None, None
        self.cumulative = {key: 0 for key in COUNTERS}
        self._timer_stack = []
        self.context_receipt = dict(status='GENERATE_INSIDE_FIRST_COLD_NATIVE_INPUT_PREPARATION',
                EasyEdit_context_transplanted=False, source_generator='util/generate.py',
                generated_by_wrapper=False, arm=self.arm, cold_seed=self.config.get('seed'),
                native_generator_calls=0, RNG_restored=False)
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
        self._bind_representations()
        original_match = self.module.upd_matrix_match_shape

        def matched(*args, **kwargs):
            result = self._call('shape_matches', original_match, *args, **kwargs)
            require(isinstance(result, torch.Tensor) and result.dtype == torch.float32
                    and tuple(result.shape) == (HIDDEN, INTERMEDIATE)
                    and torch.isfinite(result).all().item(), 'NATIVE_GPTJ_STORED_UPDATE_SHAPE_NONFINITE')
            self.current['writes'].append(dict(layer=self.layers[len(self.current['writes'])],
                delta_norm=float(torch.linalg.vector_norm(result)), stored_shape=list(result.shape)))
            return result

        self.module.upd_matrix_match_shape = matched

    @property
    def counts(self):
        return dict(self.cumulative)

    def expected_counts(self):
        return expected_counts(self.arm)

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
        timer = dict(start=time.monotonic(), child=0.)
        self._timer_stack.append(timer)
        try:
            return original(*args, **kwargs)
        finally:
            elapsed = time.monotonic() - timer['start']
            self.current['seconds'][key] += elapsed
            self.current['exclusive_seconds'][key] += max(0., elapsed - timer['child'])
            self._timer_stack.pop()
            if self._timer_stack:
                self._timer_stack[-1]['child'] += elapsed

    def _increment(self, key):
        self.current[key] += 1
        self.cumulative[key] += 1

    def _emit(self, row):
        if self.on_progress is None:
            return
        started = time.monotonic()
        try:
            self.on_progress(row)
        except Exception:
            self.current['logging_callback_errors'] += 1
        finally:
            self.current['telemetry_seconds'] += time.monotonic() - started

    def _forward(self, *unused):
        self._increment('fit_forwards')
        self._emit(dict(batch=self.current['batch'], native_z=self.cumulative['native_z'],
            native_z_completed=self.cumulative['native_z'] - 1,
            fit_global_candidate=self.cumulative['fit_forwards'], fit_updates=self.cumulative['fit_updates']))

    def _compute_z(self, *args, **kwargs):
        observed, previous = {}, sys.getprofile()
        before = (self.current['fit_forwards'], self.current['fit_updates'])
        signature = inspect.signature(self.original_z).bind(*args, **kwargs)
        layer = signature.arguments['layer']

        def trace(frame, event, unused):
            if event == 'return' and frame.f_code is self.original_z.__code__:
                observed.update({key: frame.f_locals[key] for key in
                    ('it', 'loss', 'nll_loss', 'kl_loss', 'weight_decay') if key in frame.f_locals})
            if previous is not None:
                previous(frame, event, unused)

        handle = self.model.register_forward_hook(self._forward)
        sys.setprofile(trace)
        try:
            result = self._call('native_z', self.original_z, *args, **kwargs)
        finally:
            sys.setprofile(previous)
            handle.remove()
        require(isinstance(result, torch.Tensor) and result.dtype == torch.float32
                and tuple(result.shape) == (HIDDEN,) and torch.isfinite(result).all().item(),
                'NATIVE_TARGET_SHAPE_NONFINITE')
        require(set(observed) == {'it', 'loss', 'nll_loss', 'kl_loss', 'weight_decay'},
                'NATIVE_FIT_RETURN_LOCAL_TELEMETRY')
        evaluations, updates = int(observed.pop('it')) + 1, self.current['fit_updates'] - before[1]
        require(1 <= evaluations <= 25 and updates == evaluations - 1
                and self.current['fit_forwards'] - before[0] == evaluations,
                'NATIVE_FIT_25_EVALUATIONS_24_UPDATES')
        row = dict(site=layer, request_index=self.current['native_z'],
                   evaluations=evaluations, Adam_updates=updates)
        row.update({key: float(value.detach()) for key, value in observed.items()})
        require(all(math.isfinite(value) for value in row.values()), 'NATIVE_NONFINITE_FIT_TRACE')
        row['stop'] = 'TOTAL_LOSS_BELOW_005' if row['loss'] < .05 else 'BUDGET_EXHAUSTED'
        self.current['fit_trace'].append(row)
        self._emit(dict(batch=self.current['batch'], native_z=self.cumulative['native_z'],
            native_z_completed=self.cumulative['native_z'], native_z_seconds=self.current['seconds']['native_z'],
            fit_global_candidate=self.cumulative['fit_forwards'], fit_updates=self.cumulative['fit_updates'],
            **row))
        return result

    def _compute_ks(self, *args, **kwargs):
        total = self.current['write_keys'] + self.current['history_keys']
        key = 'history_keys' if total >= len(self.layers) else 'write_keys'
        result = self._call(key, self.original_ks, *args, **kwargs)
        require(isinstance(result, torch.Tensor) and tuple(result.shape) == (100, INTERMEDIATE)
                and result.dtype == torch.float32 and torch.isfinite(result).all().item(),
                'NATIVE_KEY_FULL_REQUEST_CARDINALITY_NONFINITE')
        return result

    def _bind_representations(self):
        repr_tools = self.bundle.z_module.repr_tools
        original = repr_tools.get_reprs_at_idxs
        signature = inspect.signature(original)
        trace_code = self.module.nethook.Trace.__init__.__code__

        def checked_reprs(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            expected_rows = len(bound.arguments['contexts'])
            require(len(bound.arguments['idxs']) == expected_rows, 'NATIVE_REPR_INPUT_CARDINALITY')
            extra, previous = {'next': False, 'calls': 0}, sys.getprofile()

            def profile(frame, event, unused):
                if event == 'return' and frame.f_code is trace_code:
                    layer = frame.f_locals.get('layer')
                    if isinstance(layer, str) and layer.endswith('.ln_1'):
                        extra['next'] = True
                if previous is not None:
                    previous(frame, event, unused)

            def forward(*unused):
                self._increment('repr_forwards')
                if extra['next']:
                    self._increment('ln_1_extra_forwards')
                    extra['calls'] += 1
                    extra['next'] = False

            handle = self.model.register_forward_hook(forward)
            sys.setprofile(profile)
            try:
                result = self._call('repr_calls', original, *args, **kwargs)
            finally:
                sys.setprofile(previous)
                handle.remove()
            values = result if isinstance(result, tuple) else (result,)
            require(all(isinstance(value, torch.Tensor) and value.ndim == 2
                        and value.shape[0] == expected_rows and torch.isfinite(value).all().item()
                        for value in values), 'NATIVE_REPR_SILENT_DROP_OR_NONFINITE')
            name = bound.arguments['module_template'].format(bound.arguments['layer'])
            model = bound.arguments['model']
            actual_gptj = isinstance(model, repr_tools.GPTJForCausalLM)
            expected_extra = ((expected_rows + 127) // 128 if actual_gptj
                and bound.arguments['track'] in ('in', 'both')
                and name in ([f'transformer.h.{layer}' for layer in LAYERS]
                             if self.arm == 'CAKE' else ['transformer.h.8']) else 0)
            require(extra['calls'] == expected_extra, 'NATIVE_GPTJ_LN1_CAPTURE_FORWARD_COUNT')
            return result

        repr_tools.get_reprs_at_idxs = checked_reprs

    def history(self):
        require(self.H.device.type == 'cpu' and self.H.dtype == torch.float32
                and tuple(self.H.shape) == (len(self.layers), INTERMEDIATE, INTERMEDIATE),
                'NATIVE_PHYSICAL_LAYER_HISTORY_SCHEMA')
        return {layer: self.H[index] for index, layer in enumerate(self.layers)}

    def restore_history(self, entries):
        require(set(entries) == set(self.layers), 'NATIVE_HISTORY_ROLLBACK_LAYER_IDENTITY')
        with torch.no_grad():
            for index, layer in enumerate(self.layers):
                self.H[index].copy_(entries[layer])

    def contexts(self):
        return copy.deepcopy(self.module.CONTEXT_TEMPLATES_CACHE)

    contexts_snapshot = contexts
    context_snapshot = contexts

    def restore_contexts(self, contexts):
        self.module.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(contexts)

    restore_context = restore_contexts

    def prepare_contexts(self):
        """Actual cold native generator once, restored RNG, no fit or H update."""
        require(self.next_batch == 1 and self.current is None, 'NATIVE_CONTEXT_FIRST_COLD_ONLY')
        require(self.module.CONTEXT_TEMPLATES_CACHE is None, 'NATIVE_CONTEXT_PREPARE_ONCE')
        before = tuple((name, param.data_ptr(), param._version)
                       for name, param in self.model.named_parameters())
        hooks = {name: (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
                 for name, module in self.model.named_modules()}
        h_before = (self.H.data_ptr(), self.H._version)
        rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state(),
               torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
        calls, original = [], self.module.generate_fast

        def generate(*args, **kwargs):
            calls.append(dict(prompts=len(args[2]), n_gen_per_prompt=kwargs['n_gen_per_prompt'],
                              max_out_len=kwargs['max_out_len']))
            return original(*args, **kwargs)

        started = time.monotonic()
        self.module.generate_fast = generate
        try:
            contexts = self.module.get_context_templates(self.model, self.tokenizer)
        finally:
            self.module.generate_fast = original
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.random.set_rng_state(rng[2])
            if rng[3] is not None:
                torch.cuda.set_rng_state_all(rng[3])
        require(tuple((name, param.data_ptr(), param._version)
                      for name, param in self.model.named_parameters()) == before
                and {name: (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
                     for name, module in self.model.named_modules()} == hooks
                and (self.H.data_ptr(), self.H._version) == h_before, 'NATIVE_CONTEXT_NONMUTATION')
        self._check_contexts(contexts)
        require(calls == [dict(prompts=5, n_gen_per_prompt=1, max_out_len=10)],
                'NATIVE_CONTEXT_GENERATOR_ARGUMENTS')
        self.context_receipt.update(status='GENERATED_IN_FIRST_COLD_NATIVE_INPUT_PREPARATION',
                contexts_sha256=digest(contexts), native_generator_calls=1, native_context_calls=calls,
                native_input_preparation_seconds=time.monotonic() - started,
                native_z_calls=0, public_apply_calls=0, history_appends=0, RNG_restored=True,
                model_asset_identity=self.config.get('model_asset_identity'),
                runtime=self.config.get('runtime'),
                generator_source=member(Path(self.bundle.root) / 'util/generate.py'),
                native_getter_source=member(Path(self.bundle.root) /
                    f'{NATIVE_SPECS[self.arm]["package"]}/{NATIVE_SPECS[self.arm]["main"]}.py'))
        return copy.deepcopy(contexts)

    @staticmethod
    def _check_contexts(contexts):
        require(isinstance(contexts, list) and len(contexts) == 2 and contexts[0] == ['{}']
                and len(contexts[1]) == 5 and all(type(t) is str for group in contexts for t in group),
                'NATIVE_GENERATED_CONTEXT_1_PLUS_5')

    def reset_history_to_uninitialized(self):
        self.H.zero_()

    def apply(self, records, batch):
        require(batch == self.next_batch and 1 <= batch <= 20, 'NATIVE_BATCH_SEQUENCE_NO21')
        requests = native_requests(records)
        self._check_contexts(self.module.CONTEXT_TEMPLATES_CACHE)
        self.current = dict(batch=batch, logging_callback_errors=0, telemetry_seconds=0.,
            fit_trace=[], writes=[], seconds={key: 0. for key in COUNTERS},
            exclusive_seconds={key: 0. for key in COUNTERS}, **{key: 0 for key in COUNTERS})
        started = time.monotonic()
        before = (self.H.data_ptr(), self.H._version)
        context_before = digest(self.module.CONTEXT_TEMPLATES_CACHE)
        try:
            result = self._call('public_applies', self.native_apply, self.model, self.tokenizer,
                                requests, self.hp, cache_template=None, cache_c=self.H, P=self.P)
            require(isinstance(result, tuple) and len(result) == 2
                    and result[0] is self.model and result[1] is self.H,
                    'NATIVE_MODEL_RETURNED_HISTORY_IDENTITY')
            self.model, self.H = result
        finally:
            require(self.H.data_ptr() == before[0], 'NATIVE_HISTORY_REPLACED')
            change = self.H._version - before[1]
            require(change >= 0 and change % 2 == 0, 'NATIVE_INDEXED_HISTORY_VERSION')
            self.current['history_appends'] += change // 2
            self.cumulative['history_appends'] += change // 2
            self.current['history_tensor_version_delta'] = change
        require({key: self.current[key] for key in COUNT_FIELDS} == self.expected_counts()
                and self.current['public_applies'] == 1
                and self.current['shape_matches'] == len(self.layers),
                'NATIVE_EXACT_PUBLIC_CALL_AND_HISTORY_COUNTS')
        require(context_before == digest(self.module.CONTEXT_TEMPLATES_CACHE),
                'NATIVE_CONTEXT_CONTINUITY')
        for plane in self.H:
            require(torch.isfinite(plane).all().item(), 'NATIVE_HISTORY_NONFINITE')
        self.context_receipt.update(status='NATIVE_CONTEXT_CREATED_OR_CARRIED', cold_creation_batch=1)
        self.next_batch += 1
        receipt = dict(status='NATIVE_APPLY_RETURNED', arm=self.arm, writer=self.writer,
            batch=batch, requests=100, request_identity=digest(requests),
            counts=copy.deepcopy(self.current), delta={key: self.current[key] for key in COUNT_FIELDS},
            cumulative=self.counts, same_model_returned=True, native_has_history=True,
            caller_history_appends=0, native_z_disk_cache=False, cache_template=None,
            returned_second_value_is_native_H=True,
            history_continuity=dict(same_storage=True, tensor_version_before=before[1],
                tensor_version_after=self.H._version, physical_layers=self.layers,
                P_physical_slots=NATIVE_SPECS[self.arm]['P_slots']),
            context=copy.deepcopy(self.context_receipt), hparams=asdict(self.hp),
            seconds=time.monotonic() - started,
            timing_policy='CPU dispatch inclusive and nested-exclusive; no added timing synchronization; safety/scalar guards synchronize',
            checkpoint_saved=False, exact_resume='NOT_AVAILABLE')
        return self.model, receipt


CakeBlueEngine = NativeEngine


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
    require(config['native'].get('context_reuse') is None, 'NATIVE_CONTEXT_NO_UNVERIFIED_REUSE')
    require(model.config.model_type == 'gptj' and model.config.n_layer == 28
            and model.config.n_embd == HIDDEN and model.config.vocab_size == 50400
            and getattr(model.config, 'rotary_dim', None) == 64,
            'NATIVE_GPTJ_MODEL_CONFIG')
    require(model.get_input_embeddings().weight is not model.get_output_embeddings().weight,
            'NATIVE_GPTJ_UNTIED_HEAD')
    require(tuple(model.get_output_embeddings().bias.shape) == (50400,), 'NATIVE_GPTJ_HEAD_BIAS')
    for layer in hp.layers:
        module = bundle.module.nethook.get_module(model, hp.rewrite_module_tmp.format(layer))
        require(isinstance(module, torch.nn.Linear) and module.bias is None
                and tuple(module.weight.shape) == (HIDDEN, INTERMEDIATE)
                and module.weight.dtype == torch.float32, 'NATIVE_GPTJ_LINEAR_FC_OUT_LAYOUT')
    require(all(param.dtype == torch.float32 for param in model.parameters()), 'NATIVE_GPTJ_FP32_MODEL')
    full_projector = torch.load(_sealed_asset(binding['projector']), map_location='cpu',
                                weights_only=True, mmap=True)
    require(isinstance(full_projector, torch.Tensor) and full_projector.dtype == torch.float32
            and tuple(full_projector.shape) == (6, INTERMEDIATE, INTERMEDIATE),
            'EXISTING_GPTJ_P_STACK_PHYSICAL_3_TO8')
    for slot in NATIVE_SPECS[arm]['P_slots']:
        require(torch.isfinite(full_projector[slot]).all().item(), 'NATIVE_P_NONFINITE')
    projector = ProjectorSlots(full_projector, NATIVE_SPECS[arm]['P_slots'])
    history = torch.zeros((len(hp.layers), INTERMEDIATE, INTERMEDIATE),
                          dtype=torch.float32, device='cpu')
    engine = NativeEngine(bundle, hp, model, tokenizer, projector, history, config)
    engine.progress = progress
    # Existing C0 is bound for native namespace provenance; these two native
    # writers never call get_cov. A cache miss must never start collection.
    stats_dir = config['native'].get('stats_dir', '/mnt/raid5/janghj/EasyEdit/examples/data/stats')
    bundle.module.STATS_DIR = Path(stats_dir).resolve()
    importlib.import_module(bundle.namespace + '.util.globals').STATS_DIR = bundle.module.STATS_DIR

    def blocked_stats(*args, **kwargs):
        raise RuntimeError('NATIVE_C0_CACHE_MISS_RECOMPUTE_FORBIDDEN')

    bundle.module.layer_stats = blocked_stats
    return engine
