"""SHA-bound native GPT-J MEMIT PRUNE/RECT, with one explicit terminal repair.

Native fitting, dense lower-layer planning, restore, and public writes execute
the adopted original modules. Only import/path plumbing differs in that small
closure. PRUNE_TERMINAL_BASE_FIX changes the terminal base from edited W20 to
RAM-only cold W0; the original torch.svd/threshold/log formula is preserved.
No model/history/RNG/reconstruction payload is written by this adapter.
"""
import copy
import difflib
import hashlib
import importlib
import importlib.machinery
import inspect
import json
import math
import subprocess
import sys
import time
import types
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import torch


LAYERS = (3, 4, 5, 6, 7, 8)
BASE_FIX = 'PRUNE_TERMINAL_BASE_FIX'
ROOT = '/mnt/raid5/janghj/BLUE'
COMMIT = '311b076a92e4ed0f14f5c8b4909732da781bc5f7'
HPARAMS = 'hparams/MEMIT/EleutherAI_gpt-j-6B.json'
HPARAMS_SHA = 'e2430efa658a26f11ae9a5836951e0453fcee34fc0d952a5d05057b5e796e86f'
TERMINAL_SOURCE_SHA = '7a4ece770894a55cbf029cef38f306dfc5b8b06f623239cb5afce5b3cdfc0b80'
NATIVE_SPECS = {
    'PRUNE': dict(root=ROOT, commit=COMMIT, package='memit', main='memit_main',
                  namespace='_odeedit_gptj_prune_native', layers=list(LAYERS),
                  hparams=HPARAMS, hparams_sha=HPARAMS_SHA,
                  main_sha='943a0da1a758692fc4b38c5d1203650795f441a3952df7a4814ffe40ee2567be'),
    'RECT': dict(root=ROOT, commit=COMMIT, package='memit', main='memit_rect_main',
                 namespace='_odeedit_gptj_rect_native', layers=list(LAYERS),
                 hparams=HPARAMS, hparams_sha=HPARAMS_SHA,
                 main_sha='8e199fd49e744f6c08f0edd7933ab683df1970a5dcc9da41570e4368073ec597'),
}
SHARED_FILES = ('rome/layer_stats.py', 'rome/repr_tools.py', 'rome/tok_dataset.py',
                'util/generate.py', 'util/globals.py', 'util/hparams.py',
                'util/logit_lens.py', 'util/nethook.py', 'util/runningstats.py')
COUNT_FIELDS = ('native_z', 'write_keys', 'history_keys', 'solves', 'history_appends')
COUNTERS = (*COUNT_FIELDS, 'fit_forwards', 'fit_updates', 'public_applies',
            'executes', 'dense_provisional_writes', 'public_writes',
            'repr_forwards', 'repr_extra_capture_forwards')


def require(condition, label):
    if not condition:
        raise RuntimeError(label)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


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
    return (f'memit/{spec["main"]}.py', 'memit/compute_z.py', 'memit/compute_ks.py',
            'memit/memit_hparams.py', *SHARED_FILES, 'globals.yml', HPARAMS)


def effective_source(relative, original, arm):
    """Import-only exact compatibility edits, never numerical edits."""
    if not relative.endswith('.py'):
        return original
    text = original.decode('utf-8')
    namespace = NATIVE_SPECS[arm]['namespace']
    for package in ('util', 'rome'):
        text = text.replace(f'from {package} ', f'from {namespace}.{package} ')
        text = text.replace(f'from {package}.', f'from {namespace}.{package}.')
    if relative == 'util/globals.py':
        require(text.count('with open("globals.yml", "r") as stream:') == 1,
                'NATIVE_GLOBALS_PATH_REPAIR_EXACT')
        text = text.replace('with open("globals.yml", "r") as stream:',
                            'with open(Path(__file__).resolve().parents[1] / "globals.yml", "r") as stream:')
        require(text.count('    Path(z)\n') == 1, 'NATIVE_RELATIVE_PATH_REPAIR_EXACT')
        text = text.replace('    Path(z)\n',
                            '    (Path(z) if Path(z).is_absolute() else Path(__file__).resolve().parents[1] / z)\n')
    return text.encode('utf-8')


def terminal_source_diff(original):
    """The separate scientific repair is explicit, not an import-only claim."""
    text = original.decode('utf-8')
    before = 'adjusted_weight = original_weight + upd_matrix[k]'
    after = 'adjusted_weight = weights_copy[k].to(original_weight.device) + upd_matrix[k]'
    require(text.count(before) == 1, 'PRUNE_NATIVE_TERMINAL_BASE_EXACT')
    return ''.join(difflib.unified_diff(text.splitlines(True),
                   text.replace(before, after).splitlines(True),
                   fromfile='original/experiments/evaluate.py',
                   tofile='task-owned/PRUNE_TERMINAL_BASE_FIX'))


def adopt_native(destination):
    """Create-once small closure preparation; never copy models, P, H or stats."""
    destination = Path(destination).resolve()
    require(not destination.exists(), 'NATIVE_ADOPTION_CREATE_ONCE')
    root = Path(ROOT)
    actual = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                            check=True, capture_output=True, text=True).stdout.strip()
    require(actual == COMMIT, 'NATIVE_ORIGINAL_COMMIT')
    for arm, spec in NATIVE_SPECS.items():
        require(sha(root / f'memit/{spec["main"]}.py') == spec['main_sha'],
                'NATIVE_ORIGINAL_MAIN_SHA:' + arm)
    require(sha(root / HPARAMS) == HPARAMS_SHA, 'NATIVE_ORIGINAL_HPARAMS_SHA')
    terminal = member(root / 'experiments/evaluate.py')
    require(terminal['sha256'] == TERMINAL_SOURCE_SHA, 'NATIVE_TERMINAL_SOURCE_SHA')
    terminal['base_fix_exactdiff'] = terminal_source_diff((root / 'experiments/evaluate.py').read_bytes())
    destination.mkdir(parents=True)
    result = {}
    for arm, spec in NATIVE_SPECS.items():
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
            difference = ''.join(difflib.unified_diff(original.decode().splitlines(True),
                                 effective.decode().splitlines(True),
                                 fromfile='original/' + relative, tofile='private/' + relative))
            files.append(dict(relative=relative, original=member(source),
                              effective=member(output), compatibility_diff=difference))
        hp = next(row for row in files if row['relative'] == HPARAMS)
        result[arm] = dict(root=str(target), arm=arm, namespace=spec['namespace'],
                           original_root=str(root), original_commit=actual, files=files,
                           hparams=hp['effective'], package_shells=True,
                           original_package_initializers_executed=False,
                           terminal_source=copy.deepcopy(terminal),
                           explicit_repair=BASE_FIX if arm == 'PRUNE' else None)
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
            and binding['namespace'] == spec['namespace'], 'NATIVE_SOURCE_ORIGIN_BINDING')
    for relative, row in expected.items():
        path = verify(row['effective'])
        require(path == root / relative, 'NATIVE_PRIVATE_ROOT_IDENTITY')
        require(row['effective']['sha256'] == hashlib.sha256(effective_source(
            relative, verify(row['original']).read_bytes(), arm)).hexdigest(),
            'NATIVE_EFFECTIVE_ONLY_ALLOWED_IMPORT_DIFF:' + relative)
    require(expected[f'memit/{spec["main"]}.py']['original']['sha256'] == spec['main_sha']
            and expected[HPARAMS]['original']['sha256'] == HPARAMS_SHA,
            'NATIVE_CONTRACT_SOURCE_SHA')
    require(binding['terminal_source']['sha256'] == TERMINAL_SOURCE_SHA,
            'NATIVE_TERMINAL_SOURCE_BINDING')
    prefix = spec['namespace']
    require(not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules),
            'NATIVE_NAMESPACE_ALREADY_LOADED')
    for relative in ('', 'memit', 'rome', 'util'):
        name = prefix + ('.' + relative if relative else '')
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(root / relative) if relative else str(root)]
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
    main = importlib.import_module(prefix + '.memit.' + spec['main'])
    z = importlib.import_module(prefix + '.memit.compute_z')
    stats = importlib.import_module(prefix + '.util.runningstats')
    bundle = NativeBundle(arm, root, prefix, main, z, stats, binding)
    require({row['relative'] for row in closure(bundle)}
            == {p for p in source_files(arm) if p.endswith('.py')}, 'NATIVE_IMPORTED_CLOSURE_EXACT')
    return bundle


def closure(bundle):
    result = []
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith(bundle.namespace + '.') and path:
            path = Path(path).resolve()
            require(path.is_relative_to(bundle.root), 'NATIVE_IMPORT_ESCAPE')
            result.append(dict(module=name, relative=str(path.relative_to(bundle.root)), **member(path)))
    return result


def parse_hparams(bundle):
    hp = bundle.module.MEMITHyperParams.from_json(str(verify(bundle.binding['hparams'])))
    require(hp.model_name == 'EleutherAI_gpt-j-6B' and hp.layers == list(LAYERS)
            and (hp.v_lr, hp.v_num_grad_steps, hp.v_loss_layer, hp.v_weight_decay,
                 hp.clamp_norm_factor, hp.kl_factor) == (.5, 25, 27, .5, .75, .0625),
            'NATIVE_ACTUAL_PARSER_SCIENTIFIC_FIELDS')
    require(hp.blue is False and hp.edit_layer == -1 and hp.mom2_adjustment is True
            and hp.mom2_update_weight == 15000 and hp.fact_token == 'subject_last'
            and hp.mom2_dataset == 'wikipedia' and hp.mom2_n_samples == 100000
            and hp.mom2_dtype == 'float32', 'NATIVE_PRUNE_RECT_C15000_NO_BLUE_NO_HP')
    require(hp.rewrite_module_tmp == 'transformer.h.{}.mlp.fc_out'
            and hp.layer_module_tmp == 'transformer.h.{}'
            and hp.mlp_module_tmp == 'transformer.h.{}.mlp'
            and hp.attn_module_tmp == 'transformer.h.{}.attn'
            and hp.ln_f_module == 'transformer.ln_f' and hp.lm_head_module == 'lm_head',
            'NATIVE_GPTJ_ADAPTER_FIELDS')
    return hp


def native_requests(records):
    requests = []
    for record in records:
        rewrite = record.get('requested_rewrite', record)
        request = copy.deepcopy(rewrite)
        request['case_id'] = record['case_id']
        require(type(request['case_id']) is int
                and isinstance(request.get('target_new'), dict)
                and type(request['target_new'].get('str')) is str and request['target_new']['str']
                and all(type(request.get(key)) is str and request[key] for key in ('prompt', 'subject')),
                'NATIVE_DICT_TARGET_REQUEST_SCHEMA')
        require(request['prompt'].count('{}') == 1, 'NATIVE_SUBJECT_TEMPLATE')
        requests.append(request)
    require(len(requests) == 100, 'NATIVE_BS100_NO_EXCLUSION')
    return requests


normalize_requests = native_requests


def native_terminal_compression(cold_weight, dense_weight):
    """Original spectral operation, plus the named cold-base repair.

    Both tensors have the native storage shape, FP32, and the same device in
    actual runs. CPU controls execute this same function, never a GPU fit.
    """
    require(cold_weight.shape == dense_weight.shape and cold_weight.dtype == dense_weight.dtype
            and cold_weight.device == dense_weight.device, 'PRUNE_TERMINAL_NATIVE_TENSOR_LAYOUT')
    require(torch.isfinite(cold_weight).all().item() and torch.isfinite(dense_weight).all().item(),
            'PRUNE_TERMINAL_NONFINITE_INPUT')
    with torch.no_grad():
        update = dense_weight - cold_weight
        _, original_singular, _ = torch.svd(cold_weight)
        max_sigma = original_singular.max().item()
        update_u, update_singular, update_v = torch.svd(update)
        adjusted_singular = torch.where(
            update_singular > max_sigma,
            torch.log(update_singular)
            - torch.log(torch.tensor(max_sigma, device=dense_weight.device)) + max_sigma,
            update_singular,
        )
        compressed_update = torch.matmul(update_u, torch.matmul(
            torch.diag(adjusted_singular), update_v.t()))
        final_weight = cold_weight + compressed_update
        require(torch.isfinite(final_weight).all().item(), 'PRUNE_TERMINAL_NONFINITE_RESULT')
        stats = dict(native_svd_calls=2, max_sigma_cold=float(max_sigma),
                     max_sigma_update=float(update_singular.max().item()),
                     max_sigma_compressed=float(adjusted_singular.max().item()),
                     singular_values=int(update_singular.numel()),
                     singular_values_compressed=int((update_singular > max_sigma).sum().item()),
                     dense_delta_norm=float(torch.linalg.vector_norm(update).item()),
                     compressed_delta_norm=float(torch.linalg.vector_norm(compressed_update).item()),
                     terminal_change_norm=float(torch.linalg.vector_norm(final_weight - dense_weight).item()),
                     cold_weight_norm=float(torch.linalg.vector_norm(cold_weight).item()),
                     final_weight_norm=float(torch.linalg.vector_norm(final_weight).item()))
    return final_weight, stats


def native_rect_mask_stats(flat_scores, k, threshold):
    """Observe the already computed native scores, including threshold ties."""
    require(flat_scores.ndim == 1 and flat_scores.numel() > 1
            and k == int(flat_scores.numel() * (100 - 40) / 100), 'RECT_NATIVE_KTH_INDEX')
    require(torch.isfinite(flat_scores).all().item(), 'RECT_SCORE_NONFINITE')
    numel = int(flat_scores.numel())
    retained = int((flat_scores >= threshold).sum().item())
    ties = int((flat_scores == threshold).sum().item())
    return dict(k_percent=40, epsilon=1e-8, numel=numel, kth_index=k,
                threshold=float(threshold.item()), retained_count=retained,
                masked_count=numel - retained, threshold_tie_count=ties,
                retained_pct=100. * retained / numel,
                strict_greater_count=int((flat_scores > threshold).sum().item()),
                dense_lower_planning=True, restored_before_public_commit=True)


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
        opt = self.original.Adam(*args, **kwargs)
        original_step = opt.step

        def counted_step(*args, **kwargs):
            return self.engine._call('fit_updates', original_step, *args, **kwargs)

        opt.step = counted_step
        return opt


class _TorchCalls:
    """Native module-local proxies; no global torch monkeypatch."""
    def __init__(self, original, engine, optimizer=False):
        self.original, self.engine = original, engine
        self.linalg = _LinalgCalls(original.linalg, engine)
        self.optim = _OptimCalls(original.optim, engine) if optimizer else original.optim

    def __getattr__(self, name):
        return getattr(self.original, name)

    def kthvalue(self, *args, **kwargs):
        result = self.original.kthvalue(*args, **kwargs)
        if self.engine.arm == 'RECT' and self.engine.current is not None:
            scores = args[0] if args else kwargs['input']
            k = args[1] if len(args) > 1 else kwargs['k']
            row = native_rect_mask_stats(scores, k, result.values)
            index = len(self.engine.current['mask_trace'])
            row['layer'] = self.engine.layers[index]
            self.engine.current['mask_trace'].append(row)
        return result


class PruneRectEngine:
    """One same-model native apply per pack; PRUNE W0 is ephemeral CPU RAM."""
    def __init__(self, config, model, event_callback=None, *, tokenizer=None,
                 bundle=None, hparams=None):
        self.arm = config['arm']
        require(self.arm in NATIVE_SPECS, 'NATIVE_PRUNE_RECT_ARM')
        self.bundle = (load_native(config['native_bundle'][self.arm], self.arm)
                       if bundle is None else bundle)
        self.module = self.bundle.module
        self.hp = parse_hparams(self.bundle) if hparams is None else hparams
        self.model = model
        self.tokenizer = tokenizer if tokenizer is not None else config.get('_tokenizer')
        self.layers = list(self.hp.layers)
        require(self.layers == list(LAYERS), 'NATIVE_PRUNE_RECT_ALL_SIX_LAYERS')
        self.writer = 'memit_prune_terminal_base_fix' if self.arm == 'PRUNE' else 'memit_rect'
        self.next_batch, self.current = 1, None
        self.on_progress = event_callback
        self.cumulative = {key: 0 for key in COUNTERS}
        self.completed_native_z = 0
        self.saved_cold_weights = {}
        self.prune_applied = False
        self.terminal_receipt = None
        self._inside_execute = False
        self._active_weight_name = None
        self.context_receipt = dict(status='GENERATE_INSIDE_FIRST_REAL_NATIVE_APPLY',
                                    generated_by_wrapper=False, source_generator='util/generate.py',
                                    EasyEdit_context_transplanted=False, native_z_reuse=False)
        require(self.module.CONTEXT_TEMPLATES_CACHE is None and not self.module.COV_CACHE,
                'NATIVE_GLOBAL_INITIAL_STATE')
        require(not getattr(self.module, '_ODEEDIT_COUNTERS_BOUND', False), 'NATIVE_ENGINE_ALREADY_BOUND')
        self.original_z, self.original_ks = self.module.compute_z, self.module.compute_ks
        self.original_execute = self.module.execute_memit
        self.original_shape = self.module.upd_matrix_match_shape
        self.native_apply = (self.module.apply_memit_to_model if self.arm == 'PRUNE'
                             else self.module.apply_memit_rect_to_model)
        self.module.compute_z, self.module.compute_ks = self._compute_z, self._compute_ks
        self.module.execute_memit = self._execute
        self.module.upd_matrix_match_shape = self._match_shape
        self.module.torch = _TorchCalls(torch, self)
        self.bundle.z_module.torch = _TorchCalls(torch, self, optimizer=True)
        self.module._ODEEDIT_COUNTERS_BOUND = True
        self._bind_reprs()

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

    def _event(self):
        if self.on_progress is not None:
            try:
                self.on_progress(dict(batch=self.current['batch'],
                                      native_z=self.completed_native_z,
                                      native_z_completed=self.completed_native_z,
                                      native_z_calls=self.cumulative['native_z'],
                                      request_index=self.current['native_z'],
                                      fit_global_candidate=self.cumulative['fit_forwards'],
                                      fit_updates=self.cumulative['fit_updates']))
            except Exception:
                self.current['logging_callback_errors'] += 1

    def _fit_forward(self, *unused):
        self.current['fit_forwards'] += 1
        self.cumulative['fit_forwards'] += 1
        self._event()

    def _compute_z(self, *args, **kwargs):
        observed = {}
        previous = sys.getprofile()

        def profile(frame, event, arg):
            if event == 'return' and frame.f_code is self.original_z.__code__:
                observed.update({key: frame.f_locals[key] for key in
                                 ('it', 'loss', 'nll_loss', 'kl_loss', 'weight_decay')
                                 if key in frame.f_locals})
            if previous is not None:
                previous(frame, event, arg)

        before_forwards, before_updates = self.current['fit_forwards'], self.current['fit_updates']
        handle = self.model.register_forward_hook(self._fit_forward)
        sys.setprofile(profile)
        try:
            result = self._call('native_z', self.original_z, *args, **kwargs)
        finally:
            sys.setprofile(previous)
            handle.remove()
        require(tuple(result.shape) == (4096,) and result.dtype == torch.float32
                and torch.isfinite(result).all().item(), 'NATIVE_GPTJ_TARGET_SHAPE_NONFINITE')
        require('it' in observed, 'NATIVE_ITERATION_TELEMETRY')
        evaluations, updates = int(observed['it']) + 1, int(observed['it'])
        require(1 <= evaluations <= 25 and 0 <= updates <= 24
                and self.current['fit_forwards'] - before_forwards == evaluations
                and self.current['fit_updates'] - before_updates == updates,
                'NATIVE_25EVAL_24UPDATE_ACTUAL_FIT_COUNTS')
        row = dict(request_index=self.current['native_z'], evaluations=evaluations, Adam_updates=updates)
        row.update({key: float(value.detach()) for key, value in observed.items() if key != 'it'})
        require(all(math.isfinite(value) for value in row.values()), 'NATIVE_FIT_NONFINITE')
        row['stop'] = 'TOTAL_LOSS_BELOW_005' if row['loss'] < .05 else 'BUDGET_EXHAUSTED'
        self.current['fit_trace'].append(row)
        self.completed_native_z += 1
        self._event()
        return result

    def _compute_ks(self, *args, **kwargs):
        result = self._call('write_keys', self.original_ks, *args, **kwargs)
        require(tuple(result.shape) == (100, 16384) and result.dtype == torch.float32
                and torch.isfinite(result).all().item(), 'NATIVE_KEY_FULL100_GPTJ_NONFINITE')
        return result

    def _execute(self, *args, **kwargs):
        self._inside_execute = True
        try:
            result = self._call('executes', self.original_execute, *args, **kwargs)
            require(set(result) == {self.hp.rewrite_module_tmp.format(layer) + '.weight'
                                    for layer in self.layers}, 'NATIVE_DENSE_PLAN_ALL_SIX_LAYERS')
            return result
        finally:
            self._inside_execute = False

    def _match_shape(self, matrix, shape):
        result = self.original_shape(matrix, shape)
        require(torch.isfinite(result).all().item(), 'NATIVE_UPDATE_NONFINITE')
        key = 'dense_provisional_writes' if self._inside_execute else 'public_writes'
        index = self.current[key]
        self.current[key] += 1
        self.cumulative[key] += 1
        self.current['update_trace'].append(dict(
            layer=self.layers[index], stage='dense_provisional' if self._inside_execute else 'public',
            norm=float(torch.linalg.vector_norm(result).item()),
            dtype=str(result.dtype), rows=int(result.shape[0]), columns=int(result.shape[1])))
        return result

    def _bind_reprs(self):
        """Count actual native extra captures while preserving every hook/call."""
        repr_tools = self.bundle.z_module.repr_tools
        original = repr_tools.get_reprs_at_idxs
        signature = inspect.signature(original)

        def checked(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            rows = len(bound.arguments['contexts'])
            require(rows == len(bound.arguments['idxs']), 'NATIVE_REPR_INPUT_CARDINALITY')
            forwards = [0]
            model = bound.arguments['model']

            def count(*unused):
                forwards[0] += 1

            handle = model.register_forward_hook(count)
            started = time.monotonic()
            try:
                result = original(*args, **kwargs)
            finally:
                handle.remove()
            values = result if isinstance(result, tuple) else (result,)
            require(all(isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[0] == rows
                        and torch.isfinite(value).all().item() for value in values),
                    'NATIVE_REPR_CARDINALITY_NONFINITE')
            batches = (rows + 127) // 128
            module_name = bound.arguments['module_template'].format(bound.arguments['layer'])
            extra = module_name == 'transformer.h.8' and bound.arguments['track'] in ('in', 'both')
            from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
            extra = extra and isinstance(model, GPTJForCausalLM)
            require(forwards[0] == batches * (2 if extra else 1), 'NATIVE_GPTJ_EXTRA_CAPTURE_FORWARD_PRESERVED')
            if self.current is not None:
                self.current['repr_forwards'] += forwards[0]
                self.cumulative['repr_forwards'] += forwards[0]
                self.current['repr_extra_capture_forwards'] += batches if extra else 0
                self.cumulative['repr_extra_capture_forwards'] += batches if extra else 0
                self.current['seconds']['repr_forwards'] += time.monotonic() - started
            return result

        repr_tools.get_reprs_at_idxs = checked

    def history(self):
        return {}

    def contexts(self):
        return copy.deepcopy(self.module.CONTEXT_TEMPLATES_CACHE)

    contexts_snapshot = contexts
    context_snapshot = contexts

    def restore_contexts(self, contexts):
        self.module.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(contexts)

    restore_context = restore_contexts

    def prepare_contexts(self):
        require(self.next_batch == 1 and self.current is None
                and self.module.CONTEXT_TEMPLATES_CACHE is None, 'NATIVE_CONTEXT_ONLY_FIRST_COLD_PREPARATION')
        before = tuple((name, value.data_ptr(), value._version)
                       for name, value in self.model.named_parameters())
        hooks = {name: (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
                 for name, module in self.model.named_modules()}
        started = time.monotonic()
        contexts = self.module.get_context_templates(self.model, self.tokenizer)
        require(before == tuple((name, value.data_ptr(), value._version)
                                for name, value in self.model.named_parameters())
                and hooks == {name: (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
                              for name, module in self.model.named_modules()}, 'NATIVE_CONTEXT_MODEL_NONMUTATION')
        self._check_contexts(contexts)
        self.context_receipt.update(status='GENERATED_IN_FIRST_COLD_NATIVE_INPUT_PREPARATION',
                                    contexts_sha256=digest(contexts),
                                    native_input_preparation_seconds=time.monotonic() - started,
                                    native_z_calls=0, public_apply_calls=0,
                                    context_method='actual native get_context_templates/generate_fast')
        return copy.deepcopy(contexts)

    def _check_contexts(self, contexts):
        require(isinstance(contexts, list) and len(contexts) == 2 and contexts[0] == ['{}']
                and len(contexts[1]) == 5 and all(type(text) is str for group in contexts for text in group),
                'NATIVE_CONTEXT_1_PLUS5')

    def apply(self, records, batch):
        require(batch == self.next_batch and 1 <= batch <= 20 and not self.prune_applied,
                'NATIVE_BATCH_SEQUENCE_NO21')
        requests = native_requests(records)
        self.current = dict(batch=batch, logging_callback_errors=0, fit_trace=[], update_trace=[], mask_trace=[],
                            seconds={key: 0. for key in COUNTERS}, **{key: 0 for key in COUNTERS})
        started = time.monotonic()
        try:
            result = self._call('public_applies', self.native_apply, self.model, self.tokenizer,
                                requests, self.hp, copy=False,
                                return_orig_weights=self.arm == 'PRUNE' and batch == 1,
                                cache_template=None)
            require(isinstance(result, tuple) and len(result) == 2 and result[0] is self.model
                    and isinstance(result[1], dict), 'NATIVE_MODEL_RETURN_SAME_NO_HISTORY')
            if self.arm == 'PRUNE' and batch == 1:
                require(not self.saved_cold_weights and set(result[1]) == {
                    self.hp.rewrite_module_tmp.format(layer) + '.weight' for layer in self.layers},
                    'PRUNE_COLD_W0_FIRST_PACK_ALL_SIX_RAM_ONLY')
                self.saved_cold_weights = {name: weight.detach().to('cpu').clone()
                                           for name, weight in result[1].items()}
            else:
                require(not result[1], 'NATIVE_NO_UNUSED_WEIGHT_OR_HISTORY_RETURN')
            require(self.current['native_z'] == 100 and self.current['write_keys'] == 6
                    and self.current['solves'] == 6 and self.current['executes'] == 1
                    and self.current['dense_provisional_writes'] == self.current['public_writes'] == 6
                    and self.current['public_applies'] == 1
                    and self.current['history_keys'] == self.current['history_appends'] == 0,
                    'NATIVE_EXACT_SIX_LAYER_PUBLIC_APPLY_COUNTS')
            require(len(self.current['mask_trace']) == (6 if self.arm == 'RECT' else 0),
                    'NATIVE_RECT_EXACT_PUBLIC_MASK_COUNTS')
            contexts = self.module.CONTEXT_TEMPLATES_CACHE
            self._check_contexts(contexts)
            self.context_receipt.update(status='NATIVE_CONTEXT_CREATED_OR_CARRIED',
                                        contexts_sha256=digest(contexts), cold_creation_batch=1,
                                        no_cross_native_context_transplant=True)
            self.next_batch += 1
            receipt = dict(status='NATIVE_APPLY_RETURNED', arm=self.arm, writer=self.writer,
                           batch=batch, requests=100, request_identity=digest(requests),
                           counts=copy.deepcopy(self.current),
                           delta={key: self.current[key] for key in COUNT_FIELDS}, cumulative=self.counts,
                           same_model_returned=True, native_has_history=False, caller_history_appends=0,
                           H_required=False, P_required=False, native_z_disk_cache=False, cache_template=None,
                           context=copy.deepcopy(self.context_receipt),
                           hparams=asdict(self.hp) if is_dataclass(self.hp) else vars(self.hp),
                           prune_applied=False, explicit_repair=BASE_FIX if self.arm == 'PRUNE' else None,
                           cold_W0_RAM_only=self.arm == 'PRUNE', seconds=time.monotonic() - started,
                           timing_policy='public apply inclusive; z/key/solve/repr/update nested, not additive',
                           solve_timer='CPU dispatch wall only; no added GPU timing synchronization',
                           checkpoint_saved=False, exact_resume='NOT_AVAILABLE')
            return self.model, receipt
        finally:
            self.current = None

    def terminal_prune(self):
        """Exactly once after dense B20, before either W20 post evaluator."""
        require(self.arm == 'PRUNE' and self.next_batch == 21 and not self.prune_applied
                and self.current is None and len(self.saved_cold_weights) == 6,
                'PRUNE_TERMINAL_ONCE_AFTER20_BEFORE_W20_EVAL')
        started = time.monotonic()
        rows = []
        with torch.no_grad():
            for layer in self.layers:
                name = self.hp.rewrite_module_tmp.format(layer) + '.weight'
                weight = self.module.nethook.get_parameter(self.model, name)
                cold = self.saved_cold_weights[name].to(weight.device)
                final_weight, row = native_terminal_compression(cold, weight)
                weight.copy_(final_weight)
                rows.append(dict(layer=layer, **row))
                del cold, final_weight
        self.prune_applied = True
        self.terminal_receipt = dict(status='PRUNE_TERMINAL_APPLIED', arm='PRUNE',
                                     explicit_repair=BASE_FIX, native_spectral_formula_unchanged=True,
                                     final_weight_base='saved_cold_W0', original_bug_base='edited_W20_dense',
                                     prune_applied=True, batch=20, state_edits=2000, layers=rows,
                                     native_svd_calls=sum(row['native_svd_calls'] for row in rows),
                                     compressed_singular_values=sum(row['singular_values_compressed'] for row in rows),
                                     seconds=time.monotonic() - started, W0_RAM_only=True,
                                     checkpoint_saved=False, exact_resume='NOT_AVAILABLE')
        return copy.deepcopy(self.terminal_receipt)


NativeEngine = PruneRectEngine


def prepare_native(config, model, tokenizer, arm, attempt=None, progress=None):
    require(arm in NATIVE_SPECS, 'NATIVE_PRUNE_RECT_ARM')
    config = dict(config, arm=arm)
    bundle = load_native(config['native_bundle'][arm], arm)
    hp = parse_hparams(bundle)
    require(model.config.model_type == 'gptj' and model.config.n_positions == 2048
            and model.config.n_layer == 28 and model.config.n_embd == 4096
            and model.config.vocab_size == 50400, 'NATIVE_GPTJ_MODEL_CONFIG')
    for layer in hp.layers:
        module = bundle.module.nethook.get_module(model, hp.rewrite_module_tmp.format(layer))
        require(isinstance(module, torch.nn.Linear)
                and tuple(module.weight.shape) == (4096, 16384) and module.weight.dtype == torch.float32,
                'NATIVE_GPTJ_LINEAR_FC_OUT_STORED_LAYOUT')
    native = config['native']
    require(native.get('context_reuse') is None, 'NATIVE_CONTEXT_NO_UNVALIDATED_CACHE_REUSE')
    stats_dir = Path(native['stats_dir']).resolve()
    for layer in hp.layers:
        row = native['stats'][str(layer)]
        path = Path(row['path']).resolve()
        expected = stats_dir / 'gpt-j-6b' / 'wikipedia_stats' / (
            f'transformer.h.{layer}.mlp.fc_out_float32_mom2_100000.npz')
        require(path == expected and path.is_file() and path.stat().st_size == row['bytes'],
                'NATIVE_C0_MISSING_OR_CHANGED_NO_RECOMPUTE')
        if 'mtime_ns' in row:
            require(path.stat().st_mtime_ns == row['mtime_ns'], 'NATIVE_C0_STAT_SEAL_CHANGED')
    # Native layer_stats reads this mapped logical basename and native raw C0.
    # Its original cache loader is retained; a cache miss can never collect.
    model.config._name_or_path = 'gpt-j-6b'
    bundle.module.STATS_DIR = stats_dir

    original_layer_stats = bundle.module.layer_stats
    stats_signature = inspect.signature(original_layer_stats)

    def sealed_layer_stats(*args, **kwargs):
        bound = stats_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        layer_name = values['layer_name']
        paths = {hp.rewrite_module_tmp.format(layer): Path(native['stats'][str(layer)]['path']).resolve()
                 for layer in hp.layers}
        require(layer_name in paths and paths[layer_name].is_file()
                and Path(values['stats_dir']).resolve() == stats_dir
                and values['ds_name'] == 'wikipedia' and values['to_collect'] == ['mom2']
                and values['sample_size'] == 100000 and values['precision'] == 'float32'
                and not values['force_recompute'], 'NATIVE_C0_EXACT_CACHE_ONLY_NO_COLLECTION')
        return original_layer_stats(*args, **kwargs)

    bundle.module.layer_stats = sealed_layer_stats

    def no_collection(*args, **kwargs):
        raise RuntimeError('NATIVE_C0_CACHE_MISS_RECOMPUTE_FORBIDDEN')

    bundle.runningstats.make_loader = no_collection
    return PruneRectEngine(config, model, progress, tokenizer=tokenizer, bundle=bundle, hparams=hp)
