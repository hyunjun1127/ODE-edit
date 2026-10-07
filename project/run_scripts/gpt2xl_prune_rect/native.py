"""Pinned BLUE ordinary MEMIT PRUNE/RECT direct native runtime.

Original target fit, provisional dense writer/restore and RECT mask expressions
are preserved. Private source differs only in namespace/path plumbing plus the
approved observer-only RECT callback; PRUNE's terminal cold-base repair lives in
terminal.py and is labeled explicitly. No P, edit history or target disk cache.
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

from .terminal import apply_terminal, tensor_sha


BLUE_ROOT = '/mnt/raid5/janghj/BLUE'
BLUE_COMMIT = '311b076a92e4ed0f14f5c8b4909732da781bc5f7'
HP = 'hparams/MEMIT/gpt2-xl.json'
HP_SHA = '88f0a396b9b5573a818b91d6836ae4b50791fcb5fb1b61c8e0304e1323556861'
EVALUATE_SHA = '7a4ece770894a55cbf029cef38f306dfc5b8b06f623239cb5afce5b3cdfc0b80'
LAYERS = [13, 14, 15, 16, 17]
NATIVE_SPECS = {
    'PRUNE': dict(root=BLUE_ROOT, commit=BLUE_COMMIT, namespace='_odeedit_prune_native',
                  package='memit', main='memit_main', hparams=HP, hparams_sha=HP_SHA,
                  hparams_class='MEMITHyperParams', layers=LAYERS,
                  main_sha='943a0da1a758692fc4b38c5d1203650795f441a3952df7a4814ffe40ee2567be'),
    'RECT': dict(root=BLUE_ROOT, commit=BLUE_COMMIT, namespace='_odeedit_rect_native',
                 package='memit', main='memit_rect_main', hparams=HP, hparams_sha=HP_SHA,
                 hparams_class='MEMITHyperParams', layers=LAYERS,
                 main_sha='8e199fd49e744f6c08f0edd7933ab683df1970a5dcc9da41570e4368073ec597'),
}
SHARED_FILES = ('rome/layer_stats.py', 'rome/repr_tools.py', 'rome/tok_dataset.py',
                'util/generate.py', 'util/globals.py', 'util/hparams.py',
                'util/logit_lens.py', 'util/nethook.py', 'util/runningstats.py')
COUNT_FIELDS = ('native_z', 'write_keys', 'history_keys', 'solves', 'history_appends')
COUNTERS = COUNT_FIELDS + ('fit_forwards', 'fit_updates', 'public_applies',
                         'executes', 'rect_masks', 'terminal_transforms')


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
    result = (f'memit/{spec["main"]}.py', 'memit/compute_z.py', 'memit/compute_ks.py',
              'memit/memit_hparams.py', *SHARED_FILES, 'globals.yml', HP)
    return result + (('experiments/evaluate.py',) if arm == 'PRUNE' else ())


def effective_source(relative, original, arm):
    # evaluate.py is provenance-only and never imported/executed as a CLI.
    if not relative.endswith('.py') or relative == 'experiments/evaluate.py':
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
        text = text.replace('    Path(z)\n',
            '    (Path(z) if Path(z).is_absolute() else Path(__file__).resolve().parents[1] / z)\n')
    if relative == 'memit/memit_rect_main.py':
        require(text.count('COV_CACHE = {}\n') == 1
                and text.count('                mask = delta >= threshold\n') == 1,
                'RECT_OBSERVER_INSERTION_EXACT_SITE')
        text = text.replace('COV_CACHE = {}\n',
                            'COV_CACHE = {}\n_ODEEDIT_RECT_MASK_OBSERVER = None\n')
        text = text.replace('                mask = delta >= threshold\n',
            '                mask = delta >= threshold\n'
            '                if _ODEEDIT_RECT_MASK_OBSERVER is not None:\n'
            '                    _ODEEDIT_RECT_MASK_OBSERVER(w_name, delta, threshold, mask)\n')
    return text.encode('utf-8')


def adopt_native(destination):
    """Create-once small private execution closure; no model/tensor copied."""
    destination = Path(destination).resolve()
    require(not destination.exists(), 'NATIVE_ADOPTION_CREATE_ONCE')
    actual = subprocess.run(['git', '-C', BLUE_ROOT, 'rev-parse', 'HEAD'],
                            check=True, capture_output=True, text=True).stdout.strip()
    clean = subprocess.run(['git', '-C', BLUE_ROOT, 'status', '--porcelain'],
                           check=True, capture_output=True, text=True).stdout
    require(actual == BLUE_COMMIT and not clean.strip(), 'NATIVE_BLUE_CLEAN_COMMIT')
    destination.mkdir(parents=True)
    result = {}
    for arm, spec in NATIVE_SPECS.items():
        target = destination / arm
        target.mkdir()
        files = []
        for relative in source_files(arm):
            source = Path(BLUE_ROOT) / relative
            require(source.is_file() and not source.is_symlink(), 'NATIVE_REGULAR_SOURCE')
            original = source.read_bytes()
            effective = effective_source(relative, original, arm)
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('xb') as stream:
                stream.write(effective)
            diff = ''.join(difflib.unified_diff(original.decode().splitlines(True),
                effective.decode().splitlines(True), fromfile='original/' + relative,
                tofile='private/' + relative))
            files.append(dict(relative=relative, original=member(source),
                              effective=member(output), compatibility_diff=diff,
                              execution=relative != 'experiments/evaluate.py'))
        by_name = {row['relative']: row for row in files}
        hp, main = by_name[HP], by_name['memit/' + spec['main'] + '.py']
        require(hp['original']['sha256'] == hp['effective']['sha256'] == HP_SHA
                and main['original']['sha256'] == spec['main_sha'], 'NATIVE_HPARAMS_MAIN_BYTES')
        if arm == 'PRUNE':
            require(by_name['experiments/evaluate.py']['original']['sha256'] == EVALUATE_SHA,
                    'PRUNE_TERMINAL_ORIGINAL_SOURCE_BYTES')
        result[arm] = dict(root=str(target), arm=arm, namespace=spec['namespace'],
            original_root=BLUE_ROOT, original_commit=actual, original_clean=True,
            files=files, hparams=hp['effective'], package_shells=True,
            original_package_initializers_executed=False,
            provenance='BLUE native variant, not EasyEdit-native implementation')
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


def closure(bundle):
    result = []
    for name, module in sorted(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if name.startswith(bundle.namespace + '.') and file:
            path = Path(file).resolve()
            require(path.is_relative_to(bundle.root), 'NATIVE_IMPORT_ESCAPE')
            result.append(dict(module=name, relative=str(path.relative_to(bundle.root)), **member(path)))
    return result


def load_native(binding, arm):
    spec = NATIVE_SPECS[arm]
    root, prefix = Path(binding['root']).resolve(), spec['namespace']
    expected = {row['relative']: row for row in binding['files']}
    require(set(expected) == set(source_files(arm)) and binding['namespace'] == prefix
            and binding['original_commit'] == BLUE_COMMIT, 'NATIVE_EXACT_SOURCE_BINDING')
    for relative, row in expected.items():
        require(verify(row['effective']) == root / relative, 'NATIVE_PRIVATE_ROOT')
    require(not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules),
            'NATIVE_NAMESPACE_ALREADY_LOADED')
    for relative in ('', 'memit', 'rome', 'util'):
        name = prefix + ('.' + relative if relative else '')
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(root / relative) if relative else str(root)]
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
    module = importlib.import_module(prefix + '.memit.' + spec['main'])
    z = importlib.import_module(prefix + '.memit.compute_z')
    stats = importlib.import_module(prefix + '.util.runningstats')
    bundle = NativeBundle(arm, root, prefix, module, z, stats, binding)
    expected_py = {p for p in source_files(arm) if p.endswith('.py') and p != 'experiments/evaluate.py'}
    require({row['relative'] for row in closure(bundle)} == expected_py,
            'NATIVE_IMPORTED_CLOSURE_EXACT')
    return bundle


def parse_hparams(bundle):
    hp = bundle.module.MEMITHyperParams.from_json(str(verify(bundle.binding['hparams'])))
    require(hp.layers == LAYERS and hp.model_name == 'gpt2-xl'
            and hp.blue is False and hp.edit_layer == -1
            and (hp.v_lr, hp.v_num_grad_steps, hp.v_loss_layer, hp.v_weight_decay,
                 hp.clamp_norm_factor, hp.kl_factor, hp.mom2_update_weight) ==
                (.5, 20, 47, .5, .75, .0625, 20000), 'ORDINARY_MEMIT_NATIVE_PARSER_FIELDS')
    require(hp.fact_token == 'subject_last' and hp.mom2_dataset == 'wikipedia'
            and hp.mom2_n_samples == 100000 and hp.mom2_dtype == 'float32'
            and hp.rewrite_module_tmp == 'transformer.h.{}.mlp.c_proj'
            and hp.layer_module_tmp == 'transformer.h.{}'
            and hp.ln_f_module == 'transformer.ln_f' and hp.lm_head_module == 'lm_head',
            'NATIVE_GPT2_MODULE_AND_C0_FIELDS')
    return hp


def native_requests(records):
    requests = []
    for record in records:
        request = copy.deepcopy(record.get('requested_rewrite', record))
        request['case_id'] = record['case_id']
        require(type(request['case_id']) is int and isinstance(request.get('target_new'), dict)
                and type(request['target_new'].get('str')) is str and request['target_new']['str']
                and all(type(request.get(k)) is str and request[k] for k in ('prompt', 'subject'))
                and request['prompt'].count('{}') == 1, 'NATIVE_DICT_TARGET_REQUEST_SCHEMA')
        requests.append(request)
    require(len(requests) == 100, 'NATIVE_BS100_NO_EXCLUSIONS')
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
        original_step = opt.step

        def step(*a, **k):
            return self.engine._call('fit_updates', original_step, *a, **k)

        opt.step = step
        return opt


class _TorchCalls:
    def __init__(self, original, engine, optimizer=False):
        self.original = original
        self.linalg = _LinalgCalls(original.linalg, engine)
        self.optim = _OptimCalls(original.optim, engine) if optimizer else original.optim

    def __getattr__(self, name):
        return getattr(self.original, name)


class NativeEngine:
    def __init__(self, bundle, hp, model, tokenizer):
        self.bundle, self.module, self.hp, self.arm = bundle, bundle.module, hp, bundle.arm
        self.model, self.tokenizer = model, tokenizer
        self.writer = 'memit_prune' if self.arm == 'PRUNE' else 'memit_rect'
        self.layers, self.next_batch = LAYERS, 1
        self.current, self.on_progress, self.prune_applied = None, None, False
        self.cumulative = {key: 0 for key in COUNTERS}
        self.cold_W0 = {}
        self.terminal_receipt = None
        self.context_receipt = dict(status='FIRST_COLD_NATIVE_INPUT_PREPARATION_PENDING',
                                    source_generator='BLUE util/generate.py', EasyEdit_context_transplanted=False)
        require(self.module.CONTEXT_TEMPLATES_CACHE is None and not self.module.COV_CACHE,
                'NATIVE_INITIAL_GLOBAL_STATE')
        require(not getattr(self.module, '_ODEEDIT_COUNTERS_BOUND', False), 'NATIVE_COUNTERS_ALREADY_BOUND')
        self.original_z, self.original_ks = self.module.compute_z, self.module.compute_ks
        self.original_execute = self.module.execute_memit
        self.native_apply = (self.module.apply_memit_to_model if self.arm == 'PRUNE'
                             else self.module.apply_memit_rect_to_model)
        self.module.compute_z, self.module.compute_ks = self._compute_z, self._compute_ks
        self.module.execute_memit = self._execute
        self.module.torch = _TorchCalls(torch, self)
        bundle.z_module.torch = _TorchCalls(torch, self, optimizer=True)
        self.module._ODEEDIT_COUNTERS_BOUND = True
        if self.arm == 'RECT':
            self.module._ODEEDIT_RECT_MASK_OBSERVER = self._rect_mask
        repr_tools = bundle.z_module.repr_tools
        original_reprs, signature = repr_tools.get_reprs_at_idxs, inspect.signature(repr_tools.get_reprs_at_idxs)

        def checked_reprs(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            count = len(bound.arguments['contexts'])
            require(len(bound.arguments['idxs']) == count, 'NATIVE_REPR_INPUT_CARDINALITY')
            result = original_reprs(*args, **kwargs)
            values = result if isinstance(result, tuple) else (result,)
            require(all(isinstance(value, torch.Tensor) and value.ndim == 2
                        and value.shape[0] == count and torch.isfinite(value).all().item()
                        for value in values), 'NATIVE_REPR_DROP_OR_NONFINITE')
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
        require(self.current is not None, 'NATIVE_COUNTER_OUTSIDE_APPLY')
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
                    fit_global_candidate=self.cumulative['fit_forwards'],
                    fit_updates=self.cumulative['fit_updates'], native_z=self.cumulative['native_z'],
                    native_z_completed=self.cumulative['native_z']))
            except Exception:
                self.current['logging_callback_errors'] += 1

    def _compute_z(self, *args, **kwargs):
        handle = self.model.register_forward_hook(self._forward)
        try:
            result = self._call('native_z', self.original_z, *args, **kwargs)
            require(tuple(result.shape) == (1600,) and result.dtype == torch.float32
                    and torch.isfinite(result).all().item(), 'NATIVE_Z_SHAPE_FINITE')
            return result
        finally:
            handle.remove()

    def _compute_ks(self, *args, **kwargs):
        result = self._call('write_keys', self.original_ks, *args, **kwargs)
        require(tuple(result.shape) == (100, 6400) and result.dtype == torch.float32
                and torch.isfinite(result).all().item(), 'NATIVE_K_FULL_B100_FINITE')
        return result

    def selected(self):
        return {self.hp.rewrite_module_tmp.format(layer) + '.weight':
                self.module.nethook.get_parameter(self.model, self.hp.rewrite_module_tmp.format(layer) + '.weight')
                for layer in self.layers}

    def _execute(self, *args, **kwargs):
        before = {name: tensor_sha(value) for name, value in self.selected().items()}
        result = self._call('executes', self.original_execute, *args, **kwargs)
        after = {name: tensor_sha(value) for name, value in self.selected().items()}
        require(before == after, 'NATIVE_PROVISIONAL_DENSE_WRITER_RESTORE_MISMATCH')
        self.current['provisional_restore'] = dict(exact=True, before=before, after=after,
                                                  planning='native dense lower provisional writes')
        return result

    def _rect_mask(self, name, relative_scores, threshold, mask):
        require(self.current is not None and name in self.selected(), 'RECT_MASK_SELECTED_IDENTITY')
        require(relative_scores.shape == mask.shape and relative_scores.numel() == 6400 * 1600
                and relative_scores.dtype == torch.float64 and mask.dtype == torch.bool
                and threshold.numel() == 1 and torch.isfinite(relative_scores).all().item()
                and torch.isfinite(threshold).all().item(), 'RECT_RELATIVE_SCORE_NONFINITE_OR_SHAPE')
        count = relative_scores.numel()
        kept, tied = int(mask.sum().item()), int((relative_scores == threshold).sum().item())
        less = int((relative_scores < threshold).sum().item())
        require(torch.equal(mask, relative_scores >= threshold) and kept == count - less,
                'RECT_NATIVE_THRESHOLD_MASK_IDENTITY')
        self.current['rect_masks'] += 1
        self.cumulative['rect_masks'] += 1
        self.current['rect_mask_rows'].append(dict(weight=name, denominator=count,
            support_count=kept, support_pct=100. * kept / count, threshold_ties=tied,
            strictly_below_threshold_count=less, threshold=float(threshold.item()),
            kth_index=int(count * (100 - 40) / 100), native_k_percent=40,
            native_epsilon=1e-8, native_comparison='>=',
            ratio='abs(shape-matched dense delta/(current W+1e-8))',
            no_mask_tensor_saved=True, no_additional_model_forward=True))

    def history(self):
        return {}

    def restore_history(self, history):
        require(not history, 'ORDINARY_MEMIT_HAS_NO_NATIVE_H')

    def contexts(self):
        return copy.deepcopy(self.module.CONTEXT_TEMPLATES_CACHE)

    context_snapshot = contexts

    def restore_context(self, contexts):
        self.module.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(contexts)

    def snapshot_native_state(self):
        return dict(contexts=self.contexts(), cold_W0=dict(self.cold_W0),
                    next_batch=self.next_batch, context_receipt=copy.deepcopy(self.context_receipt),
                    prune_applied=self.prune_applied, terminal_receipt=copy.deepcopy(self.terminal_receipt))

    def restore_native_state(self, snapshot):
        self.restore_context(snapshot['contexts'])
        self.cold_W0 = dict(snapshot['cold_W0'])
        self.next_batch = snapshot['next_batch']
        self.context_receipt = copy.deepcopy(snapshot['context_receipt'])
        self.prune_applied = snapshot['prune_applied']
        self.terminal_receipt = copy.deepcopy(snapshot['terminal_receipt'])

    def native_state_signature(self):
        """Small hashes only; failed-work counters deliberately are not rolled back."""
        return dict(contexts_sha256=digest(self.contexts()), next_batch=self.next_batch,
                    prune_applied=self.prune_applied,
                    context_receipt_sha256=digest(self.context_receipt),
                    cold_W0={name: dict(sha256=tensor_sha(value), data_ptr=value.data_ptr(),
                                      shape=list(value.shape), dtype=str(value.dtype))
                             for name, value in self.cold_W0.items()},
                    terminal_receipt_sha256=digest(self.terminal_receipt))

    def prepare_contexts(self):
        require(self.next_batch == 1 and self.current is None
                and self.module.CONTEXT_TEMPLATES_CACHE is None, 'NATIVE_FIRST_COLD_CONTEXT_ONCE')
        before = tuple((name, p.data_ptr(), p._version) for name, p in self.model.named_parameters())
        contexts = self.module.get_context_templates(self.model, self.tokenizer)
        require(tuple((name, p.data_ptr(), p._version) for name, p in self.model.named_parameters()) == before,
                'NATIVE_CONTEXT_WEIGHT_MUTATION')
        require(len(contexts) == 2 and contexts[0] == ['{}'] and len(contexts[1]) == 5,
                'NATIVE_CONTEXT_1_PLUS5')
        self.context_receipt.update(status='OWN_NATIVE_COLD_CONTEXT_PREPARED', contexts_sha256=digest(contexts),
                                    target_fits=0, generated_by_wrapper=False)
        return copy.deepcopy(contexts)

    def apply(self, records, batch):
        require(batch == self.next_batch and 1 <= batch <= 20, 'NATIVE_BATCH_SEQUENCE_NO21')
        requests = native_requests(records)
        self.current = dict(batch=batch, rect_mask_rows=[], logging_callback_errors=0,
            seconds={key: 0. for key in COUNTERS}, **{key: 0 for key in COUNTERS})
        started = time.monotonic()
        capture_W0 = self.arm == 'PRUNE' and batch == 1
        result = self._call('public_applies', self.native_apply, self.model, self.tokenizer,
                           requests, self.hp, copy=False, return_orig_weights=capture_W0,
                           cache_template=None)
        require(isinstance(result, tuple) and len(result) == 2 and result[0] is self.model
                and isinstance(result[1], dict), 'NATIVE_PUBLIC_MODEL_AND_WEIGHTS_COPY')
        if capture_W0:
            require(not self.cold_W0 and set(result[1]) == set(self.selected()), 'PRUNE_FIRST_NATIVE_COLD_COPIES')
            self.cold_W0 = {name: value.detach().cpu().clone() for name, value in result[1].items()}
            require(all(tuple(v.shape) == (6400, 1600) and v.dtype == torch.float32
                        and torch.isfinite(v).all().item() for v in self.cold_W0.values()),
                    'PRUNE_NATIVE_SELECTED_W0_SCHEMA')
            require({name: tensor_sha(value) for name, value in self.cold_W0.items()} ==
                    self.current['provisional_restore']['before'],
                    'PRUNE_RETURNED_COPIES_ARE_EXACT_FIRST_COLD_ENTRY')
        else:
            require(not result[1], 'NO_REDUNDANT_NATIVE_ORIGINAL_WEIGHT_COPIES')
        require({key: self.current[key] for key in COUNT_FIELDS} ==
                dict(native_z=100, write_keys=5, history_keys=0, solves=5, history_appends=0)
                and self.current['executes'] == self.current['public_applies'] == 1,
                'ORDINARY_NATIVE_B100_EXACT_CALLS_NO_HISTORY')
        require(self.current['rect_masks'] == (5 if self.arm == 'RECT' else 0), 'RECT_FIVE_ACTUAL_MASKS')
        self.next_batch += 1
        receipt = dict(status='NATIVE_APPLY_RETURNED', arm=self.arm, writer=self.writer, batch=batch,
            requests=100, request_identity=digest(requests), counts=copy.deepcopy(self.current),
            delta={key: self.current[key] for key in COUNT_FIELDS}, cumulative=self.counts,
            native_has_history=False, native_uses_P=False, same_model_returned=True,
            return_orig_weights=capture_W0, selected_coldW0_saved_in_RAM=capture_W0,
            native_z_disk_cache=False, cache_template=None, caller_history_appends=0,
            prune_applied=self.prune_applied, terminal_transform_pending=self.arm == 'PRUNE',
            rect_mask_rows=copy.deepcopy(self.current['rect_mask_rows']),
            context=copy.deepcopy(self.context_receipt), hparams=asdict(self.hp),
            seconds=time.monotonic() - started, checkpoint_saved=False, exact_resume='NOT_AVAILABLE')
        return self.model, receipt

    def terminal_prune(self):
        require(self.arm == 'PRUNE' and self.next_batch == 21 and not self.prune_applied
                and len(self.cold_W0) == 5, 'PRUNE_TERMINAL_EXACTLY_ONCE_AFTER20')
        selected = self.selected()
        excluded = {id(value) for value in selected.values()}
        before = tuple((name, value.data_ptr(), value._version) for name, value in self.model.named_parameters()
                       if id(value) not in excluded)
        receipt = apply_terminal(selected, self.cold_W0)
        after = tuple((name, value.data_ptr(), value._version) for name, value in self.model.named_parameters()
                      if id(value) not in excluded)
        require(before == after, 'PRUNE_TERMINAL_NONSELECTED_MUTATION')
        self.prune_applied, self.terminal_receipt = True, receipt
        self.cumulative['terminal_transforms'] += 1
        receipt.update(prune_applied=True, nonselected_guard=True)
        return receipt

    def finish_batch(self, number):
        require(self.next_batch == number + 1, 'NATIVE_FINISH_BATCH_AFTER_APPLY')
        if self.arm == 'PRUNE' and number == 20:
            return self.terminal_prune()
        return dict(status='NO_TERMINAL_TRANSFORM_THIS_BATCH', arm=self.arm, batch=number,
                    prune_applied=False, terminal_transforms=0, checkpoint_saved=False,
                    no_model_mutation=True)


def prepare_native(config, model, tokenizer, arm, attempt=None, progress=None):
    binding = config['native']
    if 'arms' in binding:
        binding = binding['arms'][arm]
    adoption = binding.get('bundle', config.get('native_bundle', {}).get(arm))
    require(adoption is not None, 'NATIVE_PRIVATE_SOURCE_BINDING')
    bundle = load_native(adoption, arm)
    hp = parse_hparams(bundle)
    require(model.config.model_type == 'gpt2' and model.config.n_positions == 1024
            and model.config.n_layer == 48 and model.config.n_embd == 1600,
            'NATIVE_GPT2_COLD_MODEL_CONFIG')
    require(model.get_input_embeddings().weight is model.get_output_embeddings().weight,
            'NATIVE_GPT2_TIED_HEAD')
    for layer in LAYERS:
        weight = bundle.module.nethook.get_parameter(model, hp.rewrite_module_tmp.format(layer) + '.weight')
        bias = bundle.module.nethook.get_parameter(model, hp.rewrite_module_tmp.format(layer) + '.bias')
        require(tuple(weight.shape) == (6400, 1600) and tuple(bias.shape) == (1600,)
                and weight.dtype == bias.dtype == torch.float32, 'NATIVE_CONV1D_LAYOUT_BIAS')
    engine = NativeEngine(bundle, hp, model, tokenizer)
    engine.progress = progress

    def blocked_stats(*args, **kwargs):
        raise RuntimeError('NATIVE_C0_CACHE_MISS_RECOMPUTE_FORBIDDEN')

    bundle.module.layer_stats = blocked_stats
    # Populate original get_cov's exact key. Original C0 moment()/FP32 divide
    # comes from the pinned native runningstats implementation, not raw sum.
    model_name = model.config._name_or_path.replace('/', '_')
    for layer in LAYERS:
        stat = bundle.runningstats.CombinedStat(mom2=bundle.runningstats.SecondMoment())
        stat.load(str(verify(binding['stats'][str(layer)])))
        require(stat.mom2.count == 44068071 and stat.mom2.mom2.dtype == torch.float32
                and tuple(stat.mom2.mom2.shape) == (6400, 6400), 'C0_RAW_MASKED_TOKEN_SUM_COUNT')
        cov = stat.mom2.moment().float().cpu()
        require(torch.isfinite(cov).all().item(), 'NATIVE_C0_NONFINITE')
        bundle.module.COV_CACHE[(model_name, hp.rewrite_module_tmp.format(layer))] = cov
    return engine
