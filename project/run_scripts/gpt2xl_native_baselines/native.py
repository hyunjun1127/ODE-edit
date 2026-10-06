"""Thin direct native-source binding; no copied fitting or writing mathematics.

Package shells avoid EasyEdit's unrelated top-level optional imports. Every
executed implementation file remains the exact SHA-bound native source.
Telemetry wraps original calls without altering arguments, return values,
precision, or the process-global torch API.
"""
import ast
import copy
import importlib
import importlib.machinery
import inspect
import json
import time
import types
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .common import LAYERS, digest, member, require, sha, verify


DEFAULT_ROOT = Path('/mnt/raid5/janghj/EasyEdit/easyeditor')
NATIVE_FILES = (
    'models/alphaedit/AlphaEdit_hparams.py',
    'models/alphaedit/AlphaEdit_main.py',
    'models/alphaedit/compute_ks.py', 'models/alphaedit/compute_z.py',
    'models/memit/compute_ks.py', 'models/memit/compute_z.py',
    'models/memit/memit_hparams.py', 'models/memit/memit_main.py',
    'models/rome/layer_stats.py', 'models/rome/repr_tools.py',
    'models/rome/tok_dataset.py', 'util/generate.py', 'util/globals.py',
    'util/hparams.py', 'util/logit_lens.py', 'util/nethook.py',
    'util/runningstats.py',
)
COUNT_FIELDS = ('native_z', 'write_keys', 'history_keys', 'solves', 'history_appends')


@dataclass
class NativeBundle:
    root: Path
    namespace: str
    memit: object
    alphaedit: object


def load_native(root=DEFAULT_ROOT, expected_members=None):
    """Import complete original modules, not AST-selected mathematical copies."""
    import sys
    root = Path(root).resolve()
    if expected_members is not None:
        expected = {row['relative']: row for row in expected_members}
        require(set(expected) == set(NATIVE_FILES), 'EXACT_NATIVE_17FILE_CLOSURE')
        for relative, row in expected.items():
            path = root / relative
            require(path.is_file() and not path.is_symlink()
                    and path.stat().st_size == row['bytes']
                    and sha(path) == row['sha256'], 'NATIVE_SOURCE_CHANGED:' + relative)
    prefix = '_odeedit_gpt2_native_' + uuid.uuid4().hex
    for relative in ('', 'models', 'models.rome', 'models.memit',
                     'models.alphaedit', 'util'):
        name = prefix + ('.' + relative if relative else '')
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(root.joinpath(*relative.split('.'))) if relative else str(root)]
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
    memit = importlib.import_module(prefix + '.models.memit.memit_main')
    alpha = importlib.import_module(prefix + '.models.alphaedit.AlphaEdit_main')
    bundle = NativeBundle(root, prefix, memit, alpha)
    actual = closure(bundle)
    require({row['relative'] for row in actual} == set(NATIVE_FILES),
            'NATIVE_IMPORTED_CLOSURE_CHANGED')
    if expected_members is not None:
        require(all(row['sha256'] == expected[row['relative']]['sha256'] for row in actual),
                'NATIVE_IMPORTED_SOURCE_IDENTITY')
    return bundle


def closure(bundle=None):
    """Stable canonical module labels plus actual imported paths/SHA/size."""
    import sys
    bundle = load_native() if bundle is None else bundle
    rows = []
    for name, module in sorted(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if not name.startswith(bundle.namespace + '.') or file is None:
            continue
        path = Path(file).resolve()
        require(path.is_relative_to(bundle.root), 'NATIVE_IMPORT_ESCAPED_SOURCE_ROOT')
        rows.append(dict(module='easyeditor' + name[len(bundle.namespace):],
                         relative=str(path.relative_to(bundle.root)), **member(path)))
    return sorted(rows, key=lambda row: row['relative'])


def native_requests(records):
    """Only schema conversion; native execute retains its leading-space rule."""
    requests = []
    for record in records:
        rewrite = record.get('requested_rewrite', record)
        target = rewrite['target_new']
        target = target['str'] if isinstance(target, dict) else target
        request = dict(case_id=record['case_id'], prompt=rewrite['prompt'],
                       subject=rewrite['subject'], target_new=target)
        require(type(request['case_id']) is int
                and all(type(request[key]) is str and request[key]
                        for key in ('prompt', 'subject', 'target_new')),
                'NATIVE_REQUEST_STRING_SCHEMA')
        requests.append(request)
    require(len(requests) == 100 and len({r['case_id'] for r in requests}) == 100,
            'NATIVE_BS100_OCCURRENCE_IDENTITY')
    return requests


# Parent runner's schema-only naming; no separate native request logic.
normalize_requests = native_requests


def _function_ast(module, name):
    tree = ast.parse(Path(inspect.getsourcefile(module)).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.dump(node, include_attributes=False)


def bind_context(bundle, config, module):
    """Reuse only source/seed/runtime/model/cold-bound native generated contexts."""
    binding = config['native'].get('context_reuse')
    if binding is None:
        require(module.CONTEXT_TEMPLATES_CACHE is None, 'NATIVE_CONTEXT_INITIAL_EMPTY')
        return dict(status='GENERATE_INSIDE_FIRST_REAL_NATIVE_APPLY',
                    generated_by_wrapper=False, native_cache_preserved=True)
    contexts_path = verify(binding['contexts'])
    ready = json.loads(verify(binding['ready']).read_text())
    runtime = json.loads(verify(binding['runtime']).read_text())
    expected = {row['relative']: row for row in config['native']['closure']}
    sources = binding['source_members']
    require(len(sources) == 2, 'NATIVE_CONTEXT_GENERATOR_SOURCE_COUNT')
    for row in sources:
        verify(row)
    require(sources[0]['sha256'] == expected['util/generate.py']['sha256']
            and sources[1]['sha256'] == expected['models/memit/memit_main.py']['sha256'],
            'NATIVE_CONTEXT_GENERATOR_SOURCE_IDENTITY')
    require(_function_ast(bundle.memit, 'get_context_templates')
            == _function_ast(bundle.alphaedit, 'get_context_templates'),
            'NATIVE_CONTEXT_FUNCTION_AST_PARITY')
    require(ready['contexts_member']['sha256'] == binding['contexts']['sha256']
            and ready['model_asset_identity'] == config['model_asset_identity']
            and ready['ordered_ids_sha256'] == config['ordered_ids_sha256']
            and ready['reference_fits'] == ready['updates'] == ready['history_appends'] == 0
            and ready['RNG_restored'] is True
            and ready['native_context_calls'] == [
                dict(prompts=5, n_gen_per_prompt=1, max_out_len=10)],
            'NATIVE_CONTEXT_COLD_GENERATION_RECEIPT')
    require(runtime['cold_W0_H0'] == config['cold_W0_H0']
            and runtime['torch'] == config['runtime']['torch']
            and runtime['transformers'] == config['runtime']['transformers']
            and runtime['FP32'] and runtime['eager']
            and not runtime['TF32'] and not runtime['autocast'],
            'NATIVE_CONTEXT_COLD_RUNTIME_IDENTITY')
    contexts = json.loads(contexts_path.read_text())
    require(len(contexts) == 2 and contexts[0] == ['{}'] and len(contexts[1]) == 5
            and all(type(value) is str for group in contexts for value in group),
            'NATIVE_CONTEXT_1_PLUS_5')
    require(module.CONTEXT_TEMPLATES_CACHE is None, 'NATIVE_CONTEXT_NO_MIXED_STATE')
    module.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(contexts)
    return dict(status='EXACT_COLD_NATIVE_CONTEXT_REUSED',
                contexts_sha256=binding['contexts']['sha256'],
                ready_sha256=binding['ready']['sha256'],
                origin_runtime_sha256=binding['runtime']['sha256'],
                MEMIT_Alpha_get_context_AST_equal=True, native_z_reuse=False,
                context_from_cold_W0=True, old_ours_H_is_not_baseline_H=True,
                generated_by_wrapper=False, native_cache_preserved=True)


class _LinalgCalls:
    def __init__(self, original, engine):
        self.original, self.engine = original, engine

    def __getattr__(self, name):
        return getattr(self.original, name)

    def solve(self, *args, **kwargs):
        return self.engine._call('solves', self.original.solve, *args, **kwargs)


class _TorchCalls:
    """Only this native main module sees the proxy; global torch is untouched."""
    def __init__(self, original, engine):
        self.original = original
        self.linalg = _LinalgCalls(original.linalg, engine)

    def __getattr__(self, name):
        return getattr(self.original, name)


class NativeEngine:
    """One independent cold native process; no dry execute or caller H append."""
    def __init__(self, module, hp, model, tokenizer, arm, context_receipt):
        require(arm in ('BASE_MEMIT', 'BASE_ALPHAEDIT'), 'NATIVE_ARM')
        require(not getattr(module, '_ODEEDIT_COUNTERS_BOUND', False), 'NATIVE_ENGINE_ALREADY_BOUND')
        self.module, self.hp, self.model, self.tokenizer = module, hp, model, tokenizer
        self.arm, self.writer = arm, 'memit' if arm == 'BASE_MEMIT' else 'alphaedit'
        self.context_receipt = context_receipt
        self.next_batch = 1
        self.cumulative = dict(native_z=0, write_keys=0, history_keys=0,
                               solves=0, history_appends=0, executes=0)
        self.current = None
        self.on_progress = None
        self.original_z, self.original_ks = module.compute_z, module.compute_ks
        self.original_torch = module.torch
        self.original_execute = (module.execute_memit if self.writer == 'memit'
                                 else module.execute_AlphaEdit)
        self.native_apply = (module.apply_memit_to_model if self.writer == 'memit'
                             else module.apply_AlphaEdit_to_model)
        module.compute_z = self._compute_z
        module.compute_ks = self._compute_ks
        module.torch = _TorchCalls(self.original_torch, self)
        if self.writer == 'memit':
            module.execute_memit = self._execute
        else:
            module.execute_AlphaEdit = self._execute
        module._ODEEDIT_COUNTERS_BOUND = True

    @property
    def counts(self):
        return {key: self.cumulative[key] for key in COUNT_FIELDS}

    @property
    def progress(self):
        return self.on_progress

    @progress.setter
    def progress(self, callback):
        require(callback is None or callable(callback), 'NATIVE_PROGRESS_CALLBACK')
        self.on_progress = callback

    def _call(self, key, original, *args, **kwargs):
        require(self.current is not None, 'NATIVE_INSTRUMENTATION_OUTSIDE_APPLY')
        self.current[key] += 1
        self.cumulative[key] += 1
        started = time.monotonic()
        try:
            return original(*args, **kwargs)
        finally:
            self.current['seconds'][key] += time.monotonic() - started

    def _compute_z(self, *args, **kwargs):
        result = self._call('native_z', self.original_z, *args, **kwargs)
        if self.on_progress is not None:
            try:
                self.on_progress(dict(batch=self.current['batch'],
                                      native_z=self.cumulative['native_z'],
                                      seconds=self.current['seconds']['native_z'],
                                      request_index=self.current['native_z'],
                                      native_z_completed=self.cumulative['native_z'],
                                      native_z_seconds=self.current['seconds']['native_z']))
            except Exception:
                self.current['logging_callback_errors'] += 1
        return result

    def _compute_ks(self, *args, **kwargs):
        ordinal = self.current['write_keys'] + self.current['history_keys']
        key = 'history_keys' if self.writer == 'alphaedit' and ordinal >= len(LAYERS) else 'write_keys'
        return self._call(key, self.original_ks, *args, **kwargs)

    def _execute(self, *args, **kwargs):
        history = getattr(self.module, 'cache_c', None) if self.writer == 'alphaedit' else None
        before = None if history is None else (history.data_ptr(), history._version)
        try:
            return self._call('executes', self.original_execute, *args, **kwargs)
        finally:
            if before is not None:
                actual = self.module.cache_c
                require(actual.data_ptr() == before[0], 'NATIVE_ALPHA_HISTORY_REPLACED')
                change = actual._version - before[1]
                # Native indexed `cache_c[i,:,:] += gram` performs in-place
                # add then indexed assignment, two version bumps per append.
                require(change >= 0 and change % 2 == 0, 'NATIVE_HISTORY_VERSION_SEMANTICS')
                count = change // 2
                self.current['history_appends'] += count
                self.current['history_tensor_version_delta'] += change
                self.cumulative['history_appends'] += count

    def history(self):
        if self.writer == 'memit' or not getattr(self.module, 'cache_c_new', False):
            return {}
        value = self.module.cache_c
        require(value.device.type == 'cpu' and value.dtype == torch.float32
                and value.ndim == 3 and value.shape[0] == len(LAYERS)
                and value.shape[1] == value.shape[2], 'NATIVE_ALPHA_HISTORY_SCHEMA')
        return {layer: value[index] for index, layer in enumerate(LAYERS)}

    def reset_history_to_uninitialized(self):
        """RAM rollback only, to the genuinely empty first Alpha entry state."""
        require(self.writer == 'alphaedit', 'MEMIT_HAS_NO_NATIVE_HISTORY_TO_RESET')
        self.module.cache_c = None
        self.module.cache_c_new = False

    def apply(self, records, batch):
        require(batch == self.next_batch and 1 <= batch <= 20, 'NATIVE_BATCH_SEQUENCE')
        requests = native_requests(records)
        self.current = dict(batch=batch, native_z=0, write_keys=0, history_keys=0,
                            solves=0, history_appends=0, executes=0,
                            history_tensor_version_delta=0, logging_callback_errors=0,
                            seconds={key: 0. for key in ('native_z', 'write_keys',
                                                       'history_keys', 'solves', 'executes')})
        started = time.monotonic()
        kwargs = dict(copy=False, return_orig_weights=False, cache_template=None)
        if self.writer == 'alphaedit':
            kwargs['reset_cache'] = batch == 1
        result = self.native_apply(self.model, self.tokenizer, requests, self.hp, **kwargs)
        require(isinstance(result, tuple) and len(result) == 2 and result[0] is self.model
                and isinstance(result[1], dict) and not result[1], 'NATIVE_APPLY_RETURN_IDENTITY')
        require(self.current['native_z'] == 100 and self.current['write_keys'] == 5
                and self.current['solves'] == 5 and self.current['executes'] == 1,
                'NATIVE_ACTUAL_CALL_COUNTS')
        alpha = self.writer == 'alphaedit'
        require(self.current['history_keys'] == (5 if alpha else 0)
                and self.current['history_appends'] == (5 if alpha else 0),
                'NATIVE_ACTUAL_HISTORY_ONCE')
        self.next_batch += 1
        receipt = dict(status='NATIVE_APPLY_RETURNED', arm=self.arm, writer=self.writer,
                       batch=batch, requests=100,
                       request_identity=digest(requests), counts=dict(self.current),
                       delta={key: self.current[key] for key in COUNT_FIELDS},
                       cumulative=self.counts, native_z_disk_cache=False,
                       cache_template=None, caller_history_appends=0,
                       Alpha_reset_cache=(batch == 1) if alpha else None,
                       native_has_history=alpha, same_model_returned=True,
                       returned_weights_copy_is_history=False,
                       seconds=time.monotonic() - started,
                       timing_policy='apply/execute inclusive; z/key/solve nested, not additive',
                       solve_timer='CPU dispatch wall only; no added GPU synchronization',
                       context=self.context_receipt, hparams=asdict(self.hp),
                       checkpoint_saved=False, exact_resume='NOT_AVAILABLE')
        return self.model, receipt


def prepare_native(config, model, tokenizer, arm):
    binding = config['native']
    require(binding['device'] == 0 and binding['logical_model_name'] == 'gpt2-xl',
            'NATIVE_DEVICE_LOGICAL_MODEL_BINDING')
    bundle = load_native(binding['root'], binding['closure'])
    writer = 'memit' if arm == 'BASE_MEMIT' else 'alphaedit'
    module = bundle.memit if writer == 'memit' else bundle.alphaedit
    cls = module.MEMITHyperParams if writer == 'memit' else module.AlphaEditHyperParams
    hp = cls.from_hparams(str(verify(binding['hparams'][writer])))
    require(hp.layers == list(LAYERS)
            and (hp.v_lr, hp.v_num_grad_steps, hp.v_loss_layer, hp.v_weight_decay,
                 hp.clamp_norm_factor, hp.kl_factor) == (.5, 20, 47, .5, .75, .0625),
            'NATIVE_PARSED_SCIENTIFIC_HPARAMS')
    require(hp.mom2_dataset == 'wikipedia' and hp.mom2_n_samples == 100000
            and hp.mom2_dtype == 'float32' and hp.fact_token == 'subject_last',
            'NATIVE_STATS_LOOKUP_FIELDS')
    require((hp.mom2_update_weight == 20000) if writer == 'memit'
            else (hp.L2 == 10 and hp.nullspace_threshold == .02), 'NATIVE_WRITER_COEFFICIENT')
    hp.device, hp.batch_size = 0, 100
    hp.stats_dir = str(Path(binding['stats_dir']).resolve())
    hp.model_name = binding['logical_model_name']
    if writer == 'alphaedit':
        hp.P_loc = str(verify(binding['projector']))
        require(Path(hp.P_loc).is_file(), 'NATIVE_P_MISSING_NO_RECOMPUTE')
    for layer in LAYERS:
        name = f'transformer.h.{layer}.mlp.c_proj_float32_mom2_100000.npz'
        require((Path(hp.stats_dir) / 'gpt2-xl' / 'wikipedia_stats' / name).is_file(),
                'NATIVE_C0_MISSING_NO_RECOMPUTE')
    require(model.config.model_type == 'gpt2' and model.config.n_positions == 1024,
            'NATIVE_GPT2_MODEL_POSITION_IDENTITY')
    for layer in LAYERS:
        weight = module.nethook.get_parameter(model, f'transformer.h.{layer}.mlp.c_proj.weight')
        require(tuple(weight.shape) == (6400, 1600) and weight.dtype == torch.float32,
                'NATIVE_GPT2_CONV1D_PARAMETER_LAYOUT')
    model.config._name_or_path = 'gpt2-xl'
    context_receipt = bind_context(bundle, config, module)
    return NativeEngine(module, hp, model, tokenizer, arm, context_receipt)
