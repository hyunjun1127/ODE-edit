"""Two independent cold native PRUNE/RECT chains; one apply per batch.

Native target fit/write mathematics live in the immutable private native copy.
This module supplies input/order guards, pre/post observation and RAM rollback.
"""
import argparse
import gc
import json
import os
import random
import resource
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import TASK, NONCE, ARMS, MILESTONES, digest, require, sha, verify, write, batches, stat_seal
from .metrics import ObservationView, state, observe, rows, install_W0
from .native import prepare_native, native_requests
from project.run_scripts.experiment_tracking import init
from project.run_scripts.jlz_price_gpt2xl.tracking import SCHEMA, log_w0, log_batch
from project.run_scripts.jlz_interference_l1.cap_tracking import safe_log
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.writer import rng_snapshot, rng_restore, rng_equal
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.base_model_eval.gpt2xl_server1_common import token_identity

SOURCE_ENV = 'GPT2_PRUNE_RECT_SOURCE_COMMIT'
ARM_LAYERS = {'PRUNE': (13, 14, 15, 16, 17), 'RECT': (13, 14, 15, 16, 17)}
EXPECTED = {
    'PRUNE': dict(native_z=100, write_keys=5, history_keys=0, solves=5, history_appends=0),
    'RECT': dict(native_z=100, write_keys=5, history_keys=0, solves=5, history_appends=0)}


def locked(attempt):
    c = json.loads((attempt / 'config.json').read_text())
    lock = json.loads((attempt / 'execution.lock.json').read_text())
    require(c['instruction_id'] == lock['instruction_id'] == NONCE
            and c['task_id'] == TASK, 'TASK_AUTHORITY')
    require(os.environ.get(SOURCE_ENV) == lock['source_commit']
            and sha(attempt / 'config.json') == lock['config_sha256'], 'SOURCE_CONFIG_IDENTITY')
    for item in (lock.get('source_members', []) + lock.get('runtime_sources', [])
                 + lock.get('launchers', []) + lock.get('native_closure', [])):
        verify(item)
    require(lock.get('native_closure'), 'NATIVE_SOURCE_CLOSURE_REQUIRED')
    for item in c['assets']:
        stat_seal(item)
    require(subprocess.check_output(['git','-C',c['model'],'rev-parse','HEAD'],text=True).strip()
        == c['model_revision'],'RUNTIME_MODEL_CHECKOUT_REVISION')
    for item in c['runtime'].get('source_members', []) + c.get('evaluator_sources', []):
        verify(item)
    for key in ('archive', 'tracking_env'):
        if key in lock:
            verify(lock[key])
    for key in ('authority', 'contract', 'observer_identity'):
        if key in c:
            verify(c[key])
    require(c.get('noCP') is True and not c.get('z_disk_cache')
            and c.get('exact_resume') == 'NOT_AVAILABLE', 'NOCP_ZCACHE')
    return c, lock


def check_observer_identity(c, records, bench):
    expected = json.loads(verify(c['observer_identity']).read_text())['rows']
    actual = token_identity(records, bench)
    require(len(actual) == 26000 and actual == expected, 'ACTUAL_OBSERVER_TOKEN_ROW_ORDER')
    return digest(actual)


def nonselected(view, arm):
    selected = {id(view.weights[layer]) for layer in ARM_LAYERS[arm]}
    return tuple((name, id(p), p.data_ptr(), p._version, tuple(p.shape), str(p.dtype))
                 for name, p in view.model.named_parameters() if id(p) not in selected)


class NativeTransaction:
    """RAM-only rollback until the native write, observer and receipt all succeed."""
    def __init__(self, view, engine, bench, arm):
        self.view, self.engine, self.bench, self.arm = view, engine, bench, arm
        self.done = self.rollback_verified = False

    def __enter__(self):
        self.before = state(self.view, self.engine.history())
        self.guard = nonselected(self.view, self.arm)
        self.hooks = self.view.hook_signature()
        self.rng = rng_snapshot()
        self.observer_context = json.loads(json.dumps(self.bench.contexts))
        self.W = {layer: self.view.weights[layer].detach().cpu().clone()
                  for layer in ARM_LAYERS[self.arm]}
        require(not self.engine.history(), 'NO_NATIVE_HISTORY_SNAPSHOT')
        self.native_context = self.engine.context_snapshot()
        self.native_state = self.engine.snapshot_native_state()
        self.native_signature = self.engine.native_state_signature()
        return self

    def finish(self):
        require(nonselected(self.view, self.arm) == self.guard
                and self.view.hook_signature() == self.hooks
                and self.bench.contexts == self.observer_context, 'NATIVE_COMMIT_GUARD')
        self.done = True

    def __exit__(self, kind, value, tb):
        try:
            if not self.done:
                with torch.no_grad():
                    for layer, before in self.W.items():
                        self.view.weights[layer].copy_(before)
                self.engine.restore_native_state(self.native_state)
                rng_restore(self.rng)
                require(state(self.view, self.engine.history()) == self.before
                        and nonselected(self.view, self.arm) == self.guard
                        and self.view.hook_signature() == self.hooks
                        and self.bench.contexts == self.observer_context
                        and self.engine.context_snapshot() == self.native_context
                        and self.engine.native_state_signature() == self.native_signature
                        and rng_equal(self.rng), 'NATIVE_ROLLBACK_MISMATCH')
                self.rollback_verified = True
        finally:
            self.W.clear()
            self.native_state.clear()


def start_tracking(c, lock, out, arm):
    cfg = dict(server='server1', task_id=TASK, arm=arm, attempt=c['run_instance']['attempt'],
        source_sha=lock['source_commit'], config_sha=lock['config_sha256'],
        job_id=os.environ['SLURM_JOB_ID'], model='gpt2xl', model_family='gpt2',
        writer='prune' if arm == 'PRUNE' else 'rect', role='scientific', metric_schema=SCHEMA)
    tracker = init(env_file=c['tracking']['env_file'], spool=out / 'tracking', config=cfg)
    write(out / 'tracking-identity.json', dict(run_id=tracker.run_id,
        url=tracker.startup.get('url'), config=tracker.config_values,
        source_sha=lock['source_commit'], config_sha=lock['config_sha256'],
        startup_readback=tracker.startup, scientific_complete=False))
    return tracker


def check_native(arm, native, engine, batch):
    require(native['delta'] == EXPECTED[arm], 'NATIVE_CALL_HISTORY_COUNTS')
    require(native.get('arm', arm) == arm and native.get('batch', batch) == batch,
            'NATIVE_RECEIPT_IDENTITY')
    require(not engine.history(), 'NATIVE_NO_HISTORY_BOUNDARY')
    require(native['same_model_returned'] is True and native['native_has_history'] is False
            and native['caller_history_appends'] == 0 and native['cache_template'] is None
            and native['native_z_disk_cache'] is False and native['checkpoint_saved'] is False,
            'NATIVE_NO_EXTRA_APPLY_CACHE_OR_HISTORY')


def finish_native_batch(engine, arm, batch, view):
    """PRUNE terminal transform is before the sole final endpoint observation."""
    dense_state = state(view, engine.history())
    transform = engine.finish_batch(batch)
    require(isinstance(transform, dict), 'TERMINAL_TRANSFORM_RECEIPT')
    expected = arm == 'PRUNE' and batch == 20
    require(transform['prune_applied'] is expected, 'PRUNE_TERMINAL_ONLY_ONCE')
    if expected:
        require(transform['repair'] == 'PRUNE_TERMINAL_BASE_FIX'
                and transform['repair_authorized'] is True
                and transform['upstream_bitwise_equivalence'] is False
                and transform['terminal_transforms'] == 1
                and transform['final_base'] == 'SAVED_COLD_W0'
                and transform['spectrum_formula_changed'] is False
                and transform['checkpoint_saved'] is False, 'PRUNE_EXPLICIT_BASE_FIX_BOUNDARY')
    else:
        require(state(view, engine.history()) == dense_state, 'NO_NONTERMINAL_TRANSFORM')
    require(all(bool(torch.isfinite(w).all()) for w in view.weights.values()),
            'TERMINAL_SELECTED_NONFINITE')
    return dense_state, transform


def native_loop(c, lock, out, arm, records, model, view, engine, bench, tracker, commits):
    """Production loop separated for CPU fake-native/observer regression."""
    previous = state(view, engine.history())
    cursor = []
    w0raw = rows(out / 'W0', previous)
    for number, current, seen in batches(records):
        folder = out / f'batch-{number:02d}'
        ids = [r['case_id'] for r in current]
        started = time.monotonic()
        require(state(view, engine.history()) == previous, 'BATCH_ENTRY_LINK')
        requests = native_requests(current)
        pack = dict(record_ids=[r['case_id'] for r in current],
            identity=digest(dict(requests=requests,contexts=engine.context_snapshot(),
                                 hparams=c['native']['effective_hparams'][arm])))
        expected = c['packs'][number - 1]
        require(pack['record_ids'] == expected['ids']
                and (expected.get('identity') is None or pack['identity'] == expected['identity']),
                'NATIVE_TOKEN_ORDER_CONTEXT')
        pre = observe(view, bench, seen, current, engine.history(), f'B{number}_PRE',
                      folder / 'pre', current_ids=ids)
        with NativeTransaction(view, engine, bench, arm) as tx:
            returned, native = engine.apply(requests, number)
            require(returned is model, 'SAME_ACCUMULATED_NATIVE_MODEL')
            require(all(bool(torch.isfinite(w).all()) for w in view.weights.values())
                    and all(bool(torch.isfinite(h).all()) for h in engine.history().values()),
                    'NATIVE_NONFINITE_COMMIT')
            check_native(arm, native, engine, number)
            native_after, transform = finish_native_batch(engine, arm, number, view)
            selected = seen if number in MILESTONES else current
            post = observe(view, bench, seen, selected, engine.history(), f'W{number}',
                           folder / 'post', current_ids=ids)
            after = state(view, engine.history())
            receipt = dict(task=TASK, arm=arm, batch=number, case_ids=ids,
                source=lock['source_commit'], config=digest(c), before=previous, after=after,
                native=native, native_counts=native['delta'], pre=pre['summary'], post=post['summary'],
                native_after=native_after, terminal_transform=transform,
                prune_applied=transform['prune_applied'],
                post_current=post['current'],
                post_scope='ALL_SEEN' if number in MILESTONES else 'CURRENT',
                seen_requests=len(seen), native_pack=pack['identity'],
                ledger=digest(cursor + ids), observer_no_mutation=True,
                seconds=time.monotonic() - started, checkpoint_saved=False,
                exact_resume='NOT_AVAILABLE')
            tx.finish()
            try:
                write(folder / 'commit.json', receipt)
            except BaseException:
                tx.done = False
                raise
        commits.append(receipt)
        cursor.extend(ids)
        previous = after
        accepted = log_batch(tracker, receipt, w0raw, ids, [r['case_id'] for r in seen])
        write(folder / 'logging.json', dict(accepted=accepted,
            SDK_acceptance_is_not_remote_readback=True, scientific_state_unchanged=True))
        print(json.dumps(dict(event='native_batch_committed', arm=arm, batch=number,
            edits=len(cursor), native_counts=native['delta'])), flush=True)
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    require(len(commits) == 20 and len(cursor) == 2000, 'FULL_20_BATCH_COMPLETION')
    return previous


def run(attempt, arm):
    attempt = Path(attempt).resolve()
    require(arm in ARMS, 'ARM_IDENTITY')
    out = attempt / arm
    require(not out.exists(), 'CREATE_ONCE_ARM')
    out.mkdir()
    started = time.monotonic()
    stage, commits, engine, tracker, terminal = 'LOCK', [], None, None, {}
    try:
        c, lock = locked(attempt)
        require(str(torch.__version__) == c['runtime']['torch']
                and transformers.__version__ == c['runtime']['transformers'], 'PINNED_RUNTIME')
        torch.set_num_threads(c['resources']['cpu'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
        stage = 'ONLINE_STARTUP'
        tracker = start_tracking(c, lock, out, arm)
        stage, loaded = 'LOAD_W0', time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(c['model'], local_files_only=True,
            dtype=torch.float32, attn_implementation='eager', low_cpu_mem_usage=True,
            use_safetensors=True).to('cuda').eval()
        tok = AutoTokenizer.from_pretrained(c['model'], local_files_only=True)
        tok.pad_token = tok.eos_token
        tok.padding_side = 'right'
        view = ObservationView(model)
        require(state(view, {})['W'] == c['cold_W'], 'ACTUAL_COLD_W0')
        require(model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr(),
                'GPT2_TIED_HEAD')
        engine = prepare_native(c, model, tok, arm, attempt=attempt)
        require(not engine.history(), 'COLD_NATIVE_NO_HISTORY')
        stage = 'NATIVE_COLD_CONTEXT_PREPARATION'
        native_contexts = engine.prepare_contexts()
        require(isinstance(native_contexts, list) and native_contexts, 'NATIVE_CONTEXT_READY')
        write(out / 'native-contexts.json', dict(contexts=native_contexts,
            identity=digest(native_contexts), generator=engine.context_receipt,
            local_only=True, checkpoint_saved=False, raw_Git_W_B=False))
        bench = CounterFactAdapter(tok, native_contexts)
        records = load_prefix(Path(c['stream']).parent, 2000)
        list(batches(records))
        observer_token_order = check_observer_identity(c, records, bench)
        write(out / 'runtime.json', dict(device=torch.cuda.get_device_name(),
            torch=str(torch.__version__), transformers=transformers.__version__, model=c['model'],
            FP32=True, eager=True, TF32=False, autocast=False, CPU_threads=c['resources']['cpu'],
            source=lock['source_commit'], config=digest(c), arm=arm,
            job=os.environ['SLURM_JOB_ID'], checkpoint_saved=False,
            native_solve_dtype='native FP64 solve followed by FP32 add',
            cold_state=state(view, engine.history()), cold_history_zero_verified=True,
            native_context_identity=digest(native_contexts),
            observer_token_order_identity=observer_token_order,
            load_seconds=time.monotonic() - loaded))
        stage = 'W0_OBSERVER'
        w0 = install_W0(c, view, engine.history(), out, bench, records)
        write(out / 'W0/logging.json', dict(accepted=log_w0(tracker, w0['summary']),
            SDK_acceptance_is_not_remote_readback=True))
        def progress(value):
            safe_log(tracker, lambda:dict(batch=value['batch'],
                candidate=value['fit_global_candidate'],
                **{'fit/global_candidate':value['fit_global_candidate'],
                   'optimizer/calls':value['fit_updates']}), 'actual_native_candidate_axis')
        engine.progress = progress
        stage = 'NATIVE_20_BATCH_LOOP'
        final_state = native_loop(c, lock, out, arm, records, model, view, engine, bench,
                                  tracker, commits)
        terminal = dict(status='COMPLETED', completed_batches=20, commits=20, edits=2000,
            state=final_state, native_counts=engine.counts, source=lock['source_commit'],
            config=digest(c), terminal_transforms=1 if arm == 'PRUNE' else 0,
            prune_applied=arm == 'PRUNE', repair='PRUNE_TERMINAL_BASE_FIX' if arm == 'PRUNE' else None,
            checkpoint_saved=False, exact_resume='NOT_AVAILABLE')
    except BaseException as error:
        terminal = dict(status='FAILED', stage=stage, error_type=type(error).__name__,
            error=str(error)[:1800], completed_batches=len(commits), commits=len(commits),
            source=locals().get('lock', {}).get('source_commit', os.environ.get(SOURCE_ENV)),
            config=digest(c) if 'c' in locals() else None,
            native_counts=engine.counts if engine else {k:0 for k in EXPECTED['PRUNE']},
            checkpoint_saved=False, exact_resume='NOT_AVAILABLE')
        write(out / 'failure.json', terminal)
        traceback.print_exc(limit=8)
    finally:
        if tracker:
            try:
                write(out / 'tracking-finish.json', tracker.finish(
                    exit_code=0 if terminal.get('status') == 'COMPLETED' else 1, timeout=45))
            except Exception as error:
                write(out / 'tracking-finish-error.json', dict(type=type(error).__name__,
                    science_not_restarted=True))
        terminal.update(program_seconds=time.monotonic() - started,
            peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        write(out / 'terminal.json', terminal)
    return terminal


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--attempt', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    args = p.parse_args()
    if run(args.attempt, args.arm)['status'] != 'COMPLETED':
        raise SystemExit(1)
