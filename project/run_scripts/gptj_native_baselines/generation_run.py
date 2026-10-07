"""Six cold native GPT-J chains with explicit observer-only generation hooks.

Editing factories, R/P/N scoring and RAM transactions remain existing native
implementations. The SH2-owned bridge will adapt SH1's immutable shared
generation implementation; it is not yet bound. Milestone current results are
subsets of one measured prefix, never
additional generation. No checkpoint, scientific retry or GPU file polling.
"""
import argparse
import copy
import gc
import json
import math
import os
import random
import resource
import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import transformers

from .generation_common import (
    ARMS, ARM_LAYERS, MILESTONES, NONCE, SOURCE_ENV, TASK, authority, batches,
    digest, expected_counts, read, require, sha, stat_seal, verify, write,
    writer_identity,
)
from .generation_plan import ready
from project.run_scripts.gptj_cake_blue_prune_rect.metrics import (
    ObservationView, install_W0, observe, rows, state,
)
from project.run_scripts.gptj_cake_blue_prune_rect.run import NativeTransaction
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.writer import rng_equal, rng_snapshot
from scripts.fixed_counterfact import load_prefix


STOCK_ARMS = ('BASE_MEMIT', 'BASE_ALPHAEDIT')
HISTORY_ARMS = ('BASE_ALPHAEDIT', 'CAKE', 'ALPHAEDIT_BLUE')
BASE_FIX = 'PRUNE_TERMINAL_BASE_FIX'
IDENTITY_FIELDS = ('task_id', 'instruction_id', 'authority', 'resources', 'tracking',
                   'noCP', 'z_disk_cache', 'exact_resume')


def arm_configuration(config, arm):
    """Project the exact family binding, changing task metadata only."""
    require(arm in ARMS and set(config['arm_configs']) == set(ARMS), 'SIX_ARM_CONFIG_PROJECTION')
    result = copy.deepcopy(config['arm_configs'][arm])
    for field in IDENTITY_FIELDS:
        require(field in config, 'TASK_IDENTITY_FIELD:' + field)
        result[field] = copy.deepcopy(config[field])
    require(result['task_id'] == TASK and result['instruction_id'] == NONCE,
            'GENERATION_NATIVE_TASK_IDENTITY')
    require(result['noCP'] and not result['z_disk_cache']
            and result['exact_resume'] == 'NOT_AVAILABLE', 'GENERATION_NATIVE_NOCP')
    for field in ('model', 'model_revision', 'seed', 'runtime', 'stream'):
        require(field in result and result[field] == config[field], 'GENERATION_FAMILY_DRIFT:' + field)
    return result


def family_api(arm):
    require(arm in ARMS, 'UNKNOWN_NATIVE_ARM')
    if arm in STOCK_ARMS:
        from .native import normalize_requests, prepare_native
    elif arm in ('CAKE', 'ALPHAEDIT_BLUE'):
        from project.run_scripts.gptj_cake_blue_prune_rect.native_cake_blue import (
            normalize_requests, prepare_native,
        )
    else:
        from project.run_scripts.gptj_cake_blue_prune_rect.native_prune_rect import (
            normalize_requests, prepare_native,
        )
    return prepare_native, normalize_requests


def prepare_engine(config, model, tokenizer, arm, out):
    native_config = arm_configuration(config, arm)
    prepare, _ = family_api(arm)
    if arm in STOCK_ARMS:
        return prepare(native_config, model, tokenizer, arm)
    return prepare(native_config, model, tokenizer, arm, attempt=out)


def native_contexts(engine, arm):
    if arm in STOCK_ARMS:
        return copy.deepcopy(engine.module.CONTEXT_TEMPLATES_CACHE)
    return engine.contexts()


def initial_history(engine, arm, view):
    history = engine.history()
    expected = set(view.sites) if arm in ('CAKE', 'ALPHAEDIT_BLUE') else set()
    require(set(history) == expected, 'GENERATION_COLD_NATIVE_HISTORY_LAYOUT')
    require(all(bool((value == 0).all()) for value in history.values()),
            'GENERATION_COLD_NATIVE_HISTORY_NONZERO')
    return history


def locked(attempt):
    authority()
    config = read(attempt / 'config.json')
    lock = read(attempt / 'execution.lock.json')
    ready(config)
    require(config['instruction_id'] == lock['instruction_id'] == NONCE
            and config['task_id'] == TASK, 'GENERATION_TASK_AUTHORITY')
    require(os.environ.get(SOURCE_ENV) == lock['source_commit']
            and sha(attempt / 'config.json') == lock['config_sha256'], 'GENERATION_SOURCE_CONFIG_IDENTITY')
    for row in (lock['source_members'] + lock['runtime_sources'] + lock['launchers']
                + [lock['archive'], lock['tracking_env'], config['authority']]):
        verify(row)
    for row in config['assets']:
        stat_seal(row)
    for arm in ARMS:
        arm_configuration(config, arm)
    return config, lock


def production_ops():
    """Lazy imports keep CPU dispatch fixtures independent of SH1 availability."""
    from .generation_tracking import log_batch, log_generation, log_w0, safe_log
    return SimpleNamespace(state=state, observe=observe, install_W0=install_W0,
        rows=rows, transaction=NativeTransaction, write=write,
        log_w0=log_w0, log_batch=log_batch, log_generation=log_generation,
        safe_log=safe_log, batches=batches,
        reserve=lambda out, config: require(shutil.disk_usage(out).free
            >= config['resources']['reserve_bytes'], 'GENERATION_DISK_RESERVE'),
        cleanup=lambda: (gc.collect(), torch.cuda.empty_cache()))


def make_progress(engine, tracker, arm, safe_log):
    """Map existing fit scalars only; never equate stock z calls with Adam steps."""
    stock = dict(completed_z=0, evaluations=0, updates=0)

    def progress(value):
        current = engine.current or {}
        trace = current.get('fit_trace', [])
        last = trace[-1] if trace else None
        same = bool(last and last.get('request_index') == value.get('request_index'))

        def payload():
            data = dict(value)
            if same:
                for field in ('loss', 'nll_loss', 'kl_loss'):
                    if field in last:
                        data.setdefault(field, last[field])
            if arm in STOCK_ARMS:
                require(same and type(last.get('evaluations')) is int
                        and type(last.get('Adam_updates')) is int, 'STOCK_NATIVE_MEASURED_FIT_TRACE')
                completed = value['native_z_completed']
                require(completed == stock['completed_z'] + 1, 'STOCK_NATIVE_PROGRESS_CONTINUITY')
                require(1 <= last['evaluations'] <= 25 and 0 <= last['Adam_updates'] <= 24,
                        'STOCK_NATIVE_FIT_BUDGET_TELEMETRY')
                stock['completed_z'] = completed
                stock['evaluations'] += last['evaluations']
                stock['updates'] += last['Adam_updates']
                candidate, updates = stock['evaluations'], stock['updates']
            else:
                candidate, updates = data['fit_global_candidate'], data['fit_updates']
            result = dict(batch=data['batch'], candidate=data.get('request_index', data['native_z']),
                          **{'fit/global_candidate': candidate, 'optimizer/calls': updates})
            for field, metric in (('loss', 'fit/loss'), ('nll_loss', 'fit/nll'), ('kl_loss', 'fit/kl')):
                if field in data:
                    require(type(data[field]) in (int, float) and math.isfinite(data[field]),
                            'NONFINITE_NATIVE_PROGRESS_SCALAR')
                    result[metric] = data[field]
            if 'native_z_seconds' in data:
                result['time/phase_seconds'] = data['native_z_seconds']
            return result

        return safe_log(tracker, payload, 'native_fit_actual_candidate_axis')

    return progress


def guarded_generation(call, view, engine, arm, bench, ops, expected_cases):
    """Independent boundary guard, including native RNG/context/counters."""
    before = ops.state(view, engine.history())
    parameter_guard, hooks = view.guard(), view.hook_signature()
    counts = copy.deepcopy(engine.counts)
    contexts, bench_contexts = native_contexts(engine, arm), copy.deepcopy(bench.contexts)
    rng = rng_snapshot()
    try:
        result = call()
    finally:
        require(ops.state(view, engine.history()) == before
                and view.guard() == parameter_guard and view.hook_signature() == hooks
                and engine.counts == counts and native_contexts(engine, arm) == contexts
                and bench.contexts == bench_contexts, 'GENERATION_NATIVE_OBSERVER_MUTATION')
        require(rng_equal(rng), 'GENERATION_NATIVE_RNG_MUTATION')
    require(isinstance(result, dict) and isinstance(result.get('summary'), dict)
            and isinstance(result.get('cases'), list) and len(result['cases']) == expected_cases
            and result.get('identity') is not None, 'GENERATION_BRIDGE_RESULT_COVERAGE')
    return result


def generation_receipt(result, selected, endpoint, physical_state, raw_out, *, summary=None, subset=False):
    """Compact identity/summary only: raw text and tokens stay in bridge output."""
    reduced = result['summary'] if summary is None else summary
    require(isinstance(reduced, dict), 'GENERATION_SUBSET_SUMMARY')
    return dict(summary=reduced, identity=result['identity'], endpoint=endpoint,
                model_state=physical_state, requests=len(selected),
                cohort_identity=digest([record['case_id'] for record in selected]),
                raw_directory=str(raw_out), derived_subset=subset)


def execute_chain(config, lock, out, arm, model, tokenizer, view, engine, bench,
                  records, generation, tracker, *, ops=None, on_stage=None, on_commit=None):
    """Persistent production loop; injectable observers are CPU-only API fixtures."""
    ops = production_ops() if ops is None else ops
    native_config = arm_configuration(config, arm)
    _, normalize_requests = family_api(arm)
    out = Path(out)
    commits, cursor = [], []

    def stage(value):
        if on_stage is not None:
            on_stage(value)

    stage('W0_RPN_REUSE')
    w0_state = ops.state(view, engine.history())
    w0 = ops.install_W0(native_config, view, engine.history(), out, bench, records)
    w0raw = ops.rows(out / 'W0', w0_state)
    ops.log_w0(tracker, w0['summary'])
    stage('W0_GENERATION_LOAD_OR_PRIMARY_PUBLISH')
    w0gen = guarded_generation(generation.load_W0, view, engine, arm, bench, ops, len(records))
    w0gen_receipt = generation_receipt(w0gen, records, 'W0', w0_state, out / 'generation-W0')
    ops.write(out / 'generation-W0-reference.json', w0gen_receipt)
    ops.log_generation(tracker, 'W0_first2000', w0gen['summary'], 0, 0, 0)
    previous = ops.state(view, engine.history())
    engine.progress = make_progress(engine, tracker, arm, ops.safe_log)

    for number, current, seen in ops.batches(records):
        require(1 <= number <= 20 and len(current) == 100 and len(seen) == number * 100,
                'GENERATION_NATIVE_EXACT_20_PACKS')
        folder = out / f'batch-{number:02d}'
        ids, started = [record['case_id'] for record in current], time.monotonic()
        require(ops.state(view, engine.history()) == previous, 'GENERATION_BATCH_ENTRY_LINK')
        ops.reserve(out, config)
        normalized = normalize_requests(current)
        pack_identity = digest(normalized)
        stage(f'B{number}_PRE_RPN')
        pre = ops.observe(view, bench, seen, current, engine.history(), f'B{number}_PRE',
                          folder / 'pre', current_ids=ids)
        stage(f'B{number}_PRE_GENERATION')
        pre_directory = folder / 'generation-pre'
        pregen = guarded_generation(lambda: generation.endpoint(current, out=pre_directory,
            endpoint=f'B{number}_PRE', model_state=previous, cohort_label='CURRENT'),
            view, engine, arm, bench, ops, len(current))
        beforegen = generation_receipt(pregen, current, f'B{number}_PRE', previous, pre_directory)
        stage(f'B{number}_NATIVE_APPLY')
        with ops.transaction(view, engine, bench) as tx:
            returned, native = engine.apply(normalized, number)
            require(returned is model, 'SAME_ACCUMULATED_NATIVE_MODEL')
            require(all(bool(torch.isfinite(weight).all()) for weight in view.weights.values())
                    and all(bool(torch.isfinite(value).all()) for value in engine.history().values()),
                    'NATIVE_NONFINITE_COMMIT')
            expected = expected_counts(arm)
            counts = {field: native['delta'][field] for field in expected}
            require(counts == expected, 'NATIVE_CALL_HISTORY_COUNTS')
            require(set(engine.history()) == (set(view.sites) if arm in HISTORY_ARMS else set()),
                    'GENERATION_NATIVE_HISTORY_BOUNDARY')
            if number == 20 and arm == 'PRUNE':
                stage('TERMINAL_PRUNE_BASE_FIX')
                terminal = engine.terminal_prune()
                require(terminal.get('explicit_repair') == BASE_FIX and terminal.get('prune_applied') is True,
                        'PRUNE_EXPLICIT_TERMINAL_BASE_FIX')
                native.update(terminal_prune=terminal, prune_applied=True, explicit_repair=BASE_FIX)
            selected = seen if number in MILESTONES else current
            stage(f'B{number}_POST_RPN')
            post = ops.observe(view, bench, seen, selected, engine.history(), f'W{number}',
                               folder / 'post', current_ids=ids)
            after = ops.state(view, engine.history())
            stage(f'B{number}_POST_GENERATION')
            post_directory = folder / 'generation-post'
            postgen = guarded_generation(lambda: generation.endpoint(selected, out=post_directory,
                endpoint=f'W{number}', model_state=after,
                cohort_label='ALL_SEEN' if number in MILESTONES else 'CURRENT'),
                view, engine, arm, bench, ops, len(selected))
            currentgen = (generation.subset(postgen['cases'], current)
                          if number in MILESTONES else postgen['summary'])
            aftergen = generation_receipt(postgen, current, f'W{number}', after,
                                           post_directory, summary=currentgen,
                                           subset=number in MILESTONES)
            prefixgen = (generation_receipt(postgen, seen, f'W{number}', after, post_directory)
                         if number in MILESTONES else None)
            contexts = native_contexts(engine, arm)
            require(isinstance(contexts, list) and len(contexts) == 2 and contexts[0] == ['{}']
                    and len(contexts[1]) == 5, 'GENERATION_NATIVE_CONTEXT_CARRIED_1_PLUS5')
            proposed_cursor = cursor + ids
            receipt = dict(task=TASK, arm=arm, writer=writer_identity(arm), batch=number,
                case_ids=ids, source=lock['source_commit'], config=digest(config),
                before=previous, after=after, native=native, native_counts=counts,
                pre=pre['summary'], post=post['summary'], post_current=post['current'],
                post_scope='ALL_SEEN' if number in MILESTONES else 'CURRENT',
                seen_requests=len(seen), native_pack=pack_identity, ledger=digest(proposed_cursor),
                gen_before=beforegen, gen_current=aftergen, gen_prefix=prefixgen,
                gen_W0=dict(identity=w0gen['identity'], reference=str(out / 'generation-W0-reference.json')),
                observer_no_mutation=True, generation_observer_no_mutation=True,
                seconds=time.monotonic() - started, checkpoint_saved=False,
                exact_resume='NOT_AVAILABLE')
            ops.write(folder / 'native-context-identity.json', dict(context_sha256=digest(contexts),
                groups=[len(group) for group in contexts], generated_inside_native=True))
            tx.finish()
            try:
                ops.write(folder / 'commit.json', receipt)
            except BaseException:
                tx.done = False
                raise
        commits.append(receipt)
        cursor, previous = proposed_cursor, after
        if on_commit is not None:
            on_commit(receipt)
        ops.log_batch(tracker, receipt, w0raw, ids, [record['case_id'] for record in seen])
        pre_edits, post_edits = (number - 1) * 100, number * 100
        ops.log_generation(tracker, 'current/pre', pregen['summary'], post_edits, pre_edits, post_edits)
        ops.log_generation(tracker, 'current/post', currentgen, post_edits, pre_edits, post_edits)
        if prefixgen is not None:
            ops.log_generation(tracker, 'all_seen/post', postgen['summary'], post_edits, pre_edits, post_edits)
        print(json.dumps(dict(event='native_generation_batch_committed', arm=arm,
                             batch=number, edits=len(cursor), native_counts=counts)), flush=True)
        ops.cleanup()
    require(len(commits) == 20 and len(cursor) == 2000, 'GENERATION_FULL_20_BATCH_COMPLETION')
    return dict(status='COMPLETED', completed_batches=20, commits=20, edits=2000,
                state=previous, native_counts=engine.counts, source=lock['source_commit'],
                config=digest(config), generation_W0_identity=w0gen['identity'],
                generation_endpoints=40, checkpoint_saved=False, exact_resume='NOT_AVAILABLE')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    attempt, arm = args.attempt.resolve(), args.arm
    out = attempt / arm
    require(not out.exists(), 'CREATE_ONCE_GENERATION_ARM')
    out.mkdir()
    started = time.monotonic()
    progress_state = dict(stage='LOCK', commits=0, rollback_verified=False)
    engine = tracker = None
    terminal = {}

    def stage(value):
        progress_state['stage'] = value

    def committed(receipt):
        progress_state['commits'] += 1

    try:
        config, lock = locked(attempt)
        native_config = arm_configuration(config, arm)
        require(str(torch.__version__) == config['runtime']['torch']
                and transformers.__version__ == config['runtime']['transformers'], 'PINNED_RUNTIME')
        torch.set_num_threads(config['resources']['cpu'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        records = load_prefix(Path(config['stream']).parent, 2000)
        list(batches(records))
        stage('ONLINE_STARTUP')
        from .generation_tracking import start_tracking
        tracker = start_tracking(config, lock, out, arm)
        stage('LOAD_COLD_W0')
        loading = time.monotonic()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(config['model'], local_files_only=True,
            dtype=torch.float32, attn_implementation='eager', low_cpu_mem_usage=True,
            use_safetensors=False).to('cuda').eval()
        tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
        tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, 'right'
        model.config.use_cache = False
        view = ObservationView(model, ARM_LAYERS[arm])
        require(state(view, {})['W'] == {str(layer): native_config['cold_W'][str(layer)]
                                       for layer in view.sites}, 'ACTUAL_COLD_W0')
        require(model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr(),
                'GPTJ_UNTIED_HEAD')
        stage('NATIVE_PREPARATION')
        engine = prepare_engine(config, model, tokenizer, arm, out)
        initial_history(engine, arm, view)
        if arm not in STOCK_ARMS:
            stage('NATIVE_CONTEXT_FIRST_COLD_PREPARATION')
            engine.prepare_contexts()
            contexts = native_contexts(engine, arm)
            write(out / 'native-contexts.json', dict(contexts=contexts,
                context_sha256=digest(contexts), seed=config['seed'], native_generator=True,
                source=lock['source_commit'], training_context_not_evaluator_input=True))
        bench = CounterFactAdapter(tokenizer, [])
        write(out / 'runtime.json', dict(device=torch.cuda.get_device_name(),
            torch=str(torch.__version__), transformers=transformers.__version__,
            model=config['model'], FP32=True, eager=True, TF32=False, autocast=False,
            CPU_threads=config['resources']['cpu'], source=lock['source_commit'],
            config=digest(config), arm=arm, job=os.environ['SLURM_JOB_ID'],
            checkpoint_saved=False, initial_history_zero=True,
            initial_state=state(view, engine.history()),
            native_solve_dtype='FP64' if arm in ('BASE_MEMIT', 'PRUNE', 'RECT') else 'native FP32',
            load_seconds=time.monotonic() - loading))
        from .generation_bridge import GenerationObserver
        generation = GenerationObserver(config, lock, view, engine, tokenizer, records, out, arm)
        ops = production_ops()
        original_transaction = ops.transaction

        def transaction(*args):
            tx = original_transaction(*args)
            original_exit = tx.__class__.__exit__

            class ObservedTransaction:
                def __enter__(self):
                    return tx.__enter__()

                def __exit__(self, *exit_args):
                    try:
                        return original_exit(tx, *exit_args)
                    finally:
                        progress_state['rollback_verified'] = tx.rollback_verified

            return ObservedTransaction()

        ops.transaction = transaction
        terminal = execute_chain(config, lock, out, arm, model, tokenizer, view,
                                 engine, bench, records, generation, tracker,
                                 ops=ops, on_stage=stage, on_commit=committed)
    except BaseException as error:
        terminal = dict(status='FAILED', stage=progress_state['stage'],
            error_type=type(error).__name__, error=str(error)[:1800],
            completed_batches=progress_state['commits'], commits=progress_state['commits'],
            source=locals().get('lock', {}).get('source_commit', os.environ.get(SOURCE_ENV)),
            config=digest(config) if 'config' in locals() else None,
            native_counts=engine.counts if engine is not None else {field: 0 for field in expected_counts(arm)},
            rollback_verified=progress_state['rollback_verified'],
            checkpoint_saved=False, exact_resume='NOT_AVAILABLE')
        write(out / 'failure.json', terminal)
        traceback.print_exc(limit=8)
    finally:
        if tracker is not None:
            try:
                write(out / 'tracking-finish.json', tracker.finish(
                    exit_code=0 if terminal.get('status') == 'COMPLETED' else 1, timeout=45))
            except Exception as error:
                write(out / 'tracking-finish-error.json', dict(type=type(error).__name__, science_not_restarted=True))
        terminal.update(program_seconds=time.monotonic() - started,
            peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        write(out / 'terminal.json', terminal)
    if terminal['status'] != 'COMPLETED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
