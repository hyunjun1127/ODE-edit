"""One native cold BS100x20 baseline, with generation observers and no CP.

All editing is the bound original native apply. Observation is not training;
successful transactions persist RAM W/H, while a technical/IO fault restores
the failing batch entry and leaves its earlier sealed prefix untouched.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import resource
import shutil
import time
import traceback

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.observe import active_flags, reduce_rows
from project.run_scripts.jlz_realization.writer import rng_snapshot
from project.run_scripts.jlz_interference_l1.cap_tracking import safe_log, w0_subset

from .common import (LOCAL, METHODS, MILESTONES, NONCE, TASK, authority, batches,
                     digest, member, require, sha, stat_seal, verify, write)
from .generation import NativeGeneration, endpoint_ref, normalize_summary, require_binding
from .metrics import ObservationView, cold_weight_identity, install_W0, observe, rows, state
from .native import prepare_native
from .producer import (endpoint_values, generation_values, scientific_config,
                       transport_support, w0_values)
from .transaction import NativeTransaction

SOURCE_ENV = 'LLAMA3_NATIVE_BASELINES_SOURCE_COMMIT'


def rng_identity():
    from project.run_scripts.jlz_realization.common import tensor_sha
    value = rng_snapshot()
    return digest([repr(value[0]), value[1][0], value[1][1].tolist(), repr(value[1][2:]),
                   tensor_sha(value[2]), [tensor_sha(item) for item in value[3]]])


def sequence(number):
    require(type(number) is int and 1 <= number <= 20, 'NO_B21')
    return dict(number=number, requests=100, edits=number*100,
                pre_edits=(number-1)*100, milestone=number in MILESTONES)


def terminal_before_observation(engine, number):
    """Only approved PRUNE deployment compression, exactly once before W20."""
    sequence(number)
    return engine.terminal_prune() if number == 20 and engine.method == 'PRUNE' else None


def storage_guard(out, needed):
    require(type(needed) is int and needed > 0, 'SEALED_STORAGE_RESERVE_MISSING')
    value = shutil.disk_usage(out)
    inodes = os.statvfs(out)
    require(value.free >= needed and inodes.f_favail >= 1000, 'RESOURCE_BLOCKED_STORAGE')
    return dict(free_bytes=value.free, required_bytes=needed, free_inodes=inodes.f_favail,
                no_deletion=True, no_observation_reduction=True)


def seal_commit(transaction, path, receipt):
    """Receipt IO/hash is part of RAM commit, not an operation after it."""
    transaction.finish()
    try:
        write(path, receipt)
        sealed = member(path)
        verify(sealed)
        return sealed
    except BaseException:
        transaction.invalidate()
        raise


def verified_prefix(progress):
    """RAM success counter, not files that may survive a rolled-back IO fault."""
    value = progress['commits']
    require(type(value) is int and 0 <= value <= 20, 'SEALED_PREFIX_COUNTER')
    return value


def locked(attempt, method):
    attempt = Path(attempt).resolve()
    require(attempt.is_relative_to(LOCAL) and method in METHODS, 'TASK_ATTEMPT_METHOD_SCOPE')
    c = json.loads((attempt/'config.json').read_text())
    lock = json.loads((attempt/'execution.lock.json').read_text())
    require(c['task_id'] == lock['task_id'] == TASK and
            c['instruction_id'] == lock['instruction_id'] == NONCE, 'AUTHORITY_TASK')
    require(os.environ.get(SOURCE_ENV) == lock['source_commit'] and
            sha(attempt/'config.json') == lock['config_sha256'], 'SOURCE_CONFIG_LOCK')
    authority()
    for key in ('source_members', 'runtime_sources', 'generation_sources', 'native_sources', 'launchers'):
        for row in lock.get(key, []):
            verify(row)
    require(bool(lock.get('source_members')) and bool(lock.get('generation_sources')),
            'EXECUTION_SOURCE_CLOSURE_MISSING')
    require_binding(c)
    require(c['noCP'] and c['exact_resume'] == 'NOT_AVAILABLE' and
            len(c['packs']) == 20 and c['resources']['per_job_gpu'] == 1 and
            c['resources']['project_cap'] <= 3 and c['resources']['task_cap'] <= 3,
            'SEALED_HORIZON_RESOURCES_NO_CP')
    for row in c['assets']:
        stat_seal(row)
    require(os.environ.get('SLURM_JOB_ID', '').isdigit() and
            int(os.environ['SLURM_JOB_ID']) > 0, 'ACTUAL_SLURM_JOB_ID_REQUIRED')
    return c, lock


def setup(c, out, method, lock):
    require(torch.__version__ == c['runtime']['torch'] and
            transformers.__version__ == c['runtime']['transformers'], 'RUNTIME_VERSION')
    threads = c['resources']['cpu']
    torch.set_num_threads(threads)
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(c['model'], local_files_only=True,
        dtype=torch.float32, attn_implementation='eager', low_cpu_mem_usage=True,
        device_map={'': 'cuda:0'}).eval()
    tok = AutoTokenizer.from_pretrained(c['model'], local_files_only=True)
    tok.pad_token = tok.eos_token; tok.padding_side = 'right'
    engine = prepare_native(c, model, tok, method, attempt=out.parent)
    contexts = engine.prepare_contexts()
    bench = CounterFactAdapter(tok, contexts)
    view = ObservationView(model, sites=engine.layers)
    require(cold_weight_identity(model) == c['cold_W'], 'NATIVE_ACTUAL_COLD_W')
    require(all(bool((value == 0).all()) for value in engine.history().values()),
            'NATIVE_ACTUAL_EMPTY_OR_ZERO_H')
    records = load_prefix(Path(c['stream']).parent, 2000)
    list(batches(records))
    runtime = dict(task_id=TASK, method=method, source=lock['source_commit'],
        config=lock['config_sha256'], job=os.environ['SLURM_JOB_ID'],
        device=torch.cuda.get_device_name(), model=c['model'], torch=torch.__version__,
        transformers=transformers.__version__, FP32=True, eager=True, TF32=False,
        autocast=False, CPU_threads=threads, cold_W0_H0=state(view, engine.history()),
        full_five_cold_W=c['cold_W'], native_context=digest(engine.contexts()),
        checkpoint_saved=False, exact_resume='NOT_AVAILABLE')
    write(out/'runtime.json', runtime)
    return view, engine, bench, records


def rpn_subset(source_rows, selected, seen, before, endpoint, folder):
    """CPU reuse of an already measured identical endpoint; no extra forward."""
    ids = [record['case_id'] for record in selected]
    flags = active_flags(seen)
    chosen = set(ids)
    selected_rows = [dict(row, endpoint=endpoint,
        active_at_endpoint=flags[row['case_id']]) for row in source_rows if row['case_id'] in chosen]
    require([row['case_id'] for row in selected_rows if row['kind'] == 'R'] == ids,
            'RPN_SUBSET_OCCURRENCE_ORDER')
    summary = reduce_rows(selected_rows)
    for start in range(0, len(selected_rows), 50*13):
        write(Path(folder)/f'chunk-{start:04d}.json', dict(schema='jlz-observer-rows-v1',
            state=before, rows=selected_rows[start:start+50*13], optimizer_feedback=False))
    result = dict(schema='jlz-observer-summary-v1', endpoint=endpoint, state=before,
        requests=len(selected), summary=summary, current=summary, row_count=len(selected_rows),
        row_order=digest([row['identity'] for row in selected_rows]), seconds=0.,
        new_forwards=0, no_mutation=True, RNG_restored=True, optimizer_feedback=False,
        replay=False, reused_same_endpoint=True)
    write(Path(folder)/'summary.json', result)
    return result


class Progress:
    """Already computed native CPU scalars; no new forward or GPU sync."""
    def __init__(self, tracker, path):
        self.tracker, self.path = tracker, Path(path)
        self.stream = self.path.open('x')
        self.lines = 0

    def __call__(self, row):
        self.stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        self.stream.flush(); self.lines += 1
        def values():
            result = {'batch': row['batch'], 'candidate': row.get('evaluations', row['fit_global_candidate']),
                'fit/global_candidate': row['fit_global_candidate'], 'optimizer/calls': row['fit_updates']}
            result.update({target: row[source] for source, target in
                (('loss', 'fit/loss'), ('nll_loss', 'fit/nll'), ('kl_loss', 'fit/kl'),
                 ('weight_decay', 'fit/norm')) if source in row})
            return result
        safe_log(self.tracker, values, 'native_fit_existing_scalar')

    def close(self):
        self.stream.close()
        return dict(member=member(self.path), line_count=self.lines, authoritative_stream=True)


def drive(c, lock, attempt, method, out, tracker, view, engine, bench, records, sealed_progress=None):
    if sealed_progress is None:
        sealed_progress = {'commits': 0}
    cursor, commits = [], []
    prior = state(view, engine.history())
    generation = NativeGeneration(c, view, engine, bench, records, out/'generation', cursor)
    storage_guard(out, c['storage']['W0_required_bytes'])
    w0 = install_W0(c, view, engine.history(), out, bench, records)
    w0_rows = rows(out/'W0', prior)
    gen_w0 = generation.W0(method, lock['source_commit'], lock['config_sha256'])
    write(out/'generation-W0.json', endpoint_ref(gen_w0))
    safe_log(tracker, lambda: w0_values(w0['summary'], normalize_summary(gen_w0['summary'])),
             'W0_first2000')
    prior_rng = rng_identity()
    for number, current, seen in batches(records):
        plan = sequence(number)
        folder = out/f'batch-{number:02d}'
        folder.mkdir(exist_ok=False)
        require(state(view, engine.history()) == prior and rng_identity() == prior_rng,
                'NATIVE_OWN_W_H_RNG_NEXT_ENTRY_JOIN')
        require(digest([row['case_id'] for row in current]) == c['packs'][number-1]['occurrence_order'],
                'NATIVE_SEALED_BATCH_ORDER')
        entry = dict(task_id=TASK, method=method, model='llama3', batch=number,
            requests=100, ids=[r['case_id'] for r in current], seen_ids=[r['case_id'] for r in seen],
            state=prior, RNG=prior_rng, context=digest(engine.contexts()), ledger=digest(cursor),
            source=lock['source_commit'], config=lock['config_sha256'], job=os.environ['SLURM_JOB_ID'])
        write(folder/'entry.json', entry)
        if number == 2:
            write(out/'initial.json', dict(status='B1_COMMIT_B2_OWN_ENTRY',
                task_id=TASK, method=method, B1_commit=member(out/'batch-01/commit.json'),
                B2_entry=member(folder/'entry.json'), source=lock['source_commit'],
                config=lock['config_sha256'], job=os.environ['SLURM_JOB_ID']))
        tx = NativeTransaction(view, engine, bench, cursor)
        next_before = engine.next_batch
        prune_before = engine.prune_applied
        started = time.monotonic()
        progress = None
        sealed_commit = None
        try:
            with tx:
                write(folder/'storage-prefit.json', storage_guard(out, c['storage']['next_batch_required_bytes']))
                pre = (rpn_subset(w0_rows, current, seen, prior, 'B1_PRE', folder/'pre')
                    if number == 1 else observe(view, bench, seen, current, engine.history(),
                        f'B{number}_PRE', folder/'pre', microbatch=c['observer_microbatch']))
                gen_pre = (generation.subset(gen_w0, current, 'B1_PRE', cohort='current100')
                    if number == 1 else generation.observe(current, f'B{number}_PRE', cohort='current100'))
                progress = Progress(tracker, folder/'events.jsonl')
                engine.progress = progress
                returned, native = engine.apply(current, number)
                require(returned is view.model, 'NATIVE_RETURNED_ACTUAL_MODEL')
                engine.progress = None
                event_ref = progress.close(); progress = None
                require(native['counts']['logging_callback_errors'] == 0 and
                        event_ref['line_count'] == native['counts']['fit_forwards'] +
                        native['counts']['native_z'], 'NATIVE_AUTHORITATIVE_EVENT_IO')
                write(folder/'native.json', native)
                terminal = terminal_before_observation(engine, number)
                if terminal is not None:
                    write(folder/'prune-terminal.json', terminal)
                after = state(view, engine.history())
                generation.set_weights(after)
                selected = seen if plan['milestone'] else current
                endpoint = f'W{number}_ALLSEEN' if plan['milestone'] else f'B{number}_POST'
                post = observe(view, bench, seen, selected, engine.history(), endpoint,
                    folder/'post', current_ids=[r['case_id'] for r in current],
                    microbatch=c['observer_microbatch'])
                gen_post = generation.observe(selected, endpoint,
                    cohort='all_seen' if plan['milestone'] else 'current100')
                gen_current = (generation.subset(gen_post, current, f'B{number}_POST', cohort='current100')
                    if plan['milestone'] else gen_post)
                gen_w0_current = generation.subset(gen_w0, current, f'W0_B{number}_CURRENT',
                    cohort='current100')
                refs = dict(pre=endpoint_ref(gen_pre), current=endpoint_ref(gen_current),
                            w0_current=endpoint_ref(gen_w0_current))
                if plan['milestone']:
                    refs['all_seen'] = endpoint_ref(gen_post)
                    refs['w0_all_seen'] = endpoint_ref(generation.subset(gen_w0, seen,
                        f'W0_W{number}_ALLSEEN', cohort='all_seen'))
                    refs['first500'] = endpoint_ref(generation.subset(gen_post, records[:500],
                        f'W{number}_FIRST500', cohort='fixedfirst500'))
                    rpn_subset(rows(folder/'post', after), records[:500], seen, after,
                        f'W{number}_FIRST500', folder/'first500')
                write(folder/'generation.json', dict(schema='native-generation-endpoint-links-v1',
                    task_id=TASK, method=method, model='llama3', batch=number, endpoints=refs,
                    actual_weight_state=generation.weights_identity(),
                    native_history_identity_separate=after['H'], raw_local_only=True))
                metrics = endpoint_values(pre['summary'], post['current'],
                    post['summary'] if plan['milestone'] else None, number,
                    normalize_summary(gen_pre['summary']), normalize_summary(gen_current['summary']),
                    normalize_summary(gen_post['summary']) if plan['milestone'] else None)
                metrics.update(w0_subset(w0_rows, entry['ids'], 'w0/current/N'))
                metrics.update(generation_values('w0/current', normalize_summary(gen_w0_current['summary']), 100))
                if plan['milestone']:
                    metrics.update(w0_subset(w0_rows, entry['seen_ids'], 'w0/all_seen/N'))
                    metrics.update(generation_values('w0/all_seen',
                        normalize_summary(refs['w0_all_seen']['summary']), number*100))
                write(folder/'metrics.json', metrics)
                cursor.extend(entry['ids'])
                after_rng = rng_identity()
                receipt = dict(schema='native-baseline-commit-v1', task_id=TASK, method=method,
                    model='llama3', batch=number, requests=100, cumulative_requests=len(cursor),
                    state_before=prior, state_after=after, RNG_before=prior_rng, RNG_after=after_rng,
                    entry_context=entry['context'], exit_context=digest(engine.contexts()),
                    ledger_before=entry['ledger'], ledger_after=digest(cursor),
                    native=member(folder/'native.json'), native_counts=native['counts'],
                    history_appends=native['counts']['history_appends'],
                    caller_history_appends=0, native_has_history=engine.binding['native_has_history'],
                    pre=pre['summary'], post_current=post['current'], post=post['summary'],
                    generation=member(folder/'generation.json'), metrics=member(folder/'metrics.json'),
                    pre_observation=member(folder/'pre/summary.json'),
                    post_observation=member(folder/'post/summary.json'), events=event_ref,
                    transaction_finished=True, seconds=time.monotonic()-started,
                    timing_policy='batch inclusive; native/observers nested, not additive',
                    source=lock['source_commit'], config=lock['config_sha256'],
                    job=os.environ['SLURM_JOB_ID'], checkpoint_saved=False,
                    actual_deployed_PRUNE_endpoint=terminal is not None)
                sealed_commit = seal_commit(tx, folder/'commit.json', receipt)
            sealed_progress['commits'] = number
            commits.append(sealed_commit)
            prior, prior_rng = after, after_rng
            safe_log(tracker, lambda: metrics, 'committed_native_batch')
        except BaseException:
            engine.progress = None
            if progress is not None:
                try: progress.close()
                except Exception: pass
            if sealed_commit is not None and tx.done:
                sealed_progress['commits'] = number
                # W/H/cursor have already passed the transaction boundary.
                # A later caller/cleanup error cannot truthfully be a rollback.
                write(folder/'post-commit-error.json', dict(task_id=TASK, method=method,
                    batch=number, committed_prefix=number, sealed_commit=sealed_commit,
                    committed_state=after, rollback_attempted=False,
                    committed_prefix_preserved=True,
                    error_type=type(__import__('sys').exc_info()[1]).__name__, no_retry=True))
                raise
            engine.next_batch, engine.prune_applied = next_before, prune_before
            generation.set_weights(prior)
            write(folder/'rollback.json', dict(task_id=TASK, method=method, batch=number,
                committed_prefix=number-1, rollback_verified=tx.rollback_verified,
                state=state(view, engine.history()), expected_entry_state=prior,
                earlier_prefix_preserved=True, error_type=type(__import__('sys').exc_info()[1]).__name__,
                native_attempt_cost_counts_preserved=engine.counts, no_retry=True))
            raise
    require(len(commits) == 20 and len(cursor) == 2000 and engine.next_batch == 21,
            'NATIVE_EXACT_TWENTY_COMMITS_STOP')
    return dict(status='COMPLETE_W20', complete=True, task_id=TASK, method=method,
        model='llama3', commits=20, requests=2000, own_next_entry_joins=19,
        state=prior, RNG=prior_rng, ledger=digest(cursor), counts=engine.counts,
        history_appends=engine.counts['history_appends'], commit_receipts=commits,
        source=lock['source_commit'], config=lock['config_sha256'],
        job=os.environ['SLURM_JOB_ID'], checkpoint_saved=False, exact_resume='NOT_AVAILABLE')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    args = parser.parse_args()
    attempt = args.attempt.resolve()
    out = attempt/args.method
    out.mkdir(exist_ok=False)
    tracker, engine, started = None, None, time.monotonic()
    terminal = None
    lock = None
    sealed_progress = {'commits': 0}
    error = None
    try:
        c, lock = locked(attempt, args.method)
        storage_guard(out, c['storage']['startup_required_bytes'])
        require(transport_support()['status'] == 'KEYS_PRESENT_REQUIRES_WORKER_AND_AXIS_CHECK',
                'GENERATION_TRANSPORT_KEYS_NOT_BOUND')
        from project.run_scripts.experiment_tracking import init
        config = scientific_config(args.method, c['run_instance']['attempt'],
            lock['source_commit'], lock['config_sha256'], c['generation'])
        config['job_id'] = os.environ['SLURM_JOB_ID']
        tracker = init(env_file=c['tracking']['env_file'], spool=out/'tracking', config=config)
        view, engine, bench, records = setup(c, out, args.method, lock)
        terminal = drive(c, lock, attempt, args.method, out, tracker, view, engine, bench, records,
                         sealed_progress=sealed_progress)
    except BaseException as exc:
        error = exc
        # Raw full trace is local only. Original error is never replaced by a
        # failed logger finish or by a second attempt to fabricate completeness.
        try:
            write(out/'failed.json', dict(status='TECHNICAL_BLOCKED', technical_block=True,
                task_id=TASK, method=args.method, model='llama3', error_type=type(exc).__name__,
                error=str(exc), traceback=traceback.format_exc(),
                committed_prefix=verified_prefix(sealed_progress),
                prefix_basis='RAM_VERIFIED_TRANSACTION_EXIT_NOT_RECEIPT_GLOB',
                immutable_commit_receipts_present=len(list(out.glob('batch-*/commit.json'))),
                job=os.environ.get('SLURM_JOB_ID'), source=os.environ.get(SOURCE_ENV),
                config=None if lock is None else lock['config_sha256'],
                checkpoint_saved=False, exact_resume='NOT_AVAILABLE', automatic_retry=False))
        except Exception:
            pass
    finally:
        finish = None
        if tracker is not None:
            try: finish = tracker.finish(exit_code=int(error is not None))
            except Exception as logger_error:
                finish = dict(local_status='LOGGING_DEGRADED_FINISH',
                              error_type=type(logger_error).__name__)
        cost = dict(wall_seconds=time.monotonic()-started,
            host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            actual_GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            actual_GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            GPU_parent_allocation_accounting='CPU_COLLECTOR_EXACT_PARENT_ONLY',
            nested_stage_times_not_additive=True,
            native_attempt_counts=None if engine is None else engine.counts)
        try: write(out/'cost.json', cost)
        except Exception: pass
        if terminal is not None:
            terminal.update(tracking=finish, cost=member(out/'cost.json'),
                            SDK_finish_does_not_certify_scientific_completion=True)
            write(out/'terminal.json', terminal)
    if error is not None:
        raise error
    print(json.dumps(dict(status=terminal['status'], method=args.method,
                          commits=terminal['commits'], requests=terminal['requests']), sort_keys=True))


if __name__ == '__main__':
    main()
