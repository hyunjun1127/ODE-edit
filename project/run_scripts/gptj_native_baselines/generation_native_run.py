"""New cold six-arm entry: native math/RPN unchanged, native F/C at W20 only.

This task-private entry has no legacy MB qualification or W0 generation READY.
The production native factories and RAM rollback implementation are read-only
reuse. Native twenty commits persist before the one native observation call.
"""
import argparse
import copy
import os
import random
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import transformers

from . import generation_run as native
from .generation_common import (
    ARMS, ARM_LAYERS, SOURCE_ENV, batches, digest, expected_counts, read,
    require, sha, stat_seal, verify, write, writer_identity,
)
from .generation_native_common import TASK, NONCE, SCHEDULE, ready
from .generation_native_bridge import GenerationObserver
from project.run_scripts.gptj_cake_blue_prune_rect.metrics import ObservationView, state
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from scripts.fixed_counterfact import load_prefix


def arm_configuration(config, arm):
    """Identical native-family projection, with the explicit new task identity."""
    require(arm in ARMS and set(config['arm_configs']) == set(ARMS), 'SIX_ARM_CONFIG_PROJECTION')
    result = copy.deepcopy(config['arm_configs'][arm])
    for field in native.IDENTITY_FIELDS:
        require(field in config, 'TASK_IDENTITY_FIELD:'+field)
        result[field] = copy.deepcopy(config[field])
    require(result['task_id'] == TASK and result['instruction_id'] == NONCE,
        'NATIVE_GENERATION_TASK_IDENTITY')
    require(result['noCP'] and not result['z_disk_cache']
        and result['exact_resume'] == 'NOT_AVAILABLE', 'GENERATION_NATIVE_NOCP')
    for field in ('model', 'model_revision', 'seed', 'runtime', 'stream'):
        require(field in result and result[field] == config[field], 'GENERATION_FAMILY_DRIFT:'+field)
    return result


def prepare_engine(config, model, tokenizer, arm, out):
    projected = arm_configuration(config, arm)
    prepare, _ = native.family_api(arm)
    if arm in native.STOCK_ARMS:
        return prepare(projected, model, tokenizer, arm)
    return prepare(projected, model, tokenizer, arm, attempt=out)


def locked(attempt):
    config, lock = read(attempt/'config.json'), read(attempt/'execution.lock.json')
    ready(config)
    require(config['instruction_id'] == lock['instruction_id'] == NONCE
        and config['task_id'] == lock['task_id'] == TASK, 'NATIVE_GENERATION_TASK_AUTHORITY')
    require(os.environ.get(SOURCE_ENV) == lock['source_commit']
        and sha(attempt/'config.json') == lock['config_sha256'], 'GENERATION_SOURCE_CONFIG_IDENTITY')
    for row in (lock['source_members']+lock['runtime_sources']+lock['launchers']
            +[lock['archive'], lock['tracking_env'], config['authority']]):
        verify(row)
    for row in config['assets']:
        stat_seal(row)
    for arm in ARMS:
        arm_configuration(config, arm)
    return config, lock


def generation_receipt(result, records, physical_state, out):
    receipt = native.generation_receipt(result, records, 'W20', physical_state, out)
    for key in ('native_execution_member', 'generation_native_runtime_member',
            'native_commit_members', 'committed_batch20_member', 'generation_profile',
            'generation_route', 'sampling_scope', 'qualification_performed', 'no_fallback'):
        require(key in result, 'NATIVE_FINAL_GENERATION_RECEIPT_FIELD:'+key)
        receipt[key] = copy.deepcopy(result[key])
    return receipt


def production_ops():
    from .generation_native_tracking import log_generation
    ops = native.production_ops()
    ops.log_generation = log_generation
    return ops


def execute_chain(config, lock, out, arm, model, tokenizer, view, engine, bench, records,
                  generation, tracker, *, ops=None, on_stage=None, on_commit=None):
    """The final-only native20 loop, with task-local identity/receipt adaptation."""
    require(config['generation'].get('evaluation_schedule') == SCHEDULE,
        'NATIVE_GENERATION_EXPLICIT_SCHEDULE')
    ops = production_ops() if ops is None else ops
    projected = arm_configuration(config, arm)
    _, normalize = native.family_api(arm)
    out = Path(out)
    commits, cursor = [], []
    def stage(value):
        if on_stage is not None:
            on_stage(value)
    stage('W0_RPN_REUSE')
    w0_state = ops.state(view, engine.history())
    w0 = ops.install_W0(projected, view, engine.history(), out, bench, records)
    w0raw = ops.rows(out/'W0', w0_state)
    ops.log_w0(tracker, w0['summary'])
    previous = ops.state(view, engine.history())
    engine.progress = native.make_progress(engine, tracker, arm, ops.safe_log)
    for number, current, seen in ops.batches(records):
        require(1 <= number <= 20 and len(current) == 100 and len(seen) == number*100,
            'GENERATION_NATIVE_EXACT_20_PACKS')
        folder = out/f'batch-{number:02d}'
        ids = [record['case_id'] for record in current]
        started = time.monotonic()
        require(ops.state(view, engine.history()) == previous, 'GENERATION_BATCH_ENTRY_LINK')
        ops.reserve(out, config)
        normalized = normalize(current)
        pack_identity = digest(normalized)
        stage(f'B{number}_PRE_RPN')
        pre = ops.observe(view, bench, seen, current, engine.history(), f'B{number}_PRE',
            folder/'pre', current_ids=ids)
        stage(f'B{number}_NATIVE_APPLY')
        with ops.transaction(view, engine, bench) as tx:
            returned, receipt_native = engine.apply(normalized, number)
            require(returned is model, 'SAME_ACCUMULATED_NATIVE_MODEL')
            require(all(bool(torch.isfinite(weight).all()) for weight in view.weights.values())
                and all(bool(torch.isfinite(value).all()) for value in engine.history().values()),
                'NATIVE_NONFINITE_COMMIT')
            expected = expected_counts(arm)
            counts = {field:receipt_native['delta'][field] for field in expected}
            require(counts == expected, 'NATIVE_CALL_HISTORY_COUNTS')
            require(set(engine.history()) == (set(view.sites) if arm in native.HISTORY_ARMS else set()),
                'GENERATION_NATIVE_HISTORY_BOUNDARY')
            if number == 20 and arm == 'PRUNE':
                stage('TERMINAL_PRUNE_BASE_FIX')
                terminal = engine.terminal_prune()
                require(terminal.get('explicit_repair') == native.BASE_FIX
                    and terminal.get('prune_applied') is True, 'PRUNE_EXPLICIT_TERMINAL_BASE_FIX')
                receipt_native.update(terminal_prune=terminal, prune_applied=True, explicit_repair=native.BASE_FIX)
            selected = seen if number in native.MILESTONES else current
            stage(f'B{number}_POST_RPN')
            post = ops.observe(view, bench, seen, selected, engine.history(), f'W{number}',
                folder/'post', current_ids=ids)
            after = ops.state(view, engine.history())
            contexts = native.native_contexts(engine, arm)
            require(isinstance(contexts, list) and len(contexts) == 2 and contexts[0] == ['{}']
                and len(contexts[1]) == 5, 'GENERATION_NATIVE_CONTEXT_CARRIED_1_PLUS5')
            proposed_cursor = cursor+ids
            receipt = dict(task=config['task_id'], arm=arm, writer=writer_identity(arm), batch=number,
                case_ids=ids, source=lock['source_commit'], config=digest(config), before=previous, after=after,
                native=receipt_native, native_counts=counts, pre=pre['summary'], post=post['summary'],
                post_current=post['current'], post_scope='ALL_SEEN' if number in native.MILESTONES else 'CURRENT',
                seen_requests=len(seen), native_pack=pack_identity, ledger=digest(proposed_cursor),
                generation_schedule=SCHEDULE, generation_available=False,
                generation_unavailable_reason='FINAL_W20_ONLY_SCHEDULE',
                generation_observer_status='NOT_SCHEDULED_INTERMEDIATE', observer_no_mutation=True,
                seconds=time.monotonic()-started, checkpoint_saved=False, exact_resume='NOT_AVAILABLE')
            ops.write(folder/'native-context-identity.json', dict(context_sha256=digest(contexts),
                groups=[len(group) for group in contexts], generated_inside_native=True))
            tx.finish()
            try:
                ops.write(folder/'commit.json', receipt)
            except BaseException:
                tx.done = False
                raise
        commits.append(receipt)
        cursor, previous = proposed_cursor, after
        if on_commit is not None:
            on_commit(receipt)
        ops.log_batch(tracker, receipt, w0raw, ids, [record['case_id'] for record in seen])
        print(native.json.dumps(dict(event='native_final_generation_batch_committed', arm=arm,
            batch=number, edits=len(cursor), native_counts=counts, generation_schedule=SCHEDULE)), flush=True)
        ops.cleanup()
    require(len(commits) == 20 and len(cursor) == 2000, 'GENERATION_FULL_20_BATCH_COMPLETION')
    stage('FINAL_W20_GENERATION')
    raw_out = out/'generation-final'
    generated = native.guarded_generation(lambda:generation.endpoint(records, out=raw_out,
        endpoint='W20', model_state=previous, cohort_label='ALL_SEEN'), view, engine, arm, bench, ops, 2000)
    final = generation_receipt(generated, records, previous, raw_out)
    final.update(task=config['task_id'], arm=arm, source=lock['source_commit'], config=digest(config),
        generation_schedule=SCHEDULE, edits=2000, actual_model_edits=2000,
        generation_observer_no_mutation=True, checkpoint_saved=False, exact_resume='NOT_AVAILABLE')
    ops.write(out/'generation-final.json', final)
    ops.log_generation(tracker, 'all_seen/post', generated['summary'], 2000, 1900, 2000)
    stage('FINAL_W20_GENERATION_COMPLETE')
    return dict(status='COMPLETED', completed_batches=20, commits=20, edits=2000, state=previous,
        native_counts=engine.counts, source=lock['source_commit'], config=digest(config),
        generation_schedule=SCHEDULE, generation_endpoints=1, generation_W0_endpoints=0,
        generation_intermediate_endpoints=0, generation_final_requests=2000,
        final_generation_identity=generated['identity'],
        final_generation_identity_sha256=generated.get('identity_sha256'),
        native_execution_member=generated['native_execution_member'],
        committed_batch20_member=generated['committed_batch20_member'],
        generation_profile=generated['generation_profile'], generation_route=generated['generation_route'],
        native_scope_completed=True, final_generation_completed=True,
        checkpoint_saved=False, exact_resume='NOT_AVAILABLE')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    attempt, arm = args.attempt.resolve(), args.arm
    out = attempt/arm
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
        projected = arm_configuration(config, arm)
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
        from .generation_native_tracking import start_tracking
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
        require(state(view, {})['W'] == {str(layer):projected['cold_W'][str(layer)] for layer in view.sites},
            'ACTUAL_COLD_W0')
        require(model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr(), 'GPTJ_UNTIED_HEAD')
        stage('NATIVE_PREPARATION')
        engine = prepare_engine(config, model, tokenizer, arm, out)
        native.initial_history(engine, arm, view)
        if arm not in native.STOCK_ARMS:
            stage('NATIVE_CONTEXT_FIRST_COLD_PREPARATION')
            engine.prepare_contexts()
            contexts = native.native_contexts(engine, arm)
            write(out/'native-contexts.json', dict(contexts=contexts, context_sha256=digest(contexts),
                seed=config['seed'], native_generator=True, source=lock['source_commit'],
                training_context_not_evaluator_input=True))
        bench = CounterFactAdapter(tokenizer, [])
        write(out/'runtime.json', dict(device=torch.cuda.get_device_name(), torch=str(torch.__version__),
            transformers=transformers.__version__, model=config['model'], FP32=True, eager=True, TF32=False,
            autocast=False, CPU_threads=config['resources']['cpu'], source=lock['source_commit'],
            config=digest(config), arm=arm, job=os.environ['SLURM_JOB_ID'], checkpoint_saved=False,
            initial_history_zero=True, initial_state=state(view, engine.history()),
            native_solve_dtype='FP64' if arm in ('BASE_MEMIT', 'PRUNE', 'RECT') else 'native FP32',
            load_seconds=time.monotonic()-loading, generation_profile=config['generation']['profile']))
        stage('NATIVE_GENERATION_OBSERVER_BIND')
        generation = GenerationObserver(config, lock, view, engine, tokenizer, records, out, arm, tracker=tracker)
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
        terminal = execute_chain(config, lock, out, arm, model, tokenizer, view, engine, bench,
            records, generation, tracker, ops=ops, on_stage=stage, on_commit=committed)
    except BaseException as error:
        terminal = dict(status='FAILED', stage=progress_state['stage'], error_type=type(error).__name__,
            error=str(error)[:1800], completed_batches=progress_state['commits'], commits=progress_state['commits'],
            source=locals().get('lock', {}).get('source_commit', os.environ.get(SOURCE_ENV)),
            config=digest(config) if 'config' in locals() else None,
            native_counts=engine.counts if engine is not None else {field:0 for field in expected_counts(arm)},
            rollback_verified=progress_state['rollback_verified'], checkpoint_saved=False,
            exact_resume='NOT_AVAILABLE', generation_schedule=SCHEDULE, generation_W0_endpoints=0,
            generation_intermediate_endpoints=0, native_scope_completed=progress_state['commits'] == 20,
            native_commits_preserved=progress_state['commits'] == 20, final_generation_completed=False,
            final_generation_status='FAILED' if progress_state['stage'] == 'FINAL_W20_GENERATION' else 'NOT_COMPLETED')
        write(out/'failure.json', terminal)
        traceback.print_exc(limit=8)
    finally:
        if tracker is not None:
            try:
                write(out/'tracking-finish.json', tracker.finish(
                    exit_code=0 if terminal.get('status') == 'COMPLETED' else 1, timeout=45))
            except Exception as error:
                write(out/'tracking-finish-error.json', dict(type=type(error).__name__, science_not_restarted=True))
        terminal.update(program_seconds=time.monotonic()-started,
            peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        write(out/'terminal.json', terminal)
    if terminal['status'] != 'COMPLETED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
