"""Two independent cold processes, twenty fresh fits each, no checkpoint/retry."""
import argparse, gc, json, os, random, resource, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.observe import observe, reduce_rows, active_flags
from project.run_scripts.jlz_realization.writer import Transaction, rng_snapshot, rng_restore, rng_equal
from project.run_scripts.jlz_shared_budget.entry import prepare_entry
from project.run_scripts.jlz_shared_budget.optimize import fit
from project.run_scripts.jlz_shared_budget.telemetry import Events
from project.run_scripts.jlz_realized_writer.capture import Adapter, capture_native_sites, virtual_terminal, cache_identity
from project.run_scripts.jlz_realized_writer.writer import apply_sequential
from project.run_scripts.jlz_two_arm.baseline_pilot import _rng_hash
from .common import *

class BatchTransaction(Transaction):
    """Extend the unchanged RAM transaction to batch-local adapter/context caches."""
    def __init__(self, adapter, history, bench):
        super().__init__(adapter, history)
        self.bench = bench

    def __enter__(self):
        super().__enter__()
        self.virtual_before = self.a.last_virtual
        self.capture_before = self.a.capture_virtual
        self.context_before = json.loads(json.dumps(self.bench.contexts))
        self.hooks_before = self.a.hook_signature()
        self.a.last_virtual = {}  # row indices restart at zero in every pack
        self.a.capture_virtual = False
        return self

    def finish(self):
        require(self.a.hook_signature() == self.hooks_before, 'HOOK_LEAK')
        require(self.bench.contexts == self.context_before, 'CONTEXT_MUTATION')
        super().finish()

    def __exit__(self, kind, value, tb):
        super().__exit__(kind, value, tb)
        self.a.last_virtual = self.virtual_before if not self.done else {}
        self.a.capture_virtual = self.capture_before
        if not self.done:
            self.bench.contexts = self.context_before
        require(self.a.hook_signature() == self.hooks_before, 'TRANSACTION_HOOK_RESTORE')
        require(self.bench.contexts == self.context_before, 'TRANSACTION_CONTEXT_RESTORE')

def locked(attempt):
    c = json.loads((attempt / 'config.json').read_text())
    lock = json.loads((attempt / 'execution.lock.json').read_text())
    require(c['instruction_id'] == lock['instruction_id'] == NONCE and c['task_id'] == TASK, 'AUTHORITY')
    require(os.environ.get('ODEEDIT_SOURCE_COMMIT') == lock['source_commit'], 'FROZEN_SOURCE')
    require(sha(attempt / 'config.json') == lock['config_sha256'], 'FROZEN_CONFIG')
    for r in lock['source_members'] + lock['runtime_sources'] + lock['dependency_sources'] + lock['native_reference'] + lock['launchers'] + [lock['archive'], lock['native_hparams']]:
        verify(r)
    for r in c['authority_members'] + c['qualification_reuse']['receipts'] + [c['observer_identity'], c['native_input_alignment'], c['cpu_preflight']]:
        verify(r)
    for r in c['assets']:
        s = Path(r['path']).stat()
        require((s.st_size, s.st_ino, s.st_mtime_ns) == (r['bytes'], r['inode'], r['mtime_ns']), 'ASSET_CHANGED')
    require(c['settings']['B'] == 100 and c['settings']['batches'] == 20 and c['settings']['requests'] == 2000, 'HORIZON_20')
    require(tuple(c['settings']['arms']) == ARMS and not c['settings']['save_checkpoints'], 'MD_CD_NO_CP')
    require(len(c['packs']) == 20 and c['qualification_reuse']['actual_B1_reuse_verified'], 'INPUT_QUALIFICATION_CLOSURE')
    return c, lock

def setup(c, out, arm):
    require(shutil.disk_usage(out).free >= c['resources']['startup_free_bytes_min'], 'RESOURCE_BLOCKED_STORAGE')
    require(torch.__version__ == c['runtime']['torch'] and transformers.__version__ == c['runtime']['transformers'], 'RUNTIME_VERSION')
    torch.set_num_threads(8)
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(c['model'], local_files_only=True, dtype=torch.float32,
        attn_implementation='eager', low_cpu_mem_usage=True).to('cuda').eval()
    tok = AutoTokenizer.from_pretrained(c['model'], local_files_only=True)
    tok.pad_token = tok.eos_token; tok.padding_side = 'right'
    a = Adapter(model, c['profile'])
    bench = CounterFactAdapter(tok, json.loads(Path(c['contexts']).read_text()))
    H = {l: torch.zeros(dim[1], dim[1], dtype=torch.float32) for l, dim in a.dims.items()}
    records = load_prefix(Path(c['stream']).parent, c['settings']['requests'])
    require(digest([r['case_id'] for r in records]) == c['ordered_ids_sha256'], 'ORDERED_2000')
    cold = state(a, H)
    require(cold == c['qualification_reuse']['cold_W0_H0'], 'COLD_MODEL_HISTORY_IDENTITY')
    write(out / 'runtime.json', dict(source=os.environ['ODEEDIT_SOURCE_COMMIT'], config=digest(c), arm=arm,
        job=os.environ.get('SLURM_JOB_ID'), device=torch.cuda.get_device_name(), cold_W0_H0=cold,
        torch=torch.__version__, transformers=transformers.__version__, FP32=True, geometry_FP64=True,
        eager=True, TF32=False, autocast=False, requests=len(records), B=100, batches=20,
        actual_B1_qualification_reused=True, extra_fit=0, no_B21=True, checkpoint_saved=False))
    return a, bench, records, H

def observer(a, bench, seen, selected, H, endpoint, out, c, identities, current_ids):
    rng = rng_snapshot(); before = state(a, H)
    result = observe(a, bench, seen, selected, H, endpoint, out, c['settings']['observer_microbatch'], current_ids)
    require(rng_equal(rng) and state(a, H) == before, 'OBSERVER_STATE_RNG')
    rows = rows_from(out)
    require(validate_rows(rows, identities, [r['case_id'] for r in selected], endpoint) == result['summary'], 'OBSERVER_REDUCER')
    flags = active_flags(seen)
    require(all(r['active_at_endpoint'] == flags[r['case_id']] for r in rows), 'SEEN_PREFIX_ACTIVE')
    return result

def w0_subset(folder, current, seen, out, cold, identities):
    ids = [r['case_id'] for r in current]; flags = active_flags(seen)
    rows = [dict(r, endpoint='B1_PRE', active_at_endpoint=flags[r['case_id']])
            for r in rows_from(folder) if r['case_id'] in set(ids)]
    summary = validate_rows(rows, identities, ids, 'B1_PRE')
    write(out / 'chunk-0000.json', dict(state=cold, rows=rows, optimizer_feedback=False))
    result = dict(endpoint='B1_PRE', state=cold, requests=len(current), summary=summary, current=summary,
        seconds=0, reused_from=str(folder), new_forwards=0, no_mutation=True, optimizer_feedback=False)
    write(out / 'summary.json', result)
    return result

def drive(a, bench, records, H, c, arm, out, source):
    identities = json.loads(verify(c['observer_identity']).read_text())['rows']
    previous = state(a, H); previous_rng = rng_snapshot(); commits = []
    observer(a, bench, records, records, H, 'W0', out / 'W0', c, identities, [r['case_id'] for r in records])
    for number, current, seen in batches(records, c['settings']['B']):
        require(number <= c['settings']['batches'], 'NO_B21')
        root = out / f'batch-{number:02d}'; root.mkdir(exist_ok=False)
        require(state(a, H) == previous and rng_equal(previous_rng), 'OWN_NEXT_ENTRY_STATE_JOIN')
        pack = bench.prepare(current); expected = c['packs'][number - 1]
        require(pack['identity'] == expected['identity'] and pack['record_ids'] == expected['ids'], 'SEALED_BATCH_PACK')
        if number == 2:
            write(out / 'initial.json', dict(status='MAIN_B1_COMMIT_B2_OWN_ENTRY', arm=arm,
                B1_commit=member(out / 'batch-01/commit.json'), next_entry=previous,
                source=source, config=digest(c), observer_restored=True, own_W_H_RNG_join=True))
        rng_id = _rng_hash(previous_rng)
        write(root / 'entry.json', dict(source=source, config=digest(c), arm=arm, batch=number,
            ids=pack['record_ids'], native_pack=pack['identity'], state=previous, RNG=rng_id,
            context_hash=digest(bench.contexts), seen_ids=[r['case_id'] for r in seen], fit_count=1))
        tx = BatchTransaction(a, H, bench); started = time.monotonic(); calls_before = dict(a.calls)
        events = Events(root / 'fit-events.jsonl', source + ':' + arm, number)
        try:
            with tx:
                pre = w0_subset(out / 'W0', current, seen, root / 'pre', previous, identities) if number == 1 else observer(
                    a, bench, seen, current, H, f'B{number}_PRE', root / 'pre', c, identities, pack['record_ids'])
                entry = prepare_entry(a, bench, pack, H, c['stats'], c['settings']['fit_requests_per_group'])
                write(root / 'entry-capture.json', dict(native_pack=pack['identity'], ids=pack['record_ids'],
                    teacher_hash=entry['teacher_hash'], anchors={l: tensor_sha(v) for l, v in entry['anchors'].items()},
                    entry_hidden={l: tensor_sha(v) for l, v in entry['entry_hidden'].items()},
                    entry_weights={l: tensor_sha(v) for l, v in entry['entry_weights'].items()},
                    H_entry=previous['H'], capture_seconds=entry['seconds'], fresh_capture=True, cache_from_previous_batch=False))
                initial = capture_native_sites(a, entry, a.sites)
                cache = cache_identity(entry); guard = a.guard(); hooks = a.hook_signature()
                a.capture_virtual = True
                try:
                    plan, fit_receipt = fit(a, entry, events, root / 'fit')
                finally:
                    a.capture_virtual = False
                virtual = virtual_terminal(a, entry, plan)
                require(state(a, H) == previous and rng_equal(previous_rng), 'FIT_MUTATED_ENTRY_W_H_RNG')
                plan_id = digest({name: {l: tensor_sha(v) for l, v in plan[name].items()} for name in ('u', 'D', 'z')})
                write(root / 'plan.json', dict(identity=plan_id, terminal_z_same_forward=True, fit_count=1,
                    virtual_all_rows=digest({l: tensor_sha(v) for l, v in virtual.items()}), native_pack=pack['identity'],
                    ids=pack['record_ids'], entry_cache=cache, entry_weights=previous['W'], persisted_plan_tensors=False))
                # apply_sequential performs ALL native final-key H appends itself.
                writer = apply_sequential(a, entry, plan, virtual, initial, H, arm, root / 'writer')
                require(writer['history_appends'] == len(a.sites), 'HISTORY_ONCE')
                require(cache_identity(entry) == cache and a.guard() == guard and a.hook_signature() == hooks, 'WRITER_CACHE_GUARD')
                require(plan_id == digest({name: {l: tensor_sha(v) for l, v in plan[name].items()} for name in ('u', 'D', 'z')}), 'PLAN_MUTATION')
                del plan, virtual, initial, entry
                a.last_virtual = {}; gc.collect(); torch.cuda.empty_cache()
                selected = selected_for_post(current, seen, number)
                post = observer(a, bench, seen, selected, H, f'W{number}', root / 'post', c, identities, pack['record_ids'])
                after = state(a, H)
                require(rng_equal(previous_rng), 'BATCH_RNG_DRIFT')
                receipt = dict(source=source, config=digest(c), arm=arm, batch=number, ids=pack['record_ids'],
                    native_pack=pack['identity'], before=previous, after=after, RNG_before=rng_id,
                    RNG_after=_rng_hash(rng_snapshot()), context_hash=digest(bench.contexts),
                    fit_count=1, fit=fit_receipt, plan_identity=plan_id, history_appends=writer['history_appends'],
                    history=writer['history'], layers=list(a.sites), writer=member(root / 'writer/writer.json'),
                    observer_no_mutation=True, pre=pre['summary'], post=post['summary'], post_current=post['current'],
                    post_scope='ALL_SEEN' if number in MILESTONES else 'CURRENT', seen_requests=len(seen),
                    cache_invalidated_for_next_batch=True, calls={k: a.calls[k] - calls_before[k] for k in a.calls},
                    seconds=time.monotonic() - started, checkpoint_saved=False, exact_resume='NOT_AVAILABLE')
                write(root / 'prepared-commit.json', receipt)
                # JSON object keys are strings (fit uses integer case IDs in RAM).
                require(json.loads((root / 'prepared-commit.json').read_text()) == json.loads(json.dumps(receipt)), 'COMMIT_IO_VERIFY')
                tx.finish()
                # Keep RAM snapshots through atomic publication. An IO error
                # still aborts/rolls back THIS entry, preserving the prefix.
                try:
                    write(root / 'commit.json', receipt)
                except BaseException:
                    tx.done = False
                    raise
            commits.append(receipt); previous = after; previous_rng = rng_snapshot()
            print(json.dumps(dict(event='V13_SEQUENTIAL_COMMIT', arm=arm, batch=number, requests=len(seen))), flush=True)
        except BaseException as error:
            write(root / 'rollback.json', dict(verified=tx.rollback_verified, batch_entry=tx.before,
                earlier_committed_prefix=len(commits), logical_commit=False, rollback_after_SIGTERM='NOT_VERIFIED',
                cache_context_restored=not tx.done, error_type=type(error).__name__))
            raise
    require(len(commits) == c['settings']['batches'] and sum(r['history_appends'] for r in commits) == len(a.sites) * len(commits), 'CHAIN_COVERAGE')
    return commits

def main():
    p = argparse.ArgumentParser(); p.add_argument('--attempt', type=Path, required=True); p.add_argument('--arm', choices=ARMS, required=True)
    args = p.parse_args(); attempt = args.attempt.resolve(); out = attempt / ('main-' + args.arm)
    out.mkdir(exist_ok=False); start = time.monotonic(); status = 'TECHNICAL_BLOCKED'; a = None
    try:
        c, lock = locked(attempt); a, bench, records, H = setup(c, out, args.arm)
        drive(a, bench, records, H, c, args.arm, out, lock['source_commit'])
        imported = []
        for name, module in list(sys.modules.items()):
            path = getattr(module, '__file__', None)
            if isinstance(path, str) and Path(path).is_file() and name.startswith(('project.run_scripts.', 'memit.', 'rome.', 'util.', 'transformers.models.llama')):
                imported.append(dict(module=name, **member(path)))
        write(out / 'actual-imports.json', dict(files=imported))
        write(out / 'ready.json', dict(status='CHAIN_COMPLETE', arm=args.arm, source=lock['source_commit'],
            config=digest(c), commits=20, requests=2000, history_appends=100, no_B21=True))
        status = 'W20_COMPLETE'
    except BaseException as error:
        write(out / 'first-error.json', dict(type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), original_KEEP=True))
        raise
    finally:
        write(out / 'terminal.json', dict(status=status, arm=args.arm, source=os.environ.get('ODEEDIT_SOURCE_COMMIT'),
            commits=len(list(out.glob('batch-*/commit.json'))), seconds=time.monotonic() - start,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_VRAM_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            job=os.environ.get('SLURM_JOB_ID'), adapter_calls=None if a is None else a.calls,
            no_B21=True, checkpoint_saved=False, exact_resume='NOT_AVAILABLE'))

if __name__ == '__main__':
    main()
