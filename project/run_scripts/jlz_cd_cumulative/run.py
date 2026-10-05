"""Sealed cold 20-batch chains; no runtime retry or checkpoint path."""
import argparse, gc, json, os, random, resource, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.writer import rng_snapshot, rng_equal
from project.run_scripts.jlz_shared_budget.entry import prepare_entry
from project.run_scripts.jlz_realized_writer.capture import capture_native_sites, virtual_terminal, cache_identity
from project.run_scripts.jlz_realized_writer_sequential.run import BatchTransaction, observer, w0_subset
from project.run_scripts.jlz_writer_coupled.geometry import build_prior_from_npz
from project.run_scripts.jlz_two_arm.baseline_pilot import _rng_hash
from .common import *
from .adapter import Adapter
from .geometry import build_geometry, load_native_c0
from .calibration import CalibrationLock
from .optimize import fit
from .writer import apply

class Events:
    """Scalar/hash JSONL only, independent event namespace per arm/batch."""
    def __init__(self, path, arm, batch):
        self.path, self.arm, self.batch = path, arm, batch
    def emit(self, event, payload, request=None, candidate=None, layer=None):
        row = dict(event=event, arm=self.arm, batch=self.batch, request=request,
                   candidate=candidate, layer=layer, payload=payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a') as f:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')

def locked(attempt):
    c = json.loads((attempt / 'config.json').read_text())
    lock = json.loads((attempt / 'execution.lock.json').read_text())
    require(c['instruction_id'] == lock['instruction_id'] == NONCE and c['task_id'] == TASK, 'AUTHORITY')
    require(os.environ.get('ODEEDIT_SOURCE_COMMIT') == lock['source_commit'], 'FROZEN_SOURCE')
    require(sha(attempt / 'config.json') == lock['config_sha256'], 'FROZEN_CONFIG')
    for row in lock['source_members'] + lock['runtime_sources'] + lock['dependency_sources'] + lock['native_reference'] + lock['launchers'] + [lock['archive'], lock['native_hparams']]:
        verify(row)
    for row in c['authority_members'] + c['qualification_reuse']['receipts'] + [c['observer_identity'], c['native_input_alignment'], c['cpu_preflight']]:
        verify(row)
    for row in c['assets']:
        s = Path(row['path']).stat()
        require((s.st_size, s.st_ino, s.st_mtime_ns) == (row['bytes'], row['inode'], row['mtime_ns']), 'ASSET_CHANGED')
    require(c['settings']['B'] == 100 and c['settings']['batches'] == 20 and c['settings']['requests'] == 2000, 'HORIZON20')
    require(tuple(c['settings']['arms']) == ARMS and not c['settings']['save_checkpoints'], 'ARMS_NO_CP')
    require(c['settings']['fit_requests_per_group'] == 1, 'ORIGINAL_OWNER_GRAPH_ROUTE_ONLY')
    if 'repair_receipt' in c:
        verify(c['repair_receipt'])
        require(lock.get('repair_receipt') == c['repair_receipt'], 'FROZEN_REPAIR_RECEIPT')
    require(len(c['packs']) == 20, 'ALL20_INPUT_PACKS')
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
    a = Adapter(model, c['profile']); bench = CounterFactAdapter(tok, json.loads(Path(c['contexts']).read_text()))
    H = {l: torch.zeros(dim[1], dim[1], dtype=torch.float32) for l, dim in a.dims.items()}
    W0 = {l: w.detach().cpu().clone() for l, w in a.weights.items()}
    records = load_prefix(Path(c['stream']).parent, 2000)
    require(digest([r['case_id'] for r in records]) == c['ordered_ids_sha256'], 'ORDERED_2000')
    cold = state(a, H)
    require(cold == c['qualification_reuse']['cold_W0_H0'], 'COLD_STATE')
    write(out / 'runtime.json', dict(source=os.environ['ODEEDIT_SOURCE_COMMIT'], config=digest(c), arm=arm,
        job=os.environ.get('SLURM_JOB_ID'), device=torch.cuda.get_device_name(), cold_W0_H0=cold,
        torch=torch.__version__, transformers=transformers.__version__, FP32=True, geometry_FP64=True,
        eager=True, TF32=False, autocast=False, new_projected_qualification='NOT_YET_RUN',
        requests=2000, B=100, batches=20, checkpoint_saved=False))
    return a, bench, records, H, W0

def geometry_entry(a, entry, initial, H, W0, identity, out=None):
    owners = torch.tensor([r['request'] for r in initial['rows']], dtype=torch.long, device=a.device)
    geometries = {}
    for l in a.sites:
        rawA = build_prior_from_npz(entry['stats'][str(l)], H[l], lambda_c=15000., device=a.device)
        cov = load_native_c0(entry['stats'][str(l)], H[l].shape, device=a.device)
        geometry = build_geometry(initial['keys'][l], owners, rawA, cov, a.weights[l], W0[l],
            request_count=entry['pack']['n_requests'], identity=dict(identity, layer=l), cache_device=a.device)
        geometries[l] = geometry
        if out is not None:
            write(out / f'layer-{l}.json', geometry.receipt)
        del rawA, cov
        gc.collect(); torch.cuda.empty_cache()
    return geometries

def technical_ready(a, bench, records, H, W0, c, out):
    from .qualification import qualify
    before = state(a, H); rng = rng_snapshot(); guard = a.guard(); hooks = a.hook_signature()
    pack = bench.prepare(records[:2])
    entry = prepare_entry(a, bench, pack, H, c['stats'], 1)
    entry['source_identity'] = os.environ['ODEEDIT_SOURCE_COMMIT']
    entry['qualification_repair_receipt'] = c.get('repair_receipt')
    initial = capture_native_sites(a, entry, a.sites)
    geometries = geometry_entry(a, entry, initial, H, W0, dict(scope='TWO_NATIVE_REQUEST_FIXED_CANDIDATE'))
    result = qualify(a, entry, geometries, initial, history=H, W0=W0, out=out,
                     pad_id=bench.tokenizer.pad_token_id)
    require(state(a, H) == before and rng_equal(rng) and a.guard() == guard and a.hook_signature() == hooks, 'QUALIFICATION_MUTATION')
    a.last_virtual = {}; a.capture_virtual = False
    del entry, initial, geometries; gc.collect(); torch.cuda.empty_cache()
    write(out / 'ready.json', dict(status='MINIMAL_PROJECTED_TECHNICAL_READY', result=result,
        before=before, after=state(a, H), fixed_requests=2, fit_calls=0, updates=0,
        original_V13_math_evidence='REUSED_ONLY_UNCHANGED_WRITER_SCOPE', no_state_seed_main=True))

def reuse_w0(a, records, H, c, out, identities):
    from project.run_scripts.jlz_realization.observe import active_flags
    src = c.get('W0_reuse')
    if src is None or src.get('status') == 'NOT_AVAILABLE':
        return False
    for row in src['chunks'] + [src['summary'], src['runtime'], src['prior_config']]:
        verify(row)
    require(state(a, H) == src['state'], 'W0_REUSE_MODEL_STATE')
    ids = [r['case_id'] for r in records]; flags = active_flags(records)
    rows = rows_from(src['path'])
    summary = validate_rows(rows, identities, ids, 'W0')
    require(all(r['active_at_endpoint'] == flags[r['case_id']] for r in rows), 'W0_REUSE_ACTIVE')
    for i, start in enumerate(range(0, len(rows), 1000)):
        write(out / 'W0' / f'chunk-{i:04d}.json', dict(state=src['state'], rows=rows[start:start + 1000], optimizer_feedback=False))
    write(out / 'W0/summary.json', dict(endpoint='W0', state=src['state'], requests=2000,
        summary=summary, current=summary, seconds=0, reused_from=src['path'], new_forwards=0,
        no_mutation=True, optimizer_feedback=False, identity_receipt=digest(src)))
    return True

def drive(a, bench, records, H, W0, c, arm, out, source, attempt):
    identities = json.loads(verify(c['observer_identity']).read_text())['rows']
    previous = state(a, H); previous_rng = rng_snapshot(); commits = []
    shared_identity = dict(source=source, config=digest(c), ordered_case_ids=c['ordered_ids_sha256'],
                           B1_pack=c['packs'][0]['identity'], cold=previous, norm_price=.5, radius=.75)
    calibration = CalibrationLock(attempt / 'calibration.json', role='producer' if arm == 'CD_Q' else 'consumer', identity=shared_identity)
    if not reuse_w0(a, records, H, c, out, identities):
        observer(a, bench, records, records, H, 'W0', out / 'W0', c, identities, [r['case_id'] for r in records])
    for number, current, seen in batches(records, 100):
        require(number <= 20, 'NO_B21')
        root = out / f'batch-{number:02d}'; root.mkdir(exist_ok=False)
        require(state(a, H) == previous and rng_equal(previous_rng), 'OWN_NEXT_ENTRY_JOIN')
        pack = bench.prepare(current); expected = c['packs'][number - 1]
        require(pack['identity'] == expected['identity'] and pack['record_ids'] == expected['ids'], 'SEALED_PACK')
        if number == 2:
            write(out / 'initial.json', dict(status='MAIN_B1_COMMIT_B2_OWN_ENTRY', arm=arm,
                B1_commit=member(out / 'batch-01/commit.json'), next_entry=previous,
                source=source, config=digest(c), observer_restored=True, own_W_H_RNG_join=True))
        rng_id = _rng_hash(previous_rng)
        write(root / 'entry.json', dict(source=source, config=digest(c), arm=arm, batch=number,
            ids=pack['record_ids'], native_pack=pack['identity'], state=previous, RNG=rng_id,
            context_hash=digest(bench.contexts), seen_ids=[r['case_id'] for r in seen], fit_count=1))
        tx = BatchTransaction(a, H, bench); started = time.monotonic(); calls_before = dict(a.calls)
        events = Events(root / 'fit-events.jsonl', arm, number)
        try:
            with tx:
                pre = w0_subset(out / 'W0', current, seen, root / 'pre', previous, identities) if number == 1 else observer(
                    a, bench, seen, current, H, f'B{number}_PRE', root / 'pre', c, identities, pack['record_ids'])
                entry = prepare_entry(a, bench, pack, H, c['stats'], 1)
                require(all(len({row['request'] for row in group['rows']}) == 1
                            for group in entry['groups']), 'UNQUALIFIED_PHYSICAL_REGROUPING_DISABLED')
                initial = capture_native_sites(a, entry, a.sites)
                geometries = geometry_entry(a, entry, initial, H, W0,
                    dict(arm=arm, batch=number, native_pack=pack['identity'], source=source), root / 'geometry')
                entry['delta_zero'] = all(g.receipt['delta_zero'] for g in geometries.values())
                write(root / 'entry-capture.json', dict(native_pack=pack['identity'], ids=pack['record_ids'],
                    teacher_hash=entry['teacher_hash'], anchors={l: tensor_sha(v) for l, v in entry['anchors'].items()},
                    H_entry=previous['H'], capture_seconds=entry['seconds'], fresh_capture=True))
                cache = cache_identity(entry); guard = a.guard(); hooks = a.hook_signature()
                a.capture_virtual = True
                try:
                    plan, fit_receipt = fit(a, entry, geometries, calibration, events, root / 'fit', alpha=ALPHA[arm])
                finally:
                    a.capture_virtual = False
                virtual = virtual_terminal(a, entry, plan)
                require(state(a, H) == previous and rng_equal(previous_rng), 'FIT_MUTATED_ENTRY')
                plan_id = digest({name: {l: tensor_sha(v) for l, v in plan[name].items()} for name in ('u', 'D', 'z', 'Y')})
                write(root / 'plan.json', dict(identity=plan_id, terminal_z_same_forward=True, fit_count=1,
                    virtual_all_rows=digest({l: tensor_sha(v) for l, v in virtual.items()}),
                    native_pack=pack['identity'], ids=pack['record_ids'], persisted_plan_tensors=False))
                writer = apply(a, entry, plan, virtual, initial, H, W0, geometries, calibration.value,
                               ALPHA[arm], root / 'writer')
                require(writer['history_appends'] == len(a.sites), 'HISTORY_ONCE')
                require(cache_identity(entry) == cache and a.guard() == guard and a.hook_signature() == hooks, 'CACHE_GUARD')
                require(plan_id == digest({name: {l: tensor_sha(v) for l, v in plan[name].items()} for name in ('u', 'D', 'z', 'Y')}), 'PLAN_MUTATION')
                lambda_id = None if calibration.receipt is None else calibration.receipt['receipt_hash']
                del plan, virtual, initial, entry, geometries
                a.last_virtual = {}; gc.collect(); torch.cuda.empty_cache()
                post = observer(a, bench, seen, selected_for_post(current, seen, number), H,
                                f'W{number}', root / 'post', c, identities, pack['record_ids'])
                after = state(a, H); require(rng_equal(previous_rng), 'BATCH_RNG_DRIFT')
                receipt = dict(source=source, config=digest(c), arm=arm, batch=number, ids=pack['record_ids'],
                    native_pack=pack['identity'], before=previous, after=after, RNG_before=rng_id,
                    RNG_after=_rng_hash(rng_snapshot()), context_hash=digest(bench.contexts),
                    lambda_identity=lambda_id, allocation_price=calibration.value, fit_count=1, fit=fit_receipt,
                    plan_identity=plan_id, history_appends=writer['history_appends'], history=writer['history'],
                    layers=list(a.sites), writer=member(root / 'writer/writer.json'), observer_no_mutation=True,
                    pre=pre['summary'], post=post['summary'], post_current=post['current'],
                    post_scope='ALL_SEEN' if number in MILESTONES else 'CURRENT', seen_requests=len(seen),
                    cache_invalidated_for_next_batch=True, calls={k: a.calls[k] - calls_before[k] for k in a.calls},
                    seconds=time.monotonic() - started, checkpoint_saved=False, exact_resume='NOT_AVAILABLE')
                if number == 1:
                    match = dict(source=source, config=digest(c), plan_identity=plan_id,
                                 native_pack=pack['identity'], before=previous, after=after,
                                 lambda_identity=lambda_id, history=writer['history'])
                    if arm == 'CD_Q':
                        write(attempt / 'B1-shared-identity.json', match)
                    else:
                        require(json.loads((attempt / 'B1-shared-identity.json').read_text()) == json.loads(json.dumps(match)), 'COLD_B1_ARM_IDENTITY')
                write(root / 'prepared-commit.json', receipt)
                require(json.loads((root / 'prepared-commit.json').read_text()) == json.loads(json.dumps(receipt)), 'COMMIT_IO')
                tx.finish()
                try:
                    write(root / 'commit.json', receipt)
                except BaseException:
                    tx.done = False; raise
            commits.append(receipt); previous = after; previous_rng = rng_snapshot()
            print(json.dumps(dict(event='CD_CUMULATIVE_COMMIT', arm=arm, batch=number, requests=len(seen))), flush=True)
        except BaseException as error:
            write(root / 'rollback.json', dict(verified=tx.rollback_verified, batch_entry=tx.before,
                earlier_committed_prefix=len(commits), logical_commit=False, rollback_after_SIGTERM='NOT_VERIFIED',
                cache_context_restored=not tx.done, error_type=type(error).__name__))
            raise
    require(len(commits) == 20 and sum(r['history_appends'] for r in commits) == 100, 'CHAIN_COVERAGE')

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args(); attempt = args.attempt.resolve(); out = attempt / ('main-' + args.arm)
    out.mkdir(exist_ok=False); started = time.monotonic(); status = 'TECHNICAL_BLOCKED'; a = None
    try:
        c, lock = locked(attempt); a, bench, records, H, W0 = setup(c, out, args.arm)
        technical_ready(a, bench, records, H, W0, c, out / 'qualification')
        drive(a, bench, records, H, W0, c, args.arm, out, lock['source_commit'], attempt)
        write(out / 'ready.json', dict(status='CHAIN_COMPLETE', arm=args.arm, source=lock['source_commit'],
            config=digest(c), commits=20, requests=2000, history_appends=100, no_B21=True))
        status = 'W20_COMPLETE'
    except BaseException as error:
        write(out / 'first-error.json', dict(type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), original_KEEP=True)); raise
    finally:
        write(out / 'terminal.json', dict(status=status, arm=args.arm, source=os.environ.get('ODEEDIT_SOURCE_COMMIT'),
            commits=len(list(out.glob('batch-*/commit.json'))), seconds=time.monotonic() - started,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_VRAM_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            job=os.environ.get('SLURM_JOB_ID'), adapter_calls=None if a is None else a.calls,
            no_B21=True, checkpoint_saved=False, exact_resume='NOT_AVAILABLE'))

if __name__ == '__main__':
    main()
