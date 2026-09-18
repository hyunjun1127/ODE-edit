"""Four explicitly user-authorized S chains, never an M cold-episode loop.

Preserves the numerical core. T is explicitly skipped, not silently passed.
No job submission, automatic R/L advancement, teacher regeneration or cleanup.
"""
import argparse
import copy
import gc
import json
import resource
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .common import write, member, digest, tensor_sha, Timer
from .runtime import invariant
from .sequential_runtime import SequentialRuntime
from .sequential_state import S_ARMS, receive, past64, registry, atomic_tensor, atomic_json, state_identity, require_next, require_finalizer
from .binding import score_rows, quality_ok
from .geometry import RightSpace, edit_null_space
from .optimizer import optimize, Check, Observation, TrialNumericalOverflow, tensor_sha256 as header_sha
from .runner import EventStore, ideal_check, selection_seal
from .observer import CanonicalObserver
from .retained_native import endpoint_identity
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng, restore_rng

def validate_lock(lock):
    if (lock['stage'] != 'S_FOUR_USER_DIRECTED' or tuple(lock['S_arms']) != S_ARMS or
        lock['S_batches'] != 10 or len(lock['sample_order']) != 1000 or len(set(lock['sample_order'])) != 1000 or
        lock['full_numerical_validation'] != 'NOT_ESTABLISHED' or lock['T_status'] != 'SKIPPED_USER_DIRECTED' or
        lock['M_completion_dependency'] is not None or lock['R_L_allowed']):
        raise ValueError('S_SCOPE_OR_WAIVER_MISMATCH')
    if lock['lock_identity'] != digest({k:v for k,v in lock.items() if k != 'lock_identity'}):
        raise ValueError('S_LOCK_DIGEST')
    for field in ('authority', 'P_star_basis', 'technical_evidence'):
        if member(lock[field]['path']) != lock[field]:
            raise ValueError('S_INPUT_MEMBER_' + field)
    for m in lock['execution']['members']:
        if member(m['path']) != {k:m[k] for k in ('path','bytes','sha256')}:
            raise ValueError('S_FROZEN_SOURCE_IDENTITY')

def correct(rt, records, past, native, batchroot, arm, cov_floor):
    """Method core unchanged; only the S canonical Past64 guard is added."""
    WN = native['weight']
    ref = rt.reference_oracle()
    cur, rows, K, meta = rt.protected_oracle(records)
    old, oldrows = rt.past_oracle(past)
    write(batchroot/'protected-provenance.json', meta)
    write(batchroot/'past-provenance.json', dict(case_ids=[r['case_id'] for r in past], rows=oldrows,
        priority='SHA256(ENFC-v1|past|case_id),case_id-string', canonical_only=True,
        anchor='own_WN', success_filter=False, current_overwrites_excluded=True))
    pstar = torch.load(rt.lock['P_star_basis']['path'], weights_only=True, map_location='cpu', mmap=True)
    allowed = RightSpace(pstar['basis'].numpy(), np.empty((pstar['basis'].shape[1],0)),
        'RESOLVED' if pstar['basis'].shape[1] else 'REPAIR_SPACE_EMPTY', pstar['diagnostic'])
    keys = native['captures']['compute_ks'][0].T if arm == 'EN-S' else K
    reuse = rt.lock['B1_completed_calculations'] if batchroot.name == 'B001' else None
    if reuse:
        for m in reuse['members'].values():
            if member(m['path']) != m: raise ValueError('B1_COMPLETED_CALCULATION_IDENTITY')
        prior_meta=json.loads(Path(reuse['members']['provenance']['path']).read_text())
        if meta != prior_meta or tensor_sha(WN)!=reuse['WN'] or past:
            raise ValueError('B1_REUSED_GEOMETRY_INPUT_MISMATCH')
        name='EN-S' if arm=='EN-S' else 'EN-F'
        factors=torch.load(reuse['members'][name+'_factors']['path'],weights_only=True,map_location='cpu',mmap=True)
        diagnostic=json.loads(Path(reuse['members'][name+'_receipt']['path']).read_text())
        if factors['shared_allowed_basis']!=rt.lock['P_star_basis']: raise ValueError('B1_ALLOWED_RANGE_BINDING')
        space=RightSpace(allowed.basis,factors['blocked'].numpy(),factors['status'],diagnostic)
        if space.receipt()!=diagnostic: raise ValueError('B1_REUSED_FACTOR_HASH_OR_DIMENSION')
        write(batchroot/'geometry/reuse.json',dict(source=reuse,raw_unchanged=True,new_rank_calculation=0))
    else:
        with Timer(rt.timing, 'geometry'):
            space = edit_null_space(allowed, keys, kind='EN-S' if arm == 'EN-S' else 'EN-F')
    write(batchroot/'geometry/space.json', space.receipt())
    atomic_tensor(batchroot/'geometry/factors.pt', dict(shared_allowed_basis=rt.lock['P_star_basis'],
        blocked=torch.from_numpy(space.blocked), status=space.status))
    anchor = score_rows(cur, WN, rows)
    past_anchor = score_rows(old, WN, oldrows) if old else {}
    write(batchroot/'native-quality.json', dict(current=anchor, past=past_anchor))
    objective_id = digest(dict(teacher=rt.lock['teacher_manifest'],
        inputs=[c.input_identity for c in ref.caches], kind='S64_W0_forward_KL'))
    recorded = {}
    def kl(weight, gradient=False):
        try:
            result = ref.kl(weight, gradient=gradient)
            recorded[tensor_sha(weight)] = dict(loss=result[0], rows=result[2])
            return result
        except FloatingPointError as exc:
            if gradient or torch.equal(weight, WN) or str(exc) not in ('NONFINITE_KL_LOSS','NONFINITE_LOGITS_OR_TEACHER'):
                raise
            rt.guard()
            for i in rt.teacher.indices('S64'):
                _, teacher, _ = rt.teacher.document(i, 'cpu')
                if not torch.isfinite(teacher).all(): raise FloatingPointError('NONFINITE_FIXED_TEACHER') from exc
            raise TrialNumericalOverflow('FINITE_TRIAL_MODEL_NONFINITE_FIXED_TEACHER_VERIFIED') from exc
    def guard(weight):
        current = score_rows(cur, weight, rows)
        prior = score_rows(old, weight, oldrows) if old else {}
        cp, cr = quality_ok(current, anchor)
        pp, pr = quality_ok(prior, past_anchor)
        return Check(cp and pp, 'CURRENT_AND_PAST_PER_SEQUENCE_EXACT_IDS',
                     dict(current_reasons=cr, past_reasons=pr, current_rows=current, past_rows=prior))
    def actual_invariant(weight, ideal, actual):
        values = invariant(cur, rows, anchor, weight, WN, ideal, K, allowed)
        return Check(values['pass'], 'FULL_TOKEN_ACTUAL_INVARIANT', values)
    sink = EventStore(batchroot/'events', batchroot/'gradients')
    options = dict(objective=(lambda w,gradient=False:rt.covariance(ref,w,gradient)) if arm == 'EN-COV' else kl,
                   objective_id='W0_FULL_INPUT_ACTIVATION_DRIFT' if arm == 'EN-COV' else objective_id,
                   space=space, guard=guard, event=sink)
    if reuse and arm!='EN-COV':
        prior=json.loads(Path(reuse['members']['native_objective']['path']).read_text())
        if prior['objective_id']!=objective_id or member(prior['gradient']['path'])!=prior['gradient']:
            raise ValueError('B1_GRADIENT_STATE_OBJECTIVE_BINDING')
        G=torch.load(prior['gradient']['path'],weights_only=True,map_location='cpu',mmap=True)['gradient']
        options['initial_observation']=Observation(prior['loss'],G,prior['rows'],
            weight_sha256=header_sha(WN),objective_id=objective_id)
        recorded[tensor_sha(WN)]=dict(loss=prior['loss'],rows=prior['rows'],completed_M_calculation_reuse=True)
    if arm != 'EN-S': options.update(invariant=actual_invariant, proposal_check=lambda d:ideal_check(d,K))
    if arm == 'EN-COV': options['cov_resolution'] = cov_floor
    with Timer(rt.timing, 'correction_total_inclusive'):
        result = optimize(arm, WN, **options)
    rt.guard()
    # Pin selection BEFORE any official P/N or Dev call, including native fallback.
    endpoint = atomic_tensor(batchroot/'selected-L4.pt', dict(weight=result.weight, arm=arm,
        WN=tensor_sha(WN), W0=rt.identity['W0'], runtime_identity=endpoint_identity(rt.identity),
        source=rt.lock['execution']['head'], full_resume=False))
    ideal = atomic_tensor(batchroot/'selected-ideal-delta.pt', dict(ideal_delta=result.ideal_delta))
    receipt = result.receipt()
    receipt.update(endpoint=endpoint, ideal_delta=ideal, event_refs=sink.refs,
                   T='SKIPPED_USER_DIRECTED', full_numerical_validation='NOT_ESTABLISHED')
    ledger = write(batchroot/'selection-ledger.json', receipt)
    ids = [r['case_id'] for r in records]
    seal = selection_seal(batchroot.name, arm, result.weight, ids, ledger['sha256'])
    sealed = write(batchroot/'SELECTION_SEALED.json', dict(seal=seal, official_P_N_access_so_far=0,
        next_selected_weight_sha=tensor_sha(result.weight), method=arm, Past64=[r['case_id'] for r in past]))
    # S64 is observer-only for EN-COV; any post-selection calculation is labelled.
    for label, weight in (('native',WN), ('selected',result.weight)):
        key = tensor_sha(weight)
        if key not in recorded:
            value, _, docs = ref.kl(weight, gradient=False)
            recorded[key] = dict(loss=value, rows=docs, post_selection_observer=True)
        write(batchroot/f'S64-{label}.json', recorded[key])
    work = dict(reference=dict(ref.work), current=dict(cur.work), past={} if old is None else dict(old.work))
    write(batchroot/'oracle-work.json', work)
    selected = result.weight
    rt.oracles = []
    return selected, seal, sealed, receipt

def w0_reuse(rt, obs, records, weight, seal):
    spec = rt.lock['observer_reuse_binding']['W0_first1000']
    if member(spec['path']) != spec: raise ValueError('W0_REUSE_SOURCE_DRIFT')
    raw = Path(spec['path']).read_bytes(); prior = json.loads(raw)
    compatibility = obs.compatibility_for(records, weight, selection_seal=seal)
    proof = dict(status='COMPATIBILITY_SEALED', endpoint_role='W0_REFERENCE_SUBSET',
        source_raw_sha256=spec['sha256'], **compatibility,
        model_pretrained_weight_sha256=rt.identity['W0'],
        selected_case_ids=[r['case_id'] for r in records], source_population=1000,
        source_request_order_sha256=prior['request_order'], lineage=rt.lock['observer_reuse_binding'])
    proof['endpoint_weight_sha256'] = rt.identity['W0']
    proof['proof_sha256'] = digest(proof)
    return dict(source_bytes=raw, proof=proof)

def observe_batch(rt, records, entry, native, selected, seal, batchroot, batch):
    obs = CanonicalObserver(rt.model, rt.etok, runtime_identity=digest(rt.identity))
    observed = {}
    for role, weight in (('entry',entry), ('own-native',native), ('selected',selected)):
        h = tensor_sha(weight)
        this = selection_seal(batchroot.name, role, weight, [r['case_id'] for r in records],seal['selection_ledger_sha256'])
        if h in observed:
            source, prior = observed[h]
            value = copy.deepcopy(prior)
            value.update(selection_seal=this, same_endpoint_reuse=source, work={k:0 for k in prior['work']})
        else:
            value = obs.observe(records,weight,selection_seal=this,greedy=True,
                w0_reduced_metrics_reuse=w0_reuse(rt,obs,records,weight,this))
        ref = write(batchroot/f'observers/{role}-current.json', value)
        observed[h] = (ref, value)
    if batch in (5,10):
        seen = rt.records[:batch*100]
        whole = selection_seal(batchroot.name,'selected-fullseen',selected,[r['case_id'] for r in seen],seal['selection_ledger_sha256'])
        value = obs.observe(seen,selected,selection_seal=whole,greedy=True,
            w0_reduced_metrics_reuse=w0_reuse(rt,obs,seen,selected,whole))
        write(batchroot/'observers/selected-fullseen.json', value)
        dev = rt.reference_oracle('Dev128')
        loss, _, docs = dev.kl(selected,gradient=False)
        write(batchroot/'observers/Dev128.json',dict(loss=loss,rows=docs,work=dict(dev.work),controller_input=False))
        rt.oracles = []
    rt.guard()
    return write(batchroot/'OBSERVERS_COMPLETE.json',dict(work=obs.work, selection_sealed_before=True,
                 endpoint_restored=True, rng_restored=True, controller_feedback=0))

def run(lock, arm):
    validate_lock(lock)
    if arm not in S_ARMS: raise ValueError('UNAUTHORIZED_S_ARM')
    root = Path(lock['S_root'])/arm/'attempt-v1'
    root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); rt=None; batch=0; committed=0; stage='load'; entry=None; durable=False
    try:
        rt = SequentialRuntime(lock,root)
        technical = json.loads(Path(lock['technical_evidence']['path']).read_text())
        applicable=('W0','M0','P4','contexts','context_tokens','rng','teacher','records_digest','torch','transformers','microbatch','physical_layer')
        if any(rt.identity[k]!=technical['identity'][k] for k in applicable):
            raise ValueError('INHERITED_ASSET_BINDING_MISMATCH_NOT_T_PASS')
        write(root/'validation-status.json',dict(T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED',
            advancement='USER_DIRECTED_M_TO_S_NOT_ESTABLISHED', arm=arm, independent_from_cancelled_M=True))
        write(root/'nonselected-before.json',rt.byte_hash_nonselected())
        ledger=[]; parent=None; parent_commit=None
        for batch in range(1,11):
            durable=False; stage='entry'; records=rt.records[(batch-1)*100:batch*100]
            batchroot=root/f'B{batch:03d}'; batchroot.mkdir()
            entry=(rt.W.detach().cpu().clone(),rt.M.clone(),capture_rng())
            identity=state_identity(entry[0],entry[1],entry[2],ledger,rt.context)
            if parent is not None: require_next(parent,identity)
            link=write(batchroot/'ENTRY.json',dict(identity=identity,parent_commit=parent_commit,
                ordinal=[(batch-1)*100,batch*100],case_ids=[r['case_id'] for r in records],
                cold=(batch==1),history_appends_before=batch-1,arm=arm))
            if batch==2:
                write(root/'S_INITIAL_VALID.json',dict(status='S_INITIAL_VALID',scope='THIS_ARM_B1_COMMIT_OBSERVER_B2_ENTRY',
                    arm=arm,parent_commit=parent_commit,next_entry=link,T='SKIPPED_USER_DIRECTED',
                    full_numerical_validation='NOT_ESTABLISHED', efficacy_claim=False))
            stage='native'; native=rt.native_batch(records,batchroot/'native',batch)
            WN=native['weight']; branch_rng=capture_rng()
            past=past64(ledger,records)
            stage='correction'
            selected,seal,sealed,selection=correct(rt,records,past,native,batchroot,arm,technical['EN_COV_resolution'])
            # Controller/cache/observer work does not advance the next batch RNG.
            restore_rng(branch_rng); rt.oracles=[]; rt.copy_weight(selected)
            stage='history'; history=rt.finalize_batch(records)
            require_finalizer(history,identity['M'],tensor_sha(selected),tensor_sha(rt.M))
            next_ledger=receive(ledger,records); next_rng=capture_rng()
            next_state=state_identity(selected,rt.M,next_rng,next_ledger,rt.context)
            stage='durable_checkpoint'
            checkpoint=atomic_tensor(batchroot/'checkpoint.pt',dict(weight=selected,M4=rt.M,
                RNG=next_rng,contexts=rt.context,received_ledger=next_ledger,active_registry=registry(next_ledger),
                next_batch=batch+1,arm=arm,commit_id=f'{lock["attempt"]}:{arm}:B{batch:03d}',
                source=lock['execution']['head'],lock_identity=lock['lock_identity'],sample_order=lock['sample_order'],
                identity=endpoint_identity(rt.identity),state_identity=next_state,
                selected=sealed,history=history,full_pretrained_model_copied=False,GPU_continuation_tested=False))
            parent_commit=atomic_json(batchroot/'COMMIT.json',dict(status='COMMITTED',checkpoint=checkpoint,selected=sealed,
                state=next_state,history=history,history_append=1,inner_history_append=0,
                batch=batch,requests=100,arm=arm,next_batch=batch+1))
            durable=True;committed=batch;ledger=next_ledger;parent=next_state
            # The decision and next state are now immutable. P/N is observer-only.
            stage='observers'; rt.oracles=[]; gc.collect(); torch.cuda.empty_cache()
            observe_batch(rt,records,entry[0],WN,selected,seal,batchroot,batch)
            restore_rng(next_rng)
            require_next(parent,state_identity(rt.W,rt.M,capture_rng(),ledger,rt.context))
            write(batchroot/'BATCH_COMPLETE.json',dict(commit=parent_commit,selection=selection,
                timing_cumulative=dict(rt.timing),wall_seconds=time.monotonic()-started,
                peak_GPU_bytes=torch.cuda.max_memory_allocated(),host_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            entry=None
            del native,WN,selected,selection,past
            gc.collect()
        stage='terminal'
        after=rt.byte_hash_nonselected()
        if after!=json.loads((root/'nonselected-before.json').read_text()): raise RuntimeError('TERMINAL_NONSELECTED_BYTES')
        write(root/'TERMINAL.json',dict(status='COMPLETED',arm=arm,commits=10,requests=1000,history_appends=10,
            final_checkpoint=checkpoint,last_commit=parent_commit,fullseen_final=str(root/'B010/observers/selected-fullseen.json'),
            wall_seconds=time.monotonic()-started,timing=rt.timing,allocation_utilization='NOT_MEASURED',
            T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED'))
    except BaseException as exc:
        rollback='NOT_ATTEMPTED_NO_ENTRY'
        if rt is not None and entry is not None and not durable:
            try: rt.rollback_entry(*entry); rollback='ENTRY_W_M_CONTEXT_RNG_RESTORED'
            except BaseException as rb: rollback='FAILED:'+repr(rb)
        elif durable: rollback='NOT_ROLLED_BACK_DURABLY_COMMITTED_PREFIX_PRESERVED'
        write(root/'failure.json',dict(status='TECHNICAL_FAILURE',arm=arm,batch=batch,stage=stage,
            error=repr(exc),traceback=traceback.format_exc(),committed_prefix=committed,
            rollback=rollback,wall_seconds=time.monotonic()-started,
            full_numerical_validation='NOT_ESTABLISHED'))
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True,type=Path)
    p.add_argument('--index',required=True,type=int); a=p.parse_args()
    if not 0<=a.index<len(S_ARMS): raise ValueError('ARRAY_INDEX')
    run(json.loads(a.lock.read_text()),S_ARMS[a.index])

if __name__=='__main__': main()
