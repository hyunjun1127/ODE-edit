"""Read-only terminal CPU review; never imports/calls a model or scheduler API."""
import argparse
import csv
import io
import json
import math
from collections import Counter
from pathlib import Path
from .preparation import ROOT,sha,member,create_json,create_bytes
from .reducer import reduce_raw,pair
from .config import ARMS

LOCAL=ROOT/'completed-review-v1'
ATTEMPT=ROOT/'B1/attempt-v1'
REPORT='experiment-reports/servers/server4/en-execution-reuse-r512-g256-20260919-v1/completed-review-v1'


def read(p):return json.loads(Path(p).read_text())


def reused(observation):
    return bool(observation.get('same_episode_exact_endpoint_reuse') or observation.get('reduced_metrics_reuse_proof'))


def csvout(path,rows):
    if not rows:return create_bytes(path,b'')
    s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return create_bytes(path,s.getvalue().encode())


def inputs():
    lock=read(ATTEMPT/'execution.ready.lock.json');raw=Path(lock['output'])
    from scripts.fixed_counterfact import load_prefix
    records=load_prefix(lock['dataset_root'],100)
    return lock,raw,records


def first(accounting_source=None):
    from .accounting import parse
    accounting_source=Path(accounting_source or LOCAL/'accounting-once.txt')
    account=parse(accounting_source.read_text(),{'50410':'PREP','50449':'B1'})
    account.update(accounting_queries=1,accounting_timestamp_timezone='Asia/Seoul',
        raw_accounting=member(accounting_source),new_GPU=0,new_submissions=0)
    create_json(LOCAL/'accounting.json',account)
    lock,raw,records=inputs()
    if read(raw/'terminal.json')['status']!='B1_COMPLETE':raise ValueError('TERMINAL_REQUIRED')
    reduced={n:reduce_raw(read(raw/'observers'/f'{n}.json'),records) for n in ('W0','N4',*ARMS)}
    rows=[]
    for name,value in reduced.items():
        observation=read(raw/'observers'/f'{name}.json')
        for tag,m in value['metrics'].items():
            rows.append(dict(endpoint=name,panel=tag,numerator=m['numerator'],denominator=m['denominator'],
                percent=m['percent'],ties=m['ties'],new_strict=m['new_strict'],true_strict=m['true_strict'],
                endpoint_sha256=value['endpoint'],identity_root=value['pair_identity_root'],
                new_observer_forwards=observation.get('new_observer_forwards'),
                explicit_reuse=reused(observation)))
    table=csvout(LOCAL/'first-table.csv',rows)
    pairs={a+'__'+b:pair(reduced[a],reduced[b]) for a,b in (('N4',ARMS[0]),('N4',ARMS[1]),ARMS)}
    create_json(LOCAL/'first-paired-local-only.json',pairs)
    create_json(LOCAL/'first-table-receipt.json',dict(table=table,strict={n:v['strict'] for n,v in reduced.items()},
        inputs=[member(raw/'observers'/f'{n}.json') for n in reduced],
        status='INDEPENDENT_RAW_NLL_IDENTITY_FINITE_CARDINALITY; STATE_SOURCE_AUDIT_PENDING',new_GPU=0))
    print(json.dumps(dict(table=table,rows=rows,strict={n:v['strict'] for n,v in reduced.items()}),indent=2))


def base(repo):
    from . import report
    report.REPORT=REPORT
    print(report.run(ATTEMPT/'execution.ready.lock.json',repo,LOCAL/'accounting.json',LOCAL/'base-analysis'))


def check_capsule(cap, row, eos):
    n=cap['actual_length'];y=cap['y0'];prompt=cap['input_ids']
    assert 1<=n<=256 and len(y)==n and len(prompt)==129 and prompt==row['input_ids']
    assert cap['source_row_id']==row['source_row_id'] and cap['role']==row['role']
    assert cap['tf_input_ids']==prompt+y[:-1]
    assert cap['score_positions']==list(range(128,128+n))
    assert cap['attention_mask']==[1]*(128+n) and cap['position_ids']==list(range(128+n))
    assert not any(t in eos for t in y[:-1])
    stopped=y[-1] in eos
    assert stopped or n==256
    assert cap['length_censored']==(not stopped and n==256)
    return stopped


def replay_guard(anchor, rows):
    assert set(anchor)==set(rows)
    strict=pair_count=new=0;deltas=[]
    for key,a in anchor.items():
        b=rows[key]
        for field in ('case_id','sequence_id','labels','positions','branch','kind','context'):
            assert a[field]==b[field]
        assert math.isfinite(b['nll']) and math.isfinite(a['nll'])
        assert b['strict']==(b['predictions']==b['labels'])
        if a['branch']=='new':
            new+=1;deltas.append(b['nll']-a['nll'])
            assert b['nll']<=a['nll']+1e-4
            if a['strict']:strict+=1;assert b['strict']
            if a['kind']=='canonical':
                old=key.rsplit(':',1)[0]+':old'
                if a['nll']<anchor[old]['nll']:
                    pair_count+=1;assert b['nll']<rows[old]['nll']
    return dict(rows=len(rows),new_sequences=new,native_strict_retained=strict,
                native_canonical_pair_retained=pair_count,max_new_NLL_increase=max(deltas),status='CPU_SCALAR_ID_REPLAY_PASS')


def supplement(repo):
    from project.run_scripts.single_layer_edit_preserving_correction.common import digest
    lock,raw,records=inputs();directory=Path(repo)/REPORT
    ready=read(lock['generated_ready']['path']);mp=Path(ready['manifest']['path']);manifest=read(mp)
    assert sha(mp)==ready['manifest']['sha256']
    rp=Path(lock['reference_inputs']['path']);reference=read(rp)
    assert sha(rp)=='507b202108acc90e10810455f57da60df14e9e9e31ba567c8c75f51cf76afaeb'
    eos=set(manifest['binding']['generation']['eos_token_ids']);caps=[];payload=Counter()
    for item,row in zip(manifest['documents'],reference,strict=True):
        p=mp.parent/item['capsule']['path'];cap=read(p)
        assert sha(p)==item['capsule']['sha256'] and p.stat().st_size==item['capsule']['bytes']
        stopped=check_capsule(cap,row,eos)
        caps.append(dict(index=item['index'],role=cap['role'],source_role=row['source_role'],
                         T=cap['actual_length'],EOS=stopped,censored=cap['length_censored']))
        for key in ('logp','keys','residual'):
            target=mp.parent/item[key]['path'];assert target.stat().st_size==item[key]['bytes']
            payload[key]+=target.stat().st_size
    assert len(caps)==640 and [r['role'] for r in caps]==['R512']*512+['Dev128']*128
    summary=[]
    for role in ('R512','Dev128'):
        rs=[r for r in caps if r['role']==role];lengths=[r['T'] for r in rs]
        summary.append(dict(role=role,documents=len(rs),positions=sum(lengths),minimum_T=min(lengths),
            maximum_T=max(lengths),mean_T=sum(lengths)/len(rs),EOS_stopped=sum(r['EOS'] for r in rs),
            censored=sum(r['censored'] for r in rs),TF_tokens=sum(128+t for t in lengths)))
    csvout(directory/'generated-length-summary.csv',summary)
    csvout(directory/'generated-length-histogram.csv',[dict(role=role,T=t,documents=n)
        for role in ('R512','Dev128') for t,n in sorted(Counter(c['T'] for c in caps if c['role']==role).items())])
    create_json(directory/'generated-capsule-audit.json',dict(status='CPU_640_CAPSULE_SHA_SHIFT_EOS_IDENTITY_PASS',
        source_roles=dict(Counter(r['source_role'] for r in reference)),payload_actual_bytes=dict(payload),
        teacher_full_SHA='PRIOR_RUNTIME_FULL_MEMBER_VERIFICATION_REUSED; CURRENT_STAT_AND_MANIFEST_BOUND',
        TF_argmax_equals_y0='PRIOR_ACTUAL_CANONICAL_TF_RECEIPTS_REUSED; NO_NEW_FORWARD',
        manifest=member(mp),input=member(rp),new_GPU=0))
    work=read(mp.parent/'work.json')
    assert len(work['documents'])==640 and all(x['generation_TF_exact'] for x in work['documents'])
    totals={k:sum(d.get(k,0) for d in work['documents']) for k in ('generation_seconds','canonical_TF_seconds',
        'write_hash_seconds','document_wall_seconds','generation_decoder_forwards','canonical_TF_decoder_forwards',
        'generated_tokens','generation_input_tokens','canonical_TF_input_tokens','logical_payload_bytes')}
    create_json(directory/'preparation-work-audit.json',dict(totals=totals,new_documents=work['newly_built_documents'],
        reused_documents=work['reused_complete_documents'],timers='NESTED_NOT_ADDED',source=member(mp.parent/'work.json')))
    sweeps=[];guard=[];rowbanks=[];observer=[]
    for arm in ARMS:
        path=raw/'arms'/arm;exact=read(path/'execution-exactness.json');bank={}
        for p in sorted((path/'events').glob('*.json')):
            e=read(p)
            for key in ('objective_rows','trial_objective_rows'):
                rows=e.get('saved',{}).get(key)
                if rows:bank[digest(rows)]=rows
        assert len(bank)==5
        for i,s in enumerate(exact['full_sweep_rows']):
            rows=bank[s['rows_sha256']];assert len(rows)==512
            for j,row in enumerate(rows):
                assert row['index']==j and row['ordinal']==j and row['role']=='R512'
                assert row['source_row_id']==reference[j]['source_row_id'] and row['scored_positions']==caps[j]['T']
                assert row['input_tokens']==row['valid_tokens']==128+caps[j]['T'] and math.isfinite(row['loss'])
            mean=sum(x['loss'] for x in rows)/512
            assert mean==s['loss'] and s['coverage']['positions']==sum(c['T'] for c in caps[:512])
            assert s['coverage']['documents']==512 and s['coverage']['vocabulary_size']==128256
            assert s['coverage']['backward_documents']==(512 if s['gradient'] else 0)
            sweeps.append(dict(arm=arm,sweep=i,gradient=s['gradient'],documents=512,positions=s['coverage']['positions'],
                independently_reduced_loss=mean,rows_sha256=digest(rows),status='SAVED_ROWS_EXACT_MEAN_AND_COVERAGE_PASS'))
        assert exact['gradient_sweeps']==1 and exact['method_gradient_shared'] is False
        anchor=read(path/'native-current-anchor.json')
        for t in exact['trials']:
            f=t['full_trial'];assert f['eta']==exact['eta0']*(.5**t['trial'])
            assert f['armijo_bound']==f['current_loss']+1e-4*f['p_actual']
            assert t['armijo']==(t['loss']<=f['armijo_bound'])
            if t['guard']:
                guard.append(dict(arm=arm,trial=t['trial'],**replay_guard(anchor,t['guard']['details']['rows'])))
        rowbanks.append(bank)
    assert rowbanks[0]==rowbanks[1]
    csvout(directory/'independent-objective-row-reduction.csv',sweeps)
    csvout(directory/'independent-guard-replay.csv',guard)
    for arm in ('W0','N4',*ARMS):
        value=read(raw/'observers'/f'{arm}.json');generation=value['generation']
        observer.append(dict(endpoint=arm,explicit_observation_reuse=reused(value),
            generation_requests=len(generation),greedy_target_prefix_success=sum(x['target_prefix_match'] for x in generation),
            EOS_stopped=sum(x['stopped_on_original_eos'] for x in generation),max32_reached=sum(x['reached_max_new_tokens'] for x in generation),
            target_over32=sum(x['target_over_32_censored'] for x in generation),
            generated_tokens=sum(x['generated_length'] for x in generation),source_raw_SHA=sha(raw/'observers'/f'{arm}.json')))
    csvout(directory/'observer-reuse-and-generation.csv',observer)
    create_json(directory/'analysis-correction-log.json',dict(
        issue='초기 first-table의 explicit_reuse가 null-valued key 존재를 true로 센 분석 metadata 오류',
        correction='실제 proof 값과 별도 prior proof를 검산한다. Legacy는 새 observer, Reuse는 명시적 동일episode 관측 재사용이다.',
        numeric_counts_denominators_changed=False,raw_mutated=False,
        original_first_table=member(LOCAL/'first-table.csv'),corrected_metadata='observer-reuse-and-generation.csv'))
    sources=[]
    for value in [lock['execution']['archive'],*lock['execution']['members']]:
        p=Path(value['path']);assert p.stat().st_size==value['bytes'] and sha(p)==value['sha256'];sources.append(value)
    create_json(directory/'execution-source-recheck.json',dict(execution=lock['execution']['commit'],tree=lock['execution']['tree'],
        files=len(sources),members_root=digest(sources),status='FROZEN_ARCHIVE_AND_SOURCE_FULL_SHA_PASS',
        model_teacher='PRIOR_EXACT_IDENTITY_VERIFICATION_REUSED; NOT_FULL_REHASHED_THIS_REVIEW'))
    print(json.dumps(dict(lengths=summary,payload=dict(payload),guard=guard,observer=observer,prep=totals),indent=2))


def reconstruct(repo):
    import torch
    from project.run_scripts.single_layer_edit_preserving_correction.geometry import RightSpace
    from project.run_scripts.single_layer_edit_preserving_correction.alltoken import tensor_sha256
    torch.set_num_threads(8)
    lock,raw,_=inputs();end=read(raw/'terminal.json');arm=ARMS[0]
    exact=read(raw/'arms'/arm/'execution-exactness.json')
    G=torch.load(raw/'arms'/arm/'gradients'/(exact['gradient_sha256']+'.pt'),weights_only=True,mmap=True,map_location='cpu')['gradient']
    factors=torch.load(raw/'geometry/EN-F-factors.pt',weights_only=True,mmap=True,map_location='cpu')
    allowed=torch.load(lock['P_star_basis']['path'],weights_only=True,mmap=True,map_location='cpu')
    basis=allowed['basis'];space=RightSpace(basis.numpy(),factors['blocked'].numpy(),factors['status'])
    H=space.project(G);h=tensor_sha256(H);assert h==exact['projected_gradient_sha256']
    chi=float(torch.sum(H*H));assert chi==exact['chi'] and exact['loss']/chi==exact['eta0']
    native=torch.load(end['commits']['N4']['checkpoint']['path'],weights_only=True,mmap=True,map_location='cpu')['weight']
    rows=[]
    for t in exact['trials']:
        f=t['full_trial'];ideal=-f['eta']*H;candidate=(native.double()+ideal).float();actual=candidate.double()-native.double()
        checks={key:tensor_sha256(value)==f[key] for key,value in
            (('weight_sha256',candidate),('ideal_sha256',ideal),('actual_sha256',actual))}
        p=float(torch.sum(G*actual));checks['p_actual_exact']=p==f['p_actual']
        assert all(checks.values()),checks
        if t['decision']['accepted']:
            for name in ARMS:
                saved=torch.load(end['commits'][name]['checkpoint']['path'],weights_only=True,mmap=True,map_location='cpu')['weight']
                assert torch.equal(candidate,saved)
        rows.append(dict(trial=t['trial'],candidate_sha256=tensor_sha256(candidate),p_actual=p,**checks))
    create_json(Path(repo)/REPORT/'saved-tensor-reconstruction.json',dict(status='CPU_EXACT_H_AND_ALL_FOUR_FP32_TRIALS_PASS',
        H_sha256=h,chi=chi,eta0=exact['eta0'],trials=rows,selected_matches_both_saved_checkpoints=True,
        method_gradient='SAVED_G_REUSED_NO_AUTOGRAD',model_loads=0,forwards=0,new_GPU=0,GPU_continuation='NOT_TESTED'))
    print('CPU exact H, chi, eta0, four candidates/actual deltas/p and both selected checkpoints PASS')


def finalize(repo):
    """Publish only this new review package; never rewrite an execution artifact."""
    from .publication_checks import check
    from project.run_scripts.single_layer_edit_preserving_correction.common import digest
    directory=Path(repo)/REPORT;lock,raw,records=inputs();end=read(raw/'terminal.json')
    def output(name,value):
        p=directory/name
        assert p.parent==directory and p.suffix=='.json'
        # These are generated review outputs, not original scientific seals.
        p.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
        return member(p)
    proofrows=[]
    for role in ('W0','N4'):
        proof=read(raw/'observers'/f'prior-{role}-proof.json');source=proof['source']
        assert sha(source['path'])==source['sha256']
        old=read(source['path']);new=read(raw/'observers'/f'{role}.json')
        assert old['raw']==new['raw'] and old['raw_token_payload']==new['raw_token_payload']
        assert all(proof['old_runtime'][k]==proof['new_runtime'][k] for k in proof['matched_fields'])
        assert all(proof['old_compatibility'][k]==v for k,v in proof['new_compatibility'].items() if k!='runtime_identity')
        proofrows.append(dict(endpoint=role,mode='PRIOR_SAME_HOST_CANONICAL_REUSE',proof=member(raw/'observers'/f'prior-{role}-proof.json'),
            prior_raw=source,canonical_raw_and_tokens_equal=True,complete_nonselected_model_bytes='PRIOR_RUNTIME_BINDING_VERIFIED',new_canonical_forwards=0))
    legacy=read(raw/'observers'/f'{ARMS[0]}.json');reuse=read(raw/'observers'/f'{ARMS[1]}.json')
    assert legacy['raw']==reuse['raw'] and legacy['raw_token_payload']==reuse['raw_token_payload'] and legacy['generation']==reuse['generation']
    assert reused(reuse) and not reused(legacy)
    output('observer-proof-audit.json',dict(prior=proofrows,legacy='NEW_PHYSICAL_OBSERVER',
        reuse='EXPLICIT_IDENTICAL_ENDPOINT_OBSERVER_REUSE',same_episode_raw_tokens_generation_equal=True,
        legacy_observer=member(raw/'observers'/f'{ARMS[0]}.json'),reuse_observer=member(raw/'observers'/f'{ARMS[1]}.json')))
    oldrows=list(csv.DictReader((directory/'observer-reuse-and-generation.csv').open()))
    for row in oldrows:
        row['explicit_observation_reuse']=row['endpoint']!=ARMS[0]
        row['reuse_kind']='PRIOR_CANONICAL' if row['endpoint'] in ('W0','N4') else 'SAME_EPISODE' if row['endpoint']==ARMS[1] else 'NONE'
        row['generation_status']='NOT_MEASURED' if row['endpoint']=='W0' else 'OBSERVED_MAX32_REACHED_NOT_TARGET_LENGTH_CENSORED'
    s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(oldrows[0]));w.writeheader();w.writerows(oldrows)
    (directory/'observer-reuse-and-generation.csv').write_text(s.getvalue())
    correction=read(directory/'analysis-correction-log.json')
    correction.update(issue='초기표의 null key 존재 기반 재사용 표시 오류; 수치 변화 없음',
        correction='prior-W0/N4-proof와 actual same-episode proof를 검산하여 W0/N4 prior reuse, Legacy fresh, Reuse same-episode로 구별',
        corrected_metadata='observer-reuse-and-generation.csv')
    output('analysis-correction-log.json',correction)
    first=[]
    for name in ('N4',*ARMS):
        r=reduce_raw(read(raw/'observers'/f'{name}.json'),records)
        for metric,m in r['metrics'].items():first.append(dict(endpoint=name,metric=metric,numerator=m['numerator'],
            denominator=m['denominator'],percent=m['percent'],ties=m['ties'],observation_reuse=name!=ARMS[0]))
    if not (directory/'first-table.csv').exists():csvout(directory/'first-table.csv',first)
    cost=[];ready=read(lock['generated_ready']['path']);account=read(LOCAL/'accounting.json')
    for job in account['jobs']:
        p=job['parent'];role=job['role'];seconds=int(p['ElapsedRaw'])
        cost.append(dict(scope=role,kind='ACTUAL_PARENT_ALLOCATION',seconds=seconds,GPUh=seconds/3600,
            note='parent only; utilization NOT_MEASURED'))
    for name,seconds in [('PREP_PROGRAM',ready['seconds']),('B1_PROGRAM',end['total_program_seconds']),
                         ('PRIOR_SHARED_NATIVE',end['prior_native_seconds'])]:
        cost.append(dict(scope=name,kind='RECORDED_WALL' if name!='PRIOR_SHARED_NATIVE' else 'REUSED_PRIOR_COST_NOT_NEW',
            seconds=seconds,GPUh='',note='not added to allocation'))
    common=sum(end['setup_timing'][k] for k in ('model_load_seconds','teacher_binding_seconds','reference_cache_load_seconds',
        'native_reuse_binding_seconds','current_cache_and_keys_seconds','shared_geometry_seconds') if k in end['setup_timing'])
    for name in ARMS:
        seconds=end['arm_work'][name]['wall_seconds']
        cost.append(dict(scope=name,kind='CONTROLLER_WALL_OBSERVED',seconds=seconds,GPUh='',note='fixed order; page-cache not flushed'))
        cost.append(dict(scope=name,kind='ARITHMETIC_STANDALONE_WITH_PRIOR_NATIVE',seconds=seconds+end['prior_native_seconds'],
            GPUh='',note='constructed, not new standalone job; setup/observer excluded'))
        cost.append(dict(scope=name,kind='ARITHMETIC_WITH_COMMON_PREPARATION_AND_B1_SETUP',seconds=seconds+ready['seconds']+common,
            GPUh='',note='constructed; shared stages counted once per standalone view, not added across arms'))
    if not (directory/'cost-boundaries.csv').exists():csvout(directory/'cost-boundaries.csv',cost)
    output('cost-boundary-definitions.json',dict(common_B1_setup_seconds=common,selected_keys=[k for k in
        ('model_load_seconds','teacher_binding_seconds','reference_cache_load_seconds','native_reuse_binding_seconds',
         'current_cache_and_keys_seconds','shared_geometry_seconds') if k in end['setup_timing']],
        research_GPU_seconds=account['allocated_GPU_seconds'],new_native_fits=0,
        prior_native_not_added_to_allocation=True,all_composite_wall_views='ARITHMETIC_NOT_MEASURED_STANDALONE',
        incomplete_timer_boundaries='purewriter/checkpoint_IO_NOT_SEPARATED',new_review_GPU=0))
    if not (directory/'checkpoint-inventory.csv').exists():
        csvout(directory/'checkpoint-inventory.csv',[dict(endpoint=arm,path=r['checkpoint']['path'],
            bytes=r['checkpoint']['bytes'],file_sha256=r['checkpoint']['sha256'],W4_raw_sha256=r['identity']['W'],
            M4_raw_sha256=r['identity']['M'],CPU_schema_finite_hash='PASS',history_receipt=1,GPU_continuation='NOT_TESTED')
            for arm,r in end['commits'].items()])
    doc_hashes={
        'plans/global/2026-09-19-en-execution-reuse-r512-g256-design-v1.md':'b671e1eb26d01068b561cf32144d66291f43c8f11505d550e6945229b43b16bd',
        'plans/global/2026-09-19-en-execution-reuse-r512-g256-contract-v1.json':'84dc6d7b35afb21cb095606a490132ccfc3b72bdbd5ac6ef3de2d241ce1700a0',
        'audits/global/2026-09-19-en-execution-reuse-design-checks.json':'34a1e5a00a55e3bb6c5f61ef8315a8825e7231043e71d02b49d3f280c19c675d',
        'PROTOCOL.md':'af806a449be800251393bfcd81b2dfa5689ee34305f3bf1323e0fae82f16c87b'}
    for path,h in doc_hashes.items():assert sha(Path(repo)/path)==h
    output('full-read-binding.json',dict(exact_prior_FULL_READ_reused=doc_hashes,
        prior_ack='messages/acks/server4/2026-09-19-en-execution-reuse-r512-g256-b1.md',
        new_envelope_read_whole=True,actual_analysis_source_read=True,
        historical_pending_not_rewritten=True,new_GPU=0))
    inputs_to_seal=[ATTEMPT/'execution.lock.json',ATTEMPT/'execution.ready.lock.json',ATTEMPT/'submission.json',
        raw/'terminal.json',raw/'artifact-manifest.json',ROOT/'PREP/attempt-v1/output/READY.json',
        ROOT/'receipts/user-monitoring-pause-r1.json',ROOT/'receipts/b1-dependent-pending-handoff-r1.json']
    code=[Path(repo)/'project/run_scripts/en_execution_reuse'/n for n in
        ('review_completed_20260919.py','test_completed_review_20260919.py','reducer.py','report.py','artifact_audit.py','publication_checks.py')]
    output('analysis-manifest.json',dict(instruction='ODEEDIT-S06-EN-REUSE-R512-G256-B1-COMPLETED-REVIEW-SH4-V1',
        authority=member(Path(repo)/'messages/head/2026-09-19-sh4-en-reuse-g256-b1-completed-review.md'),
        source_files=[member(p) for p in code],input_receipts=[member(p) for p in inputs_to_seal],
        initial_table=member(LOCAL/'first-table.csv'),accounting=member(LOCAL/'accounting-once.txt'),
        actual_execution=lock['execution']['commit'],analysis_source='EXACT_FILE_SHAS; PUBLICATION_COMMIT_IN_GIT',
        scheduler_queries=1,new_model_evaluator_calls=0,new_GPU=0,separate_agent_red=False,
        old_runtime_raw_mutated=False,max_batches=1,sequential_authorized=False,auto_continue=False))
    output('publication-checks.json',check(directory))
    members=[member(p) for p in sorted(directory.iterdir()) if p.name not in ('manifest.json','rooted-receipt.json')]
    output('manifest.json',dict(members=members,member_root=digest(members),raw_tensor_prompt_stdout_in_git=False))
    output('rooted-receipt.json',dict(report=member(directory/'diagnostic-report-ko.md'),manifest=member(directory/'manifest.json'),
        analysis=member(directory/'analysis-manifest.json'),local_paired=member(LOCAL/'base-analysis/paired-exact-case-rows.json'),
        new_GPU=0,new_model_evaluations=0,status='CPU_DETAILED_REVIEW_COMPLETE',monitoring_active=False,
        automatic_resume=False,sequential_authorized=False,separate_agent_red=False))
    print(json.dumps(dict(report=member(directory/'diagnostic-report-ko.md'),manifest=member(directory/'manifest.json'),
        receipt=member(directory/'rooted-receipt.json'),cost=cost),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['first','base','supplement','reconstruct','finalize']);p.add_argument('--repo',type=Path)
    p.add_argument('--local-root',type=Path);p.add_argument('--accounting-source',type=Path)
    args=p.parse_args()
    if args.local_root:
        if not args.local_root.resolve().is_relative_to(ROOT/'completed-review-v1'):raise ValueError('REVIEW_LOCAL_SCOPE')
        LOCAL=args.local_root
    if args.phase=='first':first(args.accounting_source)
    else:globals()[args.phase](args.repo)
