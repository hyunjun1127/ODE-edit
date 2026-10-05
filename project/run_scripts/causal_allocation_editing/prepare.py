"""CPU-only fresh authority/runtime/asset and all twenty native-pack binding."""
import argparse
import copy
import csv
import json
import os
import platform
import shutil
import unicodedata
from pathlib import Path
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_writer_coupled.prepare import runtime_binding, asset_binding
from project.run_scripts.jlz_realization.observe import active_flags
from . import *
from .entry import make_native_rows

PRIOR = Path('/data/janghj/ODE-edit/local/jlz-v13-mdcd-sequential-2k/20261005-v1/attempt-r1')
MANIFEST_SHA = 'f421d54e2a85883deae62246a16b268cf9ef152fd092fbc5d254dd0f376227a4'
SCHEDULE_SHA = 'dccb4da4896aa7c3f653f7192bd01ab5d8799a0b439101417d02ee287d3fd2e2'

def verify_schedule(path, records):
    require(sha(path) == SCHEDULE_SHA, 'SCHEDULE_SHA')
    with path.open(newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
    require(fields == ['stream_index0','batch1','slot0','case_id','claim_sha256','target_new_sha256','active_at_W20']
            and len(rows) == len(records) == 2000, 'SCHEDULE_ALL_FIELDS_ROWS')
    flags = active_flags(records)
    for index, (row, record) in enumerate(zip(rows, records)):
        rw = record['requested_rewrite']
        expected = [index,index//100+1,index%100,record['case_id'],
            digest([unicodedata.normalize('NFC',' '.join(rw['subject'].split())),rw['relation_id']]),
            digest(rw['target_new']),int(flags[record['case_id']])]
        require([row[k] for k in fields] == list(map(str,expected)), 'SCHEDULE_VALUES:'+str(index))
        require(len(record['paraphrase_prompts'])==2 and len(record['neighborhood_prompts'])==10,'OBSERVER_DENOMINATORS')
    return rows

def w0_receipt(prior, identities, records):
    folder = PRIOR / 'main-CD/W0'
    if not (folder / 'summary.json').is_file():
        return dict(status='NOT_AVAILABLE', reason='EXACT_W0_RAW_ABSENT')
    summary = json.loads((folder / 'summary.json').read_text())
    cold = prior['qualification_reuse']['cold_W0_H0']
    require(summary['endpoint']=='W0' and summary['state']==cold and summary['requests']==2000
            and summary['no_mutation'] and summary['optimizer_feedback'] is False,'W0_IDENTITY_STATE')
    rows = rows_from(folder,cold)
    require(validate_rows(rows,identities,[r['case_id'] for r in records],'W0')==summary['summary'],'W0_RAW_REDUCER')
    runtime = json.loads((PRIOR/'main-CD/runtime.json').read_text())
    require(runtime['config']==digest(prior) and runtime['cold_W0_H0']==cold,'W0_RUNTIME_IDENTITY')
    return dict(status='EXACT_REUSE_PENDING_ACTUAL_COLD_CHECK',path=str(folder),state=cold,
        summary=member(folder/'summary.json'),chunks=[member(p) for p in sorted(folder.glob('chunk-*.json'))],
        runtime=member(PRIOR/'main-CD/runtime.json'),prior_config=member(PRIOR/'config.json'),
        ordered_rows=digest([r['identity'] for r in rows]),new_forwards=0,
        scope='Same-runtime/model/tokenizer/input/evaluator raw W0 only; no old fitted state or old algorithm PASS')

def full_native_binding(prior,records,packs):
    """CPU tokenizer only: sealed native FULL rows and numeric staging bound."""
    from transformers import AutoTokenizer
    from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
    tokenizer=AutoTokenizer.from_pretrained(prior['model'],local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    bench=CounterFactAdapter(tokenizer,json.loads(Path(prior['contexts']).read_text()))
    measured=[]
    for index in range(20):
        pack=bench.prepare(records[index*100:(index+1)*100]);expected=packs[index]
        require(pack['identity']==expected['identity'] and pack['record_ids']==expected['ids'],
                'FRESH_NATIVE_FULL_PACK_IDENTITY:'+str(index+1))
        rows=make_native_rows(pack);owner_widths=[]
        for owner in range(pack['n_requests']):
            own=[r for r in rows if r['request']==owner]
            require(len(own)==pack['n_rw']+1 and sum(r['kind']=='kl' for r in own)==1,'NATIVE_OWNER_ROW_COVERAGE')
            owner_widths.append(max(len(r['tokens']['input_ids']) for r in own))
        token_rows=[dict(global_row=r['global_row'],owner=r['request'],kind=r['kind'],lookup=r['lookup'],
            input_ids=r['tokens']['input_ids'].tolist(),mask=r['tokens']['attention_mask'].tolist(),
            targets=r['target'].tolist()) for r in rows]
        measured.append(dict(batch=index+1,pack=pack['identity'],requests=pack['n_requests'],rows=len(rows),
            valid_tokens=sum(len(r['tokens']['input_ids']) for r in rows),
            owner_padded_tokens=sum(owner_widths)*(pack['n_rw']+1),max_owner_width=max(owner_widths),
            full_rows_sha256=digest(token_rows),entry_source=member(ROOT/'project/run_scripts/causal_allocation_editing/entry.py')))
    T=max(v['owner_padded_tokens'] for v in measured);GiB=1024**3
    # Widths are the sealed Llama3-8B down-proj input/output dimensions.
    # Current+trial payloads coexist until the Armijo decision. Old/new
    # reverse boundaries coexist; no offload buffer is counted as a tensor CP.
    boundary=(14336+4096)*4*T
    payload=(4*(14336+4096)+2*4096)*4*T
    extra=dict(current_and_trial_payload=2*payload/GiB,entry_prefix=boundary/GiB,
        reverse_old_and_new_boundaries=2*boundary/GiB,reverse_head=4096*4*T/GiB)
    peak=27.034+sum(extra.values())
    require(peak<58,'RESOURCE_BLOCKED_HOST_STAGING_PLAN')
    return dict(status='CPU_FULL_NATIVE_INPUT_BOUND',scope='All20 original native full-token packs; no model load/GPU qualification',
        KL_future_tokens_retained=True,all20=measured,rows=sum(v['rows'] for v in measured),
        max_batch_owner_padded_tokens=T,execution_extra_GiB=extra,execution_peak_estimate_GiB=peak,
        bound_is_estimate_not_measured_peak=True,prior_receipt_scope='Original pack/token hashes only; old cropped entry route is NOT inherited')

def prepare(out, attempt, cpu_preflight):
    out, attempt, cpu_preflight = map(lambda p:Path(p).resolve(),(out,attempt,cpu_preflight))
    require(out.is_relative_to(LOCAL) and attempt.is_relative_to(LOCAL),'TASK_LOCAL_BOUNDARY')
    require(not attempt.exists() and not (out/'configuration.json').exists(),'CREATE_ONCE_PREPARATION')
    verify(member(cpu_preflight)); cpu = json.loads(cpu_preflight.read_text())
    require(cpu.get('passed') is True or cpu.get('status')=='PASS','PRODUCTION_CPU_PREFLIGHT_REQUIRED')
    envelope = json.loads((ROOT/ENVELOPE).read_text())
    require(envelope['nonce']==NONCE and envelope['task_id']==TASK,'AUTHORITY')
    manifest = ROOT/DESIGN/'artifact-manifest.json'
    require(sha(manifest)==MANIFEST_SHA==envelope['source_adoption']['manifest_sha256'],'DESIGN_MANIFEST_SHA')
    authority = [member(ROOT/p) for p in (ENVELOPE,SCHEDULE,DESIGN+'/artifact-manifest.json')]
    for relative, expected in json.loads(manifest.read_text())['sha256'].items():
        p = ROOT/relative
        require(p.is_file() and not p.is_symlink() and sha(p)==expected,'DESIGN_MEMBER:'+relative)
        authority.append(member(p))
    prior = json.loads((PRIOR/'config.json').read_text()); oldlock=json.loads((PRIOR/'execution.lock.json').read_text())
    require(sha(PRIOR/'config.json')==oldlock['config_sha256'],'PRIOR_CONFIG_SOURCE_BINDING')
    for r in oldlock['native_reference']+oldlock['dependency_sources']:verify(r)
    runtime=runtime_binding()
    require((runtime['torch'],runtime['transformers'])==(prior['runtime']['torch'],prior['runtime']['transformers']),'RUNTIME_VERSION')
    oldruntime={r['module']:r['sha256'] for r in prior['runtime']['source_members']}
    require(all(oldruntime.get(r['module'])==r['sha256'] for r in runtime['source_members']),'RUNTIME_SOURCE_CLOSURE')
    oldassets={r['path']:r for r in prior['assets']}
    assets=[asset_binding(r['path'],oldassets,member(PRIOR/'config.json')) for r in prior['assets']]
    readonly=[]
    for relative in ('project/run_scripts/jlz_realization/inputs.py','project/run_scripts/jlz_realization/observe.py',
                     'project/run_scripts/jlz_pilot/prompts.py','scripts/fixed_counterfact.py'):
        require(sha(ROOT/relative)==sha(PRIOR/'source'/relative),'INPUT_EVALUATOR_EXACT_BYTES:'+relative)
        readonly.append(member(ROOT/relative))
    alignment_path=verify(prior['native_input_alignment']);observer_path=verify(prior['observer_identity'])
    alignment=json.loads(alignment_path.read_text());identities=json.loads(observer_path.read_text())['rows']
    require(len(identities)==26000 and len({r['identity'] for r in identities})==26000,'ALL_OBSERVER_TOKEN_IDENTITIES')
    require(len(alignment['rows'])==2000 and len(alignment['packs'])==20,'ALL_NATIVE_REQUEST_IDENTITIES')
    records=load_prefix(Path(prior['stream']).parent,2000)
    require(digest([r['case_id'] for r in records])==ORDERED_SHA,'ORDERED_FIRST2000')
    verify_schedule(ROOT/SCHEDULE,records)
    packs=copy.deepcopy(prior['packs'])
    require(packs==alignment['packs'] and [p['ids'] for p in packs]==[[r['case_id'] for r in records[i:i+100]]
            for i in range(0,2000,100)],'ALL20_PACKS_ORDER_TOKEN_BINDING')
    require(sum(p['native_rows'] for p in packs)==14000,'ALL14000_NATIVE_ROWS')
    full_binding=full_native_binding(prior,records,packs)
    write(out/'native-full-input-binding.json',full_binding)
    reuse=w0_receipt(prior,identities,records)
    write(out/'W0-reuse.json',reuse);reuse['identity_receipt']=member(out/'W0-reuse.json')
    LOCAL.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(LOCAL).free;inodes=os.statvfs(LOCAL).f_favail;reserve=12*1024**3
    require(free>=reserve and inodes>=10000,'RESOURCE_BLOCKED_STORAGE')
    c={k:copy.deepcopy(prior[k]) for k in ('model','stream','contexts','stats','profile','native_root','stats_root')}
    c.update(instruction_id=NONCE,task_id=TASK,experiment_name='Causal Allocation Editing',run_instance=dict(date='2026-10-06',attempt=attempt.name),
        attempt=str(attempt),seed=20261002,runtime=runtime,assets=assets,authority_members=authority,
        native_reference=oldlock['native_reference'],dependency_sources=oldlock['dependency_sources'],
        native_hparams=prior['native_hparams'],
        observer_identity=member(observer_path),native_input_alignment=member(alignment_path),
        native_full_input_binding=member(out/'native-full-input-binding.json'),
        packs=packs,ordered_ids_sha256=ORDERED_SHA,cpu_preflight=member(cpu_preflight),
        prior_asset_binding=member(PRIOR/'config.json'),readonly_input_evaluator_sources=readonly,
        cold_W0_H0=prior['qualification_reuse']['cold_W0_H0'],W0_reuse=reuse,
        settings=dict(B=100,batches=20,requests=2000,main_arms=1,per_layer_cap=.75,shared_sum_cap=None,
            norm_coefficient=.5,lambda_KL=.0625,lambda_C=15000,accepted_update_cap=24,
            logical_candidate_cap=50,full_gradient_cap=25,fit_requests_per_group=1,
            observer_microbatch=2,milestones=list(MILESTONES),save_checkpoints=False,no_B21=True,new_baseline=0),
        qualification=dict(native_requests=2,fixed_candidates=1,extra_fit=0,updates=0,
            tolerance_action=dict(atol=2e-5,rtol=2e-4),tolerance_gradient=dict(atol=1e-6,rtol=2e-4),
            scope='Two same-candidate complete-owner graphs; actual causal loss/Q/full gradients; no standalone B100 fit'),
        resources=dict(task_cap=1,project_cap=3,cpu=8,gpu=1,host_mib=59392,hard_host_mib=60416,
            collector_host_mib=24576,wall='2-00:00:00',qualification_wall='04:00:00',collector_wall='04:00:00',reserve_bytes=reserve,
            startup_free_bytes_min=6*1024**3,free_bytes=free,free_inodes=inodes,
            output_plan_GiB=dict(scalar_metrics_events=3,source_atomic=2,margin=7),
            host_peak_plan_GiB=dict(load_phase_bound=40,execution_fixed_H_and_rollback=7.657,
                CPU_raw_A_five=7.657,CPU_selected_payload_and_rollback=2.188,
                staged_boundaries='CPU full native owner padded-token estimate; not a measured GPU/RSS peak',
                factor_scratch_one_layer=1.532,python_native_and_head_scratch=8,
                fixed_execution_sum_before_boundaries=27.034,limit=58,
                token_based_execution_extra=full_binding['execution_extra_GiB'],
                token_based_execution_peak_estimate=full_binding['execution_peak_estimate_GiB'],
                admission='Separate load and execution phases; not asserting model-load40 and all execution arrays coexist'),
            GPU_peak_plan_GiB=dict(model=30,factor_five=7.657,selected_payload=1.094,
                stage_activation_and_head='bounded one native-owner graph',memory_status='ESTIMATE_NOT_GPU_PASS'),
            ETA='UNMEASURED_NEW_METHOD;48h finite request ceiling, not extrapolated old fit timing'))
    write(out/'configuration.json',c)
    write(out/'input-runtime-binding.json',dict(nonce=NONCE,worktree=str(ROOT),host=platform.node(),
        canonical_members=authority,CSV_rows=2000,all_fields_checked=True,native_rows=14000,
        observer_rows=26000,all20_packs=packs,priorSHA_plus_freshstat_assets=assets,runtime=runtime,
        native_full_input_binding=member(out/'native-full-input-binding.json'),
        W0_reuse_receipt=member(out/'W0-reuse.json'),new_model_load=False,actual_GPU='NOT_RUN',
        synthetic_CPU_scope='12+12 mathematics only; not production model PASS'))
    return dict(configuration=str(out/'configuration.json'),packs=20,requests=2000,actual_GPU='NOT_RUN')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--cpu-preflight',type=Path,required=True);args=p.parse_args()
    print(json.dumps(prepare(args.out,args.attempt,args.cpu_preflight)))
