"""CPU binding of canonical bytes, compatible immutable S3 assets and all 20 inputs."""
import argparse,copy,csv,json,platform,shutil,subprocess,os
from pathlib import Path
from transformers import AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_writer_coupled.prepare import runtime_binding,asset_binding
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter,make_rows,batches
from .common import *

PRIOR=Path('/data/janghj/ODE-edit/local/jlz-v12-shared-budget/20261004-v1/attempt-s3-r1/config.json')
PRIOR_SHA='b09cfae0d6a8a29071818c114017b9f350ec9c58ec3143ac211b42179145458b'

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='preparation-r1');args=p.parse_args()
    out=LOCAL/args.name;require(not out.exists(),'CREATE_ONCE_PREPARATION')
    envelope=ROOT/ENVELOPE
    require(sha(envelope)=='b1a89b13d9da75f6e9b81a396ba9144c6f88c9b48f8da6c983dbbe88220b5a9a','AUTHORITY_SHA')
    manifest=ROOT/DESIGN/'artifact-manifest.json';verified=[]
    for row in json.loads(manifest.read_text())['files']:
        path=ROOT/DESIGN/row['path'];data=path.read_bytes()
        require(len(data)==row['bytes'] and sha(path)==row['sha256'],'CANONICAL_MEMBER:'+row['path'])
        verified.append(member(path))
    require(sha(PRIOR)==PRIOR_SHA,'PRIOR_CONFIG');prior=json.loads(PRIOR.read_text())
    for name in ('native_input_alignment','observer_identity'):verify(prior[name])
    for row in prior['native_reference']:verify(row)
    runtime=runtime_binding()
    require(runtime['source_root_sha256']==prior['runtime']['source_root_sha256'],'RUNTIME_CLOSURE_REUSE')
    assets=[]
    for r in prior['assets']:
        row=asset_binding(r['path'],{r['path']:r},member(PRIOR))
        if 'logical_path' in r:require(Path(r['logical_path']).resolve()==Path(row['path']).resolve(),'MODEL_LOGICAL_PATH');row['logical_path']=r['logical_path']
        assets.append(row)
    records=load_prefix(Path(prior['stream']).parent,2000)
    schedule=ROOT/'plans/global/2026-10-04-jlz-v12-marginal-allocation/experiment-2k/case-schedule-first2000.csv'
    require(sha(schedule)=='dccb4da4896aa7c3f653f7192bd01ab5d8799a0b439101417d02ee287d3fd2e2','SCHEDULE_SHA')
    with schedule.open(newline='') as stream:csvrows=list(csv.DictReader(stream))
    require(len(csvrows)==2000 and [int(r['case_id']) for r in csvrows]==[r['case_id'] for r in records],'ORDER2000')
    tok=AutoTokenizer.from_pretrained(prior['model'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
    bench=CounterFactAdapter(tok,json.loads(Path(prior['contexts']).read_text()))
    packing=[];physical=[];oldpack={p['batch']:p for p in prior['packing'] if p['phase']=='main'}
    for i in range(20):
        pack=bench.prepare(records[i*100:(i+1)*100]);old=oldpack[i+1]
        require(pack['identity']==old['identity'] and pack['record_ids']==old['ids'],'PACK_REUSE_IDENTITY')
        groups=list(batches(make_rows(pack),2,tok.pad_token_id,'cpu'))
        tokens=sum(t['input_ids'].numel() for _,t in groups);max_tokens=max(t['input_ids'].numel() for _,t in groups)
        physical.append(dict(batch=i+1,groups=len(groups),padded_tokens=tokens,max_group_tokens=max_tokens,
            valid_tokens=sum(int(t['attention_mask'].sum()) for _,t in groups)))
        packing.append(dict(batch=i+1,identity=pack['identity'],ids=pack['record_ids'],B=100,native_rows=len(pack['row_request'])))
    c={k:copy.deepcopy(prior[k]) for k in ('model','stream','contexts','stats','profile','native_root','stats_root','native_reference','native_input_alignment','observer_identity')}
    free=shutil.disk_usage(LOCAL).free;reserve=16*1024**3;require(free>=reserve,'STORAGE_RESERVE')
    modelconfig=json.loads((Path(c['model'])/'config.json').read_text());di=modelconfig['intermediate_size'];do=modelconfig['hidden_size'];L=len(c['profile']['eligible_layers'])
    maxpadded=max(r['padded_tokens'] for r in physical);boundary=maxpadded*(di+do)*4
    host=(L+2)*boundary+L*di**2*8+3*L*di**2*4+3*L*di*do*4+12*1024**3
    require(max(34*1024**3,host)<110*1024**3,'HOST_PLAN_119GiB')
    c.update(instruction_id=NONCE,task_id=TASK,authority=AUTHORITY,runtime=runtime,assets=assets,
        native_hparams=str(ROOT/'project/run_scripts/memit_history_lifelong/hparams.json'),packing=packing,
        settings=dict(B=100,batches=20,fit_count=20,branches=list(BRANCHES),candidate_cap=25,update_cap=24,
            shared_radius=1.5,local_radius=.75,fit_microbatch=2,observer_microbatch=2,no_B21=True,save_checkpoints=False),
        qualification=dict(actual_fixed_B=2,fixed_candidates=1,extra_fits=0,optimizer_updates=0,
            reverse_evaluations=3,subject_evaluations=4,full_native_hooks=True,probe_one_complete_owner=True,
            optional_stage_failure='original_same_method',CPU15_reuse=member(ROOT/DESIGN/'reference/runtime-audit.json'),
            compatible_S4_B1_source='2ab04d0b4d339573ee54fd85c7240d090a01e2b1',S4_GPU_is_S3_PASS=False),
        resources=dict(task_cap=1,project_cap=1,gpu=1,cpu=8,host_mib=121856,collector_host_mib=24576,
            wall='2-00:00:00',collector_wall='04:00:00',reserve_bytes=reserve,free_bytes=free,
            output_plan_GiB=dict(scalar_and_actions=1,evalraw_allprefixes=2,source_logs=1,atomic_scratch=1,margin=11),
            host_peak_plan_GiB=dict(model_load=34,execution=host/1024**3,limit=119),
            GPU_peak_plan_GiB=dict(model=30,entry_terminal_weights=2.2,factors=7.66,activations_head=20,transient=8,margin=12),
            physical_caps=dict(rows=2,padded_tokens=max(r['max_group_tokens'] for r in physical),boundary_bytes=max(r['max_group_tokens'] for r in physical)*(di+do)*4),
            physical_schedule=physical,route_selection='Existing native-order MB2, no MB sweep; larger route NOT_QUALIFIED',
            estimated_fit_hours_reference_only=2038.636*20/3600,ETA='NOT_MEASURED_S3; 48h requested maximum, old S4 timing not speedup or ETA',
            no_crossGPU_speedup_claim=True),authority_members=verified+[member(manifest),member(envelope),member(schedule)],
        parent_config=member(PRIOR),W0=dict(mode='FRESH_FIRST2000',reason='No exact reuse bound yet'))
    # Existing completed S3 W0: byte-identical observer, model/input/runtime and cold state checked again in job.
    w0=PRIOR.parent/'shared-W0';summary=w0/'summary.json';source=PRIOR.parent/'source'
    if summary.exists():
        imports=['project/run_scripts/jlz_realization/observe.py','project/run_scripts/jlz_realization/inputs.py']
        for rel in imports:require((ROOT/rel).read_bytes()==(source/rel).read_bytes(),'W0_EVALUATOR_BYTES')
        from .collect import rows_from
        from project.run_scripts.jlz_realization.observe import reduce_rows
        rows=rows_from(w0);s=json.loads(summary.read_text());require(reduce_rows(rows)==s['summary'],'W0_CPU_REDUCER')
        expected={r['identity']:r for r in json.loads(verify(c['observer_identity']).read_text())['rows']}
        require(len(rows)==26000 and {r['identity'] for r in rows}==set(expected),'W0_INPUT_COVERAGE')
        for r in rows:require(all(r[t+'_token_identity']==expected[r['identity']][t+'_token_identity'] for t in ('new','true')),'W0_TOKENS')
        require(prior['settings']['observer_microbatch']==2 and prior['profile']['eligible_layers']==c['profile']['eligible_layers'],'W0_PROFILE')
        c['W0']=dict(mode='EXACT_REUSE',summary=member(summary),members=[member(p) for p in sorted(w0.glob('*.json'))],
            source_config=member(PRIOR),source_evaluator=[member(source/r) for r in imports],
            binding='same runtime/model/tokenizer/evaluator/input tokens/MB2/TF32off/FP32; actual W0 hashes checked in runner')
    write(out/'configuration.json',c)
    write(out/'full-read.json',dict(nonce=NONCE,authority=AUTHORITY,members=c['authority_members'],
        manifest_members=len(verified),adopted_changed_files=15,CSV_rows=2000,host=platform.node(),worktree=str(ROOT),
        session='01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3',TeX='current V14 front; V13 historical appendix not applied',
        prior_assets_rebound=assets,runtime=runtime,CPU15_reused_not_LM_PASS=True,actual_GPU='NOT_RUN',submission='NOT_SUBMITTED'))
    print(json.dumps(dict(configuration=str(out/'configuration.json'),free_GiB=free/1024**3,host_plan_GiB=host/1024**3,W0=c['W0']['mode'],actual_GPU='NOT_RUN')))

if __name__=='__main__':main()
