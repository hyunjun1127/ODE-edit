"""CPU-only receiver/input binding; no model or old experiment execution."""
import csv,importlib,json,platform,shutil,subprocess
from pathlib import Path
from transformers import AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_writer_coupled.prepare import runtime_binding,asset_binding
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_two_arm.baseline_pilot import _import_native,_source_closure
from .common import *

def main():
    out=LOCAL/'preparation-r1';require(not out.exists(),'CREATE_ONCE_PREPARATION')
    envelope=ROOT/'messages/head/2026-10-05-jlz-v13-b1-realization-sh4.json'
    require(sha(envelope)=='b6275e6a268fb522758200f22a0a50bfbb1d1b0264a9793369f6e2aef476e886','AUTHORITY_SHA')
    manifest=ROOT/DESIGN/'artifact-manifest.json'
    require(sha(manifest)=='6e2794359186f437180afa28909f4f775b0f7f713c22a9b8fd4917fb67645d4c','MANIFEST_SHA')
    verified=[]
    for row in json.loads(manifest.read_text())['artifacts']:
        p=ROOT/DESIGN/row['file'];require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'CANONICAL_MEMBER')
        verified.append(member(p))
    command=ROOT/DESIGN/'execution-b1-sh4.json'
    require(sha(command)=='aef52c0e8715e273ff4483c1596661a46dd9fd88a2135533a282eafb342c4639','B1_OVERRIDE_SHA')
    portable=ROOT/'plans/global/2026-10-04-jlz-portable-execution-v1/runtime-contract.json'
    require(sha(portable)=='d9d9624bf873b5976a5ef22f94cc95df82a8accf8ddda469e7d0146cabde894c','PORTABLE_SHA')
    priorpath=Path('/data/janghj/ODE-edit/local/jlz-v12-shared-budget/20261004-v1/attempt-r1/config.json')
    prior=json.loads(priorpath.read_text());old={r['path']:r for r in prior['assets']}
    assets=[asset_binding(r['path'],old,member(priorpath)) for r in prior['assets']]
    runtime=runtime_binding()
    planner=[]
    for name in ('entry.py','optimize.py','optimizer.py','telemetry.py'):
        rel='project/run_scripts/jlz_shared_budget/'+name
        original=subprocess.check_output(['git','show','1d27a830274aaee49a713bd07e463b0513591c9e:'+rel],cwd=ROOT)
        require((ROOT/rel).read_bytes()==original,'FROZEN_PLANNER_BYTES:'+name)
        planner.append(member(ROOT/rel))
    data=load_prefix(Path(prior['stream']).parent,2000)
    schedule=ROOT/'plans/global/2026-10-04-jlz-v12-marginal-allocation/experiment-2k/case-schedule-first2000.csv'
    require(sha(schedule)=='dccb4da4896aa7c3f653f7192bd01ab5d8799a0b439101417d02ee287d3fd2e2','SCHEDULE_SHA')
    with schedule.open(newline='') as f:allcsv=list(csv.DictReader(f))
    require(len(allcsv)==2000 and all(int(row['case_id'])==record['case_id'] for row,record in zip(allcsv,data)),'SCHEDULE_ALL_ROWS')
    records=data[:100];tokenizer=AutoTokenizer.from_pretrained(prior['model'],local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    bench=CounterFactAdapter(tokenizer,json.loads(Path(prior['contexts']).read_text()));pack=bench.prepare(records)
    observers=[]
    for record in records:
        rw=record['requested_rewrite']
        for kind,prompts in bench.panels(record).items():
            for i,prompt in enumerate(prompts):
                row=dict(identity=digest([record['case_id'],kind,i,prompt,rw['target_new']['str'],rw['target_true']['str']]),case_id=record['case_id'],kind=kind,prompt_index=i)
                for label in ('new','true'):
                    ids,target=bench.evaluation_ids(prompt,rw['target_'+label]['str']);row[label+'_token_identity']=digest([ids,target])
                observers.append(row)
    write(out/'observer-identity.json',dict(rows=observers,raw_prompt_saved=False))
    lookup=[]
    with _import_native('/data/janghj/BLUE'):
        native=importlib.import_module('memit.memit_seq_main');closure=_source_closure('/data/janghj/BLUE')
        require(sha(native.__file__)=='f84fcf4b388ff1e5c5c9d520202d926e5b16314b8269063b57d8d25243bc716a','BLUE_SOURCE')
        for request,spec in zip(pack['requests'],pack['specs']):
            prompts=[c.format(request['prompt'])+tokenizer.decode(spec['target'][:-1]) for group in bench.contexts for c in group]+['{} is a']
            positions=[native.find_fact_lookup_idx(p,request['subject'],tokenizer,'subject_last',verbose=False) for p in prompts]
            require(positions==spec['lookup'],'NATIVE_LOOKUP')
            ref=tokenizer([p.format(request['subject']) for p in prompts],return_tensors='pt',padding=True)
            for j in range(len(prompts)):
                n=int(ref['attention_mask'][j].sum());idx=spec['offset']+j
                require(ref['input_ids'][j,:n].tolist()==pack['tokens']['input_ids'][idx,:n].tolist(),'NATIVE_TOKENS')
            lookup.append(dict(case_id=request['case_id'],lookup=positions,token_identity=digest(ref['input_ids'].tolist())))
    write(out/'native-input-alignment.json',dict(rows=lookup,B=100,fit_calls=0,GPU=False))
    free=shutil.disk_usage(LOCAL).free;require(free>=12*1024**3,'RESOURCE_BLOCKED_STORAGE')
    c={k:prior[k] for k in ('model','stream','contexts','stats','profile','native_root','stats_root')}
    c.update(instruction_id=NONCE,task_id=TASK,authority=AUTHORITY,runtime=runtime,assets=assets,native_reference=closure,
        native_hparams=str(ROOT/'project/run_scripts/memit_history_lifelong/hparams.json'),
        native_input_alignment=member(out/'native-input-alignment.json'),observer_identity=member(out/'observer-identity.json'),
        packing=dict(identity=pack['identity'],ids=pack['record_ids'],ordered100_sha256=digest(pack['record_ids']),native_rows=len(pack['row_request'])),
        settings=dict(B=100,batches=1,fit_count=1,branches=list(BRANCHES),candidate_cap=25,update_cap=24,
                      fit_requests_per_group=1,observer_microbatch=2,save_checkpoints=False,no_B2=True),
        qualification=dict(B1_subset_requests=2,extra_fits=0,FP32_atol=2e-5,FP32_rtol=2e-4,projection_atol=1e-10,projection_rtol=1e-8),
        resources=dict(task_cap=1,project_cap=3,gpu=1,cpu=8,host_mib=59392,collector_host_mib=24576,
            wall='1-00:00:00',collector_wall='04:00:00',reserve_bytes=12*1024**3,free_bytes=free,
            output_plan_GiB=dict(scalar_events=1,metrics_actions=1,source_logs=1,atomic_scratch=1,margin=8),
            host_peak_plan_GiB=dict(model_load_phase=34,execution_phase=28,limit=58),
            host_components_GiB=dict(H=3.83,rollback_H=3.83,rollback_W=1.1,entry_cache=3,native_C0_transient=4,framework=6,margin=6),
            GPU_peak_plan_GiB=dict(model=30,entry_W=1.1,activations_head=20,metric_factor_solve=14,branch_transients=5,margin=8),
            stage_policy='one layer metric/factor resident; branch snapshots RAM only; no simultaneous five endpoints',
            ETA='NOT_MEASURED_V13; finite24h ceiling, not ETA'),
        authority_members=verified+[member(manifest),member(command),member(envelope),member(portable)],planner_reuse=planner)
    write(out/'configuration.json',c)
    write(out/'full-read.json',dict(nonce=NONCE,authority=AUTHORITY,members=c['authority_members'],
        host=platform.node(),worktree=str(ROOT),session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',
        scope='정본 method/experiment/contract/telemetry/reference/audit/execution 전체 정독 및 bytes/SHA 결속',
        CSV_rows=len(allcsv),CSV_all_fields_parsed=True,executed_prefix=100,fit_count=1,branches=list(BRANCHES),
        planner_same_bytes=planner,actual_GPU='NOT_RUN',submission='NOT_SUBMITTED',old_task_mutation=False))
    print(json.dumps(dict(configuration=str(out/'configuration.json'),native_rows=len(pack['row_request']),observer_rows=len(observers),GPU='NOT_RUN')))

if __name__=='__main__':main()
