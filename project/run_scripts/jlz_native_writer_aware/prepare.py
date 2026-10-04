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
    envelope=ROOT/'messages/head/2026-10-05-jlz-v14-native-writer-b1-sh4.json'
    require(sha(envelope)=='0ac6db9ba2f799592a993dc29849737af4701af8388b66e621d4729a46020033','AUTHORITY_SHA')
    manifest=ROOT/DESIGN/'artifact-manifest.json'
    require(sha(manifest)=='6e384357a897d7f7f81e2bb3679ed9eb2ffb8f1bac431ab995e9fe5b9486554f','MANIFEST_SHA')
    verified=[]
    for row in json.loads(manifest.read_text())['files']:
        p=ROOT/DESIGN/row['path'];require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'CANONICAL_MEMBER')
        verified.append(member(p))
    command=ROOT/DESIGN/'execution-b1-sh4.json'
    require(sha(command)=='ce8aca3296fa6b666a3c2eaf63f78aa8a1ce8c3b48aaf98f689c0a75b21d1c75','B1_OVERRIDE_SHA')
    priorpath=Path('/data/janghj/ODE-edit/local/jlz-v12-shared-budget/20261004-v1/attempt-r1/config.json')
    prior=json.loads(priorpath.read_text());old={r['path']:r for r in prior['assets']}
    assets=[asset_binding(r['path'],old,member(priorpath)) for r in prior['assets']]
    runtime=runtime_binding()
    planner=[]
    for name in ('optimizer.py',):
        rel='project/run_scripts/jlz_shared_budget/'+name
        original=subprocess.check_output(['git','show','5fb35c2dc3ec04e723b9caefa1c4bb1158124708:'+rel],cwd=ROOT)
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
    from project.run_scripts.jlz_realization.inputs import make_rows,batches
    native_rows=make_rows(pack)
    padded=sum(tokens['input_ids'].numel() for _,tokens in batches(native_rows,2,tokenizer.pad_token_id,'cpu'))
    # Five layer boundaries (key+residual), one incoming adjoint, entry cache;
    # model load and execution host peaks are not simultaneously resident.
    boundary_bytes=padded*(14336+4096)*4
    host_execution=5*boundary_bytes+2*boundary_bytes+5*14336**2*8+2*5*14336**2*4+5*4096*14336*4+8*1024**3
    require(host_execution<54*1024**3,'RESOURCE_BLOCKED_HOST_PLAN')
    c.update(instruction_id=NONCE,task_id=TASK,authority=AUTHORITY,runtime=runtime,assets=assets,native_reference=closure,
        native_hparams=str(ROOT/'project/run_scripts/memit_history_lifelong/hparams.json'),
        native_input_alignment=member(out/'native-input-alignment.json'),observer_identity=member(out/'observer-identity.json'),
        packing=dict(identity=pack['identity'],ids=pack['record_ids'],ordered100_sha256=digest(pack['record_ids']),native_rows=len(pack['row_request'])),
        settings=dict(B=100,batches=1,fit_count=1,branches=list(BRANCHES),candidate_cap=25,update_cap=24,
                      fit_microbatch=2,observer_microbatch=2,save_checkpoints=False,no_B2=True),
        qualification=dict(B1_subset_shapes=[1,2,3],extra_fits=0,FP32_atol=1e-5,FP32_rtol=1e-4,
                           gradient_RMS_atol=1e-6,gradient_RMS_rtol=1e-3,ridge_relative_residual=1e-8,budget_atol=1e-6,
                           fixed_candidates='B1 zero; B2/B3 deterministic feasible sin(.025*a); no optimizer updates',
                           reference='whole-graph dense cast/add autograd plus native full hooks, same tokens and teacher',
                           loss='per-request native mean; gradients SUM, RMS across all tensor components'),
        resources=dict(task_cap=1,project_cap=3,gpu=1,cpu=8,host_mib=59392,collector_host_mib=24576,
            wall='1-00:00:00',collector_wall='04:00:00',reserve_bytes=12*1024**3,free_bytes=free,
            output_plan_GiB=dict(scalar_events=1,metrics_actions=1,source_logs=1,atomic_scratch=1,margin=8),
            host_peak_plan_GiB=dict(model_load_phase=34,execution_phase=host_execution/1024**3,limit=58),
            host_plan=dict(padded_native_tokens=padded,boundary_bytes=boundary_bytes,five_CPU_boundaries=True,
                           dense_gradient_weight_buffer=False,all_native_rows_retained=True),
            GPU_peak_plan_GiB=dict(model=30,entry_W=1.1,terminal_W=1.1,activations_head=20,five_factors=7.66,
                                  factor_transient=3,gradient_transient=2,margin=8),
            stage_policy='CPU stage boundaries; row-group reverse replay; whole-B solve once per layer; RAM-only',
            ETA='NOT_MEASURED_V14; finite24h ceiling, not ETA'),
        authority_members=verified+[member(manifest),member(command),member(envelope)],planner_reuse=planner)
    write(out/'configuration.json',c)
    write(out/'full-read.json',dict(nonce=NONCE,authority=AUTHORITY,members=c['authority_members'],
        host=platform.node(),worktree=str(ROOT),session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',
        scope='정본 11파일 method/runtime/comparison/reference/audit/validation/manifest/execution 전체 정독 및 bytes/SHA 결속',
        CSV_rows=len(allcsv),CSV_all_fields_parsed=True,executed_prefix=100,fit_count=1,branches=list(BRANCHES),
        planner_same_bytes=planner,actual_GPU='NOT_RUN',submission='NOT_SUBMITTED',old_task_mutation=False))
    print(json.dumps(dict(configuration=str(out/'configuration.json'),native_rows=len(pack['row_request']),observer_rows=len(observers),GPU='NOT_RUN')))

if __name__=='__main__':main()
