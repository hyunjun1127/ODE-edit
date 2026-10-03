"""CPU receiver verification and source/input/runtime/resource freeze inputs."""
import argparse
import importlib
import json
import platform
import shutil
from pathlib import Path
from transformers import AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_writer_coupled.prepare import runtime_binding,asset_binding,validate_schedule
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_two_arm.baseline_pilot import _import_native,_source_closure
from .common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='preparation-r1');args=p.parse_args()
    out=LOCAL/args.name;require(out.parent==LOCAL and not out.exists(),'CREATE_ONCE_PREPARATION')
    require(sha(ROOT/ENVELOPE)=='58f35d4327bba742ecf6adb7a7f75dca0211d942c7728dd0ce88fb7d1ffa1b15','ENVELOPE_SHA')
    manifest=ROOT/DESIGN/'artifact-manifest.json'
    require(sha(manifest)=='1f5f20a37419693e89278e6bcb0b841321e96492a8be697f16fee79b9c8ef560','MANIFEST_SHA')
    members=json.loads(manifest.read_text())['files'];require(len(members)==16,'PACKAGE_17_INCLUDING_MANIFEST')
    for row in members:
        path=ROOT/row['path'];data=path.read_bytes();data.decode('utf-8')
        require(len(data)==row['bytes'] and sha(path)==row['sha256'],'PACKAGE_MEMBER:'+row['path'])
    previous=Path('/data/janghj/ODE-edit/local/jlz-realization-v9/20261004-2k-v1/attempt-r2/config.json')
    prior=json.loads(previous.read_text())
    assets=[asset_binding(r['path'],{x['path']:x for x in prior['assets']},member(previous)) for r in prior['assets']]
    data=load_prefix(Path(prior['stream']).parent,2004)
    require(sha(prior['stream'])=='3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1','DATASET_SHA')
    _,schedule=validate_schedule(data,ROOT/DESIGN/'experiment-2k/case-schedule-first2000.csv',
        ROOT/'plans/global/2026-10-02-jlz-native-joint-v4/experiment-2k/evaluation-schedule.json')
    require(schedule['ordered_case_ids_sha256']=='0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4','ORDER')
    require([r['case_id'] for r in data[2000:2004]]==[541,6693,16935,17306],'PILOT_DISJOINT')
    tokenizer=AutoTokenizer.from_pretrained(prior['model'],local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    bench=CounterFactAdapter(tokenizer,json.loads(Path(prior['contexts']).read_text()))
    packing=[]
    for phase,start,B,count in [('pilot',2000,2,2),('main',0,100,20)]:
        for n in range(count):
            pack=bench.prepare(data[start+n*B:start+(n+1)*B])
            packing.append(dict(phase=phase,batch=n+1,B=B,ids=pack['record_ids'],identity=pack['identity'],
                native_rows=len(pack['row_request']),width=pack['tokens']['input_ids'].shape[1],
                valid_tokens=int(pack['tokens']['attention_mask'].sum())))
    observers=[]
    for record in data[:2000]:
        rw=record['requested_rewrite']
        for kind,prompts in bench.panels(record).items():
            for i,prompt in enumerate(prompts):
                row=dict(identity=digest([record['case_id'],kind,i,prompt,rw['target_new']['str'],rw['target_true']['str']]),
                         case_id=record['case_id'],kind=kind,prompt_index=i)
                for label in ('new','true'):
                    ids,target=bench.evaluation_ids(prompt,rw['target_'+label]['str'])
                    row[label+'_token_identity']=digest([ids,target]);row[label+'_token_count']=len(target)
                observers.append(row)
    write(out/'observer-identity.json',dict(rows=observers,raw_prompt_saved=False))
    runtime=runtime_binding()
    with _import_native('/data/janghj/BLUE'):
        mod=importlib.import_module('memit.memit_seq_main')
        native=_source_closure('/data/janghj/BLUE')
        require(sha(mod.__file__)=='f84fcf4b388ff1e5c5c9d520202d926e5b16314b8269063b57d8d25243bc716a','NATIVE_BLUE_WRITER')
        native_lookup=[]
        for phase,start,B,count in [('pilot',2000,2,2),('main',0,100,20)]:
            for n in range(count):
                pack=bench.prepare(data[start+n*B:start+(n+1)*B])
                for r,spec in zip(pack['requests'],pack['specs']):
                    prompts=[c.format(r['prompt'])+tokenizer.decode(spec['target'][:-1]) for group in bench.contexts for c in group]+['{} is a']
                    positions=[mod.find_fact_lookup_idx(p,r['subject'],tokenizer,'subject_last',verbose=False) for p in prompts]
                    require(positions==spec['lookup'],'ORIGINAL_NATIVE_LOOKUP')
                    reference=tokenizer([p.format(r['subject']) for p in prompts],return_tensors='pt',padding=True)
                    for j in range(len(prompts)):
                        length=int(reference['attention_mask'][j].sum());idx=spec['offset']+j
                        require(reference['input_ids'][j,:length].tolist()==pack['tokens']['input_ids'][idx,:length].tolist(),'ORIGINAL_NATIVE_TOKENS')
                    native_lookup.append(dict(case_id=r['case_id'],lookup=positions,token_identity=digest(reference['input_ids'].tolist())))
    write(out/'native-input-alignment.json',dict(rows=native_lookup,scope='original BLUE lookup and original per-request tokenization, all2000 plus4dev',GPU=False))
    free=shutil.disk_usage(LOCAL).free
    require(free>=40*1024**3,'RESOURCE_BLOCKED_STORAGE')
    config=dict(instruction_id=NONCE,task_id=TASK,authority=AUTHORITY,
        model=prior['model'],stream=prior['stream'],contexts=prior['contexts'],stats=prior['stats'],profile=prior['profile'],
        runtime=runtime,assets=assets,native_reference=native,native_root='/data/janghj/BLUE',
        native_hparams=str(ROOT/'project/run_scripts/memit_history_lifelong/hparams.json'),
        native_input_alignment=member(out/'native-input-alignment.json'),
        native_writer_sha=sha('/data/janghj/BLUE/memit/memit_seq_main.py'),stats_root='/data/janghj/EasyEdit/examples/data/stats',
        packing=packing,observer_identity=member(out/'observer-identity.json'),
        settings=dict(seed=20261002,fit_microbatch=2,observer_microbatch=2,save_checkpoints=False,early_stop_mean=.05,
                      candidate_cap=25,update_cap=24,chains=list(CHAINS),batches=20,allseen=list(MILESTONES),no_B21=True),
        W0=dict(mode='ONE_SHARED_FRESH_FIRST2000',reason='Historical runtime package-byte and layout equivalence not established',
                old_reference_only=True,allowed_by_current_envelope=True),
        resources=dict(task_cap=1,project_cap=2,gpu=1,cpu=8,host_mib=59392,collector_host_mib=24576,
            wall='7-00:00:00',qualification_wall='04:00:00',collector_wall='04:00:00',reserve_bytes=40*1024**3,free_bytes=free,
            output_plan_GiB=dict(percase_scalar=3,candidate_scalar=1,logs_and_source=2,atomic_temp=4,margin=30),
            host_plan_GiB=dict(history=3.83,transaction_history=3.83,rollback_weights=1.1,prior_CPU=7.66,entry_cache=3,
                               native_covariance=3.83,loader_peak=30,margin=4),
            host_plan_note='Loader peak and fit priors/cache are different phases: peak max(30+4,3.83*3+1.1+7.66+3+4)=34 GiB; actual RSS gate/receipt separate',
            GPU_plan_GiB=dict(model=30,prior_factor=7.66,weights_and_write=3,activation_and_head=20,native_dense_solve=12,margin=8),
            ETA_hours='NOT_MEASURED_V11; pilot emits cost estimate; requested168h is ceiling not ETA',
            no_implicit_resource_increase=True),
        authority_members=[member(ROOT/r['path']) for r in members]+[member(manifest),member(ROOT/ENVELOPE),member(ROOT/EXCEPTION)])
    write(out/'configuration.json',config)
    write(out/'full-read.json',dict(nonce=NONCE,authority=AUTHORITY,members=members,manifest=member(manifest),
        schedule=schedule,CSV_role='all2000/allfields verified; old eval checked for identity ONLY, v11 pre/post/W15 schedule separate',
        package_count=17,source_host=platform.node(),worktree=str(ROOT),archive_pull=0,
        implementation_status='IMPLEMENTING_NOT_SUBMITTED',actual_GPU='NOT_RUN',old_v9_mutation=False))
    print(json.dumps(dict(configuration=str(out/'configuration.json'),members=17,packs=len(packing),observer_rows=len(observers),GPU='NOT_RUN')))

if __name__=='__main__':main()
