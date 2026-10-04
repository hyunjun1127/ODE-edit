"""SH3 path/runtime port of the preserved SH4 preparation; scientific code unchanged."""
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
    require(sha(ROOT/'messages/head/2026-10-04-jlz-v12-migrate-sh4-to-sh3.json')=='57fa8e2626e0a4eb52c38d2fdab0141894ef88221c1cfa0d9a31bf0b6aefabd4','MIGRATION_ENVELOPE_SHA')
    manifest=ROOT/DESIGN/'artifact-manifest.json'
    execution=ROOT/DESIGN/'execution-manifest.json'
    require(sha(manifest)=='0f9298c77b30c2e7cfc12734ec24fffb31bb4b68349004536e9a42510497f159','MANIFEST_SHA')
    require(sha(execution)=='3028d2796efb646fca5e316ea09678b1d2f539869cdbed30b99f27d05162be45','EXEC_MANIFEST_SHA')
    members={r['path']:r for p in (manifest,execution) for r in json.loads(p.read_text())['files']}
    members=list(members.values())
    require(len(members)==26,'PACKAGE_27_INCLUDING_EXEC_MANIFEST')
    for row in members:
        path=ROOT/row['path'];data=path.read_bytes();data.decode('utf-8')
        require(len(data)==row['bytes'] and sha(path)==row['sha256'],'PACKAGE_MEMBER:'+row['path'])
    previous=Path('/data/janghj/ODE-edit/local/jlz-realized-subject-v10/20261003-v1/attempt-v2/config.json')
    prior=json.loads(previous.read_text())
    s4_config=LOCAL/'inputs/server4-handoff-r1/execution/config.json'
    require(sha(s4_config)=='18bb56e23ad996f03a7d30e7ed57171f97cdc71df0f1f6ef730198ef5b530f05','S4_FROZEN_CONFIG')
    s4=json.loads(s4_config.read_text())
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
    with _import_native('/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/blue-upstream'):
        mod=importlib.import_module('memit.memit_seq_main')
        native=_source_closure('/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/blue-upstream')
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
    require(free>=24*1024**3,'RESOURCE_BLOCKED_STORAGE')
    config=dict(instruction_id=NONCE,task_id=TASK,authority='fdfda7343eb44443ec5a846a82935206b96d6047',migration_instruction='ODEEDIT-GH-SH4-SH3-JLZ-V12-MIGRATION-20261004-R1',
        model=prior['model'],stream=prior['stream'],contexts=prior['contexts'],stats=prior['stats'],profile=s4['profile'],
        runtime=runtime,assets=assets,native_reference=native,native_root='/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/blue-upstream',
        native_hparams=str(ROOT/'project/run_scripts/memit_history_lifelong/hparams.json'),
        native_input_alignment=member(out/'native-input-alignment.json'),
        native_writer_sha=sha('/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/blue-upstream/memit/memit_seq_main.py'),stats_root='/data/janghj/EasyEdit/examples/data/stats',
        packing=packing,observer_identity=member(out/'observer-identity.json'),
        settings=dict(seed=20261002,fit_requests_per_group=1,observer_microbatch=2,save_checkpoints=False,early_stop_request=.05,
                      candidate_cap=25,update_cap=24,chains=['V12_MAIN'],batches=20,allseen=list(MILESTONES),no_B21=True),
        W0=dict(mode='ONE_SHARED_FRESH_FIRST2000',reason='Historical runtime package-byte and layout equivalence not established',
                old_reference_only=True,allowed_by_current_envelope=True),
        resources=dict(task_cap=1,project_cap=1,gpu=1,cpu=8,host_mib=59392,collector_host_mib=24576,
            wall='7-00:00:00',qualification_wall='04:00:00',collector_wall='04:00:00',reserve_bytes=24*1024**3,free_bytes=free,
            output_plan_GiB=dict(percase_scalar=3,candidate_scalar=3,logs_and_source=2,atomic_temp=4,margin=12),
            host_plan_GiB=dict(history=3.83,transaction_history=3.83,rollback_weights=1.1,prior_CPU=1.54,entry_cache=3,
                               native_covariance=3.83,loader_peak=30,margin=4),
            host_plan_note='Loader peak and fit priors/cache are different phases: loader peak34 GiB; fit history/rollback8GiB + cache3GiB + write one prior/solve scratch <12GiB, margin4GiB; actual RSS gate/receipt separate',
            GPU_plan_GiB=dict(model=30,prior_factor=0,weights_and_write=3,activation_and_head=20,native_dense_solve=10,margin=8),
            ETA_hours='NOT_MEASURED_V12; pilot emits cost estimate; requested168h is ceiling not ETA',
            no_implicit_resource_increase=True),
        authority_members=[member(ROOT/r['path']) for r in members]+[member(execution),member(ROOT/ENVELOPE),member(ROOT/EXCEPTION)])
    # v10 is an immutable asset identity reference, never v12 state or qualification.
    for asset, old in zip(assets, prior['assets']):
        if 'logical_path' in old:
            require(Path(old['logical_path']).resolve()==Path(asset['path']).resolve(),'MODEL_LOGICAL_BINDING')
            asset['logical_path']=old['logical_path']
    config['authority_members'] += [member(ROOT/'messages/head/2026-10-04-jlz-v12-migrate-sh4-to-sh3.json'),
        member(ROOT/'control/experiment-exceptions/jlz-v12-migration-sh3-20261004.json'),
        member(ROOT/'audits/servers/server4/jlz-v12-shared-budget-bs100x20-20261004-v1/migration-to-s3-r1/cancellation.json')]
    require(config['settings']==s4['settings'] and packing==s4['packing'],'S4_CONFIG_INPUT_EQUIVALENCE')
    config['source_handoff']=member(s4_config)
    config['authority_members'].append(member(s4_config))
    config['runtime_port']=dict(base_python='/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python',
        isolated_overlay=str(LOCAL/'dependencies-r1'),shared_environment_changed=False,
        previous_S3_transformers='4.44.2',selected_transformers='4.57.1',
        actual_S3_model_qualification='NOT_RUN',S4_pilot='HISTORICAL_ONLY')
    write(out/'configuration.json',config)
    write(out/'full-read.json',dict(nonce=NONCE,authority='fdfda7343eb44443ec5a846a82935206b96d6047',migration_instruction='ODEEDIT-GH-SH4-SH3-JLZ-V12-MIGRATION-20261004-R1',members=members,manifest=member(manifest),
        schedule=schedule,CSV_role='all2000/allfields verified; old eval checked for identity ONLY, v12 pre/post/W15 schedule separate',
        package_count=27,source_host=platform.node(),worktree=str(ROOT),archive_pull=0,
        implementation_status='IMPLEMENTING_NOT_SUBMITTED',actual_GPU='NOT_RUN',old_v9_mutation=False))
    print(json.dumps(dict(configuration=str(out/'configuration.json'),members=27,packs=len(packing),observer_rows=len(observers),GPU='NOT_RUN')))

if __name__=='__main__':main()

