"""CPU-only exact authority, core/runtime/Q1 bridge and all-20 input binding."""
import argparse
import copy
import json
import platform
import shutil
import subprocess
from pathlib import Path
from transformers import AutoTokenizer
from project.run_scripts.jlz_writer_coupled.prepare import runtime_binding, validate_schedule, bind_w0, asset_binding
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from .common import *
from . import baselines
from .schedule import validate_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='preparation-r2')
    args = parser.parse_args()
    out = LOCAL/args.name
    require(out.parent == LOCAL and out.name.startswith('preparation-'), 'TASK_PREPARATION_PATH')
    require(not out.exists(), 'CREATE_ONCE_PREPARATION')
    require(sha(ROOT/ENVELOPE) == '6d2e2837ae08797c6fd44a13ad0a637ecb46e8a6cab85bb03f976de675b153a4', 'ENVELOPE_SHA')
    envelope = json.loads((ROOT/ENVELOPE).read_text())
    contract = json.loads((ROOT/CONTRACT).read_text())
    require(contract == envelope['approved_execution'], 'ENVELOPE_CONTRACT')
    proof = []
    for row in envelope['bindings']:
        b = (ROOT/row['path']).read_bytes(); b.decode('utf-8')
        require(len(b) == row['bytes'] and sha(ROOT/row['path']) == row['sha256'], 'AUTHORITY_BINDING')
        proof.append(row)
    lock = json.loads((PRIOR/'execution.lock.json').read_text())
    require(sha(PRIOR/'execution.lock.json') == contract['frozen_science']['lock_sha256'], 'PRIOR_LOCK')
    require(lock['source_commit'] == CORE, 'PRIOR_SOURCE')
    prior = json.loads((PRIOR/'config.json').read_text())
    require(sha(PRIOR/'config.json') == lock['config_sha256'], 'PRIOR_CONFIG')
    evidence = [member(PRIOR/'execution.lock.json'), member(PRIOR/'config.json')]
    source_rows = []
    for old in lock['source_members']:
        relative = Path(old['path']).relative_to(PRIOR/'source')
        if str(relative).startswith(('project/run_scripts/jlz_realization/', 'project/run_scripts/jlz_writer_coupled/',
                                     'project/run_scripts/jlz_pilot/')):
            require(sha(ROOT/relative) == old['sha256'], 'FROZEN_CLOSURE_CHANGED:' + str(relative))
            source_rows.append(dict(path=str(relative), sha256=old['sha256'], bytes=old['bytes']))
    require(sum(x['path'].startswith('project/run_scripts/jlz_realization/') for x in source_rows) == 25, '25_CORE_MEMBERS')
    runtime = runtime_binding()
    for key in ('python','python_version','torch','transformers','torch_cuda','attention','model_dtype','geometry_dtype','matmul_TF32','cudnn_TF32','autocast'):
        require(runtime[key] == prior['runtime'][key], 'RUNTIME_CHANGED:' + key)
    require({r['module']:r['sha256'] for r in runtime['source_members']} ==
            {r['module']:r['sha256'] for r in prior['runtime']['source_members']}, 'RUNTIME_SOURCE_CHANGED')
    require(runtime['executable']['sha256'] == prior['runtime']['executable']['sha256'], 'PYTHON_CHANGED')
    for row in prior['native_reference']:
        verify(row)
    assets = [asset_binding(r['path'], {x['path']:x for x in prior['assets']}, member(PRIOR/'config.json')) for r in prior['assets']]
    require([(r['path'],r['bytes'],r['sha256']) for r in assets] ==
            [(r['path'],r['bytes'],r['sha256']) for r in prior['assets']], 'ASSET_IDENTITY_CHANGED')
    import_hashes = {}
    for arm in ('A','B'):
        imports_path = PRIOR/('main-'+arm)/'actual-imports.json'
        imports = json.loads(imports_path.read_text())
        evidence.extend([member(imports_path), member(PRIOR/('main-'+arm)/'terminal.json')])
        for name,row in imports.items():
            path = Path(row['path'])
            now = ROOT/path.relative_to(PRIOR/'source') if path.is_relative_to(PRIOR/'source') else path
            require(sha(now) == row['sha256'], 'ACTUAL_IMPORT_CHANGED:' + name)
            import_hashes[name] = row['sha256']
        terminal = json.loads((PRIOR/('main-'+arm)/'terminal.json').read_text())
        require(terminal['status'] == 'COMPLETED' and terminal['main_commits'] == 5, 'PRIOR_500_COMPLETE')
    q1 = lock['Q1_reuse']
    for row in [q1['bridge'], *q1['READY'].values()]:
        verify(row); evidence.append(row)
    for arm in ('A','B'):
        ready = json.loads(Path(q1['READY'][arm]['path']).read_text())
        require(ready['status'] == 'STRUCTURAL_READY' and ready['source'] == q1['source_commit']
                and ready['config_sha256'] == q1['config_sha256'], 'Q1_READY')
    write(out/'qualification-reuse.json', dict(status='REUSE_VERIFIED_CORE_RUNTIME_INPUT', instruction=INSTRUCTION,
        frozen_core=CORE, source_members=source_rows, evidence=evidence, Q1_reuse=q1,
        scope='기존 Q1 actual 및 완료500 core/runtime. 새로운20batch controller CPU검산은 별도.',
        prior_Q1_B_Slurm='FAILED_0:11_AFTER_PROGRAM_COMPLETE; NOT_ESTABLISHED', new_GPU_qualification=False))
    data = json.loads(Path(prior['stream']).read_text())
    schedule, cases = validate_schedule(data, ROOT/CASE, ROOT/EVALUATION)
    require(cases['ordered_case_ids_sha256'] == contract['inputs']['ordered_case_ids_sha256'], '2000_ORDER')
    require(digest([r['case_id'] for r in data[:500]]) == contract['inputs']['first500_ids_sha256'], '500_PREFIX')
    tokenizer = AutoTokenizer.from_pretrained(prior['model'], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = 'right'
    bench = CounterFactAdapter(tokenizer, json.loads(Path(prior['contexts']).read_text()))
    packing = []
    for phase,start,B,count in [('pilot',2000,2,2),('main',0,100,20)]:
        for n in range(count):
            pack = bench.prepare(data[start+n*B:start+(n+1)*B])
            packing.append(dict(phase=phase,batch=n+1,actual_B=B,ids=pack['record_ids'],identity=pack['identity'],
                rows=len(pack['row_request']),valid_tokens=int(pack['tokens']['attention_mask'].sum()),width=pack['tokens']['input_ids'].shape[1]))
    for old in prior['packing']:
        require(old == next(x for x in packing if x['phase']==old['phase'] and x['batch']==old['batch']), 'REUSED_PACKING_CHANGED')
    source_w0 = Path('/data/janghj/ODE-edit/local/jlz-native-joint-v4/20261002-compute-r1/user-remove-w0-r1/W0-reuse.json')
    w0 = bind_w0(source_w0, assets, data, bench, runtime)
    w0.update(instruction_id=INSTRUCTION, comparison_scope='FIRST2000 prompt/target/token binding; historical microbatch/package-byte equivalence NOT_ESTABLISHED')
    write(out/'W0-reuse.json', w0)
    observer_rows=[]
    for record in data[:2000]:
        rw=record['requested_rewrite']
        for kind,prompts in bench.panels(record).items():
            for index,prompt in enumerate(prompts):
                row=dict(case_id=record['case_id'],kind=kind,prompt_index=index,
                    identity=digest([record['case_id'],kind,index,prompt,rw['target_new']['str'],rw['target_true']['str']]))
                for label in ('true','new'):
                    prompt_ids,target_ids=bench.evaluation_ids(prompt,rw['target_'+label]['str'])
                    row[label+'_token_identity']=digest([prompt_ids,target_ids])
                    row[label+'_token_count']=len(target_ids)
                observer_rows.append(row)
    require(len(observer_rows)==26000,'OBSERVER_2K_ROWS')
    write(out/'observer-identity.json',dict(rows=observer_rows,prompt_text_saved=False,endpoint_independent=True))
    baseline_rows = baselines.bind(data[:2000], bench, out/'baselines')
    free = shutil.disk_usage(LOCAL).free
    require(free >= 30*1024**3, 'RESOURCE_BLOCKED_STORAGE_30GIB')
    resources = dict(cap=2,gpu=1,cpu=8,host_mib=60416,collector_host_mib=24576,node='server4',partition='gpu',qos='lab_gpu_s4',
        wall='2-00:00:00',qualification_wall='04:00:00',collector_wall='04:00:00',reserve_bytes=30*1024**3,free_bytes=free,
        output_plan_GiB=dict(diagnostic_tensor=2.0,scalar_raw=2.0,source_logs=2.0,atomic_temporary=4.0,margin=20.0),
        host_plan_GiB=dict(model_loading=31.0,history=3.83,history_transaction=3.83,weights_rollback=1.10,
                           streamed_prior=7.66,input_cache=3.0,runtime_margin=7.0),
        GPU_plan_GiB=dict(model=30.0,prior_factors=7.66,weights=2.2,builder=12.0,solve_aux=20.0,margin=10.0),
        prior_measured_peak_host_GiB=34717248/1024**2,prior_measured_peak_GPU_GiB=48838735360/1024**3,
        estimate_each_hours=[10,18],estimate_basis='이전500 main fit+eval 약2.3–2.7h; 20fit 및5200 evaluation request occurrences/arm, 긴batch padding 여유 포함',
        wall_not_ETA=True,new_2k_peak='NOT_MEASURED')
    require(sum(resources['host_plan_GiB'].values()) <= 59, 'HOST_PLAN')
    config = copy.deepcopy(prior)
    config.update(instruction_id=INSTRUCTION,task_id=TASK,authority=AUTHORITY,assets=assets,runtime=runtime,packing=packing,
        resources=resources,experiment=contract,evaluation_schedule=dict(all_seen_at=[5,10,20],current=list(range(1,21)),no_B21=True),
        authority_binding=dict(envelope=member(ROOT/ENVELOPE),contract=member(ROOT/CONTRACT),exception=member(ROOT/EXCEPTION),members=proof),
        qualification_reuse=dict(receipt=member(out/'qualification-reuse.json'),actual_import_hashes=import_hashes),
        w0_reuse=dict(receipt=member(out/'W0-reuse.json'),new_W0_forward=0),baselines=baseline_rows,
        observer_identity=member(out/'observer-identity.json'),
        schedule_binding=cases,scope='NEW_COLD_2K_NOT_RESUME')
    config['settings']['exact_probe'] = 'REUSE_HISTORICAL_NO_NEW_PROBE'
    validate_config(config)
    write(out/'configuration.json',config)
    write(out/'full-read.json',dict(instruction=INSTRUCTION,authority=AUTHORITY,members=proof,core_members=source_rows,
        schedule=cases,worktree=str(ROOT),host=platform.node(),status='IMPLEMENTING_NOT_SUBMITTED',job_ids=[],
        core_GPU='REUSED_SCOPED_ACTUAL_EVIDENCE_NOT_NEW_PASS',new_exact_probes=0,new_baseline_fits=0))
    print(json.dumps(dict(status='CPU_INPUT_BOUND_Q1_REUSE',configuration=str(out/'configuration.json'),packing=len(packing),
        baselines=[(r['name'],r['status']) for r in baseline_rows],GPU_submitted=0)))

if __name__ == '__main__':
    main()
