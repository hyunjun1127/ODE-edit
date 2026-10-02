"""CPU identity binding, v9 native writer and source, scope and resource freeze."""
import json
import platform
import shutil
from pathlib import Path
from transformers import AutoTokenizer
from project.run_scripts.jlz_writer_coupled.prepare import asset_binding,runtime_binding,validate_schedule,bind_w0
from .common import ROOT,LOCAL,INSTRUCTION,TASK,require,member,sha,write,digest
from .inputs import CounterFactAdapter

AUTHORITY='e147e55139d9a9c1a568a812020633b579c0b767'
ENVELOPE='messages/head/2026-10-03-jlz-v9-ridge-500-exact-pilot-sh4.json'
DESIGN='plans/global/2026-10-03-jlz-realization-v9'
EXCEPTION='control/experiment-exceptions/jlz-realization-v9-ridge-500-20261003.json'

def main():
    out=LOCAL/'preparation-v1';require(not out.exists(),'CREATE_ONCE_PREPARATION')
    authority=json.loads((ROOT/ENVELOPE).read_text());proof=[]
    for row in authority['binding']['members']:
        p=ROOT/row['path'];b=p.read_bytes();b.decode('utf-8')
        require(len(b)==row['bytes'] and sha(p)==row['sha256'],'DESIGN_SHA:'+row['path'])
        proof.append(dict(row,verification='FULL_BYTES_SHA_SIZE'))
    old=Path('/data/janghj/ODE-edit/local/jlz-writer-coupled-v5/20261002-v1/attempt-r1/config.json')
    require(sha(old)=='0e5016e6751bce0802838fc668d2d53b6d356fa878f0a54a74331b2477981403','PRIOR_CONFIG')
    previous=json.loads(old.read_text());oldreceipt=member(old)
    oldassets={r['path']:r for r in previous['assets']}
    assets=[asset_binding(r.get('requested_path',r['path']),oldassets,oldreceipt) for r in previous['assets']]
    data=json.loads(Path(previous['stream']).read_text());_,cases=validate_schedule(data)
    require(digest([r['case_id'] for r in data[:500]])==authority['execution']['main']['first500_case_ids_sha256'],'FIRST500')
    tokenizer=AutoTokenizer.from_pretrained(previous['model'],local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    bench=CounterFactAdapter(tokenizer,json.loads(Path(previous['contexts']).read_text()))
    packing=[]
    for phase,start,B,count in [('pilot',2000,2,2),('main',0,100,5)]:
        for number in range(count):
            pack=bench.prepare(data[start+number*B:start+(number+1)*B])
            packing.append(dict(phase=phase,batch=number+1,actual_B=B,ids=pack['record_ids'],identity=pack['identity'],
                rows=len(pack['row_request']),valid_tokens=int(pack['tokens']['attention_mask'].sum()),width=pack['tokens']['input_ids'].shape[1]))
    native_blue=[member(Path('/data/janghj/BLUE')/p) for p in
        ('memit/memit_seq_main.py','memit/compute_ks.py','memit/compute_z.py','memit/memit_hparams.py','rome/layer_stats.py','util/runningstats.py','rome/repr_tools.py')]
    import subprocess
    require(subprocess.check_output(['git','-C','/data/janghj/BLUE','rev-parse','HEAD'],text=True).strip()=='311b076a92e4ed0f14f5c8b4909732da781bc5f7','NATIVE_BLUE_REVISION')
    runtime=runtime_binding()
    bridge=Path('/data/janghj/ODE-edit/local/jlz-native-joint-v4/20261002-compute-r1/user-remove-w0-r1/W0-reuse.json')
    w0=bind_w0(bridge,assets,data,bench,runtime)
    w0['instruction_id']=INSTRUCTION;w0['comparison_scope']='first500_only; historical layout is not bitwise qualified'
    raw=json.loads(Path(w0['observations']['path']).read_text());ids={r['case_id'] for r in data[:500]}
    from .observe import reduce_rows
    selected=[r for r in raw['rows'] if r['case_id'] in ids]
    w0['first500_summary']=reduce_rows(selected)
    require(w0['first500_summary']['N']['numerator']==4392,'HISTORICAL_FIRST500_N')
    write(out/'W0-reuse.json',w0)
    free=shutil.disk_usage(LOCAL).free;require(free>=30*1024**3,'OUTPUT_RESERVE_30GIB')
    resources=dict(cap=2,gpu=1,cpu=8,host_mib=60416,collector_host_mib=24576,
        wall='7-00:00:00',qualification_wall='24:00:00',collector_wall='04:00:00',
        node='server4',partition='gpu',qos='lab_gpu_s4',reserve_bytes=30*1024**3,
        wall_not_ETA=True,free_bytes=free,
        host_plan_gib=dict(history=3.83,transaction_history=3.83,Q2_transaction_history=3.83,weight_rollback=3.3,same_A_CPU_prior=7.66,
            prior_loading_peak=3.1,entry_cache_and_teacher=3.,python_runtime_loader_margin=28.),
        GPU_plan_gib=dict(model=30.,prior_factors=7.66,weights=2.2,
            builder_checkpoints=12.,solve_and_native_aux_scratch=20.,safety=10.),
        peak_measured=False,estimated_seconds='NOT_ESTABLISHED; Q2 equals main_A_B1, no extra B100 fit')
    config=dict(instruction_id=INSTRUCTION,task_id=TASK,authority=AUTHORITY,
        authority_binding=dict(envelope=member(ROOT/ENVELOPE),exception=member(ROOT/EXCEPTION),members=proof),
        model=previous['model'],stream=previous['stream'],contexts=previous['contexts'],stats=previous['stats'],
        assets=assets,stats_schema=previous['stats_schema'],native_reference=previous['native_reference']+native_blue,
        runtime=runtime,packing=packing,schedule_binding=cases,resources=resources,
        profile=dict(previous['profile']),settings=dict(seed=20261002,fit_microbatch=2,observer_microbatch=2,
            candidate_cap=25,backward_cap=24,save_checkpoints=False,
            route='direct',builder_crop=False,checkpoint=True,learning_rate=.1,warmup_updates=0),
        experiment=authority['execution'],qualification=authority['review'],
        w0_reuse=dict(receipt=member(out/'W0-reuse.json'),new_W0_forward=0),
        baseline_new_runs=0,exact_resume='NOT_AVAILABLE',broadcast='NO_BROADCAST_NOT_REQUIRED')
    config['profile']={k:v for k,v in config['profile'].items() if k not in ('preservation_K','preservation_E')}
    config['settings'].update(replay=False,q_scale=True,lambda_allocation=.1,exact_probe='MAIN_A_B1_SAME_FIT')
    config['native_BLUE']=dict(commit='311b076a92e4ed0f14f5c8b4909732da781bc5f7',members=native_blue,
        adoption='mean-key FP32 nested reduction, CPU FP32 H, same-A ridge only; native z and residual divisor NOT adopted')
    config['evaluation_schedule']=dict(W0='REUSE',current=list(range(1,5)),allseen=5,
        denominators_current=dict(R=100,P=200,N=1000),denominators_W5=dict(R=500,P=1000,N=5000),
        exact='only all five causal layers qualified; R100/P200/N1000; no feedback',no_B6=True)
    config['resources']['host_plan_total_GiB']=sum(config['resources']['host_plan_gib'].values())
    require(config['resources']['host_plan_total_GiB']<59,'HOST_PLAN_59GiB')
    write(out/'configuration.json',config)
    write(out/'full-read.json',dict(instruction_id=INSTRUCTION,members=proof,case_csv=cases,archive_members_verified=27,
        owner='SH4',host=platform.node(),worktree=str(ROOT),status='IMPLEMENTING_NOT_SUBMITTED',job_ids=[],
        CPU_design_tests='GH archived v9 181/v8 404; no actual GPU claim'))
    print(json.dumps(dict(status='CPU_INPUT_BOUND_GPU_NOT_RUN',configuration=str(out/'configuration.json'),assets=len(assets),packing=len(packing))))

if __name__=='__main__':main()
