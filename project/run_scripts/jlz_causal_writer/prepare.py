"""CPU identity binding, v6 stop reconciliation, scope and resource freeze."""
import json
import platform
import shutil
from pathlib import Path
from transformers import AutoTokenizer
from project.run_scripts.jlz_writer_coupled.prepare import asset_binding,runtime_binding,validate_schedule,bind_w0
from .common import ROOT,LOCAL,INSTRUCTION,TASK,require,member,sha,write,digest
from .inputs import CounterFactAdapter

AUTHORITY='80139a375af4b62ffa4dbeedbe0f233173790eae'
ENVELOPE='messages/head/2026-10-02-jlz-v7-causal-500-sh4.json'
DESIGN='plans/global/2026-10-02-jlz-causal-writer-v7'
EXCEPTION='control/experiment-exceptions/jlz-causal-writer-v7-500-20261002.json'

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
    require(digest([r['case_id'] for r in data[:500]])==authority['execution']['main']['case_ids_sha256'],'FIRST500')
    tokenizer=AutoTokenizer.from_pretrained(previous['model'],local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    bench=CounterFactAdapter(tokenizer,json.loads(Path(previous['contexts']).read_text()))
    packing=[]
    for phase,start,B,count in [('pilot',2000,2,2),('B100',0,100,1),('main',0,100,5)]:
        for number in range(count):
            pack=bench.prepare(data[start+number*B:start+(number+1)*B])
            packing.append(dict(phase=phase,batch=number+1,actual_B=B,ids=pack['record_ids'],identity=pack['identity'],
                rows=len(pack['row_request']),valid_tokens=int(pack['tokens']['attention_mask'].sum()),width=pack['tokens']['input_ids'].shape[1]))
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
        host_plan_gib=dict(history=3.83,transaction_history=3.83,weight_rollback=1.1,
            prior_loading_peak=3.1,entry_cache_and_teacher=3.,python_runtime_loader_margin=24.),
        GPU_plan_gib=dict(model=30.,prior_factors=7.66,weights=2.2,
            builder_checkpoints=12.,solve_and_native_aux_scratch=20.,safety=10.),
        peak_measured=False,estimated_seconds='NOT_ESTABLISHED_UNTIL_V7_B100')
    config=dict(instruction_id=INSTRUCTION,task_id=TASK,authority=AUTHORITY,
        authority_binding=dict(envelope=member(ROOT/ENVELOPE),exception=member(ROOT/EXCEPTION),members=proof),
        model=previous['model'],stream=previous['stream'],contexts=previous['contexts'],stats=previous['stats'],
        assets=assets,stats_schema=previous['stats_schema'],native_reference=previous['native_reference'],
        runtime=runtime,packing=packing,schedule_binding=cases,resources=resources,
        profile=dict(previous['profile']),settings=dict(seed=20261002,fit_microbatch=2,observer_microbatch=2,
            memory_capacity=128,reference_cap=16,candidate_cap=25,backward_cap=24,save_checkpoints=False,
            route='direct',builder_crop=False,checkpoint=True,learning_rate=.1,warmup_updates=0),
        experiment=authority['execution'],qualification=authority['review'],
        w0_reuse=dict(receipt=member(out/'W0-reuse.json'),new_W0_forward=0),
        baseline_new_runs=0,exact_resume='NOT_AVAILABLE',broadcast='NO_BROADCAST_NOT_REQUIRED')
    write(out/'configuration.json',config)
    write(out/'full-read.json',dict(instruction_id=INSTRUCTION,members=proof,case_csv=cases,archive_members_verified=15,
        owner='SH4',host=platform.node(),worktree=str(ROOT),status='IMPLEMENTING_NOT_SUBMITTED',job_ids=[],
        CPU_design_tests='107 structural + 4 total-gradient; not GPU validation'))
    print(json.dumps(dict(status='CPU_INPUT_BOUND_GPU_NOT_RUN',configuration=str(out/'configuration.json'),assets=len(assets),packing=len(packing))))

if __name__=='__main__':main()
