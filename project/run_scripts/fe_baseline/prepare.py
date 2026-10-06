"""CPU-only exact FE source/assets/token/order and resource binding."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import torch
import transformers
from transformers import AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from . import *
from .adapter import pack,lookup_function
from .profile import profile

PRIOR=Path('/mnt/raid5/janghj/ODE-edit/local/causal-allocation-editing-b1/preparation-cpu6/configuration.json')
def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu',type=Path,required=True)
    p.add_argument('--requests',type=int,default=2000);p.add_argument('--out-name',default='preparation')
    p.add_argument('--online-receipt',type=Path);args=p.parse_args();horizon=profile(args.requests)
    require('/' not in args.out_name and args.out_name.startswith('preparation'),'PREPARATION_NAME')
    out=LOCAL/args.out_name;require(not out.exists(),'CREATE_ONCE_PREPARATION')
    cpu=json.loads(args.cpu.read_text());require(cpu['passed'],'CPU_TESTS_REQUIRED')
    source=LOCAL/'upstream';m=json.loads((ROOT/'plans/global/fe-sequential-2k/upstream-source-manifest.json').read_text())
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==UPSTREAM,'UPSTREAM_COMMIT')
    members=[];not_runtime=[]
    for row in m['members']:
        path=source/row['path']
        if path.suffix=='.pdf':not_runtime.append(row);continue
        require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'UPSTREAM_SHA:'+row['path']);members.append(member(path))
    # Every additional helper source is also identity-pinned; bundled data absent.
    extras=[member(x) for folder in ('util','locate_edit_utils','z_methods') for x in sorted((source/folder).glob('*.py')) if x not in [Path(r['path']) for r in members]]
    prior=json.loads(PRIOR.read_text());assets=[]
    for row in prior['assets']:
        st=Path(row['path']).stat();require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'PRIOR_ASSET_CHANGED')
        assets.append(row|{'verification':'PRIOR_FULL_SHA_PLUS_CURRENT_SIZE_INODE_MTIME'})
    require(torch.__version__==prior['runtime']['torch'] and transformers.__version__==prior['runtime']['transformers'],'RUNTIME_IDENTITY')
    for row in prior['runtime']['source_members']:verify(row)
    records=load_prefix(Path(prior['stream']).parent,args.requests)
    tok=AutoTokenizer.from_pretrained(prior['model'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
    contexts=json.loads(Path(prior['contexts']).read_text());lookup=lookup_function(source);bench=CounterFactAdapter(tok,contexts)
    specs=[];obs=[]
    for index,r in enumerate(records):
        x=pack(tok,r,contexts,lookup)
        specs.append(dict(index=index,case_id=r['case_id'],record=digest(r),pack=x['identity'],width=x['tokens']['input_ids'].shape[1],
            valid_tokens=int(x['tokens']['attention_mask'].sum()),padded_tokens=x['tokens']['input_ids'].numel(),target_tokens=len(x['target'])))
        require(len(r['paraphrase_prompts'])==2 and len(r['neighborhood_prompts'])==10,'EVALUATOR_COUNTS')
        for kind,prompts in bench.panels(r).items():
            for i,prompt in enumerate(prompts):
                rw=r['requested_rewrite'];obs.append(dict(case_id=r['case_id'],kind=kind,prompt_index=i,
                    identity=digest([r['case_id'],kind,i,prompt,rw['target_new']['str'],rw['target_true']['str']]),
                    new_token_identity=digest(bench.evaluation_ids(prompt,rw['target_new']['str'])),
                    true_token_identity=digest(bench.evaluation_ids(prompt,rw['target_true']['str']))))
    require(len(specs)==args.requests and len(obs)==13*args.requests and len({r['case_id'] for r in records})==args.requests,'WHOLE_PREFIX')
    authority=[ROOT/'messages/head/2026-10-06-fe-sequential-2k-sh2.json',*sorted((ROOT/'plans/global/fe-sequential-2k').glob('*')),
        ROOT/'control/experiment-exceptions/fe-sequential-2k-sh2-20261006.json']
    require(sha(authority[0])=='e9934c992ae6751c9a275cc1c78a3bd2f197e5b4991eb25688ed8d5a9e78e9bd','AUTHORITY_SHA')
    authority += [ROOT/'control/wandb-policy.json',ROOT/'messages/head/2026-10-06-fe-wandb-login-resume-sh2.json']
    if args.requests==10000:authority.append(ROOT/'plans/updates/server2/fe-sequential-2k/user-10k.md')
    node=subprocess.check_output(['scontrol','show','node','server2'],text=True)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True)
    reserve=(32 if args.requests==10000 else 12)*1024**3
    capacity=shutil.disk_usage(LOCAL);require(capacity.free>reserve,'DISK_RESERVE')
    out.mkdir();write(out/'token-order.json',specs);write(out/'observer-identity.json',obs)
    c=dict(instruction_id=NONCE,task_id=horizon['task_id'],profile=horizon,upstream=str(source),upstream_members=members+extras,upstream_commit=UPSTREAM,
        model=prior['model'],stream=prior['stream'],contexts=prior['contexts'],stats=prior['stats'],seed=0,
        runtime=prior['runtime'],assets=assets,asset_reuse=member(PRIOR),cold_W0_H0=prior['cold_W0_H0'],
        authority_members=[member(x) for x in authority],cpu_preflight=member(args.cpu),specs=member(out/'token-order.json'),
        observer_identity=member(out/'observer-identity.json'),record_root=digest([digest(r) for r in records]),
        settings=dict(requests=args.requests,B=100,batches=horizon['batches'],milestones=horizon['milestones'],layers=list(LAYERS),seed=0,fit_evaluations=35,Adam_max=34,
            lr=.1,KL=.0625,norm=.5,clamp=.75,early_stop_total=.05,coefficient=15000,L2=0,physical_MB=1,
            observer_microbatch=2,save_checkpoints=False,exact_resume='NOT_AVAILABLE'),
        resources=dict(cpu=6,gpu=1,host_mib=59392,hard_host_mib=60416,wall='48:00:00',collector_cpu=6,
            collector_host_mib=24576,collector_wall='04:00:00',task_cap=1,project_cap=2,reserve_bytes=reserve,
            free_bytes=capacity.free,free_inodes=os.statvfs(LOCAL).f_favail,node='server2',node_snapshot=node,GPU_snapshot=gpu,
            target_table_bytes=horizon['target_table_bytes'],H_bytes=5*14336**2*4,C0_bytes=5*14336**2*4,
            planned_eval_rows=horizon['planned_eval_rows'],new_eval_rows=horizon['new_eval_rows'],planned_fit_logical_evaluations_max=35*args.requests,
            host_plan_GiB=dict(model_load=40,steady_C0_H_target=7.813,batch_W_H_rollback=4.922,solve_scratch=5,python=8),
            GPU_plan_GiB=dict(model=30,one_layer_solve_peak_scratch=10,MB1_recompute='actual peak not measured'),
            ETA='NOT_MEASURED;48h request ceiling; include checkpoint recompute physical forwards'),
        W0_reuse='NOT_AVAILABLE_EXACT_LOCAL_BRIDGE; fresh once',no_other_baselines=True)
    if args.online_receipt:
        online=json.loads(args.online_receipt.read_text());require(online['status']=='READY_ONLINE_VERIFIED','ONLINE_SMOKE')
        c['tracking']=dict(integration='SH1_SHARED_HELPER_BOUND',online_receipt=member(args.online_receipt),
            env_file='/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env',
            env_member=member(Path('/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env')),
            helper_members=[member(p) for p in sorted((ROOT/'project/run_scripts/experiment_tracking').glob('*.py'))],
            phase_ids=dict(start=0,W0=1,target_fit=2,replay=3,write=4,evaluation=5,terminal=6),
            cohort_ids=dict(R=1,P=2,N=3),credential_recorded=False)
    write(out/'configuration.json',c)
    write(out/'source-read.json',dict(authority=c['authority_members'],upstream=members,PDF_not_runtime=not_runtime,
        prior_PROTOCOL='identical to previously FULL_READ bbe0fffa PROTOCOL',actual_model='NOT_RUN',independent_reviewer=0))
    print(json.dumps(dict(config=str(out/'configuration.json'),rows=len(obs),targets=len(specs),max_width=max(x['width'] for x in specs),new_GPU=0)))

if __name__=='__main__':main()
