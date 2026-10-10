"""Two never-submitted Qwen author FE conditions moved to SH1; no old mutation."""
import argparse,json,os,subprocess,sys,shlex,re,getpass
from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from official.runners.fe_original import config_native,validate,AUTHOR,INSTRUCTION
from official.runners.fe_original_compat import check
from official.runners.server1.common import member,verify,read
from official.runners.server1.assets import member as asset_member
from official.experiments.prepare import write_new,digest
from official.evaluation.zsre_query_parity import compare_queries
from . import submit as control

ROOT=Path('/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011')
MOVE=ROOT/'qwen-server1-move-r1'
NONCE='USER-FE-QWEN-SERVER1-MOVE-20261011-R1'
PYTHON=str(ROOT/'venv/bin/python')

def append_graph(existing):
    tails=control.frontier(existing['jobs'])
    legacy=existing['allocated_gpus']>2 or existing['admitted_DAG_width']>2
    graph=({k:tails for k in ('qwen25-cf','qwen25-zsre')} if legacy else
           {'qwen25-cf':tails[:1],'qwen25-zsre':tails[1:]})
    combined=existing['jobs']+[dict(key=k,gpus=1,parents=v) for k,v in graph.items()]
    assert control.graph_width(combined)<=max(2,existing['admitted_DAG_width'])
    if legacy:assert all(set(v)==set(tails) for v in graph.values())
    return graph,legacy,control.graph_width(combined)

def storage():
    # BF16 selected weights + native FP32 H, six independent latest + ONE host-locked tmp.
    sizes={}
    for model in ('llama3','gptj','qwen25'):
        cfg=config_native(ROOT/'author',model,'cf')
        old=read(Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1/memit-fe-history-three-model-2k/preparation-r1/assets')/(model+'.json'))
        ident=old['model']['identity'];h=ident['hidden'];d=ident['intermediate'];L=len(cfg.llms.layers)
        sizes[model]=L*(h*d*2+d*d*4)+32*1024**2
    latest=2*sum(sizes.values());tmp=max(sizes.values());z=2*1024**3;raw=16*1024**3;reserve=64*1024**3
    required=latest+tmp+z+raw+reserve;st=os.statvfs(ROOT)
    result=dict(latest_estimates=sizes,six_latest_bytes=latest,one_tmp_bytes=tmp,z_bytes=z,raw_bytes=raw,reserve_bytes=reserve,
        required_free_bytes=required,available_bytes=st.f_bavail*st.f_frsize,inodes=st.f_favail,lock=str(ROOT/'checkpoint-serialization.lock'))
    assert result['available_bytes']>=required and result['inodes']>10000,'RESOURCE_BLOCKED_STORAGE_KEEP_SOURCE'
    return result

def prepare():
    assert not (MOVE/'preparation/READY.json').exists(),'EXISTING_PREPARATION_RECONCILE'
    check(ROOT/'author');MOVE.mkdir(parents=True,exist_ok=True)
    # Read only small source-side manifests. No payload, credential, model or C0 transfer.
    program="""import json,pathlib
r=pathlib.Path('/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/source-ready-r1')
cs=[json.loads((r/'configs'/('qwen25-'+d+'.json')).read_text()) for d in ('cf','zsre')]
a=json.loads(pathlib.Path(cs[0]['assets']['path']).read_text())
print(json.dumps(dict(configs=cs,assets=a)))
"""
    result=subprocess.run(['ssh','-F','/mnt/raid5/janghj/ODE-edit/servers/local/ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=8','rke-server2','python3','-'],input=program,text=True,capture_output=True,timeout=30,check=True)
    remote=json.loads(result.stdout);write_new(MOVE/'preparation/server2-binding.json',remote)
    old=read('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1/memit-fe-history-three-model-2k/preparation-r1/configs/qwen25.json')
    assets=read(verify(old['assets']))
    assert assets['model']['identity']==remote['assets']['model']['identity']
    def hashes(a):return sorted((Path(m['path']).name,m['bytes'],m['sha256']) for m in a['model']['members'])
    assert hashes(assets)==hashes(remote['assets']),'MODEL_TOKENIZER_SOURCE_MISMATCH'
    for m in assets['model']['members']:asset_member(m['path'],allow_symlink=True,expected_sha=m['sha256'],expected_bytes=m['bytes'])
    author_members=[member(ROOT/'author'/p) for p in subprocess.check_output(['git','-C',str(ROOT/'author'),'ls-files'],text=True).splitlines() if (ROOT/'author'/p).is_file()]
    zsre=read(ROOT/'preparation-r2/configs/llama3-zsre.json')['stream']
    configs=[]
    for dataset,rc in zip(('cf','zsre'),remote['configs']):
        cfg=config_native(ROOT/'author','qwen25',dataset);stream=old['stream_member'] if dataset=='cf' else zsre
        records=read(verify(stream));records=records['records'] if isinstance(records,dict) else records
        assert len(records)==2000 and digest(records)==rc['stream_sha256'] and stream['sha256']==rc['stream']['sha256'],'STREAM_MISMATCH'
        C0={str(l):assets['C0'][str(l)]['member'] for l in cfg.llms.layers}
        for l,m in C0.items():
            assert m['sha256']==rc['C0'][l]['sha256'] and m['bytes']==rc['C0'][l]['bytes'];verify(m)
        proof=None
        if dataset=='zsre':
            tok=AutoTokenizer.from_pretrained(assets['model']['snapshot'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
            p=compare_queries(tok,records,model_family='qwen25');assert p['status']=='PASS_CPU_QUERY_ONLY'
            write_new(MOVE/'preparation/zsre-query-proof.json',p);proof=member(MOVE/'preparation/zsre-query-proof.json')
        c=dict(rc,server='server1',attempt='server1-move-r1',author_path=str(ROOT/'author'),author_members=author_members,
            assets=old['assets'],stream=stream,C0=C0,output=str(MOVE/'runs'/('qwen25-'+dataset)),storage_min_free_bytes=64*1024**3,
            checkpoint_lock=str(ROOT/'checkpoint-serialization.lock'),tracking_env_file=old['tracking_env_file'],runtime_manifest=member(ROOT/'author-lock.json'),query_proof=proof)
        assert c['llms']==OmegaConf.to_container(cfg.llms,resolve=True)
        c['config_sha256']=digest({k:v for k,v in c.items() if k!='config_sha256'});validate(c)
        path=MOVE/'preparation/configs'/('qwen25-'+dataset+'.json');write_new(path,c);configs.append(member(path))
    (MOVE/'runs').mkdir(exist_ok=True)
    write_new(MOVE/'preparation/READY.json',dict(nonce=NONCE,configs=configs,storage=storage(),server2_job_ids=[],GPU=False,server2_evidence=member(Path('audits/servers/server2/fe-original-w0-2k-20261011/source-ready-r1.json'))))
    print('QWEN_LOCAL_ASSETS_STREAMS_QUERY_READY_NO_GPU')

def register():
    prep=read(MOVE/'preparation/READY.json');attempt=MOVE/'registration-r1'
    assert not attempt.exists() and not list(MOVE.rglob('submitted-*.json')),'EXISTING_SUBMISSION_RECONCILE'
    source=control.command(['git','rev-parse','HEAD']);tree=control.command(['git','rev-parse','HEAD:official'])
    control.command(['git','merge-base','--is-ancestor',source,'origin/main']);assert not control.command(['git','status','--porcelain','--','official'])
    for m in prep['configs']:validate(read(verify(m)))
    existing=control.inventory()
    assert not any('qwen25' in r['original'] and 'fe-original' in r['original'] for r in existing['jobs']),'DUPLICATE_LIVE_QWEN'
    # Do not mutate grandfathered allocations. With an overwide old DAG, both
    # new heads wait for EVERY old terminal; thereafter their width is exactly2.
    graph,legacy_overcap,width=append_graph(existing)
    capfile=Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    row=[l.split('\t') for l in capfile.read_text().splitlines() if l.startswith('server1\t')];assert len(row)==1 and int(row[0][2])==2
    resources=dict(control.DEFAULT_RESOURCES,cpus=8,memory_MiB=98304,wall_seconds=48*3600)
    assert resources['memory_MiB']<=int(row[0][3])<=183296
    plan=dict(cap=2,source=dict(main_commit=source,official_tree=tree),resources=resources)
    physical=control.resource_preflight(plan);space=storage()
    attempt.mkdir();(attempt/'scripts').mkdir();(attempt/'logs').mkdir()
    write_new(attempt/'admission.json',dict(existing=existing,graph=graph,new_execution_width=2,combined_historical_width=width,legacy_overcap=legacy_overcap,storage=space,resources=resources,physical=physical,cap=member(capfile),old_jobs_unchanged=True))
    frozen=control.freeze_source(plan,attempt)
    lock=dict(instruction=INSTRUCTION,move_authority=NONCE,source_commit=source,official_tree=tree,source_directory=frozen['directory'],source_members=frozen['members'],configs=prep['configs'],preparation=member(MOVE/'preparation/READY.json'))
    lp=attempt/'execution-lock.json';write_new(lp,lock);jobs={};held={}
    for m in prep['configs']:
        c=read(m['path']);key='qwen25-'+c['dataset'];parents=graph[key]
        argv=[PYTHON,'-B','-u','-m','official.runners.fe_original','--config',m['path'],'--lock',str(lp)]
        env=dict(PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',OFFICIAL_CODE_COMMIT=source,OFFICIAL_TREE_SHA256=tree,WANDB_CONSOLE='off',WANDB_SAVE_CODE='false')
        text='#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())+'cd '+shlex.quote(frozen['directory'])+'\nexec '+shlex.join(argv)+'\n'
        script=attempt/'scripts'/(key+'.sh')
        with script.open('x') as f:f.write(text)
        job=dict(key=key+'-fe-original',gpus=1,mode='chain',parents=[],external_resource_parents=parents)
        cmd,deps=control.sbatch_argv(plan,job,script,attempt,{})
        response=control.command(cmd);jid=response.split(';')[0];assert re.fullmatch(r'\d+',jid)
        jobs[key]=dict(job_id=jid,name='official-s1-'+job['key'],config=m,dependencies=deps,argv=argv,script=member(script),output=c['output'])
        write_new(attempt/('submitted-'+key+'.json'),dict(response=response,command=cmd,**jobs[key]))
        d=control.metadata(control.command(['scontrol','show','job',jid,'--oneliner']))
        assert d['JobState']=='PENDING' and d['Reason']=='JobHeldUser' and d['UserId'].split('(')[0]==getpass.getuser()
        assert d['Command']==str(script) and d['WorkDir']==frozen['directory'] and d['JobName']==jobs[key]['name']
        assert d['ReqNodeList']=='devbox' and d['QOS']==resources['qos'] and d['Partition']==resources['partition'] and d['Requeue']=='0'
        assert control.gpu_count(d['ReqTRES'])==1 and not control.gpu_count(d.get('AllocTRES',''))
        assert control.requested_cpu_matches(d,8) and control.memory_MiB(d['MinMemoryNode'])==98304
        assert control.seconds(d['TimeLimit'])==48*3600 and control.dependencies(d.get('Dependency',''))==sorted(deps)
        assert control.command(['scontrol','write','batch_script',jid,'-']).strip()==text.strip();held[key]=d
    fresh=control.inventory();assert fresh['admitted_DAG_width']<=max(2,existing['admitted_DAG_width'])
    assert set(fresh['frontier'])=={v['job_id'] for v in jobs.values()},'CONCURRENT_NEW_FRONTIER_RECONCILE'
    write_new(attempt/'held-inspection.json',dict(jobs=held,inventory=fresh,lock=member(lp)))
    for row in jobs.values():control.command(['scontrol','release',row['job_id']])
    states={k:control.metadata(control.command(['scontrol','show','job',v['job_id'],'--oneliner'])) for k,v in jobs.items()}
    write_new(attempt/'submission.json',dict(nonce=NONCE,source=source,tree=tree,jobs=jobs,initial=states,lock=member(lp),released=True,held_inspected=True,WB='NOT_OBSERVED_BEFORE_STARTUP'))
    print({k:dict(job_id=v['job_id'],state=states[k]['JobState'],dependencies=v['dependencies']) for k,v in jobs.items()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','register']);a=p.parse_args()
    prepare() if a.action=='prepare' else register()
