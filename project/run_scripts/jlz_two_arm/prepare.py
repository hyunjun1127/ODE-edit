"""CPU-only exact binding, source closure and approved tiny baseline pull."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from .common import ROOT, LOCAL, CONTRACT, INSTRUCTION, PYTHON, member, sha, write, require, digest
from .references import verified_records, filter_general_pool, select_general, select_replay

def pull_primary():
    source='/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/attempt-repair-r1'
    dest=LOCAL/'inputs/memit-h-baseline'
    names=['execution.lock.json','submission.json','release.json','output/runtime.json',
           'output/actual-import-closure.json','output/B020/entry.json','output/B020/commit.json',
           'output/B020/native-observation.json','output/B020/seen-full.json']
    # Exact regular files only; no log/live trajectory/large payload discovery.
    script="import pathlib,hashlib,json,stat; root=pathlib.Path("+repr(source)+"); names="+repr(names)+"; result=[]\nfor n in names:\n p=root/n; s=p.lstat(); assert stat.S_ISREG(s.st_mode); result.append(dict(relative=n,path=str(p),bytes=s.st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))\nprint(json.dumps(result))"
    import shlex
    raw=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','rke-server3','python3 -c '+shlex.quote(script)],text=True)
    allow=json.loads(raw)
    write(LOCAL/'inputs/memit-h-allowlist.json',dict(source_keep=True,sole_receiver='SH4',files=allow))
    for row in allow:
        dst=dest/row['relative'];dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():require(dst.stat().st_size==row['bytes'] and sha(dst)==row['sha256'],'EXISTING_BASELINE_CONFLICT')
        else:
            partial=dst.with_name(dst.name+'.partial');require(not partial.exists(),'EXISTING_PARTIAL')
            subprocess.run(['scp','-q','-o','BatchMode=yes','rke-server3:'+row['path'],str(partial)],check=True)
            require(partial.stat().st_size==row['bytes'] and sha(partial)==row['sha256'],'RECEIVER_BASELINE_SHA')
            # No overwrite; atomic create-once link then remove only our verified partial.
            os.link(partial,dst);partial.unlink()
    write(dest/'receiver-receipt.json',dict(files=[member(dest/n) for n in names],source_keep=True,new_gpu=0))
    return dest

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=LOCAL/'preparation-v1');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    contract=json.loads(CONTRACT.read_text());manifest=json.loads((ROOT/'audits/global/2026-10-02-jlz-twoarm-sh4-dispatch/input-manifest.json').read_text())
    for row in manifest['members']:
        path=ROOT/row['path'];require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'AUTHORITY_BYTES')
    fullpath=Path('/data/janghj/EasyEdit/data/counterfact/counterfact.json')
    stream=Path('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json')
    context=Path('/data/janghj/ODE-edit/local/fixed10k-native-baselines/attempt-v1/output/main-cell-1/B001/contexts.json')
    full,fi=verified_records(fullpath,contract['references']['general_pool_sha256'])
    fixed,si=verified_records(stream,contract['references']['stream_sha256'])
    require(sha(context)=='33cec0eef9ec130f26c2f0e17f7c8be39e93c717ebb47eaa5e9f1c88b263524e','CONTEXT_IDENTITY')
    pool,counts=filter_general_pool(full,fixed)
    require(list(counts.values())==[11919,11544,11134,10715,34],'REFERENCE_POOL_COUNTS')
    schedules={}
    for phase,bs,n in [('pilot',4,2),('main',100,20)]:
        schedules[phase]=[]
        for i in range(n):
            current=fixed[bs*i:bs*(i+1)];past=fixed[:bs*i]
            g=select_general(pool,current,contract['references'],i);e=select_replay(past,current,contract['references'],i)
            schedules[phase].append(dict(batch=i+1,ids=[r['case_id'] for r in current],general=[r['case_id'] for r in g],replay=[r['case_id'] for r in e]))
    model=Path('/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2')
    stats=Path('/data/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats')
    inputs=[member(stream),member(fullpath),member(context)]
    import numpy as np
    stat_info=[]
    for l in range(4,9):
        file=stats/f'model.layers.{l}.mlp.down_proj_float32_mom2_100000.npz'
        with np.load(file,allow_pickle=False) as z:
            x=z['mom2.mom2'];c=int(z['mom2.count'])
            require(c>0 and x.dtype==np.float32 and x.shape==(14336,14336) and np.isfinite(x).all(),'C0_SCHEMA')
        stat_info.append(dict(layer=l,count=c));inputs.append(member(file));del x
    index=json.loads((model/'model.safetensors.index.json').read_text())
    for name in sorted(set(index['weight_map'].values())|{p.name for p in model.glob('*.json')}):inputs.append(member(model/name))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True);tok.padding_side='right';tok.pad_token=tok.eos_token
    from project.run_scripts.jlz_pilot.prompts import prepare
    pack=[]
    for phase,bs in [('pilot',4),('main',100)]:
        for cell in schedules[phase]:
            i=cell['batch']-1;records=fixed[bs*i:bs*(i+1)]
            spec=prepare(tok,[r['requested_rewrite']|{'case_id':r['case_id']} for r in records],json.loads(context.read_text()),'cpu')
            require(len(spec['row_request'])==7*bs,'CURRENT_ROWS')
            pack.append(dict(phase=phase,batch=i+1,identity=spec['identity'],rows=7*bs,width=int(spec['tokens']['input_ids'].shape[1]),valid_tokens=int(spec['tokens']['attention_mask'].sum())))
    primary=pull_primary()
    baselines=[]
    for name,cell in [('BASE_ALPHAEDIT',1),('BASE_MEMIT',2)]:
        base=Path('/data/janghj/ODE-edit/local/fixed10k-native-baselines/attempt-v1')
        raw=base/f'output/main-cell-{cell}/B020/seen-full.json'
        data=json.loads(raw.read_text());require(data['requests']==2000,'BASELINE_W20_2K')
        baselines.append(dict(name=name,mode='REUSE',raw_path=str(raw),receipt=member(raw),
                              commit=member(base/f'output/main-cell-{cell}/B020/commit.json'),
                              comparison='quality only; historical TF4.44.2/cudnnTF32true vs JLZ4.57.1/TF32off, not matched timing'))
    baselines.append(dict(name='MEMIT-H',mode='REUSE',raw_path=str(primary/'output/B020/seen-full.json'),
                          receipt=member(primary/'output/B020/seen-full.json'),
                          comparison='quality only; S3 H200/TF4.44.2 vs S4 Blackwell/TF4.57.1; no speed ratio'))
    free=shutil.disk_usage(LOCAL).free;require(free>=20*1024**3,'STORAGE_RESERVE')
    config=dict(instruction_id=INSTRUCTION,contract=json.loads(CONTRACT.read_text()),authority_members=manifest['members'],
                model=str(model),stats=str(stats),stream=str(stream),general=str(fullpath),contexts=str(context),inputs=inputs,
                stats_info=stat_info,reference_pool=counts,schedules=schedules,packing=pack,baselines=baselines,
                baseline_pilot=dict(native_root='/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/blue-source',
                  stats_root='/data/janghj/EasyEdit/examples/data/stats',projector='/data/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt',projector_source_layers=[4,5,6,7,8],
                  configs={'BASE_ALPHAEDIT':'/data/janghj/ODE-edit/local/fixed10k-native-baselines/attempt-v1/configs/AlphaEdit-native.json',
                           'BASE_MEMIT':'/data/janghj/ODE-edit/local/fixed10k-native-baselines/attempt-v1/configs/MEMIT-native.json',
                           'MEMIT-H':str(ROOT/'project/run_scripts/memit_history_lifelong/hparams.json')}),
                numerical_fidelity_policy='RECORD_ONLY_USER_DIRECTED',numerical_certification='NOT_ESTABLISHED',
                settings=dict(microbatch=4,observer_microbatch=2,seed=20261002,cross_epsilon=1e-12,
                              route_initial='dense',route_fixed_candidate_options=['original','dense','direct'],shared_technical_cap=6),
                resources=dict(cap=2,mem_mib=60416,cpu=8,gpu=1,wall='7-00:00:00',wall_not_ETA=True,
                  host_peak_estimate_gib=48,gpu_peak_estimate_gib=80,storage_reserve=20*1024**3,free_bytes=free,
                  accounting='model GPU29.915GiB; CPU H+rollback7.66GiB, W copies<=4GiB, one C0+FP64 scratch<=6GiB, teacher<4GiB, model streamed load; measured peak pending'),
                checkpoint_saved=False,exact_resume='NOT_AVAILABLE',broadcast='NO_BROADCAST_NOT_REQUIRED')
    write(a.out/'configuration.json',config)
    write(a.out/'full-read.json',dict(instruction=INSTRUCTION,authority_commit='7dcc580c1ac784844eeab2737356674b00898c22',
          current_publication=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),members=manifest['members'],
          envelope=member(ROOT/'messages/head/2026-10-02-jlz-twoarm-bs100x20-sh4.json'),
          both_lanes=member(ROOT/'messages/head/2026-10-02-jlz-twoarm-bs100x20-sh4-use-both-gpus.json'),
          protocol=member(ROOT/'PROTOCOL.md'),read_level='OWNER_FULL_READ_EXACT_SHA',gpu_validation='NOT_RUN',status='IMPLEMENTING_NOT_SUBMITTED',job_ids=[]))
    print(json.dumps({'configuration':str(a.out/'configuration.json'),'inputs':len(inputs),'pool':counts,'baselines':[(x['name'],x['mode']) for x in baselines]}))

if __name__=='__main__':main()
