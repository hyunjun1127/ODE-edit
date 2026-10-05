"""Server2 CPU input binding: exact small-source pulls, local heavy assets only."""
import argparse
import copy
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path
from . import ROOT, DESIGN, SCHEDULE, member, sha, digest, require, write
from .profile import B1_TASK, B1_NONCE
from .prepare import full_native_binding
from project.run_scripts.jlz_writer_coupled.prepare import runtime_binding
from scripts.fixed_counterfact import load_prefix

LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/causal-allocation-editing-b1')
REMOTE_CONFIG='/data/janghj/ODE-edit/local/causal-allocation-editing/preparation/configuration.json'
CONFIG_SHA='fd15741517cf6600f832384d9fd17ee6e26c4dac21fe16b07053bc256dea2f81'

def remote_bytes(path):
    require(path.startswith('/data/janghj/') and '..' not in Path(path).parts, 'REMOTE_ALLOWLIST_PATH')
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',
        'rke-server4','cat -- '+shlex.quote(path)],timeout=30)

def staged(path,data,expected):
    require(hashlib.sha256(data).hexdigest()==expected,'INPUT_SHA:'+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==data,'NO_OVERWRITE_INPUT')
    else:
        with path.open('xb') as f:f.write(data)
    return member(path)

def inputs():
    dest=LOCAL/'inputs';configpath=dest/'server4-configuration.json'
    data=configpath.read_bytes() if configpath.exists() else remote_bytes(REMOTE_CONFIG)
    staged(configpath,data,CONFIG_SHA);prior=json.loads(data)
    rows=list(prior['native_reference'])+list(prior['dependency_sources'])
    rows += [prior['native_input_alignment'],prior['observer_identity']]
    rows += [next(r for r in prior['assets'] if r.get('requested_path',r['path'])==prior['contexts'])]
    rows += [dict(path=prior['native_hparams'],sha256=sha(ROOT/'project/run_scripts/memit_history_lifelong/hparams.json'))]
    globals_path=prior['native_root']+'/globals.yml'
    globals_data=remote_bytes(globals_path)
    require(len(globals_data)<10000,'SMALL_NATIVE_GLOBALS')
    rows += [dict(path=globals_path,sha256=hashlib.sha256(globals_data).hexdigest())]
    # Remote stat of this exact file set only, never scheduler or live outputs.
    paths=[r['path'] for r in rows]
    require(len(paths)==len(set(paths)) and all(p.startswith('/data/janghj/') and '..' not in Path(p).parts for p in paths),'EXACT_SMALL_ALLOWLIST')
    script='import json,os,stat,sys; ps=json.load(sys.stdin); print(json.dumps([dict(path=p,size=os.lstat(p).st_size,regular=stat.S_ISREG(os.lstat(p).st_mode),owner=os.lstat(p).st_uid==os.getuid()) for p in ps]))'
    stats=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','rke-server4','python3 -c '+shlex.quote(script)],input=json.dumps(paths).encode(),timeout=30))
    require(all(x['regular'] and x['owner'] and x['size']<10*1024**2 for x in stats),'SMALL_REGULAR_OWNED_INPUTS')
    require(sum(x['size'] for x in stats)<20*1024**2,'SMALL_TRANSFER_BUDGET')
    write(dest/'source-allowlist.json',dict(config=REMOTE_CONFIG,config_sha=CONFIG_SHA,members=rows,stat=stats,source_KEEP=True))
    missing=[p for p in paths if not (dest/'server4'/p.removeprefix('/data/janghj/')).exists()]
    if missing:
        raw=subprocess.check_output(['ssh','-o','BatchMode=yes','rke-server4',
            'tar -cf - --no-recursion --files-from=-'],input=('\n'.join(missing)+'\n').encode(),timeout=60)
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            members=tf.getmembers();require(len(members)==len(missing),'EXACT_TAR_COUNT')
            expected={p.lstrip('/'):r for p,r in zip(paths,rows)}
            for item in members:
                require(item.isfile() and item.name in expected and '..' not in Path(item.name).parts,'SAFE_TAR_MEMBER')
                row=expected[item.name];target=dest/'server4'/row['path'].removeprefix('/data/janghj/')
                staged(target,tf.extractfile(item).read(),row['sha256'])
    mapping={r['path']:member(dest/'server4'/r['path'].removeprefix('/data/janghj/')) for r in rows}
    require(all(mapping[r['path']]['sha256']==r['sha256'] for r in rows),'RECEIVER_FULL_SHA')
    write(dest/'transfer-receipt.json',dict(members=mapping,source_KEEP=True,overwrite=False,model_transfers=0))
    return prior,mapping,configpath

def prepare(cpu):
    prior,mapping,configpath=inputs();out=LOCAL/'preparation';attempt=LOCAL/'attempt'
    require(not attempt.exists() and not (out/'configuration.json').exists(),'CREATE_ONCE_B1_PREPARATION')
    runtime=runtime_binding()
    require((runtime['torch'],runtime['transformers'])==(prior['runtime']['torch'],prior['runtime']['transformers']),'RUNTIME_VERSION')
    oldruntime={r['module']:r['sha256'] for r in prior['runtime']['source_members']}
    require(all(oldruntime[r['module']]==r['sha256'] for r in runtime['source_members']),'RUNTIME_SOURCE_EXACT')
    c=copy.deepcopy(prior)
    c.update(instruction_id=B1_NONCE,task_id=B1_TASK,execution_profile='server2-b1',attempt=str(attempt),
        run_instance=dict(date='2026-10-06',attempt='attempt'),runtime=runtime,cpu_preflight=member(cpu))
    c['model']=prior['model'].replace('/data/janghj/','/mnt/raid5/janghj/',1)
    c['stream']='/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json'
    c['contexts']=mapping[prior['contexts']]['path']
    c['stats']={l:p.replace('/data/janghj/','/mnt/raid5/janghj/',1) for l,p in prior['stats'].items()}
    c['stats_root']=prior['stats_root'].replace('/data/janghj/','/mnt/raid5/janghj/',1)
    c['native_root']=str(LOCAL/'inputs/server4/BLUE')
    c['native_hparams']=mapping[prior['native_hparams']]['path']
    c['native_reference']=[mapping[r['path']] for r in prior['native_reference']]+[mapping[prior['native_root']+'/globals.yml']]
    c['dependency_sources']=[mapping[r['path']] for r in prior['dependency_sources']]
    c['dependency_overlay']=str(Path(c['dependency_sources'][0]['path']).parents[1])
    assets=[]
    for row in prior['assets']:
        original=row.get('requested_path',row['path'])
        path=mapping[original]['path'] if original in mapping else original.replace('/data/janghj/','/mnt/raid5/janghj/',1)
        require(Path(path).is_file(),'MISSING_LOCAL_ASSET:'+path)
        current=member(path);require(current['sha256']==row['sha256'] and current['bytes']==row['bytes'],'LOCAL_ASSET_SHA:'+path)
        assets.append(dict(current,requested_path=path,verification='LOCAL_FULL_SHA_AGAINST_PRODUCTION'))
    c['assets']=assets
    authority=['messages/head/causal-allocation-editing-b1.json','plans/global/causal-allocation-editing/server2-b1.json',
        'PROTOCOL.md','control/gpu-concurrency-policy.tsv',SCHEDULE,
        'audits/servers/server4/causal-allocation-editing/preflight.json']
    authority += [str(p.relative_to(ROOT)) for p in sorted((ROOT/DESIGN).glob('*.json'))]
    c['authority_members']=[member(ROOT/p) for p in authority]
    require(sha(ROOT/authority[0])=='1f339bdc8154f44ba0d9e405e39a6375a73960a5fc0cbb579e7409416e781bd6','ENVELOPE_SHA')
    records=load_prefix(Path(c['stream']).parent,100)
    c['ordered_ids_sha256']=digest([r['case_id'] for r in records]);c['packs']=prior['packs'][:1]
    require(c['packs'][0]['ids']==[r['case_id'] for r in records],'FIRST100_SAME_PRODUCTION_ORDER')
    alignment=json.loads(Path(mapping[prior['native_input_alignment']['path']]['path']).read_text())
    write(out/'native-input-alignment.json',dict(rows=alignment['rows'][:100],packs=alignment['packs'][:1],source=mapping[prior['native_input_alignment']['path']]))
    obs=json.loads(Path(mapping[prior['observer_identity']['path']]['path']).read_text())
    ids=set(c['packs'][0]['ids']);obsrows=[r for r in obs['rows'] if r['case_id'] in ids]
    require(len(obsrows)==1300 and len({r['identity'] for r in obsrows})==1300,'B1_OBSERVER_IDENTITY')
    write(out/'observer-identity.json',dict(rows=obsrows,source=mapping[prior['observer_identity']['path']]))
    c['native_input_alignment']=member(out/'native-input-alignment.json');c['observer_identity']=member(out/'observer-identity.json')
    full=full_native_binding(c,records,c['packs'])
    write(out/'native-full-input-binding.json',full);c['native_full_input_binding']=member(out/'native-full-input-binding.json')
    c['prior_asset_binding']=member(configpath)
    c['readonly_input_evaluator_sources']=[member(ROOT/p) for p in ('project/run_scripts/jlz_realization/inputs.py','project/run_scripts/jlz_realization/observe.py','project/run_scripts/jlz_pilot/prompts.py','scripts/fixed_counterfact.py')]
    c['W0_reuse']=dict(status='NOT_AVAILABLE',reason='NO_EXACT_LOCAL_W0_IDENTITY_BRIDGE; evaluate only first100')
    c['settings'].update(B=100,batches=1,requests=100,milestones=[1],no_B2=True)
    c['settings'].pop('no_B21',None)
    c['resources'].update(project_cap=2,wall='08:00:00',qualification_wall='08:00:00',node='server2',
        partition='gpu',qos='lab_gpu_s2',ETA='NOT_MEASURED;8h request ceiling only',
        free_bytes=shutil.disk_usage(LOCAL).free,free_inodes=os.statvfs(LOCAL).f_favail)
    require(c['resources']['free_bytes']>=c['resources']['reserve_bytes'],'STORAGE_RESERVE')
    write(out/'configuration.json',c)
    write(out/'input-runtime-binding.json',dict(status='CPU_BOUND_NOT_GPU_QUALIFIED',task=B1_TASK,
        source_config=member(configpath),assets=assets,runtime=runtime,packs=1,requests=100,observer_rows=1300,
        full_native_binding=c['native_full_input_binding'],transfer=member(LOCAL/'inputs/transfer-receipt.json')))
    print(json.dumps(dict(configuration=str(out/'configuration.json'),actual_GPU='NOT_RUN')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu-preflight',type=Path,required=True)
    prepare(p.parse_args().cpu_preflight)
