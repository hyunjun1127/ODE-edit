"""Bind existing GPT2 assets and CPU generation plans; never evaluate W0."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess

from .run import TASK,NONCE
from .profile import modified_profile

LOCAL=Path('/data/janghj/ODE-edit/local')


def prepare(attempt):
    from transformers import AutoTokenizer
    from scripts.fixed_counterfact import load_prefix
    from project.run_scripts.jlz_interference_l1.cap_common import member,verify,sha,require,digest
    from project.run_scripts.experiment_generation_eval.kv_qualification import build_qualification_plan
    from .gpt2_binding import validate
    attempt.mkdir(exist_ok=False)
    data=json.loads((LOCAL/TASK/'attempt-r1/config.json').read_text())
    oldroot=LOCAL/'gpt2xl-alpha-l13-2k';oldconfig=oldroot/'attempt-logging-repair/config.json'
    gpt2=json.loads(oldconfig.read_text())
    parent_path=oldroot/'assets/server1-parent-config.json'
    parent=json.loads(parent_path.read_text());native=parent['models']['MEMIT']
    assets=LOCAL/TASK/'assets';assets.mkdir(exist_ok=True)
    stats={};transfers=[]
    for layer in range(13,18):
        source=native['stats'][str(layer)]
        row=next(r for r in native['assets'] if r['path']==source)
        dest=Path(gpt2['stats']['13']) if layer==13 else assets/Path(source).name
        require(source.startswith('/mnt/raid5/janghj/EasyEdit/examples/data/stats/gpt2-xl/wikipedia_stats/'),
                'EXACT_STAT_TRANSFER_ALLOWLIST')
        if not dest.exists():
            subprocess.run(['rsync','-a','--no-owner','--no-group','--ignore-existing','-e',
                'ssh -o BatchMode=yes -o ConnectTimeout=10','rke-server1:'+source,str(dest)],check=True,timeout=180)
        require(dest.stat().st_size==row['bytes'] and sha(dest)==row['sha256'],'EXISTING_STAT_BYTES')
        stats[str(layer)]=str(dest);transfers.append(dict(source=source,destination=member(dest),source_KEEP=True))
    with (attempt/'stat-transfer.json').open('x') as f:json.dump(transfers,f,indent=2)
    phase0=json.loads(verify(data['phase0']).read_text());realization=next(r for r in phase0['models'] if r['model']=='gpt2xl')
    cell='GPT2XL_M1_M2';profile=modified_profile(native['profiles']['CAP075'],cell,realization)
    gpt2.update(task_id=TASK,instruction_id=NONCE,cell=cell,writer='memit',model_alias='gpt2xl',
        arm_profiles={cell:profile},phase0_status='PASS',w0_policy='REUSE_ONLY_NO_FORWARD',generate_contexts=False,
        stats=stats,cold_W0_H0=native['cold_W0_H0'],full_cold_W=native['cold_W0_H0']['W'],
        parent_attempt=str(oldroot/'attempt-logging-repair'),parent_cell='ALPHAEDIT_CAP075',
        resources=dict(cpu=8,gpu=1,project_cap=2,task_cap=2,host_mib=59392,hard_host_mib=60416),
        run_instance=dict(attempt='generation-kv-repair-20261008'),generation_reserve_bytes=8*1024**3)
    gpt2['generation']['W0_producer']='DISABLED_USER'
    gpt2['task_memory_plan']=copy.deepcopy(parent['resources']['memory_plans']['MEMIT'])
    gpt2['task_memory_plan']['host_peak_with_generation_GiB']=gpt2['task_memory_plan']['host_peak_GiB']+2.5
    gpt2['assets'] += [row['destination'] for row in transfers if row['destination']['path']!=stats['13']]
    w0root=oldroot/'attempt-logging-repair/ALPHAEDIT_CAP075'
    oldruntime=json.loads((w0root/'runtime.json').read_text())
    gpt2['W0_reuse']=dict(status='QUALIFIED_EXACT_REUSE_SELECTED_STATE_EXPANSION',
        chunks=[member(p) for p in sorted((w0root/'W0').glob('chunk-*.json'))],
        summary=member(w0root/'W0/summary.json'),runtime=member(w0root/'runtime.json'),
        cold_state=oldruntime['cold_W0_H0'],observation_identity=gpt2['observation_identity'],
        source_folder=str(w0root/'W0'))
    require(gpt2['model']==oldruntime['model'],'SAME_SERVER_MODEL_PATH')
    validate(gpt2)
    data['cells'][cell]=gpt2;data['blocked_cells'].pop(cell)
    data['input_members'] += [member(oldconfig),member(parent_path),member(attempt/'stat-transfer.json')]
    for key in ('contexts_member','native_input_alignment','native_full_input_binding','observer_identity'):
        data['input_members'].append(gpt2[key])
    data['input_members'] += gpt2['W0_reuse']['chunks']+[gpt2['W0_reuse']['summary'],gpt2['W0_reuse']['runtime']]
    data['input_members'] += [x['destination'] for x in transfers]
    for cell,c in data['cells'].items():
        c['run_instance']=dict(attempt='generation-kv-repair-20261008')
        tok=AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
        records=load_prefix(Path(c['stream']).parent,2000)
        require(digest([r['case_id'] for r in records])==c['ordered_ids_sha256'],'GENERATION_PLAN_ORDER')
        identity=dict(model=c['model_alias'],revision=c['model_revision'],observation_identity=c['observation_identity'])
        plan=build_qualification_plan(tok,[dict(r,ordered_occurrence=i) for i,r in enumerate(records[:100])],
            model_identity=identity,microbatch=4,memory_alternative=True)
        require(plan['coverage']['batch_width_covered'],'GENERATION_MB4_TOKEN_COVERAGE')
        path=attempt/(cell+'-generation-plan.json')
        with path.open('x') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
        c['generation']['qualification_plan']=member(path)
        c['generation']['qualification_state']='POST_B1_EDIT_ONLY_NOT_W0'
        c['generation']['requested_microbatch']=4
        c['generation']['reference_fallback_allowed']=False
        c['task_memory_plan']['KV_cache_extra_GPU_GiB_upper']=1.0
        data['input_members'].append(member(path))
        for row in c['assets']:
            s=Path(row['path']).stat()
            require((s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_STAT_CHANGED')
    reserve=max(c['storage']['reserve_bytes'] for c in data['cells'].values())+sum(c['generation_reserve_bytes'] for c in data['cells'].values())
    free=shutil.disk_usage(attempt).free;require(free>=reserve,'STORAGE_REPAIR_ADMISSION')
    data.update(repair='USER_GENERATION_KV_AND_GPT2_INPUT_REPAIR',
        storage=dict(free_bytes=free,reserve_bytes=reserve,measured=False,shared_reservation=False))
    for row in data['input_members']:verify(row)
    with (attempt/'config.json').open('x') as f:json.dump(data,f,ensure_ascii=False,sort_keys=True,indent=2)
    print(json.dumps(dict(cells=list(data['cells']),config=member(attempt/'config.json'),storage=data['storage'],
        new_W0=0,new_stats=0,stats_transferred=sum(x['destination']['bytes'] for x in transfers[1:]))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    prepare(p.parse_args().attempt)
