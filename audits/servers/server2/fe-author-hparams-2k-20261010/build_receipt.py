"""Compact task receipt; raw/source tensors and private environment stay local."""
import json
from pathlib import Path
from datetime import datetime, timezone
from official.experiments.prepare import file_sha

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/fe-author-hparams-2k-20261010')
def read(p):return json.loads(Path(p).read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')

prep=read(LOCAL/'preparation-r1/asset-preparation.json')
assets=read(prep['assets']['path'])
report=dict(instruction_id=prep['instruction_id'],server='server2',
    accepted_turn='01a122f4-e240-7532-91fd-5295daaa902a',observed_utc=datetime.now(timezone.utc).isoformat(),
    asset_manifest_sha256=prep['assets']['sha256'],assets_identity_sha256=assets['assets_sha256'],
    model_revision=assets['model']['identity']['revision'],
    model_members=[{k:m[k] for k in ('path','sha256','bytes')} for m in assets['model']['members']],
    C0={k:dict(sha256=v['member']['sha256'],bytes=v['member']['bytes'],validation=v['validation']) for k,v in assets['C0'].items()},
    context_canonical_sha256=prep['context_canonical_sha256'],context_member_sha256=prep['context']['sha256'],
    context_ready_sha256=prep['context_ready']['sha256'],context_token_ID_CPU_match=True,
    zsre_query_parity=read(LOCAL/'preparation-r1/zsre-query-parity.json'),
    storage=prep['storage'],resources=dict(GPUs=1,CPUs=6,memory_MiB=59392,wall_hours=48,project_cap=4),
    history_OOM_repair='FP64_SYSTEM_BUFFER_LIFETIME_AND_CPU_ROLLBACK_PRESENT',
    memory_plan=dict(selected_W_H_bytes=prep['storage']['selected_weights_history_bytes'],
        C0_CPU_bytes=5*18944**2*4,H_CPU_bytes=5*18944**2*4,old_new_H_overlap=True,
        FP64_single_system_bytes=18944**2*8,CPU_rollback_weights_bytes=5*3584*18944*4,
        actual_peak_NOT_MEASURED=True,reason='Native repaired source unchanged; no GPU qualification'),
    qualification='NOT_RUN_USER_DISABLED',GPU_forward=0,downloads=0,asset_copies=0,
    existing_jobs_changed=False,broadcast='NO_BROADCAST_NOT_REQUIRED')
write(OUT/'preparation.json',report)
sub=LOCAL/'registration-r1/submission.json'
if sub.exists():
    s=read(sub); rows=[]
    for dataset,j in s['jobs'].items():
        c=read(j['config']['path']);state=s['initial_snapshot'][dataset]
        rows.append(dict(model='Qwen2.5-7B',dataset=dataset,server='server2',
            method='MEMIT_FE_HISTORY (FE author hparams)',job_id=j['job_id'],job_name=j['name'],
            state=state['JobState'],reason=state['Reason'],dependency=j['dependency'],source=s['source'],
            official_tree=s['official_tree'],config_sha256=c['config_sha256'],
            author_profile_sha256=c['author_profile_sha256'],resources=report['resources'],
            W20_checkpoint_directory=str(Path(c['output'])/'checkpoint'),
            checkpoint_locator='checkpoint/latest.json batch=20/final_W20=true required; filename content-addressed',
            evaluator='official.evaluation.zsre_paper' if dataset=='zsre' else 'official.evaluation.factual',
            flucon='NOT_APPLICABLE' if dataset=='zsre' else 'DEFERRED_CHECKPOINT_EVALUATION',
            held_inspected=s['held_inspected'],released=s['released'],WandB_stage='NOT_STARTED_DEPENDENCY_PENDING',
            W20_observed=False))
    write(OUT/'table-rows.json',rows)
    write(OUT/'submission.json',s)
    write(OUT/'source-members.json',{p:file_sha(ROOT/p) for p in [
        'official/runners/fe_author_history.py','official/baselines/fe_author_profile.py',
        'official/hparams/MEMIT_FE_HISTORY_AUTHOR/profiles.json','official/baselines/memit_fe_history.py',
        'official/evaluation/zsre_paper.py','official/baselines/easyedit/util/generate.py']})
