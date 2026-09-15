"""EP input/source lock without model loading or baseline/calibration access."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from .control import (DISPATCH, ENVELOPE, OLD, REPO, TASK, boundary, git,
                      identity, save, sha, verify_dispatch)

DATASET=REPO/'local/datasets/counterfact-fixed-10k-v1'
DEPS=REPO/'local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2'
CONTEXT=REPO/'local/blue-lifelong-b100x100/attempt-checkpoint-r2/output/main-cell-3/B001/contexts.json'
CONFIG=REPO/'local/low-cost-write-donor-pilot/20260913-v1/attempt-v1/config4.json'
PRIOR=REPO/'local/low-cost-write-donor-pilot/20260913-v1/attempt-v1/execution.lock.json'

def inputs(worktree,attempt):
    from scripts.fixed_counterfact import load_prefix, encoded, sha as digest, ORDER_ROOT
    w,a=Path(worktree),Path(attempt)
    boundary(w);verify_dispatch(w/DISPATCH)
    rows=load_prefix(DATASET,1000)
    sample=json.loads((DATASET/'source-sample.lock.json').read_text())
    assert sample['prefix1000_root']=='40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd'
    assert [r['case_id'] for r in rows]==[r['case_id'] for r in sample['records'][:1000]]
    batches=[]
    for i in range(10):
        cur=rows[i*100:(i+1)*100]
        batches.append(dict(batch=i+1,ordinals=[100*i,100*(i+1)],case_ids=[r['case_id'] for r in cur],
            record_sha256=digest(encoded(cur)),request_target_sha256=digest(encoded([r['requested_rewrite'] for r in cur])),
            inventory={'RS':100,'PS':sum(len(r['paraphrase_prompts']) for r in cur),'NS':sum(len(r['neighborhood_prompts']) for r in cur)}))
    config=json.loads(CONFIG.read_text());assert config['blue'] and config['layers']==[4] and config['L2']==1
    assert config['v_num_grad_steps']==25
    assert sha(CONTEXT)=='33cec0eef9ec130f26c2f0e17f7c8be39e93c717ebb47eaa5e9f1c88b263524e'
    for p,n in [(CONFIG,'config4.json'),(CONTEXT,'contexts.json')]:
        d=a/'inputs'/n;d.parent.mkdir(parents=True,exist_ok=True)
        with p.open('rb') as src,d.open('xb') as dst:shutil.copyfileobj(src,dst)
        d.chmod(0o400)
    return save(a/'sample.lock.json',dict(instruction_id=TASK,whole_ordered_root=ORDER_ROOT,
        prefix1000_root=sample['prefix1000_root'],dataset=identity(DATASET/'counterfact.json'),
        source_sample=identity(DATASET/'source-sample.lock.json'),batches=batches,
        ordinals=[0,1000],unique_requests=1000,model_loads=0,shuffle=False,replacement=0,
        initial='W0/coldM0; reused context bytes, no warm W/M/RNG import',seed=20260915))

def freeze(worktree,attempt):
    import torch,transformers
    from .policy import NumericalPolicy
    from .technical import TechnicalNumerics
    w,a=Path(worktree).resolve(),Path(attempt).resolve()
    b=boundary(w);d=verify_dispatch(w/DISPATCH)
    assert not git(w,'status','--porcelain','--','project/run_scripts/bg_tw_reference/ep_tw'), 'UNCOMMITTED_RUNTIME'
    assert torch.__version__=='2.9.1+cu128' and transformers.__version__=='4.44.2'
    prior=json.loads(PRIOR.read_text())
    from project.run_scripts.low_cost_write_donor_pilot.fitting import select_projector
    fullp=torch.load(prior['projector'],map_location='cpu',weights_only=True,mmap=True)
    P,pmap=select_projector(fullp,4)
    assert P.shape==(1,14336,14336) and P.dtype==torch.float32 and torch.isfinite(P).all()
    del P,fullp
    teacher=json.loads((OLD/'teacher-output-v1/teacher-manifest.json').read_text())
    reused=json.loads((a/'teacher-reuse-verification.json').read_text())
    assert reused['status']=='TEACHER192_FULLSHA_SCHEMA_FINITE_REUSED'
    assert identity(OLD/'teacher-output-v1/teacher-manifest.json')==reused['manifest']
    built=json.loads((OLD/'reference-v1/build-status.json').read_text())
    assert built['reference_identity_sha256']=='f5791dd3c986261a252796bd5d7293ece46339d261609449dcfe674970bbfde0'
    assert built['member_root']=='0c6aa4e2ddb350c61999580fe3efabaae880ec8e17aa208a8f7d85c026a6f1aa'
    for member in built['members']:
        assert Path(member['path']).stat().st_size==member['bytes'] and sha(member['path'])==member['sha256']
    sample=json.loads((a/'sample.lock.json').read_text())
    source=a/'source-v1';source.mkdir(exist_ok=False,mode=0o700)
    # All imported project modules reside in a single frozen source tree. Flat
    # legacy policy files are read-only dependencies, not launch lists.
    roots=['bg_tw_reference','low_cost_write_donor_pilot','baseline_mechanism_first']
    rels=[]
    for root in roots:
        rels += [str(p.relative_to(w)) for p in sorted((w/'project/run_scripts'/root).rglob('*.py'))]
    rels += ['scripts/fixed_counterfact.py',DISPATCH,ENVELOPE,
        'project/run_scripts/alphaedit_strength_neutral_barrier/contracts.py',
        'plans/global/2026-09-15-edit-quality-preserving-tw-contract.json',
        'project/run_scripts/bg_tw_reference/ep_tw/run.sbatch']
    for rel in rels:
        p=w/rel;dst=source/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        with p.open('rb') as f,dst.open('xb') as out:shutil.copyfileobj(f,out)
        dst.chmod(0o400)
    archive=a/'source-v1.tar'
    with archive.open('xb') as f,tarfile.open(fileobj=f,mode='w') as tar:
        for rel in rels:
            info=tar.gettarinfo(str(source/rel),arcname=rel)
            info.uid=info.gid=info.mtime=0;info.uname=info.gname='';info.mode=0o400
            with (source/rel).open('rb') as src:tar.addfile(info,src)
    members={str(source/r):identity(source/r) for r in rels}
    # Exact reused environment/model kernel closure from successful teacher.
    tl=json.loads((OLD/'teacher.lock.json').read_text())
    for m in tl['members']:
        p=m['path']
        if p.startswith(tl['snapshot']+'/') or p.startswith(str(DEPS)+'/') or '/site-packages/torch/' in p:
            assert Path(p).stat().st_size==m['bytes'];members[p]={k:m[k] for k in ('path','bytes','sha256')}
    # Native and canonical observation modules stay in original immutable roots.
    for root in [Path(prior['blue_root']),Path(prior['helper_scripts_root'])/'blue_alphaedit_sequential_comparison',
                 Path(prior['helper_scripts_root'])/'alphaedit_strength_neutral_barrier',
                 Path(prior['helper_scripts_root'])/'ordered_response_barrier_ode']:
        for p in sorted(root.rglob('*.py')):members[str(p)]=identity(p)
    for p in Path(prior['blue_root']).glob('*.yml'):members[str(p)]=identity(p)
    for p in [Path(prior['projector']),a/'inputs/config4.json',a/'inputs/contexts.json',
              a/'sample.lock.json',a/'full-read-receipt.json',a/'teacher-reuse-verification.json',
              OLD/'teacher-output-v1/teacher-manifest.json',OLD/'reference-v1/build-status.json',
              OLD/'reference-v1/source-manifest.json',DATASET/'counterfact.json',DATASET/'source-sample.lock.json',DATASET/'receipt.json']:
        members[str(p)]=identity(p)
    for m in [teacher['reference_tokens'],teacher['splits'],*teacher['cache_shards']]:members[m['path']]=m
    from dataclasses import asdict
    numerics=asdict(NumericalPolicy())
    tech=TechnicalNumerics().to_dict()
    lock=dict(instruction_id=TASK,policy='EP-TW-1',dispatch=identity(source/DISPATCH),
        source_head=b['source_head'],source_tree=b['source_tree'],source_root=str(source),source_archive=identity(archive),
        members=list(members.values()),model_revision=tl['model_revision'],snapshot=tl['snapshot'],
        torch=torch.__version__,transformers=transformers.__version__,tf32_matmul=False,tf32_cudnn=True,
        dtype='float32',attention='eager',writer_add_bos_token_attribute=False,canonical_microbatch=16,
        seed=20260915,config4=str(a/'inputs/config4.json'),contexts=str(a/'inputs/contexts.json'),
        blue_root=prior['blue_root'],projector=prior['projector'],projector_mapping=pmap,editor_sha256=prior['editor_sha256'],
        historical_evaluator_root=prior['historical_evaluator_root'],helper_scripts_root=prior['helper_scripts_root'],
        dataset_root=str(DATASET),sample_lock=identity(a/'sample.lock.json'),batches=sample['batches'],
        reference_root=str(OLD/'reference-v1'),teacher_manifest=reused['manifest'],teacher_reuse_receipt=identity(a/'teacher-reuse-verification.json'),
        numerical_policy=numerics,numerical_rationale=NumericalPolicy().receipt(),technical_numerics=tech,
        resource=dict(gpu=1,cpus=8,mem='60416M',wall='12:00:00',gpu_cap=2,gpu_hour_cap=None),
        cost_plan=dict(new_estimate_gpu_hours=[2,8],wall_reserve_factor_over_upper_estimate=1.5,
            estimate_basis='1000 native targets/10 solves +10 current/S64 gradient sweeps +<=40 candidate forwards +observations/checkpoints; not teacherwall inheritance',
            checkpoint_estimated_bytes=10*(4096*14336+14336*14336)*4,
            total_new_disk_reserve_bytes=30*(1<<30),reference_teacher_allocated_gpu_seconds_reused=98),
        output=str(a/'scientific-v1'),warm_state_imports=0,M8_imports=0,N4_calibration=False,
        new_scientific_chains=1,baseline_reruns=0,observation_feedback=False,
        after_gate='WAITING_USER_RESUME',no_broadcast='NO_BROADCAST_NOT_REQUIRED')
    return save(a/'execution.lock.json',lock)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inputs','freeze']);p.add_argument('--worktree',required=True);p.add_argument('--attempt',required=True)
    x=p.parse_args();print(json.dumps(inputs(x.worktree,x.attempt) if x.command=='inputs' else freeze(x.worktree,x.attempt)))
