"""One-shot preparation and submission for the explicit record-only recall.

No GPU diagnosis, old-job mutation, or post-release job monitoring.
"""
import argparse
import shutil
import subprocess
from .common import *
from .launch import freeze, submit
from .fidelity import POLICY, CERTIFICATION, NONCE, NLL_LIMIT, MARGIN_LIMIT

OLD=ROOT/'attempt-r2-routing'
PREP=ROOT/'record-only-20260925-r1'
ATTEMPT='attempt-r3-record-only'


def prepare():
    assert not (ROOT/ATTEMPT).exists(), 'IMMUTABLE_ATTEMPT_EXISTS'
    oldp=OLD/'execution.lock.json';old=read(oldp)
    assert sha(oldp)=='54abf9a7572eceda008db9c001ed8d1fc9de8f5a7397990a4bed26f313bc1699'
    assert old['source_commit']=='730a4a9768e5650e01fd9afdc4e0f7895c86ea92'
    for member in old['source_members']:assert sha(member['path'])==member['sha256']
    binding=read(old['T0']['path']);assert sha(old['T0']['path'])==old['T0']['sha256']
    verified=[]
    for family,bs in binding['checkpoints'].items():
        for batch,r in bs.items():
            s=Path(r['path']).stat()
            assert (s.st_size,s.st_ino,s.st_mtime_ns)==(r['bytes'],r['inode'],r['mtime_ns']), 'CP_STAT_DRIFT'
            verified.append(dict(family=family,batch=batch,path=r['path'],sha256=r['sha256'],verification='PRIOR_FULL_SHA_PLUS_UNCHANGED_STAT'))
    assert len(verified)==24
    for r in binding['model_shards']:
        s=Path(r['path']).stat();assert (s.st_size,s.st_mtime_ns)==(r['bytes'],r['mtime_ns'])
    for r in [binding['token_manifest'],binding['model_config'],binding['model_index'],binding['dataset'],*binding['tokenizer'],*binding['design_members']]:
        assert sha(r['path'])==r['sha256'],'INPUT_IDENTITY'
    published=REPO/'messages/head/2026-09-25-historical-timeaxis-numerical-record-only-sh4.md'
    assert NONCE in published.read_text()
    waiver=PREP/'numerical-record-only-authority.md'
    with waiver.open('xb') as output:output.write(published.read_bytes())
    assert sha(waiver)==sha(published)
    pol=dict(instruction_id=NONCE,numerical_fidelity_policy=POLICY,numerical_certification=CERTIFICATION,
             nll_limit=NLL_LIMIT,margin_limit=MARGIN_LIMIT,waiver=record(waiver),
             pass_semantics='STRUCTURAL_COMPLETION_NOT_NUMERICAL_CERTIFICATION',
             blocking_checks=['source/input/token/order/cardinality','whole-five-weight bytes','exact restore','finite','IO'])
    audit=REPO/'audits/servers/server4/historical-update-timeaxis-20260924-v1/rerun-20260925-r1'
    prior=read(audit/'diagnosis.json');assert prior['saved_endpoint_receipts']==9 and prior['independent_row_comparisons']==2592
    inventory={r['path']:r for r in read(audit/'artifact-index.json')}
    endpoints={f:{} for f in FAMILIES};diagonals={}
    for family,ts in ((FAMILIES[0],(0,1,20,50,100)),(FAMILIES[1],(0,1,20,50))):
        for t in ts:
            p=OLD/'output'/family/'fidelity'/f'endpoint-{t:03d}.json'
            assert record(p)==inventory[str(p)], 'PRIOR_FIDELITY_INTEGRITY'
            x=read(p);expected=binding['w0_selected'] if t==0 else binding['checkpoints'][family][str(t)]['weights']
            for s in (x['state'],x['restored']):assert s['weight_hashes']==expected and s['restore']=='EXACT_BYTES'
            endpoints[family][str(t)]=record(p)
    alpha=OLD/'output'/FAMILIES[0]/'T1/PASS.json';assert record(alpha)==inventory[str(alpha)]
    ds=read(alpha)['detail']['diagonals'];assert len(ds)==12
    for d in ds:
        expected=binding['w0_selected'] if d['start']==0 else binding['checkpoints'][FAMILIES[0]][str(d['start'])]['weights']
        for name in ('arithmetic','direct'):assert d[name]['weight_hashes']==expected and d[name]['restore']=='EXACT_BYTES'
    diagonals[FAMILIES[0]]=record(alpha)
    bridge=dict(old_lock=record(oldp),old_execution=old['source_commit'],old_jobs=[53182,53183,53184],old_allocated_GPU_seconds=1569,
        new_policy_sha256=digest(pol),T0=old['T0'],token_manifest=binding['token_manifest'],checkpoints=verified,
        model_verification='PRIOR_FULL_SHA_PLUS_UNCHANGED_STAT',endpoints=endpoints,diagonals=diagonals,
        independent_CPU_review=record(audit/'diagnosis.json'),prior_artifact_inventory=record(audit/'artifact-index.json'),
        scientific_completed_tasks=0,prior_PASS_not_relabelled=True,
        reuse_evidence_level='nine retained GPU endpoint raw comparisons; twelve Alpha GPU diagonal receipts only, no diagonal raw',
        not_reused='MEMIT W100 unsaved failed comparison and subsequent MEMIT diagonals; all scientific score tasks',
        numerical_certification=CERTIFICATION,
        source_delta='diagnostic classification/storage, explicit receipt reuse and policy join only; evaluator/token/model/materialization unchanged')
    br=save(PREP/'reuse-bridge.json',bridge)
    dest=freeze(ATTEMPT,dict(numerical_policy=pol,reuse_bridge=br,latest_instruction=NONCE,
                           prior_failure_allocation_GPU_seconds=1569,monitoring_after_release=False))
    lock=read(dest/'execution.lock.json')
    for k in ('contract_sha256','T0_sha256','token_sha256','evaluator_signature','scope','save_checkpoints'):
        assert lock[k]==old[k],('UNINTENDED_SCIENCE_CHANGE',k)
    for script in ('gpu.sbatch','collector.sbatch'):
        subprocess.run(['bash','-n',str(dest/script)],check=True)
    memory=subprocess.run(['python3','scripts/slurm_memory_policy.py','audit',str(dest/'gpu.sbatch'),str(dest/'collector.sbatch')],text=True,capture_output=True)
    save(dest/'script-memory-audit.json',dict(exit_code=memory.returncode,stdout=memory.stdout,stderr=memory.stderr))
    assert memory.returncode==0
    print(json.dumps(dict(status='FROZEN_NOT_SUBMITTED',attempt=str(dest),lock=record(dest/'execution.lock.json'),reuse_bridge=br)))


def launch():
    dest=ROOT/ATTEMPT;lock=read(dest/'execution.lock.json')
    for m in lock['source_members']:assert sha(m['path'])==m['sha256']
    assert not any(dest.glob('submitted-*.json')) and not (dest/'submission.json').exists(),'NO_DUPLICATE_SUBMISSION'
    for script in ('gpu_script','cpu_script'):
        r=read(dest/'source-and-lock-receipt.json')[script];assert sha(r['path'])==r['sha256']
    assert shutil.disk_usage(ROOT).free>=20*2**30,'STORAGE_BLOCKED'
    cap=subprocess.run(['bash','scripts/check-slurm-resource-cap.sh','server4','2','120832M'],
        env=dict(os.environ,AGENT_GPU_CAPS_FILE='/data/janghj/ODE-edit/servers/local/gpu-caps.tsv'),text=True,capture_output=True)
    save(dest/'resource-cap.json',dict(exit_code=cap.returncode,stdout=cap.stdout,stderr=cap.stderr))
    assert cap.returncode==0,('RESOURCE_CAP_BLOCK',cap.stdout,cap.stderr)
    submit(dest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','submit'));a=p.parse_args()
    prepare() if a.action=='prepare' else launch()
