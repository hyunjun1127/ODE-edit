"""CPU reuse-first admission. Original evidence is strictly read-only."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import torch
from .common import ROOT,STEM,ARMS,INSTRUCTION,member,sha,write,tensor_sha,digest

OLD=Path('/data/janghj/ODE-edit/local/local-z-adaptive-allocation/20260916-v1')
SLZ=Path('/data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2')

def csv_once(path,rows):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def exact_copy(source,destination):
    raw=source.read_bytes()
    if destination.exists():
        assert destination.read_bytes()==raw,'EXISTING_AUTHORITATIVE_BYTES_DIFFER'
    else:
        with destination.open('xb') as f:f.write(raw)

def main():
    repo=Path(__file__).resolve().parents[3]
    manifest=json.loads((repo/f'plans/global/{STEM}-sh4-dispatch/source-input-manifest.json').read_text())
    copied=[]
    for spec in manifest['files']:
        p=repo/spec['destination'];m=member(p)
        assert m['bytes']==spec['bytes'] and m['sha256']==spec['sha256']
        dst=ROOT/'authoritative'/spec['destination'];dst.parent.mkdir(parents=True,exist_ok=True)
        exact_copy(p,dst)
        copied.append(dict(**m,local=str(dst),lines=len(p.read_bytes().splitlines())))
    for relative in [f'messages/head/2026-09-18-sh4-single-layer-edit-preserving-correction-m.md',f'plans/global/{STEM}-sh4-dispatch/user-instruction-ko.md',f'plans/global/{STEM}-sh4-dispatch/source-input-manifest.json','PROTOCOL.md','control/gpu-concurrency-policy.tsv','servers/active/server4.md']:
        p=repo/relative;dst=ROOT/'authoritative'/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        exact_copy(p,dst)
        copied.append(dict(**member(p),local=str(dst),lines=len(p.read_bytes().splitlines())))
    validation=json.loads((repo/f'audits/global/{STEM}/validation_receipt.json').read_text())
    assert validation['checks_total']==1430 and all(c['passed'] for c in validation['checks'])
    assert os.environ['CODEX_THREAD_ID']=='01a04939-b5c7-7a03-ba2d-ef3343d62cfd'
    fullread_value=dict(instruction=INSTRUCTION,host=os.uname().nodename,
       session=os.environ['CODEX_THREAD_ID'],main=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
       files=copied,reading='Design/contract/cells/positioning/audit/reference source read in full;1430 receipt entries parsed and validated; unchanged PROTOCOL exact prior FULL_READ reuse',
       protocol_prior_fullread=member(SLZ/'full-read-m0.json'),CPU_prior=dict(toy=11,design=1430,actual_model_PASS=False),
       scope=['T','M_REUSE','M_EVAL_ONLY','M_RUN_MISSING'],forbidden_stages=['S','R','L'],cap=2,new_M_final_L4_required=True)
    fp=ROOT/'receipts/full-read-m0.json'
    if fp.exists():
        assert json.loads(fp.read_text())==fullread_value
        fullread=member(fp)
    else:fullread=write(fp,fullread_value)
    lock=json.loads((SLZ/'execution.lock.json').read_text())
    capsule=json.loads(Path(lock['cold_capsule']['path']).read_text())
    fitpath=OLD/'arms/N4/attempt-v1/output/B001/proposals/own-N4.pt'
    fit=torch.load(fitpath,map_location='cpu',weights_only=True,mmap=True)
    receipt=fit['receipt'];assert tensor_sha(fit['weight'])==receipt['endpoint_weight_sha256']
    assert receipt['input_state']['W']['4']==capsule['W0']['4']
    assert receipt['input_state']['contexts']==digest(capsule['contexts'])
    assert fit['target'].shape==(4096,100) and fit['captures']['compute_ks'][0].shape==(100,14336)
    assert [v['case_id'] for v in fit['target_observations']]==lock['sample_order'][:100]
    binding=dict(native=member(fitpath),receipt=receipt,source_lock=member(OLD/'execution.lock.json'),
                 cold_capsule=member(Path(lock['cold_capsule']['path'])),b1_endpoint_verified=tensor_sha(fit['weight']),
                 native_targets=100,native_fit_new_allowed=False,A='not retained; same K/P/M factor solve diagnostic only, never replace actual WN',
                 full_resume_claim=False,deleted_checkpoints='not required: retained actual proposal weight+native targets/keys/readout available')
    write(ROOT/'reuse/b001-native-binding.json',binding)
    rows=[]
    for b in range(1,11):
        for arm in ARMS+('CA-EXACT','RAND+','RAND-'):
            for item in ('shared_native','endpoint_optimization_or_geometry','observer'):
                status='RUN_MISSING';reason='No existing ENFC-v1 same-space/optimizer/per-sequence guard result';unit=item
                newfit=False;grad=arm not in ('N4','CA-EXACT','RAND+','RAND-') and item=='endpoint_optimization_or_geometry'
                if item=='shared_native':
                    status='REUSE' if b==1 else 'RUN_MISSING';newfit=b>1 and arm=='N4'
                    reason='Exact cold7 B1 retained WN/z/K/P/M0/context/seed; no fit repeat' if b==1 else 'Existing sequential B2+ entry is not W0/M0; prior independent reset family uses different target/editor configuration'
                    unit=f'b{b:03d}-shared-native-once'
                elif arm=='N4' and item=='endpoint_optimization_or_geometry':
                    status='REUSE' if b==1 else 'RUN_MISSING';reason='Native endpoint actual tensor retained' if b==1 else 'Requires this independent cold episode native endpoint'
                elif arm=='N4' and item=='observer' and b==1:
                    status='EVAL_ONLY';reason='Canonical B1 R/P/N reusable; per-sequence lock/Dev128/greedy32 diagnostics incomplete'
                rows.append(dict(cell=f'M-O0-b{b:03d}-{arm}',batch=b,arm=arm,diagnostic=item,status=status,
                    source='32a92ad6f3fff2f258d8778f3936d152e975ac1b' if b==1 else 'historical sequential or different independent editor',
                    evidence_path=str(fitpath) if b==1 else str(OLD/f'arms/N4/attempt-v1/output/B{b:03d}/entry.json'),
                    endpoint_available=b==1,endpoint_sha=receipt['endpoint_weight_sha256'] if b==1 else '',
                    source_lock=str(OLD/'execution.lock.json'),context_capsule=lock['cold_capsule']['path'],
                    comparison='W0/zeroM4,seed20260916,FP32/eager/TF32off,sameO0B100/nativehparams',
                    evidence_missing=reason,compute_unit=unit,new_native_fit=newfit,new_gradient=grad,
                    new_forward=item!='shared_native',capsule_id=f'M-O0-b{b:03d}',
                    technical_validation='T_ACTUAL_NOT_RUN_NEW_IMPLEMENTATION',
                    prior_cost='cold7 prior allocation; B1fit283.165502s nested; not rebilled' if b==1 else 'REFERENCE_ONLY',
                    forbidden='REUSE refit;warm/sequential-as-cold;S/R/L;P/N online;Report256'))
    write(ROOT/'reuse/m-reuse-decisions.json',rows);csv_once(ROOT/'reuse/m-reuse-decisions.csv',rows)
    plan=[dict(episode=f'b{b:03d}',ordinals=f'{100*(b-1)}:{100*b}',native='REUSE' if b==1 else 'RUN_MISSING',
               controllers='SCALE;CA;KL-P;EN-S;EN-F;EN-COV;EN-F4',diagnostics='CA-EXACT;RAND+;RAND-',
               final_L4_endpoints=8,admission='T_READY_REQUIRED',new_native_target_calls=0 if b==1 else 100,
               run_mode='INDEPENDENT_W0_M0',S_R_L_allowed=False) for b in range(1,11)]
    csv_once(ROOT/'reuse/m-execution-plan.csv',plan)
    # Historical roles are explicit; their metric values do not affect decisions.
    write(ROOT/'reuse/reference-only.json',[
      dict(family='EP-TW cap',reason='Different cap/residual metric/meanE guard/seed; not CA/EN-F'),
      dict(family='SL-ZFlow',reason='Actual-write target learning, not fixed-W0 ENFC objective; remote evidence not imported'),
      dict(family='fixed-z L8 screen',reason='L8/rank-one and partial token lock, not single L4/fulltoken'),
      dict(family='p1r54 independent reset',reason='Different target-time/realization controller and older context/source, not prescribed native endpoint'),
      dict(family='cold7/sequential-v2 B2..10',reason='Own sequential nonzero M/edited entry; cannot replace independent cold episode')])
    fields=('snapshot','blue_root','config4','projector','dataset_root','records_digest','sample_order','reference_root','teacher_manifest','cold_capsule','historical_evaluator_root','helper_scripts_root','editor_sha256','torch','transformers')
    runtime={k:lock[k] for k in fields}
    runtime.update(instruction_id=INSTRUCTION,seed=20260916,source_root=str(repo),allowed_stages=['T','M'],gpu_cap=2,
                   reused_native_b1=str(fitpath),M_native_new_max=9,final_endpoint_retention='ALL_M_FINAL_L4',
                   loss_microbatch=1,guard_microbatch=1,official_observer_microbatch=16,
                   TF32_matmul=False,TF32_cudnn=False,full_read=fullread,
                   historical_assets_prior_fullsha=member(SLZ/'execution.lock.json'))
    write(ROOT/'inputs/base-binding.json',runtime)
    disk=shutil.disk_usage(ROOT);stat=os.statvfs(ROOT)
    write(ROOT/'receipts/storage-plan-r0.json',dict(free=disk.free,inodes=stat.f_favail,
        final80_L4_bytes=80*4096*14336*4,native_capsules10_upper=10*270000000,
        plan_reserve_bytes=48*(1<<30),teacher_reuse_new_bytes=0,fullmodel_copy=0,
        geometry_cache_retention='compact bases/spectra/provenance; no full candidate model; selected endpoints retained',
        resource_status='PRELIMINARY_NOT_GPU_ADMISSION',new_M_gradient_upper=100,trial_upper=720))
    print(json.dumps(dict(full_read=fullread,rows=len(rows),shared_native_reuse=1,new_native_max=9,M_episodes=10,science_submitted=0)))

if __name__=='__main__':main()
