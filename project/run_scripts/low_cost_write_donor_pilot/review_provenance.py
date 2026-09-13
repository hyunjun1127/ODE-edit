"""CPU-only, bounded output rehash and recorded compute ledger; no model imports."""
import argparse, csv, hashlib, json, re
from pathlib import Path

ARMS=['N4','S875','S75','FULL8','RES8','REFIT4']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def write(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f: json.dump(o,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def csvwrite(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run(attempt,out):
    attempt=Path(attempt);out=Path(out);root=attempt/'output';out.mkdir(parents=True,exist_ok=True)
    load=lambda p:json.loads(Path(p).read_text())
    lock=load(attempt/'execution.lock.json');terminal=load(root/'core-terminal.json')
    assert sha(attempt/'execution.lock.json')=='a672a782c20543b66add9ed83dc5cf04432e9f98f6cc2646e40b68e3627a9317'
    assert sha(attempt/'execution-source.tar')=='d6cf34ab416ee313a01c8491ac0ab9d15a0f39d88a5ea8f7ead43b439dbe4312'
    assert terminal['status']=='SIX_FIXED_ENDPOINTS_COMPLETE' and list(terminal['endpoints'])==ARMS
    assert not (root/'failure.json').exists()
    inventory=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file():continue
        assert not p.is_symlink()
        before=p.stat();digest=sha(p);after=p.stat()
        assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
        inventory.append(dict(path=str(p),relative=str(p.relative_to(root)),bytes=before.st_size,sha256=digest,verification='NEW_FULL_SHA_SIZE_STABLE_STAT'))
    bypath={r['path']:r for r in inventory}
    def refs(obj):
        if isinstance(obj,dict):
            if {'path','bytes','sha256'}<=obj.keys() and obj['path'] in bypath:
                row=bypath[obj['path']];assert (obj['bytes'],obj['sha256'])==(row['bytes'],row['sha256'])
            for v in obj.values():refs(v)
        elif isinstance(obj,list):
            for v in obj:refs(v)
    for p in root.rglob('*.json'):refs(load(p))
    csvwrite(out/'raw-member-inventory.csv',inventory)
    # Rehash frozen code and newly sealed small inputs, reuse prior asset verification.
    identities=[]
    for r in lock['members']:
        p=Path(r['path']);new=str(p).startswith(str(attempt/'source')) or str(p).startswith(str(attempt)) and p.suffix!='.pt'
        if new: assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
        identities.append(dict(r,verification='NEW_REHASH' if new else 'REUSED_LOCK_AND_SUCCESSFUL_JOB_LOCK_VERIFY'))
    csvwrite(out/'execution-member-identities.csv',identities)
    write(out/'comparison-capsule.json',load(root/'comparison-capsule.json'))
    write(out/'history-provenance.json',load(root/'history-provenance.json'))
    write(out/'evidence-reuse-manifest.json',dict(execution_head=lock['worktree_head'],execution_tree='352cdd8ca3f3d7b6f2b15f3ed6a6f775fb478443',lock_sha256=sha(attempt/'execution.lock.json'),archive_sha256=sha(attempt/'execution-source.tar'),model_revision=lock['model_revision'],sample_root=lock['sample_root'],blue_head=lock['blue_head'],runtime_members=len(identities),new_member_rehash=sum(r['verification']=='NEW_REHASH' for r in identities),old_asset_rehash_reused=sum(r['verification'].startswith('REUSED') for r in identities),output_members=len(inventory),output_bytes=sum(r['bytes'] for r in inventory),new_gpu_forward=0,broadcast='NO_BROADCAST_NOT_REQUIRED',prior_initial_observation='PENDING_NOT_YET_RUN',current_stored_initial=load(root/'INITIAL_VALID.json')['status'],process_restore=load(root/'process-restore.json'),claim_decision='PENDING_GH_REVIEW'))
    # Native stdout prints one 'Computing right vector' segment per request-z.
    lines=(attempt/'logs/46451.out').read_text().splitlines();segments=[];current=None
    for line in lines:
        if line.startswith('Computing right vector'):
            if current is not None:segments.append(current)
            current=0
        elif line.startswith('loss ') and current is not None:current+=1
    if current is not None:segments.append(current)
    assert len(segments)==400 and all(1<=n<=25 for n in segments),(len(segments),segments[:5])
    zrows=[]
    for i,n in enumerate(segments):zrows.append(dict(fit=['N4','FULL8','RES8','REFIT4'][i//100],request_position=i%100,loss_evaluations=n,adam_updates=n-1))
    csvwrite(out/'z-iterations.csv',zrows)
    fits={a:load(root/a/'fit.json') for a in ['N4','FULL8','RES8','REFIT4']}
    cost=[]
    for a in ARMS:
        e=terminal['endpoints'][a];f=fits.get(a,{})
        cost.append(dict(arm=a,policy_instrumented_online_seconds=e['policy_instrumented_online_seconds'],online_ratio_to_N4=e['policy_instrumented_online_seconds']/terminal['endpoints']['N4']['policy_instrumented_online_seconds'],native_shared_fit_seconds=terminal['study_shared_native_fit_seconds'],second_fit_seconds=f.get('synchronized_wall_seconds',0),materialization_seconds=e['materialization_seconds'],finalization_seconds=e['finalization_seconds'],evaluation_seconds=load(root/a/'evaluation.json')['seconds'],policy_z_requests=100+(100 if a in ['FULL8','RES8','REFIT4'] else 0),policy_M8_setup_seconds=terminal['preparation']['M8_seconds'] if a in ['FULL8','RES8'] else 0,pure_writer_seconds='NOT_SEPARATED'))
    csvwrite(out/'compute-ledger.csv',cost)
    csvwrite(out/'fit-cost.csv',[dict(arm=a,**{k:f[k] for k in ['compute_z','compute_z_seconds','compute_ks','compute_ks_seconds','solve','solve_seconds','history_append','seconds','actual_delta_norm']},loss_evaluations=sum(z['loss_evaluations'] for z in zrows if z['fit']==a),adam_updates=sum(z['adam_updates'] for z in zrows if z['fit']==a)) for a,f in fits.items()])
    write(out/'accounting.json',dict(scheduler_state='COMPLETED',exit_code='0:0',elapsed_seconds=2132,allocated_gpus=1,allocated_GPUh=2132/3600,scheduler_start='2026-09-13T19:22:12',scheduler_end='2026-09-13T19:57:44',host_max_rss='52.27G',resource='1GPU/8CPU/60416M/12h',**{k:v for k,v in terminal.items() if k not in ['endpoints']},request_z_observed=sum(f['compute_z'] for f in fits.values()),loss_evaluations=sum(segments),adam_updates=sum(n-1 for n in segments),output_bytes=sum(r['bytes'] for r in inventory),I_O_restore_P_diagnostic_seconds='NOT_SEPARATELY_RECORDED',prior_estimate_GPUh=[2,8],prior_estimate_GiB=[20,40]))
    print(json.dumps({'output_members':len(inventory),'bytes':sum(r['bytes'] for r in inventory),'request_z':len(segments),'loss_evaluations':sum(segments),'adam_updates':sum(n-1 for n in segments)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.attempt,a.out)
