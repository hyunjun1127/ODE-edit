"""Exact EN-only cancellation and identity-checked checkpoint tombstones.

No wildcard cancellation, recursive deletion, symlink traversal or raw cleanup.
Deletion is a separate explicit command consuming the sealed inventory.
"""
import argparse
import json
import os
from pathlib import Path
import pwd
import re
import stat
import subprocess
from .provenance import ROOT, create_json, now, sha

EN = Path('/data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1')
OUT = ROOT / 'en-cleanup'
JOBS = ('50071_3', '50071_0', '50071_2', '50071_1')
SCRIPT = EN / 'S/attempt-v1/source/project/run_scripts/single_layer_edit_preserving_correction/sequential.sbatch'
ARM = {'50071_0':'EN-S','50071_1':'EN-F','50071_2':'EN-COV','50071_3':'EN-F4'}

def run(args):
    p = subprocess.run(args, text=True, capture_output=True)
    return dict(args=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr, time_utc=now())

def inspect_jobs():
    results=[]
    for job in JOBS:
        r=run(['scontrol','show','job',job,'-o'])
        fields=dict(re.findall(r'(\w+)=([^ ]*)',r['stdout']))
        if r['returncode'] == 0:
            if (fields.get('UserId') != f'janghj({os.getuid()})' or
                fields.get('JobName') != 'odeedit_enfc_S4_s4' or fields.get('ArrayJobId') != '50071' or
                fields.get('ArrayTaskId') != job.split('_')[1] or fields.get('ReqNodeList') != 'server4' or
                fields.get('Command') != str(SCRIPT)):
                raise ValueError(f'EXACT_EN_MAPPING_MISMATCH:{job}:{fields}')
        elif 'not found' not in r['stdout']+r['stderr']:
            raise RuntimeError(r)
        results.append(dict(job=job,arm=ARM[job],fields=fields,observation=r))
    return results

def cancel():
    before=inspect_jobs()
    create_json(OUT/'jobs-before-cancel.json', dict(time_utc=now(),jobs=before))
    actions=[]
    for row in before:
        if row['fields'].get('JobState') in ('RUNNING','PENDING'):
            # Re-resolve identity immediately before the exact mutation.
            current=next(x for x in inspect_jobs() if x['job']==row['job'])
            if current['fields'].get('JobState') in ('RUNNING','PENDING'):
                r=run(['scancel',row['job']]);actions.append(r)
                if r['returncode']:raise RuntimeError(r)
    create_json(OUT/'cancel-requests.json',dict(time_utc=now(),actions=actions,completed_job_not_cancelled='50071_1'))
    print(json.dumps(actions,ensure_ascii=False))

def verify():
    observed=inspect_jobs()
    accounting=run(['sacct','-X','-j','50071','--array','--format=JobIDRaw,JobID,JobName%32,User,State,ExitCode,NodeList,AllocTRES%70,ElapsedRaw,Start,End','-P'])
    if accounting['returncode']:raise RuntimeError(accounting)
    if any(x['fields'].get('JobState') in ('RUNNING','PENDING','COMPLETING','CONFIGURING','SUSPENDED') for x in observed):
        raise RuntimeError('EN_ALLOCATION_NOT_RELEASED')
    expected=set(JOBS)
    lines=accounting['stdout'].splitlines()
    for line in lines[1:]:
        cols=line.split('|')
        if len(cols)>4 and cols[1] in expected and cols[4].split()[0] in ('COMPLETED','CANCELLED','FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'):
            expected.remove(cols[1])
    if expected:raise RuntimeError(f'EN_TERMINAL_ACCOUNTING_MISSING:{expected}')
    receipt=dict(time_utc=now(),jobs=observed,accounting=accounting,allocation_released=True,
                 process_rollback='NOT_VERIFIED',scientific_resume='NOT_REQUESTED')
    create_json(OUT/'jobs-terminal.json',receipt)
    print(accounting['stdout'])

def identity(p):
    s=p.lstat()
    if not stat.S_ISREG(s.st_mode) or p.resolve()!=p or s.st_uid!=os.getuid():
        raise ValueError(f'NOT_OWNED_NONSYMLINK_REGULAR:{p}')
    return dict(path=str(p),realpath=str(p.resolve()),owner=pwd.getpwuid(s.st_uid).pw_name,
                uid=s.st_uid,dev=s.st_dev,ino=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,
                nlink=s.st_nlink,blocks=s.st_blocks,mode=s.st_mode)

def inventory():
    terminal=json.loads((OUT/'jobs-terminal.json').read_text())
    assert terminal['allocation_released']
    rows=[]
    # os.walk never follows directory symlinks; local aliases are not traversed.
    for directory, dirs, files in os.walk(EN,followlinks=False):
        dirs[:]=[d for d in dirs if not Path(directory,d).is_symlink() and d not in ('.git','source','worktree','integration')]
        for name in files:
            p=Path(directory,name)
            if p.is_symlink() or not (p.suffix in ('.pt','.pth','.safetensors','.bin') or '.pt.partial-' in name):continue
            r=identity(p);rel=str(p.relative_to(EN));r['relative']=rel
            target=(name in ('checkpoint.pt','selected-L4.pt','final-L4.pt') or
                    name.startswith('checkpoint.pt.partial-') or name.startswith('selected-L4.pt.partial-') or name.startswith('final-L4.pt.partial-'))
            # Full tensors are classified by known publisher filenames AND scope.
            target=target and (rel.startswith('S/attempt-v1/arms/') or rel.startswith('M/') or rel.startswith('T/'))
            r.update(action='DELETE_EN_CHECKPOINT' if target else 'PRESERVE',
                     classification=('EN_W_M_RNG_RESUME' if name.startswith('checkpoint.pt') else 'EN_FINAL_ENDPOINT') if target else
                     ('MIXED_NATIVE_CAPSULE_TARGETS_KEYS_PRESERVE' if name=='native-capsule.pt' else 'NOT_SELECTED_RESUME_OR_ENDPOINT'),
                     in_use=False, in_use_basis='exact EN jobs terminal/allocation released; no resume requested',
                     sha256=sha(p) if target else None,
                     preserved_hash_policy=None if target else 'NO_REDUNDANT_REHASH_REQUIRED')
            rows.append(r)
    deletes=[r for r in rows if r['action']=='DELETE_EN_CHECKPOINT']
    create_json(OUT/'checkpoint-inventory.json',dict(time_utc=now(),root=str(EN),members=rows,
                delete_files=len(deletes),delete_logical_bytes=sum(r['bytes'] for r in deletes),
                symlink_policy='never followed',aliases_dedup='realpath/dev/inode recorded; walk excludes worktree/source aliases',
                external_same_EN_paths='none identified outside authoritative EN root; common baseline capsules preserved',
                deleted_recovery='no backup created; no known other copy verified'))
    print(json.dumps(dict(files=len(rows),delete_files=len(deletes),delete_bytes=sum(r['bytes'] for r in deletes)),indent=2))

def delete():
    inspect=inspect_jobs()
    if any(r['fields'].get('JobState') in ('RUNNING','PENDING','COMPLETING','CONFIGURING') for r in inspect):
        raise RuntimeError('EN_STILL_IN_USE')
    manifest=OUT/'checkpoint-inventory.json'
    inv=json.loads(manifest.read_text())
    candidates=[r for r in inv['members'] if r['action']=='DELETE_EN_CHECKPOINT']
    # All identities checked before the first unlink and again per file.
    for r in candidates:
        current=identity(Path(r['path']))
        if any(current[k]!=r[k] for k in current):raise RuntimeError(f'DELETE_IDENTITY_CHANGED:{r["path"]}')
    before=os.statvfs(EN);actions=[]
    log=OUT/'unlink-actions.jsonl'
    with log.open('x') as f:
        for r in candidates:
            p=Path(r['path']);current=identity(p)
            if any(current[k]!=r[k] for k in current):raise RuntimeError(f'DELETE_IDENTITY_CHANGED:{p}')
            p.unlink()
            action=dict(path=str(p),sha256=r['sha256'],bytes=r['bytes'],nlink_before=r['nlink'],
                        time_utc=now(),operation='PERMANENT_UNLINK_EXACT_FILE',absent=not p.exists())
            f.write(json.dumps(action)+'\n');f.flush();os.fsync(f.fileno());actions.append(action)
    after=os.statvfs(EN)
    result=dict(time_utc=now(),inventory_sha256=sha(manifest),removed_files=len(actions),
                logical_bytes=sum(r['bytes'] for r in actions),
                unique_single_link_allocated_bytes=sum(r['blocks']*512 for r in candidates if r['nlink']==1),
                shared_hardlinks=[r['path'] for r in candidates if r['nlink']>1],
                available_bytes_before=before.f_bavail*before.f_frsize,available_bytes_after=after.f_bavail*after.f_frsize,
                observed_free_delta=(after.f_bavail*after.f_frsize)-(before.f_bavail*before.f_frsize),
                free_delta_exclusive_attribution=False,known_recoverable_copy='NOT_VERIFIED',
                EN_exact_checkpoint_resume='UNAVAILABLE_AFTER_USER_DIRECTED_REMOVAL',
                preserved='all evaluation/generation/raw/log/source/receipt; shared model/tokenizer/data/P/stats/context/reference/teacher; native target/key/mixed capsules; gradients/geometry/ideal-delta not model checkpoints')
    create_json(OUT/'removal-receipt.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('cancel','verify','inventory','delete'))
    globals()[p.parse_args().action]()
