"""Create-once FULL_READ identity reuse and one exact-job scheduler audit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/data/janghj/ODE-edit/local/local-z-adaptive-allocation/20260916-v1')
REVIEW = ROOT/'review-20260917-v1'
TASK = 'ODEEDIT-S06-LOCAL-Z-SEVENARM-DETAILED-REVIEW-SH4-V1'
ENV = 'messages/head/2026-09-17-sh4-local-z-detailed-review.md'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def identity(path):
    p=Path(path)
    return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))

def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return identity(p)

def command(args,cwd=None):
    r=subprocess.run(args,cwd=cwd,text=True,capture_output=True)
    return dict(args=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)

def run(w):
    w=Path(w).resolve()
    assert sha(w/ENV)=='26f4f0b485da7ede2602d00f3a483fb41b343d1d1cd69a085815621722689c23'
    prior=json.loads((ROOT/'full-read-receipt.json').read_text());docs=[]
    for m in prior['documents']:
        ref=identity(w/m['repo_path'])
        assert ref['sha256']==m['sha256'] and ref['bytes']==m['bytes'],m['repo_path']
        docs.append(dict(ref,reading='EXACT_PRIOR_FULL_READ_REUSED',prior_path=m['path']))
    pinned={'execution.lock.json':'093bb13dd6b42b8f2f8b8478248b067add1d94533547afa8ca9af416e6f14629',
        'resume-manifest.json':'541012971022dd245df9aa64d085379d931116e7eb4dad2eba085a31e0920ebe',
        'source-v1.tar':'6378df3a237d5a6c489b87c90f6d99c0221b409857632852916f2f5bffa7f901'}
    sources=[]
    for name,h in pinned.items():
        ref=identity(ROOT/name);assert ref['sha256']==h;sources.append(ref)
    native=[]
    for m in prior['native_sources']:
        ref=identity(m['path']);assert ref['sha256']==m['sha256'];native.append(ref)
    receipt=save(REVIEW/'full-read-m0.json',dict(instruction_id=TASK,envelope=identity(w/ENV),
        envelope_reading='FULL_READ_THIS_RECALL',prior_full_read=identity(ROOT/'full-read-receipt.json'),
        documents=docs,native_reuse=native,pins=sources,host=command(['hostname']),
        main=command(['git','rev-parse','origin/main','origin/main^{tree}'],w),
        worktree=command(['git','status','--short'],w),shared_root=command(['git','status','--short'],'/data/janghj/ODE-edit'),
        GPU_actions=0,source_runtime_modified=False,prior_observation='2026-09-16T11:36:50.122707+00:00 PENDING/GATE_NOT_OBSERVED'))
    audit_path=REVIEW/'scheduler-once.json'
    if audit_path.exists():raise FileExistsError('EXACT_JOB_AUDIT_ALREADY_PERFORMED_NO_REPOLL')
    audit=dict(time=datetime.now(timezone.utc).isoformat(),targets=['48679','48680_[0-6]'],
        accounting=command(['sacct','-n','-P','-j','48679,48680','--format=JobIDRaw,JobID,JobName%40,User,State,ExitCode,DerivedExitCode,Start,End,ElapsedRaw,AllocTRES%120,ReqTRES%120,NodeList,MaxRSS,Submit,Eligible']),
        live=command(['squeue','-h','-j','48679,48680','-o','%i|%j|%T|%b|%R']),
        detail=command(['scontrol','show','job','48679,48680','-o']),repeat_polling=False,mutation=False)
    save(audit_path,audit)
    print(json.dumps(dict(full_read=receipt,scheduler=audit),ensure_ascii=False,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);a=p.parse_args();run(a.worktree)
