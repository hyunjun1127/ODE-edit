"""Read-only bounded review admission; creates new local evidence only."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/data/janghj/ODE-edit')
LOCAL = ROOT / 'local/server4-completed-review/20260916-v1'
WT = Path(__file__).resolve().parents[3]
TASK = 'ODEEDIT-S06-SERVER4-COMPLETED-CAKE-CAP-DETAILED-REVIEW-SH4-V1'

def digest(p):
    p = Path(p)
    return {'path': str(p), 'bytes': p.stat().st_size,
            'sha256': hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()}

def save(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2); f.write('\n')

def query(args, cwd=WT):
    r = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    return {'args': args, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}

def main():
    LOCAL.mkdir(parents=True, exist_ok=False)
    docs = []
    for rel in ['messages/head/2026-09-16-sh4-completed-cake-cap-review.md',
                'plans/global/2026-09-16-server4-completed-cake-cap-review.json',
                'audits/global/2026-09-16-sh4-completed-cake-cap-review-dispatch-checks.json']:
        p = WT / rel
        target = LOCAL / 'authoritative' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f: f.write(p.read_bytes())
        docs.append(dict(digest(p), reading='FULL_READ_THIS_RECALL'))
    sweep = ROOT / 'local/ep-tw1-alpha-cap-sweep/20260915-v1/full-read-receipt.json'
    for member in json.loads(sweep.read_text())['documents']:
        old = Path(member['path'])
        prefix = ROOT / 'local/worktrees/server4-ep-tw1-alpha-cap-sweep-v1'
        p = WT / old.relative_to(prefix)
        actual = digest(p)
        assert actual['sha256'] == member['sha256'], p
        docs.append(dict(actual, reading='EXACT_PRIOR_FULL_READ_REUSE'))
    cake_read = WT / 'audits/servers/server4/2026-09-15-cake-native-lifelong/full-read.json'
    for member in json.loads(cake_read.read_text()):
        p = Path(member['path']); actual = digest(p)
        assert actual['sha256'] == member['sha256'], p
        docs.append(dict(actual, reading='EXACT_PRIOR_CAKE_FULL_READ_REUSE'))
    pause = ROOT / 'local/ep-tw1-alpha-cap-sweep/20260915-v1/pause-user-20260915-r1.json'
    with (LOCAL / 'authoritative/pause-user-20260915-r1.json').open('xb') as f:
        f.write(pause.read_bytes())
    receipt = {'instruction_id': TASK, 'time': dt.datetime.now(dt.timezone.utc).isoformat(),
               'host': query(['hostname'])['stdout'].strip(), 'worktree': str(WT),
               'analysis_base': query(['git','rev-parse','HEAD','HEAD^{tree}']),
               'shared_root': query(['git','status','--short'], ROOT),
               'old_sweep_dirty_preserved': query(['git','status','--short'], ROOT / 'local/worktrees/server4-ep-tw1-alpha-cap-sweep-v1'),
               'documents': docs, 'prior_receipts': [digest(sweep), digest(cake_read)],
               'pause_full_read': digest(pause), 'gpu_allowed': False, 'other_task_resume': False}
    assert receipt['host'] == 'server4'
    save(LOCAL / 'receipts/full-read.json', receipt)
    # Exactly one scheduler snapshot: no other jobs, no monitor or repeat loop.
    sched = {'instruction_id': TASK, 'time': dt.datetime.now(dt.timezone.utc).isoformat(),
             'jobs': [48101,48148,48149,48150], 'no_repeat_polling': True}
    sched['squeue'] = query(['squeue','-h','-j','48101,48148,48149,48150','-o','%i|%j|%T|%R'])
    sched['sacct'] = query(['sacct','-n','-P','-j','48101,48148,48149,48150',
        '--format=JobIDRaw,JobName%48,User,State,ExitCode,Submit,Start,End,ElapsedRaw,AllocTRES%160,MaxRSS,MaxVMSize,ReqMem,NodeList'])
    save(LOCAL / 'receipts/scheduler-terminal-once.json', sched)
    print(json.dumps(sched, ensure_ascii=False, indent=2))
    print(json.dumps(digest(LOCAL / 'receipts/full-read.json')))

if __name__ == '__main__': main()
