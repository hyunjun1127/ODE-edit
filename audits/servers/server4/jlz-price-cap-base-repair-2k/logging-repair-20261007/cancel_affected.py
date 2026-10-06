"""One explicit USER-approved replacement of exact faulty MEMIT/Alpha registrations."""
import hashlib,json,os,re,subprocess
from pathlib import Path
ROOT=Path('/data/janghj/ODE-edit/local')
OUT=ROOT/'jlz-price-cap-base-repair-2k/logging-repair-evidence'
TASKS={'jlz-price-cap-base-repair-2k':range(59931,59938),'jlz-price-alpha-writer-2k':range(59949,59956)}
def cmd(args):
    p=subprocess.run(args,text=True,capture_output=True)
    if p.returncode:raise RuntimeError(p.stderr)
    return p.stdout
def save(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,indent=2)
def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'cancellation.json').exists(),'duplicate cancellation invocation'
    rows=[]
    for task,expected in TASKS.items():
        attempt=ROOT/task/'attempt'; sub=json.loads((attempt/'submission.json').read_text())
        assert set(map(int,sub['jobs'].values()))==set(expected)
        lock=json.loads((attempt/'execution.lock.json').read_text())
        assert hashlib.sha256((attempt/'config.json').read_bytes()).hexdigest()==lock['config_sha256']
        frozen=attempt/'source/project/run_scripts/jlz_interference_l1/cap_tracking.py'
        assert "sum(payload['nll'])/B" in frozen.read_text()
        for role,job in sub['jobs'].items():
            detail=cmd(['scontrol','show','job',job,'-o'])
            assert 'UserId=janghj(1025)' in detail and f'JobName={task}-{role} ' in detail
            assert f'Command={attempt}/{role}.sh ' in detail and f'WorkDir={attempt}/source ' in detail
            rows.append(dict(task=task,role=role,job=job,source=lock['source_commit'],detail=detail,
                state=re.search(r'JobState=(\S+)',detail)[1],fault_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest()))
    save('pre-cancellation.json',rows)
    actions=[]
    # Hold every pending affected successor first, so cancellation cannot launch it.
    for row in rows:
        if row['state']=='PENDING':
            cmd(['scontrol','hold',row['job']]);actions.append({'job':row['job'],'action':'hold'})
    # Collectors first; then Alpha successors and MEMIT successors before roots.
    ordered=sorted(rows,key=lambda r:(r['role']=='collector',int(r['job'])),reverse=True)
    for row in ordered:
        if row['state'] not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'):
            cmd(['scancel',row['job']]);actions.append({'job':row['job'],'action':'cancel'})
    save('cancellation.json',dict(user_authorized=True,scope='faulty MEMIT6+Alpha6 and own collectors',actions=actions,
        old_source_raw_config_preserved=True,no_other_jobs_changed=True))
    print(json.dumps({'bound_jobs':len(rows),'actions':actions}))
if __name__=='__main__':main()
