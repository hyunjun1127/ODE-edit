"""One-shot exact resource-only adjustment; no submission or cancellation."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from official.runners.server1.submit import metadata, dependencies
from official.runners.server2.submit import inventory

OUT = Path(__file__).parent
BASE = Path('/mnt/raid5/janghj/ODE-edit/local')
ADD = {'62532': '62864', '62869': '62864', '62866': '62538'}
ROOTS = {'62531', '62864', '62865'}
PENDING = ['62876','62875','62874','62873','62872','62871','62870','62868','62867','62869','62866','62538','62532']

def write(name, obj):
    p = OUT / name
    assert not p.exists(), p
    p.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')

def cmd(*args):
    return subprocess.check_output(args, text=True, timeout=45)

def show(j):
    return metadata(cmd('scontrol','show','job','-o',j))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def proof(rows):
    parents = {j:{p for _,p in dependencies(v['Dependency'])} for j,v in rows.items() if j!='62877'}
    for j,p in ADD.items(): parents[j].add(p)
    assert all(ps <= parents.keys() for ps in parents.values())
    done=set()
    while len(done)<len(parents):
        ready={j for j,ps in parents.items() if j not in done and ps<=done}
        assert ready, 'cycle'
        done |= ready
    # Explore every finish/start order. Only the initial grandfathered allocation may exceed two.
    states={(frozenset(),frozenset(ROOTS))}; seen=set()
    while states:
        completed,running=states.pop()
        if (completed,running) in seen: continue
        seen.add((completed,running))
        ready={j for j,ps in parents.items() if j not in completed|running and ps<=completed}
        if len(running)>=2: assert not ready, (completed,running,ready)
        for j in ready: states.add((completed,running|{j}))
        for j in running: states.add((completed|{j},running-{j}))
    return dict(cycle=False, reachable_states=len(seen), future_allocation_limit=2,
                grandfathered_initial_allocation=3, new_start_above_cap=False,
                lanes=[['62532','62538','62866','62867','62868','62870','62873'],
                       ['62869','62871','62872','62874','62875','62876']])

def main():
    expected={}; pinned={}; sources={}
    def pin(member):
        p=Path(member['path']); assert p.is_file() and not p.is_symlink()
        assert p.stat().st_size==member['bytes'] and sha(p)==member['sha256'], p
        pinned[str(p)]=member['sha256']
    for task in ['baseline-refresh-s2-flucon-20261010','fe-author-hparams-2k-20261010']:
        root=BASE/task/'registration-r1'; receipt=json.loads((root/'submission.json').read_text())
        pin(receipt['execution_lock']); lock=json.loads(Path(receipt['execution_lock']['path']).read_text())
        sources[task]=receipt['source']
        for m in lock['source_members']: pin(m)
        for key,r in receipt['jobs'].items():
            pin(r['script'])
            if r.get('config'): pin(r['config'])
            expected[str(r['job_id'])]=dict(name=r['name'],command=r['script']['path'],workdir=str(root/'source'))
    task='qwen-zsre-sphere-b9-resume-20261010'; root=BASE/task/'registration-r1'
    r=json.loads((BASE/task/'submission-complete.json').read_text()); sources[task]=r['source']
    expected['62538']=dict(name=r['job_name'],command=str(root/'scripts/main.sh'),workdir=str(root/'source'))
    for p in root.rglob('*'):
        if p.is_file() and not p.is_symlink() and (p.suffix in ('.py','.json','.sh')) and ('source' in p.parts or p.parent.name=='scripts' or p.parent==root):
            pinned[str(p)]=sha(p)
    inv=inventory(); assert {r['job'] for r in inv['project']}==ROOTS|set(PENDING)
    before={j:show(j) for j in expected}
    def check(j, pending=False):
        d=show(j); e=expected[j]
        assert d['UserId'].startswith('janghj(') and d['ReqNodeList']=='server2'
        assert (d['JobName'],d['Command'],d['WorkDir'])==(e['name'],e['command'],e['workdir'])
        if pending:
            assert d['JobState']=='PENDING' and d['RunTime']=='00:00:00' and d['StartTime']=='Unknown'
            assert d['AllocTRES']=='(null)'
        return d
    for j in expected: check(j, j in PENDING)
    assert all(before[j]['JobState']=='RUNNING' for j in ROOTS)
    write('before.json',dict(time=datetime.now(timezone.utc).isoformat(), jobs=before, sources=sources, member_hashes=pinned))
    write('proof.json',proof(before))
    operations=[]
    for j in PENDING:
        check(j,True); cmd('scontrol','hold',j); operations.append(['hold',j])
    write('held.json',{j:check(j,True) for j in PENDING})
    for j,p in ADD.items():
        d=check(j,True); pairs=set(dependencies(d['Dependency'])); pairs.add(('afterany',p))
        dep=','.join(t+':'+i for t,i in sorted(pairs))
        cmd('scontrol','update','JobId='+j,'Dependency='+dep)
        assert set(dependencies(show(j)['Dependency']))==pairs
        operations.append(['dependency_union',j,p])
    middle={j:show(j) for j in expected}; proof(middle)
    for p,h in pinned.items(): assert sha(p)==h, p
    immutable=['Command','WorkDir','JobName','ReqTRES','TimeLimit','QOS','ReqNodeList','CPUs/Task']
    for j,d in middle.items():
        assert all(d[k]==before[j][k] for k in immutable), j
        assert set(dependencies(before[j]['Dependency']))<=set(dependencies(d['Dependency']))
    for j in reversed(PENDING):
        check(j,True); cmd('scontrol','release',j); operations.append(['release',j])
    after={j:show(j) for j in expected}
    assert all(d['Reason'] not in ('JobHeldUser','JobHeldAdmin') and d['Priority']!='0' for d in after.values())
    write('after.json',dict(time=datetime.now(timezone.utc).isoformat(),jobs=after,operations=operations,
        sources_unchanged=True,cancel_count=0,submit_count=0,temporary_holds_remaining=0,
        proof=proof(after),accepted_turn='01a12500-fa47-7fc0-85f7-1204a696bdf6'))
    print(json.dumps(dict(status='APPLIED',changes=ADD,proof=proof(after))))

if __name__=='__main__': main()
