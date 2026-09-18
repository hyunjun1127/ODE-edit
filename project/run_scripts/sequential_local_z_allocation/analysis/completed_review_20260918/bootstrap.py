"""Create-once, exact-target read-only review admission. No model imports."""
import datetime, hashlib, json, pathlib, subprocess

ROOT = pathlib.Path('/data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2')
LOCAL = ROOT/'completed-review-20260918-v1'
WT = LOCAL/'worktree'
def identity(p):
    b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'lines':b.count(b'\n')}
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(obj,f,ensure_ascii=False,indent=2); f.write('\n')
def main():
    prior=json.loads((ROOT/'full-read-m0.json').read_text())
    docs=[]
    for d in prior['documents']:
        actual=identity(WT/d['relative'])
        assert actual['sha256']==d['sha256'], d['relative']
        docs.append(dict(actual,reading='EXACT_PRIOR_FULL_READ_REUSE'))
    new=['messages/head/2026-09-18-sh4-sequential-local-z-v2-completed-review.md','messages/head/2026-09-17-sh4-slz-v2-main-initial-gate-override.md','plans/global/2026-09-17-sequential-local-z-allocation-sh4-dispatch/completed-review-20260918.json']
    docs += [dict(identity(WT/p),reading='FULL_READ_CURRENT_TURN') for p in new]
    save(LOCAL/'full-read-r1.json',{'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'documents':docs,'prior':identity(ROOT/'full-read-m0.json'),'new_gpu':0,'source':'21297ec19e7f5aecec16d2fdb14cc79380a1df94','publication_base':'3658b7b46aba0b64274bf24d5ae19009352536f6'})
    out=LOCAL/'scheduler-r1.json'
    if out.exists(): raise RuntimeError('Scheduler observation already sealed; do not repeat')
    cmd=['sacct','-j','49466_0,49466_1,49466_2,49466_3,49466_4,49466_5','--parsable2','--noheader','--format=JobID,JobIDRaw,JobName%40,User,State,ExitCode,Start,End,ElapsedRaw,AllocTRES%100,MaxRSS,NodeList,ReqMem']
    r=subprocess.run(cmd,text=True,capture_output=True,timeout=30)
    save(out,{'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'scope':'EXACT_SIX_ONLY_ONCE'})
    print(r.stdout); assert r.returncode==0,r.stderr
if __name__=='__main__': main()
