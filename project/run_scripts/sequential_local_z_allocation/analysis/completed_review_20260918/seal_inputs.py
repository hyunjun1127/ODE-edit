"""Bind old read receipts, current lightweight closure and immutable artifacts."""
import os,subprocess,datetime
from reducer import *
def main():
    lock=load(ROOT/'execution.lock.json');members=[]
    for m in lock['members']:
        p=Path(m['path']);s=p.stat();r=dict(m,current_size=s.st_size,current_stat=[s.st_dev,s.st_ino,s.st_mtime_ns])
        if m.get('verification')=='PRIOR_FULL_SHA_STABLE_STAT':
            r['current_validation']='PRIOR_SHA_PLUS_CURRENT_STAT';r['pass']=s.st_size==m['bytes'] and r['current_stat']==m['stat']
        else:r['current_validation']='CURRENT_FULL_SHA';r['pass']=s.st_size==m['bytes'] and sha(p)==m['sha256']
        members.append(r)
    env=dict(host=subprocess.check_output(['hostname'],text=True).strip(),session=os.environ.get('CODEX_THREAD_ID'),analysis_worktree=str(WT),shared_cwd='/data/janghj/ODE-edit',repository='hyunjun1127/ODE-edit',analysis_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip())
    assert env['host']=='server4' and env['session']=='01a04939-b5c7-7a03-ba2d-ef3343d62cfd'
    writejson(REPORT/'input-manifest.json',dict(environment=env,execution_head=lock['source_head'],execution_tree=lock['source_tree'],execution_lock_sha=sha(ROOT/'execution.lock.json'),archive=lock['source_archive'],FULL_READ=load(LOCAL/'full-read-r1.json'),members=members,failures=[r for r in members if not r['pass']],historical_pause=dict(path=str(ROOT/'resume-r1/resume-manifest.json'),sha256=sha(ROOT/'resume-r1/resume-manifest.json')),technical_READY=dict(path=lock['common_ready'],sha256=sha(lock['common_ready'])),reuse_policy='No model/teacher bulk rehash; reuse exact prior SHA and immutable stat',new_gpu=0))
    print('INPUT_CLOSURE',len(members),'FAIL',sum(not r['pass'] for r in members))
if __name__=='__main__':main()
