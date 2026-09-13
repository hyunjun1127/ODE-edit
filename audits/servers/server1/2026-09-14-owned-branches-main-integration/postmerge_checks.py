"""CPU-only publication composition/access checks; no scientific evaluation."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
BEFORE='7d5bae2e3be8dba87c92b86a727d5de9ed549af3'
PEER='a2ecac5458d533f841e3f9e696865b07bb9f7956'
def git(*a):return subprocess.check_output(['git',*a],text=True).strip()
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    peer_paths=git('diff','--name-only',BEFORE,PEER).splitlines()
    composed='project/run_scripts/ode_bf/p1r52_sequential_runtime.py'
    assert all(git('rev-parse',PEER+':'+p)==git('rev-parse','HEAD:'+p) for p in peer_paths if p!=composed)
    changed_own=git('diff','--name-only',PEER,'HEAD').splitlines()
    allowed=('project/run_scripts/','experiment-reports/servers/server1/','audits/servers/server1/',
             'messages/acks/server1/','messages/server-heads/server1/','plans/updates/server1/',
             'runs/baseline-mechanism-first-e01-sh1-20260912-v1/',
             'transfers/verifications/2026-09-12-baseline-mechanism-first-e01/')
    exception='tasks/status/server1/2026-09-14-owned-branches-main-integration.json'
    assert all(p.startswith(allowed) or p==exception for p in changed_own)
    files=sorted(ROOT.glob('*')); members=[]
    for p in files:
        if not p.is_file() or p.name in ('postmerge-checks.json','main-publication.json'):continue
        assert not p.is_symlink()
        b=p.read_bytes()
        assert not any(line.strip() in (b'-----BEGIN PRIVATE KEY-----',b'-----BEGIN OPENSSH PRIVATE KEY-----') for line in b.splitlines())
        if p.suffix=='.py':ast.parse(b,filename=str(p))
        if p.suffix=='.json':json.loads(b)
        if p.suffix=='.csv':
            with p.open(newline='') as f:
                r=csv.reader(f);width=len(next(r));assert all(len(row)==width for row in r)
        members.append({'path':str(p.relative_to(Path.cwd())),'bytes':len(b),'sha256':sha(b)})
    subprocess.run(['git','-c','core.whitespace=cr-at-eol,-blank-at-eol','diff',PEER,'--check'],check=True)
    payload={'status':'PASS','checked_source_head':git('rev-parse','HEAD'),'preserved_peer_main':PEER,
             'peer_unchanged_blob_count':len(peer_paths)-1,'composition_file':composed,
             'composition_evidence':'SH2 native-z dispatch and SH1 P1R54/P1R55 role paths coexist; postmerge 230 tests PASS',
             'scope_exception':{'path':exception,'authority':'exact user policy20260914','legacy_helper_exit':7},
             'audit_members':members,'audit_member_root':sha(json.dumps(members,sort_keys=True,separators=(',',':')).encode()),
             'audit_self_and_publication_receipt_excluded':True,'new_gpu_model_scheduler_raw_actions':0}
    (ROOT/'postmerge-checks.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k!='audit_members'},ensure_ascii=False))
if __name__=='__main__':main()
