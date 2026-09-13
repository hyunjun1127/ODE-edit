"""Bounded CPU verification of Git publication bytes, not experiment reruns."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import py_compile
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parent
BASE='7d5bae2e3be8dba87c92b86a727d5de9ed549af3'
SOURCE='b3b40db422bdb4d4e19ed04fca535b9d903ce8ec'
def git(*a):return subprocess.check_output(['git',*a],text=True).strip()
def blob(path):return subprocess.check_output(['git','show',SOURCE+':'+str(path)])
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    paths=git('diff','--name-only',BASE,SOURCE).splitlines()
    forbidden=[];scope=[];members=[];fields=[];reference_ok=[];reference_bad=[]
    allowed=('project/run_scripts/','experiment-reports/servers/server1/','audits/servers/server1/',
             'messages/acks/server1/','messages/server-heads/server1/','plans/updates/server1/',
             'runs/baseline-mechanism-first-e01-sh1-20260912-v1/',
             'transfers/verifications/2026-09-12-baseline-mechanism-first-e01/')
    py=[];shell=[];seen=set()
    def verify_refs(x,file):
        if isinstance(x,dict):
            p=x.get('path');h=x.get('sha256')
            if isinstance(p,str) and isinstance(h,str) and len(h)==64:
                rel=p[p.index('experiment-reports/'):] if 'experiment-reports/' in p else str(file.parent/p)
                q=Path(rel)
                if q.exists() and str(q).startswith(('experiment-reports/','audits/')) and (str(file),str(q)) not in seen:
                    seen.add((str(file),str(q)));a=sha(blob(q))
                    (reference_ok if a==h else reference_bad).append({'manifest':str(file),'member':str(q),'expected':h,'actual':a})
            for k,v in x.items():
                if k.lower() in ('prompt','target_new','target_true','text','generation','subject') and isinstance(v,str) and v:
                    fields.append({'file':str(file),'key':k,'category_only':v in ('rewrite','rephrase','neighborhood')})
                verify_refs(v,file)
        elif isinstance(x,list):
            for v in x:verify_refs(v,file)
    for name in paths:
        p=Path(name);b=blob(name)
        if not name.startswith(allowed):scope.append(name)
        if p.is_symlink() or p.suffix.lower() in ('.pt','.pth','.npy','.npz','.pkl','.safetensors','.bin','.log'):
            forbidden.append(name)
        if b'-----BEGIN PRIVATE KEY-----' in b or b'-----BEGIN OPENSSH PRIVATE KEY-----' in b:
            forbidden.append(name)
        members.append({'path':name,'bytes':len(b),'sha256':sha(b),'git_blob':git('rev-parse',SOURCE+':'+name),'mode':git('ls-tree',SOURCE,'--',name).split()[0]})
        if p.suffix=='.py':ast.parse(b,filename=name);py.append(name)
        if name.endswith(('.sh','.sbatch')):subprocess.run(['bash','-n'],input=b,check=True);shell.append(name)
        if p.suffix=='.json':verify_refs(json.loads(b),p)
        if p.suffix=='.csv':
            import io
            with io.StringIO(b.decode(),newline='') as f:
                rows=csv.reader(f);head=next(rows);width=len(head)
                for row in rows:assert len(row)==width,(name,'CSV_WIDTH')
                assert not set(head)&{'raw_prompt','prompt_text','generated_text','target_new','target_true','subject'},name
    with tempfile.TemporaryDirectory(prefix='sh1-integration-pycompile-') as td:
        for i,name in enumerate(py):
            src=Path(td)/f'{i}.py';src.write_bytes(blob(name))
            py_compile.compile(str(src),cfile=str(Path(td)/f'{i}.pyc'),doraise=True)
    assert not forbidden and not scope and not reference_bad,(forbidden,scope,reference_bad)
    assert all(x['category_only'] for x in fields),fields
    whitespace=subprocess.run(['git','-c','core.whitespace=cr-at-eol,-blank-at-eol','diff',BASE,SOURCE,'--check'],capture_output=True,text=True)
    assert whitespace.returncode==0,whitespace.stdout
    payload={'status':'PASS','publication_source_head':SOURCE, 'base_main':BASE,
             'members':members,'member_count':len(members),'total_bytes':sum(x['bytes'] for x in members),
             'members_root':sha(json.dumps(members,sort_keys=True,separators=(',',':')).encode()),
             'ast_py_compile_count':len(py),'bash_n_count':len(shell),'scope_violations':scope,'forbidden_files':forbidden,
             'manifest_report_reference_checks':reference_ok,'manifest_mismatches':reference_bad,
             'prompt_named_fields_category_only':len(fields),'raw_prompt_publication_count':0,
             'format_exceptions':'기존 CSV CRLF 및 Markdown hardbreak 보존. Git mode100644/100755는 원 실행0600 보증이 아님.',
             'model_gpu_scheduler_transfer_actions':0,'artifact_broadcast':'NO_BROADCAST_NOT_REQUIRED'}
    (ROOT/'publication-checks.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('members','manifest_report_reference_checks')},ensure_ascii=False))
if __name__=='__main__':main()
