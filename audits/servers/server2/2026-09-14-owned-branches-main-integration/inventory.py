"""Git-only SH2 ownership/ancestry/file inventory; never reads experiment raw."""
import csv,json,hashlib,re,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
START='7d5bae2e3be8dba87c92b86a727d5de9ed549af3'
def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
def run(*a):return subprocess.run(['git',*a],cwd=ROOT,text=True,capture_output=True)
def write(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def snapshot():
    refs=git('for-each-ref','--format=%(refname:short)','refs/heads','refs/remotes/origin').splitlines()
    refs=[r for r in refs if (r.startswith('codex/') or r.startswith('origin/codex/server2-') or r=='origin/codex/p1r52-joint-pc-c1-c2-tech-r6-transfer-v1') and 'owned-branches-main-integration-20260914-v1' not in r]
    result=[];files=[]
    for ref in refs:
        h=git('rev-parse',ref);base=git('merge-base',START,h);ancestor=run('merge-base','--is-ancestor',h,START).returncode==0
        creation=[l.split(' ',1)[0] for l in run('reflog','show','--format=%H %gs',ref).stdout.splitlines() if 'branch: Created from' in l]
        original=creation[-1] if creation else None
        paths=git('diff','--name-only',original or base,h).splitlines()
        if not paths:paths=git('diff-tree','--no-commit-id','--name-only','-r',h).splitlines()
        branch_changes=git('diff','--name-only',base,h).splitlines() if not ancestor else []
        diffs=[]
        for p in paths:
            same=run('diff','--quiet',START,h,'--',p).returncode==0
            exists=run('cat-file','-e',START+':'+p).returncode==0
            files.append(dict(branch=ref,path=p,at_start='EXACT_SAME' if same else 'DIFFERENT' if exists else 'ABSENT',branch_blob=run('rev-parse',h+':'+p).stdout.strip(),main_blob=run('rev-parse',START+':'+p).stdout.strip() if exists else None))
        for p in branch_changes:
            if run('diff','--quiet',START,h,'--',p).returncode:diffs.append(p)
        merge=run('merge-tree',base,START,h).stdout if diffs else ''
        conflicts=[];path=None
        for l in merge.splitlines():
            if re.match(r'  (base|our|their)\s+\d+ ',l):path=l.split()[-1]
            if l.startswith('+<<<<<<<') and path not in conflicts:conflicts.append(path)
        tasks=set()
        for p in paths:
            if p.endswith(('.md','.py','.json')) and any(s in p for s in ('server2','track_b','numerical_lock','README','contracts.py')):
                b=run('show',h+':'+p).stdout
                tasks.update(re.findall(r'ODEEDIT-[A-Z0-9][A-Z0-9_-]{8,}',b))
        result.append(dict(branch=ref,head=h,tree=git('rev-parse',h+'^{tree}'),owner='SH2',owner_evidence=git('log','-1','--format=%an <%ae>',h),task_ids=sorted(tasks),original_base=original or 'REFLOG_NOT_RECORDED',comparison_base=base,ancestor_at_start=ancestor,changed_files=paths,branch_delta_files=branch_changes,unmatched_at_start=diffs,conflict_paths=conflicts,cherry=run('cherry',START,h).stdout.splitlines() if not ancestor else []))
    write('initial-inventory.json',dict(main_head=START,main_tree=git('rev-parse',START+'^{tree}'),branches=result));write('file-comparison.json',files)
    work=[]
    for chunk in git('worktree','list','--porcelain').split('\n\n'):
        d=dict(l.split(' ',1) for l in chunk.splitlines() if ' ' in l);p=d.get('worktree')
        if p and p!=str(ROOT):
            s=subprocess.check_output(['git','-C',p,'status','--porcelain'],text=True)
            work.append(dict(path=p,head=d['HEAD'],branch=d.get('branch','DETACHED'),status=s.splitlines()))
    write('preserved-worktrees-before.json',work)
    read=[]
    for p in ('PROTOCOL.md','messages/head/2026-09-14-all-sh-owned-branches-main-integration.md'):
        b=(ROOT/p).read_bytes();read.append(dict(path=p,bytes=len(b),lines=b.count(b'\n'),sha256=sha(b),full_read=True))
    write('read-boundary-receipt.json',dict(status='FULL_READ_PASS',host=git('config','agent.hostname'),session='01a0493a-074c-7f91-9a13-769116326fef',cwd=str(ROOT),remote=git('remote','get-url','origin'),members=read))
    for r in result:
        if not r['ancestor_at_start']:print(r['branch'],len(r['unmatched_at_start']),'conflicts',r['conflict_paths'])
if __name__=='__main__':snapshot()
