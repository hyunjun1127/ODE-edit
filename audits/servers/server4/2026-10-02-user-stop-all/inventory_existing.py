"""기존 worktree의 미게시 compact 산출물 byte 보존. 실행/평가 없음."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[4]
AUDIT=Path(__file__).resolve().parent

def git(*args, cwd=ROOT):
    return subprocess.check_output(['git','-C',str(cwd),*args],text=True)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    entries=[]; worktrees=[]
    for chunk in git('worktree','list','--porcelain').split('\n\n'):
        if not chunk.strip():continue
        p=Path(chunk.splitlines()[0][9:])
        if p in (ROOT,Path('/data/janghj/ODE-edit')):continue
        lines=git('status','--porcelain','--untracked-files=all',cwd=p).splitlines()
        worktrees.append(dict(path=str(p),head=git('rev-parse','HEAD',cwd=p).strip(),
            original_deletions_untouched=sum(x[:2].strip()=='D' for x in lines)))
        for line in lines:
            if line[:2].strip()=='D':continue
            rel=line[3:]; f=p/rel; d=ROOT/rel
            if '__pycache__' in rel or not f.is_file():continue
            allowed=rel.startswith(('experiment-reports/servers/server4/',
                'audits/servers/server4/','messages/acks/server4/','messages/server-heads/server4/'))
            allowed |= rel.startswith('tasks/status/') and ('server4' in rel)
            allowed |= rel.startswith(('project/run_scripts/temporal_routing_diagnostic/',
                'project/run_scripts/single_layer_edit_preserving_correction/',
                'project/run_scripts/single_layer_mechanism_first/'))
            if not allowed:continue
            n=f.stat().st_size
            row=dict(source=str(f),relative=rel,bytes=n,sha256=sha(f))
            if d.exists():
                row['main_sha256']=sha(d)
                row['action']='ALREADY_MAIN_EXACT' if row['main_sha256']==row['sha256'] else 'KEEP_MAIN_AND_LOCAL_VARIANT_NO_OVERWRITE'
            elif n>524288 or f.suffix not in ('.md','.py','.json','.csv','.png','.svg'):
                row['action']='LOCAL_KEEP_LARGE_OR_RAW_NOT_GIT'
            elif git('log','-1','--format=%H','origin/main','--',rel).strip():
                row['action']='KEEP_LOCAL_PRIOR_MAIN_PATH_HISTORY_NO_REINTRODUCTION'
            else:
                # 기존 compact 파일만 동일 bytes 복사. 원본 KEEP, 원 main 변경0.
                d.parent.mkdir(parents=True,exist_ok=True)
                assert not d.exists()
                shutil.copyfile(f,d)
                assert sha(d)==row['sha256']
                row['action']='PUBLISHED_EXISTING_BYTES_NO_NEW_EXECUTION'
            entries.append(row)
    doc=dict(scope='등록된 local worktree의 기존 own compact/source와 main 대조; 과거 raw 전수 재분석 없음',
        worktrees=worktrees,members=entries,raw_and_user_deletions_preserved=True)
    out=AUDIT/'existing-output-publication-inventory.json'
    with out.open('x') as s:json.dump(doc,s,ensure_ascii=False,indent=2);s.write('\n')
    from collections import Counter
    print(json.dumps({'actions':dict(Counter(r['action'] for r in entries)),
        'published_bytes':sum(r['bytes'] for r in entries if r['action'].startswith('PUBLISHED')),
        'kept_variants':[r['relative'] for r in entries if 'VARIANT' in r['action']]},ensure_ascii=False))

if __name__=='__main__':main()
