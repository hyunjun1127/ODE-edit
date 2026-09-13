"""Rehash Git publication only. Never follows raw/model paths in receipts."""
import ast,csv,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
START='7d5bae2e3be8dba87c92b86a727d5de9ed549af3'
RELO=[
 ('d9521fbd','local/odebf/reports/fzcb-hard-alpha-densec-ha0-ha1-numerical-hold-v1',
  'experiment-reports/servers/server2/fzcb-hard-alpha-densec-ha0-ha1-numerical-hold-20260914-publication-v1',
  ['fzcb-hard-alpha-densec-ha0-ha1-hold-factual-ko.md','ha1-hold-summary.json','analysis-manifest.json','rooted-analysis-receipt.json']),
 ('00417f7c','local/odebf/reports/p1r52-joint-pc-c1-c2-pilot-tech-r6-baseline-comparison-v1',
  'experiment-reports/servers/server2/p1r52-joint-pc-tech-r6-baseline-comparison-20260914-publication-v1',
  ['comparison-summary.json','p1r52-joint-pc-c1-c2-pilot-tech-r6-baseline-comparison-ko.md','report-manifest.json','rooted-report-receipt.json'])]
def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a])
def sha(b):return hashlib.sha256(b).hexdigest()
def write(path,x):
    (ROOT/path).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    relocated=[]
    for ref,old,new,names in RELO:
        items=[]
        for n in names:
            original=git('show',ref+':'+old+'/'+n);actual=(ROOT/new/n).read_bytes()
            if original!=actual:raise ValueError('SEALED_BYTES_CHANGED:'+n)
            items.append(dict(original_path=old+'/'+n,published_path=new+'/'+n,bytes=len(actual),sha256=sha(actual),git_blob=git('rev-parse',ref+':'+old+'/'+n).decode().strip()))
        receipt=dict(status='EXACT_GIT_BYTES_RELOCATED',source_commit=git('rev-parse',ref).decode().strip(),members=items,
            historical_paths_and_mode_fields='PRESERVED; original mode0600 is not publication Git100644',
            raw_rehash_this_turn=False,raw_broadcast='NO_BROADCAST_NOT_REQUIRED',
            scientific_reinterpretation=False,canonical_for='historical factual package only, not a new execution or newer family replacement')
        write(new+'/publication-relocation-receipt.json',receipt);relocated.extend(items)
    # Reuse sealed member definitions, do not dereference raw evidence paths.
    report=ROOT/'experiment-reports/servers/server2/multilayer-damage-compensation-b-2026-09-11-v1/partial-recall-r1'
    m=json.loads((report/'analysis-manifest.json').read_text());verified=[]
    for item in m['members']:
        b=(report/item['path']).read_bytes()
        assert len(b)==item['bytes'] and sha(b)==item['sha256'],item['path']
        verified.append(item)
    changes=git('diff','--name-only',START,'HEAD').decode().splitlines()
    changes+=git('diff','--name-only').decode().splitlines()
    changes+=git('ls-files','--others','--exclude-standard').decode().splitlines()
    changes=sorted(set(changes));compiled=[];shell=[];jsons=[];members=[]
    for p in changes:
        f=ROOT/p
        if not f.is_file():raise ValueError('DELETION_NOT_ALLOWED:'+p)
        if not p.startswith(('project/run_scripts/','audits/servers/server2/','experiment-reports/servers/server2/','messages/acks/server2/','messages/server-heads/server2/','tasks/status/server2/')):raise ValueError('SCOPE:'+p)
        b=f.read_bytes()
        if len(b)>2_000_000 or b'\x00' in b or f.suffix not in ('.py','.md','.json','.csv','.sbatch','.sh'):raise ValueError('RAW_FREE_FILE_TYPE:'+p)
        if f.suffix=='.py':compile(b,p,'exec');compiled.append(p)
        if f.suffix in ('.sbatch','.sh'):
            subprocess.run(['bash','-n',str(f)],check=True);shell.append(p)
        if f.suffix=='.json':json.loads(b);jsons.append(p)
        members.append(dict(path=p,bytes=len(b),sha256=sha(b)))
    # Verify original checked-out worktrees, including untracked summaries.
    old=json.loads((OUT/'preserved-worktrees-before.json').read_text());post=[]
    for r in old:
        p=r['path'];head=subprocess.check_output(['git','-C',p,'rev-parse','HEAD'],text=True).strip()
        status=subprocess.check_output(['git','-C',p,'status','--porcelain'],text=True).splitlines()
        same=head==r['head'] and status==r['status']
        post.append(dict(path=p,head=head,status=status,unchanged=same))
    assert all(x['unchanged'] for x in post),'ORIGINAL_WORKTREE_CHANGED'
    write(str(OUT.relative_to(ROOT)/'publication-checks.json'),dict(status='PASS',
        compile_count=len(compiled),bash_count=len(shell),json_count=len(jsons),
        compiled=compiled,shell=shell,partial_b_manifest_members=verified,
        relocated_members=relocated,original_worktrees_unchanged=post,
        raw_free='allowlisted textual source/receipts/aggregate only; no raw paths followed',
        original_worktree_mutation=0,raw_mutation=0,GPU=0,Slurm=0,rsync=0,scientific_promotion=False))
    print(json.dumps(dict(compile=len(compiled),bash=len(shell),json=len(jsons),original_worktrees=len(post),relocated=len(relocated),B_report_members=len(verified))))
if __name__=='__main__':main()
