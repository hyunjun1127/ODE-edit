"""Read-only Git inventory and raw-free publication checks; never runs experiments."""
import csv, hashlib, json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
BASE='7d5bae2e3be8dba87c92b86a727d5de9ed549af3'
def git(*args,check=True):
 r=subprocess.run(['git',*args],cwd=ROOT,text=True,capture_output=True)
 if check and r.returncode:raise RuntimeError(r.stderr)
 return r
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def same(ref,path):
 a=git('rev-parse',ref+':'+path,check=False);b=git('rev-parse','HEAD:'+path,check=False)
 return a.returncode==b.returncode==0 and a.stdout==b.stdout
def run():
 head=git('rev-parse','HEAD').stdout.strip();rows=[];details=[]
 for line in git('for-each-ref','--format=%(refname:short)|%(objectname)','refs/heads','refs/remotes/origin').stdout.splitlines():
  name,h=line.split('|')
  if 'server4' not in name or '/codex/' not in '/'+name:continue
  base=git('merge-base',BASE,h).stdout.strip();paths=git('diff','--name-only',base,h).stdout.splitlines()
  old=git('merge-base','--is-ancestor',h,BASE,check=False).returncode==0
  equivalent=git('cherry',BASE,h).stdout.splitlines() if not old else []
  patch_equal=bool(equivalent) and all(x.startswith('-') for x in equivalent)
  oldeq=all(git('rev-parse',BASE+':'+p,check=False).stdout==git('rev-parse',h+':'+p,check=False).stdout for p in paths)
  now=git('merge-base','--is-ancestor',h,head,check=False).returncode==0
  remaining=[p for p in paths if not same(h,p)]
  if old or oldeq or patch_equal:status='ALREADY_IN_MAIN';why='ANCESTRY_OR_CHANGED_BLOB_OR_ALL_COMMIT_PATCH_EQUIVALENCE_AT_BASE'
  elif now or not remaining:status='INTEGRATED';why='ANCESTRY_OR_ALL_CHANGED_BLOBS_EQUAL_CANDIDATE'
  else:status='PARTIALLY_INTEGRATED';why='LATEST_VERIFIED_SUCCESSOR_ONLY; HISTORICAL_DIFFERENT_BYTES_NOT_REINTRODUCED'
  if 'lowcost-sixarm-seq10-v1' in name and remaining:why='EXECUTION_5e96dcb_PUBLISHED; LATER_93c3e4f_ADMISSION_RELEASE_CONTROL_AND_PRE_GATE_METADATA_NOT_PUBLISHED'
  task=('ODEEDIT-S06-LOWCOST-SIXARM-SEQUENTIAL-B100X10-SH4-V1' if 'seq10' in name else 'ODEEDIT-S06-FZCB-ATOMIC-B10-PRODUCTION-PILOT-V1' if 'fzcb' in name else 'BLUE_REPOSITORY_ORIGINAL_SEQUENTIAL' if 'blue-alphaedit' in name else 'OFFICIAL_LAYER_REALIZATION_DEBT_LIFELONG' if 'lifelong-b100' in name else 'EXISTING_OWN_BRANCH_TASK_RECORDS')
  row=dict(branch=name,head=h,tree=git('rev-parse',h+'^{tree}').stdout.strip(),owner='SH4_SERVER4',task_id=task,original_base=base,inventory_base=BASE,integration_candidate=head,status=status,evidence=why,changed_files=len(paths),residual_files=len(remaining))
  rows.append(row);details.append(dict(**row,changed_paths=paths,residual_paths=remaining))
 with (OUT/'branch-inventory.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save('branch-details.json',details)
 # These historical package bytes are copied through Git, not regenerated.
 packages=[]
 for rel in ['fzcb-completion-value-fast-kill-v1','fzcb-edit-main-method-initial-v1-hold','fzcb-tech-r1-joint-b1-controller-validity-audit-2026-09-01-v2']:
  p=ROOT/'experiment-reports/servers/server4'/rel;m=json.loads((p/'analysis-manifest.json').read_text());r=json.loads((p/'rooted-receipt.json').read_text())
  expected=r.get('manifest_sha256') or r['manifest']['sha256']
  assert expected==sha(p/'analysis-manifest.json')
  members=m.get('members',r.get('members'))
  for member in members:
   q=p/Path(member['path']).name
   assert q.is_file() and sha(q)==member['sha256'] and q.stat().st_size==member['bytes'],str(q)
  packages.append(dict(path=str(p.relative_to(ROOT)),members=len(members),manifest_sha256=sha(p/'analysis-manifest.json'),status='SEALED_MEMBER_SHA_SIZE_PASS',old_raw_reaudit=False))
 integration_base=git('rev-parse','origin/main').stdout.strip()
 changed=git('diff','--name-only',integration_base,head).stdout.splitlines()
 allowed=('project/run_scripts/blue_alphaedit_sequential_comparison/','project/run_scripts/official_layer_realization_debt/','project/run_scripts/fzcb_','project/run_scripts/low_cost_write_donor_pilot/','project/run_scripts/session06_fzcb_','project/run_scripts/session06_official_layer_realization_debt_lifelong_','experiment-reports/servers/server4/','plans/updates/server4/','audits/servers/server4/','messages/acks/server4/','messages/server-heads/server4/','tasks/status/server4/','runs/lowcost-sixarm-seq10-s4-20260913-v1/')
 for p in changed:
  assert p.startswith(allowed),('OUTSIDE_OWN_SCOPE',p)
  assert Path(p).suffix in ('.py','.json','.md','.csv','.png','.sbatch'),('PAYLOAD_EXTENSION',p)
  q=ROOT/p;assert q.is_file() and not q.is_symlink()
  if q.suffix!='.png':
   s=q.read_text();assert '-----BEGIN OPENSSH PRIVATE KEY-----' not in s and '-----BEGIN RSA PRIVATE KEY-----' not in s
 save('publication-checks.json',dict(status='OWN_SCOPE_RAW_FREE_PACKAGE_CHECK_PASS',base=BASE,candidate=head,changed_files=len(changed),historical_packages=packages,status_counts={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},residual_branches=[r['branch'] for r in rows if r['residual_files'] and r['status']=='PARTIALLY_INTEGRATED'],new_GPU=0,Slurm_mutations=0,raw_transfer=0))
 print(json.dumps({'refs':len(rows),'counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},'remaining':[{'branch':x['branch'],'paths':x['residual_paths']} for x in details if x['status']=='PARTIALLY_INTEGRATED']},indent=2))
if __name__=='__main__':run()
