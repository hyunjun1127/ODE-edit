"""Raw-free package seal, full member/source rehash, deterministic plot check."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def verify(root):
 root=Path(root);m=json.loads((root/'analysis-manifest.json').read_text());r=json.loads((root/'rooted-receipt.json').read_text())
 assert sha(root/'analysis-manifest.json')==r['manifest_sha256']
 assert {str(x.relative_to(root)) for x in root.rglob('*') if x.is_file()}=={x['path'] for x in m['members']}|{'analysis-manifest.json','rooted-receipt.json'}
 for x in m['members']:
  p=root/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p)
 for x in m['sources']:
  p=Path(x['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p)
 assert digest(m['members'])==r['member_root']==m['member_root']
 return r
def seal(root):
 root=Path(root);members=[]
 for p in sorted(root.rglob('*')):
  if not p.is_file() or p.name in ['analysis-manifest.json','rooted-receipt.json']:continue
  assert not p.is_symlink() and p.suffix in ['.json','.csv','.md','.png']
  members.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
 code=Path(__file__).resolve().parent
 sources=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(code.glob('seq_review*.py'))+sorted(code.glob('test_seq_review*.py'))+ [code/'review_metrics.py',code/'review_provenance.py']]
 mr=digest(members)
 save(root/'analysis-manifest.json',dict(schema=1,members=members,member_root=mr,sources=sources,execution_commit='5e96dcb3745977b1f273e3f5afbee61167248d49',raw_payload_in_git=False))
 save(root/'rooted-receipt.json',dict(status='PACKAGE_FULL_REHASH_PASS',member_root=mr,members=len(members),manifest_sha256=sha(root/'analysis-manifest.json'),scope='SIX_ARM_B51_B60_CPU_REVIEW',scientific_terminal_cells=6,batches=60,checkpoints=18,state_links=54,request_z=9000,fit_solve=90,history_L4=60,history_L8=20,new_GPU_forward=0,GPU_continuation_replay='NOT_TESTED',claim_decision='PENDING_GH_REVIEW',audit_evaluated=False,scientific_promotion=False))
 return verify(root)
def plot_reproduce(root):
 root=Path(root);before={p.name:sha(p) for p in root.glob('*.png')}
 subprocess.run([sys.executable,'-m','project.run_scripts.low_cost_write_donor_pilot.seq_review_plots','--root',str(root)],check=True)
 after={p.name:sha(p) for p in root.glob('*.png')};assert before==after and len(after)==7
 save(root/'plot-reproduction.json',dict(status='BYTE_IDENTICAL',plots=7,sha256=after,python=sys.executable,source_sha256=sha(Path(__file__).with_name('seq_review_plots.py'))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--verify-only',action='store_true');p.add_argument('--reproduce-plots',action='store_true');a=p.parse_args()
 if a.reproduce_plots:plot_reproduce(a.root)
 print(json.dumps((verify if a.verify_only else seal)(a.root)))
