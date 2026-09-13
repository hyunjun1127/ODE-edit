"""Seal/verify raw-free review package and exact analysis source references."""
import argparse,json,hashlib
from pathlib import Path
from .review_provenance import sha,write
def seal(root):
 root=Path(root);members=[]
 for p in sorted(root.rglob('*')):
  if not p.is_file() or p.name in ['analysis-manifest.json','rooted-receipt.json']:continue
  assert p.suffix in ['.json','.csv','.md','.png'] and not p.is_symlink()
  members.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
 code=Path(__file__).parent
 sources=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(code.glob('review*.py'))+sorted(code.glob('test_review*.py'))]
 member_root=hashlib.sha256(json.dumps(members,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 write(root/'analysis-manifest.json',dict(schema=1,members=members,member_root=member_root,analysis_sources=sources,execution_commit='7ece056c33fbb4246245c15f5f7c2a678315c05c',raw_payload_in_git=False))
 write(root/'rooted-receipt.json',dict(status='PACKAGE_FULL_REHASH_PASS',members=len(members),member_root=member_root,manifest_sha256=sha(root/'analysis-manifest.json'),scope='SIX_CORE_ENDPOINT_REVIEW_ONLY',claim_decision='PENDING_GH_REVIEW',audit_and_suffix_executed=False,scientific_promotion=False))
 return verify(root)
def verify(root):
 root=Path(root);m=json.loads((root/'analysis-manifest.json').read_text());r=json.loads((root/'rooted-receipt.json').read_text())
 assert sha(root/'analysis-manifest.json')==r['manifest_sha256']
 assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}=={x['path'] for x in m['members']}|{'analysis-manifest.json','rooted-receipt.json'}
 for row in m['members']:
  p=root/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
 for row in m['analysis_sources']:assert sha(row['path'])==row['sha256']
 assert hashlib.sha256(json.dumps(m['members'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==m['member_root']==r['member_root']
 return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--verify-only',action='store_true');a=p.parse_args();print(json.dumps((verify if a.verify_only else seal)(a.root)))
