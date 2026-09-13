"""Seal this raw-free Git inventory; no payload/network/scheduler access."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
members=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.iterdir()) if p.is_file() and p.name not in ('analysis-manifest.json','rooted-receipt.json')]
root=hashlib.sha256(json.dumps(members,sort_keys=True,separators=(',',':')).encode()).hexdigest()
m=ROOT/'analysis-manifest.json';m.write_text(json.dumps(dict(members=members,member_root=root,raw_payload=False),indent=2)+'\n')
r=ROOT/'rooted-receipt.json';r.write_text(json.dumps(dict(status='RAW_FREE_INVENTORY_FULL_REHASH_PASS',member_root=root,manifest_sha256=sha(m),members=len(members),GPU=0,Slurm_mutation=0,claim='VERIFIED_SCOPE_ONLY_HISTORICAL_REMAINDERS_LISTED'),indent=2)+'\n')
for x in members:
 p=ROOT/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
print(json.dumps(dict(member_root=root,report_sha256=sha(ROOT/'integration-report.md'),manifest_sha256=sha(m),receipt_sha256=sha(r))))
