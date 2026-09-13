"""Hash publication artifacts; never follows referenced experiment paths."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    members=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in ('analysis-manifest.json','rooted-receipt.json'):
            b=p.read_bytes();members.append(dict(path=p.name,bytes=len(b),sha256=sha(b)))
    payload=''.join(f"{r['path']}\t{r['sha256']}\t{r['bytes']}\n" for r in members).encode()
    root=sha(payload)
    manifest=dict(schema='sh2-owned-branches-main-integration-v1',status='VERIFIED_SCOPE_WITH_EXPLICIT_REMAINDER',
        members=members,member_root=root,root_basis='sorted path TAB sha256 TAB bytes LF',
        scientific_promotion=False,raw_broadcast='NO_BROADCAST_NOT_REQUIRED')
    raw=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    (OUT/'analysis-manifest.json').write_bytes(raw)
    report=(OUT/'integration-report.md').read_bytes()
    receipt=dict(status='PUBLICATION_PACKAGE_REHASH_PASS',manifest_sha256=sha(raw),member_root=root,
        report=dict(path='integration-report.md',bytes=len(report),sha256=sha(report)),
        original_head='7d5bae2e3be8dba87c92b86a727d5de9ed549af3',
        content_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        publication_identity='Git commit carrying this receipt; exact pushed HEAD/tree in terminal ACK',
        all_branches_integrated=False,GPU=0,Slurm=0,model=0,evaluator=0,monitoring=0,rsync=0)
    (OUT/'rooted-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(report_sha256=sha(report),manifest_sha256=sha(raw),root=root,members=len(members))))
if __name__=='__main__':main()
