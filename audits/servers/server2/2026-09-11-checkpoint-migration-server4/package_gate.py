"""최종 raw-free 보존 package와 원본 local evidence를 결속·재검산."""
import argparse
import csv
import json
import subprocess
from pathlib import Path
from verify_initial import REPO, CONTROL, sha, save

AUDIT=REPO/'audits/servers/server2/2026-09-11-checkpoint-migration-server4'
OUT=REPO/'transfers/verifications/2026-09-11-checkpoint-migration-server2-destination'

def check():
    summary=json.loads((OUT/'preservation-summary.json').read_text())
    assert summary['retained_count']==summary['source_removed_count_reported']==183
    assert summary['retained_bytes']==summary['source_removed_bytes_reported']==240176147811
    with (OUT/'migration-map.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==183 and len({r['source'] for r in rows})==len({r['destination'] for r in rows})==183
    assert sum(int(r['bytes']) for r in rows)==240176147811
    for name in ['initial72-retention-catalog.json','new177-v1-retention-catalog.json','jvp1k-v1-retention-catalog.json']:
        cat=json.loads((CONTROL/name).read_text())
        if name.startswith('initial72'):
            items=[(r['destination']['path'],r['destination']) for r in cat['members']]
        else:
            items=[(r['destination_path'],r['staging_verification']) for r in cat['checkpoint_members']]
        for path,v in items:
            p=Path(path); s=p.stat()
            assert not p.is_symlink()
            assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(v['dev'],v['inode'],v['bytes'],v['mtime_ns'])
    return summary

def files():
    result=[]
    for root in [AUDIT,OUT]:
        for p in sorted(root.iterdir()):
            if p.is_dir():
                assert p.name=='__pycache__'; continue
            if p.name in ['analysis-manifest.json','rooted-receipt.json']:continue
            assert not p.is_symlink() and p.suffix in {'.py','.md','.json','.csv'}
            assert p.stat().st_size < 2*1024**2
            result.append(p)
    result.extend([REPO/'messages/server-heads/server2/2026-09-11-checkpoint-migration-server4.md',REPO/'tasks/status/server4-checkpoint-migration-server2-v1/server2.json'])
    return result

def main(mode):
    check()
    manifest_path=OUT/'analysis-manifest.json'; receipt_path=OUT/'rooted-receipt.json'
    if mode=='seal':
        members=[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=sha(p)) for p in files()]
        local=[]
        for n in ['authority-receipt.json','admission-v1.json','initial72-retention-catalog.json',
                  'new177-v1-retention-catalog.json','jvp1k-v1-retention-catalog.json','closure-scope-clarification.json']:
            p=CONTROL/n;local.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
        save(manifest_path,dict(schema='S2_CHECKPOINT_PRESERVATION_PACKAGE_V1',members=members,
            local_evidence=local,implementation_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            raw_policy='Payload local-only; explicit S4-to-S2 migration only. No third-server broadcast.',
            manifest_self_excluded=True))
        save(receipt_path,dict(status='DESTINATION_PRESERVATION_COMPLETE_PACKAGE_PASS',
             manifest_path=str(manifest_path.relative_to(REPO)),manifest_sha256=sha(manifest_path),
             report_path=str((AUDIT/'factual-preservation-ko.md').relative_to(REPO)),report_sha256=sha(AUDIT/'factual-preservation-ko.md'),
             full_destination_sha256=True,checkpoint_count=183,checkpoint_bytes=240176147811,
             raw_in_git=False,gpu_replay=0,scientific_promotion=False))
    m=json.loads(manifest_path.read_text());r=json.loads(receipt_path.read_text())
    assert sha(manifest_path)==r['manifest_sha256']
    assert {x['path'] for x in m['members']}=={str(p.relative_to(REPO)) for p in files()}
    for x in m['members']:
        p=REPO/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
    for x in m['local_evidence']:
        p=Path(x['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
    print('PACKAGE_REHASH_PASS',len(m['members']),sha(manifest_path),sha(receipt_path))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['seal','verify']);main(ap.parse_args().mode)
