"""Verify the rooted sequential report, tables and optionally original inputs."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from pathlib import Path
from .round0_analysis_contracts import canonical_hash,sha256_file,verify_canonical_identity
from .sequential_analysis import require


def verify(root:Path,raw:bool=False) -> dict:
    m=json.loads((root/'analysis-manifest.json').read_text())
    r=json.loads((root/'rooted-analysis-receipt.json').read_text())
    verify_canonical_identity(m);verify_canonical_identity(r)
    require(sha256_file(root/'analysis-manifest.json')==r['manifest_sha256'],'manifest file SHA')
    require(m['identity_sha256']==r['manifest_identity'],'manifest identity')
    require(canonical_hash(m['members'])==m['members_root']==r['members_root'],'member root')
    require(sha256_file(root/r['report'])==r['report_sha256']==m['report_sha256'],'report SHA')
    expected={x['path'] for x in m['members']}|{'analysis-manifest.json','rooted-analysis-receipt.json'}
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(actual==expected,'exact package file set')
    for x in m['members']:
        p=root/x['path'];require(not p.is_symlink() and p.stat().st_size==x['bytes'] and sha256_file(p)==x['sha256'],f'package member: {p}')
    a=json.loads((root/'tables/analysis-audit.json').read_text())
    table_rows={}
    for name,expected_rows in a['table_rows'].items():
        p=root/'tables'/name;op=gzip.open if p.suffix=='.gz' else open
        with op(p,'rt',newline='') as f:
            rows=sum(1 for _ in csv.DictReader(f))
        require(rows==expected_rows,f'CSV rows {name}')
        table_rows[name]=rows
    count=0
    if raw:
        entries=json.loads((root/'tables/raw-member-inventory.json').read_text())
        require(canonical_hash(entries)==r['raw_member_root'],'raw inventory root')
        entries+=json.loads((root/'external-inputs.json').read_text())['members']
        entries+=json.loads((root/'authoritative-documents.json').read_text())
        for x in entries:
            p=Path(x['path']);require(not p.is_symlink() and p.stat().st_size==x['bytes'] and sha256_file(p)==x['sha256'],f'raw member: {p}')
            count+=1
    return {'status':'FULL_PACKAGE_REHASH_PASS','members':len(m['members']),'tables':len(table_rows),'csv_rows':sum(table_rows.values()),'raw_members_rehashed':count,'member_root':m['members_root']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('package',type=Path);p.add_argument('--raw',action='store_true')
    a=p.parse_args();print(json.dumps(verify(a.package,a.raw)))
