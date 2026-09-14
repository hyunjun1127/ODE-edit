"""Read-only independent package/member and table arithmetic checks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def verify(root):
    root=Path(root);manifest=json.loads((root/'analysis-manifest.json').read_text())
    receipt=json.loads((root/'rooted-receipt.json').read_text())
    assert sha(root/'analysis-manifest.json')==receipt['manifest_sha256']
    members=manifest['members'];names={r['path'] for r in members}
    assert len(names)==len(members)
    assert {p.name for p in root.iterdir()}==names|{'analysis-manifest.json','rooted-receipt.json'}
    for r in members:
        p=root/r['path'];assert p.parent==root and p.is_file() and not p.is_symlink()
        assert p.suffix in ['.csv','.json','.md','.png']
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
        if p.suffix=='.png':assert p.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
    assert hashlib.sha256(json.dumps(members,sort_keys=True).encode()).hexdigest()==receipt['member_root']
    for r in manifest['sources']:
        p=Path(r['path']);assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    def rows(name):
        with (root/name).open() as f:return list(csv.DictReader(f))
    final=rows('first-final-table.csv');assert len(final)==6
    for r in final:
        for m,d in [('RS',6000),('PS',12000),('NS',60000)]:
            n=int(r[m+'_n']);assert int(r[m+'_d'])==d and abs(float(r[m+'_percent'])-100*n/d)<1e-10
    rates=rows('final-populations.csv')
    lookup={(r['policy'],r['population'],r['metric']):r for r in rates if r['group']=='ALL'}
    for p in [r['policy'] for r in final]:
        for m in ['RS','PS','NS']:
            for field in ['numerator','prompt_denominator']:
                assert int(lookup[(p,'entry_old',m)][field])+int(lookup[(p,'suffix',m)][field])==int(lookup[(p,'fullseen',m)][field])
    for r in rows('paired-transitions.csv'):
        assert int(r['gained'])-int(r['lost'])==int(r['after_numerator'])-int(r['before_numerator'])==int(r['delta_numerator'])
        assert int(r['retained'])+int(r['lost'])+int(r['gained'])+int(r['failed_both'])==int(r['prompt_denominator'])
    plots=json.loads((root/'plot-reproduction.json').read_text())
    assert len(plots['plots'])==8 and all(sha(root/n)==h for n,h in plots['plots'].items())
    result=dict(status='PACKAGE_MEMBER_TABLE_ARITHMETIC_PASS',members=len(members),bytes=sum(r['bytes'] for r in members),plots=8,raw_payload=False,manifest_sha256=sha(root/'analysis-manifest.json'))
    print(json.dumps(result));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();verify(a.out)
