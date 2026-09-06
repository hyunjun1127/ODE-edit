"""Independent file/row/root verifier of a completed analysis publication."""
import argparse,csv,json
from pathlib import Path
from .inputs import sha,canonical

def verify(root):
    root=Path(root)
    for p in [root,*root.parents]:
        if p.is_symlink():raise ValueError('SYMLINK_PARENT')
    manifest=json.loads((root/'analysis-manifest.json').read_text())
    receipt=json.loads((root/'rooted-receipt.json').read_text())
    assert sha((root/'analysis-manifest.json').read_bytes())==receipt['manifest_sha256']
    assert sha((root/'factual-report-ko.md').read_bytes())==receipt['report_sha256']
    assert sha(canonical(manifest['members']))==receipt['members_root']==manifest['members_root']
    body=dict(receipt);identity=body.pop('receipt_identity');assert sha(canonical(body))==identity
    for m in manifest['members']:
        assert Path(m['path']).name==m['path']
        p=root/m['path'];assert p.is_file() and not p.is_symlink()
        b=p.read_bytes();assert len(b)==m['bytes'] and sha(b)==m['sha256'],m['path']
        # Git only represents regular executable/non-executable mode, not 0600.
        if p.suffix=='.csv':
            with p.open(newline='') as f:assert sum(1 for _ in csv.DictReader(f))==m['rows']
    return dict(status='FULL_PACKAGE_REHASH_PASS',members=len(manifest['members']),
        receipt_identity=identity,report_sha256=receipt['report_sha256'],mode_scope='ORIGINAL_GENERATION_0600;GIT_REGULAR_NONEXECUTABLE')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(verify(a.root)))
