"""Restore manifest-declared non-executable modes after Git's umask normalization.

Git tracks the executable bit, not group-write permission. This operation never
changes bytes, regenerates a receipt, or touches any external/raw input.
"""
import argparse,json,stat
from pathlib import Path
from .common import read,require,sha256_file,canonical_hash

def restore(package,apply=False):
    root=Path(package).resolve();m=read(root/'analysis-manifest.json')
    require(canonical_hash(m['members'])==m['member_root'],'manifest member root')
    changes=[]
    for x in m['members']:
        rel=Path(x['path']);require(not rel.is_absolute() and '..' not in rel.parts,'unsafe member path')
        p=root/rel;require(not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(root),'regular in-package member')
        require(p.stat().st_size==x['bytes'] and sha256_file(p)==x['sha256'],'content mismatch: no permission repair')
        mode=stat.S_IMODE(p.stat().st_mode);expected=int(x['mode'],8)
        require(mode in (0o644,0o664) and expected in (0o644,0o664),'non-executable mode scope')
        if mode!=expected:changes.append((p,mode,expected))
    if apply:
        for p,before,after in changes:p.chmod(after)
    return dict(status='DECLARED_MODE_RESTORED' if apply else 'MODE_AUDIT_ONLY',byte_changes=0,raw_changes=0,
        changes=[dict(path=str(p.relative_to(root)),before=f'{b:04o}',after=f'{a:04o}') for p,b,a in changes])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--restore',action='store_true');a=p.parse_args();print(json.dumps(restore(a.package,a.restore)))
