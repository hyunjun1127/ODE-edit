"""Selective read-only imports: create-once allowlist, no overwrite/delete."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from .contracts import ContractBoundary, digest, file_sha, member, save


def catalog_plan(catalog, destination, abc_imports=None):
    d = json.loads(Path(catalog).read_text())
    bypath = {x['path']: dict(x) for x in d['members']}
    for arm in d['arms']:
        for key in ('runtime','terminal','lock'):
            x = arm[key]; bypath.setdefault(x['path'],dict(x,kind='SOURCE_IDENTITY'))
    members=[]
    for x in sorted(bypath.values(),key=lambda r:r['path']):
        p=Path(x['path'])
        if not p.is_absolute() or '..' in p.parts or not str(p).startswith('/data/janghj/ODE-edit/local/'):
            raise ContractBoundary('OUTSIDE_APPROVED_SOURCE',path=str(p))
        reuse=None
        if abc_imports and '/main-cell-3/B' in str(p) and p.name in ('contexts.json','native-targets.pt','entry.json','native-observation.json'):
            candidate=Path(abc_imports)/'entries'/p.parent.name/p.name
            if candidate.is_file() and candidate.stat().st_size==x['bytes'] and file_sha(candidate)==x['sha256']:
                reuse=str(candidate)
        members.append(dict(source_path=str(p),bytes=x['bytes'],sha256=x['sha256'],kind=x['kind'],
                            local_reuse=reuse,destination=str(Path(destination)/p.relative_to('/'))))
    return dict(source_host='server4',ssh_alias='rke-server4',catalog=member(catalog),
                selection='All 10 singleton E1 current/seen-full plus 20 Alpha E0 next-batch companions; runtime/terminal/lock only',
                members=members,member_root=digest(members),
                download_bytes=sum(x['bytes'] for x in members if not x['local_reuse']),
                destination_writer='SH1',source_mutation=False,overwrite=False)


def pull(plan_path, receipt_path):
    plan=json.loads(Path(plan_path).read_text())
    if digest(plan['members']) != plan['member_root']:
        raise ContractBoundary('ALLOWLIST_ROOT_DRIFT')
    verified=[]
    # Each path is exact and shell-independent, with all remote source paths
    # rooted in the explicitly approved server-local tree.
    for x in plan['members']:
        p=Path(x['local_reuse'] or x['destination'])
        if not p.exists():
            if x['local_reuse']: raise ContractBoundary('REUSE_DISAPPEARED',path=str(p))
            for parent in (p.parent,*p.parent.parents):
                if parent.is_symlink():raise ContractBoundary('SYMLINK_IMPORT_PARENT')
            p.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['rsync','-a','--ignore-existing','--protect-args',
                            plan['ssh_alias']+':'+x['source_path'],str(p)],check=True)
        if p.stat().st_size!=x['bytes'] or file_sha(p)!=x['sha256']:
            raise ContractBoundary('SELECTIVE_IMPORT_MISMATCH',path=str(p))
        verified.append(dict(source=x['source_path'],local=str(p),sha256=x['sha256'],bytes=x['bytes'],reuse=bool(x['local_reuse'])))
    return save(receipt_path,dict(status='ALL_SELECTED_MEMBERS_VERIFIED',allowlist=member(plan_path),
                                 members=verified,member_root=digest(verified),source_mutation=0))


def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True)
    a=sp.add_parser('plan');a.add_argument('--catalog',required=True);a.add_argument('--destination',required=True);a.add_argument('--abc-imports');a.add_argument('--output',required=True)
    a=sp.add_parser('pull');a.add_argument('--plan',required=True);a.add_argument('--receipt',required=True)
    a=p.parse_args()
    if a.command=='plan': print(json.dumps(save(a.output,catalog_plan(a.catalog,a.destination,a.abc_imports))))
    else: print(json.dumps(pull(a.plan,a.receipt)))

if __name__=='__main__':main()
