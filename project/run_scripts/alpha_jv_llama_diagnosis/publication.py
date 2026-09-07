"""Small create-once raw-free publication helpers; no experiment execution."""
import hashlib
import json
import os
from pathlib import Path
import stat


def canonical_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def safe_path(path, *, root=None, make_parents=False):
    path=Path(os.path.abspath(path))
    if root is not None:
        approved=Path(os.path.abspath(root))
        if not path.is_relative_to(approved):raise ValueError('PUBLICATION_OUTSIDE_APPROVED_PATH')
    current=Path(path.anchor)
    for part in path.parts[1:]:
        current=current/part
        try:info=current.lstat()
        except FileNotFoundError:
            if make_parents and current!=path:current.mkdir(mode=0o700);info=current.lstat()
            elif current==path:return path
            else:raise
        if stat.S_ISLNK(info.st_mode):raise ValueError('PUBLICATION_SYMLINK_COMPONENT')
        if current!=path and not stat.S_ISDIR(info.st_mode):raise ValueError('PUBLICATION_PARENT_NOT_DIRECTORY')
    return path


def member(path, *, relative_to=None):
    path=safe_path(path);info=path.lstat()
    if not stat.S_ISREG(info.st_mode):raise ValueError('PUBLICATION_MEMBER_NOT_REGULAR')
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):sha.update(chunk)
    return dict(path=str(path.relative_to(relative_to)) if relative_to else str(path),
                sha256=sha.hexdigest(),bytes=info.st_size,mode=f'{stat.S_IMODE(info.st_mode):04o}')


def write_once(path, data, *, root):
    path=safe_path(path,root=root,make_parents=True)
    if not isinstance(data,bytes):data=(json.dumps(data,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    return member(path)


def seal_package(root):
    root=safe_path(root)
    members=[member(p,relative_to=root) for p in sorted(root.rglob('*')) if not p.is_dir()]
    manifest=dict(schema='alpha-jv-ds.preparation-manifest.v1',members=members,members_root=digest(members))
    write_once(root/'manifest.json',manifest,root=root)
    receipt=dict(schema='alpha-jv-ds.preparation-rooted-receipt.v1',members_root=manifest['members_root'],
        manifest_sha256=member(root/'manifest.json')['sha256'],scope='CPU_PREPARATION_NOT_SCIENTIFIC_RESULT',
        experiment_model_load_count=0,GPU_action_count=0,Slurm_submit_count=0,scientific_endpoint_count=0,
        scientific_promotion=False)
    receipt['identity']=digest(receipt)
    write_once(root/'rooted-receipt.json',receipt,root=root)
    return verify_package(root)


def verify_package(root):
    root=safe_path(root)
    manifest=json.loads(safe_path(root/'manifest.json').read_text())
    receipt=json.loads(safe_path(root/'rooted-receipt.json').read_text())
    expected=manifest['members']
    actual=[member(safe_path(root/r['path'],root=root),relative_to=root) for r in expected]
    inventory={str(p.relative_to(root)) for p in root.rglob('*') if not p.is_dir()}
    if (actual!=expected or digest(actual)!=manifest['members_root']
            or inventory!={r['path'] for r in expected}|{'manifest.json','rooted-receipt.json'}
            or receipt['members_root']!=manifest['members_root']
            or receipt['manifest_sha256']!=member(root/'manifest.json')['sha256']):
        raise ValueError('PACKAGE_REHASH_BOUNDARY')
    body=dict(receipt);identity=body.pop('identity')
    if digest(body)!=identity:raise ValueError('ROOTED_RECEIPT_IDENTITY_BOUNDARY')
    return dict(status='PACKAGE_REHASH_PASS',member_count=len(actual),members_root=manifest['members_root'],
        manifest=member(root/'manifest.json'),receipt=member(root/'rooted-receipt.json'),receipt_identity=identity)
