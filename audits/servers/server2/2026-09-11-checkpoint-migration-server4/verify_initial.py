"""서버2 보존 검산: 읽기 전용 payload, create-once 수신 측 기록만 생성."""
import hashlib
import json
import os
from pathlib import Path
import stat
import time

REPO = Path(__file__).resolve().parents[4]
CONTROL = Path('/mnt/raid5/janghj/ODE-edit/local/checkpoint-migration-server4/20260911-v1')
ARCHIVE = Path('/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1')
INITIAL = Path('/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1')

def save(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def stable_verify(path, expected, allow_asset_symlink=False):
    p = Path(path)
    if not allow_asset_symlink:
        assert all(not x.is_symlink() for x in [p, *p.parents]), ('SYMLINK', str(p))
    before = p.stat()
    assert stat.S_ISREG(before.st_mode) and before.st_size == expected['bytes'], str(p)
    if not allow_asset_symlink:
        assert before.st_uid == os.getuid() and before.st_nlink == 1, ('OWNER_OR_LINK',str(p))
    observed = sha(p)
    after = p.stat()
    fields = ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    assert all(getattr(before, k) == getattr(after, k) for k in fields), ('CHANGED',str(p))
    assert observed == expected['sha256'], ('SHA_MISMATCH', str(p), observed, expected['sha256'])
    return dict(path=str(p), realpath=str(p.resolve()), bytes=before.st_size, sha256=observed,
                dev=before.st_dev, inode=before.st_ino, mtime_ns=before.st_mtime_ns,
                mode=oct(stat.S_IMODE(before.st_mode)), uid=before.st_uid, nlink=before.st_nlink)

def initialize():
    for p in [CONTROL, ARCHIVE]:
        p.mkdir(mode=0o700, parents=True, exist_ok=True)
        assert not p.is_symlink() and stat.S_IMODE(p.stat().st_mode)==0o700
    (CONTROL/'incoming').mkdir(mode=0o700, exist_ok=True)
    members=[]
    for name in ['messages/head/2026-09-11-server4-checkpoint-migration-server2.md',
                 'transfers/approvals/2026-09-11-server4-checkpoint-migration-server2.md', 'PROTOCOL.md']:
        p=REPO/name; data=p.read_bytes(); target=CONTROL/(('approval-' if name.startswith('transfers/') else '')+p.name)
        if target.exists():
            assert not target.is_symlink() and target.read_bytes()==data
        else:
            fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb') as f: f.write(data)
        members.append(dict(source=str(p),path=str(target),sha256=sha(target),bytes=len(data),lines=data.count(b'\n')))
    save(CONTROL/'authority-receipt.json',dict(main='58034b67d1ddb6962796b2448240f4c51a1948a1',tree='cb905c7169047be3d4c53de9a3eaf2292e8fed96',full_read=True,members=members))
    v=os.statvfs(CONTROL)
    save(CONTROL/'capacity-initial.json',dict(time=time.time(),available_bytes=v.f_bavail*v.f_frsize,available_inodes=v.f_favail,
         reuse_candidate_bytes=62011141768,new_bytes='AWAITING_SH4_MANIFEST',
         filesystem_reservation='NO_EXCLUSIVE_FS_RESERVATION',other_task_reserve_registry='NOT_FOUND_IN_CONTROL_OR_SERVERS',
         admission_safety_reserve_bytes=200*1024**3,sole_payload_writer='SH4',gpu_model_scheduler_actions=0))

def verify():
    mpath=INITIAL/'source/checkpoint-manifest.json'; spath=INITIAL/'source/source-seal.json'
    assert sha(mpath)=='e4625ab025e6bf57c30a5c3a1e6eece01557cd2d204c3266a37777368368884a'
    assert sha(spath)=='37dacc4677e7952f16377800d3e85889b312d6352f38bd31b78f3744da28e5aa'
    manifest=json.loads(mpath.read_text()); seal=json.loads(spath.read_text())
    closure=[stable_verify(INITIAL/'source'/m['relative'],m) for m in seal['members']]
    save(CONTROL/'initial72-closure-rehash.json',dict(members=closure,source_seal_sha256=sha(spath)))
    rows=[]; start=time.monotonic()
    for i,c in enumerate(manifest['checkpoints']):
        f=c['file']; p=INITIAL/f['destination_relative']
        assert p.resolve().is_relative_to(INITIAL/'payload')
        v=stable_verify(p,f)
        rows.append(dict(source_path=f['path'],destination=v,source_sha256=f['sha256'],arm=c['arm'],batch=c['batch'],
             editcount=c['editcount'],source_head=c['source_head'],model_revision=c['model_revision'],
             weights=c['weights'],method_state=c['method_state'],rng_components=c['rng_components'],
             schema_audit_reused=c['validation'],full_model_checkpoint=False))
        if (i+1)%12==0:print('PAYLOAD_FULL_REHASH',i+1,'/',len(manifest['checkpoints']),flush=True)
    assert len(rows)==72 and sum(r['destination']['bytes'] for r in rows)==62011141768
    save(CONTROL/'initial72-payload-rehash.json',dict(status='DESTINATION_BYTES_VERIFIED_PENDING_CURRENT_SOURCE_CLOSURE',
         members=rows,count=72,bytes=62011141768,manifest_sha256=sha(mpath),seconds=time.monotonic()-start,
         retention='IN_PLACE_KEEP_NO_MOVE_NO_OVERWRITE',source_deletion_authorized_by_this_receipt=False))
    lock=json.loads((INITIAL/'source/closure/005-execution.lock.json').read_text())
    assets=manifest['base_model_members']+[m for m in lock['members'] if 'null_space_project' in m['path'] or '_mom2_' in m['path']]
    refs=[]
    for asset in assets:
        p=Path(asset['path'].replace('/data/janghj/','/mnt/raid5/janghj/',1))
        refs.append(stable_verify(p,asset,True));print('REFERENCE_FULL_REHASH',p.name,flush=True)
    save(CONTROL/'initial72-base-reference-rehash.json',dict(members=refs,model_gpu_replay=0,
          inference_restore_scope='Exact pretrained + selected weight overwrite; history retained separately',
          continuation_scope='Stored RNG/history/config/source referenced; fresh GPU continuation replay NOT_PERFORMED'))
    print('INITIAL72_PAYLOAD_CLOSURE_BASE_PASS',flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='initialize':initialize()
    elif sys.argv[1]=='verify':verify()
    else:raise ValueError(sys.argv[1])
