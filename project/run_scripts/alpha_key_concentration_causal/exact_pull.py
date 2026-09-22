"""Exact manifest pull; source read-only, verified create-once destination.

No discovery, wildcard, overwrite, delete or model execution. Input manifest
must contain source_path/destination_relative_path/bytes/sha256 for each member.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--receipt', required=True)
    a = p.parse_args()
    doc = json.loads(Path(a.manifest).read_text())
    members = doc.get('members')
    if members is None:
        assert doc['arm'] == 'BASE_ALPHAEDIT'
        members = [dict(source_path=doc['source_root'] + m['path'],
                        destination_relative_path=m['path'],bytes=m['bytes'],
                        sha256=m['sha256']) for m in doc['files']]
    root = Path(a.destination).resolve()
    receipt = Path(a.receipt)
    if receipt.exists():
        raise FileExistsError(receipt)
    root.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    paths = [m['source_path'] for m in members]
    inspect = '''import json,os,stat,sys
for p in json.loads(sys.stdin.read()):
 s=os.lstat(p)
 assert stat.S_ISREG(s.st_mode), ('NOT_REGULAR',p)
 assert os.access(p,os.R_OK), ('NOT_READABLE',p)
 print(json.dumps(dict(path=p,bytes=s.st_size,uid=s.st_uid,gid=s.st_gid,device=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns)))
'''
    import shlex
    command = 'python3 -c ' + shlex.quote(inspect)
    observed = subprocess.check_output(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', a.host, command],
        input=json.dumps(paths).encode())
    source = [json.loads(x) for x in observed.splitlines()]
    assert len(source) == len(members)
    for m, s in zip(members, source):
        assert m['source_path'] == s['path'] and m['bytes'] == s['bytes'], (m, s)
    free = os.statvfs(root)
    out = dict(status='IN_PROGRESS',host=a.host,destination=str(root),
               manifest_sha256=digest(a.manifest),start_unix=time.time(),
               free_bytes_before=free.f_bavail*free.f_frsize,source_stat=source,members=[])
    # An append-only event journal survives interruptions without claiming PASS.
    journal = receipt.with_suffix(receipt.suffix + '.events.jsonl')
    with journal.open('x') as events:
        events.write(json.dumps(out) + '\n'); events.flush(); os.fsync(events.fileno())
        for index, m in enumerate(members):
            rel = Path(m['destination_relative_path'])
            assert not rel.is_absolute() and '..' not in rel.parts
            dst = root / rel
            assert dst.resolve().is_relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            status = 'REUSED_VERIFIED'
            if not dst.exists():
                stage = dst.with_name(dst.name + '.transfer-partial')
                subprocess.run(['rsync', '-a', '--partial', '--protect-args',
                    '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
                    a.host + ':' + m['source_path'], str(stage)], check=True)
                assert stage.stat().st_size == m['bytes']
                assert digest(stage) == m['sha256'], ('SHA_MISMATCH', str(stage))
                os.link(stage, dst)  # fails rather than overwrites on collision
                stage.unlink()
                status = 'COPIED_VERIFIED'
            else:
                assert not dst.is_symlink() and dst.is_file()
                assert dst.stat().st_size == m['bytes'] and digest(dst) == m['sha256'], ('COLLISION',str(dst))
            row = dict(m, destination=str(dst),status=status,completed_unix=time.time())
            out['members'].append(row)
            events.write(json.dumps(row) + '\n'); events.flush(); os.fsync(events.fileno())
            print(json.dumps(dict(index=index+1,total=len(members),path=str(rel),status=status)),flush=True)
    out.update(status='PASS',source_keep=True,completed_unix=time.time(),
               total_bytes=sum(m['bytes'] for m in members))
    with receipt.open('x') as f:
        json.dump(out,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())


if __name__ == '__main__':
    main()
