"""Create-once exact approved design package receiver (no remote mutation)."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

def verify(archive, envelope, destination):
    authority = json.loads(envelope.read_text())
    approval = authority['transfer']
    data = archive.read_bytes()
    assert len(data) == approval['bytes']
    assert hashlib.sha256(data).hexdigest() == approval['sha256']
    allowed = {m['path']: m for m in authority['binding']['members']}
    assert len(allowed) == approval['regular_members'] == 15
    contents = {}
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert len(members) == 15
        for m in members:
            p = PurePosixPath(m.name)
            assert not p.is_absolute() and '..' not in p.parts
            assert m.isfile() and not m.issym() and not m.islnk()
            assert m.name in allowed and m.name not in contents
            b = tar.extractfile(m).read()
            a = allowed[m.name]
            assert len(b) == a['bytes']
            assert hashlib.sha256(b).hexdigest() == a['sha256']
            contents[m.name] = b
    assert contents.keys() == allowed.keys()
    for name, b in contents.items():
        dest = destination / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        assert not any(p.is_symlink() for p in [dest, *dest.parents])
        if dest.exists():
            assert dest.read_bytes() == b
        else:
            with dest.open('xb') as f:
                f.write(b)
    return dict(archive_sha256=approval['sha256'], members=15,
                all_full_sha_size='PASS', source_keep=True,
                receiver_root=str(destination))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--envelope', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.archive, a.envelope, a.destination)))
