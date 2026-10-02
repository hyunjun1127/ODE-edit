"""Read-only validation then create-once extraction of the exact v9 package."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from .common import require, member, write

def receive(archive, envelope, destination):
    authority = json.loads(envelope.read_text())
    approval = authority['transfer']
    proof = member(archive)
    require(proof['bytes'] == approval['bytes'] and proof['sha256'] == approval['sha256'], 'ARCHIVE_IDENTITY')
    allowed = {m['path']: m for m in authority['binding']['members']}
    inventory = envelope.parents[2] / authority['binding']['member_inventory']
    b = inventory.read_bytes()
    require(hashlib.sha256(b).hexdigest() == approval['inventory_sha256'], 'INVENTORY_IDENTITY')
    allowed['jlz-v9-bundle-inventory.json'] = dict(bytes=len(b), sha256=approval['inventory_sha256'])
    require(len(allowed) == approval['regular_members'] == 27, 'MEMBER_COUNT')
    contents = {}
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        require(len(members) == len(allowed), 'ARCHIVE_MEMBER_COUNT')
        for m in members:
            p = PurePosixPath(m.name)
            require(not p.is_absolute() and '..' not in p.parts and m.isfile()
                    and not m.issym() and not m.islnk(), 'UNSAFE_MEMBER')
            require(m.name in allowed and m.name not in contents, 'UNLISTED_OR_DUPLICATE')
            data = tar.extractfile(m).read()
            expected = allowed[m.name]
            require(len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'], 'MEMBER_IDENTITY:'+m.name)
            contents[m.name] = data
    for name, data in contents.items():
        dest = destination / name
        require(not any(p.is_symlink() for p in [dest, *dest.parents]), 'DESTINATION_SYMLINK')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            require(dest.read_bytes() == data, 'CREATE_ONCE_CONFLICT')
        else:
            with dest.open('xb') as f: f.write(data)
    result = dict(archive=proof, members=allowed, regular_members=27, content_members=26,
                  full_SHA_size='PASS', source_keep=True, receiver_root=str(destination))
    write(destination.parent/'receiver-proof.json', result)
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for n in ('archive','envelope','destination'): p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();receive(a.archive,a.envelope,a.destination)
