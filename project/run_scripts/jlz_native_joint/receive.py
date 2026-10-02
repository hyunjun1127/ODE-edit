"""CPU-only exact authority/package verification; never loads a model."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path('/data/janghj/ODE-edit/local/jlz-native-joint-v4/20261002-compute-r1')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def verified(path, spec):
    data = path.read_bytes()
    assert len(data) == spec['bytes'] and sha(data) == spec['sha256'], str(path)
    return data

def main():
    env = ROOT / 'messages/head/2026-10-02-jlz-v4-compute-r1-2k-sh4.json'
    assert sha(env.read_bytes()) == '4550d7ebd245e0561a0576aae86ee98c8b3ceec1e0857c4f6de7a8bf1d987b6a'
    authority = json.loads(env.read_text())
    for row in authority['binding']['FULL_READ']:
        verified(ROOT / row['path'], row)
    inp = json.loads((ROOT / authority['binding']['input_receipt']).read_text())
    dest = LOCAL / 'inputs/design-package'
    archive = dest / Path(inp['archive']['path']).name
    verified(archive, inp['archive'])
    expected = {row['path']: row for row in inp['files']}
    assert len(expected) == 33
    with tarfile.open(archive, 'r:gz') as tf:
        members = tf.getmembers()
        assert len(members) == len(expected)
        assert len({m.name for m in members}) == len(members)
        # Validate every member before any extraction.
        for member in members:
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and '..' not in path.parts
            assert member.isfile() and member.name in expected
            data = tf.extractfile(member).read()
            spec = expected[member.name]
            assert len(data) == spec['bytes'] and sha(data) == spec['sha256'], member.name
        for member in members:
            target = dest / 'members' / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            assert not target.is_symlink()
            if target.exists():
                verified(target, expected[member.name])
            else:
                with target.open('xb') as out:
                    out.write(tf.extractfile(member).read())
                verified(target, expected[member.name])
    receipt = {
        'instruction_id': authority['instruction_id'], 'status': 'RECEIVER_SHA_SIZE_PASS',
        '설명': '승인 archive와 33 regular member receiver 전체 SHA/size 확인; source KEEP. GPU 검증 아님.',
        'archive': dict(inp['archive'], receiver=str(archive)),
        'members': inp['files'], 'authority_full_read_members': authority['binding']['FULL_READ'],
        'host': os.uname().nodename, 'session': os.environ.get('CODEX_THREAD_ID'),
        'cwd': str(ROOT), 'jobs': [], 'actual_gpu_validation': 'NOT_RUN',
    }
    path = ROOT / 'audits/servers/server4/jlz-native-joint-v4-bs100x20-20261002-v1/receiver-full-read.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'
    if path.exists():
        assert path.read_text() == data
    else:
        with path.open('x') as out:
            out.write(data)
    print(json.dumps({'status': receipt['status'], 'members': len(expected), 'receipt': str(path)}))

if __name__ == '__main__':
    main()
