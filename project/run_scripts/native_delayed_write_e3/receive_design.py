"""Exact, create-once receiver for the approved 15 small design inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[3]
ROOT = Path('/data/janghj/ODE-edit/local/native-delayed-write-e3/20260924-v1')
APPROVAL = REPO/'transfers/approvals/2026-09-24-native-delayed-write-e3-sh4.json'


def digest(b):
    return hashlib.sha256(b).hexdigest()


def receive(host):
    approval = json.loads(APPROVAL.read_text())
    assert len(approval['files']) == 15 and approval['total_bytes'] == 245092
    observed = subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'hostname'], text=True).strip()
    assert observed == 'devbox', ('SOURCE_HOST', observed)
    rows=[]
    for entry in approval['files']:
        dest=Path(entry['destination'])
        assert dest.is_relative_to(ROOT/'inputs/design') and not dest.is_symlink()
        if dest.exists():
            payload=dest.read_bytes();status='REUSED_VERIFIED'
        else:
            payload=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,
                'cat -- '+shlex.quote(entry['path'])])
            status='TRANSFERRED_VERIFIED'
        assert len(payload)==entry['bytes'] and digest(payload)==entry['sha256'], ('IDENTITY_MISMATCH', entry['path'])
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:
                f.write(payload);f.flush();os.fsync(f.fileno())
        actual=dest.read_bytes()
        assert len(actual)==entry['bytes'] and digest(actual)==entry['sha256']
        rows.append(dict(entry,status=status,receiver_sha256=digest(actual)))
    receipt=dict(instruction_id=approval['instruction_id'],status='RECEIVER_VERIFIED_15_OF_15',
        utc=datetime.now(timezone.utc).isoformat(),source_alias=host,source_hostname=observed,
        source_policy='SOURCE_KEEP',receiver_hostname='server4',approval_sha256=digest(APPROVAL.read_bytes()),
        files=rows,total_bytes=sum(x['bytes'] for x in rows),full_size_SHA_checks=15,
        job_ids=[],execution_status='IMPLEMENTING_NOT_SUBMITTED')
    path=REPO/'transfers/verifications/2026-09-24-native-delayed-write-e3-sh4/design-received.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(receipt=str(path),sha256=digest(path.read_bytes()),status=receipt['status'],
        bytes=receipt['total_bytes'],execution_status=receipt['execution_status'],job_ids=[]),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',default='devbox');receive(p.parse_args().host)
