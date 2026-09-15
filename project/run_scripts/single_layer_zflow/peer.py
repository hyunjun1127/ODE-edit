"""Bounded registered peer-direct request using the already audited transport."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def main():
    ap=argparse.ArgumentParser();ap.add_argument('role',choices=['GH','SH4'])
    ap.add_argument('message',type=Path);ap.add_argument('receipt',type=Path);a=ap.parse_args()
    source=Path('/mnt/raid5/janghj/ODE-edit/local/alpha-native-response-v31-sequential-routing/handoff-20260906-v1/peer-source-request.py')
    helper=source.read_text();transport_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    start=helper.index("MESSAGE='''");end=helper.index('\ns=socket.socket',start)
    helper=helper[:start]+'MESSAGE='+repr(a.message.read_text())+helper[end:]
    host='rke-server1'
    if a.role=='SH4':
        host='rke-server4'
        helper=helper.replace("THREAD='01a04939-8873-7673-8dca-4c7fc5e31af0'","THREAD='01a04939-b5c7-7a03-ba2d-ef3343d62cfd'")
        helper=helper.replace('/mnt/raid5/janghj/.codex/app-server-control/app-server-control.sock','/data/janghj/.codex/app-server-control/app-server-control.sock')
    try:
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3','-'],
                         input=helper,text=True,capture_output=True,timeout=55)
        result=dict(exit=r.returncode,stdout=r.stdout,stderr=r.stderr)
    except subprocess.TimeoutExpired as e:
        decode=lambda s:s.decode() if isinstance(s,bytes) else s or ''
        result=dict(status='COMMUNICATION_HOLD_RESPONSE_TIMEOUT',stdout=decode(e.stdout),stderr=decode(e.stderr))
    result.update(role=a.role,transport_sha256=transport_sha,
                  message_sha256=hashlib.sha256(a.message.read_bytes()).hexdigest())
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    with a.receipt.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
