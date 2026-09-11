"""등록 SH1/GH와 bounded peer-direct 통신. Payload 전송은 별도 allowlist 단계."""
import argparse
import json
from pathlib import Path
import subprocess

CONTROL=Path('/mnt/raid5/janghj/ODE-edit/local/multilayer-joint-compensation/20260911-v1/track_b/control-v1')

def send(role, message, receipt):
    helper=Path('/mnt/raid5/janghj/ODE-edit/local/alpha-native-response-v31-sequential-routing/handoff-20260906-v1/peer-source-request.py').read_text()
    start=helper.index("MESSAGE='''");end=helper.index('\ns=socket.socket',start)
    helper=helper[:start]+'MESSAGE='+repr(message)+helper[end:]
    if role=='SH1':
        helper=helper.replace("THREAD='01a04939-8873-7673-8dca-4c7fc5e31af0'","THREAD='01a04939-f93a-7b50-bca0-65438eab2062'")
    try:
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','rke-server1','python3','-'],input=helper,text=True,capture_output=True,timeout=55)
        out=dict(exit=r.returncode,stdout=r.stdout,stderr=r.stderr)
    except subprocess.TimeoutExpired as e:
        decode=lambda x:x.decode() if isinstance(x,bytes) else x or ''
        out=dict(status='COMMUNICATION_RESPONSE_TIMEOUT',stdout=decode(e.stdout),stderr=decode(e.stderr))
    CONTROL.mkdir(parents=True,mode=0o700,exist_ok=True)
    with (CONTROL/receipt).open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=['SH1','GH']);p.add_argument('message',type=Path);p.add_argument('receipt');a=p.parse_args();send(a.role,a.message.read_text(),a.receipt)
