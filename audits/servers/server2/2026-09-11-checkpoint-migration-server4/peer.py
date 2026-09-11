"""등록 peer에게 한 번 전달하고 bounded 응답만 수집한다. Payload rsync 없음."""
import argparse
import json
from pathlib import Path
import subprocess
from verify_initial import CONTROL,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('role',choices=['SH4','GH']);ap.add_argument('message',type=Path);ap.add_argument('receipt');a=ap.parse_args()
    helper=Path('/mnt/raid5/janghj/ODE-edit/local/alpha-native-response-v31-sequential-routing/handoff-20260906-v1/peer-source-request.py').read_text()
    start=helper.index("MESSAGE='''");end=helper.index('\ns=socket.socket',start)
    helper=helper[:start]+'MESSAGE='+repr(a.message.read_text())+helper[end:]
    host='rke-server1'
    if a.role=='SH4':
        host='rke-server4'
        helper=helper.replace("THREAD='01a04939-8873-7673-8dca-4c7fc5e31af0'","THREAD='01a04939-b5c7-7a03-ba2d-ef3343d62cfd'")
        helper=helper.replace('/mnt/raid5/janghj/.codex/app-server-control/app-server-control.sock','/data/janghj/.codex/app-server-control/app-server-control.sock')
    try:
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3','-'],input=helper,text=True,capture_output=True,timeout=55)
        result=dict(exit=r.returncode,stdout=r.stdout,stderr=r.stderr)
    except subprocess.TimeoutExpired as e:
        decode=lambda s:s.decode() if isinstance(s,bytes) else s or ''
        result=dict(status='COMMUNICATION_HOLD_RESPONSE_TIMEOUT',stdout=decode(e.stdout),stderr=decode(e.stderr))
    save(CONTROL/a.receipt,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
