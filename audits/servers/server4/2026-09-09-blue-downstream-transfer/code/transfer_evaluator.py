"""추가 evaluator source leaf만 create-once 전송; CP 전송과 경로 분리."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DEST='/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1'
VERIFY='''import hashlib,json,os
from pathlib import Path
r=Path("/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1")
p=r/"evaluator-source.partial"
assert p.resolve()==p and p.stat().st_uid==os.getuid()
m=json.loads((p/"source-manifest.json").read_text())
assert hashlib.sha256((p/"source-manifest.json").read_bytes()).hexdigest()=="21badc13a1ceed8ac25edf545fcf0b3a3b9e6b69577240bf712acde84b9581e5"
for x in m["members"]:
 f=p/x["relative"];assert f.resolve()==f and f.is_file() and not f.is_symlink()
 assert f.stat().st_size==x["bytes"] and hashlib.sha256(f.read_bytes()).hexdigest()==x["sha256"]
t=r/"evaluator-source";assert not t.exists();p.rename(t)
o=dict(status="EVALUATOR_SOURCE_READY",files=len(m["members"]),source_manifest_sha256="21badc13a1ceed8ac25edf545fcf0b3a3b9e6b69577240bf712acde84b9581e5",path=str(t),dataset_retransfer=0,metric_correctness="SH2_GH_PENDING")
with (r/"evaluator-source-ready.json").open("x") as f:json.dump(o,f,indent=2)
print(json.dumps(o))
'''
def main():
    subprocess.run(['ssh','-o','BatchMode=yes','rke-server2','mkdir -m 700 '+DEST+'/evaluator-source.partial'],check=True)
    args=['rsync','-a','--no-owner','--no-group','--ignore-existing','--partial-dir=.rsync-partial','--stats',str(ROOT/'evaluator-source-package')+'/','rke-server2:'+DEST+'/evaluator-source.partial/']
    r=subprocess.run(args,check=True,capture_output=True,text=True)
    v=subprocess.run(['ssh','-o','BatchMode=yes','rke-server2','python3 -'],input=VERIFY,text=True,check=True,capture_output=True)
    with (ROOT/'evaluator-source-transfer.json').open('x') as f:json.dump(dict(command=args,rsync_stdout=r.stdout,remote_verification=json.loads(v.stdout)),f,indent=2)
    print(v.stdout)
if __name__=='__main__':main()
