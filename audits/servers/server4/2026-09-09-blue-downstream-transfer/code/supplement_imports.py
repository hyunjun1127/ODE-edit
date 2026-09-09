"""util.__init__의 정적 import 하위2파일 보완; 기존 source seal 불변."""
import ast,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BLUE=Path('/data/janghj/BLUE');DEST='/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1'
def main():
    out=ROOT/'evaluator-import-supplement';out.mkdir(exist_ok=False);(out/'util').mkdir()
    members=[]
    for rel in ['util/logit_lens.py','util/nethook.py']:
        b=(BLUE/rel).read_bytes();assert b==subprocess.check_output(['git','-C',str(BLUE),'show','311b076a92e4ed0f14f5c8b4909732da781bc5f7:'+rel]);ast.parse(b)
        with (out/rel).open('xb') as f:f.write(b)
        members.append(dict(relative=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    with (out/'manifest.json').open('x') as f:json.dump(dict(members=members,reason='util.__init__ imports logit_lens, which imports util.nethook; source unchanged; original13memberseal immutable',source_head='311b076a92e4ed0f14f5c8b4909732da781bc5f7'),f,indent=2)
    h=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()
    subprocess.run(['ssh','-o','BatchMode=yes','rke-server2','mkdir -m 700 '+DEST+'/evaluator-import-supplement.partial'],check=True)
    cmd=['rsync','-a','--no-owner','--no-group','--ignore-existing',str(out)+'/','rke-server2:'+DEST+'/evaluator-import-supplement.partial/'];subprocess.run(cmd,check=True)
    script='''import hashlib,json
from pathlib import Path
r=Path(DEST);p=r/'evaluator-import-supplement.partial'
m=p/'manifest.json';assert hashlib.sha256(m.read_bytes()).hexdigest()==EXPECTED
for x in json.loads(m.read_text())['members']:
 f=p/x['relative'];assert f.resolve()==f and not f.is_symlink();assert f.stat().st_size==x['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==x['sha256']
t=r/'evaluator-import-supplement';assert not t.exists();p.rename(t)
print(json.dumps(dict(status='IMPORT_CLOSURE_SUPPLEMENT_READY',manifest_sha256=EXPECTED,path=str(t))))
'''.replace('DEST',repr(DEST)).replace('EXPECTED',repr(h))
    result=subprocess.check_output(['ssh','-o','BatchMode=yes','rke-server2','python3 -'],input=script,text=True)
    with (ROOT/'evaluator-import-supplement-receipt.json').open('x') as f:json.dump(dict(command=cmd,source_sha256=h,remote=json.loads(result)),f,indent=2)
    print(result)
if __name__=='__main__':main()
