"""One bounded scheduler audit, source instruction seal and dataset verification."""
from pathlib import Path
import hashlib,json,subprocess,datetime

REPO=Path(__file__).resolve().parents[3]
LOCAL=Path('/data/janghj/ODE-edit/local/server4-experiments-review/20260911-v1')
JOBS=['39307','39283_1','39283_2','39283_3','39283_4','39283_5','40441','40442','40443','40444','40445','40446','42657','42658']

def main():
    LOCAL.mkdir(parents=True,exist_ok=False)
    records=[]
    for p in [REPO/'PROTOCOL.md',REPO/'messages/head/2026-09-11-server4-completed-experiments-review.md',REPO/'tasks/pending/server4-completed-experiments-review-20260911-v1.yaml']:
        b=p.read_bytes()
        with (LOCAL/p.name).open('xb') as f:f.write(b)
        records.append(dict(path=str(p),bytes=len(b),lines=len(b.splitlines()),sha256=hashlib.sha256(b).hexdigest(),full_read=True))
    commands=[['sacct','-X','-n','-P','-j',','.join(JOBS),'--format=JobIDRaw,JobID,User,JobName%80,State,ExitCode,ElapsedRaw,AllocTRES%100,NodeList,Start,End'],
              ['squeue','-h','-j',','.join(JOBS),'-o','%i|%u|%j|%T|%M|%b|%R'],
              ['python3',str(REPO/'scripts/fixed_counterfact.py'),'verify','--root','/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1']]
    outputs=[]
    for c in commands:
        p=subprocess.run(c,text=True,capture_output=True,timeout=90)
        outputs.append(dict(command=c,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
    receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reads=records,commands=outputs,base_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),base_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=REPO,text=True).strip(),gpu=0,mutations=0)
    with (LOCAL/'initial.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
