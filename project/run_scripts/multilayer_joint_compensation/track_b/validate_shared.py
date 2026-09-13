"""Bounded CPU suite receipt; no model/GPU/assets loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--owned-only',action='store_true');args=parser.parse_args()
    if args.receipt.exists():raise FileExistsError(args.receipt)
    root=Path(__file__).resolve().parents[4]
    package=root/'project/run_scripts/multilayer_joint_compensation'
    members=sorted([package/n for n in ('functional.py','linear_solve.py','elastic_qp.py','track_b/SHARED_API.md')]+
                   list((package/'tests').glob('test_*.py')))
    start=time.monotonic()
    if args.owned_only:
        names=[p.stem for p in sorted((package/'tests').glob('test_*.py')) if p.name.startswith(('test_functional','test_pcg','test_elastic','test_b','test_native_baselines'))]
        command=[sys.executable,'-m','unittest','-v']+['project.run_scripts.multilayer_joint_compensation.tests.'+name for name in names]
    else:command=[sys.executable,'-m','unittest','discover','-s',str(package/'tests'),'-v']
    result=subprocess.run(command,cwd=root,capture_output=True,text=True)
    receipt=dict(status='PASS' if result.returncode==0 else 'FAIL',scope='CPU_SYNTHETIC_ONLY',
                 command=command,exit_code=result.returncode,wall_seconds=time.monotonic()-start,
                 stdout=result.stdout,stderr=result.stderr,
                 members=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in members],
                 GPU=0,model_load=0,Slurm=0,actual_model_gate='NOT_YET_RUN')
    receipt['selection']='SH2_OWNED_TESTS_ONLY' if args.owned_only else 'COMBINED_SUITE'
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.receipt.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    print(result.stderr);print(json.dumps({'status':receipt['status'],'receipt':str(args.receipt)}))
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
