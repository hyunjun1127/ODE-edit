"""Capture focused tests and byte reproduction without accessing scientific raw."""
import argparse
import subprocess
import sys
from pathlib import Path
from .common import CAKE, TASK, WT, save, sha


def run(output, reproduced):
    command=[sys.executable,'-B','-m','unittest',
             'project.run_scripts.server4_completed_review.report_separation.test_separation']
    proc=subprocess.run(command,cwd=WT,capture_output=True,text=True)
    figures=[]
    for name in ('cake-family-comparison.png','family-cumulative.png'):
        original=CAKE/'figures'/name;repeat=reproduced/name
        equal=original.read_bytes()==repeat.read_bytes()
        figures.append(dict(name=name,sha256=sha(original),repeat_sha256=sha(repeat),byte_equal=equal))
    result=dict(instruction_id=TASK,command=command,returncode=proc.returncode,
                stdout=proc.stdout,stderr=proc.stderr,python=sys.version,
                figures=figures,PNG_byte_reproduction=all(r['byte_equal'] for r in figures),
                new_GPU_seconds=0,new_scheduler_queries=0,scientific_raw_reads=0)
    save(output,result)
    if proc.returncode or not result['PNG_byte_reproduction']:
        raise RuntimeError('Publication test/reproduction failed; receipt preserved')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reproduced',type=Path,required=True)
    args=parser.parse_args();print(run(args.output,args.reproduced))
