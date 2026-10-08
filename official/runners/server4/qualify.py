"""One authorized native continuous-B3 versus cold-B2/resumed-B3 qualification.

Three sequential processes on the same assigned GPU, never a retry loop.
No full W0 observation, generation endpoint, new method or production chain.
"""
import argparse
import os
from pathlib import Path
import subprocess
from official.experiments.prepare import read, write_new
from .resume_check import compare


def plan(python,config,assets,ready,output):
    method=read(config)['method']
    if read(config)['dataset']!='cf':raise ValueError('QUALIFICATION_CF_ONLY')
    steps=[]
    for name,directory,stop,resume in (
            ('continuous','continuous',3,False),('cold-b2','resumed',2,False),
            ('resume-b3','resumed',3,True)):
        argv=[python,'-m','official.runners.server4.run',
              '--config',str(config),'--assets',str(assets),'--ready',str(ready),
              '--output',str(output/directory),'--qualification',
              '--stop-after',str(stop),'--attempt','qualification-'+name]
        if resume:argv.append('--resume')
        # One shared same-model oracle smoke; other methods retain their own
        # independent native state/history resume checks without duplicate smoke.
        if method=='ALPHAEDIT' and name=='continuous':argv.append('--oracle-smoke')
        steps.append(dict(name=name,argv=argv))
    return steps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','assets','ready','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID','').isdigit():
        raise ValueError('ACTUAL_SLURM_QUALIFICATION_ALLOCATION_REQUIRED')
    args.output.mkdir(parents=True,exist_ok=False)
    steps=plan(read(args.assets)['python'],args.config.resolve(),args.assets.resolve(),
               args.ready.resolve(),args.output.resolve())
    write_new(args.output/'qualification-plan.json',dict(steps=steps,
        job_id=os.environ['SLURM_JOB_ID'],cold_chains=2,batch_edit_calls=6,
        request_edit_applications=600,full_W0_observations=0,automatic_retry=False))
    for step in steps:
        with (args.output/(step['name']+'.log')).open('x') as log:
            done=subprocess.run(step['argv'],stdout=log,stderr=subprocess.STDOUT,check=False)
        write_new(args.output/(step['name']+'-exit.json'),dict(
            exit_code=done.returncode,job_id=os.environ['SLURM_JOB_ID']))
        if done.returncode:
            write_new(args.output/'terminal.json',dict(status='QUALIFICATION_FAILED',
                stage=step['name'],exit_code=done.returncode,automatic_retry=False))
            raise SystemExit(done.returncode if done.returncode>0 else 1)
    result=compare(args.output/'continuous',args.output/'resumed')
    write_new(args.output/'resume-comparison.json',result)
    write_new(args.output/'terminal.json',dict(status='NATIVE_RESUME_EXACT',
        model='llama3',method=read(args.config)['method'],full_2k_parity='NOT_OBSERVED',
        production_complete=False,job_id=os.environ['SLURM_JOB_ID']))


if __name__=='__main__':main()
