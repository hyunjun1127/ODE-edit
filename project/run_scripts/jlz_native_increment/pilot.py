"""Three cold subprocess pilots, no Slurm submission/retry/quality gate."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);a=p.parse_args().attempt
    lock=json.loads((a/'execution.lock.json').read_text());config=json.loads((a/'config.json').read_text());started=time.monotonic()
    evidence=[];cold=None
    for chain in CHAINS:
        subprocess.run([sys.executable,'-m','project.run_scripts.jlz_native_increment.run','--attempt',str(a),'--chain',chain,'--pilot'],check=True)
        ready=json.loads((a/('pilot-'+chain)/'ready.json').read_text())
        require(ready['source']==lock['source_commit'] and ready['config']==digest(config) and ready['commits']==2,'PILOT_IDENTITY')
        initial=json.loads((a/('pilot-'+chain)/'initial-state.json').read_text())
        if cold is None:cold=initial
        require(initial==cold,'PILOT_COLD_INDEPENDENT_IDENTITY')
        evidence.append(member(a/('pilot-'+chain)/'ready.json'))
    ready=dict(status='STRUCTURAL_READY',source=lock['source_commit'],config=digest(config),
        pilots=evidence,cold_main_required=True,main_B100='NOT_OBSERVED',seconds=time.monotonic()-started,
        no_B100_extra_fit=True,baseline_pilot_separate=True)
    timings={chain:json.loads((a/('pilot-'+chain)/'terminal.json').read_text()) for chain in CHAINS}
    write(a/'pilot-cost-plan.json',dict(measured_pilot=timings,wall_request_not_ETA=True,
        main_ETA='BS2 cannot establish B100 throughput; MAIN B1 provides first B100 observation',
        finite_wall_seconds=168*3600,extra_B100_fit=0,task_cap=1))
    write(a/'qualification-ready.json',ready)

if __name__=='__main__':main()
