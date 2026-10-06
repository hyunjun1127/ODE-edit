"""User-approved CPU live comparison mirror, exact sealed commits only.

One process per science cell, no Slurm or model access. Stops at terminal or
after the bounded 48h job horizon plus flush allowance. Original run untouched.
"""
import argparse,json,os,time
from pathlib import Path
from project.run_scripts.jlz_interference_l1.comparison_bridge import Target,read,sha,check,atomic

class GPTJTarget(Target):
    def __init__(self,binding):
        self.b=binding;self.root=Path(binding['attempt']);self.cell=binding['cell'];self.out=self.root/self.cell
        check(sha(self.root/'config.json')==binding['config_sha256'],'CONFIG_BINDING')
        check(sha(self.root/'execution.lock.json')==binding['lock_sha256'],'LOCK_BINDING')
        self.lock=read(self.root/'execution.lock.json');self.c=read(self.root/'config.json')
        check(self.lock['source_commit']==binding['source'],'SOURCE_BINDING')
        ready=read(self.root/'inputs/ready.json')
        base=self.c['models'][self.cell.split('_',1)[0]]
        check(base['model_asset_identity']==ready['model_asset_identity'],'INPUT_MODEL_BINDING')
        self.mc=base|ready
        identity=ready['observer_identity'];check(sha(identity['path'])==identity['sha256'],'TOKEN_BINDING')
        self.identities=read(identity['path'])['rows'];self.packs=ready['packs']
        self.last=0;self.previous=None;self.previous_rng=None;self.rows={};self.hashes={}

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--cell',required=True)
    p.add_argument('--once',action='store_true')
    args=p.parse_args();attempt=args.attempt;cell=args.cell;out=attempt/cell/'comparison'
    if not (attempt/'inputs/ready.json').exists():return
    out.mkdir(exist_ok=True);lock=read(attempt/'execution.lock.json')
    import fcntl
    process_lock=(out/'process.lock').open('a')
    fcntl.flock(process_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=read(attempt/'submission.json')['jobs'] if (attempt/'submission.json').exists() else {}
    binding=dict(attempt=str(attempt),cell=cell,model='GPTJ',writer='memit' if cell.startswith('MEMIT_') else 'alphaedit',
        source=lock['source_commit'],config_sha256=lock['config_sha256'],lock_sha256=sha(attempt/'execution.lock.json'),job=jobs.get(cell,os.environ['SLURM_JOB_ID']))
    target=GPTJTarget(binding);deadline=time.monotonic()+49*3600;last=0
    while time.monotonic()<deadline:
        try:
            target.collect()
            if target.last>last:
                receipt=target.publish(out);last=target.last
                atomic(out/'receipt.json',dict(status='ONLINE_COMPARISON_VERIFIED',**receipt))
        except Exception as error:
            atomic(out/'error.json',dict(status='LOGGING_DEGRADED',type=type(error).__name__,reason=str(error)[:160],
                science_unchanged=True))
            if isinstance(error,(AssertionError,KeyError,ValueError)):break
        if args.once or (attempt/cell/'terminal.json').exists():break
        time.sleep(60)

if __name__=='__main__':main()
