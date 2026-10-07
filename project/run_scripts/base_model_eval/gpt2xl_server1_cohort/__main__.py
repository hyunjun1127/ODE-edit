import argparse
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description='One fresh cold GPT2 W0 plus CPU cohort curves')
    parser.add_argument('command',choices=('prepare','freeze','launchers','run','collect','submit','tests'))
    parser.add_argument('--attempt',type=Path);parser.add_argument('--config',type=Path);parser.add_argument('--lock',type=Path)
    parser.add_argument('--source');args=parser.parse_args()
    if args.command=='prepare':
        from .prepare import prepare
        c=prepare(args.attempt);value=dict(status='CPU_PREPARED_NOT_SUBMITTED',config=str(Path(c['attempt'])/'config.json'),run_id=c['run_instance']['run_id'])
    elif args.command=='freeze':
        from .prepare import freeze
        from .common import member
        lock=freeze(args.config,args.source);value=dict(source=lock['source_commit'],tree=lock['source_tree'],lock=member(args.config.parent/'execution.lock.json'),source_members=len(lock['source_members']))
    elif args.command=='launchers':
        from .prepare import launchers
        value=launchers(args.config,args.lock)
    elif args.command=='submit':
        from .submit import submit
        value=submit(args.config,args.lock)
    elif args.command=='collect':
        from .collect import collect
        value=collect(args.config,args.lock)
    elif args.command=='tests':
        from .test_pipeline import cpu_gate
        value=cpu_gate(args.attempt)
    else:
        from .run import run
        value=run(args.config,args.lock)
    print(json.dumps(value,ensure_ascii=False))
    if value.get('status')=='FAILED':raise SystemExit(1)

if __name__=='__main__':main()
