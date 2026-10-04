"""Finite CPU dependency barrier for shared W0; no model or scheduler calls."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate(binding, summary, initial, config, source, config_digest):
    require(binding['completed'] is True and binding['requests'] == 2000, 'W0_NOT_COMPLETE')
    require(binding['source'] == source and binding['config'] == config_digest, 'W0_SOURCE_CONFIG')
    require(binding['observer_identity'] == config['observer_identity'], 'W0_INPUT_IDENTITY')
    require(binding['state'] == summary['state'] == initial, 'W0_COLD_STATE')
    require(summary['endpoint'] == 0 and summary['requests'] == 2000 and summary['row_count'] == 26000, 'W0_COVERAGE')
    require(summary['no_mutation'] is True, 'W0_OBSERVER_MUTATION')
    require({k:v['denominator'] for k,v in summary['summary'].items()} == {'R':2000,'P':4000,'N':20000}, 'W0_DENOMINATORS')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',type=Path,required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--config-sha',required=True)
    parser.add_argument('--config-digest',required=True)
    parser.add_argument('--pilot-ready-sha',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-wait-seconds',type=int,default=13800)
    args=parser.parse_args();a=args.attempt;start=time.monotonic()
    require(0<args.max_wait_seconds<=13800, 'FINITE_GATE_WALL')
    require(not args.output.exists(), 'CREATE_ONCE_GATE_RECEIPT')
    result={'status':'TECHNICAL_FAILED','main_initial':'NOT_OBSERVED','GPU':0,'new_forward':0,'scheduler_calls':0}
    try:
        require(sha(a/'config.json')==args.config_sha, 'CONFIG_CHANGED')
        require(sha(a/'qualification-ready.json')==args.pilot_ready_sha, 'PILOT_READY_CHANGED')
        config=read(a/'config.json');pilot=read(a/'qualification-ready.json')
        require(pilot['status']=='STRUCTURAL_READY' and pilot['source']==args.source and pilot['config']==args.config_digest, 'PILOT_NOT_READY')
        binding=a/'shared-W0/binding.json';summary=a/'shared-W0/summary.json'
        while not binding.exists():
            require(not (a/'main-MAIN/terminal.json').exists(), 'MAIN_EXITED_BEFORE_W0_READY')
            require(time.monotonic()-start<args.max_wait_seconds, 'W0_READY_TIMEOUT')
            time.sleep(10)
        validate(read(binding),read(summary),read(a/'main-MAIN/initial-state.json'),config,args.source,args.config_digest)
        result.update(status='SHARED_W0_READY',source=args.source,config=args.config_digest,
                      binding_sha256=sha(binding),summary_sha256=sha(summary),task_cap=2,
                      science_source_unchanged=True)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        result['seconds']=time.monotonic()-start
        args.output.parent.mkdir(parents=True,exist_ok=True)
        temp=args.output.with_suffix('.tmp')
        with temp.open('x') as f:
            json.dump(result,f,sort_keys=True);f.flush();os.fsync(f.fileno())
        os.replace(temp,args.output)


if __name__=='__main__':
    main()
