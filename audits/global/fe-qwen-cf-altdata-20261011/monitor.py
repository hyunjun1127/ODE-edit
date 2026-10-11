"""Read-only status and milestone monitor for the data-only Qwen contrast."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parent
ROOT = TASK / 'native-r1'
sys.path.insert(0, '/mnt/raid5/janghj/ODE-edit/local/fe-collapse-repair-20261011')
from monitor_native import parse_log


def capture():
    registration = json.loads((ROOT/'submission.json').read_text())
    jobs = {mode: value['job_id'] for mode, value in registration['jobs'].items()}
    accounting = subprocess.check_output(['sacct', '-n', '-X', '-P', '-j', ','.join(jobs.values()),
                                          '-o', 'JobIDRaw,State,ExitCode,Elapsed,Start,End'], text=True)
    states = {line.split('|')[0]: line.split('|')[1:] for line in accounting.strip().splitlines()}
    queue = subprocess.check_output(['squeue', '-h', '-j', ','.join(jobs.values()), '-o', '%i|%T|%R'], text=True)
    parsed = parse_log(ROOT/f"train-{jobs['train']}.out")
    errors = {}
    for mode, job in jobs.items():
        path = ROOT/f'{mode}-{job}.err'
        content = path.read_text(errors='replace') if path.exists() else ''
        errors[mode] = content[-2500:] if any(x in content for x in ('Traceback', 'Error:', 'Killed', 'CANCELLED')) else ''
    evaluation = ROOT/'evaluations/qwen25'
    metrics = {path.name.removesuffix('-receipt.json'): json.loads(path.read_text())['summary']
               for path in sorted(evaluation.glob('B*-receipt.json'))}
    result = dict(observed_at_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
                  jobs=jobs, states=states, queue=queue.strip(), errors=errors,
                  completed_batches=max([row['batch'] for row in parsed['batches'] if row['complete']], default=0),
                  current=parsed['batches'][-1] if parsed['batches'] else None,
                  checkpoints=[entry['completed_batches'] for entry in parsed['checkpoints']],
                  warnings=parsed['warnings'], metrics=metrics,
                  evaluation_complete=(evaluation/'COMPLETE.json').exists(),
                  evaluation_failed=(evaluation/'FAILURE.json').exists(),
                  free_GiB=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize/2**30)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    folder = TASK/'monitor'
    folder.mkdir(exist_ok=True)
    previous = None
    while True:
        result = capture()
        temporary = folder/'latest.partial'
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        os.replace(temporary, folder/'latest.json')
        signature = json.dumps(dict(states={key: value[:2] for key, value in result['states'].items()},
                                     completed=result['completed_batches'], checkpoints=result['checkpoints'],
                                     metrics=result['metrics'], warnings=result['warnings'], errors=result['errors']), sort_keys=True)
        if signature != previous:
            with (folder/'events.jsonl').open('a') as handle:
                handle.write(json.dumps(result)+'\n')
            print(json.dumps(result), flush=True)
            previous = signature
        terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'DEPENDENCY'}
        if not args.watch or (len(result['states']) == 2 and all(value[0] in terminal for value in result['states'].values())):
            break
        time.sleep(30)
