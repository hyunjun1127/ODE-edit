"""One explicit bounded read; no loop/daemon/callback or scheduler mutation."""
import argparse
import datetime
import json
from pathlib import Path
from .common import ROOT,write,member
from .control import call
from .sequential_state import S_ARMS

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);a=p.parse_args()
    attempt=ROOT/'S/attempt-v1';sub=json.loads((attempt/'submission.json').read_text());job=sub['job']
    queue=call(['squeue','-h','-j',job,'-o','%i|%T|%u|%j|%R|%b|%S'])
    arms={}
    for arm in S_ARMS:
        root=attempt/'arms'/arm/'attempt-v1';d=dict(started=root.exists())
        for name in ('failure.json','S_INITIAL_VALID.json','TERMINAL.json'):
            path=root/name
            if path.exists():d[name]=dict(member=member(path),value=json.loads(path.read_text()))
        b=root/'B001'
        d['B1_stages']=[name for name in ('ENTRY.json','native/native-binding.json','protected-provenance.json',
            'geometry/space.json','native-quality.json','selection-ledger.json','SELECTION_SEALED.json',
            'COMMIT.json','OBSERVERS_COMPLETE.json','BATCH_COMPLETE.json') if (b/name).exists()]
        events=sorted((b/'events').glob('*.json'))
        if events:
            record=json.loads(events[-1].read_text())['record']
            d['latest_event']={k:record[k] for k in ('event_index','event','round','trial','reason','passed','accepted') if k in record}
        arms[arm]=d
    value=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=job,queue=queue,arms=arms)
    path=ROOT/'S/receipts'/f'observation-{a.index}.json'
    result=write(path,value)
    print(json.dumps(dict(receipt=result,**value),ensure_ascii=False))

if __name__=='__main__':main()
