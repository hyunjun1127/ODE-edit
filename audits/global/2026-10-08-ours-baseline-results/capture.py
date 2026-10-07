"""Bounded read-only capture from the three owners. Run from repository root."""
import concurrent.futures
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
REPORT=HERE.parents[2]/'experiment-reports/global/2026-10-08-ours-baseline-results'

def capture(server):
    if (REPORT/(server+'-snapshot.json')).exists():
        return server,'PRESERVED_PREVIOUS_SUCCESSFUL_CAPTURE'
    cells=[r for r in json.loads((HERE/'registry.json').read_text()) if r['server']==server]
    source=(HERE/'collect_host.py').read_text()+'\nprint(json.dumps(collect('+repr(cells)+'),allow_nan=False))\n'
    cmd=['python3','-'] if server=='server1' else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','rke-'+server,'python3','-']
    run=subprocess.run(cmd,input=source,text=True,capture_output=True,timeout=55)
    if run.returncode: raise RuntimeError(server+': '+run.stderr[-3000:])
    result=json.loads(run.stdout)
    path=REPORT/(server+'-snapshot.json')
    if path.exists(): raise RuntimeError('Immutable snapshot already exists: '+str(path))
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return server,len(result['cells']),sum(len(r['summaries']) for r in result['cells'])

if __name__=='__main__':
    REPORT.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(capture,s) for s in ['server1','server2','server4']]
        errors=[]
        for f in futures:
            try: print(f.result())
            except Exception as e: errors.append(str(e))
        if errors: raise RuntimeError('\n'.join(errors))
