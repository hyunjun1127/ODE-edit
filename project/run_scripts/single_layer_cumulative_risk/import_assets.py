"""Create-once, hash-bound import of exact published checkpoint/source members."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get('CUMRISK_ASSET_ROOT', str(REPO / 'local/single-layer-cumulative-risk-abc/20260910-v1')))
REPORT = REPO / 'experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v2'
SOURCE = '/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/blue-source/'
CHAIN = '/data/janghj/ODE-edit/local/blue-lifelong-b100x100/attempt-checkpoint-r2/'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def rows(name):
    with (REPORT / name).open() as f:
        return list(csv.DictReader(f))

def selected():
    result = {}
    for r in rows('source-member-inventory.csv'):
        p = r['path']; dest = None
        if p.startswith(SOURCE):
            rel = p[len(SOURCE):]
            if rel.endswith('.py') and rel.split('/')[0] in {'AlphaEdit','rome','util','dsets','experiments'}:
                dest = 'blue-source/' + rel
            elif rel == 'globals.yml': dest = 'blue-source/' + rel
        if p == '/data/janghj/ODE-edit/local/blue-lifelong-b100x100/attempt-v1/configs/AlphaEdit-L4_ONLY.json':
            dest = 'config.json'
        if p.startswith(CHAIN+'lifelong/') and p.endswith('.py'):
            dest = 'historical/' + p[len(CHAIN):]
        helper='/source-tech-r2/project/run_scripts/blue_alphaedit_sequential_comparison/'
        if helper in p and p.endswith('.py'):
            dest='historical/blue_alphaedit_sequential_comparison/'+p.split(helper)[1]
        if dest: result[dest] = r
    for r in rows('raw-member-inventory.csv'):
        p = r['path']; prefix = CHAIN+'output/main-cell-3/'
        if not p.startswith(prefix): continue
        rel = p[len(prefix):]; parts = rel.split('/')
        if len(parts) != 2: continue
        batch, name = parts
        needed = (batch in {'B010','B050','B090'} and name in {'W-method-state.pt','commit.json','contexts.json','entry.json'})
        needed |= (batch in {'B011','B051','B091'} and name in {'native-targets.pt','native-observation.json','entry.json','contexts.json'})
        if needed: result['entries/'+rel] = r
    return result

def main():
    manifest=[]
    for dest,r in sorted(selected().items()):
        path=ROOT/'imports'/dest
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            partial=path.with_name(path.name+'.incoming')
            if partial.exists(): raise FileExistsError(partial)
            subprocess.run(['rsync','--checksum','--ignore-existing','--partial','--protect-args',
                            'rke-server4:'+r['path'],str(partial)],check=True)
            assert partial.is_file() and not partial.is_symlink()
            assert partial.stat().st_size==int(r['bytes']) and sha(partial)==r['sha256'], dest
            os.link(partial,path)
            partial.unlink()
        assert not path.is_symlink() and path.stat().st_size==int(r['bytes']) and sha(path)==r['sha256'],dest
        manifest.append(dict(source=r['path'],path=str(path),bytes=path.stat().st_size,sha256=r['sha256']))
        print(json.dumps(dict(member=dest,status='VERIFIED')),flush=True)
    output=ROOT/'imports/manifest-v2.json'
    content=json.dumps(dict(status='IMPORT_VERIFIED',members=manifest),sort_keys=True,indent=2)+'\n'
    if output.exists(): assert output.read_text()==content
    else:
        with output.open('x') as f:f.write(content)

if __name__=='__main__':main()
