"""기존 산출물 게시의 좁은 schema/hash/경로 검사. 모델·scheduler 호출0."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[4]
AUDIT=Path(__file__).resolve().parent
REPORT=ROOT/'experiment-reports/servers/server4/jlz-twoarm-bs100x20-20261002-v1/user-stop-20261002-r1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):
    with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    inv=json.loads((AUDIT/'existing-output-publication-inventory.json').read_text())
    paths={ROOT/r['relative'] for r in inv['members'] if r['action'].startswith('PUBLISHED')}
    for row in inv['members']:
        if row['action'].startswith('PUBLISHED'):assert sha(ROOT/row['relative'])==row['sha256']
    paths.update(p for d in (AUDIT,REPORT) for p in d.rglob('*') if p.is_file())
    paths.update(ROOT/r for r in (
        'experiment-reports/servers/server4/2026-10-02-user-stop-publication.md',
        'messages/acks/server4/2026-10-02-user-stop-all.json',
        'messages/server-heads/server4/2026-10-02-user-stop-all.json',
        'tasks/status/jlz-twoarm-bs100x20-20261002-v1/server4.json'))
    for p in paths:
        assert p.suffix in ('.md','.json','.csv','.py','.png','.svg'),str(p)
        assert p.stat().st_size<=524288,str(p)
        if p.suffix=='.json':json.loads(p.read_text())
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
        if p.suffix=='.csv':
            rows=list(csv.reader(p.open(newline='')))
            assert all(len(r)==len(rows[0]) for r in rows[1:]),str(p)
    new_docs=[REPORT/'report-ko.md',ROOT/'experiment-reports/servers/server4/2026-10-02-user-stop-publication.md']
    links=[]
    for p in new_docs:
        for href in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            target=(p.parent/href).resolve();assert target.exists(),str(target);links.append(str(target))
    rows=list(csv.DictReader((REPORT/'stored-endpoint-metrics.csv').open()))
    checks={(r['arm'],r['kind']):r for r in rows if r['stage']=='main' and r['batch']=='11'}
    for key,pair in {('A','R'):(1100,1100),('B','R'):(1100,1100),('A','P'):(1493,2200),
                     ('B','P'):(1514,2200),('A','N'):(882,1000),('B','N'):(883,1000)}.items():
        r=checks[key];assert (int(r['numerator']),int(r['denominator']))==pair
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]
    dump(AUDIT/'package-manifest.json',dict(members=manifest,bytes=sum(x['bytes'] for x in manifest),
        hash_scope='이번 게시 source/compact. manifest 자신과 후속 검산receipt는 제외, Git commit으로 결속.',
        previous_main=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    dump(AUDIT/'publication-checks.json',dict(status='PASS_BOUNDED_PUBLICATION',
        exact_copies=110,member_count=len(paths),json_parse=True,python_ast=True,csv_columns=True,
        new_report_links=len(links),W11_table_matches_csv=True,
        old_report_links='기존 bytes 보존. local-only 대용량 링크는 Git 미포함; 전체 구보고 재검토0',
        browser_render='NOT_VERIFIED',new_independent_red='NOT_PERFORMED',GPU_calls=0,
        producer_raw_changed=False,source_payload_in_git='SOURCE_ONLY_NO_TENSOR_PROMPT_LOG',
        source_preserved=True,configured_gpu_caps_unchanged=True))
    paths.update({AUDIT/'package-manifest.json',AUDIT/'publication-checks.json'})
    subprocess.run(['git','add','--',*[p.relative_to(ROOT).as_posix() for p in sorted(paths)]],cwd=ROOT,check=True)
    print(json.dumps({'staged_files':len(paths),'package_bytes':sum(r['bytes'] for r in manifest)}))

if __name__=='__main__':main()
