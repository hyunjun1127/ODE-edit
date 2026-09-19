"""CPU-only package validation/rendering/seal; no scientific runtime imports."""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse
from html.parser import HTMLParser
from .review_completed_b1 import member, read
from .review_b1 import dump

CODE_NAMES = (
    'review_b1.py','test_review_b1.py','review_completed_b1.py','test_review_completed_b1.py',
    'review_tensor_inventory.py','test_review_tensor_inventory.py','review_local_solver.py',
    'review_provenance.py','review_supplement.py','test_review_supplement.py',
    'publish_b1_review.py','review_final_evidence.py','review_publication.py','test_review_publication.py',
)

def tables(text):
    result=[];current=[];fence=False
    for line in text.splitlines()+['']:
        if line.startswith('```'):
            fence=not fence
        isrow=not fence and line.startswith('|') and line.endswith('|')
        if isrow:
            # GFM escaped pipes cannot be counted as delimiters.
            cells=re.split(r'(?<!\\)\|',line)[1:-1]
            current.append([c.strip() for c in cells])
        elif current:
            width=len(current[0])
            if len(current)<2 or any(len(row)!=width for row in current):
                raise ValueError('MARKDOWN_TABLE_COLUMNS')
            if not all(re.fullmatch(r':?-{3,}:?',c) for c in current[1]):
                raise ValueError('MARKDOWN_TABLE_SEPARATOR')
            result.append(current);current=[]
    if fence:
        raise ValueError('UNCLOSED_CODE_FENCE')
    return result

def links(text, base, planned=()):
    checked=[]
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
        if urlparse(target).scheme or target.startswith('#'):
            continue
        dest=(base/unquote(target.split('#')[0])).resolve()
        if not dest.is_file() and target not in planned:
            raise ValueError('MISSING_LINK:'+target)
        checked.append(target)
    return checked

class RenderCheck(HTMLParser):
    def __init__(self):
        super().__init__();self.tables=0;self.cells=0;self.images=0
    def handle_starttag(self,tag,attrs):
        if tag=='table':self.tables+=1
        if tag in ('th','td'):self.cells+=1
        if tag=='img':self.images+=1

def no_raw(package):
    forbidden_keys={'prompt','target','target_token_ids','input_ids','tf_input_ids','y0','logp','received_ledger'}
    def walk(value,path):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in forbidden_keys:raise ValueError('RAW_KEY:'+path+':'+k)
                walk(v,path)
        elif isinstance(value,list):
            for v in value:walk(v,path)
    for p in package.rglob('*'):
        if not p.is_file():continue
        if p.suffix not in ('.csv','.json','.md','.png'):raise ValueError('UNEXPECTED_PUBLICATION_TYPE:'+str(p))
        if p.suffix=='.json':walk(read(p),p.name)
        if p.suffix=='.csv':
            with p.open() as f:
                rr=list(csv.reader(f))
            if not rr or any(len(r)!=len(rr[0]) for r in rr):raise ValueError('CSV_COLUMNS:'+p.name)
            if forbidden_keys.intersection(rr[0]):raise ValueError('RAW_CSV_HEADER:'+p.name)
        if p.stat().st_size>10*2**20:raise ValueError('COMPACT_FILE_LIMIT:'+p.name)
    return 'NO_RAW_KEYS_OR_TENSOR_PAYLOADS; metadata/IDs/aggregate scalars allowed'

def canonical_bytes(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def seal(repo,package,local):
    repo,package,local=map(Path,(repo,package,local))
    # The exact terminal output was inventoried by the prior reducer.
    inv=read(package/'raw-inventory.json');changed=[]
    for old in inv['members']:
        now=member(old['path'])
        if (old['bytes'],old['sha256'])!=(now['bytes'],now['sha256']):changed.append(old['path'])
    if changed:raise ValueError('RAW_POSTRUN_CHANGED:'+repr(changed))
    body=(package/'report-ko.md').read_text();tab=tables(body)
    # Count/denominator cell check against independently reduced CSV.
    with (package/'final-table.csv').open() as f:final=list(csv.DictReader(f))
    expected=[('N4','100/100 (100%)','194/200 (97%)','865/1000 (86.5%)'),
              ('EN-KL-Q','100/100 (100%)','194/200 (97%)','865/1000 (86.5%)'),
              ('DEC-LINE','100/100 (100%)','194/200 (97%)','865/1000 (86.5%)'),
              ('DEC-MODES-CUM','100/100 (100%)','194/200 (97%)','865/1000 (86.5%)')]
    if [tuple(r[:4]) for r in tab[0][2:6]]!=expected:raise ValueError('REPORT_FIRST_TABLE')
    for r in final:
        if [int(r[k]) for k in ('RS_count','PS_count','NS_count','RS_denominator','PS_denominator','NS_denominator')]!=[100,194,865,100,200,1000]:
            raise ValueError('REPORT_CSV_DISAGREEMENT')
    no_raw(package)
    pngs=sorted(package.glob('*.png'));before=[member(p) for p in pngs]
    py='/data/janghj/EasyEdit/.venv/bin/python'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    subprocess.run([py,'-m','project.run_scripts.single_layer_mechanism_first.publish_b1_review','--plots-only','--package',str(package)],
                   cwd=repo,env=env,check=True,capture_output=True,text=True)
    after=[member(p) for p in pngs]
    if before!=after:raise ValueError('PNG_BYTE_REPRODUCTION')
    tests=['test_review_b1','test_review_completed_b1','test_review_tensor_inventory','test_review_supplement','test_review_publication']
    cp=subprocess.run([py,'-m','unittest']+['project.run_scripts.single_layer_mechanism_first.'+s for s in tests],
                      cwd=repo,env=env,capture_output=True,text=True)
    log=cp.stdout+cp.stderr
    with (local/'review-tests.txt').open('x') as f:f.write(log)
    count=re.search(r'Ran (\d+) tests?',log)
    if cp.returncode or not count:raise ValueError('CPU_TEST_FAILURE')
    import markdown_it
    from markdown_it import MarkdownIt
    html=MarkdownIt('commonmark').enable('table').render(body)
    renderer=RenderCheck();renderer.feed(html)
    expected_cells=sum(sum(len(r) for i,r in enumerate(t) if i!=1) for t in tab)
    if renderer.tables!=len(tab) or renderer.cells!=expected_cells or renderer.images!=2 or '완료' not in html:
        raise ValueError('RENDER_STRUCTURAL_MISMATCH')
    rendered=local/'report-ko.rendered.html'
    with rendered.open('x') as f:f.write('<!doctype html><meta charset="utf-8">'+html)
    links(body,package,('analysis-manifest.json','rooted-receipt.json','publication-checks.json'))
    links((package/'reproduce.md').read_text(),package)
    source=repo/'project/run_scripts/single_layer_mechanism_first'
    inputs=dict(execution=read(package/'provenance.json')['execution'],raw_inventory=member(package/'raw-inventory.json'),
        source_and_large_input_verification=member(package/'provenance.json'),capsules=member(package/'capsule-summary.json'),
        accounting=member(repo/'audits/servers/server4/single-layer-mechanism-first-20260919-v1/completed-b1-review-20260920-v1/accounting.json'),
        raw_policy='immutable; full scoped rehash twice; large common inputs prior verifiedSHA+stat',new_model_calls=0)
    dump(package/'input-manifest.json',inputs)
    dump(package/'analysis-manifest.json',dict(schema='SLMF_B1_CPU_REVIEW_V1',analysis_base='0150da2f02c830baa070944e1f504852f2fe6b01',
        execution_commit='5f79085629b10b2bb8bdee88d017e18a46bb4c74',source=[dict(repo_path=str((source/n).relative_to(repo)),**member(source/n)) for n in CODE_NAMES],
        inputs=member(package/'input-manifest.json'),partial_analysis_attempts_preserved=True,
        authority='messages/head/2026-09-20-sh4-slmf-b1-completed-review.md',GPU=0,analysis_did_not_import_runtime=True))
    dump(package/'publication-checks.json',dict(CPU_tests=int(count.group(1)),CPU_tests_result='PASS',
        raw_postrun_fullSHA_unchanged=len(inv['members']),raw_bytes=sum(r['bytes'] for r in inv['members']),
        tensor_inventory=member(package/'tensor-inventory.json'),report_first_table_csv_match=True,
        markdown_tables=len(tab),rendered_cells=renderer.cells,rendered_images=renderer.images,
        renderer='markdown-it-py '+markdown_it.__version__+' CommonMark+table',rendered_HTML=member(rendered),
        Korean_UTF8_render=True,GUI_browser_pixel_render='NOT_RUN',PNG_byte_reproducibility=True,PNGs=after,
        PNG_visual_inspection='OWNER_VIEWED_BOTH',links='checked before and after final seal',raw_free=no_raw(package),
        access_helper='NOT_PASS_FOR_EXPLICITLY_ALLOWED_RUNS_PATH',
        access_helper_exact_rejection='runs/odeedit_slmf_b1_completed_review_s4_20260920/receipt.json',
        explicit_path_authority='current envelope section4; no shared helper/policy change',
        audit_level='owner audit + independent arithmetic reducer + bounded peer source review; NOT independent full red',
        peer_prior_authorship='hook/controller portions self re-review',new_GPU=0,new_scheduler_writes=0))
    members=[dict(path=str(p.relative_to(package)),bytes=p.stat().st_size,sha256=member(p)['sha256'])
             for p in sorted(package.rglob('*')) if p.is_file() and p.name!='rooted-receipt.json']
    root=hashlib.sha256(canonical_bytes(members)).hexdigest()
    dump(package/'rooted-receipt.json',dict(schema='SLMF_B1_CPU_REVIEW_ROOT_V1',root_algorithm='SHA256 canonical sorted member records; self excluded',
        payload_root=root,members=members,report_sha256=member(package/'report-ko.md')['sha256'],
        analysis_manifest_sha256=member(package/'analysis-manifest.json')['sha256'],
        source_snapshot='analysis source members; publication commit supplied by handoff, no self-reference',
        terminal='TASK_COMPLETE_STOP',monitoring_active=False,automatic_resume=False))
    links(body,package)
    return dict(payload_root=root,report=member(package/'report-ko.md'),receipt=member(package/'rooted-receipt.json'),
                tests=int(count.group(1)),tables=len(tab),raw_postrun_unchanged=len(inv['members']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--package',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();print(json.dumps(seal(a.repo,a.package,a.local),ensure_ascii=False))
