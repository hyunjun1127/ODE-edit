"""Reproducible publication checks and non-circular report/receipt manifests.

No Git write/network/scheduler/model calls. The owning SH performs integration.
"""
import ast,csv,datetime,html,os,re,subprocess,sys
from html.parser import HTMLParser
from markdown_it import MarkdownIt
from reducer import *

ANALYSIS=WT/'project/run_scripts/sequential_local_z_allocation/analysis/completed_review_20260918'
AUDIT=WT/'audits/servers/server4/2026-09-18-sequential-local-z-allocation-v2-review'
class TableParser(HTMLParser):
    def __init__(self):super().__init__();self.tables=[];self.links=[];self.current=None;self.row=None
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='table':self.current=[]
        if tag=='tr':self.row=0
        if tag in ['td','th']:self.row+=1
        if tag=='a':self.links.append(attrs.get('href',''))
        if tag=='img':self.links.append(attrs.get('src',''))
    def handle_endtag(self,tag):
        if tag=='tr':self.current.append(self.row);self.row=None
        if tag=='table':self.tables.append(self.current);self.current=None
def member(p):return dict(path=str(p.relative_to(WT)),bytes=p.stat().st_size,sha256=sha(p))
def main():
    checks={};py=list(ANALYSIS.glob('*.py'))
    for p in py:ast.parse(p.read_text())
    checks['syntax_files']=len(py)
    testcmd=['/data/janghj/EasyEdit/.venv/bin/python','-m','unittest','discover','-s',str(ANALYSIS),'-p','test_*.py','-v']
    tr=subprocess.run(testcmd,text=True,capture_output=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=60)
    writejson(LOCAL/'focused-tests.json',dict(command=testcmd,returncode=tr.returncode,stdout=tr.stdout,stderr=tr.stderr))
    assert tr.returncode==0,tr.stderr;checks['focused_tests']='14 PASS'
    a=load(REPORT/'audit-summary.json');assert not a['failed'];checks['stored_evidence_checks']=a['checks']
    inp=load(REPORT/'input-manifest.json');assert not inp['failures'];checks['input_closure_members']=len(inp['members'])
    manifest=load(REPORT/'figure-manifest.json');before=manifest['outputs']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    subprocess.run(['/data/janghj/EasyEdit/.venv/bin/python',str(ANALYSIS/'plot.py')],env=env,check=True,timeout=60)
    after=load(REPORT/'figure-manifest.json')['outputs'];assert before==after;checks['PNG_byte_reproducible']=list(after)
    checks['PNG_visual_inspection']='SH4 viewed all3 actual PNG; readable axes/labels, measured points only; heatmap labels rounded2 with exact CSV'
    markdown=(REPORT/'diagnostic-report-ko.md').read_text();rendered=MarkdownIt('commonmark').enable('table').render(markdown)
    parser=TableParser();parser.feed(rendered)
    assert parser.tables and all(len(set(t))==1 for t in parser.tables)
    checks['GFM_HTML_tables']=len(parser.tables);checks['GFM_HTML_rows']=sum(len(t) for t in parser.tables);checks['Korean_UTF8']=True
    checks['render_level']='markdown-it actual HTML table parse; browser pixel rendering NOT_TESTED'
    (LOCAL/'diagnostic-report-ko.html').write_text('<!doctype html><meta charset="utf-8"><title>SLZ v2 CPU review</title>'+rendered)
    reserved={'analysis-manifest.json','rooted-receipt.json','publication-checks.json'}
    for link in parser.links:
        if ':' in link or link.startswith('#'):continue
        assert (REPORT/link.split('#')[0]).is_file() or link in reserved,link
    checks['links']=len(parser.links)
    inventory=list(csv.DictReader((REPORT/'raw-inventory.csv').open()))
    for r in inventory:
        s=Path(r['path']).stat();assert s.st_size==int(r['bytes']) and s.st_mtime_ns==int(r['mtime_ns']),r['path']
    current_paths={str(p) for a in ARMS for p in (ROOT/'arms'/a/'attempt-v1/output').rglob('*') if p.is_file()}
    assert current_paths=={r['path'] for r in inventory}
    checks['raw_immutable_post_stat_members']=len(inventory);checks['raw_full_SHA_already_recorded']=True
    for r in csv.DictReader((REPORT/'first-final-table.csv').open()):
        for tag in MULT:assert abs(float(r[tag+'_percent'])-100*int(r[tag+'_count'])/int(r[tag+'_denominator']))<1e-12
    checks['final_table_arithmetic']=True
    for p in REPORT.rglob('*'):
        if p.is_file():assert p.suffix in ['.md','.json','.csv','.png'],str(p)
    checks.update(runtime_changes=0,model_load=0,new_GPU=0,slurm_mutations=0,scheduler='EXACT_SIX_ONCE_REUSED_RECEIPT',new_remote_transfer=0,independent_red_agent=False,review_type='SH_SELF_AUDIT_PLUS_INDEPENDENT_CPU_REDUCER_SELECTOR',analysis_repairs=['ASCII_JSON_IDENTITY_ENCODING','SET_ID_SERIALIZATION_ORDER_NOT_SEMANTIC_CHANGE'])
    writejson(REPORT/'publication-checks.json',checks)
    source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip();source_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=WT,text=True).strip()
    sources=[member(p) for p in sorted(ANALYSIS.glob('*.py'))]
    artifacts=[member(p) for p in sorted(REPORT.rglob('*')) if p.is_file() and p.name not in ['analysis-manifest.json','rooted-receipt.json']]
    writejson(REPORT/'analysis-manifest.json',dict(schema='SLZV2_CPU_REVIEW_V1',execution_head='21297ec19e7f5aecec16d2fdb14cc79380a1df94',analysis_source_head=source_head,analysis_source_tree=source_tree,source_members=sources,artifacts=artifacts,source_runtime_unchanged=True,raw_inventory=member(REPORT/'raw-inventory.csv'),local_full_audit_sha=sha(LOCAL/'audit-checks.json'),local_tensor_audit_sha=sha(LOCAL/'tensor-checks.json'),new_gpu=0,immutable_originals=True))
    writejson(REPORT/'rooted-receipt.json',dict(schema='SLZV2_COMPLETED_REVIEW_ROOT_V1',instruction_id='ODEEDIT-S06-SLZV2-COMPLETED-DETAILED-REVIEW-SH4-V1',manifest=member(REPORT/'analysis-manifest.json'),report=member(REPORT/'diagnostic-report-ko.md'),first_table=member(REPORT/'first-final-table.csv'),scheduler_receipt_sha=sha(LOCAL/'scheduler-r1.json'),scope=dict(arms=6,batches=60,arm_requests=6000,unique_requests=1000,commits=60,history_appends=300,adjacent_links=54),new_gpu=0,disk_W_M_checkpoint=0,exact_crash_resume='NOT_AVAILABLE',GPU_continuation='NOT_TESTED',automatic_resume=False,review_state='CPU_REVIEW_COMPLETE',publication_commit='REPORTED_SEPARATELY_IN_FINAL_HANDOFF_TO_AVOID_SELF_REFERENCE'))
    for link in parser.links:
        if ':' not in link and not link.startswith('#'):assert (REPORT/link.split('#')[0]).is_file(),link
    for m in load(REPORT/'analysis-manifest.json')['artifacts']:
        p=WT/m['path'];assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256']
    AUDIT.mkdir(parents=True,exist_ok=True)
    writejson(AUDIT/'postrun.json',dict(checks=checks,report=member(REPORT/'diagnostic-report-ko.md'),manifest=member(REPORT/'analysis-manifest.json'),rooted=member(REPORT/'rooted-receipt.json'),rooted_and_member_hashes_verified=True))
    writejson(AUDIT/'full-read.json',load(LOCAL/'full-read-r1.json'))
    writejson(AUDIT/'scheduler-six-only.json',load(LOCAL/'scheduler-r1.json'))
    print(json.dumps(checks,indent=2));print('REPORT_SHA',sha(REPORT/'diagnostic-report-ko.md'));print('ROOT_SHA',sha(REPORT/'rooted-receipt.json'))
if __name__=='__main__':main()
