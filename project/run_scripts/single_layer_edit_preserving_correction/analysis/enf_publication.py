"""CPU-only publication checks/seals. Does not query scheduler or run science."""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from .enf_sequential_review import REPORT,ROOT,ARM,read,write,sha,member,digest

CODE=Path('project/run_scripts/single_layer_edit_preserving_correction/analysis')
FUTURE={'analysis-manifest.json','rooted-receipt.json','validation-receipt.json'}

class Tables(HTMLParser):
    def __init__(self):
        super().__init__();self.widths=[];self.current=[];self.cells=0;self.active=False
    def handle_starttag(self,tag,attrs):
        if tag=='table':self.current=[];self.active=True
        if self.active and tag=='tr':self.cells=0
        if self.active and tag in ('th','td'):self.cells+=1
    def handle_endtag(self,tag):
        if tag=='tr' and self.active:self.current.append(self.cells)
        if tag=='table':
            assert len(self.current)>=2 and len(set(self.current))==1,self.current
            self.widths.append(self.current[0]);self.active=False

def check(repo,scratch):
    out=repo/REPORT;doc=out/'diagnostic-report-ko.md';raw=doc.read_text();previous=None;count=0
    for line in raw.splitlines():
        if line.startswith('|'):
            assert line.endswith('|');n=len(line.split('|'))-2
            if previous is not None:assert n==previous,('GFM_COLUMN_MISMATCH',line)
            else:count+=1
            previous=n
        else:previous=None
    assert raw.count('```')%2==0 and raw.count('`')%2==0
    links=[]
    for p in (doc,out/'README.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('http://','https://','#')):continue
            q=p.parent/target.split('#')[0]
            assert q.is_file() or q.name in FUTURE,(p,target)
            links.append(str(q.relative_to(repo)))
    html=scratch/'diagnostic-report-ko.html'
    # System interpreter has markdown-it-py; the model runtime does not.
    render="from pathlib import Path;from markdown_it import MarkdownIt;import sys;p=Path(sys.argv[1]);Path(sys.argv[2]).write_text('<!doctype html><meta charset=\"utf-8\">'+MarkdownIt('commonmark').enable('table').render(p.read_text()))"
    r=subprocess.run(['/usr/bin/python3','-B','-c',render,str(doc),str(html)],capture_output=True,text=True)
    assert r.returncode==0,('HTML_RENDER_FAILED',r.stderr)
    parser=Tables();parser.feed(html.read_text());assert len(parser.widths)==count and '검증' in html.read_text()
    png=out/'figures/en-f-observations.png';before=sha(png)
    r=subprocess.run([sys.executable,'-B','-m','project.run_scripts.single_layer_edit_preserving_correction.analysis.enf_review_package',
                     '--scratch',str(scratch),'--phase','plots'],cwd=repo,capture_output=True,text=True)
    assert r.returncode==0,r.stderr;assert sha(png)==before,'PNG_BYTE_REPRODUCTION'
    tests=subprocess.run([sys.executable,'-B','-m','unittest','project.run_scripts.single_layer_edit_preserving_correction.analysis.test_enf_review','-v'],
                         cwd=repo,capture_output=True,text=True)
    assert tests.returncode==0,tests.stderr
    inventory=list(csv.DictReader((out/'raw-inventory.csv').open()))
    for row in inventory:
        p=Path(row['path']);assert p.is_relative_to(ARM) and p.stat().st_size==int(row['bytes'])
        # Post-analysis compact control/metric bytes independently match initial inventory.
        if p.suffix=='.json':assert sha(p)==row['sha256'],('RAW_POSTCHECK_CHANGED',p)
    lock=read(ROOT/'execution.lock.json')
    for item in lock['execution']['members']:
        actual=member(item['path'])
        assert all(actual[k]==item[k] for k in ('path','bytes','sha256'))
    assert sha(ROOT/'execution.lock.json')=='4d600d4b57df58203fb21f447116c0362b8a731b9b6d71c4485353d913224d56'
    for p in out.rglob('*'):
        if p.is_file():
            assert p.suffix in ('.md','.csv','.json','.png') and p.stat().st_size<2_000_000,(p,p.stat().st_size)
            if p.suffix in ('.csv','.json'):
                text=p.read_text()
                assert '"prompt":' not in text and '"target_token_ids":' not in text and '"generated_token_ids":' not in text
    t=read(scratch/'tensor-summary.json');m=read(scratch/'metrics-summary.json')
    for metric,num,den in [('RS',997,1000),('PS',1931,2000),('NS',8034,10000)]:
        row=next(r for r in m['tables']['final-table'] if r['scope']=='W10_full1000' and r['metric']==metric)
        assert(row['numerator'],row['denominator'])==(num,den)
    receipt=dict(status='CPU_PUBLICATION_CHECKS_PASS',tests=dict(command='unittest analysis.test_enf_review -v',result=tests.stderr),
        raw_scope=str(ARM),raw_files=len(inventory),raw_full_SHA_inventory='raw-inventory.csv',
        postcheck='all files size + all JSON fullSHA unchanged; large tensor SHA reused from completed CPU inventory',
        CP=t['current_CP'],links=t['adjacent_links'],tensor_objects_checked=t['checked_tensor_objects'],
        independent_reducer=True,separate_reviewer_agent=False,new_GPU=0,new_model_execution=0,
        additional_scheduler_queries_in_publication=0,total_exact_accounting_queries_this_recall=1,
        HTML=dict(renderer='system python3 markdown_it commonmark + table extension',rendered=member(html),tables=count,columns=parser.widths,
                  browser_visual='NOT_RUN',Korean_UTF8='CHECKED'),links_checked=len(links),PNG_byte_reproduced=True,PNG_visual_inspection='MAIN_AGENT_VIEWED',
        raw_free='COMPACT_AGGREGATES_IDENTITIES_ONLY_NO_PROMPT_TENSOR_LOG_PAYLOAD',scientific_numerical_validation='NOT_ESTABLISHED')
    receipt['analysis_only_repair']={'scope':'publication checker only','first_error':'Compared full member dict to lock dict with extra metadata keys',
        'fix':'Compare exact path/bytes/SHA fields; all original validations retained','runtime_or_raw_changed':False}
    write(out/'validation-receipt.json',receipt)
    print('PUBLICATION_CHECKS_PASS',count,'tables',len(links),'links')

def seal(repo,scratch):
    out=repo/REPORT
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=repo,text=True).strip()
    # Enforce that the analysis source itself is already immutable in this commit.
    assert not subprocess.check_output(['git','status','--porcelain','--',str(CODE)],cwd=repo,text=True).strip()
    code=[member(p) for p in sorted((repo/CODE).glob('*.py'))]
    package=[member(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('analysis-manifest.json','rooted-receipt.json')]
    relative=[dict(path=str(Path(r['path']).relative_to(repo)),bytes=r['bytes'],sha256=r['sha256']) for r in package]
    manifest=dict(schema='enf-sequential-cpu-review-v1',job='50071_1',arm='EN-F',execution='9e5884f5a2f8dcb7fc7406084f3fde94df450a55',
        analysis_source_commit=head,analysis_source_tree=tree,analysis_code=code,package=relative,
        source_raw_access='EXACT_ENF_PLUS_COMMON_IMMUTABLE_AND_SEALED_HISTORICAL_REFERENCE',new_GPU=0,
        local_analysis_inputs=[member(scratch/n) for n in ('metrics-summary.json','tensor-summary.json','supplement-summary.json')],
        excludes=['analysis-manifest.json (self)','rooted-receipt.json (manifest root)','later publication commit (Git ancestry)'],
        raw_root_inventory=member(out/'raw-inventory.csv'),validation=member(out/'validation-receipt.json'))
    write(out/'analysis-manifest.json',manifest)
    write(out/'rooted-receipt.json',dict(schema='rooted-enf-review-v1',root=digest(relative),
        manifest=member(out/'analysis-manifest.json'),report=member(out/'diagnostic-report-ko.md'),
        first_table=member(out/'first-final-table.csv'),analysis_source_commit=head,execution=manifest['execution'],
        scope='50071_1 EN-F only',status='COMPLETED_1000_CPU_REVIEW_COMPLETE',
        T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED',M_to_S='USER_DIRECTED_NOT_ESTABLISHED',
        terminal_policy='TASK_COMPLETE_STOP',automatic_resume=False,new_GPU=0))
    for row in relative:assert sha(repo/row['path'])==row['sha256']
    print(json.dumps(dict(report=member(out/'diagnostic-report-ko.md'),manifest=member(out/'analysis-manifest.json'),receipt=member(out/'rooted-receipt.json')),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--scratch',type=Path,required=True);p.add_argument('--phase',choices=['check','seal'],required=True)
    a=p.parse_args();globals()[a.phase](a.repo,a.scratch)
