"""GFM HTML rendering and scope/count checks using compact publications only."""
import argparse
from html.parser import HTMLParser
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import markdown_it
from markdown_it import MarkdownIt
from .common import *

class DOM(HTMLParser):
    def __init__(self):
        super().__init__();self.tables=[];self.current=None;self.row=None;self.cell=None;self.ids=set();self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.add(a['id'])
        if tag in ('a','img'):self.links.append(a.get('href',a.get('src','')))
        if tag=='table':self.current=[];self.tables.append(self.current)
        if tag=='tr':self.row=[];self.current.append(self.row)
        if tag in ('td','th'):self.cell=[]
    def handle_data(self,text):
        if self.cell is not None:self.cell.append(text)
    def handle_endtag(self,tag):
        if tag in ('td','th'):
            self.row.append(''.join(self.cell));self.cell=None

def render(p,output=None):
    text=p.read_text();engine=MarkdownIt('commonmark',{'html':True}).enable('table')
    rendered=engine.render(text);dom=DOM();dom.feed(rendered)
    expected=[];current=None;code=False
    for i,line in enumerate(text.splitlines()):
        if line.startswith('```'):code=not code
        if code:continue
        if line.startswith('|'):
            cells=re.split(r'(?<!\\)\|',line.strip().strip('|'))
            if all(re.fullmatch(r'\s*:?-+:?\s*',x) for x in cells):continue
            if current is None:current=[];expected.append(current)
            current.append(len(cells))
        else:current=None
    actual=[[len(row) for row in table] for table in dom.tables]
    assert actual==expected,(p,actual,expected)
    assert all(len(set(counts))==1 for counts in actual),p
    for target in dom.links:
        if target.startswith('#'):assert target[1:] in dom.ids,(p,target)
        elif not target.startswith(('http:','https:')):assert (p.parent/target.split('#')[0]).exists(),(p,target)
    if output:
        css='body{font-family:sans-serif;max-width:1440px;margin:32px auto;line-height:1.6}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #bbb;padding:6px;white-space:normal}img{max-width:100%}code{overflow-wrap:anywhere}'
        # file:// base is local-only and never transmitted or deployed.
        write(output,'<!doctype html><html lang="ko"><meta charset="utf-8"><base href="'+p.parent.as_uri()+'/"><style>'+css+'</style><body>'+rendered+'</body></html>')
    return dom

def check_views():
    mapped=json.loads((AUDIT/'publication-mapping.json').read_text())['entries'];n=0
    ignored={'arm','arm_display_label','comparison','source_arm','source_comparison'}
    for m in mapped:
        src=WT/m['source']['path'];dst=WT/m['destination']['path']
        assert sha(src)==m['source']['sha256']
        assert sha(dst)==m['destination']['sha256']
        if m['mode']=='EXACT_BYTES_REUSED':assert src.read_bytes()==dst.read_bytes()
        elif dst.suffix=='.csv' and m['mode'] in ('ROW_VIEW_LABEL_ONLY_NUMERIC_VALUES_UNCHANGED','EXACT_BATCH10_POPULATION_VIEW'):
            before=readcsv(src);after=readcsv(dst)
            def core(r):return {k:v for k,v in r.items() if k not in ignored}
            # Preserve missing strings, floats, denominators and provenance strings exactly.
            keys={json.dumps(core(r),sort_keys=True) for r in before}
            assert all(json.dumps(core(r),sort_keys=True) in keys for r in after),dst
        n+=1
    return n

def check_scope():
    cake=(CAKE/'diagnostic-report-ko.md').read_text();cap=(CAP/'diagnostic-report-ko.md').read_text()
    assert not re.search(r'\bCAP(?:1|10|100)\b|NORM_ONLY|EP-TW-1|REFIT4|SL-ZFlow|lowcost|BG-1',cake)
    assert 'CAKE' not in cap
    assert 'SKIPPED_USER_DIRECTED / numerical_validation=NOT_ESTABLISHED' in cap
    assert 'checkpoint0' in cake and 'GPUcontinuationPASS' in cake
    assert not re.search(r'(?:AlphaEdit|MEMIT)_L[4-8]_ONLY',cake)
    names={r['arm'] for r in readcsv(CAKE/'family-final-comparison.csv')}
    expected={'CAKE_NATIVE','BASE_ALPHAEDIT_NATIVE','BASE_MEMIT_NATIVE','AlphaEdit_BLUE(L4+L8)','MEMIT_BLUE(L4+L8)'}
    expected|={f'{f}_BLUE(L{l}-only)' for f in ('AlphaEdit','MEMIT') for l in range(4,9)}
    assert names==expected
    for r in readcsv(CAKE/'family-final-comparison.csv'):
        for m,d in [('RS',10000),('PS',20000),('NS',100000)]:assert int(r[m+'_denominator'])==d
    for r in readcsv(CAP/'first-final-table.csv'):
        assert r['arm'] in ('CAP1','CAP10','CAP100','NORM_ONLY')
        assert int(r['denominator'])==dict(RS=1000,PS=2000,NS=10000)[r['metric']]
    index=(INDEX/'README.md').read_text();assert '\n|' not in index and '![' not in index
    assert '합산하지 않는다' in index
    assert not (CAP/'CAKE-B10-first1000.csv').exists()
    return dict(CAKE_unique_edited_runs=len(names),W0_separate=True,alpha_arms=4,index_scientific_table_count=0)

def run(output):
    output.mkdir(parents=True,exist_ok=False)
    trees=protected_trees();assert all(t['git_tree']==t['evidence_tree'] for t in trees)
    subprocess.run(['git','diff','--exit-code','HEAD','--',*[t['path'] for t in trees]],cwd=WT,check=True,capture_output=True)
    n=check_views();scope=check_scope();docs=[]
    for name,p in [('cake',CAKE/'diagnostic-report-ko.md'),('cap',CAP/'diagnostic-report-ko.md'),('index',INDEX/'README.md'),('server4',BASE/'README.md')]:
        dom=render(p,output/(name+'.html'))
        docs.append(dict(input=ref(p),html_path=str(output/(name+'.html')),html_sha256=sha(output/(name+'.html')),
                         rendered_tables=len(dom.tables),rendered_rows=sum(len(t) for t in dom.tables),links_checked=len(dom.links)))
    # Scope transformations may alter prose/labels, never the original scientific table cells.
    old_cap=render(CAP_OLD/'diagnostic-report-ko.md');new_cap=render(CAP/'diagnostic-report-ko.md')
    assert old_cap.tables==new_cap.tables,'cap numerical/table-cell drift'
    old_cake=render(CAKE_OLD/'diagnostic-report-ko.md');new_cake=render(CAKE/'diagnostic-report-ko.md')
    normalized=[[[label_text(v) for v in row] for row in table] for table in old_cake.tables]
    assert all(t in new_cake.tables for t in normalized),'CAKE original table drift'
    result=dict(instruction_id=TASK,status='CPU_PUBLICATION_SCOPE_NUMERIC_AND_GFM_HTML_PASS',
        input_mapping_entries=n,scope=scope,render_engine=f'markdown-it-py {markdown_it.__version__}',
        actual_render='HTML table DOM cells and links checked; not claimed browser pixel rendering',
        documents=docs,protected_v1_trees=trees,new_scheduler_queries=0,new_GPU_seconds=0,
        scientific_raw_reads=0,numerical_validation_change=False,
        cap_all_old_table_cells_identical=True,cake_all_old_table_cells_preserved_except_display_labels=True)
    save(output/'checks.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(json.dumps(run(p.parse_args().output),ensure_ascii=False))
