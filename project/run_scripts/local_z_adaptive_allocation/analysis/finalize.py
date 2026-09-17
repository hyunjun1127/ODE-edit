"""CPU-only publication checks and create-once metadata; no Git mutations."""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from .bootstrap import ROOT,REVIEW,identity,save
from .publish import REPORT,write,copy,rows
from .plot import plots

AUDIT='audits/servers/server4/2026-09-17-local-z-adaptive-allocation-review'
def git(w,*args):return subprocess.check_output(['git','-C',str(w),*args],text=True).strip()
def run(worktree):
    w=Path(worktree).resolve();p=w/REPORT;a=w/AUDIT;a.mkdir(parents=True,exist_ok=False)
    code=w/'project/run_scripts/local_z_adaptive_allocation/analysis'
    for f in code.glob('*.py'):ast.parse(f.read_text())
    test=subprocess.run(['/data/janghj/EasyEdit/.venv/bin/python','-B','-m','unittest','project.run_scripts.local_z_adaptive_allocation.analysis.test_analysis','-v'],cwd=w,text=True,capture_output=True)
    assert test.returncode==0,test.stderr
    names=re.findall(r'^test_[^ ]+',test.stderr,re.M);assert len(names)==26
    # Actual Markdown→HTML render with installed system markdown-it, no install/network.
    script="""from markdown_it import MarkdownIt
import sys
from pathlib import Path
p=Path(sys.argv[1]);out=Path(sys.argv[2]);html=MarkdownIt('commonmark').enable('table').render(p.read_text())
with out.open('x') as f:f.write('<!doctype html><meta charset="utf-8">'+html)
print(html.count('<table>'))
"""
    render=REVIEW/'rendered-report-v1.html'
    count=int(subprocess.check_output(['/usr/bin/python3','-c',script,str(p/'diagnostic-report-ko.md'),str(render)],text=True).strip())
    text=(p/'diagnostic-report-ko.md').read_text();assert '�' not in text
    tables=[];active=None
    for i,line in enumerate(text.splitlines(),1):
        if line.startswith('|'):
            n=len(line.split('|'))-2
            if active is None:active=n;tables.append(dict(line=i,columns=n,rows=0))
            assert n==active,('TABLE_COLUMNS',i,n,active)
            tables[-1]['rows']+=1
        else:active=None
    assert len(tables)==count
    links=[]
    for doc in p.glob('*.md'):
        for href in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',doc.read_text()):
            if href.startswith(('https:','http:')):continue
            target=(doc.parent/href.split('#')[0]).resolve();assert target.exists(),(doc,href)
            links.append(dict(document=doc.name,href=href,exists=True))
    figures=REVIEW/'figures-reproduced-v1';plots(p,figures)
    png=[]
    for f in (p/'figures').glob('*.png'):
        actual=identity(f);assert actual['sha256']==identity(figures/f.name)['sha256'];png.append(dict(file=f.name,bytes=actual['bytes'],sha256=actual['sha256'],byte_reproduced=True))
    for f in p.glob('*.csv'):
        with f.open() as h:
            rr=list(csv.reader(h));assert rr and all(len(r)==len(rr[0]) for r in rr)
    first=rows(p/'first-final-table.csv');final=rows(p/'final-seven-arm.csv')
    for x,y in zip(first,final):
        assert x['arm']==y['arm']
        for metric in ('RS','PS','NS'):
            for suffix in ('count','denominator','percent','vs_N4_pp'):assert float(x[metric+'_'+suffix])==float(y[metric+'_'+suffix])
    # Exact immutable task sources still match the sealed execution, no runtime edits.
    frozen=ROOT/'source-v1/project/run_scripts/local_z_adaptive_allocation'
    for f in frozen.glob('*'):
        if f.is_file():assert identity(f)['sha256']==identity(w/'project/run_scripts/local_z_adaptive_allocation'/f.name)['sha256']
    old=git(w,'show','31c02006affc2f94f64ff43539c8892206ca4b17:experiment-reports/servers/server4/README.md')
    current=(w/'experiment-reports/servers/server4/README.md').read_text()
    assert all(line in current for line in old.splitlines())
    post=dict(status='CPU_PUBLICATION_CHECKS_PASS_WITH_OPERATIONAL_DEVIATION',test_count=len(names),tests=names,
        source_AST=True,CSV_columns=True,GFM_table_count=count,GFM_tables=tables,rendered_HTML=identity(render),
        render_scope='markdown-it CommonMark plus table; HTML checked, no browser screenshot claim',links=links,PNG=png,
        first_table_metrics_unchanged=True,runtime_bytes_unchanged=True,prior_README_entries_preserved=True,new_GPU=0,
        independent_agent_review='NOT_USED; independent reducer implementation by same SH4',
        limitations=['GPU_off_on_NOT_TESTED','full_backbone_bytes_NOT_REHASHED','cap1_actual_max2_DEVIATION'])
    save(a/'postrun-checks.json',post)
    summary=json.loads((REVIEW/'analysis-r2/summary.json').read_text());t=json.loads((REVIEW/'tensor-audit/tensor-audit.json').read_text())
    copy(REVIEW/'tensor-audit/tensor-audit.json',p/'tensor-audit-summary.json')
    copy(REVIEW/'analysis-r2/summary.json',p/'metric-selector-summary.json')
    copy(REVIEW/'supplement-v1/summary.json',p/'supplement-summary.json')
    save(a/'preflight.json',dict(instruction='ODEEDIT-S06-LOCAL-Z-SEVENARM-DETAILED-REVIEW-SH4-V1',
        scope='CPU_ONLY_EXACT_48679_48680_ARRAY_COMPLETED_OUTPUT',full_read=identity(REVIEW/'full-read-m0.json'),
        scheduler_once=identity(REVIEW/'scheduler-once.json'),prior_pending_preserved=identity(ROOT/'resume-manifest.json'),
        source_exec= '32a92ad6f3fff2f258d8778f3936d152e975ac1b',resource_change=False,model_calls=0,
        raw_writes=False,helper='No generic helper PASS claimed; explicit user write paths and host/repo/branch checked'))
    manifest=dict(instruction_id='ODEEDIT-S06-LOCAL-Z-SEVENARM-DETAILED-REVIEW-SH4-V1',analysis_source_head=git(w,'rev-parse','HEAD'),
        analysis_source_tree=git(w,'rev-parse','HEAD^{tree}'),analysis_branch=git(w,'branch','--show-current'),
        execution_source_head='32a92ad6f3fff2f258d8778f3936d152e975ac1b',execution_tree='75a96b2e2122d2be6b096af12c22df8a4365a8e1',
        execution_lock=identity(ROOT/'execution.lock.json'),source_archive=identity(ROOT/'source-v1.tar'),
        source=[identity(f) for f in sorted(code.glob('*')) if f.is_file()],
        data_root=str(ROOT),review_root=str(REVIEW),scheduler_single_audit=identity(REVIEW/'scheduler-once.json'),
        reused_evidence=identity(p/'evidence-reuse-manifest.json'),postrun=identity(a/'postrun-checks.json'),
        report=identity(p/'diagnostic-report-ko.md'),GPU_calls=0,monitoring_active=False,automatic_resume=False,
        verification='CPU metric/selector/state/CP; runtime GPU technical evidence reused; no independent GPU continuation',
        operational_deviation=dict(cap_required=1,actual_max_GPU=2,seconds_above_cap=19428,cause='NOT_RECORDED'),
        artifacts=[dict(path=str(f.relative_to(p)),bytes=f.stat().st_size,sha256=identity(f)['sha256']) for f in sorted(p.rglob('*')) if f.is_file()])
    save(p/'analysis-manifest.json',manifest)
    receipt=dict(status='TASK_COMPLETE_STOP',report=identity(p/'diagnostic-report-ko.md'),manifest=identity(p/'analysis-manifest.json'),
        postrun=identity(a/'postrun-checks.json'),execution=manifest['execution_source_head'],analysis=manifest['analysis_source_head'],
        jobs=['48679']+[f'48680_{i}' for i in range(7)],committed_batches=70,requests_per_arm=1000,unique_requests=1000,
        CP_count=21,CPU_reconstructed_candidates=171,history_append=110,state_links=63,
        science_GPU_seconds=summary['science_GPU_seconds'],preparation_GPU_seconds=summary['preparation_GPU_seconds'],new_review_GPU_seconds=0,
        numerical_scope='Stored actual technical checks and CPU arithmetic, not fresh model parity',
        operational_deviation=manifest['operational_deviation'],automatic_resume=False,monitoring_active=False)
    save(p/'rooted-receipt.json',receipt)
    write(a/'audit-report-ko.md',f'''# Local-z CPU postrun 감사

26 CPU tests, {count} Markdown tables→HTML, CSV columns, local relative links, 4 PNG byte재현, 원 첫표 count/denominator/percent 불변을 확인했다.
실행 runtime bytes와 기존 README 항목은 그대로다. 70commit/63links/21CP/171 candidate재구성 및 selector/Past/NLL 재집계의 범위를 상세보고와 CSV에 분리했다.

운영 cap1 이탈(max2/19428초)은 숨기지 않았다. 원인 NOT_RECORDED이며 새scheduler질의/변경은 하지 않았다.
Source/상태/runtimeguard 확인과 GPU off-on/backbone독립byte검증은 다르다. 후자는 NOT_TESTED다.
독립 reducer는 실행 evaluator와 별도 구현했으나 같은 SH4가 자체검산했다. 별도 red/subagent PASS를 주장하지 않는다.
Raw prompt/teacher/tensor/fullstdout은 local-only, 이번 Git은 source/집계/path/hash/보고뿐이다. 새GPU0, 자동monitoring0.

보고 SHA: {receipt['report']['sha256']}
manifest SHA: {receipt['manifest']['sha256']}
''')
    print(json.dumps(dict(report=receipt['report'],manifest=receipt['manifest'],receipt=identity(p/'rooted-receipt.json')),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);a=p.parse_args();run(a.worktree)
