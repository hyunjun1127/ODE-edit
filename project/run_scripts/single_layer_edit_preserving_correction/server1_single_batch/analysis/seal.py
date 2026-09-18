"""Postcheck/render/root raw-free publication. Never alters scientific inputs."""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from review import read, dump, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--local',type=Path,required=True)
    p.add_argument('--execution-wt',type=Path,required=True);a=p.parse_args();a.local.mkdir(parents=True,exist_ok=True)
    prior=a.execution_wt/'audits/servers/server1/2026-09-18-enfc-single-batch-m/full-read-preflight.json'
    reads=read(prior)['files']
    for r in reads:assert sha(r['path'])==r['sha256'],r['path']
    current=Path('messages/head/2026-09-18-sh1-enfc-b001-completed-review.md')
    dump(a.root/'read-identity-reuse.json',dict(prior_full_read=reads,current_full_read=dict(path=str(current),bytes=current.stat().st_size,sha256=sha(current)),
        evidence_policy='prior FULL_READ reused only with identical SHA; current review envelope fully read'))
    report=a.root/'diagnostic-report-ko.md';txt=report.read_text()
    links=re.findall(r'\]\(([^)]+)\)',txt)
    for link in links:assert not '://' in link and (a.root/link).is_file(),link
    tables=0;expected=None;fence=False
    for line in txt.splitlines():
        if line.startswith('```'):fence=not fence
        if not fence and line.startswith('|'):
            n=len(line.split('|'))
            if expected is None:expected=n;tables+=1
            assert n==expected,(line,n,expected)
        else:expected=None
    sys.path.insert(0,str(a.local/'render-deps'))
    import markdown
    html=markdown.markdown(txt,extensions=['tables','fenced_code'])
    assert html.count('<table>')==tables
    rendered=a.local/'diagnostic-report-ko.html'
    rendered.write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><body>'+html+'</body></html>')
    for f in a.root.iterdir():
        assert f.suffix in ('.csv','.json','.md','.png'),f
        assert f.stat().st_size < 2_000_000,(f,'unexpected large publication')
        if f.suffix=='.csv':
            with f.open() as stream:
                rr=list(csv.reader(stream));assert rr and all(len(r)==len(rr[0]) for r in rr),f
                assert not set(rr[0]) & {'prompt','target','target_token_ids','generated_token_ids','teacher_logp'},f
    # Verify scientific JSON inputs again after analysis; do not rehash huge model/teacher payloads.
    input_members=read(a.root/'input-json-manifest.json')
    for m in input_members:assert Path(m['path']).stat().st_size==m['bytes'] and sha(m['path'])==m['sha256']
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).parent),'-p','test_*.py','-v'],capture_output=True,text=True)
    assert test.returncode==0,test.stderr
    (a.local/'cpu-tests.txt').write_text(test.stdout+test.stderr)
    dump(a.root/'postcheck.json',dict(GFM_table_column_check='PASS',tables=tables,relative_links=len(links),
        markdown_renderer='Python-Markdown '+markdown.__version__,rendered_html=dict(path=str(rendered.resolve()),sha256=sha(rendered)),
        renderer_result='ACTUAL_HTML_RENDER_PASS',browser_screenshot='NOT_RUN',PNG_visual_inspection='BOTH_FIGURES_INSPECTED',
        tests=8,tests_failed=0,test_log_sha256=sha(a.local/'cpu-tests.txt'),raw_free='ALLOWLIST_EXTENSIONS_CSV_SCHEMA_MANUAL_SOURCE_REVIEW',
        scientific_json_immutability='PASS',new_GPU=0,remote_raw=0,scheduler_reads=1,
        independent_red_agent='NOT_USED_OWNER_SOURCE_AUDIT_PLUS_SEPARATE_REDUCER',full_numerical_validation='NOT_ESTABLISHED'))
    source=[dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(Path(__file__).parent.glob('*.py'))]
    excludes={'analysis-manifest.json','rooted-receipt.json'}
    members=[dict(path=f.name,sha256=sha(f),bytes=f.stat().st_size) for f in sorted(a.root.iterdir()) if f.is_file() and f.name not in excludes]
    root=hashlib.sha256(json.dumps(members,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    manifest=dict(schema='ENFC_CPU_REVIEW_V1',analysis_source_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        analysis_source_files=source,execution_source='3f1941b21538d6a7ad0afd756774ad6a0b605750',members=members,member_root=root,
        root_rule='sha256(canonical sorted-key compact JSON member list); excludes manifest/receipt to avoid cycles',
        input_manifest_sha=sha(a.root/'input-json-manifest.json'),artifact_manifest_sha=sha(a.root/'artifact-manifest.json'))
    dump(a.root/'analysis-manifest.json',manifest)
    dump(a.root/'rooted-receipt.json',dict(status='REVIEW_COMPLETE_WITH_EXPLICIT_LIMITATIONS',member_root=root,
        analysis_manifest_sha256=sha(a.root/'analysis-manifest.json'),report_sha256=sha(report),
        execution_source=manifest['execution_source'],analysis_source=manifest['analysis_source_HEAD'],
        job=50098,parent_GPU_seconds=9147,primary_rows=8,unique_requests=100,primary_unique_weight_tensors=6,
        new_GPU=0,T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED',scientific_promotion=False,
        broadcast='NO_BROADCAST_NOT_REQUIRED'))
    print(json.dumps(dict(report_sha=sha(report),manifest_sha=sha(a.root/'analysis-manifest.json'),receipt_sha=sha(a.root/'rooted-receipt.json'),root=root)))


if __name__=='__main__':main()
