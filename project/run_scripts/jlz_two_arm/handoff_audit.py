"""Static initial-handoff audit only: never query jobs or later scientific output."""
import csv
import importlib.util
import json
from pathlib import Path
from .common import ROOT,member,sha,digest,write,require

TAG='jlz-twoarm-bs100x20-20261002-v1'

def main():
    report=ROOT/'experiment-reports/servers/server4'/TAG
    audit=ROOT/'audits/servers/server4'/TAG
    status=ROOT/'tasks/status'/TAG/'server4.json'
    state=json.loads(status.read_text())
    require(state['state']=='INITIAL_GATE_PASS_MONITORING_STOPPED' and
            not state['monitoring_active'] and not state['automatic_resume'],'STOP_BOUNDARY')
    require(state['execution_source']=='2af2af3ba7a4e2e6b5d0ac5c8e53841fcb275b1b','EXECUTION_SOURCE_NOT_ANALYSIS')
    rows=list(csv.DictReader((report/'initial-metrics.csv').open()))
    require(len(rows)==9 and len({(r['scope'],r['kind']) for r in rows})==9,'COMPACT_TABLE_CARDINALITY')
    markdown=(report/'report-ko.md').read_text()
    for row in rows:
        group=state['independent_CPU_metrics'] if row['scope'].startswith('main-') else state['independent_pilot_review'][row['scope'][6]]['metrics']
        expected=group[row['kind']]
        for key,value in expected.items():
            require(row[key]==str(value),'CSV_SOURCE_VALUE '+key)
        if row['scope'].startswith('main-'):
            require(f"| {row['kind']} | {row['numerator']}/{row['denominator']} |" in markdown,'MARKDOWN_TABLE_VALUE')
    source=ROOT/'project/run_scripts/jlz_two_arm'
    files=[*source.glob('*.py'),*report.glob('*'),*audit.glob('*'),status,
           ROOT/'messages/acks/server4/2026-10-02-jlz-twoarm-bs100x20.json',
           ROOT/'messages/server-heads/server4/2026-10-02-jlz-twoarm-bs100x20.json',
           ROOT/'plans/updates/server4'/TAG/'execution-plan.json']
    exclude={'package-manifest.json','rooted-receipt.json','publication-checks.json'}
    files=sorted({p for p in files if p.is_file() and p.name not in exclude})
    require(all(p.suffix in ('.py','.json','.md','.csv') for p in files),'COMPACT_EXTENSION_ALLOWLIST')
    require(all(p.stat().st_size<2_000_000 for p in files),'NO_LARGE_ARTIFACT')
    # No raw prompt/token/teacher/tensor payload keys in the compact JSON package.
    forbidden={'input_ids','logits','requested_rewrite','paraphrase_prompts','neighborhood_prompts'}
    def check(value):
        if isinstance(value,dict):
            require(not forbidden.intersection(value),'RAW_PAYLOAD_KEY')
            for v in value.values():check(v)
        elif isinstance(value,list):
            for v in value:check(v)
    for path in files:
        if path.suffix=='.json':check(json.loads(path.read_text()))
    checks=dict(independent_reducer='standalone CPU raw reducer; owner executed',
        separate_red='bounded source audit only, not GPU/whole postrun',
        CSV_values='PASS',Markdown_table_values='PASS',raw_free_allowlist='PASS',
        actual_Markdown_render='NOT_VERIFIED_RENDERER_UNAVAILABLE',
        renderer_modules={x:importlib.util.find_spec(x) is not None for x in ('markdown','markdown_it','bs4')},
        source_commit='2af2af3ba7a4e2e6b5d0ac5c8e53841fcb275b1b',
        analysis_code=member(__file__),new_scheduler_queries=0,new_model_calls=0,
        monitoring_active=False,automatic_resume=False,science_completion='NOT_CLAIMED')
    write(audit/'publication-checks.json',checks)
    files.append(audit/'publication-checks.json')
    members=[dict(relative_path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    write(audit/'package-manifest.json',dict(schema=1,members=members,root_sha256=digest(members),raw_free=True))
    write(audit/'rooted-receipt.json',dict(manifest=member(audit/'package-manifest.json'),
          report=member(report/'report-ko.md'),status=member(status),scope='initial handoff, not 2k completion'))
    print(json.dumps(dict(files=len(files),root_sha256=digest(members),report_sha256=sha(report/'report-ko.md'))))

if __name__=='__main__':main()
