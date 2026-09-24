"""CPU publication package closure; no scheduler/model imports or operations."""
import csv
import re
import subprocess
from .common import *
from .report import figures, rows

def main():
    # Full independent CSV rerun used fresh scratch, not original collector.
    reproduced=LOCAL/'reproduction-r1/report';repro=[]
    for p in sorted(reproduced.glob('*.csv')):
        target=REPORT/p.name;assert sha(p)==sha(target),(p.name,'REPRO_MISMATCH')
        repro.append(record(target))
    assert len(repro)==12
    first={p.name:sha(p) for p in (REPORT/'figures').glob('*.png')}
    figures(rows('first-endpoint-table.csv'),rows('paired-summary.csv'),rows('coverage.csv'))
    assert first=={p.name:sha(p) for p in (REPORT/'figures').glob('*.png')}
    assert len(first)==2
    sources=[record(p) for p in sorted((REPO/'project/run_scripts/native_delayed_write_e3_completed_review').glob('*.py'))]
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    for r in sources:
        rel=str(Path(r['path']).relative_to(REPO))
        b=subprocess.check_output(['git','show',source_commit+':'+rel],cwd=REPO)
        assert hashlib.sha256(b).hexdigest()==r['sha256'],('UNCOMMITTED_ANALYSIS_SOURCE',rel)
    save(REPORT/'analysis-manifest.json',dict(instruction_id=INSTRUCTION,analysis_commit=source_commit,analysis_members=sources,
        execution_commit='3ebe0b07078940c2d46f9ea2226ccc20c0446162',execution_tree='7dece41be8ed2666d05a96cdb8432828c88fb5c0',
        execution_lock_sha256=sha(ATTEMPT/'execution.lock.json'),frozen_source_unchanged=True,CP_model_fullrehash=False,
        accounting=record(AUDIT/'accounting.json'),input_binding=record(AUDIT/'input-source-binding.json'),
        full_output_inventory=read(REPORT/'artifact-index-root.json')['file'],
        extra_atwrite_IDs=record(LOCAL/'atwrite-transition-ids.json'),
        new_model_GPU_calls=0,Slurm_write=0,NO_BROADCAST_NOT_REQUIRED=True))
    # Rooted manifest intentionally excludes itself and the receipt (no circular hashes).
    manifests={};
    for p in sorted(REPORT.rglob('*')):
        if p.is_file() and p.name not in ('package-manifest.json','rooted-receipt.json'):
            manifests[str(p.relative_to(REPORT))]=dict(bytes=p.stat().st_size,sha256=sha(p))
    save(REPORT/'package-manifest.json',dict(root=str(REPORT.relative_to(REPO)),members=manifests,raw_in_git=False))
    save(REPORT/'rooted-receipt.json',dict(instruction_id=INSTRUCTION,status='CPU_REVIEW_COMPLETE',report=record(REPORT/'report-ko.md'),
        analysis_manifest=record(REPORT/'analysis-manifest.json'),package_manifest=record(REPORT/'package-manifest.json'),
        exact_jobs={'52823':'COMPLETED 0:0','52824':'COMPLETED 0:0'},coverage=dict(E1=25,E3_pairs=12,factorial=48,modified=72),
        output_members=45592,output_bytes=814562794,GPU_allocated_seconds=12288,collector_CPU_elapsed_seconds=25,
        monitoring_active=False,automatic_resume=False,independent_agent=False,
        limitations=['No full tensor/K/v replay','General KL not independently recomputed from RAM-only teacher','GFM renderer unavailable']))
    links=[];tables=0;tablecols=None
    for p in sorted(REPORT.glob('*.md')):
        for line in p.read_text().splitlines():
            if line.startswith('|'):
                cols=len(line.split('|'))-2
                if tablecols is None:tablecols=cols;tables+=1
                assert cols==tablecols,('TABLE_COLUMN_COUNT',p,line)
            else:tablecols=None
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if link.startswith(('http:','https:','#')):continue
            target=p.parent/link.split('#')[0];assert target.exists(),('BROKEN_LINK',p,link);links.append(link)
    for p in REPORT.glob('*.csv'):
        data=list(csv.reader(p.open()));assert data and all(len(r)==len(data[0]) for r in data)
    save(AUDIT/'postrun-checks.json',dict(CSV_full_reproduction=repro,PNG_reproduction_sha256=first,PNG_owner_visual_inspection='PASS_2_FIGURES',
        GFM_table_blocks=tables,table_column_check='PASS',relative_links_checked=len(links),relative_links='PASS',
        GFM_HTML_render='NOT_RUN_RENDERER_NOT_INSTALLED',independent_red_agent=False,tests=record(AUDIT/'unit-and-collector-checks.json'),
        original_collector_readonly=True,runtime_modified=False,threshold_changed=False,raw_modified=False,full_model_CP_rehash=False,
        observer_evaluation_calls=0,publication='own-scope source and compact derived reports only'))
    print('PACKAGE_COMPLETE',source_commit,sha(REPORT/'report-ko.md'),sha(REPORT/'rooted-receipt.json'))

if __name__=='__main__':main()
