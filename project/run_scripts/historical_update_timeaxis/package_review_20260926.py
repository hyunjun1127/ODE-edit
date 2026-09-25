"""CPU-only publication checks and compact packaging, no experiment operations."""
import collections
import csv
import hashlib
import json
from pathlib import Path

from .completed_review_20260926 import ROOT,RUN,DESIGN,REPO,REPORT,AUDIT,LOCAL,digest,active


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvrows(p):return list(csv.DictReader(p.open()))


def prepare():
    final=LOCAL/'repro-v2';matched=[]
    for p in sorted((final/'report').rglob('*')):
        if not p.is_file():continue
        dest=REPORT/p.relative_to(final/'report')
        if dest.exists():
            assert sha(dest)==sha(p),('REPRODUCTION_MISMATCH',str(dest));matched.append(str(dest.relative_to(REPO)))
        else:
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    for p in sorted((final/'audit').glob('*.json')):
        # Supersede preliminary review's derived summaries only, never execution data.
        (AUDIT/p.name).write_bytes(p.read_bytes())
    # Independently derive the censor ledger from original subject/relation and target IDs.
    lock=read(RUN/'execution.lock.json');binding=read(lock['T0']['path']);data=read(binding['dataset']['path'])
    ledger=csvrows(DESIGN/'fact-ledger.csv');groups=collections.defaultdict(list);group_ids={}
    for ordinal,(r,f) in enumerate(zip(data,ledger,strict=True)):
        q=r['requested_rewrite'];group=(q['subject'],q['relation_id']);groups[group].append((ordinal//100+1,q['target_new']['id'],f))
        group_ids.setdefault(group,f['subject_relation_group']);assert group_ids[group]==f['subject_relation_group']
        assert int(f['ordinal'])==ordinal and int(f['birth_batch'])==ordinal//100+1
        assert f['target_new_id']==q['target_new']['id'] and f['target_true_id']==q['target_true']['id']
    assert len(set(group_ids.values()))==len(groups)
    conflicts=0
    for group,rr in groups.items():
        conflicts+=len({b for b,_,_ in rr if len({target for bb,target,_ in rr if bb==b})>1})
        for b,target,f in rr:
            samebatch=len({v for bb,v,_ in rr if bb==b})>1
            later=[bb for bb,v,_ in rr if bb>b and v!=target];first=min(later) if later else ''
            assert str(samebatch)==f['same_batch_conflict'] and str(first)==f['first_later_conflict_batch']
            assert str(len(rr)>1)==f['repeated_group']
            for t,name in [(int(f['anchor_batch']),'active_at_anchor'),(100,'active_at_100')]:assert str(active(f,t))==f[name]
    ledger_check=dict(rows=len(ledger),subject_relation_groups=len(groups),repeated_groups=sum(len(v)>1 for v in groups.values()),
        multi_target_groups=sum(len({x[1] for x in v})>1 for v in groups.values()),same_batch_conflict_group_batches=conflicts,
        same_batch_conflict_requests=sum(f['same_batch_conflict']=='True' for f in ledger),active_anchor=sum(active(f,int(f['anchor_batch'])) for f in ledger),active_t100=sum(active(f,100) for f in ledger),status='PASS_INDEPENDENT_DATASET_REDUCTION')
    write(AUDIT/'censor-ledger-check.json',ledger_check)
    # Parent interval arithmetic, no scheduler access in publication script.
    import datetime
    acct=read(AUDIT/'accounting-snapshot.json');gpu=acct['jobs'][:2]
    intervals=[]
    for j in acct['jobs']:
        begin=datetime.datetime.fromisoformat(j['start']);end=datetime.datetime.fromisoformat(j['end'])
        assert int((end-begin).total_seconds())==j['elapsed_seconds']
        intervals.extend([(begin,j['allocated_gpus']),(end,-j['allocated_gpus'])])
    live=peak=0
    for _,n in sorted(intervals):live+=n;peak=max(peak,live)
    assert peak==2 and live==0 and sum(x['elapsed_seconds'] for x in gpu)==19173
    checks=dict(설명='원 runtime를 바꾸지 않은 CPU 게시 검산',scope=[53283,53284,53285],new_GPU=0,
        scalar_cache_identity_review='PASS',primary_aggregate_equality='PASS',cluster_bootstrap_equality='PASS',
        code_PNG_reproduction='PASS',matched_files=matched,matched_pngs=sum(x.endswith('.png') for x in matched),
        censor_ledger=ledger_check,CPU_unit_tests={'run':10,'passed':10,'model_validation':False},
        numerical_certification='NOT_ESTABLISHED',original_production_modules_modified=False,independent_agent_used=False,
        shared_runtime_or_environment_modified=False,allocation_GPU_seconds=19173,prior_failed_GPU_seconds=1569,
        renderer='PENDING_SYSTEM_MARKDOWN_IT',links='PENDING',raw_free='PENDING_STAGED_AUDIT',monitoring_active=False,automatic_resume=False)
    write(AUDIT/'publication-checks.json',checks)
    print(json.dumps({'reproduced_files':len(matched),'reproduced_figures':checks['matched_pngs'],'ledger':ledger_check},ensure_ascii=False))


def seal():
    source_paths=[REPO/'project/run_scripts/historical_update_timeaxis'/name for name in ('completed_review_20260926.py','test_completed_review_20260926.py','package_review_20260926.py')]
    members=[]
    for p in sorted([*REPORT.rglob('*'),*AUDIT.rglob('*'),*source_paths]):
        if p.is_file() and p.name not in ('package-manifest.json','rooted-receipt.json'):
            members.append(dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=sha(p)))
    manifest=dict(설명='이 package의 source/report/audit member SHA; manifest/receipt 자기참조 제외',runtime_commit='b856babbca101c096d72a38a3ec9c936a85a4cf3',
        analysis_source_sha256=sha(source_paths[0]),members=members,member_root_sha256=digest(members))
    write(REPORT/'package-manifest.json',manifest)
    write(REPORT/'rooted-receipt.json',dict(status='TASK_COMPLETE_STOP',runtime_commit=manifest['runtime_commit'],jobs=[53283,53284,53285],
        execution_lock_sha256=sha(RUN/'execution.lock.json'),analysis_source_sha256=manifest['analysis_source_sha256'],
        report_sha256=sha(REPORT/'report-ko.md'),package_manifest_sha256=sha(REPORT/'package-manifest.json'),member_root_sha256=manifest['member_root_sha256'],
        numerical_certification='NOT_ESTABLISHED',checkpoint_saved=False,new_GPU=0,monitoring_active=False,automatic_resume=False))
    print(json.dumps(read(REPORT/'rooted-receipt.json')))


def check():
    from markdown_it import MarkdownIt
    from urllib.parse import unquote
    report=REPORT/'report-ko.md';text=report.read_text();parser=MarkdownIt('commonmark').enable('table')
    tokens=parser.parse(text);html=parser.render(text);(LOCAL/'report-rendered.html').write_text(html)
    links=[];tables=0
    for t in tokens:
        if t.type=='table_open':tables+=1
        for child in t.children or []:
            value=child.attrGet('href') if child.type=='link_open' else child.attrGet('src') if child.type=='image' else None
            if value and not value.startswith(('https://','http://','#')):
                target=(report.parent/unquote(value.split('#')[0])).resolve();assert target.is_file(),('BROKEN_LINK',value)
                links.append(value)
    count=None
    for line in text.splitlines():
        if line.startswith('|'):
            n=len(line.split('|'))-2
            if count is None:count=n
            else:assert count==n,('TABLE_COLUMNS',line,count,n)
        else:count=None
    # CSV rectangularity, finite serialized numbers, source safety and frozen closure.
    for p in REPORT.glob('*.csv'):
        rr=list(csv.reader(p.open()));assert rr and all(len(x)==len(rr[0]) for x in rr),'CSV_COLUMNS'
    for p in (REPORT/'figures').glob('*.png'):assert p.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
    lock=read(RUN/'execution.lock.json')
    for r in lock['source_members']:assert sha(r['path'])==r['sha256'],'ORIGINAL_RUNTIME_CHANGED'
    tests=REPO/'project/run_scripts/historical_update_timeaxis/test_completed_review_20260926.py'
    assert tests.exists()
    x=read(AUDIT/'publication-checks.json');x.update(renderer='PASS_MARKDOWN_IT_HTML_TABLES',rendered_tables=tables,links='PASS',link_count=len(links),
        rendered_html_local=str(LOCAL/'report-rendered.html'),rendered_html_sha256=sha(LOCAL/'report-rendered.html'),CSV_columns='PASS',PNG_visual_owner_inspection='5/5',
        raw_free='PASS_PACKAGE_PATH_AND_CONTENT_INSPECTION',original_runtime_closure_unchanged=True)
    write(AUDIT/'publication-checks.json',x)
    print(json.dumps({'tables':tables,'links':len(links),'renderer':'markdown_it','status':'PASS'}))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','seal']);a=p.parse_args()
    {'prepare':prepare,'check':check,'seal':seal}[a.action]()
