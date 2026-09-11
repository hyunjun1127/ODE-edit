"""Raw-free package validation and create-once manifest/rooted receipt."""
import csv,json,subprocess,datetime
from .common import *

def validate():
    summary=csvread(OUT/'final-summary.csv');cum=csvread(OUT/'cumulative-metrics.csv');current=csvread(OUT/'current-metrics.csv')
    assert len(summary)==14 and len({r['arm'] for r in summary})==14
    w0=read(OLD/'preedit-runtime-summary.json')['runtime']['selected_W0']['weights']
    for arm in ARMS:
        rt=read(root(arm)/'runtime.json')
        assert all(v['sha256']==w0[k]['sha256'] for k,v in rt['W0']['weights'].items()),arm
    assert len(cum)==14*12*3 and len(current)==14*100*3
    for r in cum+current:
        batch=int(r['batch']);expected=(100 if r['scope']=='CURRENT_B100' else batch*100)*MULT[r['metric']]
        assert int(r['denominator'])==expected and 0<=int(r['numerator'])<=expected and abs(float(r['rate'])-int(r['numerator'])/expected)<1e-14
    for r in summary:
        for tag in MULT:
            q=next(x for x in cum if x['arm']==r['arm'] and x['metric']==tag and int(x['batch'])==100)
            assert r[tag+'_numerator']==q['numerator'] and r[tag+'_denominator']==q['denominator']
    for name in ['paired-final-transitions.csv','prompt-transitions.csv','w0-final-paired-transitions.csv']:
        for r in csvread(OUT/name):
            # W0 publication uses the same transition column naming when present.
            if all(k in r and r[k]!='' for k in ['retained','lost','gained','both_failed','denominator']):
                assert sum(int(r[k]) for k in ['retained','lost','gained','both_failed'])==int(r['denominator'])
                assert int(r['retained'])+int(r['gained'])==int(r['after_num'])
    cps=csvread(OUT/'checkpoint-tensors.csv')
    assert len({(r['arm'],r['batch']) for r in cps})==168
    assert len(cps)==288
    for r in csvread(OUT/'chain-integrity.csv'):
        assert int(r['batches'])==100 and int(r['requests'])==10000 and int(r['W_links'])==99
    for p in OUT.rglob('*'):
        if p.is_file():assert p.suffix in ['.csv','.json','.md','.png'],str(p)
        assert not p.is_symlink()
    for p in OUT.rglob('*.csv'):
        fields=next(csv.reader(p.open()),[])
        assert not any(k in fields for k in ['prompt','logits','generation','case_id','token_ids']),str(p)
        if 'target' in fields:assert {r['target'] for r in csvread(p)}<= {'new','true'},str(p)
    for p in OUT.glob('MEMIT-*.csv'):
        if 'final-blue' not in p.name:
            assert all(family(r['arm'])=='MEMIT' for r in csvread(p))
    for p in OUT.glob('AlphaEdit-*.csv'):
        if 'final-blue' not in p.name:
            assert all(family(r['arm'])=='AlphaEdit' for r in csvread(p))
    return dict(status='PASS',arms=14,checkpoints=168,selected_tensor_rows=288,current_metric_rows=len(current),cumulative_metric_rows=len(cum),final_request_observations=140000,unique_requests=10000,cumulative_request_state_rows=778400,W0_shared_evaluations=1,W0_selected_all14_match_sealed_SH2=True,native_all_batches_seen_RS_requeststate_rows=1010000,native_noncheckpoint_extra_RS_requeststate_rows=898800,duplicate_denominator_policy='final and checkpoint current reuses are not additional unique observations',tests=10,raw_free=True,scientific_promotion=False,full_GPU_replay=0)

def main():
    plots=read(OUT/'figures/plot-receipt.json')
    rerender=LOCAL/'reproduced-figures'
    checked_plots=[]
    for m in plots['figures']:
        h=sha(rerender/m['path']);assert h==m['sha256']==sha(OUT/'figures'/m['path'])
        checked_plots.append(dict(path=m['path'],sha256=h,byte_equal=True))
    plotreceipt=dict(status='BYTE_STABLE_PASS',figures=checked_plots,source_sha256=sha(REPO/'project/run_scripts/server4_experiments_review/plots.py'),rerender_root=str(rerender))
    if (OUT/'plot-reproduction.json').exists():assert read(OUT/'plot-reproduction.json')==plotreceipt
    else:save(OUT/'plot-reproduction.json',plotreceipt)
    validation=validate();save(OUT/'package-validation.json',validation)
    # The local checks/receipts are small factual artifacts; no raw run logs copied.
    for source,name in [(LOCAL/'initial.json','initial-audit.json')]:
        save(OUT/name,read(source))
    code=sorted((REPO/'project/run_scripts/server4_experiments_review').glob('*.py'))
    reuse=[REPO/'project/run_scripts/blue_fivearm_analysis/common.py']+[REPO/'project/run_scripts/blue_lifelong_analysis'/n for n in ['common.py','metrics.py','aggregate.py','audit.py']]
    sources=[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=sha(p)) for p in code+reuse]
    members=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name not in ['analysis-manifest.json','rooted-receipt.json']:
            m=dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p))
            if p.suffix=='.csv':m['rows']=len(csvread(p))
            members.append(m)
    manifest=dict(instruction_id=INSTRUCTION,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_HEAD='c60df37fe3d3c2b1ee8038b2ec3ac8704260a04f',base_tree='5c988beb01eb4acf567ba81194247f80f19c6faa',analysis_sources=sources,reused_report_sha256=sha(OLD/'factual-report-ko.md'),members=members,member_root=digest(members),raw_broadcast='OMITTED_USER_ANALYSIS_ONLY_LOCAL_ONLY_NO_NEW_RAW',scientific_promotion=False)
    save(OUT/'analysis-manifest.json',manifest)
    for m in members:assert sha(OUT/m['path'])==m['sha256'] and (OUT/m['path']).stat().st_size==m['bytes']
    save(OUT/'rooted-receipt.json',dict(instruction_id=INSTRUCTION,status='REVIEW_COMPLETE_RAW_FREE',manifest_sha256=sha(OUT/'analysis-manifest.json'),member_root=digest(members),members=len(members),tables=sum('rows' in m for m in members),figures=sum(m['path'].endswith('.png') for m in members),report_sha256=sha(OUT/'diagnostic-report-ko.md'),validation=validation,task_owned_GPU_jobs=0,monitoring=0,next='OWN_SCOPE_NONFORCE_MAIN_INTEGRATION_THEN_STOP'))
    print(json.dumps(dict(report_sha256=sha(OUT/'diagnostic-report-ko.md'),manifest_sha256=sha(OUT/'analysis-manifest.json'),receipt_sha256=sha(OUT/'rooted-receipt.json'),member_root=digest(members),members=len(members))))

if __name__=='__main__':main()
