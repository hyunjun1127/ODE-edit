"""Read-only CPU W20 publication; no model, scheduler, evaluation or retry.

Recompute PRICE raw-row metrics/state/controller; baseline comparisons reuse
published integer-count own-W20 tables, not final-W100 or new paired raw.
"""
import csv
import io
import json
import math
import sys
from pathlib import Path
from . import collect as C
from .review_w15 import ATTEMPT, SOURCE, CONFIG, LOCK, compact_stats

OLD = 'experiment-reports/servers/server4/jlz-v13-mdcd-sequential-2k-20261005-v1/review-20261005-r1/'
V12 = 'experiment-reports/servers/server3/jlz-v12-shared-budget-bs100x20-20261004-v1/review-20261005-v1/'
BLUE = 'experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/'
HISTORY = 'experiment-reports/servers/server3/memit-history-fixed10k-20260928-v1/completion-review-r1/all-seen-metrics.csv'
CAKE = 'experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v2/seen-prefix.csv'
DEN = dict(R=2000, P=4000, N=20000)


def table(reader, rel):
    return list(csv.DictReader(io.StringIO(reader.bytes(C.ROOT/rel).decode())))


def historical(reader):
    binding=reader.json(C.ROOT/('audits/servers/server4/jlz-v13-mdcd-sequential-2k-20261005-v1/review-20261005-r1/historical-baseline-binding.json'))
    for item in [binding['source'],binding['source_manifest']]+binding['underlying_tables']:
        p=C.ROOT/item['path'];data=reader.bytes(p)
        C.require(len(data)==item['bytes'] and C.sha(p)==item['sha256'],'HISTORICAL_TABLE_SHA_SIZE')
    grouped={}
    def add(name,rows,kind='metric',num='numerator',den='denominator',source=''):
        result={}
        for row in rows:
            k=row[kind][0]
            C.require(k in DEN and k not in result,'EXACT_HISTORICAL_METRIC_SET')
            n,d=int(row[num]),int(row[den]);C.require(d==DEN[k] and 0<=n<=d,'HISTORICAL_W20_DENOMINATOR')
            result[k]=dict(numerator=n,denominator=d,rate=n/d)
        C.require(set(result)==set(DEN),'HISTORICAL_COMPLETE_RPN')
        grouped[name]=dict(metrics=result,source=source,scope='HISTORICAL_REFERENCE')
    add('MEMIT-H',[r for r in table(reader,HISTORY) if r['batch']=='20'],source=HISTORY)
    for family in ('AlphaEdit','MEMIT'):
        source=BLUE+family+'-cumulative-metrics.csv';rows=table(reader,source)
        for arm,name in ((family+'_ORIGINAL',family+'-BLUE'),('BASE_'+family.upper(),family)):
            selected=[r for r in rows if r['batch']=='20' and r['arm']==arm]
            C.require(all(r['scope']=='CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS' for r in selected),'OWN_W20_NOT_FINAL_W100')
            if arm.endswith('_ORIGINAL'):
                C.require(all(r['arm_display_label']==family+'_BLUE (L4+L8)' for r in selected),'CORRECT_DISPLAY_LABEL_NOT_LEGACY_ARM_NAME')
            add(name,selected,source=source)
    # Rewrite-only and full-seen R coexist: select the complete R/P/N endpoint.
    add('CAKE',[r for r in table(reader,CAKE) if r['batch']=='20' and r['population']=='ACTUAL_FULL_SEEN'],source=CAKE)
    add('v12-MEMIT',[r for r in table(reader,V12+'endpoint-metrics.csv') if r['endpoint']=='20'],
        kind='family',num='success',den='n',source=V12+'endpoint-metrics.csv')
    for arm in ('MD','CD'):
        add('V13 '+arm,[r for r in table(reader,OLD+'comparison-W20.csv') if r['endpoint']=='W20' and r['arm']==arm],
            kind='kind',source=OLD+'comparison-W20.csv')
    # Previous published rounded displays are cross-checks, never HM inputs.
    for row in table(reader,OLD+'baseline-W20.csv'):
        ref=grouped[row['method']]['metrics']
        C.require(row['endpoint']=='W20' and row['requests']=='2000','PREVIOUS_BASELINE_ENDPOINT')
        for k in DEN:C.require(math.isclose(100*ref[k]['rate'],float(row[k+'S_percent']),abs_tol=1e-10),'BASELINE_REUSE_COUNTS')
        C.require(math.isclose(100*C.harmonic(ref),float(row['score_harmonic_percent']),abs_tol=1e-10),'BASELINE_HARMONIC_RECOMPUTED')
    return grouped


def write_csv(path, rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    out=Path(sys.argv[1]).resolve();C.require(not out.exists(),'CREATE_ONCE')
    reader=C.Reader();c=reader.json(ATTEMPT/'config.json');lock=reader.json(ATTEMPT/'execution.lock.json')
    C.require(C.sha(ATTEMPT/'config.json')==CONFIG and C.sha(ATTEMPT/'execution.lock.json')==LOCK
              and lock['source_commit']==SOURCE and c['execution_arms']==['PRICE'],'ACTUAL_EXECUTION_IDENTITY')
    for item in lock['source_members']:
        data=reader.bytes(item['path']);C.require(len(data)==item['bytes'] and C.sha(item['path'])==item['sha256'],'SOURCE_HASH')
    for rel in ('project/run_scripts/jlz_interference_l1/collect.py','project/run_scripts/jlz_interference_l1/w0_reuse.py',
                'project/run_scripts/jlz_realized_writer_sequential/review_completed.py'):
        C.require(C.sha(C.ROOT/rel)==C.sha(ATTEMPT/'source'/rel),'REVIEW_REUSED_FROZEN_SOURCE')
    records=C.load_prefix(Path(c['stream']).parent,2000)
    C.require(C.digest([r['case_id'] for r in records])==C.ORDERED_SHA,'FIRST2000_ORDER')
    identities=reader.bound(c['observer_identity'])['rows']
    arm,prefix=C._arm_review(reader,ATTEMPT,c,lock,'PRICE',records,identities,{})
    C.require(arm['status']=='W20_COMPLETE' and arm['commits']==20 and arm['actual']==dict(joins=19,history_appends=100)
              and set(prefix)=={5,10,15,20} and not arm['first_error'] and not arm['uncommitted_attempts'],'W20_FULL_COVERAGE')
    term=arm['terminal'];C.require(term['source']==SOURCE and term['job']=='59768' and term['no_B21'],'GPU_TERMINAL')
    collector_term=reader.json(ATTEMPT/'collector/terminal.json')
    C.require(collector_term['scientific_coverage_complete'] and collector_term['source']==SOURCE,'COLLECTOR_TERMINAL')
    inv=reader.bound(collector_term['inventory'])
    for item in inv['files']:
        data=reader.bytes(item['path']);C.require(len(data)==item['bytes'] and C.sha(item['path'])==item['sha256'],'COLLECTOR_FILE_SHA')
    stored=reader.json(ATTEMPT/'collector/reduction.json')
    for key in ('metrics','paired','counters','actual','cost'):
        C.require(arm[key]==stored['arms']['PRICE'][key],'CPU_REDUCTION_COLLECTOR_PARITY_'+key)
    base=historical(reader)
    metrics=next(p['metrics'] for p in arm['metrics'] if p['endpoint']=='W20_ALL_SEEN')
    C.require({k:v['denominator'] for k,v in metrics.items()}==DEN,'W20_EXACT_DENOMINATORS')
    w0=next(p['metrics'] for p in arm['metrics'] if p['endpoint']=='W0')
    rows=[]
    groups=[('W0 (ours)',dict(metrics=w0,scope='SAME_RUNTIME_W0_EXACT_REUSE',source=c['W0_reuse']['prior_source'])),
            ('PRICE (ours)',dict(metrics=metrics,scope='CURRENT_CPU_RAW_VERIFIED',source=SOURCE))]+list(base.items())
    for name,group in groups:
        m=group['metrics'];is_w0=name=='W0 (ours)'
        row=dict(method=name,endpoint='W0' if is_w0 else 'W20',edits=0 if is_w0 else 2000,
                 evaluated_requests=2000,scope=group['scope'])
        for k in DEN:row.update({k+'_numerator':m[k]['numerator'],k+'_denominator':m[k]['denominator'],k+'S_percent':100*m[k]['rate']})
        row['harmonic_mean_percent']=100*C.harmonic(m);row['source']=group['source'];rows.append(row)
    realization=[]
    for batch in arm['realization']:
        for layer,data in batch['realization']['layers'].items():
            for role in ('mean','canonical','rewrite','KL'):
                values=data[role];values=[values] if isinstance(values,dict) else values
                entry=dict(batch=batch['batch'],layer=int(layer),role=role)
                for k in ('normratio','directionalratio','cosine','relative_error','zero_target_leakage'):
                    entry.update({k+'_'+n:v for n,v in compact_stats(r[k] for r in values).items()})
                entry.update({k:data[k] for k in ('ideal_Q','effective_Q','Q_C0','Q_H','effective_update_norm')});realization.append(entry)
    summary=dict(status='PRICE_W20_CPU_REVIEWED',execution_source=SOURCE,config_sha256=CONFIG,lock_sha256=LOCK,
        commits=20,joins=19,history_appends=100,counters=arm['counters'],metrics=arm['metrics'],paired=arm['paired'],
        cost=arm['cost'],terminal=term,accounting=stored['accounting'],W0_observation_cost=arm['W0_observation_cost'],
        baseline_scope='HISTORICAL_OWN_W20_AGGREGATES_NO_CROSS_RUN_PAIRED_RAW',comparison=rows,
        harmonic_formula='100*3/(1/(Rnum/Rden)+1/(Pnum/Pden)+1/(Nnum/Nden)); any zero => 0',
        raw_read_files=len(reader.files),reviewer='OWNER_USING_INDEPENDENT_STDLIB_ROW_REDUCER',
        separate_reviewer=False,model_load=0,new_model_forward=0,new_slurm=0,checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
    C.require('torch' not in sys.modules and 'transformers' not in sys.modules,'NO_MODEL_IMPORT')
    for item in reader.files.values():C.require(C.sha(item['path'])==item['sha256'],'IMMUTABLE_READ_SET')
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (out/'input-manifest.json').write_text(json.dumps(dict(files=list(reader.files.values())),indent=2)+'\n')
    write_csv(out/'comparison-W20.csv',rows);write_csv(out/'realization.csv',realization)
    allmetrics=[]
    for point in arm['metrics']:
        for kind,m in point['metrics'].items():
            allmetrics.append(dict(endpoint=point['endpoint'],kind=kind,**{k:m[k] for k in
                ('numerator','denominator','rate','strict_numerator','strict_denominator','strict_rate','token_micro','prompt_macro',
                 'true_nll_mean','new_nll_mean')},harmonic_mean_percent=100*C.harmonic(point['metrics'])))
    write_csv(out/'metrics.csv',allmetrics)
    paired=[]
    for p in arm['paired']:
        for k,types in p['paired'].items():
            for met,v in types.items():paired.append(dict(from_endpoint=p['from_endpoint'],to_endpoint=p['to_endpoint'],
                birth_cohort=p.get('birth_cohort',''),kind=k,metric=met,**v))
    write_csv(out/'paired-cohorts.csv',paired)
    write_csv(out/'cost.csv',[{k:v for k,v in row.items() if isinstance(v,(int,float))} for row in arm['cost']])
    print(json.dumps(dict(status=summary['status'],counts={k:[v['numerator'],v['denominator']] for k,v in metrics.items()},
        harmonic_mean_percent=100*C.harmonic(metrics),historical_methods=list(base),files=len(reader.files))))


if __name__=='__main__':main()
