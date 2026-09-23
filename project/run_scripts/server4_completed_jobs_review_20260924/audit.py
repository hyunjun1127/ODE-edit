"""Terminal metadata/source/state audit; never invokes Slurm or a model."""
import ast
import csv
import json
import re
from collections import Counter,defaultdict
from datetime import datetime
from .review import ROOT,REPO,PACKAGE,SCRATCH,WRITERS,ENTRIES,BRANCHES,read,sha,table,write_json
import numpy as np

# Read-only scheduler tool output from the initial snapshot, transcribed without
# inferring state from historical reports. All timestamps below are Asia/Seoul.
ACCOUNTING='''50974|slmf_T0|FAILED|1:0|2026-09-19T21:01:50|2026-09-20T01:50:04|2026-09-20T01:52:20|136|1
50983|slmf_S10|CANCELLED by 1025|0:0|2026-09-19T22:14:35|None|2026-09-20T02:20:06|0|0
51055|slmf_repair|FAILED|1:0|2026-09-20T02:13:17|2026-09-20T02:13:17|2026-09-20T02:15:01|104|1
51056|slmf_S10r2|FAILED|1:0|2026-09-20T02:37:05|2026-09-20T02:37:05|2026-09-20T02:42:40|335|1
51057|slmf_B1r3|FAILED|1:0|2026-09-20T03:15:09|2026-09-20T03:15:09|2026-09-20T03:24:07|538|1
51058|slmf_B1r4|COMPLETED|0:0|2026-09-20T03:37:37|2026-09-20T03:37:37|2026-09-20T04:38:06|3629|1
51260|en_adapt_B300|FAILED|1:0|2026-09-20T18:36:31|2026-09-20T18:36:31|2026-09-20T20:20:09|6218|1
52527|alpha_key_gate|FAILED|1:0|2026-09-23T07:00:36|2026-09-23T09:59:04|2026-09-23T09:59:48|44|1
52528|alpha_key_geometry|CANCELLED|0:0|2026-09-23T07:00:36|None|2026-09-23T10:00:04|0|0
52529|alpha_key_writers|CANCELLED|0:0|2026-09-23T07:00:36|None|2026-09-23T10:00:04|0|0
52530|alpha_key_reduce|COMPLETED|0:0|2026-09-23T07:00:36|2026-09-23T10:00:04|2026-09-23T10:00:04|0|0
52563|alpha_key_gate|COMPLETED|0:0|2026-09-23T10:37:05|2026-09-23T10:43:34|2026-09-23T10:46:50|196|1
52564|alpha_key_geometry|COMPLETED|0:0|2026-09-23T10:37:05|2026-09-23T10:47:04|2026-09-23T16:04:53|19069|1
52565|alpha_key_writers|FAILED|1:0|2026-09-23T10:37:06|2026-09-23T12:03:04|2026-09-23T12:27:58|1494|1
52566|alpha_key_reduce|COMPLETED|0:0|2026-09-23T10:37:06|2026-09-23T16:05:04|2026-09-23T16:06:57|113|0
52575|alpha_key_writers_r4|COMPLETED|0:0|2026-09-23T12:59:06|2026-09-23T15:33:31|2026-09-23T23:50:13|29802|1
52576|alpha_key_reduce_r4|COMPLETED|0:0|2026-09-23T12:59:06|2026-09-23T23:50:23|2026-09-23T23:55:25|302|0'''

def metadata():
    rows=[]
    for line in ACCOUNTING.splitlines():
        j,n,s,x,submit,start,end,secs,gpu=line.split('|')
        rows.append(dict(job_id=j,job_name='odeedit_'+n+'_s4',owner='janghj',state=s,exit_signal=x,submit=submit,start=start,end=end,
            elapsed_seconds=int(secs),allocated_gpu=int(gpu),allocated_gpu_seconds=int(secs)*int(gpu),
            timezone='Asia/Seoul',scope='FRESH_ALPHA_REVIEW' if 'alpha' in n else 'PRIOR_COMPLETED_REVIEW_REUSED',
            dependency='',source='',tree='',lock='',archive_sha256='',command='',raw='',requested_mem='59G' if gpu=='1' else '',requested_cpus=8 if gpu=='1' else 0))
    sources=[]
    for attempt,control in [('r1',ROOT/'control'),('r3',ROOT/'controls/attempt-r3'),('r4',ROOT/'controls/attempt-r4')]:
        sub=read(control/'submission.json');lock=read(control/'execution.lock.json')
        inspect=read(control/'held-inspection.json');inspect=inspect.get('inspection',inspect.get('jobs',inspect))
        for phase,j in sub['jobs'].items():
            row=next(r for r in rows if r['job_id']==str(j))
            row.update(dependency=sub.get('dependencies',sub.get('dependency_graph',{})).get(phase),
                source=lock.get('source_commit'),tree=lock.get('source_tree'),lock=str(control/'execution.lock.json'),
                lock_sha256=sha(control/'execution.lock.json'),archive_sha256=lock.get('archive',{}).get('sha256'),
                command=str(control/(phase+'.sh')),raw=str(ROOT/'execution'/('attempt-'+attempt)),
                full_argv=(control/(phase+'.sh')).read_text(),requested_mem='8G' if phase=='reduce' else '59G',requested_cpus=4 if phase=='reduce' else 8)
        src=__import__('pathlib').Path(lock['repo'])
        for m in lock.get('execution_source_members',[]):
            p=src/m['relative_path'];exists=p.exists();actual=sha(p) if exists else None
            sources.append(dict(attempt=attempt,path=str(p),exists=exists,sha256=actual,expected=m['sha256'],matches=actual==m['sha256']))
    slmf=__import__('pathlib').Path('/data/janghj/ODE-edit/local/single-layer-mechanism-first/20260919-v1')
    subs=[slmf/'T0/hook-attempt-v1/submission.json',*sorted((slmf/'PROGRAM').glob('*/submission.json')),
          __import__('pathlib').Path('/data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/server4-migration-r1/attempt-v1/submission.json')]
    for p in subs:
        s=read(p);jid=str(s.get('job_id',s.get('job','')))
        candidates=[r for r in rows if r['job_id']==jid]
        if not candidates:continue
        r=candidates[0];lp=s.get('lock',{})
        lp=lp.get('path') if isinstance(lp,dict) else lp
        if not lp:lp=str(p.parent/'execution.lock.json')
        lock=read(lp) if __import__('pathlib').Path(lp).exists() else {}
        command=s.get('command',[])
        r.update(source=s.get('source',lock.get('source_commit')),tree=lock.get('source_tree',lock.get('tree','SEE_PRIOR_REVIEW')),
                 lock=lp,lock_sha256=sha(lp) if __import__('pathlib').Path(lp).exists() else 'NOT_AVAILABLE',
                 raw=str(p.parent),full_argv=json.dumps(command),submission_receipt=str(p),submission_sha256=sha(p),requested_cpus=8,requested_mem='59G',
                 dependency=next((x.split('=',1)[1] for x in command if isinstance(x,str) and x.startswith('--dependency=')),''))
    table('job-inventory.csv',rows);table('frozen-source-members.csv',sources)
    events=[]
    for r in rows:
        if r['allocated_gpu'] and r['start']!='None':
            events.extend([(r['start'],1,r['job_id']),(r['end'],-1,r['job_id'])])
    active=set();peak=0;interval=[]
    for t,d,j in sorted(events,key=lambda x:(x[0],x[1])):
        if d==1:active.add(j)
        else:active.remove(j)
        peak=max(peak,len(active));interval.append(dict(time=t,job=j,event='start' if d==1 else 'end',gpus=len(active),active=';'.join(sorted(active))))
    table('allocation-intervals.csv',interval)
    write_json(PACKAGE/'accounting-receipt.json',dict(snapshot_utc='2026-09-23T15:52:05Z',
        current_queue_empty=True,query_owner='janghj',census_node='server4',census_start='2026-09-20T00:00:00+09:00',
        exact_terminal_parents=[r['job_id'] for r in rows],supplement_50983='exact metadata-linked query; no execution',
        provenance='squeue/sacct read-only tool results transcribed; no status extrapolation',max_concurrent_gpus=peak,
        alpha_gpu_seconds=sum(r['allocated_gpu_seconds'] for r in rows if 'alpha' in r['job_name']),
        all_recent_gpu_seconds=sum(r['allocated_gpu_seconds'] for r in rows),batch_extern_added=False,
        commands=['squeue -h -u janghj','sacct -X -u janghj -N server4 -S 2026-09-20T00:00:00 -E now','sacct -X -j exact IDs'],
        scheduler_mutations=0,monitoring_active=False))
    reports=[]
    for p in sorted((REPO/'experiment-reports/servers/server4').rglob('*.md')):
        if PACKAGE in p.parents:continue
        text=p.read_text();reports.append(dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=sha(p),
            title=next((x.lstrip('# ') for x in text.splitlines() if x.startswith('#')),''),
            evidence='HISTORICAL_PUBLISHED_REPORT_INDEX_NOT_RAW_REREDUCED'))
    table('historical-report-index.csv',reports)
    print('METADATA',len(rows),'reports',len(reports),'maxGPU',peak,flush=True)

def geometry():
    root=ROOT/'execution/attempt-r3/geometry';rows=[];errors=[]
    for p in sorted(root.glob('E*/*/*/*/L*-*-geometry.json')):
        obj=read(p)
        for label,g in [('writer_mean',obj['writer_mean'])]+[(f'context{i}',v) for i,v in enumerate(obj['contexts'])]:
            meta=g['metadata'];energies=np.array(g['norm_energy_per_sample'],dtype=np.float64)
            for variant in ('raw','centered','unit_raw','unit_centered'):
                v=g[variant];eig=np.array(v['eigenvalues_descending'],dtype=np.float64);trace=float(eig.sum());pr=trace*trace/float(eig@eig) if eig@eig else None
                if pr is not None:assert np.isclose(pr,v['participation_ratio'],rtol=1e-10,atol=1e-10)
                rows.append(dict(phase=meta['phase'],state=meta['state'],condition=meta['condition'],panel=meta['panel'],layer=meta['layer'],space=obj['space'],
                    context=label,variant=variant,n=g['n'],PR=pr,top1=float(eig[0]/trace) if trace else None,top5=float(eig[:5].sum()/trace) if trace else None,
                    trace=trace,mean_energy=g['mean_energy_fraction'],norm_ESS=float(energies.sum()**2/(energies@energies)),
                    norm_top1pct=float(np.sort(energies)[-g['top1pct_sample_count']:].sum()/energies.sum()),zero_norm=g['zero_norm_count'],source=str(p)))
            assert np.isclose(energies.sum()**2/(energies@energies),g['norm_energy_ess'],rtol=1e-10)
    full=table(str(SCRATCH/'geometry-all-context-spectrum-arithmetic.csv'),rows)
    table('geometry-independent-spectrum-arithmetic.csv',[r for r in rows if r['context']=='writer_mean'])
    groups=defaultdict(list)
    for r in rows:
        if r['context']!='writer_mean':groups[tuple(r[k] for k in ('phase','state','condition','panel','layer','space','variant'))].append(r)
    table('geometry-context-ranges.csv',[dict(zip(('phase','state','condition','panel','layer','space','variant'),k),
        contexts=len(v),PR_min=min(x['PR'] for x in v),PR_max=max(x['PR'] for x in v),PR_mean=float(np.mean([x['PR'] for x in v])),
        mean_energy_min=min(x['mean_energy'] for x in v),mean_energy_max=max(x['mean_energy'] for x in v)) for k,v in groups.items()])
    rec=read(root/'cell-receipts.json');table('geometry-observation-coverage.csv',rec)
    write_json(PACKAGE/'geometry-audit.json',dict(rows=len(rows),panel_observations=len(rec),
        verification='Independent PR/top-energy/ESS from saved eigenvalues and per-request norms; no model rerun; full eigenvectors not reconstructed',
        full_context_arithmetic_path=str(full),full_context_arithmetic_sha256=sha(full),
        raw_tensor_recompute='Representative CPU check recorded separately',aggregate_source_overwrite=False))
    print('GEOMETRY',len(rows),flush=True)

def states():
    rows=[];events=[];num=[];components=[];counters=Counter()
    for e in ENTRIES:
        for b in BRANCHES:
            p=WRITERS/f'W{e:03d}/{b}/write/terminal.json';o=read(p);counters.update(o['counters'])
            assert o['status']=='COMPLETED' and len(o['history'])==5 and o['counters']['history_appends']==5
            assert o['counters']['final_l8_residual_calls']==1 and o['new_W_M_RNG_checkpoint_saved'] is False
            assert [s['stage'] for s in o['stages']]==['entry','z','W4','W5','W6','W7','W8','history']
            for h in o['history']:
                assert h['append_completed'] and h['columns']==100
                rows.append(dict(entry=e,branch=b,layer=h['layer'],preM=h['pre_M_sha256'],postM=h['post_M_sha256'],key=h['key_sha256'],
                    validation=h['validation'],independent_fullM=h['independent_full_M_reconstruction'],reused=(e==50 and b in ('NATIVE','SHAM'))))
            for event in o['events']:events.append({**event,'entry':e,'branch':b,'reused':e==50 and b in ('NATIVE','SHAM')})
            for obs in (p.parent.parent/'stages').glob('*/observation.json'):
                z=read(obs);assert z['observer_nonmutation'] and z['history_appends']==0
            if b=='SHAM':
                z=read(p.parent.parent/'sham-control.json')
                for d in z['differences']:num.append(dict(entry=e,kind='SHAM',status=z['status'],comparison=z.get('comparison_verdict'),**d))
        for p in (WRITERS/f'W{e:03d}/components').glob('L*/full-hook-physical-parity.json'):
            z=read(p);num.append(dict(entry=e,kind='HOOK',layer=p.parent.name,**z))
        for p in (WRITERS/f'W{e:03d}/components').rglob('observer-nonmutation.json'):
            z=read(p);assert z['state_before']==z['state_after'];components.append(dict(entry=e,path=str(p),**{k:v for k,v in z.items() if not isinstance(v,(dict,list))}))
    table('history-exactly-once.csv',rows);table('writer-events.csv',events);table('numerical-comparisons.csv',num);table('component-nonmutation.csv',components)
    write_json(PACKAGE/'state-audit.json',dict(branches=24,layer_history_appends=len(rows),runtime_counters=dict(counters),
        new_r4_appends=sum(not r['reused'] for r in rows),reused_r3_appends=sum(r['reused'] for r in rows),component_observers=len(components),
        whole_W_M_reconstruction='NOT_PERFORMED',GPU_continuation='NOT_TESTED',no_new_full_state_checkpoint=True))
    print('STATE',len(rows),len(components),flush=True)

if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
