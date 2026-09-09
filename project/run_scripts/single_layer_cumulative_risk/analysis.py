"""Observed-row analysis only; no model access, interpolation or selection by NS."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from .records import save,digest

def summary(values):
    a=np.asarray(values,dtype=np.float64)
    if not len(a) or not np.isfinite(a).all():raise ValueError('EMPTY_OR_NONFINITE_OBSERVATIONS')
    return dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max()))

def key(row):return row['panel'],row['metric'],row['case_id'],row['prompt_index']

def paired_rows(post,entry,w0):
    base={key(r):r for r in entry};origin={key(r):r for r in w0}
    assert len(base)==len(entry) and len(origin)==len(w0)
    out=[]
    for r in post:
        e,z=base[key(r)],origin[key(r)]
        assert r['identity']==e['identity']==z['identity']
        out.append(dict(**r,entry_success=e['success'],W0_success=z['success'],
             new_nll_delta=r['new_nll']-e['new_nll'],true_nll_delta=r['true_nll']-e['true_nll'],
             inherited_margin=e['margin']-z['margin'],additional_margin=r['margin']-e['margin'],
             entry_success_to_failure=bool(e['success'] and not r['success']),
             entry_failure_to_success=bool(not e['success'] and r['success'])))
    return out

def bootstrap(rows,seed=20260910,repetitions=2000,clusters=None):
    # Each request keeps every prompt in its metric; prompt rows are not IID.
    groups={}
    for r in rows:
        group=r['case_id'] if clusters is None else clusters[r['case_id']]
        groups.setdefault(group,[]).append(r)
    ordered=sorted(groups,key=str)
    totals=np.array([[sum(float(r['success'])-float(r['entry_success']) for r in groups[g]),
                      sum(r['new_nll_delta'] for r in groups[g]),len(groups[g])] for g in ordered])
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(ordered),(repetitions,len(ordered)))
    draws=totals[indices].sum(axis=1)
    return dict(cluster_count=len(groups),repetitions=repetitions,
                success_delta_ci95=np.quantile(draws[:,0]/draws[:,2],[.025,.975]).tolist(),
                new_nll_delta_ci95=np.quantile(draws[:,1]/draws[:,2],[.025,.975]).tolist(),
                scope='panel sampling variability; not edit-order or optimizer-seed variability')

def aggregate(rows):
    n=len(rows);den=sum(r['entry_success'] for r in rows)
    output=dict(numerator=sum(r['success'] for r in rows),denominator=n,
                rate=sum(r['success'] for r in rows)/n,
                strict_numerator=sum(r['true_strict'] if r['metric']=='NS' else r['new_strict'] for r in rows),
                strict_denominator=n,entry_success_denominator=den,
                loss=sum(r['entry_success_to_failure'] for r in rows),
                recovery=sum(r['entry_failure_to_success'] for r in rows),
                entry_failure_denominator=n-den,
                conditional_loss_rate=sum(r['entry_success_to_failure'] for r in rows)/den if den else None)
    for metric in ['new_nll','true_nll','margin','new_nll_delta','true_nll_delta','inherited_margin','additional_margin']:
        output.update({metric+'_'+stat:v for stat,v in summary([r[metric] for r in rows]).items()})
    output.update(bootstrap(rows))
    return output

def csv_bytes(rows):
    columns=sorted({k for row in rows for k in row});s=io.StringIO(newline='')
    writer=csv.DictWriter(s,fieldnames=columns,lineterminator='\n');writer.writeheader()
    for row in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    return s.getvalue().encode()

def write_csv(path,rows,compress=False):
    data=csv_bytes(rows)
    if compress:data=gzip.compress(data,mtime=0)
    with Path(path).open('xb') as f:f.write(data)
    return dict(path=Path(path).name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),rows=len(rows))

def collect(root,registry):
    """Registry gives exact successful native roots, avoiding automatic retry promotion."""
    aggregates=[];requests=[];trajectories=[];members=[]
    for entry,config in registry.items():
        native=Path(config['native'])
        def load(path):
            data=path.read_bytes();members.append(dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
            return json.loads(data)
        base=load(native/'ENTRY-full.json')['rows'];w0=load(native/'W0-full.json')['rows']
        endpoints=[(native/name,name.removesuffix('.json')) for name in ['W0-full.json','ENTRY-full.json','N-full.json']]
        endpoints += [(p,p.stem) for p in sorted(native.glob('native-scale-*-curve.json'))]
        for direct in config.get('direct',[]):
            d=Path(direct)
            for candidate in sorted(d.glob('*-alpha-*')):
                endpoints += [(p,candidate.name+'/'+p.stem) for p in sorted(candidate.glob('eval-*.json'))]
                for p in sorted(candidate.glob('step-*.json')):
                    trajectories.append(dict(entry=entry,endpoint=candidate.name,**load(p)))
        for path,endpoint in endpoints:
            observed=load(path);joined=paired_rows(observed['rows'],base,w0)
            for row in joined:requests.append(dict(entry=entry,endpoint=endpoint,resolution=observed['resolution'],**row))
            for panel,metric in sorted({(r['panel'],r['metric']) for r in joined}):
                group=[r for r in joined if (r['panel'],r['metric'])==(panel,metric)]
                aggregates.append(dict(entry=entry,endpoint=endpoint,panel=panel,metric=metric,resolution=observed['resolution'],**aggregate(group)))
    return aggregates,requests,trajectories,members

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    registry=json.loads(a.registry.read_text());tables=collect(a.root,registry)
    outputs=[write_csv(a.output/'paired-summary.csv',tables[0]),write_csv(a.output/'request-metrics.csv.gz',tables[1],True),
             write_csv(a.output/'trajectory.csv',tables[2])]
    save(a.output/'analysis-manifest.json',dict(inputs=tables[3],outputs=outputs,registry_sha=digest(registry),
         input_root=digest(tables[3]),output_root=digest(outputs),imputation=0,interpolation=0,model_action=0,
         scientific_promotion=False,full_campaign_completeness='MUST_BE_CHECKED_AGAINST_REQUIREMENTS_EVIDENCE'))

if __name__=='__main__':main()
