"""Observed GF sign-pair derivatives and curvature; no additional model evaluation."""
import argparse
import json
from pathlib import Path
from .analysis import key,summary,write_csv
from .records import save,digest
from .import_assets import sha

def central(minus,plus,base,epsilon):
    if epsilon<=0:raise ValueError('POSITIVE_RECORDED_AMPLITUDE_REQUIRED')
    return dict(derivative_along_GFminus=(minus-plus)/(2*epsilon),
                central_curvature=(minus+plus-2*base)/(epsilon*epsilon))

def collect(registry):
    rows=[];probes=[];inputs=[]
    def load(path):
        inputs.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
        return json.loads(path.read_text())
    for entry,config in registry.items():
        if 'B' not in config:continue
        root=Path(config['B']);native=Path(config['native'])
        probe=load(root/'direction-probes.json');base={key(r):r for r in load(native/'N-full.json')['rows']}
        for name,r in probe['probes'].items():
            probes.append(dict(entry=entry,direction=name,native_norm=probe['native_norm'],Q_rank=probe['Q_rank'],
                               GF_LF_cosine=probe['GF_LF_cosine'],**r))
        for amplitude in [.03,.1,.3]:
            pminus=root/f'GFminus-amplitude-{amplitude}';pplus=root/f'GFplus-amplitude-{amplitude}'
            minus=load(pminus/'eval.json');plus={key(r):r for r in load(pplus/'eval.json')['rows']}
            eps=amplitude*probe['native_norm']
            minus_receipt=load(pminus/'receipt.json');plus_receipt=load(pplus/'receipt.json')
            for r in minus['rows']:
                p,n=plus[key(r)],base[key(r)]
                assert r['identity']==p['identity']==n['identity']
                for metric in ['new_nll','true_nll','margin']:
                    rows.append(dict(entry=entry,amplitude=amplitude,epsilon=eps,panel=r['panel'],metric=r['metric'],
                         case_id=r['case_id'],prompt_index=r['prompt_index'],quantity=metric,resolution=minus['resolution'],
                         **central(r[metric],p[metric],n[metric],eps),
                         derivative_axis='intended physical epsilon along GFminus; actual FP32 norm recorded separately',
                         minus_actual_norm=minus_receipt['actual_extra_norm'],plus_actual_norm=plus_receipt['actual_extra_norm']))
    groups={}
    for row in rows:
        group=tuple(row[k] for k in ['entry','amplitude','panel','metric','quantity','resolution'])
        groups.setdefault(group,[]).append(row)
    aggregates=[]
    for values,group in sorted(groups.items()):
        r=dict(zip(['entry','amplitude','panel','metric','quantity','resolution'],values));r['denominator']=len(group)
        for name in ['derivative_along_GFminus','central_curvature']:
            r.update({name+'_'+stat:value for stat,value in summary([x[name] for x in group]).items()})
        aggregates.append(r)
    return rows,aggregates,probes,inputs

def main():
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows,aggregates,probes,inputs=collect(json.loads(a.registry.read_text()))
    a.output.mkdir(parents=True,exist_ok=False)
    outputs=[write_csv(a.output/'GF-sign-paired.csv.gz',rows,True),write_csv(a.output/'GF-central-summary.csv',aggregates),
             write_csv(a.output/'direction-probes.csv',probes)]
    save(a.output/'direction-analysis-manifest.json',dict(inputs=inputs,outputs=outputs,input_root=digest(inputs),
         output_root=digest(outputs),imputation=0,new_model_action=0,discrete_success_derivative_claim=False))

if __name__=='__main__':main()
