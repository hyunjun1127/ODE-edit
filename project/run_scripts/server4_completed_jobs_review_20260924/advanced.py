"""Independent CPU coverage, masked metrics, causal-matrix arithmetic."""
from .review import *
from collections import Counter,defaultdict
import re

def run():
    kr=[];pair_rows=[];masked=[];ntrans=[];coverage=[];panels=[]
    geo=ROOT/'execution/attempt-r3/geometry'
    cells=list(csv.DictReader((ROOT/'inputs/design/cells.csv').open()))
    for cell in cells:
        phase=cell['phase'];cid=cell['cell_id'];p=None
        if phase=='E1':
            _,state,panel=cid.split('-');p=geo/'E1'/state/'native'/panel/'capture.json'
        elif phase=='E2':
            m=re.match(r'E2-W(\d+)-(.*)',cid);state,condition=m.groups();d=geo/'E2'/f'W{int(state):03d}'/condition
            if condition=='upper-control':p=d.parent/'upper-control-parity.json'
            else:p=d/'calibration512'/'capture.json'
            captures=[read(d/panel/'capture.json') for panel in ('calibration512','assessment-early','assessment-middle','assessment-onset','assessment-late')]
            ids=[set(x['case_ids']) for x in captures]
            assert [len(x) for x in ids]==[512,872,872,872,872]
            assert sum(len(x) for x in ids)==len(set.union(*ids))==4000
            if condition=='upper-control':assert all(x['bitwise_equal'] for x in read(p).values())
        elif phase in ('E3','E4-H'):
            m=re.match(r'E[34]-W(\d+)-(.*)',cid);e,b=m.groups();b='NATIVE' if phase=='E3' else b;p=WRITERS/f'W{int(e):03d}'/b/'write/terminal.json'
        elif phase in ('E4-W','E4-KR'):
            m=re.match(r'E4-W(\d+)-L(\d+)-(.*)',cid);e,l,f=m.groups();e,l=int(e),int(l)
            if phase=='E4-KR':l-=1;family='KR';variants=('aa','ab','ba','bb')
            elif f=='matrix':family='matrix';variants=('parallel','perpendicular','norm_control')
            else:family='components';variants=('no','mean','centered','full')
            ps=[WRITERS/f'W{e:03d}/components/L{l}'/family/v/'observer/current.json' for v in variants]
            assert all(x.exists() for x in ps),(cid,ps)
            p=ps[0]
        if phase in ('SEQ','ORDER','FUTURE'):status='FOLLOWUP_NOT_SUBMITTED'
        else:status='COMPLETED' if p and p.exists() else 'NOT_RECORDED'
        coverage.append(dict(phase=phase,cell_id=cid,status=status,evidence=str(p) if p else '',
            numerical_equivalence='OBSERVATION_ONLY_NOT_GENERAL_PASS' if phase.startswith('E4') else 'SEE_CONFORMANCE'))
    table('family-coverage.csv',coverage)
    for e in ENTRIES:
        for b in BRANCHES:
            p=WRITERS/f'W{e:03d}/{b}/stages/history/H512.json';o=read(p);data=reduce_raw(p);mask={r['case_id']:r for r in o['compact']['masks']['rows']}
            for subset,key in [('ACTIVE','functional_current_valid'),('SUPERSEDED','functional_superseded')]:
                for metric,rs in data.items():
                    part=[r for r in rs if mask[r['case_id']][key]]
                    if part:masked.append(dict(entry=e,branch=b,subset=subset,metric=metric,**aggregate(part)))
            paths={s:WRITERS/f'W{e:03d}/{b}/stages/{s}/N512.json' for s in ('entry','W4','W5','W6','W7','W8','history')}
            dataN={s:reduce_raw(p)['NS'] for s,p in paths.items()}
            for a,z in zip(list(dataN),list(dataN)[1:]):ntrans.append(dict(entry=e,branch=b,from_stage=a,to_stage=z,**paired(dataN[a],dataN[z])))
            if b=='NATIVE':nativeN=dataN['history']
            else:pair_rows.append(dict(entry=e,layer='',family='E4-H',variant=b,panel='N512',metric='NS',**paired(nativeN,dataN['history'],True)))
        for l in (5,6):
            root=WRITERS/f'W{e:03d}/components/L{l}'
            d={v:reduce_raw(root/'KR'/v/'observer/current.json') for v in ('aa','ab','ba','bb')}
            seals=[read(root/'KR'/v/'endpoint-seal.json')['operation'] for v in d]
            for key in ('receiving_state_id','P_sha256','M_sha256','ridge','residual_divisor'):
                assert len({str(s[key]) for s in seals})==1
            assert seals[0]['ridge']==10 and seals[0]['residual_divisor']==8-l
            for metric in ('RS','PS','NS'):
                for v in ('ab','ba','bb'):pair_rows.append(dict(entry=e,layer=l,family='KR',variant=v,panel='Current',metric=metric,**paired(d['aa'][metric],d[v][metric])))
                a={v:aggregate(d[v][metric]) for v in d}
                kr.append(dict(entry=e,upstream=l,receiving=l+1,metric=metric,denominator=a['aa']['denominator'],
                    aa=a['aa']['numerator'],ab=a['ab']['numerator'],ba=a['ba']['numerator'],bb=a['bb']['numerator'],
                    interaction_count=a['bb']['numerator']-a['ba']['numerator']-a['ab']['numerator']+a['aa']['numerator'],
                    interaction_desired_nll=a['bb']['desired_nll_mean']-a['ba']['desired_nll_mean']-a['ab']['desired_nll_mean']+a['aa']['desired_nll_mean'],
                    receiving_state=seals[0]['receiving_state_id'],same_MP=True))
            baseline=reduce_raw(root/'components/no/observer/current.json')
            for family,variants in [('components',('mean','centered','full')),('matrix',('parallel','perpendicular','norm_control'))]:
                for v in variants:
                    d=reduce_raw(root/family/v/'observer/current.json')
                    for metric in d:pair_rows.append(dict(entry=e,layer=l,family=family,variant=v,panel='Current',metric=metric,**paired(baseline[metric],d[metric])))
    table('KR-interactions.csv',kr);table('component-and-N512-paired.csv',pair_rows);table('H512-active-superseded.csv',masked);table('N512-stage-transitions.csv',ntrans)
    write_json(PACKAGE/'coverage-receipt.json',dict(by_phase={p:dict(Counter(r['status'] for r in coverage if r['phase']==p)) for p in set(r['phase'] for r in coverage)},
        raw_metric_pair_reductions=len(pair_rows),KR_matrices=len(kr)//3,KR_receiving_MP_identity=True))
    print('ADVANCED',Counter(r['status'] for r in coverage),flush=True)

def tensor_sample():
    import torch
    torch.set_num_threads(4);rows=[]
    # Chosen by declared endpoints, not observed effect: W0 and W100, early,
    # native, L5/L6. CPU exact stored inputs; no checkpoint/model loading.
    for e in (0,100):
        for layer in (5,6):
            d=ROOT/f'execution/attempt-r3/geometry/E1/W{e:03d}/native/early'
            p=d/f'L{layer}-keys.pt';obj=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
            print('TENSOR_SCHEMA',e,layer,{k:list(v.shape) if torch.is_tensor(v) else type(v).__name__ for k,v in obj.items()},flush=True)
            for k,v in obj.items():
                if torch.is_tensor(v):assert torch.isfinite(v).all()
            # Schema captures actual writer mean directly; do not replace it by
            # a differently rounded reconstruction from individual contexts.
            candidates=[(k,v) for k,v in obj.items() if torch.is_tensor(v) and tuple(v.shape)==(1000,14336)]
            for key,v in candidates:
                x=v.numpy().astype(np.float64);g=x@x.T;trace=np.trace(g);raw=trace**2/np.sum(g*g)
                xc=x-x.mean(0);gc=xc@xc.T;center=np.trace(gc)**2/np.sum(gc*gc)
                stored=read(d/f'L{layer}-raw-geometry.json')['writer_mean']
                rows.append(dict(state=e,layer=layer,tensor_key=key,shape='1000x14336',finite=True,
                    raw_PR=raw,stored_raw_PR=stored['raw']['participation_ratio'],centered_PR=center,
                    stored_centered_PR=stored['centered']['participation_ratio'],
                    raw_close=bool(np.isclose(raw,stored['raw']['participation_ratio'],rtol=1e-9)),
                    centered_close=bool(np.isclose(center,stored['centered']['participation_ratio'],rtol=1e-9))))
                assert rows[-1]['raw_close'] and rows[-1]['centered_close']
    table('geometry-CPU-tensor-sample.csv',rows)
    print('TENSOR_SAMPLE',len(rows),flush=True)

if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
