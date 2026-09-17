"""Supplemental CPU source/terminal-residual/strict-transition audit.

No import of an execution module, model class, scheduler or evaluator.
"""
import ast
import csv
import json
from pathlib import Path
from collections import Counter
import torch
from .bootstrap import ROOT, REVIEW, identity, save
from .metrics import ARMS, MULT, read, digest, table, select

def strict_pair(a,b,tag):
    key=lambda r:(r['case_id'],r['prompt_index'],r['identity'])
    aa={key(r):r for r in a};bb={key(r):r for r in b};assert len(aa)==len(a) and aa.keys()==bb.keys()
    side='true' if tag=='NS' else 'new'
    x=[bool(aa[k][side+'_strict']) for k in aa];y=[bool(bb[k][side+'_strict']) for k in aa]
    out=dict(metric=tag,denominator=len(x),before=sum(x),after=sum(y),lost=sum(v and not w for v,w in zip(x,y)),gained=sum(not v and w for v,w in zip(x,y)))
    if tag=='PS':
        groups={}
        for k in aa:groups.setdefault(k[0],[]).append(k)
        xx=[all(aa[k]['new_strict'] for k in keys) for keys in groups.values()]
        yy=[all(bb[k]['new_strict'] for k in keys) for keys in groups.values()]
        out.update(twoP_denominator=len(xx),twoP_before=sum(xx),twoP_after=sum(yy),twoP_lost=sum(v and not w for v,w in zip(xx,yy)),twoP_gained=sum(not v and w for v,w in zip(xx,yy)))
    return out

def run():
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    out=REVIEW/'supplement-v1';out.mkdir(exist_ok=False)
    lock=read(ROOT/'execution.lock.json');source=Path(lock['source_root']);members=[];functions=[]
    for m in lock['members']:
        p=Path(m['path']);s=p.stat();assert s.st_size==m['bytes']
        if m.get('verification')=='PRIOR_FULL_SHA_STABLE_STAT':
            assert [s.st_dev,s.st_ino,s.st_mtime_ns]==m['stat']
            method='PRIOR_FULL_SHA_CURRENT_STABLE_STAT_NO_REHASH'
        else:
            assert identity(p)['sha256']==m['sha256'];method='CURRENT_FULL_SHA'
        members.append(dict(path=str(p),bytes=s.st_size,sha256=m['sha256'],verification=method))
    for p in sorted((source/'project/run_scripts/local_z_adaptive_allocation').glob('*.py')):
        tree=ast.parse(p.read_text())
        for n in ast.walk(tree):
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
                functions.append(dict(file=str(p.relative_to(source)),name=n.name,line=n.lineno,end_line=n.end_lineno,sha256=identity(p)['sha256']))
    table(out/'source-functions.csv',functions);save(out/'input-source-inventory.json',members)
    tech=ROOT/'technical/attempt-v1';fwd={r['candidate_id']:r for r in (read(p) for p in (tech/'order-forward').glob('candidate-*.json'))}
    rev={r['candidate_id']:r for r in (read(p) for p in (tech/'order-reverse').glob('candidate-*.json'))}
    repeat=[]
    for name,a in fwd.items():
        b=rev[name];row=dict(candidate=name,strict_same=a['S_cur']==b['S_cur'] and a['S_past']==b['S_past'])
        for k,tolerance in [('E',5e-5),('H',5e-5),('D',5e-7)]:
            diff=abs(a[k]-b[k]);assert diff<=tolerance;row[k+'_abs_diff']=diff
        assert row['strict_same'];repeat.append(row)
    table(out/'technical-repeat.csv',repeat)
    terminal=[];counts=[]
    for arm in ('T75','TD'):
        for b in range(1,11):
            folder=ROOT/'arms'/arm/'attempt-v1/output'/f'B{b:03d}/proposals'
            z=torch.load(folder/'terminal-Z8.pt',map_location='cpu',weights_only=True,mmap=True)
            counts.append(dict(arm=arm,batch=b,terminal_target_seconds=z['receipt']['seconds'],terminal_Adam=z['receipt']['adam_updates'],terminal_loss=z['receipt']['loss_evaluations']))
            for p in sorted(folder.glob('terminal[48]*.pt')):
                f=torch.load(p,map_location='cpu',weights_only=True,mmap=True);c=f['captures']
                expected=z['Z8']-c['H8'];assert torch.equal(expected,c['R'])
                assert c['K'].shape==(14336,100) and c['H8'].shape==c['R'].shape==(4096,100)
                terminal.append(dict(arm=arm,batch=b,fit=p.stem,writer_layer=f['receipt']['layer'],readout_layer=f['receipt']['readout_layer'],
                    residual_exact_Z8_minus_current_H8=True,Z8_norm=float(z['Z8'].norm()),H8_norm=float(c['H8'].norm()),residual_norm=float(c['R'].norm()),
                    terminal_key_readout_solve_inclusive_seconds=f['receipt']['seconds'],input_state_hash=digest(f['receipt']['input_state'])))
    table(out/'terminal-residual.csv',terminal);table(out/'terminal-target-cost.csv',counts)
    strict=[];equality=[];fulls={}
    for arm in ARMS:
        root=ROOT/'arms'/arm/'attempt-v1/output';final=read(root/'B010/seen-full.json');fulls[arm]=final
        pooled={k:[] for k in MULT}
        for b in range(1,11):
            d=root/f'B{b:03d}';entry=read(d/'entry-current.json');selected=read(d/'selected-current.json')
            ids=[r['case_id'] for r in selected['metrics']['RS']['rows']];later=select(final,ids)
            for tag in MULT:
                a=selected['metrics'][tag]['rows'];pooled[tag].extend(a)
                strict.append(dict(arm=arm,batch=b,contrast='ENTRY_TO_ATWRITE',**strict_pair(entry['metrics'][tag]['rows'],a,tag)))
                strict.append(dict(arm=arm,batch=b,contrast='ATWRITE_TO_W10',**strict_pair(a,later[tag],tag)))
            if b in (5,10):
                whole=read(d/'seen-full.json');subset=select(whole,ids)
                assert all(subset[t]==selected['metrics'][t]['rows'] for t in MULT)
        full5=read(root/'B005/seen-full.json');ids=[r['case_id'] for r in full5['metrics']['RS']['rows']]
        later=select(final,ids)
        for tag in MULT:
            strict.append(dict(arm=arm,contrast='POOLED_ATWRITE_TO_W10',**strict_pair(pooled[tag],final['metrics'][tag]['rows'],tag)))
            strict.append(dict(arm=arm,contrast='W5_FIRST500_TO_W10_SAME500',**strict_pair(full5['metrics'][tag]['rows'],later[tag],tag)))
        for name,ids in [('first500',ids),('last500',[r['case_id'] for r in final['metrics']['RS']['rows'][500:]])]:
            part=read(root/f'B010/{name}.json');assert all(select(final,ids)[t]==part['metrics'][t]['rows'] for t in MULT)
    for a,b in [(a,'N4') for a in ARMS if a!='N4']+[('LD','L4D'),('LD','L75'),('LD','REFIT4'),('LD','TD'),('T75','L75')]:
        for tag in MULT:strict.append(dict(arm=a,reference=b,contrast='POLICY_W10',**strict_pair(fulls[b]['metrics'][tag]['rows'],fulls[a]['metrics'][tag]['rows'],tag)))
    for b in range(1,11):
        a=read(ROOT/f'arms/N4/attempt-v1/output/B{b:03d}/commit.json');c=read(ROOT/f'arms/TD/attempt-v1/output/B{b:03d}/commit.json')
        equality.append(dict(batch=b,N4_TD_W4_same=a['state']['W']['4']==c['state']['W']['4'],N4_TD_W8_same=a['state']['W']['8']==c['state']['W']['8'],
            M4_same=a['state']['M']['4']==c['state']['M']['4'],M8_same=a['state']['M']['8']==c['state']['M']['8']))
    table(out/'strict-transitions.csv',strict);table(out/'N4-TD-state-comparison.csv',equality)
    save(out/'summary.json',dict(status='CPU_SOURCE_RESIDUAL_STRICT_REUSE_CHECKS_PASS',terminal_saved_residuals=len(terminal),
        source_members=len(members),runtime_source_modified=False,technical_repeat_max={k:max(r[k+'_abs_diff'] for r in repeat) for k in ('E','H','D')},
        N4_TD_weights_M4_identical_all_batches=all(r['N4_TD_W4_same'] and r['N4_TD_W8_same'] and r['M4_same'] for r in equality),
        N4_TD_M8_different_all_batches=all(not r['M8_same'] for r in equality),subset_reuse_exact=True,new_GPU=0))
    print((out/'summary.json').read_text())

if __name__=='__main__':run()
