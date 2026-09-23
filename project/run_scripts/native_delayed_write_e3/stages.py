"""One persistent GPU lane, internal fail-closed atomic DAG; no dynamic submit."""
import argparse
import collections
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
import numpy as np
from .common import ROOT, BASE, PANELS, INSTRUCTION, BATCHES, digest, read, save, sha
from .backend import Backend, groups, tensor_sha, norms, rotation

ORDER = ['G00','G10','G20','G21','G30','G31','G40','G50','G51','G60','G70']


class Run:
    def __init__(self, lock, output):
        self.lock_path=Path(lock);self.lock=read(lock);self.output=Path(output)
        self.binding={k:self.lock[k] for k in ('instruction_id','attempt','source_sha256','panel_sha256','data_sha256')}
        self.output.mkdir(parents=True,exist_ok=False)
        self.rows=read(PANELS/'rows.json');self.groups=groups(self.rows)
        self.general_teachers={}
        self.backend=None;self.stage='INIT';self.start=time.monotonic()
        self.plan=read(ROOT/'inputs/design/review/e3-dependency-plan.json')

    def gate(self, stage, detail):
        i=ORDER.index(stage)
        prior=None
        if i:
            p=self.output/ORDER[i-1]/'gate-result.json'
            x=read(p)
            assert x['status']=='PASS' and x['binding']==self.binding, ('DEPENDENCY_IDENTITY',stage)
            prior=sha(p)
        return save(self.output/stage/'gate-result.json',dict(status='PASS',stage=stage,binding=self.binding,
            predecessor_sha256=prior,detail=detail,scientific_positive_effect_required=False,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),elapsed_seconds=time.monotonic()-self.start))

    def require(self, stage):
        self.stage=stage
        i=ORDER.index(stage)
        if i:
            x=read(self.output/ORDER[i-1]/'gate-result.json')
            assert x['status']=='PASS' and x['binding']==self.binding

    def save_rows(self, stage, key, chunk, rows):
        assert len({r['row_id'] for r in rows})==len(rows)
        return save(self.output/stage/key/f'{chunk:04d}.json',rows)

    def teacher(self, chunk):
        return self.general_teachers[chunk]

    def save_teacher(self, chunk, rows):
        self.general_teachers[chunk]=rows
        save(self.output/'G20/general-teacher-receipts'/f'{chunk:04d}.json',dict(tensor_sha256=[tensor_sha(x) for x in rows],bytes=sum(x.numel()*x.element_size() for x in rows),
            row_ids=[r['row_id'] for r in self.groups[chunk]],source_endpoint='W0',dtype='float32',purpose='bounded GeneralEval full-vocabulary KL teacher; not edited checkpoint'))

    def g00(self):
        self.stage='G00'
        assert self.lock['instruction_id']==INSTRUCTION
        assert self.lock['native_fitting']==self.lock['history_append']==0
        assert self.lock['save_checkpoints'] is False and self.lock['task_gpu_cap']==1
        for m in self.lock['members']:
            p=Path(m['path']);assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256'], ('INPUT_SOURCE_DRIFT',str(p))
        for family, cps in self.lock['checkpoints'].items():
            for b,m in cps.items():
                p=Path(m['path']);st=p.stat()
                assert (st.st_size,st.st_mtime_ns,st.st_ino)==(m['bytes'],m['mtime_ns'],m['inode']), ('CHECKPOINT_BINDING_DRIFT',str(p))
        assert len(self.rows)==self.lock['panel_rows'] and len({r['row_id'] for r in self.rows})==len(self.rows)
        assert len(self.lock['checkpoints'])==2 and all(len(v)==12 for v in self.lock['checkpoints'].values())
        self.gate('G00',dict(cpu_preflight=self.lock['preflight_receipt'],asset_count=24,logical_endpoints=25,panel_rows=len(self.rows)))

    def g10(self):
        self.require('G10');self.backend=Backend(self.lock);b=self.backend;torch=b.torch
        fidelity=read(PANELS/'fidelity-rows.json')
        results=[];repeat_max=0.;old_max=0.;key_repeat=0.
        for family,cell in [('BASE_ALPHAEDIT',1),('BASE_MEMIT',2)]:
            b.endpoint(f'{family}_W001')
            old=read(BASE/f'output/main-cell-{cell}/B001/current.json')['metrics']['RS']['rows']
            byid={x['case_id']:x for x in old}
            for group in groups(fidelity):
                a,k,v,mask,_=b.forward(group);a2,k2,v2,_,_=b.forward(group)
                r=max(abs(x['nll']-y['nll']) for x,y in zip(a,a2,strict=True));repeat_max=max(repeat_max,r)
                kr=max(float((k[l]-k2[l]).abs().max()) for l in k);key_repeat=max(key_repeat,kr)
                for x,source in zip(a,group,strict=True):
                    prior=byid[source['case_id']][source['label']+'_nll']
                    d=abs(x['nll']-prior);old_max=max(old_max,d)
                    # Predeclared scale-aware FP32 envelope, not chosen from task outcome.
                    threshold=self.lock['numerics']['nll_atol']+self.lock['numerics']['nll_rtol']*abs(prior)+10*r
                    assert d<=threshold, ('ORIGINAL_ROW_MISMATCH',family,source['row_id'],d,threshold)
                    assert x['strict']==byid[source['case_id']][source['label']+'_strict'],('ORIGINAL_STRICT_MISMATCH',source['row_id'])
                zero=torch.zeros_like(v[8]);az,kz,vz,_,_=b.forward(group,patch=zero)
                assert all(x['nll']==y['nll'] and x['token_predictions']==y['token_predictions'] for x,y in zip(a,az,strict=True)), 'HOOK_ZERO_PARITY'
                assert all(torch.equal(k[l],kz[l]) for l in k), 'HOOK_KEY_PARITY'
                results.append(dict(family=family,label=group[0]['label'],rows=len(group),repeat_nll_max=r,original_max_so_far=old_max,key_repeat_max=kr,hook_zero_exact=True))
        # L4 invariance against W0 and every physical module mapping on actual panel.
        panel=self.groups[0][:2]
        b.endpoint('W0');a,k0,_,_,_=b.forward(panel)
        b.endpoint('BASE_ALPHAEDIT_W100');_,kt,_,_,_=b.forward(panel)
        assert torch.equal(k0[4],kt[4]),'L4_INPUT_CHANGED'
        b.restore()
        self.gate('G10',dict(results=results,repeat_nll_max=repeat_max,key_repeat_max=key_repeat,
            L4_invariant=True,W0_selected_bytes_restored=True,nonselected_version_pointer_unchanged=True,
            RNG_unchanged=True,M_context_history='input immutable; not installed because no writer; disk stat invariant',
            GPU_continuation='NOT_TESTED_NOT_IN_SCOPE',cost=b.cost()))

    def drift(self, rows, k0, v0, kt, vt, mask, state, k1=None, ws1=None):
        b=self.backend;torch=b.torch;stats=[]
        for l in range(4,9):
            d=kt[l]-k0[l]
            w0=b.w0[l].to('cuda');wt=b.weights[l]
            dk=torch.nn.functional.linear(k0[l],wt-w0)
            wt_dk=torch.nn.functional.linear(d,wt)
            observed=vt[l]-v0[l]
            residual=observed-dk-wt_dk
            baseaction=torch.nn.functional.linear(d,w0)
            history=None;future=None;split_residual=None
            if k1 is not None:
                ws=ws1[l].to('cuda');d1=kt[l]-k1[l]
                history=torch.nn.functional.linear(d1,ws-w0)
                future=torch.nn.functional.linear(kt[l],wt-ws)
                oldout=torch.nn.functional.linear(k1[l],ws)
                split_residual=vt[l]-oldout-torch.nn.functional.linear(d1,w0)-history-future
            for i,row in enumerate(rows):
                m=mask[i];a=k0[l][i,m].double();c=kt[l][i,m].double()
                err=float(residual[i,m].double().norm());den=max(float(observed[i,m].double().norm()),float(v0[l][i,m].double().norm()),1e-12)
                stats.append(dict(row_id=row['row_id'],state=state,layer=l,valid_tokens=int(m.sum()),
                    key0_norm=float(a.norm()),keyt_norm=float(c.norm()),key_delta_norm=float((c-a).norm()),
                    cosine=float((a*c).sum()/max(float(a.norm()*c.norm()),1e-30)),
                    observed_norm=float(observed[i,m].double().norm()),direct_write_norm=float(dk[i,m].double().norm()),
                    key_drift_action_norm=float(wt_dk[i,m].double().norm()),base_action_norm=float(baseaction[i,m].double().norm()),
                    direct_drift_signed_dot=float((dk[i,m].double()*wt_dk[i,m].double()).sum()),
                    mapping_residual=err,mapping_relative=err/den,
                    H1_deltaK_norm=None if history is None else float(history[i,m].double().norm()),
                    F1t_Kt_norm=None if future is None else float(future[i,m].double().norm()),
                    time_split_residual=None if split_residual is None else float(split_residual[i,m].double().norm()),
                    key_mapping='physical down_proj input, all valid TF tokens',not_NLL_additive=True))
                assert err/den<=self.lock['numerics']['module_relative_ceiling'],'MAPPING_IDENTITY'
            if l==4:assert torch.equal(kt[l],k0[l]),'E1_L4_INVARIANCE'
        return stats

    def e1(self):
        self.require('G20');b=self.backend;torch=b.torch
        states=['W0']+[f'{f}_W{x:03d}' for f in ('BASE_ALPHAEDIT','BASE_MEMIT') for x in BATCHES]
        for ci,group in enumerate(self.groups):
            b.endpoint('W0');r0,k0,v0,mask,teacher=b.forward(group,return_general=group[0]['kind']=='GENERAL')
            if teacher:
                self.save_teacher(ci,teacher)
                for r in r0:r.update(w0_forward_kl=0.,w0_top1_agree=r['target_count'])
            self.save_rows('G20','W0',ci,r0)
            k1={}
            for state in states[1:]:
                b.endpoint(state);r,kt,vt,mask,_=b.forward(group,general_teacher=teacher or None)
                family=state.rsplit('_W',1)[0];batch=int(state.rsplit('_W',1)[1])
                if batch==1:k1[family]={l:k.clone() for l,k in kt.items()}
                stats=self.drift(group,k0,v0,kt,vt,mask,state,k1[family],b.states[f'{family}_W001'])
                self.save_rows('G20',state,ci,r)
                save(self.output/'G20/layer-key-drift'/state/f'{ci:04d}.json',stats)
            print('E1_CHUNK_COMPLETE',ci,len(self.groups),flush=True)
        b.restore();self.gate('G20',dict(logical_endpoints=25,rows_per_endpoint=len(self.rows),cost=b.cost(),keys='bounded GPU chunk; recompute for E3; no unbounded raw-key store'))

    def e1_validate(self):
        self.require('G21');expected=[r['row_id'] for r in self.rows]
        names=['W0']+[f'{f}_W{x:03d}' for f in ('BASE_ALPHAEDIT','BASE_MEMIT') for x in BATCHES]
        for name in names:
            got=[r['row_id'] for p in sorted((self.output/'G20'/name).glob('*.json')) for r in read(p)]
            assert got==expected and len(set(got))==len(expected), ('E1_COMPLETENESS',name)
        self.gate('G21',dict(endpoints=25,rows_per_endpoint=len(expected),identity_order_exact=True))

    def pair_name(self,p):return f"{p['family']}_s{p['s']:03d}_t{p['t']:03d}"

    def factorial(self,phase,stage):
        self.require(stage);b=self.backend;torch=b.torch
        for pair in [p for p in self.plan['pairs'] if p['phase']==phase]:
            key=self.pair_name(pair);f,s,t=pair['family'],pair['s'],pair['t']
            A=(b.states[f'{f}_W{s:03d}'][8].double()-b.w0[8].double()).float().to('cuda')
            for ci,group in enumerate(self.groups):
                teacher=self.teacher(ci) if group[0]['kind']=='GENERAL' else None
                captures={};values={}
                for state in ('00','10','01','11'):
                    b.hybrid(f,s,t,state);rs,ks,vs,mask,_=b.forward(group,general_teacher=teacher)
                    captures[state]=ks[8];values[state]=vs[8]
                    if state=='11':
                        prior=read(self.output/'G20'/f'{f}_W{t:03d}'/f'{ci:04d}.json')
                        assert [(r['row_id'],r['nll'],r['token_predictions']) for r in rs]==[(r['row_id'],r['nll'],r['token_predictions']) for r in prior], 'ACTUAL11_E1_PARITY'
                        for r in rs:r['reused_metric_from']=str(self.output/'G20'/f'{f}_W{t:03d}'/f'{ci:04d}.json')
                    self.save_rows(stage,f'{key}/{state}',ci,rs)
                assert torch.equal(captures['11'],captures['01']) and torch.equal(captures['10'],captures['00']), 'E3_KEY_INVARIANCE'
                delta=captures['01']-captures['00'];action=torch.nn.functional.linear(delta,A)
                cross=values['11']-values['10']-values['01']+values['00'];err=cross-action
                checks=[]
                for i,r in enumerate(group):
                    m=mask[i];den=max(float(values['11'][i,m].double().norm()),1e-12)
                    rel=float(err[i,m].double().norm())/den
                    checks.append(dict(row_id=r['row_id'],pair=key,valid_tokens=int(m.sum()),K11_K01_exact=True,K10_K00_exact=True,
                        action_norm=float(action[i,m].double().norm()),cross_norm=float(cross[i,m].double().norm()),
                        residual_norm=float(err[i,m].double().norm()),relative_to_v11=rel,
                        A_definition=pair['write_kind'],FP32_hybrid_subtraction_recorded=True,module_not_NLL_additive=True))
                save(self.output/stage/key/'module-interaction-checks'/f'{ci:04d}.json',checks)
            print('FACTORIAL_PAIR_COMPLETE',stage,key,flush=True)
        b.restore();self.gate(stage,dict(phase=phase,pairs=sum(p['phase']==phase for p in self.plan['pairs']),cost=b.cost()))

    def interaction_validate(self,source,stage,phase):
        self.require(stage);maxrel=0.;n=0
        for p in [p for p in self.plan['pairs'] if p['phase']==phase]:
            rows=[r for f in sorted((self.output/source/self.pair_name(p)/'module-interaction-checks').glob('*.json')) for r in read(f)]
            assert [r['row_id'] for r in rows]==[r['row_id'] for r in self.rows]
            for r in rows:
                n+=1;maxrel=max(maxrel,r['relative_to_v11'])
                assert r['K11_K01_exact'] and r['K10_K00_exact']
                assert r['relative_to_v11']<=self.lock['numerics']['module_relative_ceiling'], ('MODULE_CROSS_IDENTITY',r)
        self.gate(stage,dict(rows=n,max_module_residual_relative=maxrel,predeclared_ceiling=self.lock['numerics']['module_relative_ceiling']))

    def patch(self,phase,stage):
        self.require(stage);b=self.backend;torch=b.torch
        for p in [p for p in self.plan['pairs'] if p['phase']==phase]:
            key=self.pair_name(p);f,s,t=p['family'],p['s'],p['t']
            A=(b.states[f'{f}_W{s:03d}'][8].double()-b.w0[8].double()).float().to('cuda')
            for ci,group in enumerate(self.groups):
                teacher=self.teacher(ci) if group[0]['kind']=='GENERAL' else None
                b.hybrid(f,s,t,'00');_,ks,_,mask,_=b.forward(group)
                k0=ks[8];del ks
                b.hybrid(f,s,t,'11');baseline,kt,_,mask,_=b.forward(group,general_teacher=teacher)
                prior=read(self.output/'G20'/f'{f}_W{t:03d}'/f'{ci:04d}.json')
                assert [(r['row_id'],r['nll'],r['token_predictions']) for r in baseline]==[(r['row_id'],r['nll'],r['token_predictions']) for r in prior], 'PATCH_RECEIVER_PARITY'
                action=torch.nn.functional.linear(kt[8]-k0,A)*mask.unsqueeze(-1)
                variants=[('dose_0p5',-.5*action),('dose_1',-action),('dose_minus1',action)]
                variants += [(f'rotation_{seed}',-rotation(action,seed)) for seed in self.lock['rotation_seeds']]
                audits=[]
                for name,shift in variants:
                    if name.startswith('rotation'):
                        a=action.double().square().sum(-1);c=shift.double().square().sum(-1)
                        assert torch.allclose(a,c,rtol=1e-12,atol=1e-12), 'RMS_CONTROL_MISMATCH'
                    rs,_,_,_,_=b.forward(group,capture=False,patch=shift,general_teacher=teacher)
                    self.save_rows(stage,f'{key}/{name}',ci,rs)
                    audits.append(dict(variant=name,shift=norms(shift,mask),rows=len(rs),all_valid_input_positions=True,pad_patch_zero=True))
                save(self.output/stage/key/'patch-coverage'/f'{ci:04d}.json',dict(rows=[r['row_id'] for r in group],variants=audits,
                    lambda0_reused_from=f'G20/{f}_W{t:03d}/{ci:04d}.json',receiving_state=f'{f}_W{t:03d}'))
            print('PATCH_PAIR_COMPLETE',stage,key,flush=True)
        b.restore();self.gate(stage,dict(phase=phase,new_variants_per_pair=6,lambda0='REUSED_EXACT_E1_ENDPOINT',cost=b.cost()))

    def run(self):
        try:
            self.g00();self.g10();self.e1();self.e1_validate()
            self.factorial('core','G30');self.interaction_validate('G30','G31','core');self.patch('core','G40')
            self.factorial('extension','G50');self.interaction_validate('G50','G51','extension');self.patch('extension','G60')
            save(self.output/'science-terminal.json',dict(status='SCIENCE_COMPLETED_G70_PENDING_CPU_REDUCER',binding=self.binding,
                 cost=self.backend.cost(),host_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 new_checkpoint=False,write=0,history_append=0,fit=0,all_twelve_fixed_pairs=True))
        except BaseException as exc:
            restore=None
            if self.backend is not None:
                try:self.backend.restore();restore='SELECTED_W0_BYTES/RNG/NONSELECTED/INPUT_STAT_VERIFIED'
                except BaseException as e:restore=repr(e)
            save(self.output/'failure.json',dict(status='TECHNICAL_FAILED',stage=self.stage,error=repr(exc),traceback=traceback.format_exc(),
                binding=self.binding,restore=restore,cost=None if self.backend is None else self.backend.cost(),elapsed_seconds=time.monotonic()-self.start))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();Run(args.lock,args.output).run()
