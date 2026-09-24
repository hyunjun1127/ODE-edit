"""All completed E1/E3 rows, paired effects, module receipts, cost and inventory."""
import collections
import csv
import math
import numpy as np
from .common import *
from .reducer import load_scores,summarize,compare,ci,write_csv

def general_compare(label,x,y):
    assert [r['row_id'] for r in x]==[r['row_id'] for r in y]
    d=[b['nll']-a['nll'] for a,b in zip(x,y,strict=True)]
    row=dict(contrast=label,panel='GeneralEval128',kind='GENERAL',count=len(x),natural_nll_delta=float(np.mean(d)),
        KL_delta=float(np.mean([b['w0_forward_kl']-a['w0_forward_kl'] for a,b in zip(x,y)])),
        W0_top1_delta=sum(b['w0_top1_agree']-a['w0_top1_agree'] for a,b in zip(x,y)),
        TF_token_correct_delta=sum(b['token_correct']-a['token_correct'] for a,b in zip(x,y)),
        strict_lost=sum(a['strict'] and not b['strict'] for a,b in zip(x,y)),strict_gained=sum(not a['strict'] and b['strict'] for a,b in zip(x,y)))
    for q in (.01,.05,.5,.95,.99):row[f'natural_nll_delta_q{int(q*100):02d}']=float(np.quantile(d,q))
    row.update(ci(d,x));return row

def all_json(path):return [r for p in sorted(Path(path).glob('*.json')) for r in read(p)]

def active_counts(name,pairs,ledger):
    batch=0 if name=='W0' else int(name.rsplit('_W',1)[1]);selected=[r for r in ledger if r['batch']==batch]
    orig={(r['case_id'],r['kind'],r['prompt_index']):r for r in pairs if r['panel']=='H_diag_B1_R100_P200'}
    variants={(r['case_id'],r['kind'],r['prompt_index'],r['active_target_hash']):r for r in pairs if r['panel']=='H_active_variants'}
    out=[]
    for status in ('ACTIVE','SUPERSEDED','NOT_RECEIVED'):
        for kind in ('R','P'):
            original=[];current=[]
            for item in selected:
                if item['status']!=status:continue
                for pi in ([0] if kind=='R' else [0,1]):
                    r=orig[item['case_id'],kind,pi];original.append(r)
                    if item['active_target'] is not None:
                        current.append(r if item['active_target']==item['original_target'] else variants[item['case_id'],kind,pi,digest(item['active_target'])])
            out.append(dict(endpoint=name,status=status,kind=kind,original_count=len(original),original_success=sum(r['success'] for r in original),
                current_count=len(current),current_success=sum(r['success'] for r in current),current_strict=sum(r['strict'] for r in current)))
    return out

def main():
    assert not (LOCAL/'detail-terminal.json').exists()
    panel=read(ROOT/'inputs/panels-v2/rows.json');pidx={r['row_id']:r for r in panel};lock=read(ATTEMPT/'execution.lock.json')
    plan=read(ROOT/'inputs/design/review/e3-dependency-plan.json');ledger=read(ROOT/'inputs/panels-v2/active-overwrite-ledger.json')
    endpoints={};valid=[];paired=[];ids=[];active=[];module=[];factorial=[];summaries=[];coverage=[];patchchecks=[];drift=[]
    for p in sorted((OUTPUT/'G20').iterdir()):
        if p.name!='W0' and not p.name.startswith('BASE_'):continue
        rr,g,v=load_scores(p,panel);endpoints[p.name]=(rr,g);valid.append(v);active+=active_counts(p.name,rr,ledger)
    for name,(rr,g) in endpoints.items():
        if name!='W0':
            paired+=compare(name+' minus W0',endpoints['W0'][0],rr,ids)
            paired.append(general_compare(name+' minus W0',endpoints['W0'][1],g))
            raw=all_json(OUTPUT/'G20/layer-key-drift'/name)
            assert len(raw)==5*len(panel)
            for layer in range(4,9):
                sr=[r for r in raw if r['layer']==layer]
                assert [r['row_id'] for r in sr]==[p['row_id'] for p in panel]
                assert all(r['valid_tokens']==len(pidx[r['row_id']]['input_ids']) for r in sr)
                assert all(r['mapping_relative']<=lock['numerics']['module_relative_ceiling'] for r in sr)
                if layer==4:assert all(r['key_delta_norm']==0 for r in sr)
                for pn in sorted({p['panel'] for p in panel}):
                    x=[r for r in sr if pidx[r['row_id']]['panel']==pn]
                    row=dict(endpoint=name,layer=layer,panel=pn,completion_rows=len(x),valid_positions=sum(r['valid_tokens'] for r in x),
                        mapping_relative_max=max(r['mapping_relative'] for r in x),time_split_residual_max=max(r['time_split_residual'] for r in x),
                        signed_cross_negative=sum(r['direct_drift_signed_dot']<0 for r in x))
                    for k in ('key_delta_norm','cosine','direct_write_norm','key_drift_action_norm','base_action_norm','H1_deltaK_norm','F1t_Kt_norm','direct_drift_signed_dot'):
                        row[k+'_mean']=float(np.mean([r[k] for r in x]))
                    drift.append(row)
    for pair in plan['pairs']:
        key=f"{pair['family']}_s{pair['s']:03d}_t{pair['t']:03d}";fs='G30' if pair['phase']=='core' else 'G50';ps='G40' if pair['phase']=='core' else 'G60'
        states={};general={}
        for s in ('00','10','01','11'):
            rr,g,v=load_scores(OUTPUT/fs/key/s,panel);valid.append(v);states[s]=rr;general[s]=g;summaries+=summarize(key+'/'+s,rr,g)
        expected=endpoints[f"{pair['family']}_W{pair['t']:03d}"]
        # Compare numerical payload, not provenance-only reused_metric_from annotation.
        assert states['11']==expected[0]
        assert [{k:v for k,v in r.items() if k!='reused_metric_from'} for r in general['11']]==expected[1]
        maps={s:{r['pair_id']:r for r in rr} for s,rr in states.items()}
        for pn,k in sorted({(r['panel'],r['kind']) for r in states['00']}):
            xx=[r for r in states['00'] if (r['panel'],r['kind'])==(pn,k)];row=dict(pair=key,phase=pair['phase'],write_kind=pair['write_kind'],panel=pn,kind=k,count=len(xx))
            for metric in ('true_nll','new_nll','desired_nll','g'):
                d=[maps['11'][r['pair_id']][metric]-maps['10'][r['pair_id']][metric]-maps['01'][r['pair_id']][metric]+maps['00'][r['pair_id']][metric] for r in xx]
                row[metric+'_interaction']=float(np.mean(d))
                for q in (.01,.05,.5,.95,.99):row[f'{metric}_q{int(q*100):02d}']=float(np.quantile(d,q))
                if metric=='desired_nll':row.update(ci(d,xx))
            factorial.append(row)
        dg=[d['nll']-b['nll']-c['nll']+a['nll'] for a,b,c,d in zip(general['00'],general['10'],general['01'],general['11'],strict=True)]
        factorial.append(dict(pair=key,phase=pair['phase'],panel='GeneralEval128',kind='GENERAL',count=128,natural_nll_interaction=float(np.mean(dg)),**ci(dg,general['00'])))
        mr=all_json(OUTPUT/fs/key/'module-interaction-checks');assert [r['row_id'] for r in mr]==[p['row_id'] for p in panel]
        assert all(r['K11_K01_exact'] and r['K10_K00_exact'] and r['valid_tokens']==len(pidx[r['row_id']]['input_ids']) for r in mr)
        assert all(r['relative_to_v11']<=lock['numerics']['module_relative_ceiling'] for r in mr)
        for pn in sorted({p['panel'] for p in panel}):
            x=[r for r in mr if pidx[r['row_id']]['panel']==pn]
            module.append(dict(pair=key,panel=pn,rows=len(x),relative_max=max(r['relative_to_v11'] for r in x),residual_max=max(r['residual_norm'] for r in x),
                action_mean=float(np.mean([r['action_norm'] for r in x])),cross_mean=float(np.mean([r['cross_norm'] for r in x])),
                zero_action_rows=sum(r['action_norm']==0 for r in x),source_runtime_key_equal=True,tensor_reconstruction='NOT_AVAILABLE'))
        variants=['dose_0p5','dose_1','dose_minus1']+[f'rotation_{s}' for s in lock['rotation_seeds']]
        for variant in variants:
            rr,g,v=load_scores(OUTPUT/ps/key/variant,panel);valid.append(v);summaries+=summarize(key+'/'+variant,rr,g)
            paired+=compare(key+'/'+variant+' minus actual11',states['11'],rr,ids)
            paired.append(general_compare(key+'/'+variant+' minus actual11',general['11'],g))
            w0={r['pair_id']:r for r in endpoints['W0'][0]};lost={r['pair_id'] for r in states['11'] if w0[r['pair_id']]['success'] and not r['success']}
            a=[r for r in states['11'] if r['pair_id'] in lost];b=[r for r in rr if r['pair_id'] in lost]
            if a:paired+=compare(key+'/'+variant+' LOST_ONLY_auxiliary',a,b,ids)
        seen=[];maxratio=0.;zerochunks=0
        for fp in sorted((OUTPUT/ps/key/'patch-coverage').glob('*.json')):
            c=read(fp);seen+=c['rows'];vv={r['variant']:r for r in c['variants']};assert set(vv)==set(variants)
            validtokens=sum(len(pidx[i]['input_ids']) for i in c['rows']);base=vv['dose_1']['shift'];zerochunks+=base['norm']==0
            for v,r in vv.items():
                assert r['all_valid_input_positions'] and r['pad_patch_zero'] and r['rows']==len(c['rows']) and r['shift']['elements']==4096*validtokens
                expected_norm=base['norm']*(.5 if v=='dose_0p5' else 1)
                error=abs(r['shift']['norm']-expected_norm)/max(expected_norm,1e-30);maxratio=max(maxratio,error)
                assert error<=1e-12
            assert c['receiving_state']==f"{pair['family']}_W{pair['t']:03d}" and (OUTPUT/c['lambda0_reused_from']).is_file()
        assert seen==[p['row_id'] for p in panel]
        patchchecks.append(dict(pair=key,chunks=len(list((OUTPUT/ps/key/'patch-coverage').glob('*.json'))),completion_rows=len(seen),norm_matching_max_relative=maxratio,zero_action_chunks=zerochunks,
            allvalid_and_pad_ledger=True,per_token_RMS='runtime assertion; scalar CPU norm verification, not independent tensor replay'))
        coverage.append(dict(**pair,pair=key,factorial_states=4,patch_variants=6,lambda0_reused=True,status='COMPLETED'))
        print('REVIEW_PAIR',key,flush=True)
    assert len(valid)==145
    for name,rr in [('paired-summary.csv',paired),('active-overwrite.csv',active),('module-summary.csv',module),('factorial-summary.csv',factorial),
        ('factorial-patch-metrics.csv',summaries),('coverage.csv',coverage),('patch-audit.csv',patchchecks),('layer-drift.csv',drift),('row-validation.csv',valid)]:write_csv(REPORT/name,rr)
    save(LOCAL/'paired-transition-ids.json',ids)
    costs=[];previous={};last='start'
    for s in plan['stages']:
        name=s['stage_id'];g=read(OUTPUT/name/'gate-result.json');c=g.get('detail',{}).get('cost')
        if c:
            d={k:c[k]-previous.get(k,0) for k in ('forward_calls','forward_seconds','program_seconds','selected_weight_H2D_bytes')};assert all(v>=0 for v in d.values())
            costs.append(dict(stage=name,since_previous=last,**d,peak_gpu_bytes_cumulative=c['peak_gpu_bytes'],forward_nested_in_program=True,
                nonforward_NOT_SEPARATED=d['program_seconds']-d['forward_seconds']));previous=c;last=name
    write_csv(REPORT/'compute.csv',costs)
    # Full new output inventory, old model/checkpoint SHA not recomputed.
    inventory=[dict(path=str(p.relative_to(OUTPUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUTPUT.rglob('*')) if p.is_file()]
    assert not any(x['path'].endswith(('.partial','.pt','.npy')) for x in inventory)
    prior=read(ATTEMPT/'report/artifact-index.json')['members'];assert inventory==prior
    save(LOCAL/'artifact-index.json',dict(root=str(OUTPUT),members=inventory))
    save(REPORT/'artifact-index-root.json',dict(file=record(LOCAL/'artifact-index.json'),members=len(inventory),bytes=sum(x['bytes'] for x in inventory),new_fullSHA_matches_original_index=True,
        paired_ID_artifact=record(LOCAL/'paired-transition-ids.json'),input_CP_fullSHA_reused_current_stat=True))
    result=dict(status='CPU_REVIEW_PASS_STORED_SCOPE',score_datasets=len(valid),completion_rows=sum(v['completion_rows'] for v in valid),
        endpoint25=True,factorial48=True,patch72=True,actual11_exact=True,panel_cases_verified=True,original_runtime_modified=False,
        independent_agent=False,GPU_calls=0,scheduler_extra_queries=0,limitations=['No independent full tensor/K/v replay','KL scalars source/runtime bound; full teacher RAM-only',
        'state SHA bound through CP/config/path/runtime not per-row tensor hash','E1 temporal split stored for s1 only; s>1 factorial separately available'])
    save(AUDIT/'detail-checks.json',result);save(LOCAL/'detail-terminal.json',result)

if __name__=='__main__':main()
