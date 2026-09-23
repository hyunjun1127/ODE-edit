"""Post-terminal CPU checks and GeneralEval supplement, never imports a model.

Separate analysis source: submitted runtime/collector remain immutable. Original
row files are read-only and are not copied into Git. No live partial reduction.
"""
import argparse
import collections
import csv
import math
from pathlib import Path
import numpy as np
from .common import PANELS, ROOT, read, save, sha, digest
from .reduce import read_rows, write_csv, cluster_ci


def validate_rows(rows, panel):
    assert [r['row_id'] for r in rows]==[r['row_id'] for r in panel], 'ORDER_CARDINALITY'
    max_mean_diff=0.
    for r,p in zip(rows,panel,strict=True):
        assert r['input_sha']==digest([p['input_ids'],p['positions'],p['target_ids']])
        assert all(r[k]==p[k] for k in ('row_id','pair_id','panel','kind','label','case_id','subject','prompt_cluster'))
        assert r['target_count']==len(p['target_ids'])==len(r['token_predictions'])==len(r['token_nll'])
        assert all(math.isfinite(float(x)) for x in [r['nll'],*r['token_nll']])
        correct=[a==b for a,b in zip(r['token_predictions'],p['target_ids'],strict=True)]
        assert sum(correct)==r['token_correct'] and all(correct)==r['strict']
        # CPU FP64 mean and GPU FP32 reduction are distinct routes: expose, do not
        # silently replace original model NLL with a differently rounded value.
        max_mean_diff=max(max_mean_diff,abs(float(np.mean(r['token_nll']))-r['nll']))
    return dict(rows=len(rows),token_positions=sum(r['target_count'] for r in rows),
        token_predictions_recount_exact=True,nll_mean_route_max_abs=max_mean_diff)


def independent_counts(rows):
    grouped=collections.defaultdict(dict)
    for r in rows:
        if r['label']!='natural':grouped[(r['panel'],r['kind'],r['pair_id'])][r['label']]=r
    bykind=collections.defaultdict(list)
    for (panel,kind,_),d in grouped.items():
        assert set(d)=={'true','new'}
        t,n=d['true'],d['new'];preferred=t if kind in ('N','BASE') else n
        success=t['nll']<n['nll'] if kind in ('N','BASE') else n['nll']<t['nll']
        bykind[panel,kind].append(dict(success=success,tie=t['nll']==n['nll'],strict=preferred['strict'],
            token_correct=preferred['token_correct'],token_count=preferred['target_count'],
            prompt_accuracy=preferred['token_correct']/preferred['target_count'],
            desired_nll=preferred['nll'],true_nll=t['nll'],new_nll=n['nll'],margin=n['nll']-t['nll']))
    result=[]
    for (panel,kind),rr in sorted(bykind.items()):
        result.append(dict(panel=panel,kind=kind,count=len(rr),success=sum(r['success'] for r in rr),
            ties=sum(r['tie'] for r in rr),strict=sum(r['strict'] for r in rr),
            token_correct=sum(r['token_correct'] for r in rr),token_total=sum(r['token_count'] for r in rr),
            TF_prompt_macro=float(np.mean([r['prompt_accuracy'] for r in rr])),
            desired_nll=float(np.mean([r['desired_nll'] for r in rr])),
            true_nll=float(np.mean([r['true_nll'] for r in rr])),new_nll=float(np.mean([r['new_nll'] for r in rr])),
            g_p01=float(np.quantile([r['margin'] for r in rr],.01)),g_p50=float(np.median([r['margin'] for r in rr])),
            g_p99=float(np.quantile([r['margin'] for r in rr],.99))))
    return result


def general_metrics(name, rows):
    rr=[r for r in rows if r['kind']=='GENERAL'];assert len(rr)==128
    return dict(endpoint=name,documents=len(rr),positions=sum(r['target_count'] for r in rr),
        nll=float(np.mean([r['nll'] for r in rr])),strict=sum(r['strict'] for r in rr),
        token_correct=sum(r['token_correct'] for r in rr),
        TF_prompt_macro=float(np.mean([r['token_correct']/r['target_count'] for r in rr])),
        forward_KL_docmean=float(np.mean([r['w0_forward_kl'] for r in rr])),
        w0_top1_matches=sum(r['w0_top1_agree'] for r in rr))


def general_contrast(name, before, after):
    x=[r for r in before if r['kind']=='GENERAL'];y=[r for r in after if r['kind']=='GENERAL']
    assert [r['row_id'] for r in x]==[r['row_id'] for r in y] and len(x)==128
    d=[b['nll']-a['nll'] for a,b in zip(x,y,strict=True)]
    row=dict(contrast=name,documents=128,NLL_delta=float(np.mean(d)),NLL_delta_p05=float(np.quantile(d,.05)),
        NLL_delta_p95=float(np.quantile(d,.95)),
        strict_lost=sum(a['strict'] and not b['strict'] for a,b in zip(x,y)),
        strict_gained=sum(not a['strict'] and b['strict'] for a,b in zip(x,y)),
        token_correct_delta=sum(b['token_correct']-a['token_correct'] for a,b in zip(x,y)),
        W0_agreement_delta=sum(b['w0_top1_agree']-a['w0_top1_agree'] for a,b in zip(x,y)),
        KL_delta=float(np.mean([b['w0_forward_kl']-a['w0_forward_kl'] for a,b in zip(x,y)])))
    row.update(cluster_ci(d,x));return row


def run(attempt, destination):
    attempt=Path(attempt);output=attempt/'output';dest=Path(destination)
    assert (output/'G70/gate-result.json').is_file(), 'COLLECTOR_NOT_TERMINAL'
    assert not (output/'failure.json').exists(), 'USE_PARTIAL_FAILURE_REVIEW'
    lock=read(attempt/'execution.lock.json');panel=read(PANELS/'rows.json')
    plan=read(ROOT/'inputs/design/review/e3-dependency-plan.json')
    previous=None;binding=None;stage_rows=[]
    for spec in plan['stages']:
        sid=spec['stage_id'];p=output/sid/'gate-result.json';r=read(p)
        assert r['status']=='PASS' and r['stage']==sid
        binding=binding or r['binding'];assert r['binding']==binding
        assert r['predecessor_sha256']==previous
        previous=sha(p);stage_rows.append(dict(stage=sid,sha256=previous,elapsed_seconds=r.get('elapsed_seconds')))
    assert binding['panel_sha256']==sha(PANELS/'panel-manifest.json')==lock['panel_sha256']
    dest.mkdir(parents=True,exist_ok=False)
    counts=[];general=[];contrasts=[];validations=[];endpoints={};checks=[];drift=[]
    for p in sorted((output/'G20').iterdir()):
        if p.name!='W0' and not p.name.startswith('BASE_'):continue
        rows=read_rows(p);v=validate_rows(rows,panel);validations.append(dict(dataset='E1/'+p.name,**v))
        counts.extend(dict(endpoint=p.name,**r) for r in independent_counts(rows))
        general.append(general_metrics(p.name,rows));endpoints[p.name]=rows
        groups=collections.defaultdict(list)
        for f in sorted((output/'G20/layer-key-drift'/p.name).glob('*.json')):
            for r in read(f):groups[r['layer']].append(r)
        for layer,rr in sorted(groups.items()):
            assert len(rr)==len(panel)
            drift.append(dict(endpoint=p.name,layer=layer,completion_rows=len(rr),valid_input_positions=sum(r['valid_tokens'] for r in rr),
                mean_key_delta_norm=float(np.mean([r['key_delta_norm'] for r in rr])),mean_cosine=float(np.mean([r['cosine'] for r in rr])),
                mean_H1_deltaK_norm=float(np.mean([r['H1_deltaK_norm'] for r in rr])),
                mean_F1t_Kt_norm=float(np.mean([r['F1t_Kt_norm'] for r in rr])),
                max_mapping_relative=max(r['mapping_relative'] for r in rr),
                completion_rows_not_unique_queries=True))
    assert len(endpoints)==25
    for pair in plan['pairs']:
        key=f"{pair['family']}_s{pair['s']:03d}_t{pair['t']:03d}"
        st='G30' if pair['phase']=='core' else 'G50';pt='G40' if pair['phase']=='core' else 'G60'
        states={}
        for state in ('00','10','01','11'):
            rr=read_rows(output/st/key/state);states[state]=rr
            validations.append(dict(dataset=f'{st}/{key}/{state}',**validate_rows(rr,panel)))
            general.append(general_metrics(key+'/'+state,rr))
        actual=endpoints[f"{pair['family']}_W{pair['t']:03d}"]
        assert [(r['nll'],r['token_predictions']) for r in actual]==[(r['nll'],r['token_predictions']) for r in states['11']]
        gen={s:[r for r in rr if r['kind']=='GENERAL'] for s,rr in states.items()}
        interaction=[d['nll']-b['nll']-c['nll']+a['nll'] for a,b,c,d in zip(gen['00'],gen['10'],gen['01'],gen['11'],strict=True)]
        row=dict(contrast=key+'/GENERAL_FACTORIAL_NLL',documents=128,NLL_interaction=float(np.mean(interaction)))
        row.update(cluster_ci(interaction,gen['00']));contrasts.append(row)
        module=[r for f in sorted((output/st/key/'module-interaction-checks').glob('*.json')) for r in read(f)]
        assert [r['row_id'] for r in module]==[r['row_id'] for r in panel]
        checks.append(dict(pair=key,rows=len(module),max_relative=max(r['relative_to_v11'] for r in module),
            all_key_equalities=all(r['K11_K01_exact'] and r['K10_K00_exact'] for r in module)))
        for variant in ['dose_0p5','dose_1','dose_minus1']+[f'rotation_{s}' for s in lock['rotation_seeds']]:
            rr=read_rows(output/pt/key/variant)
            validations.append(dict(dataset=f'{pt}/{key}/{variant}',**validate_rows(rr,panel)))
            general.append(general_metrics(key+'/'+variant,rr));contrasts.append(general_contrast(key+'/'+variant,actual,rr))
    expected=25+48+72;assert len(validations)==expected
    with (attempt/'report/endpoint-summary.csv').open() as f:collected=list(csv.DictReader(f))
    for r in counts:
        found=[q for q in collected if all(q[k]==str(r[k]) for k in ('endpoint','panel','kind'))];assert len(found)==1
        for k in ('count','success','strict','token_correct','token_total','ties'):assert int(found[0][k])==r[k]
    for name,rr in [('independent-endpoints.csv',counts),('general-eval.csv',general),('general-paired.csv',contrasts),
                    ('module-identity.csv',checks),('layer-drift-summary.csv',drift),('row-validation.csv',validations),('stage-chain.csv',stage_rows)]:
        write_csv(dest/name,rr)
    save(dest/'review-receipt.json',dict(status='PASS_STORED_EVIDENCE',model_calls=0,GPU_calls=0,
        analysis_source_sha256=sha(__file__),execution_lock_sha256=sha(attempt/'execution.lock.json'),
        score_datasets=expected,completion_rows=expected*len(panel),source_binding=binding,
        independent_counts_match_collector=True,token_and_strict_recount=True,
        new_checkpoint=False,independent_agent=False,GPU_continuation='NOT_TESTED_NOT_IN_SCOPE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();run(a.attempt,a.destination)
