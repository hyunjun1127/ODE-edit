"""CPU statistical/reference/mechanism reduction of immutable completed B1.

Only reads this output and its already bound generated capsule metadata.
No model, runtime, solver or scheduler invocation. No raw text/token publication.
"""
import argparse
import math
from collections import Counter
from pathlib import Path
import numpy as np
from .review_b1 import ARMS, reduce_raw, dump
from .review_completed_b1 import read, member, csv_write, quantile, paired


def stats(values):
    values=list(values)
    if not values:return dict(n=0,mean=None,min=None,max=None,p50=None,p95=None,p99=None)
    if not all(math.isfinite(x) for x in values):raise ValueError('NONFINITE_ANALYSIS_INPUT')
    return dict(n=len(values),mean=sum(values)/len(values),min=min(values),max=max(values),
                p50=quantile(values,.5),p95=quantile(values,.95),p99=quantile(values,.99))


def cluster(base,now,indices):
    if [r['pair_id'] for r in base]!=[r['pair_id'] for r in now]:raise ValueError('PAIR_ID')
    groups={}
    for a,b in zip(base,now):groups.setdefault(a['case_id'],[]).append((a,b))
    if len(groups)!=indices.shape[1]:raise ValueError('CLUSTER_CARDINALITY')
    result={}
    for key,fun in [('success_pp',lambda a,b:100*(int(b['success'])-int(a['success']))),
                    ('true_nll',lambda a,b:b['true_nll']-a['true_nll']),
                    ('new_nll',lambda a,b:b['new_nll']-a['new_nll']),
                    ('desired_margin',lambda a,b:b['desired_margin']-a['desired_margin'])]:
        values=np.array([sum(fun(a,b) for a,b in g)/len(g) for g in groups.values()])
        boot=values[indices].mean(1)
        result[key]=dict(mean=float(values.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)))
    return result


def choice(scan):
    rows=scan['rows'];total=0;mismatches=0;mus=[];ties=0
    for r in rows:
        n=r['scored_positions'];total+=n
        if any(len(r[k])!=n for k in ('margins','correct','labels','predictions','positions')):raise ValueError('CHOICE_CARDINALITY')
        if r['positions']!=list(range(128,128+n)):raise ValueError('POSITION_SHIFT')
        correct=[a==b for a,b in zip(r['labels'],r['predictions'])]
        if correct!=r['correct'] or sum(not x for x in correct)!=r['mismatches']:raise ValueError('CHOICE_ID_REDUCER')
        if min(r['margins'])!=r['mu']:raise ValueError('WORST_MARGIN')
        if not all(math.isfinite(x) for x in r['margins']):raise ValueError('CHOICE_FINITE')
        mus.append(r['mu']);mismatches+=r['mismatches'];ties+=sum(x==0 for x in r['margins'])
    phi=sum(min(x,0.)**2 for x in mus)/len(rows)
    if not math.isclose(phi,scan['phi_reference'],abs_tol=1e-14,rel_tol=1e-12):raise ValueError('PHI_REDUCER')
    if mismatches!=scan['mismatches']:raise ValueError('MISMATCH_REDUCER')
    return dict(documents=len(rows),positions=total,Phi=phi,mismatches=mismatches,
        all_tokens_preserved_documents=sum(all(r['correct']) for r in rows),worst_margin=min(mus),ties=ties)


def choice_pair(a,b):
    lost=gained=0;change=[];doclost=docgain=0
    for x,y in zip(a['rows'],b['rows'],strict=True):
        if (x['source_row_id'],x['capsule_sha256'],x['positions'],x['labels'])!=(y['source_row_id'],y['capsule_sha256'],y['positions'],y['labels']):raise ValueError('REFERENCE_PAIR_ID')
        lost+=sum(p and not q for p,q in zip(x['correct'],y['correct']))
        gained+=sum(not p and q for p,q in zip(x['correct'],y['correct']))
        doclost+=all(x['correct']) and not all(y['correct']);docgain+=not all(x['correct']) and all(y['correct'])
        change.append(y['mu']-x['mu'])
    return dict(native_safe_new_flips=lost,old_flips_recovered=gained,
                document_lost=doclost,document_recovered=docgain,mu_change=stats(change))


def resolve_reference(path,root):
    d=read(path);seen=set()
    while 'same_endpoint_reuse' in d:
        p=Path(d['same_endpoint_reuse']['path'])
        if not p.resolve().is_relative_to(root.resolve()) or str(p) in seen:raise ValueError('ALIAS_SCOPE_OR_CYCLE')
        seen.add(str(p));assert member(p)['sha256']==d['same_endpoint_reuse']['sha256'];d=read(p)
    return d


def run(output,destination):
    out=Path(output);dest=Path(destination)
    assert read(out/'terminal.json')['status']=='B1_COMPLETE_USER_LIMIT'
    dest.mkdir(parents=True,exist_ok=False);obsroot=out/'B1/observers/current'
    obs={arm:read(obsroot/f'{arm}.json') for arm in (*ARMS,'W0')}
    lock=read(out.parent/'execution.lock.json')
    expected=lock['sample_order'][:100]
    for d in obs.values():
        if [r['case_id'] for r in d['raw']['rewrite_target_new']]!=expected:raise ValueError('LOCKED_ORDER')
    if read(out/'B1/native/native-binding.json')['case_ids']!=expected:raise ValueError('NATIVE_REQUEST_ORDER')
    reduced={arm:reduce_raw(d) for arm,d in obs.items()}
    rng=np.random.default_rng(20260920);indices=rng.integers(0,100,size=(10000,100));distributions=[];cis=[]
    for arm,(summary,rows) in reduced.items():
        for metric,rr in rows.items():
            for field in ('true_nll','new_nll','desired_margin'):
                distributions.append(dict(arm=arm,metric=metric,field=field,**stats(r[field] for r in rr)))
            if arm!='W0':
                for field,s in cluster(reduced['N4'][1][metric],rr,indices).items():
                    cis.append(dict(arm=arm,metric=metric,field=field,**s,unit='100 request clusters',bootstrap=10000,seed=20260920))
    csv_write(dest/'NLL-distributions.csv',distributions);csv_write(dest/'paired-cluster-CI.csv',cis)
    dump(dest/'W0-and-final.json',{arm:r[0] for arm,r in reduced.items()})
    # True-token strict and token accuracy stay separate from pair preference.
    strict=[]
    for arm,d in obs.items():
        for metric,prefix,desired in [('RS','rewrite','new'),('PS','rephrase','new'),('NS','locality','true')]:
            rr=d['raw'][prefix+'_target_'+desired]
            strict.append(dict(arm=arm,metric=metric,prompt_strict=sum(r['all_tokens_correct'] for r in rr),
                prompts=len(rr),token_correct=sum(sum(r['token_correct']) for r in rr),tokens=sum(len(r['token_correct']) for r in rr)))
    csv_write(dest/'strict-token.csv',strict)
    trial_rows=[];acceptance={}
    for arm in ARMS:
        s=read(out/'B1/arms'/arm/'selection-ledger.json')
        for t in s['trials']:
            row=dict(arm=arm,trial=t['trial'],scale=t['scale'],reason=t['reason'],accepted=t['accepted'],
                actual_delta_norm=t.get('actual_delta_norm'),ideal_delta_norm=t.get('ideal_delta_norm'),
                loss=t.get('loss'),actual_p=t.get('actual_p'),actual_armijo_rhs=t.get('actual_armijo_rhs'),
                current=t.get('current',{}).get('pass'),reference_phi=t.get('phi_reference'),
                reference_mismatches=t.get('mismatches'))
            if arm=='EN_KL_Q':
                expected_armijo=t['actual_p']<0 and t['loss']<=t['actual_armijo_rhs'] and t['actual_loss_decrease']>1e-6
                if expected_armijo!=t['armijo']:raise ValueError('ARMIJO_REPLAY')
                if not math.isclose(s['details']['native_loss']-t['loss'],t['actual_loss_decrease'],abs_tol=1e-15):raise ValueError('KL_GAIN_REPLAY')
            trial_rows.append(row)
        if arm=='EN_KL_Q':
            chosen=next(t for t in s['trials'] if t['accepted']);iv=chosen['current']['invariant'];n=lock['numerical']
            checks=dict(ideal=iv['ideal_response_relative']<=n['projector_fp64'],
                actual_DK=iv['actual_max_token_normalized_response']<=n['actual_DK'],
                leakage=iv['actual_projection_leakage_relative']<=n['actual_leak'],
                logitmax=iv['logit_max']<=n['logit_max'],logitrms=iv['logit_rms']<=n['logit_rms'],
                NLL=iv['max_NLL_difference']<=n['current_nll'],
                strict_ID=not iv['strict_symmetric_difference'],pair_ID=not iv['pair_symmetric_difference'])
            if not all(checks.values()) or chosen['current']['reasons']:raise ValueError('SAVED_GUARD_REPLAY')
            acceptance[arm]=dict(trial=chosen['trial'],checks=checks,invariant=iv,
                protected_sequence_rows=len(chosen['current']['rows']),anchor_rows='NOT_SAVED_FOR_INDEPENDENT_PER_SEQUENCE_SUBTRACTION',
                verification='STORED_SCALAR_THRESHOLD_ARITHMETIC; NOT_TENSOR_REPLAY')
    csv_write(dest/'candidate-arithmetic.csv',trial_rows);dump(dest/'acceptance-arithmetic.json',acceptance)
    native=read(out/'B1/reference-native.json');reference=[dict(arm='N4',panel='R512_choice_native',**choice(native))]
    dev=[];kl=[];resolved={}
    for arm in ARMS:
        r=resolve_reference(obsroot/f'{arm}-reference.json',out);resolved[arm]=r
        dev.append(dict(arm=arm,**choice(r['Dev128_decision'])))
        for panel,key in [('R512','R512_train_KL'),('Dev128','Dev128_KL')]:
            block=r[key];rows=block['rows'];loss=sum(x['loss'] for x in rows)/len(rows)
            if not math.isclose(loss,block['loss'],rel_tol=1e-12,abs_tol=1e-14):raise ValueError('KL_DOCUMENT_MEAN')
            kl.append(dict(arm=arm,panel=panel,documents=len(rows),positions=sum(x['scored_positions'] for x in rows),KL=loss,
                reduction='document actual-position mean then document mean',reuse=arm not in ('N4','EN_KL_Q')))
    dump(dest/'Dev128-transitions.json',{arm:choice_pair(resolved['N4']['Dev128_decision'],r['Dev128_decision']) for arm,r in resolved.items()})
    csv_write(dest/'reference-native.csv',reference);csv_write(dest/'Dev128-choice.csv',dev);csv_write(dest/'KL.csv',kl)
    dump(dest/'reference-coverage.json',dict(R512_native_choice='INDEPENDENT_TOKEN_ROWS_REDUCED',
        EN_KL_Q_selected_R512_choice='NOT_MEASURED; postseal stores train KL but no full R512 choice scan',
        DEC_selected_R512_choice='EXACT_NATIVE_FALLBACK; native scan reused by source/receipt binding',
        DEC_candidate_actual_scans=0,Dev128='INDEPENDENT_TOKEN_ROWS_REDUCED',Report256='UNOPENED'))
    # Only immutable metadata capsules, not teacher payloads, are rehashed here.
    generated=read(out/'generated-input-binding.json');ready=read(generated['preparation']['path'])
    mp=Path(ready['manifest']['path']);manifest=read(mp)
    assert member(mp)['sha256']==generated['store']['manifest_sha256']
    caps=[];seen=set()
    for d in manifest['documents']:
        p=mp.parent/d['capsule']['path'];assert member(p)['sha256']==d['capsule']['sha256'];c=read(p)
        if c['status']!='COMPLETE' or c['actual_length']!=len(c['y0']):raise ValueError('CAPSULE_COMPLETE')
        if c['source_row_id'] in seen:raise ValueError('DUPLICATE_REFERENCE')
        seen.add(c['source_row_id']);n=c['actual_length']
        if len(c['input_ids'])!=129 or len(c['tf_input_ids'])!=128+n or c['score_positions']!=list(range(128,128+n)):raise ValueError('GENERATED_SHIFT')
        caps.append(dict(role=c['role'],ordinal=c['ordinal'],Ti=n,censored=c['length_censored'],stop_reason=c['stop_reason']))
    csv_write(dest/'capsule-lengths.csv',caps)
    dump(dest/'capsule-summary.json',dict(manifest=member(mp),capsules_rehashed=len(caps),
        roles={r:dict(length=stats(c['Ti'] for c in caps if c['role']==r),
            stops=dict(Counter(c['stop_reason'] for c in caps if c['role']==r)),
            censored=sum(c['censored'] for c in caps if c['role']==r)) for r in ('R512','Dev128')},
        prior_tf_argmax_verified=ready['store']['canonical_tf_argmax_verified'],
        current_run_tf_argmax_repeated=generated['store']['canonical_tf_argmax_verified'],
        large_teacher_full_rehash=False,current_new_GPU=0))
    # Postselection algebra: all512 row identity and document-normalized means.
    actions=read(out/'B1/mechanism/reference-actions.json');ar=actions['rows']
    for r in ar:
        if not math.isclose(r['native_net_energy']-r['entry_energy'],r['twice_entry_step_inner']+r['native_step_energy'],abs_tol=1e-10,rel_tol=1e-12):raise ValueError('ACTION_IDENTITY')
    means={k:sum(r[k]/r['valid_input_tokens'] for r in ar)/len(ar) for k in actions['document_normalized_means']}
    for k,v in means.items():
        if not math.isclose(v,actions['document_normalized_means'][k],abs_tol=1e-12,rel_tol=1e-12):raise ValueError('ACTION_NORMALIZATION')
    modes=read(out/'B1/mechanism/modes.json')['rows']
    for r in modes:
        s=r['sigma']
        if not math.isclose(r['ridge_gain_sigma_over_one_plus_sigma_squared'],s/(1+s*s),rel_tol=1e-12):raise ValueError('RIDGE_GAIN')
    csv_write(dest/'writer-modes.csv',modes)
    dump(dest/'mechanism-algebra.json',dict(documents=len(ar),valid_tokens=sum(r['valid_input_tokens'] for r in ar),
        normalized_means=means,signed_scalar_response=stats(r['signed_native_scalar_derivative'] for r in ar),
        positive_signed=sum(r['signed_native_scalar_derivative']>0 for r in ar),
        modes=len(modes),H5='NOT_APPLICABLE_COLD_B1',new_model_calls=0))
    components=[]
    for p in sorted((out/'B1/mechanism/interventions').glob('*.json')):
        d=read(p);row=dict(component=p.stem,reference_documents=len(d['reference_rows']),
            reference_mismatches=sum(r['mismatches'] for r in d['reference_rows']),
            reference_Phi=sum(min(r['mu'],0)**2 for r in d['reference_rows'])/len(d['reference_rows']))
        for tag in ('Current','N'):
            data=d[tag];count=correct=ns=0
            for k,r in data.items():
                if r['branch']!='new':continue
                old=data[k[:-3]+'old'];assert r['case_id']==old['case_id']
                # Old/new targets may have different token lengths/positions.
                # Pair by the frozen sequence stem, not target cardinality.
                if not all(math.isfinite(x['nll']) and len(x['positions'])==len(x['labels']) for x in (r,old)):
                    raise ValueError('COMPONENT_SEQUENCE_INVALID')
                count+=1;correct+=r['nll']<old['nll'] if tag=='Current' else old['nll']<r['nll'];ns+=r['strict']
            row[tag+'_count']=correct;row[tag+'_denominator']=count;row[tag+'_new_strict']=ns
        components.append(row)
    csv_write(dest/'postselection-components.csv',components)
    writer=read(out/'B1/mechanism/writer.json')
    dump(dest/'timing-exclusions.json',dict(invalid=writer['timing']['total_inclusive_seconds'],
        field='mechanism/writer.json:timing.total_inclusive_seconds',
        cause='frozen mechanism.py start monotonic overwritten by for start in range(0,m,128)',
        action='EXCLUDED_NOT_REPAIRED_RAW',valid_upper_scope=read(out/'B1/mechanism/COMPLETE.json')['seconds']))
    return dict(reference=reference,Dev128=dev,KL=kl,components=components)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();print(run(a.output,a.destination))
