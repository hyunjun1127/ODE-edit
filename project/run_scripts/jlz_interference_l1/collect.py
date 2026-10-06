"""Independent CPU scalar/row reducer; no model/Slurm writes or backfill."""
import argparse,copy,csv,hashlib,io,json,math,os,pwd,subprocess,traceback
from pathlib import Path
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    Reader,validate_rows as independent_rows,reduce_rows,compare_summary,paired,harmonic,active_flags)
from . import *
from .profile import ARMS,history_expected
from .storage import (write as bounded_write,guard,storage_plan,encode,CANDIDATE_BYTES,
    STATIC_BYTES,COLLECTOR_BYTES,ERROR_RESERVE_BYTES,ROW_BYTES,CHUNK_ROWS,CHUNK_BYTES)
from .storage import write_bytes,ConsoleBudget


def write(path,value):
    path=Path(path);encoded=encode(value,COLLECTOR_BYTES)
    used=sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file() and p!=path and not p.name.endswith('.tmp'))
    require(used+len(encoded)<=COLLECTOR_BYTES,'COLLECTOR_TOTAL_SERIALIZER_BOUND')
    bounded_write(path,value,limit=COLLECTOR_BYTES)


def text_file(path,data):
    raw=data.encode();path=Path(path)
    used=sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file() and p!=path and not p.name.endswith('.tmp'))
    require(used+len(raw)<=COLLECTOR_BYTES,'COLLECTOR_TOTAL_SERIALIZER_BOUND')
    write_bytes(path,raw,limit=8*1024**2)


def optional_receipt(reader,path):
    """Missing/empty/torn terminal or error evidence is not verified success."""
    path=Path(path)
    if not path.exists():return None
    if not path.is_file() or path.is_symlink() or path.stat().st_size==0:
        return dict(status='NOT_VERIFIED',reason='EMPTY_OR_UNSAFE_RECEIPT',path=str(path))
    try:return reader.json(path)
    except (json.JSONDecodeError,UnicodeDecodeError) as error:
        return dict(status='NOT_VERIFIED',reason='INVALID_RECEIPT_BYTES',path=str(path),error=str(error))

def finite(value):
    if isinstance(value,float):require(math.isfinite(value),'NONFINITE_RAW_SCALAR')
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(tuple,list)):
        for v in value:finite(v)

def close(left,right,name,rtol=1e-12,atol=1e-12):
    require(math.isclose(left,right,rel_tol=rtol,abs_tol=atol),name)


def matrix(value,L,B,name):
    require(isinstance(value,list) and len(value)==L and all(isinstance(v,list) and len(v)==B for v in value),name)
    finite(value);return value


def validate_price(record,profile,B,batch):
    finite(record);L=len(profile['eligible_layers'])
    require(record['B']==B and record['layers']==profile['eligible_layers'] and record['arm']==profile['arm']
        and record['batch']==batch and record['price_extra_model_calls']==record['price_extra_solves']==0
        and record['no_durable_matrices'],'STATIC_PRICE_SOURCE_SCOPE')
    anchors=matrix(record['anchors'],L,B,'STATIC_NATIVE_ANCHORS');caps=matrix(record['local_caps'],L,B,'STATIC_NATIVE_CAPS')
    raw=matrix(record['raw_kappa'],L,B,'STATIC_RAW_SCORE');floor=matrix(record['floored_kappa'],L,B,'STATIC_FLOORED_SCORE')
    computed=matrix(record['computed_pi'],L,B,'STATIC_COMPUTED_PRICE');effective=matrix(record['effective_pi'],L,B,'STATIC_APPLIED_PRICE')
    d=matrix(record['denominator'],L,B,'STATIC_DENOMINATORS')
    require(all(a>0 for row in anchors for a in row) and all(v>=0 for row in raw for v in row),'PRICE_ANCHOR_SCORE_DOMAIN')
    require(B==1 or all(v>1e-8 for row in d for v in row),'PRICE_DENOMINATOR_NOT_EPSILON_FALLBACK')
    require(set(record['mean_M_hash'])==set(map(str,profile['eligible_layers']))
        and all(row['mean_definition_exact'] for row in record['layer_checks']),'C0_MEAN_M_DEFINITION_RECEIPT')
    reverse=record['reverse_permutation'];changed=[];tie_counts=[]
    for r in range(B):
        maximum=max(raw[l][r] for l in range(L));allzero=maximum==0
        minimum=min(max(raw[l][r],1e-6*maximum,1e-12) for l in range(L))
        require(record['allzero_request'][r]==allzero,'PRICE_ALL_ZERO_NEUTRAL')
        close(record['max_raw'][r],maximum,'PRICE_MAX_RAW')
        close(record['min_floored'][r],minimum,'PRICE_MIN_FLOORED')
        close(record['relative_floor'][r],1e-6*maximum,'PRICE_RELATIVE_FLOOR')
        close(record['absolute_floor'][r],1e-12,'PRICE_ABSOLUTE_FLOOR')
        for l in range(L):
            close(caps[l][r],.75*anchors[l][r],'NATIVE_LAYER_LOCAL_CAP')
            expected=max(raw[l][r],1e-6*maximum,1e-12)
            close(floor[l][r],expected,'PRICE_FLOOR_NO_CLIP')
            value=1. if B==1 or allzero else expected/minimum
            close(computed[l][r],value,'PRICE_NORMALIZATION')
            require(record['floor_mask'][l][r]==(expected>raw[l][r]) and record['zero_score_mask'][l][r]==(raw[l][r]==0),'PRICE_FLOOR_MASK')
        order=sorted(range(L),key=lambda l:(computed[l][r],l));assignment=list(reversed(order))
        require(reverse[r]['owner']==r and reverse[r]['sorted_layer_indices']==order
            and reverse[r]['reverse_assignment_indices']==assignment,'STABLE_PRICE_REVERSE_PERMUTATION')
        expected=[computed[l][r] for l in range(L)]
        if profile['arm']=='FLAT':expected=[1.]*L
        elif profile['arm']=='REVERSE':
            for destination,source in zip(order,assignment):expected[destination]=computed[source][r]
        for l in range(L):close(effective[l][r],expected[l],'APPLIED_ARM_PRICE')
        if profile['arm']=='REVERSE':require(sorted(expected)==sorted(computed[l][r] for l in range(L)),'REVERSE_SAME_MULTISET')
        changed.append(sum(effective[l][r]!=computed[l][r] for l in range(L)))
        tie_counts.append(sum(computed[order[l]][r]==computed[order[l-1]][r] for l in range(1,L)))
        close(record['beta_base'][r],.75,'DIMENSIONLESS_BETA_BASE')
        close(record['beta_max'][r],.75*max(expected),'DIMENSIONLESS_BETA_CEILING')
        close(record['effective_pi_max'][r],max(expected),'APPLIED_PRICE_MAX')
    require(record['changed_assignment_count']==changed and record['exact_tie_count']==tie_counts
        and record['unchanged_request_count']==sum(x==0 for x in changed),'REVERSE_ASSIGNMENT_COUNTS')
    loo=record['loo']
    require(loo['extra_model_calls']==loo['extra_solves']==0,'CACHED_LOO_NO_EXTRA_MODEL_SOLVE')
    if batch==1 and B>1:
        require(loo['status']=='ACTUAL_C0_CACHED_MATVEC_CHECKED' and loo['extra_factorizations']==0
            and [(row['layer'],row['source'],row['recipient']) for row in loo['pairs']]==
                [(layer,r,j) for layer in profile['eligible_layers'] for r,j in ((0,1),(1,0))],'ACTUAL_B1_ALL_LAYER_FIXED_LOO_PAIRS')
        for row in loo['pairs']:
            require(row['raw_A_unsymmetrized'] and row['orientation']=='q.T@k_recipient','RAW_A_LOO_ORIENTATION')
            if row['key_norm']==0:require(row['residual_absolute']<=1e-8,'LOO_ZERO_KEY_RESIDUAL')
            else:
                close(row['residual_relative'],row['residual_absolute']/row['key_norm'],'LOO_RELATIVE_DEFINITION')
                require(row['residual_relative']<=1e-6,'LOO_RELATIVE_RESIDUAL')
            error=abs(row['coefficient_observed']-row['coefficient_expected'])
            close(row['error'],error,'LOO_COEFFICIENT_ERROR')
            close(row['limit'],1e-8+1e-6*abs(row['coefficient_expected']),'LOO_FIXED_COEFFICIENT_LIMIT')
            require(error<=row['limit'],'LOO_COEFFICIENT_TOLERANCE')
    elif batch>1:require(loo['status']=='NOT_REQUESTED_AFTER_B1' and not loo['pairs'],'NO_REPEATED_LOO_STUDY')
    return anchors,caps,effective


def validate_projection(p,caps,weights,beta,B):
    L=len(caps);pre=matrix(p['pre_norm'],L,B,'PROPOSAL_NORM_SCHEMA')
    target=matrix(p['fp64_projected_norm'],L,B,'FP64_PROJECTED_NORM_SCHEMA')
    stored=matrix(p['post_norm'],L,B,'POSTCAST_NORM_SCHEMA')
    require(p['coordinate']=='absolute_R_Euclidean' and not p['postcast_repair'] and p['lr']==.1 and p['eps']==1e-8
        and not p['moment_reset'],'ABSOLUTE_ADAM_NO_HIDDEN_REPAIR')
    for r in range(B):
        tau=p['tau'][r];require(tau>=0,'PROJECTION_DUAL_NONNEGATIVE')
        close(p['beta'][r],beta[r],'PROJECTION_SAME_CONTROLLER_BETA')
        budget=beta[r];spend=0.;postspend=0.
        for l in range(L):
            n=pre[l][r];cap=caps[l][r];w=weights[l][r];t=target[l][r];post=stored[l][r]
            require(n>=0 and t>=0 and post>=0,'PROJECTION_NONNEGATIVE_NORMS')
            expected=min(cap,max(n-tau*w,0.))
            require(abs(t-expected)<=1e-10*max(1,n,cap),'INDEPENDENT_FP64_LENGTH_KKT')
            require(post-cap<=1e-6,'INDEPENDENT_FP32_LOCAL_CAP')
            close(p['local_excess'][l][r],max(post-cap,0.),'POSTCAST_LOCAL_EXCESS_RECEIPT')
            require(p['zero_mask'][l][r]==(t==0) and p['capped_mask'][l][r]==(t==cap and t!=0)
                and p['free_mask'][l][r]==(t!=0 and t!=cap),'ZERO_CAPPED_FREE_MASKS')
            spend+=w*t;postspend+=w*post
        require(spend-budget<=1e-10*max(1,budget)
            and abs(tau*(spend-budget))<=1e-10*max(1,tau*budget),'INDEPENDENT_FP64_PRIMAL_COMPLEMENTARITY')
        require(postspend-budget<=1e-6*max(1,budget),'INDEPENDENT_FP32_SHARED_SPEND')
        close(p['weighted_spend'][r],postspend,'INDEPENDENT_WEIGHTED_SPEND',atol=1e-10)
        close(p['shared_excess'][r],max(postspend-budget,0.),'INDEPENDENT_SHARED_EXCESS',atol=1e-10)
        close(p['shared_limit'][r],1e-6*max(1,budget),'FP32_FIXED_SHARED_LIMIT')
        if not p['shared_active'][r]:require(tau==0,'INACTIVE_SHARED_DUAL_ZERO')


def read_candidate_stream(reader,fit,arm,batch):
    ref=fit['candidate_stream'];path=Path(ref['path'])
    require(path.name=='events.jsonl' and path.parent.name==f'batch-{batch:02d}'
        and path.parent.parent.name==arm and path.stat().st_size<=25*CANDIDATE_BYTES,'EXACT_BOUNDED_CANDIDATE_STREAM')
    data=reader.bytes(path)
    require(len(data)==ref['bytes'] and sha(path)==ref['sha256'] and data.endswith(b'\n'),'STREAM_IMMUTABLE_BYTES')
    lines=data.splitlines();require(1<=len(lines)<=25 and ref['line_count']==len(lines),'CANDIDATE_STREAM_LINE_COUNT')
    result=[];record_digest=hashlib.sha256()
    for k,line in enumerate(lines):
        require(len(line)+1<=CANDIDATE_BYTES,'CANDIDATE_RECORD_SERIALIZER_MAX')
        row=json.loads(line);require(row['task']==TASK and row['arm']==arm and row['batch']==batch
            and row['event']=='candidate' and row['payload']['candidate']==k,'AUTHORITATIVE_STREAM_IDENTITY_ORDER')
        record_digest.update(json.dumps(row['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n')
        result.append(row['payload'])
    require(record_digest.hexdigest()==fit['candidate_record_digest'],'FIT_RECORD_DIGEST_SAME_AUTHORITATIVE_ROWS')
    return result


def validate_fit(fit,profile,events,price,B=100):
    finite(fit);n=fit['candidates'];batch=price['batch'];anchors,caps,prices=validate_price(price,profile,B,batch)
    require('events' not in fit and 1<=n<=25 and len(events)==n and fit['updates']==n-1<=24
        and fit['logical_builds']==fit['logical_subject_forwards']==n
        and fit['logical_subject_backwards']<=24,'STREAM_ONLY_25_24_BUILD_SUBJECT_BUDGET')
    require(fit['terminal_candidate']==n-1 and fit['terminal_no_backward'] and fit['terminal_extra_forward']==0
        and fit['norm_analytic_once'] and fit['production_builder_reverse']==fit['production_solve_VJP']==0
        and fit['requested_gradient']=='FP64_Lambda_M_rows_T','FIXED_APPROXIMATE_PULLBACK_AND_TERMINAL')
    price_hash=digest({k:v for k,v in price.items() if k!='record_sha256'})
    require(fit['entry_price_sha256']==price['record_sha256']==price_hash,'STATIC_PRICE_HASH_FIXED_FOR_ENTIRE_FIT')
    weights=[[prices[l][r]/anchors[l][r] for r in range(B)] for l in range(len(anchors))]
    oldactive=[False]*B;oldt=[0]*B;olde=[0]*B;oldbeta=[.75]*B;backwards=0
    for k,row in enumerate(events):
        finite(row);require(row['candidate']==k and row['ordinal']==k+1 and len(row['F'])==B,'CANDIDATE_ORDER_CARDINALITY')
        require(row['entry_price_sha256']==price_hash,'EACH_CANDIDATE_SAME_FROZEN_PRICE_HASH')
        mask=[oldactive[r] or row['F'][r]>=.05 for r in range(B)]
        ctrl=row['controller'];require(row['active_mask']==mask and ctrl['active']==mask,'IRREVERSIBLE_NATIVE_LOSS_ACTIVE')
        require(ctrl['update_counts']==oldt and ctrl['expansion']==olde,'OWN_REQUEST_COUNTER_JOIN')
        require(ctrl['beta']==oldbeta,'BETA_CURRENT_CANDIDATE_JOIN')
        for r in range(B):
            expected=.75 if olde[r]==0 else (.75*max(prices[l][r] for l in range(len(anchors))) if olde[r]==4
                else .75*math.exp(olde[r]/4*math.log(max(prices[l][r] for l in range(len(anchors))))))
            close(ctrl['beta'][r],expected,'INDEPENDENT_DIMENSIONLESS_EXPANSION')
        terminal=k==24 or not any(mask)
        require(row['terminal']==terminal and (k==n-1)==terminal and row['backward']==(not terminal and any(mask)),
            'LAST_EVALUATED_TERMINAL_NO_EXTRA_BACKWARD')
        backwards+=row['backward']
        close(row['full_task_sum'],sum(row['F']),'FULL_FORWARD_TASK_SUM',atol=1e-8)
        close(row['masked_backward_sum'],sum(f for f,m in zip(row['F'],mask) if m),'ACTIVE_REQUEST_SUM',rtol=1e-7,atol=1e-7)
        telemetry=row['telemetry']
        require(telemetry['candidate']==k and set(telemetry['layers'])==set(map(str,profile['eligible_layers'])),'CANDIDATE_SCALAR_TELEMETRY_LAYERS')
        current_norm=[telemetry['layers'][str(l)]['requested_norm'] for l in profile['eligible_layers']]
        for r in range(B):
            for li,l in enumerate(profile['eligible_layers']):
                close(telemetry['layers'][str(l)]['normalized_norm'][r],current_norm[li][r]/anchors[li][r],'NORMALIZED_NORM_SCALAR')
            close(row['norm'][r],.5/price['anchor_star'][r]**2*sum(values[r] for values in current_norm),'NATIVE_NORM_ONCE',atol=1e-10)
        if terminal:
            require(row['gradient_status']=='NO_BACKWARD_TERMINAL' and 'projection' not in row,'TERMINAL_GRADIENT_NULL')
            oldactive=mask;break
        after=row['post_update_controller'];projection=row['projection']
        expected_t=[t+int(active) for t,active in zip(oldt,mask)]
        expected_e=[e+int(active and t>=12 and F>=.05 and e<4 and maximum>.75)
            for e,active,t,F,maximum in zip(olde,mask,oldt,row['F'],price['beta_max'])]
        require(after['update_counts']==expected_t and after['expansion']==expected_e
            and projection['adam_updates']==expected_t,'OWN_GRACE12_BEFORE13_EXPANSION')
        expectedbeta=[price['beta_max'][r] if expected_e[r]==4 else .75*math.exp(expected_e[r]/4*math.log(price['effective_pi_max'][r])) for r in range(B)]
        for r in range(B):close(after['beta'][r],expectedbeta[r],'POST_UPDATE_BETA')
        validate_projection(projection,caps,weights,after['beta'],B)
        for r in range(B):
            if not mask[r]:require(all(projection['post_norm'][l][r]==current_norm[l][r] for l in range(len(anchors))),'INACTIVE_OWNER_STATE_UNCHANGED')
        oldactive,oldt,olde,oldbeta=mask,expected_t,expected_e,after['beta']
    require(backwards==fit['logical_subject_backwards'] and sum(oldt)==fit['request_updates']<=B*24
        and fit['controller']['update_counts']==oldt and fit['controller']['expansion']==olde
        and fit['controller']['beta']==oldbeta,'TERMINAL_COUNTERS_AND_PRICE_STATE')
    states=['ZERO_STEP' if not oldactive[r] else ('SATISFIED_EXPANDED' if olde[r] else 'SATISFIED_BASE') if events[-1]['F'][r]<.05
        else 'UNSATISFIED_MAX' if oldbeta[r]>=price['beta_max'][r] else 'UNSATISFIED' for r in range(B)]
    require(fit['terminal_states']==states,'TERMINAL_STATE_POPULATION')
    return dict(builds=n,subject_forwards=n,subject_backwards=backwards,request_updates=sum(oldt),
        expanded_requests=sum(e>0 for e in olde),statuses={s:states.count(s) for s in set(states)},
        entry_price_sha256=fit['entry_price_sha256'],price_frozen_entire_batch=True)

def endpoint(reader,folder,identities,ids,name,state_value,seen):
    folder=Path(folder)
    if not (folder/'summary.json').exists():return None
    rows=[]
    for path in sorted(folder.glob('chunk-*.json')):
        require(path.stat().st_size<=CHUNK_BYTES,'BOUNDED_METRIC_CHUNK_BYTES')
        chunk=reader.json(path);require(chunk['state']==state_value and chunk['optimizer_feedback'] is False,'RAW_ENDPOINT_STATE')
        require(len(chunk['rows'])<=CHUNK_ROWS,'BOUNDED_METRIC_CHUNK_ROWS')
        for row in chunk['rows']:
            encode(row,ROW_BYTES)
            close(row['margin_new_minus_true'],row['new_nll']-row['true_nll'],'EXPLICIT_NEW_MINUS_TRUE_MARGIN',atol=0.,rtol=0.)
        rows.extend(chunk['rows'])
    independent_rows(rows,expected_rows(identities,ids),name)
    result=reduce_rows(rows);saved=reader.json(folder/'summary.json')
    require(saved['endpoint']==name and saved['state']==state_value and saved['requests']==len(ids)
        and saved['no_mutation'] and saved['optimizer_feedback'] is False,'ENDPOINT_RECEIPT')
    require({k:v['denominator'] for k,v in result.items()}==dict(R=len(ids),P=2*len(ids),N=10*len(ids)),'EXACT_ENDPOINT_DENOMINATORS')
    compare_summary(result,saved['summary']);flags=active_flags(seen)
    require(all(r['active_at_endpoint']==flags[r['case_id']] for r in rows),'SEEN_PREFIX_METADATA')
    return dict(rows=rows,metrics=result,seconds=saved['seconds'])

def _arm_review(reader,attempt,c,lock,arm,records,identities,progress):
    out=attempt/arm;ids=[r['case_id'] for r in records];profile=c['arm_profiles'][arm];layers=list(map(str,profile['eligible_layers']))
    cold={key:{l:c['cold_W0_H0'][key][l] for l in layers} for key in ('W','H')}
    commits=[];metrics=[];pairs=[];cost=[];realizations=[];prefix={};atwrite=[]
    counts=dict(builds=0,subject_forwards=0,subject_backwards=0,request_updates=0)
    w0=endpoint(reader,out/'W0',identities,ids,'W0',cold,records)
    if w0:metrics.append(dict(endpoint='W0',requests=2000,metrics=w0['metrics']))
    for number in range(1,21):
        folder=out/f'batch-{number:02d}'
        if not (folder/'commit.json').exists():break
        commit=optional_receipt(reader,folder/'commit.json')
        if commit is None or commit.get('status')=='NOT_VERIFIED':break
        entry=reader.json(folder/'entry.json');pack=c['packs'][number-1]
        require(commit['task']==entry['task']==TASK and commit['arm']==entry['arm']==arm
            and commit['batch']==entry['batch']==number and commit['source']==entry['source']==lock['source_commit']
            and commit['config']==entry['config']==digest(c) and commit['profile']==entry['profile']==digest(profile),'ARM_SOURCE_CONFIG_BATCH')
        require(commit['ids']==entry['ids']==pack['ids'] and commit['native_pack']==entry['native_pack']==pack['identity'],'NATIVE_PACK_ORDER')
        require(commit['before']==entry['state'] and commit['RNG_before']==entry['RNG'],'ENTRY_STATE_RNG')
        if commits:
            previous=commits[-1];require(commit['before']==previous['after'] and commit['RNG_before']==previous['RNG_after']
                and entry['ledger']==previous['ledger'] and entry['context']==previous['context'],'19_OWN_STATE_JOINS')
        else:require(commit['before']==cold,'OWN_COLD_W0_H0')
        require(commit['observer_no_mutation'] and commit['RNG_after']==commit['RNG_before'],'OBSERVER_RNG_NONMUTATION')
        require(commit['history_appends']==len(layers) and set(commit['history'])==set(layers),'ELIGIBLE_HISTORY_COUNT')
        for l,h in commit['history'].items():
            require(h['appends']==1 and h['requests']==100 and h['rows']=='native_rewrite_only_nestedmean'
                and h['zero_and_unsatisfied_requests_included'] and h['before']==commit['before']['H'][l]
                and h['after']==commit['after']['H'][l],'HISTORY_ALL_REQUESTS_ONCE')
        fit=reader.json(verify(commit['fit']));events=read_candidate_stream(reader,fit,arm,number)
        price_path=Path(fit['candidate_stream']['entry_price']['path'])
        require(price_path==folder/'entry-price.json' and price_path.stat().st_size<=STATIC_BYTES,'EXACT_ONCE_ENTRY_PRICE_FILE')
        price_envelope=reader.json(verify(fit['candidate_stream']['entry_price']))
        require(price_envelope['task']==TASK and price_envelope['arm']==arm and price_envelope['batch']==number
            and price_envelope['event']=='entry_price','ENTRY_PRICE_ENVELOPE_IDENTITY')
        price=price_envelope['payload']
        require(price['pack_identity']==entry['native_pack'] and price['entry_weight_hash']==entry['state']['W']
            and price['entry_state_sha256']==digest(entry['state'])
            and price['source_binding']==dict(source=lock['source_commit'],config_sha256=lock['config_sha256'],
                profile=digest(profile),batch=number,arm=arm,pack=pack['identity'],state_sha256=digest(entry['state'])),
            'STATIC_PRICE_OWN_ENTRY_SOURCE_CONFIG_BINDING')
        cnt=validate_fit(fit,profile,events,price)
        for k in counts:counts[k]+=cnt[k]
        writer=reader.json(verify(commit['writer']))
        require(writer['accepted_weight_copy_exact'] and writer['terminal_last_evaluated_not_best'] and writer['no_resolve']
            and writer['no_double_add'] and writer['candidate']==fit['terminal_candidate']==commit['accepted_candidate'],'LAST_EVALUATED_EXACT_COMMIT')
        require(writer['weight_hashes']==commit['after']['W'] and writer['history']==commit['history'],'PAYLOAD_AND_H_IDENTITIES')
        capture=reader.json(folder/'entry-capture.json');require(capture['fresh_capture'] and capture['H_entry']==entry['state']['H']
            and capture['anchor_layer']==8 and capture['eligible_layers']==profile['eligible_layers'],'FRESH_TEACHER_ANCHOR_FACTOR')
        realization=reader.json(verify(writer['realization']));finite(realization)
        require(realization['candidate']==commit['accepted_candidate'] and realization['B']==100 and set(realization['layers'])==set(layers)
            and realization['terminal_extra_planner_forward']==realization['terminal_extra_planner_backward']==0,'TERMINAL_REALIZATION_BINDING')
        realizations.append(dict(batch=number,realization=realization))
        gap=reader.json(verify(writer['actual_gap']));finite(gap)
        require(gap['record_only'] and len(gap['subject_F'])==len(gap['actual_alltoken_F'])==100,'SUBJECT_ACTUAL_GAP_NOT_EXACTNESS')
        current=records[(number-1)*100:number*100];seen=records[:number*100];birth=pack['ids']
        pre=endpoint(reader,folder/'pre',identities,birth,f'B{number}_PRE',commit['before'],seen)
        selected=seen if number in MILESTONES else current;selected_ids=[r['case_id'] for r in selected]
        post=endpoint(reader,folder/'post',identities,selected_ids,f'W{number}',commit['after'],seen)
        require(pre is not None and post is not None,'COMMITTED_OBSERVATION_MISSING');compare_summary(pre['metrics'],commit['pre']);compare_summary(post['metrics'],commit['post'])
        birthrows=[r for r in post['rows'] if r['case_id'] in set(birth)];compare_summary(reduce_rows(birthrows),commit['post_current'])
        metrics.extend([dict(endpoint=f'B{number}_PRE',requests=100,metrics=pre['metrics']),
            dict(endpoint=f'W{number}_CURRENT',requests=100,metrics=reduce_rows(birthrows))])
        pairs.append(dict(from_endpoint=f'B{number}_PRE',to_endpoint=f'W{number}_CURRENT',paired=paired(pre['rows'],birthrows)));atwrite.extend(birthrows)
        if number in MILESTONES:
            prefix[number]=post['rows'];metrics.append(dict(endpoint=f'W{number}_ALL_SEEN',requests=len(seen),metrics=post['metrics']))
            metrics.append(dict(endpoint=f'W{number}_FIRST500',requests=500,metrics=reduce_rows([r for r in post['rows'] if r['case_id'] in set(ids[:500])])))
            metrics.append(dict(endpoint=f'W{number}_FIRST100',requests=100,metrics=reduce_rows([r for r in post['rows'] if r['case_id'] in set(ids[:100])])))
            pairs.append(dict(from_endpoint='AT_WRITE',to_endpoint=f'W{number}',paired=paired(atwrite,post['rows'])))
            for born in range(1,number+1):
                cohort=set(ids[(born-1)*100:born*100])
                pairs.append(dict(from_endpoint=f'B{born}_AT_WRITE',to_endpoint=f'W{number}',birth_cohort=born,
                    paired=paired([r for r in atwrite if r['case_id'] in cohort],[r for r in post['rows'] if r['case_id'] in cohort])))
            if w0:pairs.append(dict(from_endpoint='W0',to_endpoint=f'W{number}',paired=paired([r for r in w0['rows'] if r['case_id'] in set(ids[:number*100])],post['rows'])))
        cost.append(dict(batch=number,batch_inclusive_seconds=commit['seconds'],entry_capture_seconds=capture['seconds'],
            fit_inclusive_seconds=fit['seconds'],build_seconds=fit['build_seconds'],subject_seconds=fit['subject_seconds'],
            pullback_seconds=fit['pullback_seconds'],optimizer_seconds=fit['optimizer_seconds'],projection_seconds=fit['projection_seconds'],
            candidate_scalar_telemetry_seconds=fit['candidate_scalar_telemetry_seconds'],
            writer_inclusive_seconds=writer['seconds'],scalar_terminal_seconds=realization['seconds'],
            observer_pre_seconds=pre['seconds'],observer_post_seconds=post['seconds'],logical_and_physical_fit_calls=fit['call_counts'],
            additional_actual_native_groups=gap['extra_actual_native_forward_groups'],
            price_seconds=fit['price_seconds'],LOO_seconds=fit['loo_seconds'],
            candidate_stream_IO_seconds=fit['candidate_stream']['io_seconds'],entry_price_sha256=fit['entry_price_sha256'],
            actual_B1_assertions=price['loo'] if number==1 else 'NOT_REQUESTED_AFTER_B1',
            fixed_price_scope='own batch c0 only; upper candidate K/P still fresh',
            timing_policy='exclusive fit stages; fit/writer/batch inclusive timers not additive again'))
        commits.append(commit)
        progress.update(commits=copy.deepcopy(commits),metrics=copy.deepcopy(metrics),paired=copy.deepcopy(pairs),
            realization=copy.deepcopy(realizations),cost=copy.deepcopy(cost),counters=dict(counts),prefix=copy.deepcopy(prefix))
    terminal=optional_receipt(reader,out/'terminal.json')
    firsterror=optional_receipt(reader,out/'first-error.json')
    abandoned=[]
    for folder in sorted(out.glob('batch-*')):
        if folder.name not in {f'batch-{i:02d}' for i in range(1,len(commits)+1)}:
            abandoned.append(dict(batch=folder.name,logical_commit=False,rollback=optional_receipt(reader,folder/'rollback.json'),
                nonempty_files=[dict(path=str(p),bytes=p.stat().st_size) for p in folder.rglob('*') if p.is_file() and p.stat().st_size>0],
                zero_byte_files=[str(p) for p in folder.rglob('*') if p.is_file() and p.stat().st_size==0],
                rollback_or_terminal_without_valid_receipt='NOT_VERIFIED'))
    complete=w0 is not None and len(commits)==20 and 20 in prefix and terminal is not None and terminal['status']=='W20_COMPLETE'
    if len(commits)==20:require({k:v['denominator'] for k,v in reduce_rows(prefix[20]).items()}==dict(R=2000,P=4000,N=20000),'W20_FULL_DENOMINATORS')
    result=dict(arm=arm,status='W20_COMPLETE' if complete else 'PARTIAL_OR_TECHNICAL_BLOCKED',commits=len(commits),requests=len(commits)*100,
        expected=dict(commits=20,requests=2000,joins=19,history_appends=history_expected(arm)),
        actual=dict(joins=max(0,len(commits)-1),history_appends=sum(r['history_appends'] for r in commits)),
        counters=counts,metrics=metrics,paired=pairs,realization=realizations,cost=cost,terminal=terminal,first_error=firsterror,
        uncommitted_attempts=abandoned,
        checkpoint_saved=False,exact_resume='NOT_AVAILABLE',no_missing_as_zero=True)
    return result,prefix


def arm_review(reader,attempt,c,lock,arm,records,identities):
    """A corrupt arm cannot erase already-verified prefix or other arm evidence."""
    progress={}
    try:return _arm_review(reader,attempt,c,lock,arm,records,identities,progress)
    except Exception as error:
        commits=progress.get('commits',[])
        result=dict(arm=arm,status='CPU_REVIEW_TECHNICAL_BLOCKED_PREFIX_ONLY',commits=len(commits),requests=100*len(commits),
            expected=dict(commits=20,requests=2000,joins=19,history_appends=100),
            actual=dict(joins=max(0,len(commits)-1),history_appends=sum(v['history_appends'] for v in commits)),
            counters=progress.get('counters',dict(builds=0,subject_forwards=0,subject_backwards=0,request_updates=0)),
            metrics=progress.get('metrics',[]),paired=progress.get('paired',[]),realization=progress.get('realization',[]),
            cost=progress.get('cost',[]),terminal=optional_receipt(reader,attempt/arm/'terminal.json'),
            first_error=dict(type=type(error).__name__,error=str(error),source='independent CPU reducer',
                traceback=traceback.format_exc(),original_KEEP=True,automatic_retry=False),
            uncommitted_attempts='NOT_CERTIFIED_AFTER_FIRST_REDUCER_ERROR',no_missing_as_zero=True,
            checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
        return result,progress.get('prefix',{})

def accounting_snapshot(attempt):
    receipt=attempt/'submission.json'
    if not receipt.exists():return dict(status='NOT_AVAILABLE',reason='NO_EXACT_LOCAL_SUBMISSION')
    data=json.loads(receipt.read_text());ids=[str(v) for v in data['jobs'].values()]
    require(all(x.isdigit() for x in ids),'EXACT_JOB_IDS')
    response=subprocess.run(['sacct','-n','-P','-j',','.join(ids),
        '--format=JobIDRaw,User,JobName%120,State,ElapsedRaw,AllocTRES,TotalCPU,ExitCode'],check=True,capture_output=True,text=True,timeout=30)
    roles={str(job):role for role,job in data['jobs'].items()}
    parents=[]
    for line in response.stdout.splitlines():
        v=line.split('|')
        if len(v)<8 or v[0] not in ids:continue
        require(v[1]==pwd.getpwuid(os.getuid()).pw_name and v[2]==TASK+'-'+roles[v[0]],'EXACT_PARENT_OWNER_NAME')
        tres=dict(x.split('=',1) for x in v[5].split(',') if '=' in x);gpu=int(tres.get('gres/gpu',0));elapsed=int(v[4] or 0)
        require(gpu==0 if roles[v[0]]=='collector' else gpu in (0,1),'EXACT_PARENT_ALLOCATED_GPU')
        parents.append(dict(job=int(v[0]),role=roles[v[0]],state=v[3],elapsed_seconds=elapsed,GPUs=gpu,allocated_GPU_seconds=gpu*elapsed,TotalCPU=v[6],exit=v[7]))
    observed={str(v['job']) for v in parents};missing=sorted(set(ids)-observed)
    return dict(status='ONE_BOUNDED_EXACT_PARENT_QUERY' if not missing else 'PARTIAL_ACCOUNTING_NOT_AVAILABLE',parents=parents,
        expected_job_ids=ids,observed_job_ids=sorted(observed),missing_job_ids=missing,cost_complete=not missing,
        observed_allocated_GPU_seconds=sum(v['allocated_GPU_seconds'] for v in parents),
        allocated_GPU_seconds=sum(v['allocated_GPU_seconds'] for v in parents) if not missing else None,
        pending_not_allocated=True,steps_not_double_counted=True,collector_GPU0=True)

def _collect(attempt,out,accounting):
    guard(attempt,2*COLLECTOR_BYTES+ERROR_RESERVE_BYTES,inodes=16)
    out.mkdir(parents=True,exist_ok=False);reader=Reader();c=reader.json(attempt/'config.json');lock=reader.json(attempt/'execution.lock.json')
    require(c['task_id']==lock['task_id']==TASK and c['instruction_id']==lock['instruction_id']==NONCE
        and sha(attempt/'config.json')==lock['config_sha256'],'COLLECTOR_SOURCE_CONFIG_AUTHORITY')
    require(c['storage']==storage_plan(),'COLLECTOR_SEALED_STORAGE_BOUND')
    records=load_prefix(Path(c['stream']).parent,2000);require(digest([r['case_id'] for r in records])==ORDERED_SHA,'ORDERED_FIXED_COHORT')
    identities=reader.json(verify(c['observer_identity']))['rows'];arms={};prefix={}
    for arm in ARMS:arms[arm],prefix[arm]=arm_review(reader,attempt,c,lock,arm,records,identities)
    cross=[]
    for arm in ARMS[1:]:
        for k in MILESTONES:
            if k in prefix['PRICE'] and k in prefix[arm]:
                cross.append(dict(reference='PRICE',arm=arm,endpoint=f'W{k}',cohort='ALL_SEEN',
                    paired=paired(prefix['PRICE'][k],prefix[arm][k])))
                fixed=set(r['case_id'] for r in records[:500])
                cross.append(dict(reference='PRICE',arm=arm,endpoint=f'W{k}',cohort='SAME_FIRST500',
                    paired=paired([r for r in prefix['PRICE'][k] if r['case_id'] in fixed],
                                  [r for r in prefix[arm][k] if r['case_id'] in fixed])))
    complete=all(v['status']=='W20_COMPLETE' for v in arms.values());parents=accounting_snapshot(attempt) if accounting else dict(status='NOT_REQUESTED')
    result=dict(task=TASK,status='THREE_ARMS_W20_COMPLETE' if complete else 'PARTIAL_OR_TECHNICAL_BLOCKED',source=lock['source_commit'],config_sha256=lock['config_sha256'],
        arms=arms,standalone_qualification=False,toy_runs=0,cross_arm_paired=cross,actual=dict(commits=sum(v['commits'] for v in arms.values()),
        history_appends=sum(v['actual']['history_appends'] for v in arms.values()),joins=sum(v['actual']['joins'] for v in arms.values())),
        expected=dict(commits=60,joins=57,history_appends=300),accounting=parents,
        CPU_review='Independent immutable-row arithmetic/token identity/state linkage, no target-model replay',
        source_review='Static/import only; no toy or numerical qualification; actual B1 c0 and normal proposal assertions are in the trajectory',
        static_price_verification='Scalar raw-score/floor/normalization/permutation/beta/projector arithmetic; M hashes are identity evidence, not tensor replay',
        storage_bound=c['storage'],baseline_new_fits=0,
        no_missing_as_zero=True,checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
    write(out/'reduction.json',result);write(out/'paired.json',cross)
    rows=[]
    for arm,v in arms.items():
        for point in v['metrics']:
            m=point['metrics']
            for kind,item in m.items():rows.append(dict(arm=arm,endpoint=point['endpoint'],kind=kind,
                **{k:item[k] for k in ('numerator','denominator','rate','token_micro','prompt_macro','strict_numerator','strict_denominator')},
                harmonic_RSPSNS=harmonic(m)))
    require(len(rows)<=1000,'COMPACT_COMPARISON_ROW_BOUND')
    buffer=io.StringIO(newline='')
    names=list(rows[0]) if rows else ['arm','endpoint','kind','numerator','denominator']
    w=csv.DictWriter(buffer,fieldnames=names);w.writeheader();w.writerows(rows)
    text_file(out/'comparison.csv',buffer.getvalue())
    lines=['# 간섭가격 group-L1 2k 사실 보고','',f'- 상태: {result["status"]}',f'- 실행 source: `{lock["source_commit"]}`',
        f'- Commit {result["actual"]["commits"]}/60, own join {result["actual"]["joins"]}/57, H append {result["actual"]["history_appends"]}/300.',
        '- PRICE/FLAT/REVERSE는 각각 cold W0/H0에서 자기 2000 occurrence의 W/H trajectory를 따른다. 공유되는 것은 readonly 입력·모델 자산뿐이다.',
        '- 별도 qualification/toy/small fit/full-builder-gradient 진단 0. 실제 B1 c0의 cached LOO 및 정상 proposal KKT/assertion을 trajectory 안에서 검산한다.',
        '- 계산가격·적용가격·정규화 anchor/local cap은 entry-price.json에 batch당1회, 후보는 authoritative events.jsonl에1회 저장한다. fit.json은 SHA/line count 참조만 저장한다.',
        '', '| Arm/endpoint | RS | PS | NS |', '|---|---:|---:|---:|']
    for arm,v in arms.items():
        for point in v['metrics']:
            if point['endpoint']=='W0' or point['endpoint'].endswith('ALL_SEEN'):
                m=point['metrics'];lines.append('| '+arm+'/'+point['endpoint']+' | '+' | '.join(f'{100*m[k]["rate"]:.3f} ({m[k]["numerator"]}/{m[k]["denominator"]})' for k in ('R','P','N'))+' |')
    lines.extend(['','## Coverage·기제·비용',''])
    for arm,v in arms.items():
        lines.append(f'- {arm}: {v["status"]}; commit {v["commits"]}/20, join {v["actual"]["joins"]}/19, H {v["actual"]["history_appends"]}/{history_expected(arm)}; '+
            '/'.join(str(v['counters'][k]) for k in ('builds','subject_forwards','subject_backwards','request_updates'))+' BUILD/subjectF/subjectB/request-update.')
        if v['first_error']:lines.append(f'  최초 기술 오류/증거: {v["first_error"].get("type",v["first_error"].get("status"))}: {v["first_error"].get("error",v["first_error"].get("reason"))}')
    lines.extend(['','- Terminal requested/realized norm·direction·cosine·error/share, zero-owner leakage, ideal/effective Q·capacity·rawSPD applicability는 reduction.json arm realization에 있다.',
        '- Subject loss와 실제 all-token loss gap은 별도로 관측했으며 동일payload가 두 경로의 hidden/loss/전체gradient 동일성을 뜻하지 않는다.',
        '- PRICE–REVERSE가 primary, PRICE–FLAT은 secondary다. Paired는 같은 실제 endpoint/case/token 분모에서만 계산했다. 미측정 endpoint는 0점/완료로 대체하지 않았다.',
        '- Raw/floored/computed/applied 가격, exact ties/reverse multiset, beta stage·own update·weighted spend·FP64 KKT/FP32 cap은 원 scalar stream으로 독립 재검산했다. M/P/K tensor replay는 수행하지 않았다.',
        '- Price proxy 또는 같은 budget/norm은 같은 semantic strength가 아니다. NS 개선과 R/P 획득 감소가 함께 있으면 tradeoff이며 단일 seed에서 보편적 우위를 주장하지 않는다.',
        f'- 부모 allocated GPU seconds(단일계상): {parents.get("allocated_GPU_seconds","NOT_AVAILABLE")}. Pending·jobsteps 중복 가산 없음.',
        '- Fit exclusive price/LOO/BUILD/subject/pullback/Adam/projection/scalar time과 inclusive fit/writer/batch time을 구분했다. Candidate I/O·실제 alltoken gap·observer/history 비용을 별도 보존한다.',
        '- Serializer의 전체3arm 상한과 atomic/error reserve를 source/config에 봉인했다. 매 batch fit 전에 fresh free/inode guard를 실시한다. 공유 filesystem 예약·향후 quota 여유를 보장한 것은 아니다.',
        '- prior ENOSPC 소비 주체는 NOT_IDENTIFIED다. 유효 commit prefix만 인정하며 empty/torn rollback/terminal은 NOT_VERIFIED다. 원자료 삭제·로그 누락·평가 축소·자동 retry는 없다.',
        '- 새 baseline fit 0; 조건 검산 없는 기존 baseline은 HISTORICAL_REFERENCE/NOT_AVAILABLE. noCP, exact resume NOT_AVAILABLE.',
        '- 원 source/raw/log KEEP. Source·compact report/countCSV/manifest만 Git; NO_BROADCAST_NOT_REQUIRED.'])
    report=out/'report-ko.md';text_file(report,'\n'.join(lines)+'\n')
    files=[member(p) for p in sorted(out.iterdir()) if p.is_file()]
    write(out/'inventory.json',dict(task=TASK,files=files,reader_files=reader.files,raw_local_KEEP=True))
    write(out/'terminal.json',dict(task=TASK,status=result['status'],scientific_coverage_complete=complete,
        source=lock['source_commit'],report=member(report),inventory=member(out/'inventory.json')))
    return result

def collect(attempt,out=None,accounting=True):
    attempt=Path(attempt).resolve();out=Path(out).resolve() if out else attempt/'collector';require(not out.exists(),'COLLECTOR_CREATE_ONCE')
    try:return _collect(attempt,out,accounting)
    except Exception as error:
        out.mkdir(parents=True,exist_ok=True);r=dict(task=TASK,status='CPU_REVIEW_TECHNICAL_BLOCKED',error_type=type(error).__name__,
            error=str(error),traceback=traceback.format_exc(),scientific_coverage_complete=False,original_KEEP=True,automatic_retry=False)
        write(out/'reducer-first-error.json',r)
        report=out/('failure-report-ko.md' if (out/'report-ko.md').exists() else 'report-ko.md')
        text_file(report,'# 간섭가격 group-L1 CPU 검산 기술 차단\n\n'+str(error)+'\n\n원 source/raw KEEP; 새 GPU/평가/자동 retry 0. W20 완료를 인증하지 않는다.\n')
        files=[member(p) for p in out.iterdir() if p.is_file() and p.name not in ('inventory.json','terminal.json')]
        inv=out/'failure-inventory.json';write(inv,dict(task=TASK,files=files,raw_local_KEEP=True))
        write(out/'terminal.json',dict(task=TASK,status=r['status'],report=member(report),inventory=member(inv),scientific_coverage_complete=False))
        return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--out',type=Path)
    a=p.parse_args();ConsoleBudget(Path(a.attempt)/'collector-console-bound-failure.json').install()
    r=collect(a.attempt,a.out);print(json.dumps(dict(status=r['status'],actual=r.get('actual'))))
    if r['status']=='CPU_REVIEW_TECHNICAL_BLOCKED':raise SystemExit(1)
