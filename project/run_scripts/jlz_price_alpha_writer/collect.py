"""Independent CPU scalar/row reducer; no model/Slurm writes or backfill."""
import argparse,copy,csv,hashlib,io,json,math,os,pwd,subprocess,traceback
from pathlib import Path
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    Reader,validate_rows as independent_rows,reduce_rows,compare_summary,paired,harmonic,active_flags)
from .common import *
from .common import ARMS,history_expected
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
    require(record['writer']==profile['writer']=='alphaedit' and record['lambda_alpha']==1.
        and record['projector_sha256']==profile['projector_sha256'],'ALPHA_PRICE_PROJECTOR_IDENTITY')
    require(record['B']==B and record['layers']==profile['eligible_layers'] and record['arm']==profile['arm']
        and record['batch']==batch and record['price_extra_model_calls']==record['price_extra_solves']==0
        and record['no_durable_matrices'],'STATIC_PRICE_SOURCE_SCOPE')
    anchors=matrix(record['anchors'],L,B,'STATIC_NATIVE_ANCHORS');caps=None if profile['cap_mode']=='none' else matrix(record['local_caps'],L,B,'STATIC_NATIVE_CAPS')
    require((record['local_caps'] is None)==(profile['cap_mode']=='none'),'NULL_UNCAPPED')
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
            if caps is not None:close(caps[l][r],.75*anchors[l][r],'NATIVE_LAYER_LOCAL_CAP')
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
        close(record['beta_base'][r],profile['beta_base'],'DIMENSIONLESS_BETA_BASE')
        close(record['beta_max'][r],max(profile['beta_base'],.75*max(expected)),'DIMENSIONLESS_BETA_CEILING')
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
            require(row['writer']=='alphaedit' and row['operator']=='(I+NH)q+NKminus(Kminus.Tq)=Nkr'
                and row['normalization']=='original_unprojected_key_norm','ALPHA_NOT_MEMIT_LOO')
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
    L=len(weights);pre=matrix(p['pre_norm'],L,B,'PRE_NORMS')
    target=matrix(p['fp64_projected_norm'],L,B,'FP64_NORMS');stored=matrix(p['post_norm'],L,B,'STORED_NORMS')
    require(p['coordinate']=='absolute_R_Euclidean' and not p['postcast_repair'] and not p['moment_reset']
        and p['lr']==.1 and p['eps']==1e-8,'NATIVE_ADAM_NO_RESCUE')
    require((p['capped_mask'] is None)==(caps is None) and (p['local_excess'] is None)==(caps is None),'UNCAPPED_NULL_FIELDS')
    for r in range(B):
        tau=p['tau'][r];require(tau>=0,'DUAL_NONNEGATIVE');close(p['beta'][r],beta[r],'SAME_BETA')
        ends=p['endpoint_corrections'][r];require(ends['owner']==r,'ENDPOINT_OWNER')
        selected={e['block']:e for e in ends['selected_endpoints']}
        spend=postspend=0.
        for l in range(L):
            n,w,t,post=pre[l][r],weights[l][r],target[l][r],stored[l][r]
            cap=None if caps is None else caps[l][r];raw=max(n-tau*w,0.)
            if cap is not None:raw=min(cap,raw)
            require(n>=0 and t>=0 and post>=0 and abs(t-raw)<=1e-10*max(1,n,cap or 0),'LENGTH_KKT')
            if l in selected:
                e=selected[l];require(e['kind'] in ('ZERO','CAP'),'TAG_KIND')
                knot=n/w if e['kind']=='ZERO' else (n-cap)/w
                require(e['value']==tau==knot,'TAG_EXACT_EQUALITY')
                require(t==(0. if e['kind']=='ZERO' else cap),'SELECTED_ENDPOINT_EXACT')
            close(ends['raw_expression'][l],raw,'RAW_ENDPOINT_EXPR')
            close(ends['correction'][l],t-raw,'ENDPOINT_CORRECTION')
            require(p['zero_mask'][l][r]==(t==0)==(post==0),'EXACT_STORED_ZERO_NO_PRUNING')
            capped=cap is not None and t==cap and t!=0
            require(p['free_mask'][l][r]==(t!=0 and not capped),'INTERIOR_SUPPORT')
            if cap is not None:
                require(post-cap<=1e-6 and p['capped_mask'][l][r]==capped,'LOCAL_CAP')
                close(p['local_excess'][l][r],max(post-cap,0.),'LOCAL_EXCESS')
            spend+=w*t;postspend+=w*post
        require(spend-beta[r]<=1e-10*max(1,beta[r]) and abs(tau*(spend-beta[r]))<=1e-10*max(1,tau*beta[r]),'PRIMAL_COMPLEMENTARITY')
        require(postspend-beta[r]<=1e-6*max(1,beta[r]),'STORED_SHARED')
        close(p['weighted_spend'][r],postspend,'STORED_SPEND',atol=1e-10)
        close(p['shared_excess'][r],max(postspend-beta[r],0.),'STORED_EXCESS',atol=1e-10)
        if not p['shared_active'][r]:require(tau==0,'MINIMUM_INACTIVE_DUAL')


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
    oldactive=[False]*B;oldt=[0]*B;olde=[0]*B;oldbeta=[profile['beta_base']]*B;backwards=0
    for k,row in enumerate(events):
        finite(row);require(row['candidate']==k and row['ordinal']==k+1 and len(row['F'])==B,'CANDIDATE_ORDER_CARDINALITY')
        require(row['entry_price_sha256']==price_hash,'EACH_CANDIDATE_SAME_FROZEN_PRICE_HASH')
        mask=[oldactive[r] or row['F'][r]>=.05 for r in range(B)]
        ctrl=row['controller'];require(row['active_mask']==mask and ctrl['active']==mask,'IRREVERSIBLE_NATIVE_LOSS_ACTIVE')
        require(ctrl['update_counts']==oldt and ctrl['expansion']==olde,'OWN_REQUEST_COUNTER_JOIN')
        require(ctrl['beta']==oldbeta,'BETA_CURRENT_CANDIDATE_JOIN')
        for r in range(B):
            base=price['beta_base'][r];maximum=price['beta_max'][r]
            expected=maximum if olde[r]==4 else base*math.exp(olde[r]/4*math.log(maximum/base))
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
        expected_e=[e+int(active and t>=12 and F>=.05 and e<4 and maximum>profile['beta_base'])
            for e,active,t,F,maximum in zip(olde,mask,oldt,row['F'],price['beta_max'])]
        require(after['update_counts']==expected_t and after['expansion']==expected_e
            and projection['adam_updates']==expected_t,'OWN_GRACE12_BEFORE13_EXPANSION')
        expectedbeta=[price['beta_max'][r] if expected_e[r]==4 else price['beta_base'][r]*math.exp(expected_e[r]/4*math.log(price['beta_max'][r]/price['beta_base'][r])) for r in range(B)]
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
    reuse=(folder/'cap-reuse.json').exists()
    if reuse:
        from .w0 import chunks as source_chunks
        require(name=='W0','W0_REUSE_ENDPOINT_ONLY');paths=source_chunks(folder,state_value)
        reader.json(folder/'cap-reuse.json')
    else:paths=sorted(folder.glob('chunk-*.json'))
    for path in paths:
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
    if reuse:require(saved['seconds']==0 and saved['new_forwards']==0 and saved['reference_only'],'W0_REUSE_NEW_COST_ZERO')
    return dict(rows=rows,metrics=result,seconds=saved['seconds'],
        original_evaluation_seconds=saved.get('original_evaluation_seconds') if reuse else None,
        reference_only=reuse)

def _arm_review(reader,attempt,c,lock,arm,records,identities,progress):
    out=attempt/arm;ids=[r['case_id'] for r in records];profile=c['arm_profiles'][arm];layers=list(map(str,profile['eligible_layers']))
    cold={key:{l:c['cold_W0_H0'][key][l] for l in layers} for key in ('W','H')}
    commits=[];metrics=[];pairs=[];cost=[];realizations=[];prefix={};atwrite=[]
    counts=dict(builds=0,subject_forwards=0,subject_backwards=0,request_updates=0)
    w0=endpoint(reader,out/'W0',identities,ids,'W0',cold,records)
    if w0:metrics.append(dict(endpoint='W0',requests=2000,metrics=w0['metrics']))
    W0_cost=None if w0 is None else dict(new_evaluation_seconds=w0['seconds'],reference_only=w0['reference_only'],
        original_evaluation_seconds=w0['original_evaluation_seconds'],
        old_observation_not_charged_to_repair=True if w0['reference_only'] else False)
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
        require(writer['writer']=='alphaedit' and writer['lambda_alpha']==1.
            and writer['projector_sha256']==c['projector']['sha256'],'ALPHA_COMMIT_PROJECTOR_IDENTITY')
        require(writer['accepted_weight_copy_exact'] and writer['terminal_last_evaluated_not_best'] and writer['no_resolve']
            and writer['no_double_add'] and writer['candidate']==fit['terminal_candidate']==commit['accepted_candidate'],'LAST_EVALUATED_EXACT_COMMIT')
        require(writer['weight_hashes']==commit['after']['W'] and writer['history']==commit['history'],'PAYLOAD_AND_H_IDENTITIES')
        capture=reader.json(folder/'entry-capture.json');require(capture['fresh_capture'] and capture['H_entry']==entry['state']['H']
            and capture['anchor_layer']==8 and capture['eligible_layers']==profile['eligible_layers'],'FRESH_TEACHER_ANCHOR_FACTOR')
        realization=reader.json(verify(writer['realization']));finite(realization)
        for layer in realization['layers'].values():
            require(layer['C0_scale']==1. and layer['solve']['writer']=='alphaedit'
                and layer['solve']['projector_sha256']==c['projector']['sha256'],'ALPHA_DIAGNOSTIC_IDENTITY')
            close(layer['ideal_Q'],layer['Q_C0']+layer['Q_H'],'DIRECT_DIAGNOSTIC_COST_SPLIT')
            close(layer['effective_Q'],layer['effective_Q_C0']+layer['effective_Q_H'],'DIRECT_EFFECTIVE_COST_SPLIT')
        require(realization['candidate']==commit['accepted_candidate'] and realization['B']==100 and set(realization['layers'])==set(layers)
            and realization['terminal_extra_planner_forward']==realization['terminal_extra_planner_backward']==0,'TERMINAL_REALIZATION_BINDING')
        mechanism=mechanism_summary(events,price,realization,profile)
        realizations.append(dict(batch=number,realization=realization,mechanism=mechanism))
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
        expected=dict(commits=20,requests=2000,joins=19,history_appends=history_expected(arm.split('_')[-1])),
        actual=dict(joins=max(0,len(commits)-1),history_appends=sum(r['history_appends'] for r in commits)),
        counters=counts,metrics=metrics,paired=pairs,realization=realizations,cost=cost,W0_observation_cost=W0_cost,terminal=terminal,first_error=firsterror,
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

def mechanism_summary(events,price,realization,profile):
    L=len(profile['eligible_layers']);B=price['B'];last=events[-1]
    def stats(values):
        values=[v for v in values if v is not None]
        return dict(valid_count=len(values),mean=None if not values else sum(values)/len(values),
            minimum=None if not values else min(values),maximum=None if not values else max(values))
    layers={}
    for i,l in enumerate(profile['eligible_layers']):
        key=str(l);values=realization['layers'][key];roles={}
        for role in ('canonical','rewrite','KL'):
            owners=values[role];sums={k:sum(o['sufficient_statistics'][k] for o in owners)
                for k in ('action_squared_norm','target_squared_norm','action_target_dot','residual_squared_norm')}
            xx,yy,xy,ee=[sums[k] for k in ('action_squared_norm','target_squared_norm','action_target_dot','residual_squared_norm')]
            roles[role]=dict(unit='pooled actual context-row sufficient statistics; not mean of ratios',
                owners=len(owners),context_rows=sum(o['rows'] for o in owners),sufficient_statistics=sums,
                normratio=None if yy==0 else math.sqrt(xx/yy),directionalratio=None if yy==0 else xy/yy,
                cosine=None if xx==0 or yy==0 else xy/math.sqrt(xx*yy),relative_error=None if yy==0 else math.sqrt(ee/yy),
                owner_mean_ratios={k:stats([o[k] for o in owners]) for k in ('normratio','directionalratio','cosine','relative_error')},
                valid_context_counts={k:sum(o['valid_counts'][k] for o in owners) for k in ('normratio','directionalratio','cosine','relative_error')},
                target_classes={k:sum(o['target_class']==k for o in owners) for k in ('exact_zero','positive_tiny_diagnostic','positive_above_diagnostic_threshold')})
        minimum=[min(price['effective_pi'][j][r] for j in range(L)) for r in range(B)]
        layers[key]=dict(raw_kappa=stats(price['raw_kappa'][i]),price=stats(price['effective_pi'][i]),
            minimum_price_owners=sum(price['effective_pi'][i][r]==minimum[r] for r in range(B)),
            floor_count=sum(price['floor_mask'][i]),roles=roles,mean=values['mean'],
            zero_to_positive=sum(sum(e.get('support_transition',{}).get('zero_to_positive',[[False]*B for _ in range(L)])[i]) for e in events),
            positive_to_zero=sum(sum(e.get('support_transition',{}).get('positive_to_zero',[[False]*B for _ in range(L)])[i]) for e in events),
            ideal_Q=values['ideal_Q'],effective_Q=values['effective_Q'],effective_update_norm=values['effective_update_norm'])
    return dict(cap_mode=profile['cap_mode'],base=stats(price['beta_base']),maximum=stats(price['beta_max']),
        current_beta=stats(last['controller']['beta']),stage_counts={str(s):last['controller']['expansion'].count(s) for s in range(5)},
        own_updates=stats(last['controller']['update_counts']),weighted_spend=stats(last['telemetry']['weighted_spend']),
        spend_slack=stats(last['telemetry']['spend_slack']),exact_support=stats(last['telemetry']['exact_support']),
        native_F=stats(last['F']),rewrite_NLL=stats([sum(row)/len(row) for row in last['nll']]),weighted_KL=stats([.0625*x for x in last['KL']]),
        tiny_report_only=True,normalization_never_prunes=True,layers=layers)

def requested_cells(c):
    """Subset registration is explicit; absent cells are not failed runs."""
    selected=c.get('selected_cells',CELLS);retained=c.get('retained_cells',{})
    require(isinstance(selected,(list,tuple)) and len(selected)==len(set(selected))
        and all(cell in CELLS for cell in selected),'SELECTED_CELL_SCOPE')
    require(isinstance(retained,dict) and all(cell in CELLS for cell in retained)
        and not set(selected).intersection(retained),'RETAINED_CELL_SCOPE')
    coverage=list(selected)+list(retained)
    require(bool(coverage),'EMPTY_CELL_COVERAGE')
    return list(selected),retained,coverage

def retained_binding(reader,cell,record,records,c):
    """Read a kept run under its original config/lock, never the new config."""
    require(isinstance(record,dict),'RETAINED_BINDING_METADATA')
    attempt=Path(record['attempt']);config_path=Path(record['config_path'])
    lock_path=Path(record.get('lock_path',attempt/'execution.lock.json'))
    require(attempt.is_absolute() and config_path==attempt/'config.json'
        and lock_path==attempt/'execution.lock.json','RETAINED_EXACT_ATTEMPT_PATHS')
    original=reader.json(config_path);lock=reader.json(lock_path)
    config_sha=reader.files[str(config_path)]['sha256'];lock_sha=reader.files[str(lock_path)]['sha256']
    require(original['task_id']==lock['task_id']==TASK and original['instruction_id']==lock['instruction_id']==NONCE
        and config_sha==lock['config_sha256'],'RETAINED_ORIGINAL_SOURCE_CONFIG')
    require(config_sha==record['config_sha256'] and lock_sha==record['lock_sha256']
        and lock['source_commit']==record['execution_source'],'RETAINED_IMMUTABLE_BINDING')
    job=str(record['job_id']);require(job.isdigit() and int(job)>0,'RETAINED_EXACT_JOB_ID')
    require(original['stream']==c['stream'] and digest([r['case_id'] for r in records])==ORDERED_SHA,
        'RETAINED_SAME_ORDERED_COHORT')
    cc=cell_config(original,cell);identities=reader.json(verify(cc['observer_identity']))['rows']
    require(cc['packs']==c['models'][cell.split('_',1)[0]]['packs'],'RETAINED_SAME_NATIVE_PACKS')
    binding=dict(origin='RETAINED_UNCHANGED_EXECUTION',attempt=str(attempt),config_path=str(config_path),
        config_sha256=config_sha,lock_path=str(lock_path),lock_sha256=lock_sha,
        execution_source=lock['source_commit'],job_id=job,no_job_mutation=True,no_duplicate_run=True)
    return attempt,cc,lock,identities,binding

def accounting_snapshot(attempt):
    receipt=attempt/'submission.json'
    if not receipt.exists():return dict(status='NOT_AVAILABLE',reason='NO_EXACT_LOCAL_SUBMISSION')
    data=json.loads(receipt.read_text());jobs=dict(data['jobs']);retained=data.get('retained_jobs',{})
    require(isinstance(retained,dict) and not set(jobs).intersection(retained),'ACCOUNTING_RETAINED_ROLES')
    for role,item in retained.items():
        jobs[role]=item['job_id'] if isinstance(item,dict) else item
    ids=[str(v) for v in jobs.values()]
    require(len(ids)==len(set(ids)),'ACCOUNTING_NO_DUPLICATE_PARENT')
    require(all(x.isdigit() for x in ids),'EXACT_JOB_IDS')
    response=subprocess.run(['sacct','-n','-P','-j',','.join(ids),
        '--format=JobIDRaw,User,JobName%120,State,ElapsedRaw,AllocTRES,TotalCPU,ExitCode'],check=True,capture_output=True,text=True,timeout=30)
    roles={str(job):role for role,job in jobs.items()}
    parents=[]
    for line in response.stdout.splitlines():
        v=line.split('|')
        if len(v)<8 or v[0] not in ids:continue
        require(v[1]==pwd.getpwuid(os.getuid()).pw_name and v[2]==TASK+'-'+roles[v[0]],'EXACT_PARENT_OWNER_NAME')
        tres=dict(x.split('=',1) for x in v[5].split(',') if '=' in x);gpu=int(tres.get('gres/gpu',0));elapsed=int(v[4] or 0)
        require(gpu==0 if roles[v[0]]=='collector' else gpu in (0,1),'EXACT_PARENT_ALLOCATED_GPU')
        parents.append(dict(job=int(v[0]),role=roles[v[0]],origin='RETAINED_UNCHANGED_EXECUTION' if roles[v[0]] in retained else 'NEW_REGISTRATION',
            state=v[3],elapsed_seconds=elapsed,GPUs=gpu,allocated_GPU_seconds=gpu*elapsed,TotalCPU=v[6],exit=v[7]))
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
    arms={};prefix={};selected,retained,coverage=requested_cells(c);source_by_arm={}
    for cell in selected:
        cc=cell_config(c,cell);identities=reader.json(verify(cc['observer_identity']))['rows']
        arms[cell],prefix[cell]=arm_review(reader,attempt,cc,lock,cell,records,identities)
        source_by_arm[cell]=dict(origin='NEW_REGISTRATION',attempt=str(attempt),execution_source=lock['source_commit'],config_sha256=lock['config_sha256'])
    for cell,record in retained.items():
        original_attempt,cc,original_lock,identities,binding=retained_binding(reader,cell,record,records,c)
        arms[cell],prefix[cell]=arm_review(reader,original_attempt,cc,original_lock,cell,records,identities)
        source_by_arm[cell]=binding
    cross=[]
    for model in MODELS:
        ref=model+'_AE_CAP075'
        for cell in (model+'_AE_CAP100',model+'_AE_FREE100'):
            for k in MILESTONES:
                if k in prefix.get(ref,{}) and k in prefix.get(cell,{}):
                    cross.append(dict(reference=ref,arm=cell,endpoint=f'W{k}',cohort='ALL_SEEN',
                        paired=paired(prefix[ref][k],prefix[cell][k])))
                    fixed=set(r['case_id'] for r in records[:500])
                    cross.append(dict(reference=ref,arm=cell,endpoint=f'W{k}',cohort='SAME_FIRST500',
                        paired=paired([r for r in prefix[ref][k] if r['case_id'] in fixed],
                                      [r for r in prefix[cell][k] if r['case_id'] in fixed])))
    complete=all(v['status']=='W20_COMPLETE' for v in arms.values());parents=accounting_snapshot(attempt) if accounting else dict(status='NOT_REQUESTED')
    if tuple(coverage)==CELLS:
        from .comparison import existing_memit
        memit_comparison=existing_memit(c,attempt,prefix,reader)
    else:
        # The historical six-cell helper binds an older fixed MEMIT attempt.
        # It cannot certify correspondence with replacement source/rows.
        memit_comparison=[dict(cell=cell,reference=cell.replace('_AE_','_'),endpoint=f'W{k}',
            status='PENDING_COMPARISON',reason='New subset/replacement MEMIT source requires exact independent endpoint binding; historical fixed-source comparison not invoked',
            no_new_evaluation=True,no_result_monitoring=True) for cell in coverage for k in MILESTONES]
    completion_status='SIX_CELLS_W20_COMPLETE' if tuple(coverage)==CELLS else 'REQUESTED_CELLS_W20_COMPLETE'
    result=dict(task=TASK,status=completion_status if complete else 'PARTIAL_OR_TECHNICAL_BLOCKED',source=lock['source_commit'],config_sha256=lock['config_sha256'],
        execution_arms=coverage,new_execution_arms=selected,retained_execution_arms=list(retained),source_by_arm=source_by_arm,
        not_requested_cells=[cell for cell in CELLS if cell not in coverage],excluded_arms=['FREE075','FLAT','REVERSE'],
        arms=arms,standalone_qualification=False,toy_runs=0,cross_arm_paired=cross,memit_comparison=memit_comparison,actual=dict(commits=sum(v['commits'] for v in arms.values()),
        history_appends=sum(v['actual']['history_appends'] for v in arms.values()),joins=sum(v['actual']['joins'] for v in arms.values())),
        expected=dict(commits=20*len(coverage),joins=19*len(coverage),history_appends=100*len(coverage)),accounting=parents,
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
    require(len(rows)<=2000,'COMPACT_COMPARISON_ROW_BOUND')
    buffer=io.StringIO(newline='')
    names=list(rows[0]) if rows else ['arm','endpoint','kind','numerator','denominator']
    w=csv.DictWriter(buffer,fieldnames=names);w.writeheader();w.writerows(rows)
    text_file(out/'comparison.csv',buffer.getvalue())
    lines=['# PRICE AlphaEdit writer — 요청 cell 2k 사실 보고','',f'- 상태: {result["status"]}',f'- 신규 실행 source: `{lock["source_commit"]}`',
        f'- Commit {result["actual"]["commits"]}/{result["expected"]["commits"]}, own join {result["actual"]["joins"]}/{result["expected"]["joins"]}, H append {result["actual"]["history_appends"]}/{result["expected"]["history_appends"]}.',
        '- 실제 coverage arm: '+', '.join(coverage)+'. 신규 등록: '+', '.join(selected)+'; 원 실행 보존: '+(', '.join(retained) or '없음')+'. 보존 arm은 원 config/lock/source로 독립 검산하며 신규 source 실행으로 간주하지 않는다. 제외 cell은 NOT_REQUESTED다.',
        '- 별도 qualification/toy/small fit/full-builder-gradient 진단 0. 실제 B1 c0의 cached LOO 및 정상 proposal KKT/assertion을 trajectory 안에서 검산한다.',
        '- 계산가격·적용가격·정규화 anchor/local cap은 entry-price.json에 batch당1회, 후보는 authoritative events.jsonl에1회 저장한다. fit.json은 SHA/line count 참조만 저장한다.',
        '', '| Model/arm/endpoint | RS | PS | NS | Harmonic |', '|---|---:|---:|---:|---:|']
    for arm,v in arms.items():
        for point in v['metrics']:
            if point['endpoint']=='W0' or point['endpoint'].endswith('ALL_SEEN'):
                m=point['metrics'];lines.append('| '+arm+'/'+point['endpoint']+' | '+' | '.join(f'{100*m[k]["rate"]:.3f} ({m[k]["numerator"]}/{m[k]["denominator"]})' for k in ('R','P','N'))+f' | {100*harmonic(m):.3f} |')
    lines.extend(['','## 기존 MEMIT 동일 cell 관측','', '| Alpha cell/endpoint | 비교 상태 | MEMIT RS | MEMIT PS | MEMIT NS | MEMIT Harmonic |', '|---|---|---:|---:|---:|---:|'])
    for item in memit_comparison:
        m=item.get('metrics')
        values=' | '.join(f'{100*m[k]["rate"]:.3f}' for k in ('R','P','N'))+f' | {100*harmonic(m):.3f}' if m else 'NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE'
        lines.append('| '+item['cell']+'/'+item['endpoint']+' | '+item['status']+' | '+values+' |')
    lines.extend(['','## Coverage·기제·비용',''])
    for arm,v in arms.items():
        lines.append(f'- {arm}: {v["status"]}; commit {v["commits"]}/20, join {v["actual"]["joins"]}/19, H {v["actual"]["history_appends"]}/{history_expected(arm.split('_')[-1])}; '+
            '/'.join(str(v['counters'][k]) for k in ('builds','subject_forwards','subject_backwards','request_updates'))+' BUILD/subjectF/subjectB/request-update.')
        if v['first_error']:lines.append(f'  최초 기술 오류/증거: {v["first_error"].get("type",v["first_error"].get("status"))}: {v["first_error"].get("error",v["first_error"].get("reason"))}')
    lines.extend(['','- Terminal requested/realized norm·direction·cosine·error/share, zero-owner leakage, ideal/effective Q_C0·Q_H(각각 직접계산, C0 scale1), projected-key ratio/Mrr·LU/operator residual은 reduction.json arm realization에 있다. 비대칭 A0를 energy로 사용하지 않았다.',
        '- Subject loss와 실제 all-token loss gap은 별도로 관측했으며 동일payload가 두 경로의 hidden/loss/전체gradient 동일성을 뜻하지 않는다.',
        '- 같은 model의 CAP075–CAP100/FREE100만 같은 endpoint/case/token에 paired 계산한다. 서로 다른 model token을 paired로 혼합하지 않는다. 미측정 endpoint는 NOT_AVAILABLE이다.',
        '- W0 exact reuse는 원 chunk/summary/runtime/config/lock/SHA와 cold 상태를 결속한 원자료 참조다. 재사용 시간/forward는 0이며 원 관측 시간은 provenance로 분리한다.',
        '- Raw/floored/computed/applied 가격, exact ties/reverse multiset, beta stage·own update·weighted spend·FP64 KKT/FP32 cap은 원 scalar stream으로 독립 재검산했다. M/P/K tensor replay는 수행하지 않았다.',
        '- 같은 model×arm×endpoint/token/runtime 관측만 MEMIT 비교에 사용했다. Writer/가격/이후 own trajectory는 다르며 과학적 해석은 포함하지 않는다.',
        f'- 부모 allocated GPU seconds(단일계상): {parents.get("allocated_GPU_seconds","NOT_AVAILABLE")}. Pending·jobsteps 중복 가산 없음.',
        '- Fit exclusive price/LOO/BUILD/subject/pullback/Adam/projection/scalar time과 inclusive fit/writer/batch time을 구분했다. Candidate I/O·실제 alltoken gap·observer/history 비용을 별도 보존한다.',
        '- Serializer의 source/config 상한과 atomic/error reserve를 봉인했다. 신규 batch fit 전에 fresh free/inode guard를 실시한다. 보존 arm의 기존 실행 bytes는 바꾸지 않았다. 공유 filesystem 예약·향후 quota 여유를 보장한 것은 아니다.',
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
