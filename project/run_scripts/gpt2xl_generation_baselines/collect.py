"""Independent scalar/raw CPU reducer; no model/GPU/native apply or W&B upload.

The qualification receipt verifier imports torch's Python package, but this
collector does not initialize CUDA, instantiate a model or run tensor inference.

Source-preserving partial reports are written before collector terminal. Stored
generation texts/tokens remain local and are never included in compact reports.
"""
import argparse
import hashlib
import json
import math
import os
import pwd
import re
import subprocess
from pathlib import Path

from .common import TASK, NONCE, ARMS, MILESTONES, digest, member, require, sha, write
from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    Reader, active_flags, compare_summary, reduce_rows)
from project.run_scripts.gpt2xl_native_baselines.collect import _metric_rows, _paired_rows, _csv, _atomic_text
from project.run_scripts.gpt2xl_cake_blue.collect import endpoint, zero_history_hash
from project.run_scripts.gpt2xl_prune_rect.collect import transform_guard, rect_masks

COUNTERS=('native_z','write_keys','history_keys','solves','history_appends')
LAYERS=(13,14,15,16,17)
ARM_LAYERS={a:LAYERS for a in ARMS};ARM_LAYERS['ALPHAEDIT_BLUE']=(13,17)
HISTORY_LAYERS=dict(BASE_MEMIT=(),BASE_ALPHAEDIT=LAYERS,CAKE=LAYERS,ALPHAEDIT_BLUE=(13,17),PRUNE=(),RECT=())
EXPECTED={a:dict(native_z=200 if a=='ALPHAEDIT_BLUE' else 100,write_keys=len(ARM_LAYERS[a]),
    history_keys=len(HISTORY_LAYERS[a]),solves=len(ARM_LAYERS[a]),history_appends=len(HISTORY_LAYERS[a])) for a in ARMS}
REASONS=('missing_generation_prompts','missing_reference','zero_generated_vector','zero_reference_vector',
    'nonfinite_score','length_cap_no_continuation','asset_not_available','tokenizer_not_available')


def reduce_generation(rows):
    """Independent per-occurrence numerator/count reduction, not chunk means."""
    values={k:[] for k in ('ngram_entropy','reference_score')}
    reasons={r:0 for r in REASONS};prompts=tokens=long_prompts=0
    for row in rows:
        item=row['metrics'];actual_reasons=item['reasons']
        require(type(actual_reasons) is list and len(set(actual_reasons))==len(actual_reasons)
            and set(actual_reasons)<=set(REASONS),'COLLECT_GENERATION_REASONS')
        for r in actual_reasons:reasons[r]+=1
        for metric,valid in (('ngram_entropy','fluency_valid'),('reference_score','consistency_valid')):
            require(type(item[valid]) is bool,'COLLECT_GENERATION_VALIDITY')
            value=item[metric]
            if item[valid]:
                require(type(value) in (int,float) and math.isfinite(value) and value>=0,
                    'COLLECT_GENERATION_FINITE')
                if metric=='reference_score':require(value<=1+1e-12,'COLLECT_COSINE_RANGE')
                values[metric].append(value)
            else:require(value is None,'COLLECT_MISSING_NOT_ZERO')
        for key in ('generation_prompt_count','generated_token_count','length_cap_no_continuation_count'):
            require(type(item[key]) is int and item[key]>=0,'COLLECT_GENERATION_INTEGER')
        require(item['length_cap_no_continuation_count']<=item['generation_prompt_count'],'COLLECT_LONG_PROMPT_COUNT')
        prompts+=item['generation_prompt_count'];tokens+=item['generated_token_count']
        long_prompts+=item['length_cap_no_continuation_count']
    result=dict(planned_count=len(rows),fluency_count=len(values['ngram_entropy']),
        consistency_count=len(values['reference_score']),fluency_sum=math.fsum(values['ngram_entropy']),
        consistency_sum=math.fsum(values['reference_score']),missing_reason_counts=reasons,
        generation_prompt_count=prompts,generated_token_count=tokens,
        length_cap_no_continuation_prompt_count=long_prompts,
        reason_count_unit='request_occurrences_nonexclusive',fluency_unit='bits',consistency_unit='cosine_0_to_1')
    for metric,kind in (('ngram_entropy','fluency'),('reference_score','consistency')):
        if result[kind+'_count']:result[metric]=result[kind+'_sum']/result[kind+'_count']
    return result


def qualified_generation(c):
    generation=dict(c['generation'])
    if 'qualification_plan_sha256' in generation:
        from project.run_scripts.experiment_generation_eval.kv_qualification import verify_actual_receipt
        from project.run_scripts.experiment_generation_eval.compatibility import member as generation_member
        actual_member=generation.get('qualification_receipt_member') or generation_member(Path(generation['qualification_receipt']))
        actual=verify_actual_receipt(actual_member,expected_plan_sha256=generation['qualification_plan_sha256'],
            expected_model_identity=generation['model_identity'])
        require(actual['qualification_pass'] is True,'COLLECT_ACTUAL_QUALIFICATION_REQUIRED')
        generation.update(generation_route=actual['selected_route'],generation_microbatch=actual['fixed_microbatch'],
            qualification_receipt_member=actual_member)
    return generation


def runtime_identity(c):
    generation=qualified_generation(c)
    value=dict(schema=generation['schema'],profile=generation['profile'],eval_seed=generation['eval_seed'],
        model_identity=generation['model_identity'],generation_source_sha=generation['generation_source_sha'],
        reference_assets_sha256=generation['reference_assets_sha256'],route=generation.get('generation_route','UNPADDED_FULL_PREFIX_NO_CACHE'))
    if 'qualification_receipt_member' in generation:
        value.update(generation_microbatch=generation['generation_microbatch'],
            qualification_receipt_sha256=generation['qualification_receipt_member']['sha256'])
    return value


def generation_endpoint(reader,receipt,c,records,endpoint_name,cohort,physical_state,ordinal_start=1):
    """Bind stored metric rows to original prompts/tokens/state/occurrences."""
    saved=reader.bound(receipt['rows']);identity=saved['identity']
    expected_occurrences=list(range(ordinal_start,ordinal_start+len(records)))
    state_identity=dict(W=physical_state['W'],H={})
    require(identity['runtime']==digest(runtime_identity(c)) and identity['endpoint']==endpoint_name
        and identity['cohort']==cohort and identity['state_sha256']==digest(state_identity)
        and identity['ordered_occurrences']==expected_occurrences
        and saved['identity_sha256']==receipt['identity_sha256']==digest(identity)
        and receipt['identity']==identity and saved['RNG_restored'] is receipt['RNG_restored'] is True
        and saved['observer_no_mutation'] is receipt['observer_no_mutation'] is True,
        'COLLECT_GENERATION_ENDPOINT_IDENTITY_STATE_RNG')
    rows=saved['rows']
    require(len(rows)==len(records) and [r['occurrence'] for r in rows]==expected_occurrences
        and identity['observation_identities']==[r['identity_sha256'] for r in rows],
        'COLLECT_GENERATION_OCCURRENCE_ORDER')
    compatibility={};qualified=qualified_generation(c)
    if 'qualification_receipt_member' in qualified:
        actual_member=qualified['qualification_receipt_member']
        require(saved['qualification_receipt_member']==receipt['qualification_receipt_member']==actual_member
            and identity['qualification_receipt_sha256']==actual_member['sha256'],
            'COLLECT_ENDPOINT_ACTUAL_QUALIFICATION_SHA')
        reader.bound(actual_member)
    if 'compatibility_member' in saved:
        from project.run_scripts.experiment_generation_eval.compatibility import load_compatibility
        compat=load_compatibility(saved['compatibility_member'],expected_runtime=identity['runtime'],
            expected_qualification=identity['qualification_receipt_sha256'])
        require(saved['compatibility_member']==receipt['compatibility_member']
            and identity['compatibility_sha256']==compat['identity_sha256'],'COLLECT_COMPATIBILITY_IDENTITY')
        reader.bound(saved['compatibility_member'])
        compatibility={r['occurrence']:r for r in compat['identity']['original_entries']}
    if 'provenance_sha256' in identity:
        require(identity['provenance_sha256']==digest([r['provenance'] for r in rows]),'COLLECT_PROVENANCE_IDENTITY')
    for row,record,ordinal in zip(rows,records,expected_occurrences):
        path=Path(row['observation_path']);require(path.is_file() and not path.is_symlink(),'COLLECT_RAW_SAFE_PATH')
        raw=reader.json(path);rewrite=record['requested_rewrite'];prompts=record.get('generation_prompts',[])
        record_identity=dict(ordered_occurrence=ordinal,case_id=record['case_id'],generation_prompts=prompts,
            relation_id=rewrite.get('relation_id'),target_new_id=rewrite['target_new'].get('id'))
        raw_runtime=raw['identity']['runtime'];route=qualified.get('generation_route','UNPADDED_FULL_PREFIX_NO_CACHE')
        if raw_runtime!=identity['runtime']:
            entry=compatibility.get(ordinal)
            require(entry is not None and entry['original_raw_member']['path']==str(path)
                and entry['original_identity_sha256']==row['identity_sha256']
                and entry['original_payload_sha256']==row['payload_sha256']
                and entry['original_runtime_sha256']==raw_runtime,'COLLECT_OLD_PROVENANCE_REQUIRED')
            reader.bound(entry['original_raw_member']);route=entry['original_route']
        if 'provenance' in row:
            provenance=row['provenance'];reader.bound(provenance['raw_member'])
            require(provenance['raw_member']['path']==str(path) and provenance['runtime_sha256']==raw_runtime
                and provenance['route']==route,'COLLECT_RAW_PROVENANCE')
            if raw_runtime!=identity['runtime']:
                require(provenance['origin']=='COMPATIBLE_ORIGINAL_W0'
                    and provenance['generation_source_sha']==entry['original_generation_source_sha'],
                    'COLLECT_OLD_SOURCE_PROVENANCE')
        require(raw['identity']==dict(runtime=raw_runtime,state_identity=state_identity,record_identity=record_identity)
            and raw['identity_sha256']==row['identity_sha256']==digest(raw['identity'])
            and raw['payload_sha256']==row['payload_sha256']==digest({k:v for k,v in raw.items() if k!='payload_sha256'})
            and raw['occurrence']==row['occurrence']==ordinal and raw['case_id']==row['case_id']==record['case_id']
            and raw['metrics']==row['metrics'] and raw['raw_local_only'] is True
            and raw['checkpoint_saved'] is False,'COLLECT_GENERATION_RAW_IDENTITY')
        observations=raw['observations'];require(len(observations)==len(prompts),'COLLECT_GENERATION_ALL_PROMPTS')
        for index,(observation,prompt) in enumerate(zip(observations,prompts)):
            sampling=observation['sampling'];inp=observation['input_token_ids'];continuation=observation['continuation_token_ids']
            require(observation['prompt']==prompt and observation['occurrence']==ordinal
                and observation['prompt_index']==index and observation['profile']==c['generation']['profile']
                and sampling==dict(top_k=5,temperature=1,top_p=1,max_total_tokens=100)
                and observation['RNG_restored'] is True and observation['route']==route
                and observation['full_token_ids']==inp+continuation and len(inp)>0
                and observation['input_token_count']==len(inp)
                and observation['continuation_token_count']==len(continuation),
                'COLLECT_GENERATION_PROFILE_TOKEN_IDENTITY')
            expected_seed=int(digest(dict(model_identity=c['generation']['model_identity'],
                ordered_occurrence=ordinal,prompt_index=index,eval_seed=c['generation']['eval_seed']))[:16],16)%(2**63-1)
            require(observation['seed']==expected_seed,'COLLECT_GENERATION_ARM_JOB_INDEPENDENT_RNG_KEY')
            require(all(type(t) is int and t>=0 for t in inp+continuation),'COLLECT_TOKEN_SCHEMA')
            if len(inp)>=100:
                require(continuation==[] and observation['stop_reason']=='length_cap_no_continuation'
                    and observation['model_forwards']==0,'COLLECT_LONG_PROMPT_NO_FORWARD')
            else:
                require(len(inp)+len(continuation)<=100 and observation['model_forwards']==len(continuation),
                    'COLLECT_TOTAL_LENGTH_FORWARD_COUNT')
                if observation['stop_reason']=='eos':
                    require(continuation and continuation[-1] in observation['eos_ids']
                        and not any(token in observation['eos_ids'] for token in continuation[:-1]),'COLLECT_NATIVE_EOS')
                else:require(observation['stop_reason']=='length_cap' and len(inp)+len(continuation)==100,
                    'COLLECT_GENERATION_STOP_REASON')
        item=row['metrics']
        require(item['generation_prompt_count']==len(observations)
            and item['generated_token_count']==sum(o['continuation_token_count'] for o in observations)
            and item['length_cap_no_continuation_count']==sum(o['stop_reason']=='length_cap_no_continuation' for o in observations),
            'COLLECT_GENERATION_PROMPT_TOKEN_COUNTS')
    reduced=reduce_generation(rows)
    require(reduced==saved['summary']==receipt['summary'],'COLLECT_GENERATION_RAW_SUM_COUNT_REDUCTION')
    work=receipt['work']
    require(all(type(work[k]) is int and work[k]>=0 for k in
        ('new_case_observations','cached_case_observations','generation_forwards','full_prefix_token_work'))
        and work['new_case_observations']+work['cached_case_observations']==len(rows)
        and type(work['seconds']) in (int,float) and math.isfinite(work['seconds']) and work['seconds']>=0,
        'COLLECT_GENERATION_WORK_ACCOUNTING')
    return dict(summary=reduced,rows=rows,work=work)


def native_guard(commit,arm,batch,cold):
    native=commit['native'];hp=native['hparams'];history=HISTORY_LAYERS[arm]
    require(commit['native_counts']==native['delta']==EXPECTED[arm]
        and native['arm']==arm and native['batch']==batch and native['requests']==100
        and native['same_model_returned'] is True and native['native_has_history'] is bool(history)
        and native['caller_history_appends']==0 and native['cache_template'] is None
        and native['native_z_disk_cache'] is False and native['checkpoint_saved'] is False,
        'COLLECT_NATIVE_ONE_APPLY_HISTORY_COUNTS')
    require(hp['layers']==list(ARM_LAYERS[arm]) and hp['v_lr']==.5 and hp['v_num_grad_steps']==20
        and hp['v_loss_layer']==47 and hp['v_weight_decay']==.5 and hp['clamp_norm_factor']==.75
        and hp['kl_factor']==.0625,'COLLECT_NATIVE_HPARAMS')
    if arm=='BASE_ALPHAEDIT':require(hp['L2']==10 and native['Alpha_reset_cache'] is (batch==1),'COLLECT_STOCK_ALPHA_RESET_ONCE')
    elif arm in ('CAKE','ALPHAEDIT_BLUE'):
        require(hp['L2']==(40 if arm=='CAKE' else 80) and (arm=='CAKE' or hp['blue'] is True),'COLLECT_NATIVE_PROJECTED_WRITER')
    else:require(hp['mom2_update_weight']==20000,'COLLECT_NATIVE_MEMIT_COVARIANCE')
    if arm in ('PRUNE','RECT'):
        require(hp['blue'] is False and native['return_orig_weights'] is (arm=='PRUNE' and batch==1)
            and native['selected_coldW0_saved_in_RAM'] is (arm=='PRUNE' and batch==1)
            and native['native_uses_P'] is False,'COLLECT_PRUNE_RECT_NATIVE_STATE')
        transform_guard(commit,arm,batch,cold)
        return rect_masks(native,arm,batch)
    require(commit['prune_applied'] is False and commit['terminal_transform']['prune_applied'] is False
        and commit['native_after']==commit['after'],'COLLECT_NO_FOREIGN_TERMINAL_TRANSFORM')
    return []


def generation_metric_row(arm,name,edits,value):
    return dict(arm=arm,endpoint=name,edits=edits,**{k:v for k,v in value['summary'].items()
        if k!='missing_reason_counts'},**{'missing_'+k+'_count':v for k,v in value['summary']['missing_reason_counts'].items()})


def returned_work(reader,out,c,commits,W0_recorded=True):
    """Preserve returned-but-uncommitted phase costs without nested double count."""
    result=[]
    for path in sorted((Path(out)/'generation-work').glob('*.json')):
        value=reader.json(path);receipt=value['observation'];endpoint=reader.bound(receipt['rows'])
        expected_endpoint=value['phase'][:-5] if value['phase'].endswith('_POST') else value['phase']
        require(value['phase']==path.stem and type(value['batch']) is int and 0<=value['batch']<=20
            and value['scientific_commit_not_asserted'] is True and value['raw_local_only'] is True
            and value['checkpoint_saved'] is False and receipt['identity']==endpoint['identity']
            and endpoint['identity_sha256']==receipt['identity_sha256']==digest(endpoint['identity'])
            and endpoint['identity']['runtime']==digest(runtime_identity(c))
            and endpoint['identity']['endpoint']==expected_endpoint
            and endpoint['summary']==receipt['summary'] and receipt['RNG_restored'] is True
            and receipt['observer_no_mutation'] is True,'COLLECT_RETURNED_PHASE_WORK_BINDING')
        work=receipt['work']
        require(all(type(work[k]) is int and work[k]>=0 for k in
            ('new_case_observations','cached_case_observations','generation_forwards','full_prefix_token_work'))
            and type(work['seconds']) in (int,float) and math.isfinite(work['seconds']) and work['seconds']>=0,
            'COLLECT_RETURNED_PHASE_WORK_SCALARS')
        accounted=(value['batch']==0 and W0_recorded) or 0<value['batch']<=commits
        result.append(dict(phase=value['phase'],batch=value['batch'],**work,
            committed_batch=accounted,
            counted_already_in_commit_receipt=accounted,
            cost_not_additive_to_program_or_allocation=True))
    return result


def review_arm(reader,attempt,c,lock,arm,identities,records,progress=None):
    out=Path(attempt)/arm;packs=c['packs'];all_ids=[case for p in packs for case in p['ids']]
    require(len(packs)==20 and all(len(p['ids'])==100 for p in packs) and len(set(all_ids))==2000,'COLLECT_20X100_ORDER')
    metrics=[];pairs=[];costs=[];counts=[];generation_table=[];masks=[];commits=[];at_write=[];prefixes={}
    result=dict(arm=arm,scientific_status='PARTIAL_OR_NOT_VERIFIED',commits=0,requests=0,state_links=0,
        W0_available=False,generation_W0_available=False,metric_rows=metrics,paired_rows=pairs,compute_rows=costs,
        counter_rows=counts,generation_rows=generation_table,rect_mask_rows=masks,missing=[],new_model_forwards=0)
    if progress is not None:progress.update(result)
    if not (out/'runtime.json').exists():result['missing'].append('RUNTIME_NOT_RECORDED');return result
    runtime=reader.json(out/'runtime.json');cold=runtime['cold_state']
    require(cold['W']==c['cold_W'] and runtime['source']==lock['source_commit'] and runtime['config']==digest(c)
        and runtime['arm']==arm and runtime['cold_history_zero_verified'],'COLLECT_RUNTIME_COLD_BINDING')
    initial=HISTORY_LAYERS[arm] if arm in ('CAKE','ALPHAEDIT_BLUE') else ()
    require(cold['H']=={str(l):zero_history_hash() for l in initial},'COLLECT_COLD_HISTORY_ZERO')
    w0=endpoint(reader,out/'W0',identities,all_ids,'W0',cold,records)
    if w0:
        result['W0_available']=True;metrics.extend(_metric_rows(arm,'W0_FIRST2000',0,w0['summary']))
        costs.append(dict(arm=arm,phase='W0_RPN',batch=0,seconds=w0['seconds'],reference_only=w0['reference_only']))
    wg=None
    if (out/'generation-W0.json').exists():
        value=reader.json(out/'generation-W0.json');ready=reader.bound(value['shared_ready'])
        require(value['model_weights_only'] and value['method_state_reused'] is False
            and (ready.get('cold_W0_completed_observation') is True if 'qualification_plan_sha256' in c['generation'] else ready['fresh_actual_W0'])
            and ready['raw_local_only'] and ready['no_checkpoint']
            and ready['identity']==dict(runtime=digest(runtime_identity(c)),model_W=cold['W'],
                occurrences=list(range(1,2001)),case_ids=all_ids),'COLLECT_SHARED_GENERATION_W0_READY')
        if 'qualification_plan_sha256' in c['generation']:
            actual_member=qualified_generation(c)['qualification_receipt_member']
            require(ready['completion_verified'] is True and ready['generation_repair_nonce']==NONCE
                and ready['qualification_receipt']==actual_member
                and ready['qualification_receipt_sha256']==actual_member['sha256']
                and ready['compatibility_sha256']==ready['compatibility']['sha256'],
                'COLLECT_SHARED_W0_ACTUAL_QUALIFICATION_COMPATIBILITY')
            reader.bound(ready['qualification_receipt']);reader.bound(ready['compatibility'])
        wg=generation_endpoint(reader,value['observation'],c,records,'W0','FIRST2000',cold)
        require(value['reused'] is (arm!=c['generation']['primary_arm']),'COLLECT_W0_GENERATED_ONCE_PRIMARY')
        if value['reused']:require(wg['work']['generation_forwards']==0,'COLLECT_SHARED_W0_NO_SECOND_FORWARD')
        result['generation_W0_available']=True;generation_table.append(generation_metric_row(arm,'W0_FIRST2000',0,wg))
        costs.append(dict(arm=arm,phase='W0_GENERATION',batch=0,**wg['work']))
    for number in range(1,21):
        folder=out/f'batch-{number:02d}'
        if not (folder/'commit.json').exists():break
        commit=reader.json(folder/'commit.json');ids=packs[number-1]['ids'];before=cold if not commits else commits[-1]['after']
        require(commit['task']==TASK and commit['arm']==arm and commit['batch']==number and commit['case_ids']==ids
            and commit['source']==lock['source_commit'] and commit['config']==digest(c) and commit['before']==before
            and set(commit['after']['W'])==set(cold['W']) and commit['observer_no_mutation']
            and set(commit['after']['H'])=={str(l) for l in HISTORY_LAYERS[arm]},'COLLECT_COMMIT_EXACT_LINK')
        if arm=='ALPHAEDIT_BLUE':require(all(commit['after']['W'][str(l)]==before['W'][str(l)] for l in (14,15,16)),'COLLECT_BLUE_ONLY_TWO_LAYERS')
        seen=records[:number*100];seen_ids=all_ids[:number*100];current=records[(number-1)*100:number*100]
        pre=endpoint(reader,folder/'pre',identities,ids,f'B{number}_PRE',before,seen)
        post=endpoint(reader,folder/'post',identities,seen_ids if number in MILESTONES else ids,f'W{number}',commit['after'],seen)
        require(pre is not None and post is not None,'COLLECT_COMMITTED_RPN_MISSING')
        compare_summary(pre['summary'],commit['pre']);compare_summary(post['summary'],commit['post'])
        current_rows=[r for r in post['rows'] if r['case_id'] in set(ids)];post_current=reduce_rows(current_rows)
        compare_summary(post_current,commit['post_current'])
        require({k:post_current[k]['denominator'] for k in 'RPN'}==dict(R=100,P=200,N=1000),'COLLECT_CURRENT_ALWAYS100')
        metrics.extend(_metric_rows(arm,f'B{number}_PRE',number*100,pre['summary']))
        metrics.extend(_metric_rows(arm,f'W{number}_CURRENT',number*100,post_current))
        pairs.extend(_paired_rows(arm,f'B{number}_PRE',f'W{number}_CURRENT',pre['rows'],current_rows));at_write.extend(current_rows)
        if number in MILESTONES:
            prefixes[number]=post['rows'];metrics.extend(_metric_rows(arm,f'W{number}_ALL_SEEN',number*100,post['summary']))
            metrics.extend(_metric_rows(arm,f'W{number}_FIRST500',number*100,reduce_rows([r for r in post['rows'] if r['case_id'] in set(all_ids[:500])])))
            pairs.extend(_paired_rows(arm,'AT_WRITE',f'W{number}_ALL_SEEN',at_write,post['rows']))
            for born in range(1,number+1):
                cohort=set(packs[born-1]['ids']);pairs.extend(_paired_rows(arm,f'B{born}_AT_WRITE',f'W{number}',
                    [r for r in at_write if r['case_id'] in cohort],[r for r in post['rows'] if r['case_id'] in cohort],born))
            if w0:pairs.extend(_paired_rows(arm,'W0',f'W{number}',[r for r in w0['rows'] if r['case_id'] in set(seen_ids)],post['rows']))
            flags=active_flags(seen)
            for active in (True,False):
                subset=[r for r in post['rows'] if flags[r['case_id']] is active]
                if subset:metrics.extend(_metric_rows(arm,f'W{number}_'+('ACTIVE' if active else 'SUPERSEDED'),number*100,reduce_rows(subset)))
        masks.extend(native_guard(commit,arm,number,cold));counts.append(dict(arm=arm,batch=number,**commit['native_counts']))
        gen=commit['generation'];require(wg is not None,'COLLECT_GENERATION_W0_REQUIRED')
        checks=[('pre',current,f'B{number}_PRE','CURRENT',before,(number-1)*100+1),
            ('post',seen if number in MILESTONES else current,f'W{number}','ALL_SEEN' if number in MILESTONES else 'CURRENT',commit['after'],1 if number in MILESTONES else (number-1)*100+1),
            ('current',current,f'W{number}_CURRENT','CURRENT',commit['after'],(number-1)*100+1),
            ('w0_current',current,f'W0_CURRENT_B{number}','CURRENT',cold,(number-1)*100+1)]
        if number in MILESTONES:checks.append(('w0_all_seen',seen,f'W0_ALL_SEEN_B{number}','ALL_SEEN',cold,1))
        else:require(gen['w0_all_seen'] is None,'COLLECT_UNMEASURED_ALL_SEEN_OMITTED')
        for key,selected,name,cohort,physical,ordinal in checks:
            value=generation_endpoint(reader,gen[key],c,selected,name,cohort,physical,ordinal)
            if key in ('current','w0_current','w0_all_seen') or (key=='pre' and number==1):
                require(value['work']['new_case_observations']==value['work']['generation_forwards']==0,
                    'COLLECT_SAME_ENDPOINT_SUBSET_NO_FORWARD')
            generation_table.append(generation_metric_row(arm,name,number*100,value))
            costs.append(dict(arm=arm,phase='GENERATION_'+key,batch=number,**value['work']))
        current_gen=reader.bound(gen['current']['rows']);post_gen=reader.bound(gen['post']['rows'])
        selected_rows=[r for r in post_gen['rows'] if r['occurrence'] in set(range((number-1)*100+1,number*100+1))]
        require(current_gen['rows']==selected_rows,'COLLECT_CURRENT_ALL_SEEN_OVERLAP_REUSE')
        costs.extend([dict(arm=arm,phase='RPN_PRE',batch=number,seconds=pre['seconds']),
            dict(arm=arm,phase='RPN_POST',batch=number,seconds=post['seconds']),
            dict(arm=arm,phase='BATCH_INCLUSIVE',batch=number,seconds=commit['seconds'],nested_cost_not_added_again=True)])
        commits.append(commit);result.update(commits=len(commits),requests=100*len(commits),state_links=max(0,len(commits)-1))
        if progress is not None:progress.update(result)
    terminal=reader.json(out/'terminal.json') if (out/'terminal.json').exists() else None
    totals={k:sum(r[k] for r in counts) for k in COUNTERS}
    if terminal:
        require(terminal['source']==lock['source_commit'] and terminal['config']==digest(c)
            and terminal['commits']==len(commits) and all(terminal['native_counts'][k]>=totals[k] for k in COUNTERS),'COLLECT_TERMINAL_AND_FAILED_COST')
        if terminal['status']=='COMPLETED':
            require(terminal['completed_batches']==20 and terminal['edits']==2000 and terminal['state']==commits[-1]['after']
                and all(terminal['native_counts'][k]==totals[k] for k in COUNTERS)
                and terminal['terminal_transforms']==(1 if arm=='PRUNE' else 0)
                and terminal['prune_applied'] is (arm=='PRUNE'),'COLLECT_COMPLETE_20_COMMITS_NATIVE_COUNTS')
        costs.append(dict(arm=arm,phase='PROGRAM_WALL',seconds=terminal['program_seconds'],not_added_to_allocation=True))
    if 5 in prefixes and 20 in prefixes:pairs.extend(_paired_rows(arm,'W5_FIRST500','W20_FIRST500',prefixes[5],[r for r in prefixes[20] if r['case_id'] in set(all_ids[:500])]))
    complete=len(commits)==20 and w0 is not None and wg is not None and terminal is not None and terminal['status']=='COMPLETED'
    phase_work=returned_work(reader,out,c,len(commits),W0_recorded=wg is not None)
    # Committed work is already in the table above. Only failed/uncommitted
    # returned phases are appended; full parent allocation still covers partial
    # work that failed before an observation returned.
    costs.extend(dict(arm=arm,generation_phase=row['phase'],phase='UNCOMMITTED_GENERATION',
        **{k:v for k,v in row.items() if k!='phase'})
        for row in phase_work if not row['counted_already_in_commit_receipt'])
    result.update(scientific_status='COMPLETED_VALIDATED' if complete else 'PARTIAL_OR_NOT_VERIFIED',
        measured_native_counts=totals,terminal_status=terminal.get('status') if terminal else None,
        failed_or_uncommitted_native_counts={k:terminal['native_counts'][k]-totals[k] for k in COUNTERS} if terminal else None,
        terminal_prune_applied=bool(commits and commits[-1]['prune_applied']),no_PRICE_KKT=True,
        returned_generation_phase_count=len(phase_work),
        uncommitted_returned_generation_forwards=sum(r['generation_forwards'] for r in phase_work if not r['counted_already_in_commit_receipt']))
    if (out/'tracking-identity.json').exists():
        tracking=reader.json(out/'tracking-identity.json');cfg=tracking['config'];g=c['generation']
        require(cfg['task_id']==TASK and cfg['arm']==arm and cfg['model']=='gpt2xl' and cfg['role']=='scientific'
            and cfg['generation_metric_schema']==g['schema'] and cfg['generation_profile']==g['profile']
            and cfg['generation_eval_seed']==g['eval_seed'] and cfg['reference_assets_sha256']==g['reference_assets_sha256']
            and cfg['generation_source_sha']==g['generation_source_sha'] and cfg['baseline']==arm
            and ('qualification_plan_sha256' not in g or
                (cfg['generation_qualification_plan_sha256']==g['qualification_plan_sha256']
                 and cfg['generation_repair_instruction']==NONCE))
            and tracking['source_sha']==lock['source_commit'] and tracking['config_sha']==lock['config_sha256'],
            'COLLECT_TRACKING_GENERATION_IDENTITY')
        finish=reader.json(out/'tracking-finish.json') if (out/'tracking-finish.json').exists() else {}
        result['tracking']=dict(run_id=tracking['run_id'],url=tracking.get('url'),job_id=cfg['job_id'],
            startup_status=tracking.get('startup_readback',{}).get('status'),finish_status=finish.get('status','NOT_RECORDED'),
            SDK_acceptance_not_remote_readback=True,scientific_completion_is_separate=True)
    return result


def allocation_once(reader,attempt,lock,runner=None,owner=None):
    path=Path(attempt)/'submission.json'
    if not path.exists():return dict(status='NOT_RECORDED',queries=0,reason='OWN_SUBMISSION_NOT_FOUND')
    queries=0
    try:
        submission=reader.json(path)
        require(submission['instruction_id']==NONCE and submission['task_id']==TASK and submission['source_commit']==lock['source_commit'],'OWN_ACCOUNTING_BINDING')
        ids={arm:str(submission['jobs'][arm]) for arm in ARMS}
        require(len(set(ids.values()))==6 and all(re.fullmatch('[1-9][0-9]*',job) for job in ids.values()),'OWN_ACCOUNTING_SIX_IDS')
        expected_owner=pwd.getpwuid(os.getuid()).pw_name if owner is None else owner
        require(lock['owner']==expected_owner,'OWN_ACCOUNTING_OWNER');queries=1
        value=(subprocess.run if runner is None else runner)(['sacct','-X','-n','-P','-j',','.join(ids.values()),
            '--format=JobIDRaw,JobName%120,User,State,ExitCode,ElapsedRaw,AllocTRES'],text=True,capture_output=True,timeout=20,check=False)
        require(value.returncode==0 and len(value.stdout)<=1024**2,'OWN_ACCOUNTING_BOUNDED')
        records={}
        for line in value.stdout.splitlines():
            fields=line.strip().split('|')
            if fields and fields[-1]=='':fields.pop()
            require(len(fields)==7,'OWN_ACCOUNTING_COLUMNS')
            job,name,user,status,exit_code,elapsed,tres=fields
            require(job in ids.values() and job not in records,'OWN_ACCOUNTING_EXACT_PARENT')
            arm=next(a for a,j in ids.items() if j==job)
            require(name==TASK+'-'+arm and user==expected_owner and elapsed.isdigit(),'OWN_ACCOUNTING_SOURCE_OWNER_NAME')
            resources=dict(p.split('=',1) for p in tres.split(',') if '=' in p);gpu=resources.get('gres/gpu')
            if gpu is None:
                typed=[v for k,v in resources.items() if k.startswith('gres/gpu:')]
                require(len(typed)<=1,'OWN_ACCOUNTING_GPU_SCHEMA');gpu=typed[0] if typed else '0'
            require(gpu in ('0','1'),'OWN_ACCOUNTING_SINGLE_GPU')
            records[job]=dict(arm=arm,job_id=job,owner=user,scheduler_state=status,exit_code=exit_code,
                parent_elapsed_seconds=int(elapsed),allocated_GPUs=int(gpu),allocated_GPU_seconds=int(elapsed)*int(gpu),
                AllocTRES=tres,child_steps_excluded=True,allocation_not_program_timer_sum=True)
        require(set(records)==set(ids.values()),'OWN_ACCOUNTING_SIX_ROWS')
        return dict(status='RECORDED',queries=queries,records=[records[ids[a]] for a in ARMS],scheduler_completion_not_scientific_completion=True)
    except Exception as error:return dict(status='NOT_RECORDED',queries=queries,error_type=type(error).__name__,reason='OWN_ACCOUNTING_UNAVAILABLE_OR_IDENTITY_ERROR',no_retry=True)


def collect(attempt):
    attempt=Path(attempt);reader=Reader();c=reader.json(attempt/'config.json');lock=reader.json(attempt/'execution.lock.json')
    require(c['task_id']==TASK and c['instruction_id']==lock['instruction_id']==NONCE and sha(attempt/'config.json')==lock['config_sha256'],'COLLECT_CONFIG_SOURCE')
    for field in ('source_members','runtime_sources','launchers','native_closure','source_config_members'):
        for item in lock.get(field,[]):
            data=reader.bytes(item['path'])
            require(len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],
                'COLLECT_SOURCE_MEMBER_BYTES')
    identities=reader.bound(c['observer_identity'])['rows']
    stream=next(item for item in c['assets'] if item['path']==c['stream']);records=reader.bound(stream)[:2000]
    require([r['case_id'] for r in records]==[i for p in c['packs'] for i in p['ids']],'COLLECT_ORDERED_FIRST2000')
    out=attempt/'collector';require(not out.exists(),'COLLECT_CREATE_ONCE');out.mkdir();reviews=[]
    for arm in ARMS:
        progress={}
        try:reviews.append(review_arm(reader,attempt,c,lock,arm,identities,records,progress))
        except Exception as error:reviews.append(dict(progress,arm=arm,scientific_status='TECHNICAL_BLOCKED_REVIEW',error_type=type(error).__name__,error=str(error),valid_prefix_preserved=True))
    allocation=allocation_once(reader,attempt,lock);write(out/'allocation.json',allocation)
    for rec in allocation.get('records',[]):next(r for r in reviews if r['arm']==rec['arm']).setdefault('compute_rows',[]).append(dict(rec,phase='PARENT_ALLOCATION'))
    tables=(('metric_rows','metrics.csv'),('paired_rows','paired.csv'),('compute_rows','compute.csv'),
        ('counter_rows','native-counts.csv'),('generation_rows','generation.csv'),('rect_mask_rows','rect-support.csv'))
    for key,name in tables:_csv(out/name,[row for review in reviews for row in review.get(key,[])])
    compact=[{k:v for k,v in review.items() if k not in {key for key,_ in tables}} for review in reviews]
    complete=all(r['scientific_status']=='COMPLETED_VALIDATED' for r in compact)
    report=['# GPT2-XL native baseline generation CPU 검산','',
        '저장 scalar/raw를 독립 CPU 재집계했습니다. 새 모델 forward/fit/편집·W&B upload는 없습니다.','',
        '| Arm | 상태 | 완료 batch | 요청 | 연결 |','| --- | --- | ---: | ---: | ---: |']
    for review in compact:report.append('| %s | %s | %s | %s | %s |'%(review['arm'],review['scientific_status'],review.get('commits','NA'),review.get('requests','NA'),review.get('state_links','NA')))
    report.extend(['','Fluency는 prompt-inclusive text의 ngram entropy(bits), consistency는 고정 reference TF-IDF cosine입니다.',
        'Occurrence별 valid count/sum을 집계하고 missing을 0점으로 채우지 않았습니다. 낮은 결과는 기술 failure gate가 아닙니다.',
        'W0 generation은 원 source/runtime를 보존한 검증된 완료 행을 재사용하고, 미완료·미검증 행만 새 cold W0 route로 관측합니다.',
        'Primary의 전체2k 완료 READY는 new/reused mixed provenance와 실제 qualification receipt를 결속하며, 다른 arm은 그 exact subset을 공유합니다.',
        'Current는 항상100, all-seen은 W5/W10/W15/W20 prefix입니다. overlap/원 W0 subset은 추가 forward가 없습니다.',
        'PRUNE W20은 명시적 PRUNE_TERMINAL_BASE_FIX 이후이고, RECT actual support/ties는 별도 표입니다.',
        'Native 수식/계수/fit은 기존 source를 재사용했습니다. R/P desired=new, N desired=true; TF와 자유생성 지표는 별개입니다.',
        'SDK 접수·원격 readback·과학 완료는 분리합니다. Agent 반복 monitoring은 만들지 않았습니다.',
        'Program/nested stages/parent GPU allocation 비용을 중복 합산하지 않았습니다. Accounting: '+allocation['status']+'.',''])
    _atomic_text(out/'report-ko.md','\n'.join(report));write(out/'review.json',dict(task=TASK,source=lock['source_commit'],reviews=compact,scientific_complete=complete,new_model_forwards=0,GPU_validation_by_collector=False))
    outputs=[member(p) for p in sorted(out.iterdir()) if p.is_file()]
    write(out/'manifest.json',dict(task=TASK,source=lock['source_commit'],inputs=list(reader.files.values()),outputs=outputs,raw_copied=False,raw_free_report=True,no_checkpoints=True))
    write(out/'terminal.json',dict(status='COMPLETED',scientific_complete=complete,
        scientific_status='COMPLETED_VALIDATED' if complete else 'PARTIAL_OR_TECHNICAL_BLOCKED',source=lock['source_commit'],
        report=member(out/'report-ko.md'),manifest=member(out/'manifest.json'),new_model_forwards=0))
    return dict(collector_status='COMPLETED',scientific_complete=complete,output=str(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    print(json.dumps(collect(p.parse_args().attempt)))
