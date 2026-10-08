"""GPT-J caller adapter of SH1 immutable cache APIs; no generator copy/refit.

Qualification executes shared generate_rows exactly once per fixed route.
Captured probabilities replay independent prompt RNG streams; that is derived
evidence, not an assertion that internal Generator states were measured.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from . import generation_cache_qualification as q
from .generation_bridge import GenerationObserver as LegacyBridge, OCCURRENCE_FIELDS
from .generation_common import read, verify
from project.run_scripts.experiment_generation_eval.common import (
    PROFILE, SCHEMA, EVAL_SEED, case_seed, digest, immutable_write, require)
from project.run_scripts.experiment_generation_eval.compatibility import (
    member, verify_member, runtime_identity, verified_endpoint_row, load_compatibility)
from project.run_scripts.experiment_generation_eval.kv_qualification import (
    QUALIFICATION_SCHEMA, TOLERANCES as SHARED_TOLERANCES, verify_actual_receipt)


EXECUTION_ADAPTER='TASK_PRIVATE_SHARED_GENERATE_ROWS_PROOF_CONVERSION_NOT_SHARED_RUN_QUALIFICATION'
REPLAY_EVIDENCE='DEVICE_LOCAL_GENERATOR_REPLAY_NOT_INTERNAL_STATE_MEASUREMENT'
LINK_SCHEMA='gptj-generation-qualification-link-v1'
SHARED_SOURCE=q.SHARED_SOURCE
PACKAGE_TREE='e6935b9a099b14d7ad8e5e39e5455da08b2159f8'


def api_binding():
    """Public compact read-bound API, never actual qualification/route PASS."""
    return dict(status='READ_BOUND_SHARED_API',source_sha=SHARED_SOURCE,package_tree=PACKAGE_TREE,
        generator='project.run_scripts.experiment_generation_eval.generator.generate_rows',
        plan='project.run_scripts.experiment_generation_eval.kv_qualification.build_qualification_plan',
        actual_verifier='project.run_scripts.experiment_generation_eval.kv_qualification.verify_actual_receipt',
        actual_execution=EXECUTION_ADAPTER,source_namespace_modified=False,actual_GPU_qualification=False)


def _list(value):
    if hasattr(value,'detach'):
        value=value.detach().cpu().tolist()
    if value and isinstance(value[0],list):
        require(len(value)==1,'QUALIFICATION_TRACE_SINGLE_LOGICAL_ROW')
        value=value[0]
    return value


def _state_sha(generator):
    return hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()


def _batch_coverage(requests, trace, microbatch):
    """Derive actual logical width/shrink only from executed cached traces."""
    groups={}
    for index,row in enumerate(requests):
        if row['token_length']<100:
            groups.setdefault(row['token_length'],[]).append(index)
    observed_width=0;removed=False
    for indices in groups.values():
        for offset in range(0,len(indices),microbatch):
            keys={(requests[i]['occurrence'],requests[i]['prompt_index'])
                  for i in indices[offset:offset+microbatch]}
            counts={}
            for row in trace:
                if (row['occurrence'],row['prompt_index']) in keys:
                    counts[row['step']]=counts.get(row['step'],0)+1
            observed_width=max(observed_width,max(counts.values(),default=0))
            removed |= any(0<counts[step]<counts[step-1] for step in counts if step>0)
    return dict(actual_max_microbatch=observed_width,active_row_removal=bool(removed))


def shared_measure(model, tokenizer, assets, plan, cohort):
    """Return an injected measure backed by real shared API and trace callback."""
    import torch
    from project.run_scripts.experiment_generation_eval.generator import generate_rows, model_device
    from project.run_scripts.experiment_generation_eval.metrics import score_case
    q.validate_plan(plan,cohort)
    requests=copy.deepcopy(cohort['shared_plan']['requests']);device=model_device(model)
    calls=[]
    def measure(*,route,cohort,microbatch,reference):
        require(route not in calls and route==q.ROUTES[len(calls)],'QUALIFICATION_SHARED_ROUTE_ONCE_ORDER')
        calls.append(route)
        require([(r['occurrence'],r['prompt_index'],r['prompt']) for r in cohort]
                ==[(r['occurrence'],r['prompt_index'],r['prompt']) for r in requests],
                'QUALIFICATION_SHARED_REQUESTS_FROZEN')
        captured=[];streams={};steps={};native_classes=set();native_positions=True
        def trace(value):
            nonlocal native_positions
            key=value['occurrence'],value['prompt_index'];step=steps.get(key,0);steps[key]=step+1
            if key not in streams:
                seed=case_seed(plan['model_identity'],*key,EVAL_SEED)
                streams[key]=torch.Generator(device=device).manual_seed(seed)
            stream=streams[key];before=_state_sha(stream)
            probabilities=value['topk_probabilities'].to(device)
            choice=torch.multinomial(probabilities,1,generator=stream)
            replay=int(torch.gather(value['topk_ids'].to(device),1,choice).item())
            positions=_list(value['position_ids']);prefix=_list(value['input_ids'])
            if route!=q.REFERENCE:
                native_classes.add(value['cache_class'])
                native_positions &= value.get('cache_position_explicit') is True
            captured.append(dict(occurrence=key[0],prompt_index=key[1],step=step,
                prefix_token_ids=prefix,query_token_ids=_list(value['query_input_ids']),
                attention_mask=_list(value['attention_mask']),position_ids=positions,
                cache_position=_list(value.get('cache_position',positions)),
                logits=value['logits'],topk_ids=_list(value['topk_ids']),
                topk_probabilities=value['topk_probabilities'],
                generator_state_before=before,generator_state_after=_state_sha(stream),
                replayed_sample_exact=replay==value['sampled_token']))
        if device.type=='cuda':
            torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
            start_event,stop_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start_event.record()
        started=time.perf_counter()
        forced=None if route==q.REFERENCE else [row['continuation_token_ids'] for row in reference['rows']]
        rows=generate_rows(model,tokenizer,requests,route=q.SHARED_ROUTES[route],microbatch=microbatch,
                           eval_seed=EVAL_SEED,trace=trace,forced_prefixes=forced)
        if device.type=='cuda':
            stop_event.record();torch.cuda.synchronize(device)
            gpu_seconds=start_event.elapsed_time(stop_event)/1000
            allocated=torch.cuda.max_memory_allocated(device);reserved=torch.cuda.max_memory_reserved(device)
        else:
            gpu_seconds=allocated=reserved=None
        require([(r['occurrence'],r['prompt_index']) for r in rows]
                ==[(r['occurrence'],r['prompt_index']) for r in requests],'QUALIFICATION_SHARED_ROWS_ORDER')
        rank={(r['occurrence'],r['prompt_index']):i for i,r in enumerate(requests)}
        captured.sort(key=lambda r:(rank[(r['occurrence'],r['prompt_index'])],r['step']))
        metric_rows=[]
        for request,row in zip(requests,rows):
            refs=assets.snippets_for(request['relation_id'],request['target_new_id']) if assets is not None else None
            metric_rows.append(score_case([row],refs,getattr(assets,'vectorizer',None),
                                          getattr(assets,'word_tokenize',None)))
        coverage={key:True for key in q.REQUIRED_COVERAGE}
        if route!=q.REFERENCE:
            coverage['forced_prefix']=native_positions and native_classes=={'transformers.cache_utils.DynamicCache'}
        if route==q.BATCH:
            coverage.update(_batch_coverage(requests,captured,microbatch))
        return dict(rows=rows,trace=captured,metric_rows=metric_rows,
            metric_summary={'prompt_count':len(metric_rows)},coverage=coverage,
            work=dict(physical_forward_calls=sum(r['physical_forward_calls'] for r in rows),
                prefill_query_tokens=sum(r['prefill_query_tokens'] for r in rows),
                decode_query_tokens=sum(r['decode_query_tokens'] for r in rows),
                logical_row_token_decisions=sum(r['model_forwards'] for r in rows)),
            cost=dict(elapsed_seconds=time.perf_counter()-started,synchronized_GPU_seconds=gpu_seconds,
                peak_gpu_allocated_bytes=allocated,peak_gpu_reserved_bytes=reserved,
                peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    measure.calls=calls
    return measure


def shared_receipt_body(proof, proof_member, plan):
    """Explicit conversion of actual caller evidence, NOT shared-run execution."""
    q.validate_actual_receipt(proof,plan)
    results={}
    costs={}
    for route in q.ROUTES:
        result=proof['route_results'][route];g=result['gates'];work=proof['work'][route];cost=proof['cost'][route]
        results[q.SHARED_ROUTES[route]]=dict(passed=result['status']=='PASS',executed=True,
            token_sequence_exact=g['tokens'] and g['EOS'] and g['row_mapping'] and g['seed_stream'],
            topk_mask_position_exact=g['topk_ids'] and g['positions'],logits_close=g['logits'],
            topk_probabilities_close=g['topk_probabilities'],
            metric_values_close_and_validity_reasons_exact=g['metrics'],
            max_abs_logit_error=result['max_logit_abs_error'],
            max_abs_topk_probability_error=result['max_topk_probability_abs_error'],
            coverage=copy.deepcopy(result['coverage']))
        costs[q.SHARED_ROUTES[route]]=dict(elapsed_sec=cost['elapsed_seconds'],
            GPU_seconds=cost['synchronized_GPU_seconds'],peak_allocated_bytes=cost['peak_gpu_allocated_bytes'],
            peak_reserved_bytes=cost['peak_gpu_reserved_bytes'],host_max_RSS_bytes=cost['peak_host_RSS_bytes'],
            logical_row_forward_decisions=work['logical_row_token_decisions'],**{k:work[k]
                for k in ('physical_forward_calls','prefill_query_tokens','decode_query_tokens')})
    selected=q.SHARED_ROUTES[proof['selected_route']]
    body=dict(schema=QUALIFICATION_SCHEMA,actual_qualification=True,qualification_pass=True,
        selected_route=selected,microbatch=proof['fixed_microbatch'],fixed_microbatch=proof['fixed_microbatch'],
        plan_sha256=plan['shared_plan_sha256'],private_plan_sha256=q.digest(plan),
        model_identity=plan['model_identity'],source_identity=plan['shared_source_sha'],
        profile=PROFILE,eval_seed=EVAL_SEED,tolerances=copy.deepcopy(SHARED_TOLERANCES),
        coverage=copy.deepcopy(plan['coverage']),route_results=results,cost=costs,
        model_no_mutation=proof['state_unchanged'],native_state_no_mutation=proof['state_unchanged'],
        RNG_restored=proof['RNG_restored'],fit_calls=0,edit_commits=0,retry_count=0,
        raw_local_only=True,pretrained_GPU_PASS=proof['actual_GPU'],
        caller_proof_member=copy.deepcopy(proof_member),execution_adapter=EXECUTION_ADAPTER,
        seed_stream_evidence=REPLAY_EVIDENCE,native_models_scope=['gptj'],
        selection_reason='FIXED_GATE_ORDER_NOT_SCIENTIFIC_QUALITY')
    body['identity_sha256']=digest(body)
    return body


def qualify_and_link(plan, cohort, model, tokenizer, assets, state_callback,
                     receipt_path, private_plan_member, shared_plan_member):
    """Actual first-job proof then separately linked real shared-schema member."""
    private_plan=read(verify_member(private_plan_member));shared_plan=read(verify_member(shared_plan_member))
    require(private_plan==plan and shared_plan==cohort['shared_plan'],'QUALIFICATION_PLAN_MEMBERS')
    measure=shared_measure(model,tokenizer,assets,plan,cohort)
    proof=q.qualify_runtime(plan,cohort,model,measure,state_callback,receipt_path)
    proof_member=member(receipt_path)
    require(measure.calls==list(q.ROUTES),'QUALIFICATION_THREE_ACTUAL_ROUTES_ONCE')
    shared_path=Path(receipt_path).parent/'shared-qualification-actual.json'
    immutable_write(shared_path,shared_receipt_body(proof,proof_member,plan));shared_member=member(shared_path)
    verify_actual_receipt(shared_member,expected_plan_sha256=plan['shared_plan_sha256'],
                          expected_model_identity=plan['model_identity'])
    link=dict(schema=LINK_SCHEMA,qualification=proof_member,qualification_plan=private_plan_member,
        qualification_plan_sha256=q.digest(plan),shared_qualification_receipt_member=shared_member,
        shared_qualification_plan=shared_plan_member,shared_qualification_plan_sha256=plan['shared_plan_sha256'],
        selected_route=proof['selected_route'],shared_selected_route=q.SHARED_ROUTES[proof['selected_route']],
        fixed_microbatch=proof['fixed_microbatch'],actual_GPU=True,checkpoint_saved=False)
    immutable_write(Path(receipt_path).parent/'qualification-link.json',link)
    return link


class GenerationObserver(LegacyBridge):
    """First cold owner qualifies once, then observes missing original-W0 rows."""
    def __init__(self,config,lock,view,engine,tokenizer,records,out,arm,tracker=None):
        from project.run_scripts.experiment_generation_eval.assets import load_assets
        from project.run_scripts.experiment_generation_eval.observer import GenerationObserver as SharedObserver
        self.config,self.view,self.engine=config,view,engine
        self.out,self.arm,self.gen=Path(out),arm,config['generation']
        self.repair,self.tracker=self.gen['repair'],tracker
        require(self.gen['schema']==SCHEMA and self.gen['profile']==PROFILE
                and self.gen['eval_seed']==EVAL_SEED and self.gen['W0_owner']=='BASE_MEMIT'
                and self.gen['source_sha']==q.SHARED_SOURCE,'CACHE_BRIDGE_PROFILE_SOURCE')
        self.records,self.by_case=[],{}
        records=list(records);require(len(records)==2000,'CACHE_BRIDGE_FIRST2000')
        for ordinal,original in enumerate(records,1):
            require(type(original.get('case_id')) is int and original['case_id'] not in self.by_case
                    and all(type(original[k]) is int and original[k]==ordinal
                        for k in OCCURRENCE_FIELDS if k in original),'CACHE_BRIDGE_RECORD_ORDER')
            record=dict(copy.deepcopy(original),occurrence_index=ordinal)
            self.records.append(record);self.by_case[record['case_id']]=record
        self.cohort_sha=digest(self.records);self.cold_state=copy.deepcopy(self.gen['W0_state_identity'])
        self.cache=Path(self.gen['W0_cache']);self._w0,self._receipts=None,{}
        self._progress_offset=None
        self.assets=load_assets(dict(generation_assets=self.gen['generation_assets'],
                                    asset_paths=self.gen.get('asset_paths',{})))
        require(self.assets.sha==self.gen['reference_assets_sha256'],'CACHE_BRIDGE_REFERENCE_IDENTITY')
        self._cold_weights()
        self.plan=read(verify_member(self.repair['qualification_plan']))
        self.cohort=read(verify_member(self.repair['qualification_cohort']))
        q.validate_plan(self.plan,self.cohort)
        require(q.digest(self.plan)==self.repair['qualification_plan_sha256']
                and self.plan['model_identity']==self.gen['model_identity']
                and self.plan['reference_identity']==self.assets.sha
                and self.plan['shared_source_sha']==self.gen['source_sha'], 'CACHE_BRIDGE_PLAN_BINDING')
        shared_plan_member=self.repair['shared_qualification_plan']
        require(read(verify_member(shared_plan_member))==self.cohort['shared_plan'],'CACHE_BRIDGE_SHARED_PLAN_MEMBER')
        receipt_path=Path(self.repair['qualification_receipt_path'])
        link_path=receipt_path.parent/'qualification-link.json'
        if link_path.exists():
            self.qualification_link=read(link_path)
            self._verify_link()
        else:
            require(arm=='BASE_MEMIT','CACHE_BRIDGE_QUALIFICATION_NOT_READY_NO_POLL')
            self.qualification_link=qualify_and_link(self.plan,self.cohort,view.model,tokenizer,self.assets,
                self._native_signature,receipt_path,self.repair['qualification_plan'],shared_plan_member)
            self._verify_link()
        self.qualification_link_member=member(link_path)
        shared_member=self.qualification_link['shared_qualification_receipt_member']
        shared_config=dict(model_identity=self.gen['model_identity'],generation_source_sha=self.gen['source_sha'],
            reference_assets_sha256=self.assets.sha,profile=PROFILE,eval_seed=EVAL_SEED,
            generation_route=self.qualification_link['shared_selected_route'],
            generation_microbatch=self.qualification_link['fixed_microbatch'],
            qualification_plan_sha256=self.plan['shared_plan_sha256'],qualification_receipt_member=shared_member)
        self.shared=SharedObserver(view.model,tokenizer,self.assets,shared_config,self.out/'generation-raw',
            state_callback=self._native_signature,progress_callback=self._progress)
        from .generation_cache_reuse import build_compatibility
        proof=read(verify_member(self.qualification_link['qualification']))
        self.private_compatibility=build_compatibility(self.repair['old_complete_case_inventory'],
            self.qualification_link['qualification'],self.shared.runtime_identity,
            {k:proof[k] for k in ('plan_sha256','cohort_sha256','shared_source_sha',
                'native_source_binding','selected_route','fixed_microbatch')},
            qualification_plan_member=self.repair['qualification_plan'],
            out=self.repair['compatibility_manifest_path'])
        self.private_compatibility_member=member(self.repair['compatibility_manifest_path'])
        self.runtime_aux=dict(self.qualification_link,source_commit=lock['source_commit'],
            config_sha256=lock['config_sha256'],shared_runtime_identity=copy.deepcopy(self.shared.runtime_identity),
            shared_runtime_sha256=self.shared.runtime_sha,qualification_link_member=self.qualification_link_member,
            shared_qualification_plan_sha256=self.plan['shared_plan_sha256'],
            compatibility_manifest=self.private_compatibility_member,
            checkpoint_saved=False,raw_local_only=True)
        immutable_write(self.out/'generation-repair-runtime.json',self.runtime_aux)
        self.runtime_aux_member=member(self.out/'generation-repair-runtime.json')

    def _native_signature(self):
        from project.run_scripts.experiment_generation_eval.observer import _cache_signature
        result=super()._native_signature()
        result['engine_contexts']=_cache_signature(getattr(self.engine,'contexts',None))
        return result

    def _verify_link(self):
        link=self.qualification_link
        require(link['schema']==LINK_SCHEMA and link['qualification_plan']==self.repair['qualification_plan']
                and link['qualification_plan_sha256']==q.digest(self.plan)
                and link['shared_qualification_plan']==self.repair['shared_qualification_plan']
                and link['shared_qualification_plan_sha256']==self.plan['shared_plan_sha256']
                and link['actual_GPU'] is True and link['checkpoint_saved'] is False,'CACHE_BRIDGE_DUAL_PLAN_LINK')
        proof=read(verify_member(link['qualification']));q.validate_actual_receipt(proof,self.plan)
        actual=verify_actual_receipt(link['shared_qualification_receipt_member'],
            expected_plan_sha256=self.plan['shared_plan_sha256'],expected_model_identity=self.gen['model_identity'])
        require(actual['caller_proof_member']==link['qualification']
                and actual['private_plan_sha256']==q.digest(self.plan)
                and actual['source_identity']==self.gen['source_sha']
                and actual['execution_adapter']==EXECUTION_ADAPTER
                and proof['selected_route']==link['selected_route']
                and q.SHARED_ROUTES[proof['selected_route']]==actual['selected_route']==link['shared_selected_route']
                and proof['fixed_microbatch']==actual['fixed_microbatch']==link['fixed_microbatch'],
                'CACHE_BRIDGE_DUAL_ACTUAL_LINK')

    def _progress(self,payload):
        from .generation_tracking import log_generation_progress
        values=copy.deepcopy(payload)
        if values['phase']=='W0_generation' and self._progress_offset is not None:
            offset=self._progress_offset
            for key in ('completed_cases','completed_prompts','generated_tokens','reused_cases'):
                values['generation_progress/'+key]+=offset[key]
            values['generation_progress/total_cases']=2000
            values['generation_progress/total_prompts']=sum(len(r.get('generation_prompts',[])) for r in self.records)
            # Rates retain actual newly measured phase work; old physical work
            # and prior generation wall/GPU cost are never added or relabelled.
        job=(self.tracker.config_values['job_id'] if self.tracker is not None
             else os.environ['SLURM_JOB_ID'])
        values.update(route=self.qualification_link['shared_selected_route'],model='gptj',job_id=job)
        from .generation_tracking_schema import metrics
        metrics(values,scientific=True,canonical=True)
        self.out.mkdir(parents=True,exist_ok=True)
        journal_name='generation-progress-W0.jsonl' if values['phase']=='W0_generation' else 'generation-progress.jsonl'
        with (self.out/journal_name).open('a') as journal:
            journal.write(json.dumps(values,sort_keys=True,allow_nan=False)+'\n')
        if self.tracker is not None:
            log_generation_progress(self.tracker,values)

    def _ready_identity(self):
        result=super()._ready_identity()
        result.update(qualification_sha256=self.qualification_link['qualification']['sha256'],
            shared_qualification_sha256=self.qualification_link['shared_qualification_receipt_member']['sha256'],
            shared_plan_sha256=self.plan['shared_plan_sha256'])
        return result

    def _wrap(self,receipt,state_identity,out=None):
        result=super()._wrap(receipt,state_identity)
        result.update(qualification=self.qualification_link['qualification'],
            qualification_receipt_member=self.qualification_link['shared_qualification_receipt_member'],
            shared_qualification_receipt_member=self.qualification_link['shared_qualification_receipt_member'],
            shared_runtime_identity=copy.deepcopy(self.shared.runtime_identity),
            shared_qualification_plan_sha256=self.plan['shared_plan_sha256'],
            qualification_link_member=self.qualification_link_member,
            qualification_plan_sha256=q.digest(self.plan),
            compatibility_manifest=self.private_compatibility_member,
            generation_repair_runtime_member=self.runtime_aux_member)
        if receipt['identity']['endpoint']=='W0' and getattr(self,'_w0_progress_binding',None) is not None:
            result['generation_progress']=copy.deepcopy(self._w0_progress_binding)
        if receipt['identity']['endpoint']=='W0' and hasattr(self,'_w0_reused_ready'):
            result['reused_complete_READY']=copy.deepcopy(self._w0_reused_ready)
        if 'compatibility_member' in receipt:
            result['compatibility_member']=copy.deepcopy(receipt['compatibility_member'])
        self._receipts[digest(result['cases'])]=result
        if out is not None:
            path=Path(out)/'receipt.json';immutable_write(path,result);result['receipt_path']=str(path.resolve())
        return result

    def load_W0(self):
        from .generation_cache_reuse import build_shared_compatibility, read_reusable_case, read_member
        from project.run_scripts.experiment_generation_eval.metrics import reduce_cases
        from project.run_scripts.experiment_generation_eval.observer import model_signature
        from project.run_scripts.experiment_generation_eval.generator import isolated_rng, rng_snapshot, rng_equal
        self._cold_weights()
        if self._w0 is not None:
            return copy.deepcopy(self._w0)
        ready_path=self.cache/'READY.json';identity=self._ready_identity()
        if ready_path.exists():
            ready=read(ready_path)
            require(ready['status']=='READY' and ready['identity']==identity
                and ready['identity_sha256']==digest(identity) and ready['producer_arm']=='BASE_MEMIT',
                'CACHE_BRIDGE_W0_READY_IDENTITY')
            require(ready['qualification']==self.qualification_link['qualification']
                and ready['qualification_receipt_member']==self.qualification_link['shared_qualification_receipt_member']
                and ready['compatibility_manifest']==self.private_compatibility_member
                and ready['shared_runtime_identity']==self.shared.runtime_identity,
                'CACHE_BRIDGE_READY_DUAL_ACTUAL')
            verify_member(ready['generation_progress'])
            self._w0_progress_binding=None
            self._w0_reused_ready=dict(status='REUSED_COMPLETE_READY',completed_cases=2000,
                READY=member(ready_path),new_generation=0,
                producer_generation_progress_member=ready['generation_progress'])
            receipt=self.shared.read_observed(verify_member(ready['endpoint']))
        else:
            require(self.arm=='BASE_MEMIT','CACHE_BRIDGE_W0_NOT_READY_NO_POLL')
            before=model_signature(self.view.model);native=copy.deepcopy(self._native_signature());rng=rng_snapshot()
            with isolated_rng():
                inventory_member=self.repair['old_complete_case_inventory']
                inventory,_=read_member(inventory_member['path'],inventory_member)
                compatibility=build_shared_compatibility(inventory_member,
                    self.qualification_link['shared_qualification_receipt_member'],
                    self.repair['old_cold_observation_guard'],self.shared.runtime_identity,self.cold_state,
                    out=self.cache/'shared-compatibility.json',qualification_plan_member=self.repair['qualification_plan'],
                    actual_qualification_member=self.qualification_link['qualification'])
                reused={};old_prompts=old_tokens=0
                manifest=compatibility['manifest'];compat_sha=manifest['identity_sha256']
                for entry in manifest['identity']['original_entries']:
                    ordinal=entry['occurrence'];record=self.records[ordinal-1]
                    old=read_reusable_case(inventory,ordinal,record,tokenizer=self.shared.tokenizer)
                    raw=old['raw'];row=copy.deepcopy(old['row'])
                    row['provenance']=dict(origin='COMPATIBLE_ORIGINAL_W0',raw_member=entry['original_raw_member'],
                        runtime_sha256=entry['original_runtime_sha256'],
                        generation_source_sha=entry['original_generation_source_sha'],route=entry['original_route'],
                        compatibility_sha256=compat_sha)
                    reused[ordinal]=row;old_prompts+=len(raw['observations'])
                    old_tokens+=sum(r['continuation_token_count'] for r in raw['observations'])
                missing=[r for r in self.records if r['occurrence_index'] not in reused]
                self._progress_offset=dict(completed_cases=len(reused),reused_cases=len(reused),
                    completed_prompts=old_prompts,generated_tokens=old_tokens)
                try:
                    measured=self.shared.observe(missing,'W0',cohort='MISSING_FIRST2000',state_identity=self.cold_state)
                finally:
                    self._progress_offset=None
                rows={**reused,**{r['occurrence']:r for r in measured['rows']}}
                require(set(rows)==set(range(1,2001)),'CACHE_BRIDGE_W0_COMPLETE2000')
                ordered=[rows[i] for i in range(1,2001)]
                endpoint_identity=dict(runtime=self.shared.runtime_sha,state_sha256=digest(self.cold_state),
                    endpoint='W0',cohort='FIRST2000',ordered_occurrences=list(range(1,2001)),
                    observation_identities=[r['identity_sha256'] for r in ordered],
                    provenance_sha256=digest([r['provenance'] for r in ordered]),
                    qualification_receipt_sha256=self.qualification_link['shared_qualification_receipt_member']['sha256'],
                    compatibility_sha256=compat_sha)
                receipt=dict(identity=endpoint_identity,identity_sha256=digest(endpoint_identity),
                    summary=reduce_cases(ordered),rows=ordered,RNG_restored=True,observer_no_mutation=True,
                    raw_local_only=True,qualification_receipt_member=self.qualification_link['shared_qualification_receipt_member'],
                    compatibility_member=compatibility['member'],
                    observer_identity_member=member(self.shared.out/'observer-identity.json'))
                for record,row in zip(self.records,ordered):
                    from project.run_scripts.experiment_generation_eval.observer import _record_identity
                    verified_endpoint_row(row,receipt,expected_record_identity=_record_identity(record,record['occurrence_index']),
                                          expected_state=self.cold_state)
                require(model_signature(self.view.model)==before and self._native_signature()==native
                    and rng_equal(rng),'CACHE_BRIDGE_W0_COMPOSITION_STATE_RNG')
                path=self.cache/'W0-endpoint.json';immutable_write(path,receipt)
                self.shared.read_observed(path)
                receipt.update(rows_path=str(path.resolve()),work=dict(measured['work'],
                    cached_case_observations=measured['work']['cached_case_observations']+len(reused),
                    completed_prompts=measured['work'].get('completed_prompts',
                        sum(len(r.get('generation_prompts',[])) for r in missing))+old_prompts,
                    generated_tokens=measured['work'].get('generated_tokens',
                        sum(r['metrics']['generated_token_count'] for r in measured['rows']))+old_tokens))
                self._validate_W0(receipt)
                progress_member=member(self.out/'generation-progress-W0.jsonl')
                final=json.loads(Path(progress_member['path']).read_text().splitlines()[-1])
                require(final['phase']=='W0_generation' and final['generation_progress/completed_cases']==2000
                    and final['generation_progress/total_cases']==2000
                    and final['generation_progress/completed_prompts']==final['generation_progress/total_prompts'],
                    'CACHE_BRIDGE_W0_FINAL_PROGRESS')
                self._w0_progress_binding=progress_member
                immutable_write(ready_path,dict(status='READY',identity=identity,identity_sha256=digest(identity),
                    producer_arm='BASE_MEMIT',endpoint=member(path),producer_work=receipt['work'],
                    qualification=self.qualification_link['qualification'],
                    qualification_receipt_member=self.qualification_link['shared_qualification_receipt_member'],
                    shared_qualification_receipt_member=self.qualification_link['shared_qualification_receipt_member'],
                    compatibility_manifest=self.private_compatibility_member,generation_progress=progress_member,
                    shared_runtime_identity=copy.deepcopy(self.shared.runtime_identity),
                    observer_identity_member=member(self.shared.out/'observer-identity.json'),
                    **{key:copy.deepcopy(self.qualification_link[key]) for key in
                        ('qualification_plan','qualification_plan_sha256','shared_qualification_plan',
                         'shared_qualification_plan_sha256','selected_route','shared_selected_route','fixed_microbatch')},
                    compatibility_member=compatibility['member'],qualification_link_member=self.qualification_link_member,
                    checkpoint_saved=False,raw_local_only=True))
        self._validate_W0(receipt)
        self._w0=self._wrap(receipt,self.cold_state,self.out/'generation-W0')
        return copy.deepcopy(self._w0)
