"""Qualification CPU fixtures only; never actual model/GPU qualification."""
import copy
import unittest

import numpy as np

from . import generation_cache_qualification as q


class Tokenizer:
    def __call__(self, prompt, **kwargs):
        if kwargs not in (dict(padding=False,truncation=False),
                         dict(return_tensors='pt',padding=False,truncation=False)):
            raise AssertionError('No padding/truncation allowed')
        ids=list(range(int(prompt)))
        if kwargs.get('return_tensors')=='pt':
            import torch
            return {'input_ids':torch.tensor([ids])}
        return {'input_ids':ids}


def records(lengths=(4,)*8+(99,100,101,2,130)):
    return [dict(case_id=i, generation_prompts=[str(lengths[i % len(lengths)])]) for i in range(2000)]


def plan(microbatch=8):
    return q.build_plan(records(), Tokenizer(), model_identity='model-bound',
        tokenizer_identity='tokenizer-bound', reference_identity='r'*64,
        shared_source_sha='s'*40, native_source_binding={'fixture': True},
        batch_microbatch=microbatch,
        fixture_only=True,
        admission_reason='DEFAULT_MB8' if microbatch == 8 else 'PREDECLARED_MEMORY_MB4')


def proof(cohort, route):
    rows, trace = [], []
    for prompt in cohort['prompts']:
        size = prompt['input_token_count']
        ids = list(range(size)); new = [201,202][:max(0,min(2,100-size))]
        rows.append(dict(occurrence=prompt['occurrence'],prompt_index=prompt['prompt_index'],
            input_token_ids=ids,continuation_token_ids=new,full_token_ids=ids+new,
            input_token_count=size,continuation_token_count=len(new),
            stop_reason='length_cap_no_continuation' if not new else 'eos' if new[-1]==202 else 'length_cap',
            eos_ids=[202],eos_binding={'tokenizer':[202]},seed=prompt['occurrence']))
        for step, _ in enumerate(new):
            prefix=ids+new[:step]
            query=prefix if route==q.REFERENCE or step==0 else prefix[-1:]
            positions=list(range(len(prefix)-len(query),len(prefix)))
            trace.append(dict(occurrence=prompt['occurrence'],prompt_index=prompt['prompt_index'],step=step,
                prefix_token_ids=prefix,query_token_ids=query,attention_mask=[1]*len(prefix),
                position_ids=positions,cache_position=positions,
                logits=np.linspace(.1,.6,50400,dtype=np.float32),
                topk_ids=[5,4,3,2,1],topk_probabilities=np.array([.3,.25,.2,.15,.1],dtype=np.float32),
                generator_state_before='stream-'+str(step),generator_state_after='stream-'+str(step+1)))
    return dict(rows=rows,trace=trace,metric_rows=[{'count':len(rows),'mean':.5,'missing':[]}],
        metric_summary={'planned_count':len(rows),'reference_score':.5,'missing_reason_counts':{'x':0}},
        coverage={**{k:True for k in q.REQUIRED_COVERAGE},'actual_max_microbatch':8,
                  'active_row_removal':True},
        work={'physical_forward_calls':len(trace),'prefill_query_tokens':32,
              'decode_query_tokens':8,'logical_row_token_decisions':len(trace)},
        cost={'elapsed_seconds':.1,'synchronized_GPU_seconds':.09,
              'peak_gpu_allocated_bytes':16,'peak_gpu_reserved_bytes':32,'peak_host_RSS_bytes':64})


class QualificationPlanTests(unittest.TestCase):
    def test_plan_is_deterministic_length_only_and_ids_remain_local(self):
        before=records(); original=copy.deepcopy(before)
        frozen,local=plan()
        self.assertEqual((frozen,local),plan())
        self.assertEqual(before,original)
        self.assertEqual(frozen['coverage']['same_length_width'],4)
        self.assertEqual(frozen['cohort_count'],8)
        self.assertNotIn('prompts',frozen)
        self.assertNotIn('case_id',str(frozen))
        self.assertNotEqual(frozen['status'],'QUALIFIED_ACTUAL_GPU_ROUTE')
        self.assertTrue(frozen['coverage']['length100_boundary'])
        self.assertTrue(frozen['coverage']['shortest_input'])
        self.assertTrue(frozen['coverage']['longest_input'])
        self.assertEqual(local['prompts'][0]['input_token_count'],2)
        self.assertEqual(q.digest(local['shared_plan']),frozen['shared_plan_sha256'])

    def test_memory_alternative_is_frozen_before_measurement(self):
        frozen,local=plan(4)
        self.assertEqual(frozen['batch_microbatch'],4)
        self.assertEqual(frozen['admission_reason'],'PREDECLARED_MEMORY_MB4')
        self.assertTrue(frozen['coverage']['length100_boundary'])
        self.assertTrue(any(row['input_token_count']>=100 for row in local['prompts']))
        with self.assertRaisesRegex(RuntimeError,'ADMISSION_CHOICE'):
            q.build_plan(records(),Tokenizer(),model_identity='m',tokenizer_identity='t',
                reference_identity='r',shared_source_sha='s',native_source_binding={},batch_microbatch=4)

    def test_tolerance_and_local_cohort_are_immutable_gates(self):
        frozen,local=plan()
        q.validate_plan(frozen,local)
        frozen['tolerances']['logits']['atol']=.01
        with self.assertRaisesRegex(RuntimeError,'FROZEN_PLAN_CHANGED'):
            q.validate_plan(frozen,local)
        frozen,local=plan()
        local['prompts'][0]['prompt']='changed'
        with self.assertRaisesRegex(RuntimeError,'FROZEN_COHORT_CHANGED'):
            q.validate_plan(frozen,local)

    def test_native_kv_positions_logits_stream_and_output_agree(self):
        frozen,local=plan()
        reference=proof(local,q.REFERENCE)
        cached=proof(local,q.SINGLETON)
        self.assertEqual(q.compare_route(frozen,local,reference,cached,q.SINGLETON)['status'],'PASS')
        self.assertEqual(q.compare_route(frozen,local,reference,proof(local,q.BATCH),q.BATCH)['status'],'FAILED_GATE')
        frozen,local=plan(4); reference=proof(local,q.REFERENCE); batch=proof(local,q.BATCH)
        batch['coverage']['actual_max_microbatch']=4
        self.assertEqual(q.compare_route(frozen,local,reference,batch,q.BATCH)['status'],'PASS')

    def test_close_logits_are_not_exact_token_or_topk_claim(self):
        frozen,local=plan(); reference=proof(local,q.REFERENCE)
        changed=proof(local,q.SINGLETON)
        changed['trace'][0]['logits']+=1e-5
        self.assertEqual(q.compare_route(frozen,local,reference,changed,q.SINGLETON)['status'],'PASS')
        changed['trace'][0]['topk_ids']=[1,2,3,4,5]
        result=q.compare_route(frozen,local,reference,changed,q.SINGLETON)
        self.assertTrue(result['gates']['logits']); self.assertFalse(result['gates']['topk_ids'])
        self.assertEqual(result['status'],'FAILED_GATE')

    def test_position_mapping_eos_and_generator_stream_fail_separately(self):
        frozen,local=plan(); reference=proof(local,q.REFERENCE)
        for field,gate in [('cache_position','positions'),('generator_state_after','seed_stream')]:
            changed=proof(local,q.SINGLETON); changed['trace'][1][field]=[]
            self.assertFalse(q.compare_route(frozen,local,reference,changed,q.SINGLETON)['gates'][gate])
        changed=proof(local,q.SINGLETON); changed['rows'][0]['eos_ids']=[999]
        self.assertFalse(q.compare_route(frozen,local,reference,changed,q.SINGLETON)['gates']['EOS'])
        changed=proof(local,q.BATCH); changed['coverage']['active_row_removal']=False
        self.assertFalse(q.compare_route(frozen,local,reference,changed,q.BATCH)['gates']['coverage'])

    def test_metrics_float_tolerance_does_not_relax_count_or_missing_gate(self):
        frozen,local=plan(); reference=proof(local,q.REFERENCE)
        changed=proof(local,q.SINGLETON); changed['metric_summary']['reference_score']+=5e-7
        self.assertTrue(q.compare_route(frozen,local,reference,changed,q.SINGLETON)['gates']['metrics'])
        changed['metric_summary']['planned_count']-=1
        self.assertFalse(q.compare_route(frozen,local,reference,changed,q.SINGLETON)['gates']['metrics'])
        changed=proof(local,q.SINGLETON); changed['metric_summary']['missing_reason_counts']['x']=1
        self.assertFalse(q.compare_route(frozen,local,reference,changed,q.SINGLETON)['gates']['metrics'])

    def test_missing_equal_length_or_full_width_coverage_is_not_pass(self):
        frozen,local=plan(); reference=proof(local,q.REFERENCE)
        changed=proof(local,q.BATCH); changed['coverage']['actual_max_microbatch']=4
        self.assertEqual(q.compare_route(frozen,local,reference,changed,q.BATCH)['status'],'FAILED_GATE')
        changed=proof(local,q.BATCH); changed['coverage'].pop('forced_prefix')
        self.assertEqual(q.compare_route(frozen,local,reference,changed,q.BATCH)['status'],'FAILED_GATE')

    def test_empty_trace_or_consistently_overflowed_reference_cannot_qualify(self):
        frozen,local=plan();reference=proof(local,q.REFERENCE)
        reference['trace']=[]
        self.assertEqual(q.compare_route(frozen,local,reference,reference,q.REFERENCE)['status'],'FAILED_GATE')
        reference=proof(local,q.REFERENCE)
        row=next(row for row in reference['rows'] if row['input_token_count']==100)
        row['continuation_token_ids']=[202];row['continuation_token_count']=1
        row['full_token_ids'].append(202)
        self.assertFalse(q.compare_route(frozen,local,reference,reference,q.REFERENCE)['gates']['tokens'])

    def test_measured_work_cost_are_compact_and_missing_proof_fails(self):
        _,local=plan(); bundle=proof(local,q.SINGLETON)
        bundle['work']['raw_token_ids']=[1,2]
        bundle['cost']['private_prompt']='PRIVATE'
        work,cost=q.compact_work_cost(bundle)
        self.assertNotIn('raw_token_ids',work)
        self.assertNotIn('private_prompt',cost)
        bundle['cost']['peak_gpu_allocated_bytes']=0
        with self.assertRaisesRegex(RuntimeError,'MEASURED_COST_REQUIRED'):
            q.compact_work_cost(bundle)

    def test_fixed_fallback_order_requires_all_routes_once(self):
        results={route:{'status':'PASS'} for route in q.ROUTES}
        self.assertEqual(q.select_route(results),q.BATCH)
        results[q.BATCH]['status']='FAILED_GATE'
        self.assertEqual(q.select_route(results),q.SINGLETON)
        results[q.SINGLETON]['status']='FAILED_GATE'
        self.assertEqual(q.select_route(results),q.REFERENCE)
        del results[q.BATCH]
        self.assertIsNone(q.select_route(results))

    def test_unqualified_singleton_precludes_even_passing_batch(self):
        results={route:{'status':'PASS'} for route in q.ROUTES}
        results[q.SINGLETON]['status']='FAILED_GATE'
        self.assertEqual(q.select_route(results),q.REFERENCE)

    def test_consumer_receipt_rejects_boolean_only_or_missing_actual_cost(self):
        frozen,local=plan(); frozen['fixture_only']=False
        reference=proof(local,q.REFERENCE)
        bundles={route:proof(local,route) for route in q.ROUTES}
        results={route:q.compare_route(frozen,local,reference,bundles[route],route)
                 for route in q.ROUTES}
        compact={route:q.compact_work_cost(bundles[route]) for route in q.ROUTES}
        receipt=dict(schema=q.RECEIPT_SCHEMA,status='QUALIFIED_ACTUAL_GPU_ROUTE',actual_GPU=True,
            plan_sha256=q.digest(frozen),cohort_sha256=frozen['cohort_sha256'],
            model_identity=frozen['model_identity'],shared_source_sha=frozen['shared_source_sha'],
            native_source_binding=frozen['native_source_binding'],tolerances=frozen['tolerances'],
            shared_plan_sha256=frozen['shared_plan_sha256'],
            selected_route=q.SINGLETON,fixed_microbatch=1,route_results=results,
            selected_route_passed=True,state_unchanged=True,RNG_restored=True,no_fit=True,
            checkpoint_saved=False,work={r:compact[r][0] for r in q.ROUTES},
            cost={r:compact[r][1] for r in q.ROUTES})
        # This exercises a scalar consumer contract, not actual GPU proof.
        self.assertTrue(q.validate_actual_receipt(receipt,frozen))
        changed=copy.deepcopy(receipt); changed['route_results'][q.SINGLETON]['gates']={}
        with self.assertRaisesRegex(RuntimeError,'MEASURED_GATES'):
            q.validate_actual_receipt(changed,frozen)
        changed=copy.deepcopy(receipt); changed['work']={}
        with self.assertRaisesRegex(RuntimeError,'THREE_ROUTE_COSTS'):
            q.validate_actual_receipt(changed,frozen)
        changed=copy.deepcopy(receipt); changed['actual_GPU']=False
        with self.assertRaisesRegex(RuntimeError,'ACTUAL_RECEIPT_REQUIRED'):
            q.validate_actual_receipt(changed,frozen)

    def test_cpu_fixture_never_enters_actual_gpu_execution(self):
        frozen,local=plan()
        with self.assertRaisesRegex(RuntimeError,'CPU_FIXTURE_NOT_GPU_PROOF'):
            q.qualify_runtime(frozen,local,object(),None,None,'/fixture/not-written')


if __name__=='__main__':
    unittest.main()
