"""CPU software fixtures only: no pretrained model/actual GPU qualification.

W0/READY tests bypass constructor and mock source-backed row validators; those
fixtures certify protocol assembly, not real old rows or native/model science.
"""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from . import generation_cache_bridge as b
from . import generation_cache_qualification as q
from .test_generation_cache_qualification import plan
from .test_generation_bridge import records
from project.run_scripts.experiment_generation_eval.common import digest, immutable_write
from project.run_scripts.experiment_generation_eval.metrics import reduce_cases


def metric_row(record):
    ordinal=record['occurrence_index'];identity=digest(['CPU_FIXTURE',ordinal])
    return dict(occurrence=ordinal,case_id=record['case_id'],identity_sha256=identity,
        payload_sha256=digest(identity),observation_path='/CPU_FIXTURE_NO_ACTUAL_RAW/'+identity+'.json',
        metrics=dict(ngram_entropy=0.,reference_score=None,fluency_valid=True,consistency_valid=False,
            reasons=['missing_reference'],generation_prompt_count=1,generated_token_count=0,
            length_cap_no_continuation_count=0),provenance={'origin':'CPU_FIXTURE'})


class FakeShared:
    def __init__(self,out,callback):
        self.out=Path(out);self.callback=callback;self.tokenizer=None;self.calls=[]
        self.runtime_identity={'CPU_fixture_only':True,'actual_GPU':False}
        self.runtime_sha=digest(self.runtime_identity)
        immutable_write(self.out/'observer-identity.json',dict(identity=self.runtime_identity,
            identity_sha256=self.runtime_sha,CPU_fixture_only=True))

    def observe(self,records,endpoint,cohort=None,state_identity=None):
        from project.run_scripts.experiment_generation_eval.progress import GenerationProgress
        self.calls.append((endpoint,[r['occurrence_index'] for r in records],copy.deepcopy(state_identity)))
        progress=GenerationProgress(len(records),len(records),self.callback)
        progress.emit('start',force=True)
        rows=[]
        for record in records:
            rows.append(metric_row(record))
            progress.complete_case({'observations':[dict(continuation_token_count=0,model_forwards=0,
                full_prefix_token_work=0,physical_forward_calls=0,prefill_query_tokens=0,decode_query_tokens=0)]})
        progress.finish()
        return dict(rows=rows,work=dict(new_case_observations=len(rows),cached_case_observations=0,
            generation_forwards=0,full_prefix_token_work=0,physical_forward_calls=0,
            prefill_query_tokens=0,decode_query_tokens=0,seconds=0.))

    def read_observed(self,path):
        receipt=json.loads(Path(path).read_text())
        if receipt['summary']!=reduce_cases(receipt['rows']):raise AssertionError('Fixture reduction')
        return dict(receipt,rows_path=str(Path(path).resolve()),work=dict(new_case_observations=0,
            cached_case_observations=len(receipt['rows']),generation_forwards=0,full_prefix_token_work=0,seconds=0.))

    def subset(self,observed,records,endpoint,cohort=None):
        lookup={row['occurrence']:row for row in observed['rows']}
        rows=[copy.deepcopy(lookup[r['occurrence_index']]) for r in records]
        identity=dict(runtime=self.runtime_sha,state_sha256=observed['identity']['state_sha256'],
            endpoint=endpoint,cohort=cohort,ordered_occurrences=[r['occurrence'] for r in rows])
        return dict(identity=identity,identity_sha256=digest(identity),summary=reduce_cases(rows),rows=rows,
            rows_path='/CPU_FIXTURE/subsets/NOT_ACTUAL.json',work={'generation_forwards':0},
            RNG_restored=True,observer_no_mutation=True)


class CacheBridgeTests(unittest.TestCase):
    def test_api_binding_is_read_only_plan_not_actual_pass(self):
        api=b.api_binding()
        self.assertEqual(api['source_sha'],'199cfe5664355f6f1c9069c72396ec759bb25cec')
        self.assertEqual(api['package_tree'],b.PACKAGE_TREE)
        self.assertFalse(api['actual_GPU_qualification'])
        self.assertFalse(api['source_namespace_modified'])
        self.assertIn('NOT_SHARED_RUN_QUALIFICATION',api['actual_execution'])

    def test_real_shared_plan_is_frozen_separately_and_mutation_rejected(self):
        frozen,local=plan()
        with tempfile.TemporaryDirectory() as directory:
            binding=q.freeze_plan(frozen,local,directory)
            self.assertEqual(json.loads(Path(binding['shared_qualification_plan']['path']).read_text()),
                             local['shared_plan'])
            self.assertEqual(binding['shared_qualification_plan_sha256'],digest(local['shared_plan']))
        local['shared_plan']['requests'][0]['input_token_ids'].append(1)
        with self.assertRaisesRegex(RuntimeError,'FROZEN_COHORT_CHANGED'):q.validate_plan(frozen,local)

    def test_native_runtime_and_distribution_metadata_are_separate_exact_pins(self):
        binding=q.installed_native_binding()
        self.assertEqual(binding['versions']['torch'],'2.9.1+cu128')
        self.assertEqual(binding['distribution_versions']['torch'],'2.9.1')
        self.assertEqual(len(binding['files']),2)
        with patch.object(q.importlib.metadata,'version',side_effect=lambda n:'2.9.9' if n=='torch' else '4.57.1'):
            with self.assertRaisesRegex(RuntimeError,'DISTRIBUTION_PIN'):q.installed_native_binding()

    def test_active_removal_requires_observed_partial_survivor_shrink(self):
        requests=[dict(token_length=2,occurrence=i,prompt_index=0) for i in range(4)]
        trace=[dict(occurrence=i,prompt_index=0,step=s) for s in (0,1) for i in range(4)]
        self.assertFalse(b._batch_coverage(requests,trace,4)['active_row_removal'])
        trace += [dict(occurrence=i,prompt_index=0,step=2) for i in range(1,4)]
        self.assertEqual(b._batch_coverage(requests,trace,4),
                         {'actual_max_microbatch':4,'active_row_removal':True})

    def test_cpu_fixture_cannot_convert_to_shared_actual_gpu_proof(self):
        frozen,_=plan()
        with self.assertRaises((KeyError,RuntimeError)):
            b.shared_receipt_body({'schema':q.RECEIPT_SCHEMA,'status':'QUALIFIED_ACTUAL_GPU_ROUTE',
                                  'actual_GPU':False},{},frozen)

    def _fixture(self,root,arm='BASE_MEMIT'):
        observer=b.GenerationObserver.__new__(b.GenerationObserver)
        observer.out=Path(root)/arm;observer.arm=arm
        observer.records=[dict(record,occurrence_index=i) for i,record in enumerate(records(),1)]
        observer.by_case={r['case_id']:r for r in observer.records}
        observer.cohort_sha=digest(observer.records);observer.cold_state={'original_fullcold':'CPU_FIXTURE'}
        observer.gen=dict(model_identity='CPU_FIXTURE_NO_MODEL',source_sha=b.SHARED_SOURCE)
        observer.config={'cold_W':{'3':'fixture'}};observer.view=types.SimpleNamespace(model=object(),sites=(3,))
        observer.assets=types.SimpleNamespace(sha='a'*64);observer.cache=Path(root)/'cache'
        observer.plan,_=plan();observer._w0=None;observer._receipts={};observer._progress_offset=None
        dummy=Path(root)/'CPU_FIXTURE_NOT_GPU_PROOF.json'
        if not dummy.exists():immutable_write(dummy,{'CPU_fixture_only':True,'actual_GPU':False})
        item=b.member(dummy)
        observer.repair=dict(old_complete_case_inventory=item,old_cold_observation_guard=item,
                            qualification_plan=item)
        observer.qualification_link=dict(qualification=item,shared_qualification_receipt_member=item,
            qualification_plan=item,qualification_plan_sha256=digest(observer.plan),
            shared_qualification_plan=item,shared_qualification_plan_sha256=observer.plan['shared_plan_sha256'],
            selected_route=q.SINGLETON,shared_selected_route=q.SHARED_ROUTES[q.SINGLETON],fixed_microbatch=1)
        observer.qualification_link_member=item;observer.private_compatibility_member=item;observer.runtime_aux_member=item
        observer._cold_weights=lambda:None;observer._native_signature=lambda:{'CPU_fixture':True}
        observer.tracker=types.SimpleNamespace(config_values={'job_id':'72001'},log=lambda values:True)
        observer.shared=FakeShared(observer.out/'generation-raw',observer._progress)
        return observer,item

    def test_protocol_missing_only_ready_progress_and_read_subset_no_new_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            observer,item=self._fixture(directory)
            entries=[dict(occurrence=i,original_raw_member=item,original_runtime_sha256='b'*64,
                original_generation_source_sha='c'*40,original_route=q.REFERENCE) for i in range(1,397)]
            manifest={'identity':{'original_entries':entries},'identity_sha256':digest(entries)}
            def old_case(inventory,ordinal,record,**kwargs):
                return dict(raw={'observations':[{'continuation_token_count':0}]},row=metric_row(record))
            patches=[patch('project.run_scripts.gptj_native_baselines.generation_cache_reuse.read_member',return_value=({},item)),
                patch('project.run_scripts.gptj_native_baselines.generation_cache_reuse.build_shared_compatibility',
                      return_value={'manifest':manifest,'member':item}),
                patch('project.run_scripts.gptj_native_baselines.generation_cache_reuse.read_reusable_case',side_effect=old_case),
                patch('project.run_scripts.experiment_generation_eval.observer.model_signature',return_value={'CPU_fixture':True}),
                patch.object(b,'verified_endpoint_row',return_value={'CPU_fixture_only':True})]
            for mock in patches:mock.start()
            try:
                full=observer.load_W0()
                self.assertEqual(observer.shared.calls[0][1],list(range(397,2001)))
                self.assertEqual(observer.shared.calls[0][2],observer.cold_state)
                self.assertEqual(full['summary']['planned_count'],2000)
                self.assertEqual(full['work']['completed_prompts'],2000)
                journal=Path(full['generation_progress']['path'])
                progress=[json.loads(line) for line in journal.read_text().splitlines()]
                self.assertEqual(progress[0]['generation_progress/reused_cases'],396)
                self.assertEqual(progress[-1]['generation_progress/completed_cases'],2000)
                self.assertEqual(progress[-1]['generation_progress/new_cases'],1604)
                self.assertEqual(progress[-1]['generation_progress/physical_forward_calls'],0)
                before=b.member(journal)
                observer._progress_offset=None
                edited=dict(progress[-1],phase='generation_evaluation')
                for key in ('completed_cases','total_cases','new_cases'):
                    edited['generation_progress/'+key]=100
                edited['generation_progress/reused_cases']=0
                for key in ('completed_prompts','total_prompts'):edited['generation_progress/'+key]=100
                observer._progress(edited)
                self.assertEqual(b.member(journal),before)
                peer,_=self._fixture(directory,'CAKE')
                reused=peer.load_W0()
                self.assertFalse(peer.shared.calls)
                self.assertEqual(reused['reused_complete_READY']['status'],'REUSED_COMPLETE_READY')
                self.assertEqual(reused['reused_complete_READY']['new_generation'],0)
                self.assertNotIn('generation_progress',reused)
                selected=observer.subset_receipt(full,observer.records[:100])
                self.assertEqual(selected['summary']['planned_count'],100)
                self.assertEqual(selected['work']['generation_forwards'],0)
                self.assertEqual(len(observer.shared.calls),1)
            finally:
                for mock in reversed(patches):mock.stop()

    def test_nonowner_without_ready_fails_without_polling_or_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            observer,_=self._fixture(directory,'CAKE')
            with self.assertRaisesRegex(RuntimeError,'W0_NOT_READY_NO_POLL'):observer.load_W0()
            self.assertFalse(observer.shared.calls)


if __name__=='__main__':unittest.main()
