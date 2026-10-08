"""CPU final observer identity/control fixtures; not actual GPU route PASS."""
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from . import generation_cache_bridge as cache
from . import generation_final_bridge as final
from .test_generation_bridge import records
from .test_generation_cache_qualification import plan
from project.run_scripts.experiment_generation_eval.common import (
    EVAL_SEED,PROFILE,SCHEMA,digest,immutable_write)
from project.run_scripts.experiment_generation_eval.compatibility import member
from .test_generation_tracking_shared import raw_summary


class FinalBridgeTests(unittest.TestCase):
    def test_final_progress_adapts_only_metadata_and_no_input_mutation(self):
        observer=final.GenerationObserver.__new__(final.GenerationObserver)
        payload=dict(phase='generation_evaluation',**{'generation_progress/total_cases':2000,
            'generation_progress/step':0})
        before=copy.deepcopy(payload)
        with patch.object(cache.GenerationObserver,'_progress') as parent:
            observer._progress(payload)
            parent.assert_called_once_with(dict(payload,phase='W20_generation'))
            for bad in (dict(payload,phase='W0_generation'),
                        dict(payload,**{'generation_progress/total_cases':100})):
                with self.assertRaisesRegex(RuntimeError,'PROGRESS_ENDPOINT'):
                    observer._progress(bad)
        self.assertEqual(payload,before)

    def test_actual_wrap_runner_receipt_reducer_route_link(self):
        from .generation_run import generation_receipt
        from .generation_collect import qualification_link, Reader
        private, _=plan()
        state={'W':'CPU_FIXTURE_W20','H':{}}
        observer=final.GenerationObserver.__new__(final.GenerationObserver)
        observer.plan=private;observer.final_only=True;observer._receipts={}
        observer.shared=types.SimpleNamespace(runtime_sha='runtime-fixture',
            runtime_identity={'CPU_fixture':True})
        identity=dict(endpoint='W20',runtime=observer.shared.runtime_sha,
            state_sha256=digest(state),ordered_occurrences=list(range(1,2001)))
        with tempfile.TemporaryDirectory(prefix='final-route-link-fixture-') as directory:
            root=Path(directory)
            proof=root/'proof.json';raw=root/'raw.json';runtime=root/'runtime.json';link=root/'link.json'
            for path in (proof,raw,runtime,link):
                immutable_write(path,{'CPU_fixture_only':True,'actual_GPU':False})
            observer.qualification_link=dict(qualification=member(proof),
                shared_qualification_receipt_member=member(proof),
                selected_route=cache.q.SINGLETON,fixed_microbatch=1)
            observer.qualification_link_member=member(link);observer.runtime_aux_member=member(runtime)
            observed=dict(summary=raw_summary(2000),rows=[{'CPU_fixture':True} for _ in range(2000)],
                identity=identity,identity_sha256=digest(identity),rows_path=str(raw),work={},
                RNG_restored=True,observer_no_mutation=True)
            wrapped=observer._wrap(observed,state,root/'endpoint')
            compact=generation_receipt(wrapped,records(),'W20',state,root/'endpoint')
            evidence=dict(actual_qualification=True,actual_member_sha256=member(proof)['sha256'],
                plan_sha256=digest(private),selected_route=cache.q.SINGLETON,fixed_microbatch=1)
            # Actual production functions, not a mocked route-link reader.
            qualification_link(Reader(),{'generation':{'repair':{
                'qualification_receipt_path':str(proof)}}},compact,evidence,root)
            self.assertEqual(compact['selected_route'],cache.q.SINGLETON)
            self.assertEqual(compact['fixed_microbatch'],1)
            from .generation_common import read
            saved=read(root/'endpoint/receipt.json')
            self.assertEqual(saved['selected_route'],cache.q.SINGLETON)
            self.assertNotIn('compatibility_manifest',saved)

    def test_ctor_keeps_qualification_binding_but_never_builds_old_w0_compatibility(self):
        private,cohort=plan()
        private['shared_source_sha']=cache.q.SHARED_SOURCE
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);private_path=root/'private-plan.json';shared_path=root/'shared-plan.json'
            cohort_path=root/'cohort.json';proof_path=root/'CPU_FIXTURE_ONLY.json'
            for path,value in ((private_path,private),(shared_path,cohort['shared_plan']),
                               (cohort_path,cohort),(proof_path,{'actual_GPU':False,'CPU_fixture':True})):
                immutable_write(path,value)
            link=dict(shared_qualification_receipt_member=member(proof_path),
                shared_selected_route='UNPADDED_KV_SINGLETON',fixed_microbatch=1)
            immutable_write(root/'qualification-link.json',link)
            config={'generation':dict(schema=SCHEMA,profile=PROFILE,eval_seed=EVAL_SEED,
                source_sha=cache.q.SHARED_SOURCE,W0_owner='BASE_MEMIT',evaluation_schedule=final.SCHEDULE,
                model_identity=private['model_identity'],reference_assets_sha256=private['reference_identity'],
                generation_assets={},W0_state_identity={'cold':'CPU_FIXTURE'},W0_cache=str(root/'DO_NOT_USE'),
                repair=dict(qualification_plan=member(private_path),qualification_cohort=member(cohort_path),
                    qualification_plan_sha256=digest(private),shared_qualification_plan=member(shared_path),
                    qualification_receipt_path=str(root/'private-actual.json')))}
            class SharedFixture:
                def __init__(self,*args,**kwargs):
                    self.runtime_identity={'CPU_fixture':True};self.runtime_sha=digest(self.runtime_identity)
            with patch('project.run_scripts.experiment_generation_eval.assets.load_assets',
                       return_value=types.SimpleNamespace(sha=private['reference_identity'])),\
                 patch('project.run_scripts.experiment_generation_eval.observer.GenerationObserver',SharedFixture),\
                 patch.object(cache.GenerationObserver,'_cold_weights'),\
                 patch.object(cache.GenerationObserver,'_verify_link'),\
                 patch('project.run_scripts.gptj_native_baselines.generation_cache_reuse.build_compatibility') as compat:
                observer=final.GenerationObserver(config,{'source_commit':'CPU_FIXTURE_SOURCE',
                    'config_sha256':'CPU_FIXTURE_CONFIG'},types.SimpleNamespace(model=object()),object(),
                    None,records(),root/'arm','BASE_MEMIT')
            compat.assert_not_called()
            self.assertFalse(hasattr(observer,'private_compatibility_member'))
            self.assertNotIn('compatibility_manifest',observer.runtime_aux)
            self.assertEqual(observer.runtime_aux['W0_generation'],'NOT_SCHEDULED')
            self.assertFalse((root/'DO_NOT_USE'/'READY.json').exists())
            for method in (observer.load_W0,lambda:observer.subset(None,None),
                           lambda:observer.subset_receipt(None,None)):
                with self.assertRaisesRegex(RuntimeError,'NOT_SCHEDULED'):method()

    def test_one_final_ordered_full_cohort_actual_state_and_no_intermediate_route(self):
        observer=final.GenerationObserver.__new__(final.GenerationObserver)
        observer.records=[dict(r,occurrence_index=i) for i,r in enumerate(records(),1)]
        observer.by_case={r['case_id']:r for r in observer.records};observer._final_attempted=False
        observer.engine=types.SimpleNamespace(next_batch=21,history=lambda:{})
        observer.view=object();physical={'W':'ACTUAL_CPU_FIXTURE_W20','H':{}}
        calls=[]
        def observe(selected,endpoint,cohort,state_identity):
            calls.append((endpoint,len(selected),copy.deepcopy(state_identity)))
            return {'identity':dict(ordered_occurrences=list(range(1,2001)),
                                   state_sha256=digest(state_identity))}
        observer.shared=types.SimpleNamespace(observe=observe)
        observer._wrap=lambda receipt,state,out:receipt
        with patch('project.run_scripts.gptj_cake_blue_prune_rect.metrics.state',return_value=physical):
            for endpoint,chosen in (('B1_PRE',observer.records[:100]),('W5',observer.records[:500]),
                                    ('W20',list(reversed(observer.records)))):
                with self.assertRaisesRegex(RuntimeError,'ONE_W20_FULL_FIRST2000'):
                    observer.endpoint(chosen,out='/CPU_FIXTURE',endpoint=endpoint,
                                      model_state=physical,cohort_label='ALL_SEEN')
            observer.engine.next_batch=20
            with self.assertRaisesRegex(RuntimeError,'AFTER_TWENTY_NATIVE_COMMITS'):
                observer.endpoint(observer.records,out='/CPU_FIXTURE',endpoint='W20',
                                  model_state=physical,cohort_label='ALL_SEEN')
            observer.engine.next_batch=21
            with self.assertRaisesRegex(RuntimeError,'ACTUAL_FINAL_STATE'):
                observer.endpoint(observer.records,out='/CPU_FIXTURE',endpoint='W20',
                                  model_state={'W':'wrong'},cohort_label='ALL_SEEN')
            observer.endpoint(observer.records,out='/CPU_FIXTURE',endpoint='W20',
                              model_state=physical,cohort_label='ALL_SEEN')
            with self.assertRaisesRegex(RuntimeError,'ONE_W20_FULL_FIRST2000'):
                observer.endpoint(observer.records,out='/CPU_FIXTURE',endpoint='W20',
                                  model_state=physical,cohort_label='ALL_SEEN')
        self.assertEqual(calls,[('W20',2000,physical)])


if __name__=='__main__':unittest.main()
