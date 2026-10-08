"""CPU target-only collector guards/accounting, without scheduler calls."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import native_w20_collect as w20_collect


class Tests(unittest.TestCase):
    def test_full20_three_native_histories_flat_actual_receipt_and_saved_raw(self):
        for arm in w20_collect.ARMS:
            with self.subTest(arm=arm):self._review_full20_fixture(arm)

    def _review_full20_fixture(self,arm):
        """RPN heavy seams are mocked; native scalar guards and raw reducer run.

        The only verifier override explicitly allows this CPU fake-model actual
        receipt. Production still requires native GPU qualification unchanged.
        No scientific fit, edited model, or pretrained/GPU observation occurs.
        """
        from project.run_scripts.jlz_realized_writer_sequential.review_completed import Reader
        from project.run_scripts.experiment_generation_eval import kv_qualification as qualification
        from project.run_scripts.experiment_generation_eval.common import immutable_write, digest
        from project.run_scripts.experiment_generation_eval.compatibility import member
        from project.run_scripts.experiment_generation_eval.observer import GenerationObserver
        from project.run_scripts.experiment_generation_eval.test_kv_generator import CacheModel, Tokenizer, IDENTITY
        from project.run_scripts.experiment_generation_eval.test_observer import FakeAssets
        from .run import generation_receipt
        from .collect import EXPECTED

        records=[dict(case_id=index,occurrence_index=index+1,generation_prompts=['1 2'],
            requested_rewrite=dict(relation_id='r',target_new={'str':'new','id':'t'})) for index in range(2000)]
        native=EXPECTED[arm]
        layers=(13,14,15,16,17)
        cold=dict(W={str(layer):'fixture-cold-weight' for layer in layers},
            H={str(layer):w20_collect.zero_history_hash() for layer in layers} if arm == 'CAKE' else {})
        states=[cold]+[dict(W={str(layer):'fixture-batch-'+str(number)+'-weight' for layer in layers},
            H={str(layer):'fixture-batch-'+str(number)+'-history' for layer in layers}
                if arm in ('BASE_ALPHAEDIT','CAKE') else {}) for number in range(1,21)]
        physical=dict(W=states[-1]['W'],H={})
        hparams=dict(layers=list(layers),v_lr=.5,v_num_grad_steps=20,v_loss_layer=47,
            v_weight_decay=.5,clamp_norm_factor=.75,kl_factor=.0625)
        hparams.update(mom2_update_weight=20000) if arm == 'BASE_MEMIT' else hparams.update(L2=10 if arm == 'BASE_ALPHAEDIT' else 40)
        source='a'*40
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp);out=attempt/arm;out.mkdir()
            model,tok,assets=CacheModel(),Tokenizer(),FakeAssets()
            plan=qualification.build_qualification_plan(tok,records,model_identity=IDENTITY)
            immutable_write(out/'qualification-plan.json',plan)
            actual=qualification.run_qualification(model,tok,assets,plan,out=out/'qualification')
            generation=dict(schema='counterfact-cake-generation-metrics-v1',
                profile='cf-cake-prompt-inclusive-total100-eos-corrected-v1',eval_seed=20261007,
                model_identity=IDENTITY,generation_source_sha='fixed-generation-source',
                reference_assets_sha256=assets.sha,qualification_plan_member=member(out/'qualification-plan.json'),
                qualification_plan_sha256=digest(plan),schedule=w20_collect.SCHEDULE,
                source_identity='fixed-qualified-source')
            config=dict(task_id=w20_collect.TASK,instruction_id=w20_collect.NONCE,cold_W=cold['W'],
                packs=[dict(ids=list(range(index*100,(index+1)*100))) for index in range(20)],generation=generation)
            binding=dict(plan=generation['qualification_plan_member'],plan_sha256=digest(plan),actual=actual['member'],
                selected_route=actual['selected_route'],fixed_microbatch=actual['fixed_microbatch'],
                route_results=actual['route_results'],qualification_reused=False,
                technical_qualification_not_scientific_generation=True,generation_schedule=w20_collect.SCHEDULE)
            immutable_write(out/'generation-qualification-binding.json',binding)
            observer_config=dict(generation,generation_route=actual['selected_route'],
                generation_microbatch=actual['fixed_microbatch'],qualification_receipt_member=actual['member'],
                qualification_allow_cpu_fixture=True)
            observer=GenerationObserver(model,tok,assets,observer_config,out/'generation-W20-raw')
            observed=observer.observe(records,'W20',cohort='ALL_SEEN',state_identity=physical)
            flat=generation_receipt(observed)
            immutable_write(out/'runtime.json',dict(cold_state=cold,source=source,config=digest(config),arm=arm,
                cold_history_zero_verified=True,generation_schedule=w20_collect.SCHEDULE))
            for number in range(1,21):
                native_receipt=dict(arm=arm,batch=number,requests=100,delta=native,hparams=hparams,
                    same_model_returned=True,native_has_history=arm != 'BASE_MEMIT',caller_history_appends=0,
                    cache_template=None,native_z_disk_cache=False,checkpoint_saved=False)
                if arm == 'BASE_ALPHAEDIT':native_receipt['Alpha_reset_cache']=number == 1
                immutable_write(out/f'batch-{number:02d}'/'commit.json',dict(task=w20_collect.TASK,arm=arm,batch=number,
                    case_ids=config['packs'][number-1]['ids'],source=source,config=digest(config),
                    before=states[number-1],after=states[number],native=native_receipt,native_after=states[number],
                    prune_applied=False,terminal_transform=dict(prune_applied=False),
                    observer_no_mutation=True,native_counts=native,pre={},post={},post_current={},seconds=0.0,
                    generation_schedule=w20_collect.SCHEDULE,generation_measured=number == 20,
                    generation=flat if number == 20 else None))
            immutable_write(out/'generation-W20.json',dict(generation_schedule=w20_collect.SCHEDULE,
                completion_verified=True,generation_measured_at_edits=2000,observation=flat,physical_state=physical,
                qualification_receipt=actual['member']))
            immutable_write(out/'terminal.json',dict(status='COMPLETED',source=source,config=digest(config),commits=20,
                completed_batches=20,edits=2000,state=states[-1],native_counts={key:value*20 for key,value in native.items()},
                generation_schedule=w20_collect.SCHEDULE,generation_W20_complete=True,terminal_transforms=0,
                prune_applied=False,program_seconds=0.0))
            original_verify=qualification.verify_actual_receipt
            def fake_actual_only(*args,**kwargs):
                return original_verify(*args,**dict(kwargs,allow_cpu_fixture=True))
            fake_current={kind:dict(denominator=count) for kind,count in dict(R=100,P=200,N=1000).items()}
            def endpoint(reader,folder,identities,ids,name,expected_state,seen):
                return dict(summary={},rows=[],seconds=0.0,reference_only=name == 'W0')
            seams=(patch.object(w20_collect,'endpoint',side_effect=endpoint),
                patch.object(w20_collect,'compare_summary'),patch.object(w20_collect,'reduce_rows',return_value=fake_current),
                patch.object(w20_collect,'_metric_rows',return_value=[]),patch.object(w20_collect,'_paired_rows',return_value=[]),
                patch.object(w20_collect,'active_flags',return_value={record['case_id']:True for record in records}),
                patch.object(qualification,'verify_actual_receipt',side_effect=fake_actual_only))
            from contextlib import ExitStack
            with ExitStack() as stack:
                for seam in seams:stack.enter_context(seam)
                review=w20_collect.review_arm(Reader(),attempt,config,dict(source_commit=source),arm,[],records)
                self.assertEqual(review['scientific_status'],'COMPLETED_VALIDATED')
                self.assertEqual(review['commits'],20)
                self.assertEqual(review['requests'],2000)
                self.assertEqual(review['state_links'],19)
                self.assertTrue(review['generation_W20_available'])
                self.assertEqual(len(review['generation_rows']),1)
                self.assertEqual(review['generation_rows'][0]['planned_count'],2000)
                self.assertEqual(review['qualification_actual_member'],actual['member'])
                self.assertFalse(review['earlier_generation_measured'])
                self.assertEqual(review['measured_native_counts'],{key:value*20 for key,value in native.items()})
                path=out/'generation-W20.json';wrong=json.loads(path.read_text())
                wrong['observation']['identity_sha256']='0'*64
                path.write_text(json.dumps(wrong))  # deliberate CPU fixture corruption only
                with self.assertRaisesRegex(RuntimeError,'TERMINAL_GENERATION_STATE'):
                    w20_collect.review_arm(Reader(),attempt,config,dict(source_commit=source),arm,[],records)
                # A forged raw observation remains rejected by the saved byte
                # member even if the outer terminal receipt is restored.
                wrong['observation']=flat;path.write_text(json.dumps(wrong))
                raw_path=Path(observed['rows'][0]['observation_path']);raw=json.loads(raw_path.read_text())
                raw['observations'][0]['text']='deliberately changed CPU fixture only'
                raw_path.write_text(json.dumps(raw))
                with self.assertRaisesRegex(ValueError,'BOUND_FILE_SHA'):
                    w20_collect.review_arm(Reader(),attempt,config,dict(source_commit=source),arm,[],records)

    def test_w20_subset_identity_windows_are_distinct_and_raw_tamper_is_rejected(self):
        from project.run_scripts.experiment_generation_eval.common import GenerationError
        from project.run_scripts.experiment_generation_eval.observer import GenerationObserver
        from project.run_scripts.experiment_generation_eval.test_observer import FakeAssets,record
        from project.run_scripts.experiment_generation_eval.test_generator import FakeModel,FakeTokenizer
        with tempfile.TemporaryDirectory() as tmp:
            model=FakeModel();records=[record(1),record(2,501)]
            observer=GenerationObserver(model,FakeTokenizer(),FakeAssets(),
                dict(model_identity=dict(model='CPUfixture'),generation_source_sha='fixed-source'),Path(tmp))
            full=observer.observe(records,'W20','ALL_SEEN_FIXTURE',dict(W='fixture-W20',H={}))
            first=observer.subset(full,records[:1],'W20_FIRST_FIXTURE','FIRST_FIXTURE')
            last=observer.subset(full,records[1:],'W20_CURRENT_FIXTURE','CURRENT_FIXTURE')
            repeated=observer.subset(full,records[1:],'W20_CURRENT_FIXTURE','CURRENT_FIXTURE')
            self.assertNotEqual(first['rows_path'],last['rows_path'])
            self.assertEqual(last['rows_path'],repeated['rows_path'])
            self.assertEqual(Path(first['rows_path']).stem,first['identity_sha256'])
            self.assertEqual(Path(last['rows_path']).stem,last['identity_sha256'])
            self.assertEqual(len(model.calls),2) # subsets create no model forward.
            raw_path=Path(full['rows'][0]['observation_path']);raw=json.loads(raw_path.read_text())
            raw['observations'][0]['text']='forged CPU fixture only';raw_path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(GenerationError,'OBSERVATION_BYTES_IDENTITY'):
                observer.read_observed(full['rows_path'])

    def test_compact_errors_do_not_publish_raw_exception_strings(self):
        self.assertEqual(w20_collect.compact_error_code(RuntimeError('W20_COLLECT_TECHNICAL_TAG')),
                         'W20_COLLECT_TECHNICAL_TAG')
        self.assertEqual(w20_collect.compact_error_code(ValueError('private prompt fixture 123')),
                         'UNCLASSIFIED_REVIEW_ERROR')

    def test_earlier_generation_measurement_or_phase_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            commits=[dict(batch=1,generation_schedule=w20_collect.SCHEDULE,generation_measured=False,generation=None)]
            w20_collect.require_only_terminal_generation(out,commits)
            commits[0]['generation']={'fake':'W0'}
            with self.assertRaisesRegex(RuntimeError,'GENERATION_SCHEDULE'):
                w20_collect.require_only_terminal_generation(out,commits)
            commits[0]['generation']=None
            w20_collect.write(out/'generation-work'/'W5_POST.json',{})
            with self.assertRaisesRegex(RuntimeError,'NO_EARLY_GENERATION_PHASE'):
                w20_collect.require_only_terminal_generation(out,commits)

    def test_no_shared_W0_generation_dependency_or_fake_W20_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            w20_collect.require_only_terminal_generation(out,[])
            w20_collect.write(out/'generation-W0.json',{})
            with self.assertRaisesRegex(RuntimeError,'NO_W0_GENERATION'):
                w20_collect.require_only_terminal_generation(out,[])

    def test_target_only_accounting_query_exactthree_and_no_step_doublecount(self):
        from project.run_scripts.jlz_realized_writer_sequential.review_completed import Reader
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp);jobs=dict(zip(w20_collect.ARMS,('1','2','3')))
            w20_collect.write(attempt/'submission.json',dict(instruction_id=w20_collect.NONCE,
                task_id=w20_collect.TASK,source_commit='source',jobs=dict(jobs,collector='4')))
            calls=[]
            def runner(argv,**kwargs):
                calls.append(argv)
                text='\n'.join('|'.join((job,w20_collect.TASK+'-'+arm,'fixture','COMPLETED','0:0','12','gres/gpu=1'))
                    for arm,job in jobs.items())
                return SimpleNamespace(returncode=0,stdout=text)
            result=w20_collect.allocation_once(Reader(),attempt,dict(source_commit='source',owner='fixture'),runner=runner,owner='fixture')
            self.assertEqual(result['status'],'RECORDED')
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][calls[0].index('-j')+1],'1,2,3')
            self.assertEqual(len(result['records']),3)
            self.assertEqual(sum(row['allocated_GPU_seconds'] for row in result['records']),36)
            self.assertTrue(all(row['child_steps_excluded'] for row in result['records']))

    def test_missing_runtime_is_explicit_partial_no_invented_generation_zero(self):
        from project.run_scripts.jlz_realized_writer_sequential.review_completed import Reader
        records=[dict(case_id=i) for i in range(2000)]
        config=dict(packs=[dict(ids=list(range(index*100,(index+1)*100))) for index in range(20)])
        with tempfile.TemporaryDirectory() as tmp:
            result=w20_collect.review_arm(Reader(),Path(tmp),config,{},'BASE_MEMIT',[],records)
            self.assertEqual(result['scientific_status'],'PARTIAL_OR_NOT_VERIFIED')
            self.assertFalse(result['generation_W20_available'])
            self.assertEqual(result['generation_rows'],[])
            self.assertNotIn('ngram_entropy',result)
            self.assertEqual(result['missing'],['RUNTIME_NOT_RECORDED'])


if __name__=='__main__':unittest.main()
