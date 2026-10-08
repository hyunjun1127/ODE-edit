"""Local JSON/source fixture tests, NOT actual GPU qualification evidence."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from official.evaluation import reduce as factual_reduce
from official.experiments.prepare import METHODS, ROOT, digest, file_sha, read, write_new
from official.runners.server2 import collect
from official.runners.server2 import qualification_input as bridge


class QualificationInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_temp = tempfile.TemporaryDirectory()
        cls.old_source = Path(cls.source_temp.name)/'old-source'
        cls.new_source = Path(cls.source_temp.name)/'consumer-source'
        shutil.copytree(bridge.PRODUCER_ATTEMPT/'source/official',cls.old_source,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(ROOT,cls.new_source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))

    @classmethod
    def tearDownClass(cls):
        cls.source_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.attempt = Path(self.temp.name)/'CPU_FIXTURE_producer'
        self.attempt.mkdir()
        shutil.copytree(self.old_source,self.attempt/'source/official')
        self.old = read(bridge.PRODUCER_ATTEMPT/'manifest.json')
        self.consumer = deepcopy(self.old)
        self.consumer.update(code_commit='f'*40,official_tree_sha256='e'*40,
            assets_identity_sha256='NEW_SOURCE_PROVENANCE_ONLY_ASSETS_ID',
            _test_source_root=str(self.new_source))
        self.consumer['source_members'] = self.source_members(self.new_source)
        self.old_manifest = self.attempt/'manifest.json'
        write_new(self.old_manifest,self.old)
        (self.attempt/'source.tar').write_bytes(b'TEST_ONLY_CPU_SOURCE_ARCHIVE_NOT_A_GPU_ATTEMPT')
        self.lock = dict(code_commit=bridge.PRODUCER_COMMIT,official_tree_sha256=bridge.PRODUCER_TREE,
            manifest=collect.member(self.old_manifest),archive=collect.member(self.attempt/'source.tar'))
        write_new(self.attempt/'execution.lock.json',self.lock)
        self.submission = dict(status='SUBMISSION_HANDOFF',stage='qualification',
            source=dict(code_commit=bridge.PRODUCER_COMMIT,official_tree_sha256=bridge.PRODUCER_TREE),
            base_manifest_sha256=self.old['base_manifest_sha256'],
            lock=collect.member(self.attempt/'execution.lock.json'),
            jobs={method:str(90000+index) for index,method in enumerate([*METHODS,'collector'])})
        write_new(self.attempt/'submission.json',self.submission)
        self.frozen = self.plan()

    @staticmethod
    def source_members(root):
        return {str(path.relative_to(root)):file_sha(path) for path in root.rglob('*')
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc'}

    def plan(self, consumer=None, *, root=None):
        return bridge.plan(self.attempt,self.consumer if consumer is None else consumer,
            test_only_cpu=True,source_root=self.new_source if root is None else root)

    def owner_proof(self,method):
        plan = self.old['native_parity_plans']['cf'][method]
        source = plan['source']
        layers = list(dict.fromkeys((source['native_hparams']['layers'][0],source['native_hparams']['layers'][-1])))
        affine = lambda width:dict(shape=[2,width],dtype='float32',max_abs_error=0.,
            atol=plan['tolerance']['affine_readout']['atol'],rtol=plan['tolerance']['affine_readout']['rtol'],close=True)
        return dict(schema='official-server2-first-trajectory-native-parity-v1',
            status='PASS_ACTUAL_OWNER_FORMULA_PARITY',actual_GPU=True,method=method,dataset='cf',
            plan_sha256=plan['plan_sha256'],code_commit=source['code_commit'],
            official_tree_sha256=source['official_tree_sha256'],
            factual_source_sha256=source['factual_sha256'],native_hparams_sha256=source['hparams_sha256'],
            candidate_nll_close=True,candidate_nll_abs_error=0.,candidate_token_prefix_exact=True,
            token_predictions_exact=True,subject_lookup_exact=True,observer_no_mutation=True,RNG_restored=True,
            extra_LM_forward_calls=0,target_fit_calls=0,optimizer_calls=0,writer_calls=0,
            existing_factual_forward_calls=1,native_candidate_identity_sha256='a'*64,
            fc_out={f'transformer.h.{layer}.mlp.fc_out':affine(4096) for layer in layers},
            readout27=affine(50400),block_output_schemas={f'transformer.h.{layer}':
                dict(container='Tensor',hidden_shape=[2,4096]) for layer in [*layers,27]},
            independent_public_native_evaluator=plan['independent_public_native_evaluator'],
            independent_oracle_PASS=False,bitwise_full_evaluator_claim=False,scientific_quality_gate=False,
            evidence_scope=plan['evidence_scope'],fixture='CPU_JSON_ONLY_NO_REAL_MODEL_OR_GPU')

    def factual(self,rows):
        from official.evaluation.factual import SCHEMA,TOKENIZATION
        cases = [dict(case_id=row['case_id'],occurrence_index=row['occurrence_index'],
            rewrite_prompts_probs=[dict(target_new=1.,target_true=2.)],
            paraphrase_prompts_probs=[dict(target_new=1.,target_true=2.)]*len(row['paraphrase_prompts']),
            neighborhood_prompts_probs=[dict(target_new=2.,target_true=1.)]*len(row['neighborhood_prompts']))
            for row in rows]
        summary = factual_reduce.counterfact(cases)
        summary['Score_AlphaEdit_display'] = factual_reduce.harmonic([
            round(summary[label],2) for label in ('Efficacy','Generalization','Specificity')])
        identity = dict(schema=SCHEMA,dataset='cf',tokenization=TOKENIZATION,
            cohort_sha256=digest({'CPU_fixture':cases}),
            ordered_occurrences=[row['occurrence_index'] for row in rows],
            external_identity=collect.expected_factual_external_identity(self.old,'cf'),
            padding='RIGHT_EXPLICIT_ATTENTION_MASK',use_cache=False)
        return dict(cases=cases,summary=summary,identity=identity,identity_sha256=digest(identity),
            observer_no_mutation=True,RNG_restored=True,fixture='CPU_JSON_ONLY_NO_REAL_MODEL_OR_GPU')

    def install_fixture_proof(self):
        rows = read(self.old['streams']['cf']['member']['path'])[200:300]
        entries = {}
        for method in METHODS:
            out = self.attempt/method
            out.mkdir()
            write_new(out/'owner.json',self.owner_proof(method))
            for name in ('continuous.json','resumed.json'):
                write_new(out/name,self.factual(rows))
            identity = self.old['checkpoint_identities']['cf'][method]
            cp = out/'checkpoints';cp.mkdir()
            (cp/'batch-03-fixture.pt').write_bytes(b'CPU_NOT_A_TENSOR_CHECKPOINT')
            final = dict(batch=3,file='batch-03-fixture.pt',sha256='b'*64,
                identity_sha256=digest(identity),final_W20=False)
            durable = dict(batch=2,file='batch-02-previously-durable.pt',sha256='c'*64,
                identity_sha256=digest(identity),final_W20=False)
            write_new(cp/'latest.json',final)
            native = [dict(batch=batch,method=method,requests=100,
                native_source_sha256=self.old['native_source_sha256'][method][0],
                history_appends_expected=collect.HISTORY_COUNTS[method]//20,
                **({'checkpoint':durable} if index==1 else {})) for index,batch in enumerate((1,2,3,3))]
            value = dict(status='PASS_ACTUAL_QUALIFICATION',actual_GPU=True,model='gptj',dataset='cf',
                method=method,code_commit=self.old['code_commit'],official_tree_sha256=self.old['official_tree_sha256'],
                manifest_sha256=self.old['base_manifest_sha256'],checkpoint_identity=identity,
                continuous_batches=3,resume_after_batch=2,resumed_batches=[3],weights_equal=True,
                history_equal=True,contexts_equal=True,rng_equal=True,metrics_equal=True,
                durable_B2=durable,checkpoint=final,native_commits=native,
                actual_native_batch_calls=4,actual_native_request_applications=400,
                native_owner_formula_parity=collect.member(out/'owner.json'),
                continuous_metric=collect.member(out/'continuous.json'),resumed_metric=collect.member(out/'resumed.json'),
                fixture='CPU_JSON_ONLY_NO_REAL_MODEL_OR_GPU')
            write_new(out/'qualification.json',value)
            entries[method] = value
        collector = self.attempt/'collector';collector.mkdir()
        aggregate = collect.aggregate_qualification(entries,self.old)
        aggregate['fixture'] = 'CPU_JSON_ONLY_NO_REAL_MODEL_OR_GPU'
        write_new(collector/'qualification.json',aggregate)
        result = dict(status='SCIENTIFIC_COMPLETE',scientific_complete=True,stage='qualification',
            code_commit=self.old['code_commit'],official_tree_sha256=self.old['official_tree_sha256'],
            manifest_sha256=self.old['base_manifest_sha256'],failures={},scheduler_failures={},
            unobserved_scientific_accounting=[],accounting=dict(exact_requested_jobs=self.submission['jobs'],
                jobs={method:dict(state='COMPLETED') for method in METHODS}))
        write_new(collector/'result.json',result)
        terminal = dict(status='COLLECTOR_COMPLETE',scientific_complete=True,source=self.old['code_commit'],
            official_tree_sha256=self.old['official_tree_sha256'],result=collect.member(collector/'result.json'),
            final_atomic_terminal=True)
        write_new(collector/'terminal.json',terminal)

    @staticmethod
    def replace(path,value):
        path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')

    def test_preregistered_physical_same_source_projection_separate_asset_provenance(self):
        self.assertEqual(self.frozen['status'],'PREREGISTERED_PROVENANCE_PRESERVING_NATIVE_COMPATIBILITY')
        self.assertFalse(self.frozen['actual_GPU'])
        self.assertNotEqual(self.frozen['producer']['assets_identity_sha256'],
                            self.frozen['consumer']['assets_identity_sha256'])
        self.assertEqual(set(self.frozen['computational_projection']['critical_run_AST']),
                         set(bridge.RUN_FUNCTIONS))
        self.assertEqual(self.frozen['computational_projection']['display_exception']['only_changed_field'],
                         'Score_AlphaEdit_display')
        self.assertFalse(self.frozen['old_receipts_relabelled'])

    def test_absent_proof_is_pending_not_actual_and_no_model_or_job_query(self):
        result = bridge.verify(self.frozen,self.consumer)
        self.assertEqual(result['status'],'INPUT_PENDING_NOT_ACTUAL_QUALIFIED')
        self.assertFalse(result['actual_GPU'])
        self.assertEqual(result['consumer_binding']['checkpoint_tensor_loads'],0)
        self.assertFalse(result['recurring_monitor'])

    def test_full_CPU_fixture_validation_never_actual_promotion_or_new_source_relabel(self):
        self.install_fixture_proof()
        result = bridge.verify(self.frozen,self.consumer)
        self.assertEqual(result['status'],'VERIFIED_CPU_FIXTURE_INPUT_NOT_ACTUAL')
        self.assertFalse(result['actual_GPU'])
        self.assertEqual(set(result['consumer_binding']['methods']),set(METHODS))
        self.assertEqual(result['producer_proof']['code_commit'],bridge.PRODUCER_COMMIT)
        self.assertEqual(result['consumer_binding']['consumer_code_commit'],'f'*40)
        self.assertEqual(result['consumer_binding']['CF_original_evaluator_parity'],
                         'NOT_ESTABLISHED_BY_THIS_INPUT')
        changed = deepcopy(self.frozen)
        changed['test_only_cpu_fixture'] = False
        changed['plan_sha256'] = digest({key:item for key,item in changed.items() if key!='plan_sha256'})
        with self.assertRaisesRegex(ValueError,'ONLY_EXACT_PRODUCER'):
            bridge.verify(changed,self.consumer)

    def test_semantic_asset_order_runtime_hparams_and_source_tamper_failclosed(self):
        for field in ('slot','model','runtime','order','config','source'):
            changed = deepcopy(self.consumer)
            if field=='slot':changed['assets']['projector']['BLUE_physical_slots']=[1,5]
            elif field=='model':changed['model_assets'][3]['sha256']='0'*64
            elif field=='runtime':changed['runtime']['torch']='ANOTHER_VERSION'
            elif field=='order':changed['streams']['cf']['lock']['batches'][0]['ordered_case_ids_sha256']='0'*64
            elif field=='config':changed['checkpoint_identities']['cf']['MEMIT']['config_sha256']='0'*64
            else:changed['source_members']['baselines/registry.py']='0'*64
            with self.assertRaises(ValueError):self.plan(changed)

    def test_missing_extra_or_new_computational_member_not_optional(self):
        for key,action in [('baselines/registry.py','missing'),('baselines/new_math.py','extra')]:
            changed = deepcopy(self.consumer)
            if action=='missing':del changed['source_members'][key]
            else:changed['source_members'][key]='0'*64
            with self.assertRaisesRegex(ValueError,'MEMBER_SET_CHANGED'):
                self.plan(changed)

    def test_actual_model_and_stream_load_paths_bound_to_declared_members(self):
        for field in ('model_snapshot','stream'):
            changed = deepcopy(self.consumer)
            if field == 'model_snapshot':changed['model_snapshot']='/unrelated/model-snapshot'
            else:changed['streams']['cf']['path']='/unrelated/first2000.json'
            with self.assertRaisesRegex(ValueError,'LOAD_PATH_MEMBER_BINDING'):
                self.plan(changed)

    def test_critical_helpers_imports_constants_and_shadowing_failclosed(self):
        alternate = Path(self.temp.name)/'changed-bindings'
        shutil.copytree(self.new_source,alternate)
        run = alternate/'runners/server2/run.py'
        original = run.read_text()
        for text, expected in (
            (original.replace("'stream_sha256'", "'other_stream_sha256'"),'CRITICAL_RUN_AST_CHANGED'),
            (original.replace('import torch\n','import json as torch\n'),'IMPORT_CONSTANT_BINDINGS_CHANGED'),
            (original.replace("INSTRUCTION = 'USER-OFFICIAL-BASELINES-20261008-R1'",
                              "INSTRUCTION = 'CHANGED_INSTRUCTION'"),'IMPORT_CONSTANT_BINDINGS_CHANGED'),
            (original+'\nload_model = lambda manifest: None\n','ASSIGNMENT_SHADOWS_DEFINITION'),
            (original+'\nUNDECLARED_EXTRA_CONSTANT = 3\n','IMPORT_CONSTANT_BINDINGS_CHANGED')):
            run.write_text(text)
            changed = deepcopy(self.consumer);changed['source_members']=self.source_members(alternate)
            with self.assertRaisesRegex(ValueError,expected):
                self.plan(changed,root=alternate)
        run.write_text(original)

    def test_exact_native_run_AST_and_display_exception_only(self):
        alternate = Path(self.temp.name)/'changed_source'
        shutil.copytree(self.new_source,alternate)
        run = alternate/'runners/server2/run.py'
        run.write_text(run.read_text().replace("'ONE_ALLOCATED_GPU_REQUIRED'","'CHANGED_NATIVE_GUARD'"))
        changed = deepcopy(self.consumer);changed['source_members']=self.source_members(alternate)
        with self.assertRaisesRegex(ValueError,'CRITICAL_RUN_AST_CHANGED'):
            self.plan(changed,root=alternate)
        shutil.copy2(self.new_source/'runners/server2/run.py',run)
        reducer = alternate/'evaluation/reduce.py';reducer.write_text(reducer.read_text().replace('requests=len(cases)','requests=len(cases)+1'))
        changed['source_members']=self.source_members(alternate)
        with self.assertRaisesRegex(ValueError,'ONLY_DECLARED_DISPLAY'):
            self.plan(changed,root=alternate)

    def test_actual_resume_flags_call_count_H_and_source_required(self):
        self.install_fixture_proof()
        path = self.attempt/'MEMIT/qualification.json'
        original = read(path)
        for key in ('contexts_equal','actual_native_request_applications','native_commits','code_commit'):
            changed = deepcopy(original)
            if key=='contexts_equal':changed[key]=False
            elif key=='actual_native_request_applications':changed[key]=300
            elif key=='code_commit':changed[key]='f'*40
            else:changed[key][1]['history_appends_expected']=1
            self.replace(path,changed)
            with self.assertRaises(ValueError):bridge.verify(self.frozen,self.consumer)
            self.replace(path,original)

    def test_percase_continuous_resumed_raw_and_member_integrity_are_required(self):
        self.install_fixture_proof()
        out = self.attempt/'MEMIT'
        proof = read(out/'qualification.json')
        raw = read(proof['resumed_metric']['path'])
        raw['cases'][0]['rewrite_prompts_probs'][0]['target_new'] = 3.
        raw['summary'] = factual_reduce.counterfact(raw['cases'])
        raw['summary']['Score_AlphaEdit_display'] = factual_reduce.harmonic([
            round(raw['summary'][key],2) for key in ('Efficacy','Generalization','Specificity')])
        self.replace(Path(proof['resumed_metric']['path']),raw)
        with self.assertRaisesRegex(ValueError,'SHA_SIZE'):bridge.verify(self.frozen,self.consumer)
        proof['resumed_metric']=collect.member(proof['resumed_metric']['path'])
        self.replace(out/'qualification.json',proof)
        with self.assertRaisesRegex(ValueError,'RAW_EQUALITY'):bridge.verify(self.frozen,self.consumer)

    def test_owner_formula_is_not_native_original_oracle(self):
        self.install_fixture_proof()
        out = self.attempt/'MEMIT';proof = read(out/'qualification.json')
        owner = read(proof['native_owner_formula_parity']['path'])
        owner['independent_oracle_PASS'] = True
        self.replace(out/'owner.json',owner)
        proof['native_owner_formula_parity']=collect.member(out/'owner.json')
        self.replace(out/'qualification.json',proof)
        with self.assertRaisesRegex(ValueError,'NOT_INDEPENDENT'):bridge.verify(self.frozen,self.consumer)

    def test_collector_incomplete_source_or_terminal_not_promoted(self):
        self.install_fixture_proof()
        path = self.attempt/'collector/result.json';original=read(path)
        for field in ('scientific_complete','code_commit','accounting'):
            changed=deepcopy(original)
            if field=='scientific_complete':changed[field]=False
            elif field=='code_commit':changed[field]='f'*40
            else:changed[field]['jobs']['MEMIT']['state']='FAILED'
            self.replace(path,changed)
            with self.assertRaises(ValueError):bridge.verify(self.frozen,self.consumer)
            self.replace(path,original)
        terminal=self.attempt/'collector/terminal.json';value=read(terminal)
        value['scientific_complete']=False;self.replace(terminal,value)
        with self.assertRaisesRegex(ValueError,'ATOMIC_TERMINAL'):bridge.verify(self.frozen,self.consumer)

    def test_plan_member_producer_and_expected_path_tamper_rejected(self):
        changed=deepcopy(self.frozen);changed['consumer']['code_commit']='0'*40
        with self.assertRaisesRegex(ValueError,'PLAN_SHA'):bridge.verify(changed,self.consumer)
        with self.assertRaisesRegex(ValueError,'EXPECTED_PROOF_PATH'):
            bridge.verify(self.frozen,self.consumer,Path(self.temp.name)/'other.json')
        self.old['model_revision']='0'*40;self.replace(self.old_manifest,self.old)
        with self.assertRaises(ValueError):bridge.verify(self.frozen,self.consumer)


if __name__=='__main__':
    unittest.main()
