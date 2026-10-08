"""New native-three authority, immutable inputs, duplicate and cap2 fixtures."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import native_w20_common as common,native_w20_prepare as prepare,native_w20_submit as submit
from . import w20_common as protected_common

class Control(unittest.TestCase):
    def fixture(self,tmp):
        root=Path(tmp);source=root/'old';attempt=root/'new'
        plan=dict(model_identity={'model':'gpt2xl'},requests=[dict(prompt='LOCAL_FIXTURE')],actual_qualification=False)
        common.write(root/'plan.json',plan)
        six=(*common.ARMS,'ALPHAEDIT_BLUE','PRUNE','RECT')
        prior=dict(instruction_id='old',task_id='old',attempt=str(source),arms=list(six),
            source_configs={arm:dict(native={'algorithm':arm,
                'layers':[13,17] if arm=='ALPHAEDIT_BLUE' else [13,14,15,16,17],
                'v_lr':.5,'v_num_grad_steps':20,'mom2_update_weight':20000,
                'L2':40 if arm=='CAKE' else 10}) for arm in six},
            generation=dict(schema='counterfact-cake-generation-metrics-v1',
                profile='cf-cake-prompt-inclusive-total100-eos-corrected-v1',eval_seed=20261007,
                model_identity=plan['model_identity'],qualification_plan_member=common.member(root/'plan.json'),
                qualification_plan_sha256=common.digest(plan),qualification_receipt='/old/qualification.json',
                shared_W0_root='/old/W0',old_w0_reuse={'old':True},primary_arm='BASE_MEMIT'),
            cpu_preflight={'old':True},manual_retry_authority_member={'old':True},manual_recall_id='old',
            noCP=True,z_disk_cache=False,exact_resume='NOT_AVAILABLE',cold_W={'unchanged':True},
            runtime={'torch':'unchanged'},observer_identity={'unchanged':True},tracking={'unchanged':True},
            stream='/unchanged/counterfact.json',packs=[{'unchanged':True}])
        common.write(source/'config.json',prior);common.write(source/'execution.lock.json',{'source_commit':'a'*40})
        common.write(root/'transition.json',dict(instruction_id=common.NONCE,task_id=common.TASK,
            owner=dict(server='server1',session=common.SESSION),source_attempt=str(source),
            target_jobs={arm:dict(job=common.OLD_TARGET_IDS[arm],state='FAILED' if arm=='BASE_MEMIT'
                else 'CANCELLED') for arm in common.ARMS},
            protected_jobs_preserved=True,old_source_raw_preserved=True,snapshot_at='2026-10-08T20:00:00+09:00'))
        return root,source,attempt,prior,common.member(root/'transition.json')

    def test_fresh_authority_and_explicit_existing_schedule_inheritance(self):
        envelope,policy=common.authority()
        self.assertEqual(envelope['scope']['methods'],list(common.ARMS))
        self.assertEqual(common.ARMS,('BASE_MEMIT','BASE_ALPHAEDIT','CAKE'))
        self.assertEqual(policy['generation_schedule'],common.SCHEDULE)
        self.assertEqual(policy['instruction_id'],common.SCHEDULE_INSTRUCTION)
        self.assertNotEqual(policy['instruction_id'],common.NONCE)

    def test_transform_removes_generation_W0_without_native_science_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            local,source,attempt,prior,transition=self.fixture(tmp);before=copy.deepcopy(prior)
            with patch.object(common,'LOCAL',local):
                value=prepare.prepared_config(prior,attempt,source,prior['generation']['qualification_plan_member'],transition)
            self.assertEqual(prior,before)
            self.assertEqual(value['arms'],list(common.ARMS))
            self.assertEqual(set(value['source_configs']),set(common.ARMS))
            for arm in common.ARMS:self.assertEqual(value['source_configs'][arm],prior['source_configs'][arm])
            self.assertEqual(value['source_configs']['BASE_MEMIT']['native']['mom2_update_weight'],20000)
            self.assertEqual(value['source_configs']['BASE_ALPHAEDIT']['native']['L2'],10)
            self.assertEqual(value['source_configs']['CAKE']['native']['L2'],40)
            for field in ('cold_W','runtime','observer_identity','tracking','stream','packs'):
                self.assertEqual(value[field],prior[field])
            for field in ('shared_W0_root','old_w0_reuse','primary_arm'):
                self.assertNotIn(field,value['generation'])
            self.assertNotIn('cpu_preflight',value);self.assertNotIn('manual_retry_authority_member',value)
            self.assertFalse(value['W0_generation_required'])
            self.assertEqual(value['generation_endpoints'],[dict(endpoint='all_seen/post',state='W20',edits=2000,requests=2000)])
            self.assertEqual(value['generation']['qualification_plan_member'],prior['generation']['qualification_plan_member'])
            self.assertEqual(value['generation']['plan']['three_arm_case_observations_max'],6000)
            self.assertEqual(value['generation_schedule_authority'],common.SCHEDULE_INSTRUCTION)
            self.assertEqual(set(value['generation']['qualification_receipts']),set(common.ARMS))

    def test_W0_extra_method_or_intermediate_generation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            local,source,attempt,prior,transition=self.fixture(tmp)
            with patch.object(common,'LOCAL',local):
                value=prepare.prepared_config(prior,attempt,source,prior['generation']['qualification_plan_member'],transition)
                for mutation in ('W0','method','endpoint'):
                    changed=copy.deepcopy(value)
                    if mutation=='W0':changed['generation']['shared_W0_root']='/forbidden/W0'
                    if mutation=='method':changed['source_configs']['PRUNE']={}
                    if mutation=='endpoint':changed['generation_endpoints'].append({'state':'W5'})
                    with self.subTest(mutation=mutation),self.assertRaisesRegex(RuntimeError,'NATIVE_W20_'):
                        common.validate_config(changed)

    def test_profiles_do_not_override_or_bypass_each_other(self):
        self.assertEqual(protected_common.ARMS,('ALPHAEDIT_BLUE','PRUNE','RECT'))
        self.assertEqual(protected_common.TASK,'gpt2xl-blue-prune-rect-w20-generation')
        self.assertNotEqual(common.NONCE,protected_common.NONCE)
        self.assertNotEqual(common.SOURCE_ENV,protected_common.SOURCE_ENV)
        with tempfile.TemporaryDirectory() as tmp:
            local,source,attempt,prior,transition=self.fixture(tmp)
            with patch.object(common,'LOCAL',local):
                value=prepare.prepared_config(prior,attempt,source,prior['generation']['qualification_plan_member'],transition)
                with self.assertRaisesRegex(RuntimeError,'W20_CONFIG_THREE_ONLY'):
                    protected_common.validate_config(value)
                changed=copy.deepcopy(value);changed['instruction_id']=protected_common.NONCE
                with self.assertRaisesRegex(RuntimeError,'NATIVE_W20_CONFIG_EXACT_THREE'):
                    common.validate_config(changed)

    def test_prior_profile_is_history_but_started_or_partial_new_registration_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            local=Path(tmp)
            common.write(local/'old/config.json',dict(instruction_id=protected_common.NONCE))
            common.write(local/'old/submission.json',dict(jobs={}))
            with patch.object(common,'LOCAL',local):
                self.assertEqual(common.registered_attempts(),[])
                common.write(local/'new/config.json',dict(instruction_id=common.NONCE))
                common.write(local/'new/registration-pass-started.json',dict(automatic_retry=False))
                self.assertEqual(common.registered_attempts(),[local/'new'])
                common.write(local/'partial/config.json',dict(instruction_id=common.NONCE))
                common.write(local/'partial/submitted-BASE_MEMIT.json',dict(job='100'))
                self.assertEqual(set(common.registered_attempts()),{local/'new',local/'partial'})

    def test_exact_terminal_targets_and_protected_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            local,source,attempt,prior,row=self.fixture(tmp)
            self.assertEqual(set(common.transition_receipt(row,source)['target_jobs']),set(common.ARMS))
            for field,new in (('protected_jobs_preserved',False),('source_attempt','/wrong/attempt')):
                value=common.read(common.verify(row));value[field]=new
                common.write(local/(field+'.json'),value)
                with self.subTest(field=field),self.assertRaisesRegex(RuntimeError,'TRANSITION_SCOPE'):
                    common.transition_receipt(common.member(local/(field+'.json')),source)
            value=common.read(common.verify(row));value['target_jobs']['CAKE']['state']='RUNNING'
            common.write(local/'running.json',value)
            with self.assertRaisesRegex(RuntimeError,'TERMINAL'):
                common.transition_receipt(common.member(local/'running.json'),source)
            value=common.read(common.verify(row));value['target_jobs']['CAKE']['job']='61436'
            common.write(local/'wrong-job.json',value)
            with self.assertRaisesRegex(RuntimeError,'TERMINAL'):
                common.transition_receipt(common.member(local/'wrong-job.json'),source)

    def test_cap1_or2_afterany_native_lanes_no_generation_READY_gate(self):
        for cap in (1,2):
            ids={};jobs=[dict(job='90',gpus=1,resource_detail='Dependency=(null) ')]
            parents=submit.order(cap)
            for index,arm in enumerate(common.ARMS):
                ids[arm]=str(index+1);dep=submit.role_dependencies(arm,['90'],parents,ids)
                self.assertIn('90',dep)
                jobs.append(dict(job=ids[arm],gpus=1,resource_detail='Dependency=afterany:'+':'.join(dep)+' '))
            self.assertEqual(submit.width(jobs),cap)
            self.assertEqual(submit.role_dependencies('collector',[],parents,ids),list(ids.values()))
        self.assertEqual(submit.order(2),dict(BASE_MEMIT=[],BASE_ALPHAEDIT=[],CAKE=['BASE_MEMIT']))

    def test_protected_pending_frontier_keeps_both_GPU_leaves_and_cap2(self):
        existing=[dict(job='61436',gpus=1,resource_detail='Dependency=(null) '),
            dict(job='61437',gpus=1,resource_detail='Dependency=(null) '),
            dict(job='61438',gpus=1,resource_detail='Dependency=afterany:61436 ')]
        frontier=submit.validate_frontier(existing,submit.dependencies(existing))
        self.assertEqual(frontier,['61437','61438'])
        jobs=copy.deepcopy(existing);ids={};parents=submit.order(2)
        for index,arm in enumerate(common.ARMS):
            ids[arm]=str(62000+index);dep=submit.role_dependencies(arm,frontier,parents,ids)
            self.assertTrue({'61437','61438'}<=set(dep))
            self.assertNotIn('61436',dep);self.assertNotIn('61439',dep)
            jobs.append(dict(job=ids[arm],gpus=1,resource_detail='Dependency=afterany:'+':'.join(dep)+' '))
        self.assertEqual(submit.width(jobs),2)
        for invalid in (['61436'],['61437'],['61437','61439']):
            with self.subTest(invalid=invalid),self.assertRaisesRegex(RuntimeError,'NATIVE_W20_'):
                submit.validate_frontier(existing,invalid)
        self.assertEqual(submit.validate_frontier([],[]),[])

    def test_exact_native_GPU_launchers_and_own_three_GPU0_collector(self):
        resources=dict(cpu=8,collector_cpu=8,host_mib=65536,collector_host_mib=24576,
            wall='2-00:00:00',collector_wall='04:00:00')
        for role in submit.ROLES:
            argv=submit.arguments(role,['61437','61438'],Path('/attempt'),resources)
            for field in ('--hold','--export=NONE','--no-requeue','--dependency=afterany:61437:61438',
                '--job-name='+common.TASK+'-'+role):self.assertIn(field,argv)
            self.assertFalse(any('afterok' in field or 'kill-on-invalid-dep' in field for field in argv))
            self.assertEqual('--gres=gpu:1' in argv,role!='collector')
            script=submit.launcher(Path('/source'),'a'*40,role,Path('/attempt'))
            self.assertIn('native_w20_collect' if role=='collector' else 'native_w20_run',script)
            self.assertIn(common.SOURCE_ENV,script)
            for forbidden in ('WANDB_API_KEY','SLURM_JOB_ID=','W0_READY','--arm PRUNE','--arm RECT','--arm ALPHAEDIT_BLUE'):
                self.assertNotIn(forbidden,script)
        with self.assertRaisesRegex(RuntimeError,'TARGET_ROLE_ONLY'):
            submit.arguments('ALPHAEDIT_BLUE',[],Path('/attempt'),resources)

    def test_actual_transition_metadata_without_scheduler_queries(self):
        row=common.LOCAL/'w20-only-native-three-r1/transition-receipt.json'
        if not row.is_file():self.skipTest('Exact new server1 native task transition not present')
        value=common.read(row)
        common.transition_receipt(common.member(row),Path(value['source_attempt']))
        self.assertEqual({arm:r['job'] for arm,r in value['target_jobs'].items()},common.OLD_TARGET_IDS)

    def test_compile_new_owned_control(self):
        for name in ('native_w20_common.py','native_w20_prepare.py','native_w20_submit.py','native_w20_preflight.py'):
            path=Path(__file__).with_name(name);compile(path.read_bytes(),str(path),'exec')

if __name__=='__main__':unittest.main()
