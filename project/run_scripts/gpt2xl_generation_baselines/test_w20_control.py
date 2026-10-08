"""Three-only scope, immutable schedule, duplicate and resource-control fixtures."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import w20_common as common,w20_prepare as prepare,w20_submit as submit

class Control(unittest.TestCase):
    def fixture(self,tmp):
        root=Path(tmp);source=root/'old';attempt=root/'new'
        plan=dict(model_identity={'model':'gpt2xl'},requests=[dict(prompt='LOCAL_FIXTURE')],actual_qualification=False)
        common.write(root/'plan.json',plan)
        prior=dict(instruction_id='old',task_id='old',attempt=str(source),arms=['old'],
            source_configs={arm:dict(native={'algorithm':arm,'layers':[13,17] if arm=='ALPHAEDIT_BLUE' else [13,14,15,16,17],
                'v_lr':.5,'v_num_grad_steps':20,'mom2_update_weight':20000})
                for arm in (*common.ARMS,'BASE_MEMIT','BASE_ALPHAEDIT','CAKE')},
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
            target_jobs={arm:dict(job=common.OLD_TARGET_IDS[arm],state='CANCELLED') for arm in common.ARMS},
            protected_jobs_preserved=True,old_source_raw_preserved=True,snapshot_at='2026-10-08T19:00:00+09:00'))
        return root,source,attempt,prior,common.member(root/'transition.json')

    def test_fresh_canonical_authority_exact_three_and_latest_policy(self):
        envelope,policy=common.authority()
        self.assertEqual(envelope['scope']['methods'],['AlphaEdit-BLUE','PRUNE','RECT'])
        self.assertEqual(policy['generation_schedule'],common.SCHEDULE)
        self.assertEqual(common.ARMS,('ALPHAEDIT_BLUE','PRUNE','RECT'))

    def test_transform_removes_W0_and_intermediate_generation_without_science_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            local,source,attempt,prior,transition=self.fixture(tmp);before=copy.deepcopy(prior)
            with patch.object(common,'LOCAL',local):
                value=prepare.prepared_config(prior,attempt,source,prior['generation']['qualification_plan_member'],transition)
            self.assertEqual(prior,before)
            self.assertEqual(set(value['source_configs']),set(common.ARMS))
            for arm in common.ARMS:self.assertEqual(value['source_configs'][arm],prior['source_configs'][arm])
            for field in ('cold_W','runtime','observer_identity','tracking','stream','packs'):
                self.assertEqual(value[field],prior[field])
            for field in ('shared_W0_root','old_w0_reuse','primary_arm'):
                self.assertNotIn(field,value['generation'])
            self.assertNotIn('cpu_preflight',value);self.assertNotIn('manual_retry_authority_member',value)
            self.assertFalse(value['W0_generation_required'])
            self.assertEqual(value['generation_endpoints'],[dict(endpoint='all_seen/post',state='W20',edits=2000,requests=2000)])
            self.assertEqual(value['generation']['qualification_plan_member'],prior['generation']['qualification_plan_member'])
            self.assertEqual(value['generation']['plan']['three_arm_case_observations_max'],6000)

    def test_W0_prerequisite_extra_arm_or_other_endpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            local,source,attempt,prior,transition=self.fixture(tmp)
            with patch.object(common,'LOCAL',local):
                value=prepare.prepared_config(prior,attempt,source,prior['generation']['qualification_plan_member'],transition)
                for mutation in ('W0','arm','endpoint'):
                    changed=copy.deepcopy(value)
                    if mutation=='W0':changed['generation']['shared_W0_root']='/forbidden/W0'
                    if mutation=='arm':changed['source_configs']['CAKE']={}
                    if mutation=='endpoint':changed['generation_endpoints'].append({'state':'W5'})
                    with self.subTest(mutation=mutation),self.assertRaisesRegex(RuntimeError,'W20_'):
                        common.validate_config(changed)

    def test_old_nonce_is_history_new_registration_or_unknown_ID_attempt_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            local=Path(tmp)
            common.write(local/'old/config.json',dict(instruction_id='old'))
            common.write(local/'old/submission.json',dict(jobs={}))
            with patch.object(common,'LOCAL',local):
                self.assertEqual(common.registered_attempts(),[])
                common.write(local/'new/config.json',dict(instruction_id=common.NONCE))
                common.write(local/'new/registration-pass-started.json',dict(automatic_retry=False))
                self.assertEqual(common.registered_attempts(),[local/'new'])

    def test_exact_terminal_transition_and_protected_jobs_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            local,source,attempt,prior,row=self.fixture(tmp)
            self.assertEqual(set(common.transition_receipt(row,source)['target_jobs']),set(common.ARMS))
            for field,new in (('protected_jobs_preserved',False),('source_attempt','/wrong/attempt')):
                value=common.read(common.verify(row));value[field]=new
                common.write(local/(field+'.json'),value)
                with self.subTest(field=field),self.assertRaisesRegex(RuntimeError,'TRANSITION_SCOPE'):
                    common.transition_receipt(common.member(local/(field+'.json')),source)
            value=common.read(common.verify(row));value['target_jobs']['RECT']['state']='RUNNING'
            common.write(local/'running.json',value)
            with self.assertRaisesRegex(RuntimeError,'TERMINAL'):
                common.transition_receipt(common.member(local/'running.json'),source)

    def test_three_afterany_resource_lanes_cap1_or2_no_generation_GATE(self):
        for cap in (1,2):
            ids={};jobs=[dict(job='90',gpus=1,resource_detail='Dependency=(null) ')]
            parents=submit.order(cap)
            for index,arm in enumerate(common.ARMS):
                ids[arm]=str(index+1);dep=submit.role_dependencies(arm,['90'],parents,ids)
                self.assertIn('90',dep)
                jobs.append(dict(job=ids[arm],gpus=1,resource_detail='Dependency=afterany:'+':'.join(dep)+' '))
            self.assertEqual(submit.width(jobs),cap)
            self.assertEqual(submit.role_dependencies('collector',[],parents,ids),list(ids.values()))
        self.assertEqual(submit.order(2),dict(ALPHAEDIT_BLUE=[],PRUNE=[],RECT=['ALPHAEDIT_BLUE']))

    def test_exact_three_GPU_launcher_and_target_only_CPU_collector(self):
        resources=dict(cpu=8,collector_cpu=8,host_mib=65536,collector_host_mib=24576,
            wall='2-00:00:00',collector_wall='04:00:00')
        for role in submit.ROLES:
            argv=submit.arguments(role,['100'],Path('/attempt'),resources)
            for field in ('--hold','--export=NONE','--no-requeue','--dependency=afterany:100',
                '--job-name='+common.TASK+'-'+role):self.assertIn(field,argv)
            self.assertFalse(any('afterok' in field or 'kill-on-invalid-dep' in field for field in argv))
            self.assertEqual('--gres=gpu:1' in argv,role!='collector')
            script=submit.launcher(Path('/source'),'a'*40,role,Path('/attempt'))
            self.assertIn('w20_collect' if role=='collector' else 'w20_run',script)
            self.assertIn(common.SOURCE_ENV,script)
            for forbidden in ('WANDB_API_KEY','SLURM_JOB_ID=','W0_READY','--arm CAKE','--arm BASE_MEMIT'):
                self.assertNotIn(forbidden,script)
        with self.assertRaisesRegex(RuntimeError,'TARGET_ROLE_ONLY'):
            submit.arguments('BASE_MEMIT',[],Path('/attempt'),resources)

    def test_actual_transition_metadata_without_scheduler_query(self):
        row=common.LOCAL/'w20-only-three-r1/transition-receipt.json'
        if not row.is_file():self.skipTest('Exact server1 new task transition not present')
        value=common.read(row)
        common.transition_receipt(common.member(row),Path(value['source_attempt']))
        self.assertEqual(set(value['target_jobs']),set(common.ARMS))

    def test_compile_new_owned_control(self):
        for name in ('w20_common.py','w20_prepare.py','w20_submit.py'):
            path=Path(__file__).with_name(name);compile(path.read_bytes(),str(path),'exec')

if __name__=='__main__':unittest.main()
