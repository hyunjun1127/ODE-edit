"""Synthetic launch/accounting tests; no scheduler command or model load."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from .contracts import INSTRUCTION_ID
from .launch import (BUDGET_SHA256, CELL_WALL_SECONDS, JOB_NAME, PACKAGE, LaunchBoundary,
                     empty_campaign_registry, preparelaunch, reconcile_accounting,
                     register_submission, reserve_first_s_wave, validate_launch)
from .publication import write_once


def registry():
    value = empty_campaign_registry()
    value['jobs'] = [dict(job_id='123_0', instruction_id=INSTRUCTION_ID, track='S',
        node='devbox', gpus=1, owner='janghj', job_name=JOB_NAME,
        source_head='a'*40, run_root='/repo/local/alpha-jv-llama-diagnosis-sweep/attempt-TECH-R1', wall_seconds=CELL_WALL_SECONDS)]
    return value


def accounting(value, state='COMPLETED', elapsed=100, live='', *, tres='cpu=8,gres/gpu=1', owner='janghj', name=JOB_NAME):
    text = f'123_0|{owner}|{name}|{state}|{elapsed}|{tres}|480\n'
    return reconcile_accounting(value, sacct_text=text, squeue_text=live, sacct_ok=True, squeue_ok=True)


class LaunchTests(unittest.TestCase):
    def test_initial_reservation_is_16_not48_per_cell(self):
        snap = reconcile_accounting(empty_campaign_registry(), sacct_text='', squeue_text='', sacct_ok=True, squeue_ok=True)
        fact = reserve_first_s_wave(snap)
        self.assertEqual(snap['charged_GPU_seconds'], 0)
        self.assertEqual(fact['new_wave_reservation_GPU_seconds'], 57600)
        self.assertEqual(fact['remaining_after_wave_reservation_GPU_seconds'], 115200)

    def test_failed_attempt_charges_all_allocation_seconds(self):
        for state in ('COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED by 1025'):
            with self.subTest(state=state):
                snap=accounting(registry(),state=state,elapsed=1234)
                self.assertEqual(snap['charged_GPU_seconds'],1234)
                self.assertEqual(snap['reserved_running_or_queued_GPU_seconds'],0)

    def test_pending_wait_zero_consumption_full_reservation(self):
        snap=accounting(registry(),state='PENDING',elapsed=0,live='123_0|PENDING|0:00',tres='')
        self.assertEqual(snap['charged_GPU_seconds'],0)
        self.assertEqual(snap['reserved_running_or_queued_GPU_seconds'],28800)

    def test_live_and_cancelled_completing_charged_conservatively(self):
        for state in ('RUNNING','COMPLETING','CANCELLED by 1025'):
            with self.subTest(state=state):
                snap=accounting(registry(),state=state,elapsed=60,live='123_0|COMPLETING|02:00')
                self.assertEqual(snap['charged_GPU_seconds'],120)
                self.assertEqual(snap['reserved_running_or_queued_GPU_seconds'],28680)

    def test_bad_accounting_and_wrong_owner_fail(self):
        for kwargs in ({'owner':'other'}, {'name':'unrelated_job'}, {'tres':''},
                       {'tres':'gres/gpu=1,gres/gpu:a100=2'}, {'elapsed':'NaN'}, {'state':'UNKNOWN'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(LaunchBoundary):
                accounting(registry(),**kwargs)
        with self.assertRaises(LaunchBoundary):
            reconcile_accounting(registry(),sacct_text='',squeue_text='',sacct_ok=True,squeue_ok=True)
        with self.assertRaises(LaunchBoundary):
            reconcile_accounting(empty_campaign_registry(),sacct_text='',squeue_text='',sacct_ok=False,squeue_ok=True)

    def test_live_missing_or_malformed_fails(self):
        with self.assertRaises(LaunchBoundary):accounting(registry(),state='RUNNING')
        with self.assertRaises(LaunchBoundary):accounting(registry(),live='123_0|COMPLETING|UNKNOWN')
        with self.assertRaises(LaunchBoundary):accounting(registry(),live='124_0|COMPLETING|02:00')
        with self.assertRaises(LaunchBoundary):accounting(registry(),state='PENDING',elapsed=5,tres='')

    def test_insufficient_campaign_remaining_blocks_whole_wave(self):
        value=registry();value['jobs'][0]['wall_seconds']=172800
        snap=reconcile_accounting(value,sacct_text=f'123_0|janghj|{JOB_NAME}|FAILED|120000|gres/gpu=1|2880',
                                  squeue_text='',sacct_ok=True,squeue_ok=True)
        with self.assertRaisesRegex(LaunchBoundary,'INSUFFICIENT'):
            reserve_first_s_wave(snap)
        snap['charged_GPU_seconds']=0
        with self.assertRaisesRegex(LaunchBoundary,'IDENTITY'):
            reserve_first_s_wave(snap)

    def test_duplicate_steps_do_not_double_charge(self):
        row=f'123_0|janghj|{JOB_NAME}|COMPLETED|60|gres/gpu=1|480\n'
        steps=f'123_0.batch|janghj|batch|COMPLETED|60|gres/gpu=1|480\n'
        snap=reconcile_accounting(registry(),sacct_text=row+steps,squeue_text='',sacct_ok=True,squeue_ok=True)
        self.assertEqual(snap['charged_GPU_seconds'],60)
        with self.assertRaises(LaunchBoundary):
            reconcile_accounting(registry(),sacct_text=row+row,squeue_text='',sacct_ok=True,squeue_ok=True)

    def build_temp(self,base):
        repo=Path(base);src=repo/PACKAGE
        src.mkdir(parents=True)
        script=Path(__file__).with_name('server1_sweep.sbatch').read_bytes()
        for name,data in (('server1_sweep.sbatch',script),('gpu_runtime.py',b'# synthetic test fixture, no model\n'),('launch.py',b'# synthetic source\n')):
            write_once(src/name,data,root=repo)
        for name,value in (('budget.json',{'synthetic':True}),('sample.json',{'synthetic':True}),
                           ('assets.json',{'status':'CPU_ASSET_BINDING'}),('cpu.json',{'status':'PASS_CPU_ONLY'})):
            write_once(repo/name,value,root=repo)
        def git(_repo,*args):
            if args[0]=='status':return ''
            if args==('rev-parse','HEAD'):return 'a'*40
            if args==('rev-parse','HEAD^{tree}'):return 'b'*40
            if args[0]=='branch':return 'codex/synthetic-only'
            if args[:2]==('ls-files','--'):return args[2]
            if args[0]=='ls-files':return '\n'.join(f'{PACKAGE}/{x}' for x in ('server1_sweep.sbatch','gpu_runtime.py','launch.py'))
            raise AssertionError(args)
        return repo,git

    def prepare_temp(self,repo):
        root=repo/'local/alpha-jv-llama-diagnosis-sweep/TECH-R1'
        preparelaunch(root,repo,budget_lock=repo/'budget.json',sample_lock=repo/'sample.json',
                      assets_lock=repo/'assets.json',cpu_checks_lock=repo/'cpu.json',
                      campaign_registry=empty_campaign_registry())
        register_submission(root,array_job_id=123,owner='janghj',owner_uid=1025,
                            held_inspection_sha256='c'*64,live_admission_sha256='d'*64)
        return root

    def test_premodel_checks_source_locks_owner_task(self):
        with tempfile.TemporaryDirectory() as base:
            repo,git=self.build_temp(base)
            with patch('project.run_scripts.alpha_jv_llama_diagnosis.launch._git',git),patch(
                    'project.run_scripts.alpha_jv_llama_diagnosis.launch.verify_budget_authority',return_value={'synthetic':True}):
                root=self.prepare_temp(repo)
                env=dict(SLURM_ARRAY_JOB_ID='123',SLURM_ARRAY_TASK_ID='0',SLURM_JOB_NAME=JOB_NAME)
                result=validate_launch(repo,root,0,environ=env,current_uid=1025)
                self.assertEqual(result['status'],'PREMODEL_MECHANICAL_PASS')
                with self.assertRaisesRegex(LaunchBoundary,'OWNERSHIP'):
                    validate_launch(repo,root,0,environ=env,current_uid=1)
                with self.assertRaisesRegex(LaunchBoundary,'UNREGISTERED'):
                    validate_launch(repo,root,1,environ=env,current_uid=1025)
                with self.assertRaises(FileExistsError):
                    register_submission(root,array_job_id=123,owner='janghj',owner_uid=1025,
                        held_inspection_sha256='c'*64,live_admission_sha256='d'*64)

    def test_source_drift_before_model_is_blocking(self):
        with tempfile.TemporaryDirectory() as base:
            repo,git=self.build_temp(base)
            with patch('project.run_scripts.alpha_jv_llama_diagnosis.launch._git',git),patch(
                    'project.run_scripts.alpha_jv_llama_diagnosis.launch.verify_budget_authority',return_value={'synthetic':True}):
                root=self.prepare_temp(repo)
            def changed(path,*args):return 'f'*40 if args==('rev-parse','HEAD') else git(path,*args)
            with patch('project.run_scripts.alpha_jv_llama_diagnosis.launch._git',changed):
                with self.assertRaisesRegex(LaunchBoundary,'QUEUED_SOURCE_DRIFT'):
                    validate_launch(repo,root,0,environ={},current_uid=1025)

    def test_uncommitted_launch_source_is_blocking(self):
        with tempfile.TemporaryDirectory() as base:
            repo,git=self.build_temp(base)
            def missing(path,*args):return '' if args==('ls-files','--',PACKAGE+'/gpu_runtime.py') else git(path,*args)
            with patch('project.run_scripts.alpha_jv_llama_diagnosis.launch._git',missing),patch(
                    'project.run_scripts.alpha_jv_llama_diagnosis.launch.verify_budget_authority',return_value={'synthetic':True}):
                with self.assertRaisesRegex(LaunchBoundary,'NOT_COMMITTED'):
                    self.prepare_temp(repo)


if __name__=='__main__':unittest.main()
