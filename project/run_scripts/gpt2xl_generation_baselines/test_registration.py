"""Explicit USER recall control fixtures; no scheduler/model/GPU operations."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import common


class ManualRegistration(unittest.TestCase):
    def fixture(self,tmp):
        local=Path(tmp);prior=local/'attempt-cache-repair-r1';target=local/'attempt-cache-repair-r2'
        owner=dict(server='server1',session=common.SESSION)
        source='a'*40;jobs={role:str(60000+i) for i,role in enumerate((*common.ARMS,'collector'))}
        common.write(prior/'config.json',dict(instruction_id=common.NONCE,task_id=common.TASK,attempt=str(prior)))
        config=common.member(prior/'config.json')
        common.write(prior/'execution.lock.json',dict(instruction_id=common.NONCE,task_id=common.TASK,
            source_commit=source,config_sha256=config['sha256']))
        lock=common.member(prior/'execution.lock.json')
        common.write(prior/'submission.json',dict(instruction_id=common.NONCE,task_id=common.TASK,
            source_commit=source,jobs=jobs,config=config,lock=lock))
        terminal=dict(status='TERMINAL_RECONCILED',prior_attempt=str(prior),source_commit=source,
            owner=owner,snapshot_at='2026-10-08T12:00:00+09:00',
            jobs={role:dict(job=job,state='COMPLETED' if role=='collector' else 'FAILED') for role,job in jobs.items()})
        common.write(local/'terminal.json',terminal)
        value=dict(schema=1,instruction_id=common.NONCE,task_id=common.TASK,
            manual_recall_id=common.MANUAL_RECALL,authority_type='USER_EXPLICIT_MANUAL_REPAIR',
            user_quote=common.MANUAL_USER_QUOTE,owner=owner,target_attempt=str(target),prior_attempt=str(prior),
            prior_source_commit=source,prior_config_member=config,prior_lock_member=lock,
            prior_submission_member=common.member(prior/'submission.json'),prior_jobs=jobs,
            terminal_reconciliation_member=common.member(local/'terminal.json'),automatic_retry=False)
        common.write(local/'manual.json',value)
        return local,prior,target,value,common.member(local/'manual.json'),terminal

    def test_default_original_nonce_duplicate_protection_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            local,prior,target,*_=self.fixture(tmp)
            with patch.object(common,'LOCAL',local):
                with self.assertRaisesRegex(RuntimeError,'NO_DUPLICATE_NONCE'):
                    common.registration_authority(target)

    def test_only_exact_explicit_prior_r1_permitted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            local,prior,target,value,row,_=self.fixture(tmp)
            with patch.object(common,'LOCAL',local):
                self.assertEqual(common.registration_authority(target,row),value)
                common.write(target/'config.json',dict(instruction_id=common.NONCE))
                common.write(target/'submitted-BASE_MEMIT.json',dict(job='70000'))
                with self.assertRaisesRegex(RuntimeError,'NO_OTHER_REGISTERED_ATTEMPT'):
                    common.registration_authority(target,row)

    def test_user_quote_target_source_job_or_auto_retry_mismatch_rejected(self):
        mutations=(('user_quote','do something else','EXPLICIT_USER_AUTHORITY'),
            ('target_attempt','/wrong/attempt','EXACT_ATTEMPT'),
            ('prior_source_commit','b'*40,'PRIOR_SOURCE_CONFIG'),
            ('automatic_retry',True,'EXPLICIT_USER_AUTHORITY'),
            ('prior_jobs',{'BASE_MEMIT':'99999'},'PRIOR_SEVEN_IDS'))
        for key,new,error in mutations:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                local,prior,target,value,row,_=self.fixture(tmp)
                changed=copy.deepcopy(value);changed[key]=new
                common.write(local/'changed.json',changed)
                with patch.object(common,'LOCAL',local),self.assertRaisesRegex(RuntimeError,error):
                    common.registration_authority(target,common.member(local/'changed.json'))

    def test_active_or_unknown_prior_job_not_authorized_for_replacement(self):
        for state in ('RUNNING','PENDING','COMPLETING','UNKNOWN'):
            with self.subTest(state=state),tempfile.TemporaryDirectory() as tmp:
                local,prior,target,value,row,terminal=self.fixture(tmp)
                terminal['jobs']['BASE_MEMIT']['state']=state
                common.write(local/'active-terminal.json',terminal)
                value['terminal_reconciliation_member']=common.member(local/'active-terminal.json')
                common.write(local/'changed.json',value)
                with patch.object(common,'LOCAL',local),self.assertRaisesRegex(RuntimeError,'PRIOR_NOT_TERMINAL'):
                    common.registration_authority(target,common.member(local/'changed.json'))

    def test_immutable_runtime_lock_receipt_binding_not_registration_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            local,prior,target,value,row,_=self.fixture(tmp)
            config=dict(manual_retry_authority_member=row,manual_recall_id=common.MANUAL_RECALL)
            lock=dict(config,automatic_retry=False)
            with patch.object(common,'LOCAL',local):
                self.assertEqual(common.bound_manual_authority(target,config,lock),value)
                with self.assertRaisesRegex(RuntimeError,'CONFIG_LOCK_BINDING'):
                    common.bound_manual_authority(target,config,dict(lock,manual_recall_id='wrong'))

    def test_actual_r1_manual_recall_metadata_without_scheduler_query(self):
        path=common.LOCAL/'cache-repair-20261008-r2/manual-retry-authority.json'
        if not path.is_file():self.skipTest('Actual server1 deliberate recall receipt not present')
        value=common.manual_retry_authority(common.LOCAL/'attempt-cache-repair-r2',common.member(path))
        self.assertEqual(value['manual_recall_id'],common.MANUAL_RECALL)
        self.assertEqual(set(value['prior_jobs']),set(common.ARMS)|{'collector'})
        self.assertFalse(value['automatic_retry'])


if __name__=='__main__':unittest.main()
