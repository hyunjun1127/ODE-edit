"""Tiny local CPU storage fixtures, never Slurm/model/network evidence."""
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from project.run_scripts import server4_qwen_archive as q
from project.run_scripts.checkpoint_archive import archive as a


class ArchiveTests(unittest.TestCase):
    def fixture(self,root):
        logical='qwen25-cf-ft';job='12345';out=root/'runs'/logical
        identity={k:(hashlib.sha1 if k in a.GIT_IDENTITY_FIELDS else hashlib.sha256)(('CPU_FIXTURE_'+k).encode()).hexdigest() for k in a.IDENTITY_FIELDS}
        q.write(root/'archive-policy.json',dict(schema='final-checkpoint-central-archive-v1',instruction_id=a.INSTRUCTION,
            scope=dict(prospective_only=True),destination=dict(root='/unused/test')))
        q.write(root/'cutover.json',dict(schema='final-checkpoint-archive-server-cutover-v1',server='server4',instruction_id=a.INSTRUCTION,
            scope=a.SCOPE,policy_sha256=q.member(root/'archive-policy.json')['sha256'],ack_nonce='CPU_FIXTURE_ONLY',received_at_utc='2020-01-01T00:00:00Z'))
        q.write(root/'assets.json',dict(CPU_FIXTURE=True));q.write(root/'source-lock.json',dict(CPU_FIXTURE=True))
        q.write(root/'configs'/f'{logical}.json',dict(CPU_FIXTURE=True));q.write(root/'streams/cf-stream.lock.json',dict(CPU_FIXTURE=True))
        q.write(root/'receiver.json',dict(host='CPU_FIXTURE_NO_NETWORK'))
        q.adopt(root,logical,identity);q.registered(root,logical,job,q.now())
        (out/'checkpoint').mkdir(parents=True)
        payload=out/'checkpoint/final.pt';payload.write_bytes(b'CPU fixture, not tensor')
        pointer=dict(batch=20,final_W20=True,file=payload.name,sha256=q.member(payload)['sha256'],identity_sha256=a.digest(identity))
        q.write(out/'checkpoint/latest.json',pointer)
        commits=[]
        for b in range(1,21):
            p=out/'commits'/f'b{b:02d}.json';q.write(p,dict(completed_batch=b,CPU_FIXTURE=True));commits.append(q.member(p))
        calc={}
        for name in ('factual','generation'):
            p=out/(name+'.json');q.write(p,dict(CPU_FIXTURE_NOT_ACTUAL_MEASUREMENT=True))
            calc[name]=dict(state='COMPLETE',observed_requests=2000,**q.member(p))
        q.write(out/'terminal.json',dict(schema='server4-qwen-official-terminal-v1',actual_job_id=job,logical_main_row=logical,
            checkpoint_identity=identity,checkpoint=pointer,completed_edits=2000,status='W20_COMPLETE',
            completed_at_utc=q.now(),commits=commits,calculation_evidence=calc))
        receiver=a.Receiver.from_policy_file(root/'archive-policy.json',cutover_members={'server4':q.member(root/'cutover.json')},root=root/'receiver',test_only=True)
        return logical,payload,receiver

    def execute_fixture(self,fail_verify=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);logical,payload,r=self.fixture(root)
            def remote(root,action,value):
                if action=='admit':return r.admit(value['manifest'])
                if action=='verify':
                    if fail_verify:raise RuntimeError('CPU_FIXTURE_VERIFY_FAILURE')
                    return r.verify(value['admission'],value['manifest'])
                return r.recheck(value['receipt'])
            def copy(argv,**kwargs):
                self.assertEqual(argv[0],'rsync');self.assertNotIn('--delete',argv)
                shutil.copyfile(argv[-2],argv[-1].split(':',1)[1])
            with patch.object(q,'remote',remote),patch.object(q,'scheduler_terminal',return_value='12345|COMPLETED|0:0|'),patch.object(q.subprocess,'run',copy):
                if fail_verify:
                    with self.assertRaisesRegex(RuntimeError,'VERIFY_FAILURE'):q.archive(root,logical)
                    self.assertTrue(payload.exists())
                else:
                    q.archive(root,logical)
                    self.assertFalse(payload.exists())
                    receipt=q.read(root/'archive'/logical/'source-removed.json')
                    self.assertEqual(receipt['stage'],'SOURCE_REMOVED_ARCHIVE_VERIFIED')
                    self.assertTrue((root/'runs'/logical/'terminal.json').exists())

    def test_full_tiny_CPU_roundtrip(self):self.execute_fixture()
    def test_receiver_failure_keeps_source(self):self.execute_fixture(True)


if __name__=='__main__':unittest.main()
