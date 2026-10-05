"""Narrow port regression; no CUDA/model qualification claims."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import TASK, NONCE, digest, member, write, sha
from .profile import execution, check_horizon, B1_TASK, B1_NONCE
from . import b1_submit as submit

def config():
    return dict(task_id=B1_TASK,instruction_id=B1_NONCE,execution_profile='server2-b1',
        settings=dict(B=100,batches=1,requests=100),packs=[{}])

class B1Tests(unittest.TestCase):
    def test_default20_preserved(self):
        p=execution({});self.assertEqual((p['task'],p['nonce'],p['batches'],p['requests']), (TASK,NONCE,20,2000))
        self.assertEqual(p['status'],'W20_COMPLETE')
    def test_b1_exact_authority_horizon(self):
        c=config();p=check_horizon(c);self.assertEqual((p['batches'],p['requests'],p['milestones']),(1,100,(1,)))
        self.assertEqual(p['no_next'],'no_B2')
        for k,v in [('batches',20),('requests',2000),('B',2)]:
            bad=copy.deepcopy(c);bad['settings'][k]=v
            with self.assertRaises(RuntimeError):check_horizon(bad)
        c['instruction_id']=NONCE
        with self.assertRaises(RuntimeError):check_horizon(c)
    def test_single_runner_qualification_and_main_not_skipped(self):
        c=dict(runtime={'python':'/python'},dependency_overlay='/deps')
        s=submit.launcher(Path('/source'),'commit','main',Path('/attempt'),c)
        self.assertNotIn('--main-only',s);self.assertNotIn('--qualification-only',s)
        self.assertIn('causal_allocation_editing.run',s);self.assertIn('/source:/deps',s)
        self.assertEqual(submit.ROLES,('main','collector'))
    def test_resources_exact_name_cap_one(self):
        r=dict(host_mib=59392,collector_host_mib=24576)
        for role in submit.ROLES:
            a=submit.arguments(role,'afterany:123',Path('/attempt'),r)
            for flag in ('--hold','--export=NONE','--no-requeue','--cpus-per-task=6','--nodelist=server2','--job-name='+B1_TASK):self.assertIn(flag,a)
            self.assertIn('--dependency=afterany:123',a)
            self.assertIn('--time='+('04:00:00' if role=='collector' else '08:00:00'),a)
            self.assertEqual('--gres=gpu:1' in a,role=='main')
    def test_b1_terminal_does_not_invent_commit_or_w20(self):
        from .collect import collect
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=config();records=[dict(case_id=i) for i in range(100)]
            write(root/'observer.json',dict(rows=[]))
            c.update(stream='/data/counterfact.json',ordered_ids_sha256=digest(list(range(100))),
                observer_identity=member(root/'observer.json'),cold_W0_H0=dict(W={},H={}))
            write(root/'config.json',c);write(root/'execution.lock.json',dict(instruction_id=B1_NONCE,config_sha256=sha(root/'config.json'),source_commit='source'))
            write(root/'main/terminal.json',dict(status='W1_COMPLETE'))
            with patch('project.run_scripts.causal_allocation_editing.collect.load_prefix',return_value=records) as load:
                result=collect(root,accounting=False)
            load.assert_called_once_with(Path('/data'),100)
            self.assertEqual(result['expected'],dict(commits=1,requests=100,joins=0,history_appends=5))
            self.assertEqual(result['status'],'PARTIAL_OR_TECHNICAL_BLOCKED')
            self.assertEqual(result['commits'],0)
            report=(root/'collector/report-ko.md').read_text()
            self.assertNotIn('/2000',report);self.assertNotIn('/20,',report)

if __name__=='__main__':unittest.main()
