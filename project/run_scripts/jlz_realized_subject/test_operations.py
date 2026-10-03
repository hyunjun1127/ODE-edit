"""CPU tests for production transaction, dependency and independent raw audit."""
import datetime
import getpass
import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from .common import write,sha,INSTRUCTION,state
from .test_core import fixture,record
from .run import batch,aux_identity
from .collect import reduce,pair,validate_rows,validate_budget
from . import submit,collect


class OperationsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_two_batch_RAM_continuity_and_IO_rollback(self):
        a,bench,H,e,stats=fixture(self.root)
        config=dict(stats=stats,settings=dict(fit_microbatch=2))
        metadata=dict(ledger=[],next_batch=1)
        with patch.dict(os.environ,ODEEDIT_SOURCE_COMMIT='test-synthetic'):
            first=batch(a,bench,[record(0),record(1)],H,metadata,config,'A',self.root/'b1',1)
            second=batch(a,bench,[record(1),record(2)],H,metadata,config,'A',self.root/'b2',2)
            self.assertEqual(second['before'],first['after'])
            self.assertEqual(second['before_aux'],first['after_aux'])
            self.assertEqual(metadata['ledger'],[0,1,1,2])
            validate_budget(self.root/'b1/fit')
            before=state(a,H);aux=aux_identity(bench,metadata)
            from . import run
            original=run.write
            def fail_commit(path,value):
                if Path(path).name=='commit.json':raise OSError('injected receipt failure')
                return original(path,value)
            with patch.object(run,'write',side_effect=fail_commit):
                with self.assertRaisesRegex(OSError,'injected'):
                    batch(a,bench,[record(0),record(2)],H,metadata,config,'A',self.root/'failed',3)
            self.assertEqual(state(a,H),before);self.assertEqual(aux_identity(bench,metadata),aux)
            self.assertFalse((self.root/'failed/commit.json').exists())
    def test_upfront_DAG_all_held_before_any_release(self):
        write(self.root/'execution.lock.json',dict(instruction=INSTRUCTION,source_commit='test-source',source_root=str(self.root)))
        launchers={}
        for stage in ('q1','A','B','collector'):
            script=self.root/(stage+'.sh');script.write_text('#!/bin/bash\ntrue\n')
            launchers[stage]=dict(file=dict(path=str(script),sha256=sha(script)))
        write(self.root/'launchers.json',launchers)
        admission=self.root/'admission.json'
        write(admission,dict(instruction=INSTRUCTION,cap=1,host_memory_mib=60416,
            observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),predecessor_job_ids=['123']))
        commands=[];registered={}
        def fake(argv):
            commands.append(argv)
            if argv[0]=='sbatch':
                jid=str(900+len(registered));registered[jid]=argv;return jid
            if argv[:3]==['scontrol','show','job']:
                jid=argv[-1];sb=registered[jid];options=dict(x[2:].split('=',1) for x in sb if x.startswith('--') and '=' in x)
                mem='59G' if '--gres=gpu:1' in sb else '24G'
                return (f"JobId={jid} UserId={getpass.getuser()}(1025) JobName={options['job-name']} ReqNodeList=ubuntu "
                        f"Partition=gpu JobState=PENDING Priority=0 Command={sb[-1]} Requeue=0 NumCPUs=8 "
                        f"ReqTRES=cpu=8,mem={mem}"+(',gres/gpu=1' if mem=='59G' else '')+
                        f" Dependency={options.get('dependency','(null)')} SubmitLine="+shlex.join(sb))
            if argv[:2]==['scontrol','release']:
                self.assertEqual(len(registered),4);self.assertTrue((self.root/'submission.json').exists());return ''
            raise AssertionError(argv)
        with patch.object(submit,'run',side_effect=fake),patch('sys.argv',['submit','--attempt',str(self.root),'--admission',str(admission)]):
            submit.main()
        receipt=json.loads((self.root/'submission.json').read_text());jobs=receipt['jobs']
        self.assertEqual(jobs['q1']['dependencies'],['afterany:123'])
        self.assertEqual(jobs['A']['dependencies'],['afterok:900'])
        self.assertEqual(jobs['B']['dependencies'],['afterok:900','afterany:901'])
        self.assertEqual(jobs['collector']['dependencies'],['afterany:900:901:902'])
        self.assertEqual([c[-1] for c in commands if c[:2]==['scontrol','release']],['903','902','901','900'])
        with patch.object(submit,'run',side_effect=AssertionError('duplicate external call')),patch('sys.argv',['submit','--attempt',str(self.root),'--admission',str(admission)]):
            with self.assertRaisesRegex(RuntimeError,'DUPLICATE'):submit.main()
    def test_independent_raw_ties_N_direction_tokens_and_identity(self):
        row=dict(identity='id',case_id=1,kind='N',prompt_index=0,new_nll=2.,true_nll=2.,
            new_token_correct=0,new_token_count=2,new_strict=False,new_token_identity='nt',
            true_token_correct=1,true_token_count=2,true_strict=False,true_token_identity='tt')
        self.assertEqual(reduce([row])['N']['numerator'],0)
        after=dict(row,true_nll=1.)
        self.assertEqual(reduce([after])['N']['numerator'],1)
        self.assertEqual(reduce([after])['N']['token_micro'],.5)
        self.assertEqual(pair([row],[after])['N']['gained'],['id'])
        validate_rows([row],[row],[dict(case_id=1)])
        with self.assertRaisesRegex(RuntimeError,'TOKEN_TARGET'):validate_rows([dict(row,new_token_identity='wrong')],[row],[dict(case_id=1)])
        with self.assertRaisesRegex(RuntimeError,'NONFINITE'):reduce([dict(row,true_nll=float('nan'))])
    def test_missing_admission_never_registers_or_consumes_lock(self):
        write(self.root/'execution.lock.json',dict(instruction=INSTRUCTION))
        write(self.root/'launchers.json',{})
        with patch.object(submit,'run',side_effect=AssertionError('external mutation')),patch('sys.argv',['submit','--attempt',str(self.root),'--admission',str(self.root/'missing.json')]):
            with self.assertRaises(FileNotFoundError):submit.main()
        self.assertFalse((self.root/'submission.started').exists())
    def test_family_major_W0_and_case_major_actual_order(self):
        rows=[]
        for case in (8,3):
            for kind in ('R','P','N'):
                rows.append(dict(case_id=case,kind=kind,prompt_index=0,identity=str(case)+kind,
                    new_token_identity='n',true_token_identity='t',new_token_count=1,true_token_count=1))
        family=[r for k in ('R','P','N') for r in rows if r['kind']==k]
        validate_rows(rows,family,[dict(case_id=8),dict(case_id=3)])
        with self.assertRaisesRegex(RuntimeError,'ORDER_IDENTITY'):
            validate_rows(list(reversed(rows)),family,[dict(case_id=8),dict(case_id=3)])
    def test_collector_incomplete_is_not_scientific_completion(self):
        data=self.root/'data.json';write(data,[])
        w0=self.root/'W0.json';write(w0,dict(rows=[]))
        config=self.root/'config.json';write(config,dict(stream=str(data),W0_reuse=dict(path=str(w0),sha256=sha(w0))))
        write(self.root/'execution.lock.json',dict(instruction=INSTRUCTION,source_commit='test',config_sha256=sha(config)))
        with patch('sys.argv',['collect','--attempt',str(self.root),'--config',str(config)]):collect.main()
        out=self.root/'collection'
        self.assertEqual(json.loads((out/'terminal.json').read_text())['status'],'SCIENTIFIC_INCOMPLETE')
        for name in ('manifest.json','report-ko.md','summary.json'):self.assertTrue((out/name).exists())


if __name__=='__main__':unittest.main()
