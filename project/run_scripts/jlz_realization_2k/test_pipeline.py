"""Bounded controller/collector contracts. No model loading or scheduler calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from .common import INSTRUCTION, digest, member, write
from .schedule import selection, denominators, verify_commit, validate_config
from . import run, collect, submit

def config():
    return dict(experiment={'main':dict(batches_per_arm=20,batch_size=100,edits_per_arm=2000,no_batch21=True)},
        evaluation_schedule={'all_seen_at':[5,10,20]},settings={'exact_probe':'REUSE_HISTORICAL_NO_NEW_PROBE',
            'save_checkpoints':False,'observer_microbatch':2},baseline_new_runs=0,
        packing=[dict(phase='main',batch=b,identity=str(b),ids=list(range((b-1)*100,b*100))) for b in range(1,21)])

def receipt(number, c):
    after=dict(W={'4':str(number)},H={'4':str(number)})
    return dict(source='test',config=digest(c),batch=number,actual_B=100,current_ids=list(range((number-1)*100,number*100)),
        candidate_count=25,backward_count=24,Adam_updates=24,history_appends=5,
        history={'accepted_weight_copy_exact':True},exact_commit_sha=after['W'],after=after,
        Q2=False,replay=False,terminal_gradient_measured=False,no_checkpoint=True)

class PipelineTests(unittest.TestCase):
    def test_all_twenty_endpoints_no21(self):
        records=[{'case_id':i} for i in range(2000)]
        total=0
        for b in range(1,21):
            current,seen,selected=selection(records,b)
            self.assertEqual(len(current),100)
            self.assertEqual([x['case_id'] for x in current],list(range((b-1)*100,b*100)))
            self.assertEqual(len(selected),denominators(b)['R'])
            total+=len(selected)
        self.assertEqual(total,5200)
        for b in (0,21):
            with self.assertRaises(RuntimeError):selection(records,b)

    def test_exact_milestone_counts(self):
        for b,n in ((5,500),(10,1000),(20,2000)):
            self.assertEqual(denominators(b),dict(R=n,P=2*n,N=10*n))
        self.assertEqual(denominators(19),dict(R=100,P=200,N=1000))

    def test_config_no_legacy_scope(self):
        c=config();validate_config(c)
        for key,value in [('batches_per_arm',5),('edits_per_arm',500),('no_batch21',False)]:
            bad=copy.deepcopy(c);bad['experiment']['main'][key]=value
            with self.assertRaises(RuntimeError):validate_config(bad)

    def test_real_controller_routes20_and_noQ2(self):
        c=config();records=[{'case_id':i} for i in range(2000)];events=[];current_state={}
        def fake_batch(a,bench,records,h,c,arm,out,number,callback,q2):
            self.assertFalse(q2)
            callback(current_state,dict(identity=str(number),record_ids=[r['case_id'] for r in records]))
            result=receipt(number,c);current_state.clear();current_state.update(result['after'])
            events.append(('fit',number,len(records)));return result
        def fake_observe(a,bench,seen,selected,h,number,out,mb,current_ids):
            events.append(('eval',number,len(selected)))
            return dict(summary={k:dict(denominator=v) for k,v in denominators(number).items()},
                        current={k:dict(denominator=v) for k,v in dict(R=100,P=200,N=1000).items()})
        with tempfile.TemporaryDirectory() as td, patch.object(run,'state',side_effect=lambda *x:dict(current_state)),\
             patch.object(run,'batch',side_effect=fake_batch),patch.object(run,'observe',side_effect=fake_observe):
            output=run.drive(None,None,records,None,c,'A',Path(td),'test')
            self.assertEqual(len(output),20)
            self.assertEqual(json.loads((Path(td)/'initial.json').read_text())['state']['W'],{'4':'1'})
        self.assertEqual(events[-2:],[('fit',20,100),('eval',20,2000)])
        self.assertEqual(len(events),40)

    def test_commit_failures_are_blocking(self):
        c=config();r=receipt(20,c)
        verify_commit(r,20,list(range(1900,2000)),'test',digest(c))
        for key,value in [('Q2',True),('replay',True),('history_appends',4),('actual_B',2),('Adam_updates',25)]:
            bad=dict(r);bad[key]=value
            with self.assertRaises(RuntimeError):verify_commit(bad,20,list(range(1900,2000)),'test',digest(c))

    def test_old_completed500_is_not_2k(self):
        lock=dict(source_commit='s',config_sha256='c')
        term=dict(status='COMPLETED',instruction=INSTRUCTION,source='s',config_sha256='c',main_commits=20)
        self.assertTrue(collect.complete(term,20,19,[],lock))
        self.assertFalse(collect.complete(term,5,4,[],lock))
        self.assertFalse(collect.complete(term,20,19,['missing'],lock))
        self.assertFalse(collect.complete(dict(term,source='old'),20,19,[],lock))

    def test_partial_afterany_emits_report_before_terminal(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);write(root/'data.json',[]);write(root/'expected.json',dict(rows=[]));write(root/'w0.json',dict(rows=[]))
            write(root/'bridge.json',dict(observations=member(root/'w0.json')))
            c=dict(instruction_id=INSTRUCTION,stream=str(root/'data.json'),observer_identity=member(root/'expected.json'),
                w0_reuse={'receipt':member(root/'bridge.json')},baselines=[])
            write(root/'config.json',c);write(root/'execution.lock.json',dict(source_commit='test',config_sha256=member(root/'config.json')['sha256']))
            collect.collect(root,root/'report')
            t=json.loads((root/'report/terminal.json').read_text())
            self.assertEqual(t['status'],'PARTIAL_OR_TECHNICAL_FAILED')
            self.assertEqual(t['main_commits'],0)
            self.assertTrue(Path(t['report']['path']).exists() and Path(t['artifacts']['path']).exists())

    def test_scheduler_argv_two_independent_oneGPU_lanes(self):
        resources=dict(host_mib=60416,collector_host_mib=24576,wall='2-00:00:00',collector_wall='04:00:00')
        for name in ('main-A','main-B'):
            argv=submit.arguments(name,None,Path('/test'),resources)
            for option in ('--gres=gpu:1','--mem=60416M','--cpus-per-task=8','--export=NONE','--no-requeue','--hold'):
                self.assertIn(option,argv)
            self.assertFalse(any(x.startswith('--dependency=') for x in argv))
        cpu=submit.arguments('collector','afterany:1:2',Path('/test'),resources)
        self.assertIn('--dependency=afterany:1:2',cpu)
        self.assertFalse(any(x.startswith('--gres=') for x in cpu))

    def test_early_collector_exception_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with self.assertRaises(FileNotFoundError):collect.dispatch(root,root/'report')
            failure=root/'report-failure'
            self.assertEqual(json.loads((failure/'terminal.json').read_text())['status'],'COLLECTOR_TECHNICAL_FAILED')
            self.assertTrue((failure/'first-error.json').exists())
            self.assertTrue((failure/'artifact-index.json').exists())

if __name__=='__main__':unittest.main()
