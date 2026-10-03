import copy
import datetime
import getpass
import json
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from .reduce import metrics,paired,paired_rows,interaction,validate,distribution
from .common import NONCE,write,sha
from . import submit
from .collect import collect


def row(i=0,new=1.,true=2.,kind='N'):
    return dict(identity=str(i),case_id=i,kind=kind,prompt_index=0,
        new_nll=new,true_nll=true,margin_true_minus_new=true-new,
        new_token_identity='n'+str(i),true_token_identity='t'+str(i),
        new_token_count=2,new_token_correct=1,new_strict=False,
        true_token_count=1,true_token_correct=1,true_strict=True)


class WorkflowTests(unittest.TestCase):
    def test_true_new_sign_ties_TF(self):
        m=metrics([row(0,2,2),row(1,3,1)])['N']
        self.assertEqual((m['numerator'],m['ties'],m['TF_token_micro']),(1,1,1.))
        self.assertEqual(metrics([row(kind='R')])['R']['numerator'],1)
    def test_equal_totals_lost_and_gain(self):
        p=paired([row(0,3,1),row(1,1,3)],[row(0,1,3),row(1,3,1)])['N']
        self.assertEqual((p['lost'],p['gained']),(1,1))
        detail=paired_rows([row(0,3,1)],[row(0,1,3)])[0]
        self.assertTrue(detail['lost']);self.assertEqual(detail['desired_margin_delta'],-4)
    def test_corrupt_rows_block(self):
        r=row()
        for values in ([r,r],[dict(r,true_strict=False)],[dict(r,new_nll=float('nan'))]):
            with self.assertRaises(ValueError):metrics(values)
        with self.assertRaises(ValueError):validate([dict(r,new_token_identity='x')],[r])
    def test_signed_interaction_not_damage_sum(self):
        m={k:[row(new=n,true=1)] for k,n in [('NONE',2),('SUBJECT_ONLY',4),('NONSUBJECT_ONLY',3),('ALL',7)]}
        r,s=interaction(m);self.assertEqual(r[0]['interaction'],2);self.assertEqual(s['mean'],2)
    def test_quantile_tail(self):self.assertEqual(distribution([0,10])['p95'],9.5)
    def test_partial_collector_preserves_no_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write(p/'execution.lock.json',dict(instruction=NONCE,source='fixture'))
            write(p/'config.json',{});write(p/'lookup-local.json',dict(denominator=1000,identifiable=0))
            collect(p)
            self.assertEqual(json.loads((p/'collection/terminal.json').read_text())['status'],'PARTIAL_OR_TECHNICAL_BLOCKED')
            self.assertIn('NOT_MEASURED',(p/'collection/report-ko.md').read_text())
    def test_held_gpu_and_collector_all_before_release_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write(p/'execution.lock.json',dict(instruction=NONCE,source='fixture',source_root=str(p),members=[],runtime_sources=[]))
            launches={}
            for stage in ('gpu','collector'):
                script=p/(stage+'.sh');script.write_text('#!/bin/bash\ntrue\n')
                launches[stage]=dict(file=dict(path=str(script),sha256=sha(script)))
            write(p/'launchers.json',launches);ad=p/'admission.json'
            write(ad,dict(instruction=NONCE,project_GPU_after=1,effective_cap=2,task_existing_GPU=0,
                observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            registered={};calls=[]
            def fake(argv):
                calls.append(argv)
                if argv[0]=='sbatch':
                    jid=str(100+len(registered));registered[jid]=argv;return jid
                if argv[:3]==['scontrol','show','job']:
                    jid=argv[-1];sb=registered[jid];opts=dict(x[2:].split('=',1) for x in sb if x.startswith('--') and '=' in x)
                    gpu='--gres=gpu:1' in sb;script=p/('gpu.sh' if gpu else 'collector.sh')
                    return (f"JobId={jid} UserId={getpass.getuser()}(1000) JobName={opts['job-name']} ReqNodeList=devbox Partition=gpu "
                        f"JobState=PENDING Priority=0 Requeue=0 NumCPUs=8-14 CPUs/Task=8 Command={script} ReqTRES=cpu=8,mem="+('96G,gres/gpu=1' if gpu else '24G')+
                        f" TimeLimit={opts['time']} Dependency={opts.get('dependency','(null)')} SubmitLine="+shlex.join(sb))
                if argv[:2]==['scontrol','release']:
                    self.assertEqual(len(registered),2);self.assertTrue((p/'submission.json').exists());return ''
                raise AssertionError(argv)
            with patch.object(submit,'command',side_effect=fake),patch('sys.argv',['submit','--attempt',str(p),'--admission',str(ad)]):submit.main()
            self.assertEqual([x[-1] for x in calls if x[:2]==['scontrol','release']],['101','100'])
            with patch.object(submit,'command',side_effect=AssertionError('duplicate')),patch('sys.argv',['submit','--attempt',str(p),'--admission',str(ad)]):
                with self.assertRaisesRegex(RuntimeError,'NONCE'):submit.main()


if __name__=='__main__':unittest.main()
