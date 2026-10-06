"""Narrow horizon/telemetry regressions; reuse prior CPU7 math evidence."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from . import ROOT,member,write
from .profile import profile
from . import telemetry
from .collect import collect
from .submit import argv,launcher,SOURCES
from project.run_scripts.experiment_tracking.schema import metrics,load_env

class Sink:
    def __init__(self):self.values=[]
    def log(self,v):self.values.append(metrics(v));return True

class Tests(unittest.TestCase):
    def test_horizons(self):
        old=profile();new=profile(10000)
        self.assertEqual((old['batches'],old['planned_eval_rows']),(20,137800))
        self.assertEqual((new['batches'],new['layer_solves'],new['own_joins']),(100,500,99))
        self.assertEqual(new['target_table_bytes'],819200000)
        self.assertEqual(new['milestones'],list(range(5,101,5)))
        self.assertEqual(new['final_denominators'],dict(R=10000,P=20000,N=100000))
        with self.assertRaises(RuntimeError):profile(5000)

    def test_scalar_mapping(self):
        sink=Sink();telemetry.fit(sink,99,dict(iteration=34,total=1.,nll=.8,kl=.1,delta_norm=2.))
        v=dict(rate=.5,numerator=1,denominator=2,token_micro=.6,prompt_macro=.6,strict_numerator=1,
            strict_denominator=2,desired_token_correct=3,desired_token_count=5,new_nll_mean=.4,true_nll_mean=.5)
        telemetry.evaluation(sink,dict(summary={k:v for k in ('R','P','N')},seconds=2.),100)
        self.assertEqual(len(sink.values),5);self.assertEqual(sink.values[1]['eval/NS'],.5)
        self.assertNotIn('case_id',str(sink.values));self.assertNotIn('prompt',str(sink.values).replace('prompt_macro',''))

    def test_full_100_commit_order_without_model(self):
        with tempfile.TemporaryDirectory() as temp:
            a=Path(temp);write(a/'ids.json',[]);write(a/'spec.json',[])
            write(a/'config.json',dict(observer_identity=member(a/'ids.json'),specs=member(a/'spec.json'),
                settings=dict(requests=10000),cold_W0_H0='cold'))
            write(a/'execution.lock.json',dict(source_commit='0'*40));previous='cold'
            for b in range(1,101):
                write(a/f'main/batch-{b:02d}/commit.json',dict(batch=b,before=previous,after=str(b)))
                previous=str(b)
            write(a/'main/terminal.json',dict(status='TECHNICAL_FAILED'))
            r=collect(a,accounting=False)
            self.assertEqual((r['commits'],r['own_joins']),(100,99))
            self.assertEqual(r['status'],'PARTIAL_OR_TECHNICAL_FAILED')
            self.assertIn('W100',(a/'collector/report-ko.md').read_text())

    def test_launcher_and_source(self):
        a=Path('/attempt');args=argv('main',a,task='fe-sequential-10k')
        for x in ('--cpus-per-task=6','--mem=59392M','--time=48:00:00','--export=NONE','--no-requeue','--gres=gpu:1','--job-name=fe-sequential-10k'):self.assertIn(x,args)
        self.assertIn('project/run_scripts/experiment_tracking',SOURCES)
        script=launcher('main',a,'a'*40,dict(runtime=dict(python='/python')))
        self.assertNotIn('WANDB_API_KEY',script);self.assertNotIn('WANDB_DISABLED',script)
        source=(ROOT/'project/run_scripts/fe_baseline/run.py').read_text()
        self.assertLess(source.index('telemetry.start('),source.index('AutoModelForCausalLM.from_pretrained('))
        calls=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertEqual(sum(n.func.attr=='fit' for n in calls),1)
        self.assertFalse(any(n.func.attr in ('save','save_pretrained') for n in calls))

if __name__=='__main__':unittest.main()
