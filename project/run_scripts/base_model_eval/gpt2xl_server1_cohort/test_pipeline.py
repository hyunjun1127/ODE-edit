"""Production assembly CPU fixtures; fake scores, no model/SDK/Slurm call."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .common import REFERENCE_CONFIG,SCHEMA,read,write
from . import collect as collector,run as runner,submit
from .curves import DEFAULT_CONFIG
from .test_curves import fixture_rows


class FakeView:
    def __init__(self):
        self.calls=0;self.physical_candidate_rows=0;self.version=0
    def guard(self):return {'weight':(11,self.version,(6400,1600),'FP32')}
    def selected_state(self):return {'13':'sealed-cold-weight'}
    def hooks(self):return {'model':((),(),())}


class FakeTracker:
    last_rejection={'code':'INJECTED_CPU_REJECTION','point_not_enqueued':True}
    def __init__(self,accept=False):self.payloads=[];self.accept=accept
    def log(self,payload):self.payloads.append(copy.deepcopy(payload));return self.accept


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows,cls.expected=fixture_rows()

    def observe(self,root,view=None,tracker=None,mutate=False,nonfinite=False):
        view=view or FakeView()
        def pairs(records,bench):
            start=records[0]*13;stop=(records[-1]+1)*13
            specs=[{key:row[key] for key in ('identity','case_id','kind','prompt_index','endpoint')}
                   for row in self.rows[start:stop]]
            return specs,[row for row in self.rows[start:stop] for _ in (0,1)]
        def scores(actual,bench,pairs,microbatch):
            self.assertEqual(microbatch,2);actual.calls+=len(pairs)//2
            actual.physical_candidate_rows+=len(pairs)
            values=[]
            for index,row in enumerate(pairs):
                label='new' if index%2==0 else 'true'
                values.append({key:row[label+'_'+key] for key in
                    ('nll','token_count','token_correct','strict','token_identity')})
            if mutate:actual.version+=1
            if nonfinite:values[0]['nll']=float('nan')
            return values
        with patch.object(runner,'pair_specs',side_effect=pairs),patch.object(runner,'capacity',return_value={}), \
             patch('project.run_scripts.jlz_price_gpt2xl.scores.scores',side_effect=scores):
            return runner.observe(view,None,list(range(2000)),self.expected,root,tracker)

    def test_production_observer_once_and_cpu_emit_rejections_are_not_silent(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);tracker=FakeTracker()
            raw,summary=self.observe(root,tracker=tracker)
            self.assertEqual(summary['physical_candidate_rows'],52000)
            self.assertEqual(summary['physical_forward_calls'],26000)
            self.assertEqual(len(list(root.glob('chunk-*.json'))),40)
            self.assertEqual([row['occurrence_ordinal'] for row in raw[650:663]],[50]*13)
            self.assertEqual(len(tracker.payloads),40)
            self.assertTrue(all('edits' not in value for value in tracker.payloads))
            result=runner.emit_curves(raw,self.expected,dict(reference_config=REFERENCE_CONFIG),root,tracker)
            self.assertEqual(result['status'],'LOGGING_DEGRADED')
            self.assertEqual(len(result['acceptances']),21)
            self.assertTrue(all(not value['accepted'] and value['rejection'] for value in result['acceptances']))
            self.assertEqual(result['new_model_forward_calls'],0)
            self.assertEqual([p['edits'] for p in tracker.payloads[40:]],list(range(0,2001,100)))

    def test_production_guard_and_nonfinite_block_without_chunks(self):
        for mode in ('mutate','nonfinite'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                with self.assertRaises((ValueError,RuntimeError)):
                    self.observe(Path(temp),**{mode:True})
                self.assertEqual(list(Path(temp).glob('chunk-*.json')),[])

    def test_collector_assembled_complete_and_corruption_is_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'W0';out.mkdir()
            raw,summary=self.observe(out)
            runner.emit_curves(raw,self.expected,dict(reference_config=REFERENCE_CONFIG),out,FakeTracker(True))
            write(root/'identity.json',dict(rows=self.expected))
            config=dict(attempt=str(root),observer_identity=runner.member(root/'identity.json'),
                        cold_selected_W=summary['selected_W'],reference_config=REFERENCE_CONFIG)
            lock=dict(source_commit='a'*40,config_sha256='b'*64)
            write(out/'terminal.json',dict(status='COMPLETED',source='a'*40,config='b'*64))
            with patch.object(collector,'verify_lock',return_value=(config,lock)):
                result=collector.collect('unused','unused')
            self.assertEqual(result['status'],'COMPLETED')
            self.assertEqual(result['curve_coverage']['current_coverage'],20)
            self.assertEqual(result['curve_coverage']['all_seen_coverage'],4)
            self.assertEqual(result['new_forward_calls_by_reducer'],0)
            self.assertEqual(read(root/'collector/terminal.json')['status'],'COMPLETED')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'W0';out.mkdir()
            bad=copy.deepcopy(self.rows[:650]);bad[0]['new_nll']=float('inf')
            # JSON writer rejects nonfinite before the collector can see it.
            with self.assertRaises(ValueError):write(out/'bad.json',bad)
            self.assertFalse((out/'bad.json').exists())

    def test_collector_partial_keeps_missing_unmeasured(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'W0').mkdir();write(root/'identity.json',dict(rows=self.expected))
            config=dict(attempt=str(root),observer_identity=runner.member(root/'identity.json'),
                        cold_selected_W={},reference_config=REFERENCE_CONFIG)
            with patch.object(collector,'verify_lock',return_value=(config,dict(source_commit='a'*40,config_sha256='b'*64))):
                result=collector.collect('unused','unused')
            self.assertEqual(result['status'],'PARTIAL_OR_FAILED')
            self.assertEqual(result['curve_coverage']['status'],'NOT_MEASURED')
            self.assertEqual(result['independently_reduced'],{})

    def test_production_argv_real_dependencies_and_no_cpu_gpu_request(self):
        r=dict(cpu=8,host_mib=65536,wall='04:00:00',collector_cpu=8,collector_host_mib=24576,
               collector_wall='02:00:00',dependency=['60156','60157'])
        gpu=submit.argv_for('GPU',Path('/tmp/cpu-fixture'),r)
        cpu=submit.argv_for('collector',Path('/tmp/cpu-fixture'),r,'7654321')
        self.assertIn('--dependency=afterany:60156:60157',gpu)
        self.assertIn('--dependency=afterany:7654321',cpu)
        self.assertIn('--gres=gpu:1',gpu);self.assertFalse(any('gres' in item for item in cpu))
        self.assertIn('--export=NONE',cpu);self.assertIn('--no-requeue',gpu)
        with self.assertRaises(RuntimeError):submit.argv_for('collector',Path('/tmp/cpu-fixture'),r)


def cpu_gate(path):
    from .test_curves import CohortCurvesTests
    from .test_tracking import W0TrackingTests
    stream=io.StringIO();suite=unittest.TestSuite()
    for case in (CohortCurvesTests,W0TrackingTests,PipelineTests):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
    with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    value=dict(status='CPU_PASS_NOT_MODEL_OR_ONLINE_PASS' if result.wasSuccessful() else 'FAILED',
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        skipped=len(result.skipped),GPU=False,model_loaded=False,Slurm=False,SDK_online=False,
        fixture_only=True,production_functions=True,output=stream.getvalue())
    if path is not None:write(path,value)
    return value


if __name__=='__main__':unittest.main()
