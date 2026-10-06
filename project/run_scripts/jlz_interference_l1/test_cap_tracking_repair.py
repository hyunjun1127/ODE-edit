"""CPU replay of the failed real candidate/endpoint; not a scientific toy/fit."""
import copy,json,math,tempfile,unittest
from pathlib import Path
from .cap_tracking import TrackedEvents,candidate_metrics,log_endpoint
from project.run_scripts.experiment_tracking.schema import metrics

RAW=Path('/data/janghj/ODE-edit/local/jlz-price-cap-base-repair-2k/attempt/LLAMA_CAP075/batch-01')
class Sink:
    batch=1
    def __init__(self):self.rows=[]
    def emit(self,event,payload):self.rows.append((event,payload))
class Tracker:
    def __init__(self,spool):self.spool=Path(spool);self.points=[]
    def log(self,point):self.points.append(metrics(point));return True

class RecordedLoggingRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (RAW/'events.jsonl').open() as f:cls.payload=json.loads(next(f))['payload']
        cls.endpoint=json.loads((RAW/'pre/summary.json').read_text())['summary']
    def test_actual_B100_context_reduction_and_no_mutation(self):
        p=copy.deepcopy(self.payload);before=copy.deepcopy(p)
        self.assertEqual((len(p['nll']),len(p['nll'][0])),(100,6))
        with self.assertRaises(TypeError):sum(p['nll'])
        row=candidate_metrics(p,1);metrics(row)
        self.assertAlmostEqual(row['fit/nll'],sum(map(sum,p['nll']))/600)
        self.assertTrue(math.isclose(row['fit/nll']+.0625*row['fit/kl'],sum(p['F'])/100,abs_tol=2e-5))
        self.assertEqual(p,before)
    def test_authoritative_once_and_all_endpoint_fields(self):
        with tempfile.TemporaryDirectory() as d:
            sink=Sink();tracker=Tracker(d);TrackedEvents(sink,tracker).emit('candidate',self.payload)
            self.assertEqual(len(sink.rows),1);self.assertEqual(len(tracker.points),1)
            self.assertTrue(log_endpoint(tracker,self.endpoint,1,100))
            self.assertEqual(len(tracker.points),5)
            self.assertEqual([r['phase_id'] for r in tracker.points],[1,2,3,4,5])
    def test_caller_conversion_failure_isolated_and_receipted(self):
        with tempfile.TemporaryDirectory() as d:
            p=copy.deepcopy(self.payload);p['nll']=[[]]*100;sink=Sink();tracker=Tracker(d)
            TrackedEvents(sink,tracker).emit('candidate',p)
            self.assertEqual(len(sink.rows),1);self.assertFalse(tracker.points)
            self.assertEqual(json.loads((Path(d)/'caller-receipt.json').read_text())['status'],'LOGGING_DEGRADED_CALLER')
    def test_transport_exception_isolated(self):
        with tempfile.TemporaryDirectory() as d:
            tracker=Tracker(d)
            def broken(_):raise OSError('test transport')
            tracker.log=broken;TrackedEvents(Sink(),tracker).emit('candidate',self.payload)
            self.assertEqual(tracker.caller_dropped_points,1)
    def test_authoritative_IO_not_swallowed(self):
        class BrokenSink(Sink):
            def emit(self,*_):raise OSError('authoritative write')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OSError):TrackedEvents(BrokenSink(),Tracker(d)).emit('candidate',self.payload)

if __name__=='__main__':unittest.main()
