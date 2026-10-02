"""CPU-only registration/partial collection contracts; no scheduler/model I/O."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from .common import write,sha
from .collect import collect,paired
from .submit import arguments,dependencies,dependency_members,expected_dependencies

class Pipeline(unittest.TestCase):
    def test_all_modules_parse(self):
        for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_dependency_and_resources(self):
        r=dict(host_mib=60416,collector_host_mib=24576,qualification_wall='24:00:00',wall='7-00:00:00',collector_wall='04:00:00')
        dep=dependencies(('afterok',['1','2']),('afterany',['3']))
        self.assertEqual(dependency_members('afterok:1(unfulfilled):2(unfulfilled),afterany:3(unfulfilled)'),expected_dependencies(dep))
        for name in ['prep-A','prep-B','main-A','main-B','collector']:
            args=arguments(name,dep,Path('/tmp/scoped-fixture'),r)
            self.assertIn('--hold',args);self.assertIn('--export=NONE',args);self.assertIn('--no-requeue',args)
            self.assertEqual('--gres=gpu:1' in args,name!='collector')
            self.assertIn('--kill-on-invalid-dep=yes',args)
    def test_prep_failure_collects_partial(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);write(p/'w0.json',dict(rows=[]))
            write(p/'bridge.json',dict(observations=dict(path=str(p/'w0.json'),sha256=sha(p/'w0.json'))))
            write(p/'config.json',dict(w0_reuse=dict(receipt=dict(path=str(p/'bridge.json')))))
            write(p/'execution.lock.json',dict(source_commit='fixture'))
            collect(p,p/'report')
            result=json.loads((p/'report/terminal.json').read_text())
            self.assertEqual(result['status'],'PARTIAL_OR_TECHNICAL_FAILED');self.assertEqual(result['main_commits'],0)
    def test_paired_ties_lost_gained(self):
        def row(i,t,n):return dict(identity=str(i),kind='N',true_nll=t,new_nll=n,true_token_count=1,new_token_count=1)
        result=paired([row(1,1,2),row(2,2,1)],[row(1,1,1),row(2,1,2)])['N']
        self.assertEqual((result['denominator'],result['lost'],result['gained']),(2,1,1))

if __name__=='__main__':unittest.main()
