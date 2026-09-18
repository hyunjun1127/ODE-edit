"""CPU-only admission/schema tests, no scheduler or source mutation."""
import tempfile
from pathlib import Path
import unittest
from project.run_scripts.single_layer_edit_preserving_correction.admission import inspect_M,function_hash

class AdmissionTests(unittest.TestCase):
    def test_exact_held_M_and_memory_equivalence(self):
        lock={'source_root':'/task/source'};path=Path('/task/execution.lock.json')
        text='UserId=janghj JobState=PENDING Reason=JobHeldUser NumCPUs=8 Requeue=0 NodeList=server4 mem=59G gres/gpu=1 ArrayTaskId=0-9 ArrayTaskThrottle=2 Command=/task/source/project/run_scripts/single_layer_edit_preserving_correction/run.sbatch /task/source /task/execution.lock.json M'
        inspect_M(text,lock,path,2)
        inspect_M(text.replace('mem=59G','mem=60416M'),lock,path,2)
        for old,new in (('mem=59G','mem=60G'),('ArrayTaskThrottle=2','ArrayTaskThrottle=3'),(' M',' S'),('Requeue=0','Requeue=1')):
            with self.assertRaises(ValueError):inspect_M(text.replace(old,new),lock,path,2)
    def test_AST_binding_ignores_comments_not_arithmetic(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a.py';b=Path(d)/'b.py';c=Path(d)/'c.py'
            a.write_text('def f(x):\n return x+1\n');b.write_text('# comment\ndef f(x):\n return x+1\n')
            c.write_text('def f(x):\n return x+2\n')
            self.assertEqual(function_hash(a,'f'),function_hash(b,'f'))
            self.assertNotEqual(function_hash(a,'f'),function_hash(c,'f'))

if __name__=='__main__':unittest.main()
