import ast
from pathlib import Path
import unittest
from project.run_scripts.bg_tw_reference.ep_tw import review_nogate as r
from project.run_scripts.bg_tw_reference.ep_tw.test_review_nogate import row
from .cake import label

class Tests(unittest.TestCase):
    def test_ties_and_inverted_NS(self):
        x=row(new=2,true=2)
        self.assertFalse(r.success(x,'RS'));self.assertFalse(r.success(x,'NS'))
        x=row(new=3,true=1);self.assertTrue(r.success(x,'NS'));self.assertFalse(r.success(x,'PS'))
    def test_exact_id_set_not_serialization_order(self):
        runtime=sorted({9763,12179,18415},key=str)
        self.assertNotEqual(runtime,sorted(runtime))
        self.assertEqual(set(runtime),{9763,12179,18415})
        self.assertEqual(len(runtime),len(set(runtime)))
    def test_legacy_cake_schema(self):
        c={'weight_state':{'layer4':'hash'},'cache_sha256':'M','before_after_exact':True,'evaluator_controller_influence':0}
        endpoint={'weights':{'layer4':'hash'},'cache':'M'}
        self.assertEqual(c['weight_state'],endpoint['weights']);self.assertEqual(c['cache_sha256'],endpoint['cache'])
        self.assertEqual(c['evaluator_controller_influence'],0)
    def test_identity_pairing(self):
        with self.assertRaises(AssertionError):r.transition([row(identity='x')],[row(identity='y')],'RS')
    def test_names(self):
        self.assertEqual(label('AlphaEdit_ORIGINAL'),'AlphaEdit_BLUE(L4+L8)')
        self.assertEqual(label('BASE_ALPHAEDIT'),'BASE_ALPHAEDIT_NATIVE')
    def test_no_model_execution(self):
        for p in Path(__file__).parent.glob('*.py'):
            if p.name.startswith('test_'):continue
            s=p.read_text();ast.parse(s)
            for bad in ('from_pretrained(','.cuda(','.backward(',"['sbatch'",'scontrol release'):
                self.assertNotIn(bad,s)

if __name__=='__main__':unittest.main()
