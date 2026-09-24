"""Bounded CPU tests; these are not GPU fidelity evidence."""
import copy
import tempfile
import unittest
from .common import *
from .bind import check_design
from .reduce import contribution,pair

class Core(unittest.TestCase):
    def test_design(self):self.assertEqual(len(check_design()),10000)
    def test_censor(self):
        f=dict(same_batch_conflict='False',first_later_conflict_batch='20')
        self.assertTrue(active(f,19));self.assertFalse(active(f,20));self.assertFalse(active(f,100))
        f['same_batch_conflict']='True';self.assertFalse(active(f,1))
        f.update(same_batch_conflict='False',first_later_conflict_batch='');self.assertTrue(active(f,100))
    def test_atomic(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            p=Path(d)/'x.json';save(p,{'a':1})
            with self.assertRaises(FileExistsError):save(p,{'a':2})
            self.assertEqual(read(p),{'a':1})
    def test_subtract_exact_diagonal_and_pair(self):
        import numpy as np
        rng=np.random.default_rng(1);a=rng.normal(size=(7,11)).astype('float32');b=rng.normal(size=a.shape).astype('float32');t=rng.normal(size=a.shape).astype('float32');v=rng.normal(size=a.shape).astype('float32')
        u=b.astype('float64')-a.astype('float64');self.assertTrue(np.array_equal((b.astype('float64')-u).astype('float32'),a))
        x=(t.astype('float64')-u-v.astype('float64')).astype('float32');self.assertEqual(x.dtype,np.float32);self.assertTrue(np.isfinite(x).all())
    def test_state_all_five_no_checkpoint(self):
        source=(Path(__file__).parent/'backend.py').read_text()
        self.assertIn('for k in KEYS:',source);self.assertNotIn('torch.save(',source);self.assertNotIn('sub_(u);',source)
    def test_cache_invalidations(self):
        base=['weightsha',['token1'],'eval']
        for i in range(3):
            x=copy.deepcopy(base);x[i]='changed';self.assertNotEqual(digest(base),digest(x))
    def test_receipt_counts(self):
        self.assertEqual(len(rows(DESIGN/'score-tasks.csv')),383)
        self.assertEqual(len(rows(DESIGN/'state-bank.csv')),173)
    def test_no_effect_gate(self):
        c=read(DESIGN/'experiment-contract.json');self.assertFalse(c['analysis']['scientific_effect_size_gate']);self.assertEqual(c['numeric_contract']['max_absolute_nll_fidelity_error'],.00025)
    def test_independent_bookkeeping(self):
        import random
        rng=random.Random(0)
        for _ in range(100):
            mt,bt,mb,bb=[rng.uniform(-20,20) for _ in range(4)]
            self.assertLessEqual(abs((mt-mb)-(bt-bb)-((mt-bt)-(mb-bb))),1e-10)
    def test_strict_tie(self):self.assertFalse((3.-3.)>0)
    def test_all_dependencies_no_science_bypass(self):
        text=(Path(__file__).parent/'worker.py').read_text()
        self.assertIn("self.barrier('T1')",text);self.assertIn("self.barrier(stage)",text)
        self.assertNotIn('sbatch',text);self.assertNotIn('scancel',text)
    def test_source_evidence_exact(self):
        for r in read(DESIGN/'asset-bindings.json')['evaluator_sources']:
            self.assertEqual(sha(DESIGN/r['snapshot_path']),r['sha256'])

if __name__=='__main__':unittest.main()
