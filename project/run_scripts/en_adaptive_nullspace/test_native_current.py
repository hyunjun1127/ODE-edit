"""CPU contracts; no claim of actual Llama/T0 numerical equivalence."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from .current import complete_packs, weighted_columns, digest
from .native import requests_from_records, HOOK_SHA256
from project.run_scripts.single_layer_mechanism_first import z_hook


def cache(values):
    return SimpleNamespace(keys=torch.tensor([values], dtype=torch.float32))


def identity(ids, policy=None):
    value=dict(input_ids=ids, attention_mask=[1]*len(ids),
        position_ids=list(range(len(ids))), tokenizer=policy or {'padding':'none','bos':True})
    return dict(sha256=digest(value), **value)


def row(case, ci, name):
    return dict(case_id=case, cache=ci, sequence_id=name, kind='native', branch='new',context=0)


class Contracts(unittest.TestCase):
    def test_source_identity(self):
        self.assertEqual(hashlib.sha256(Path(z_hook.__file__).read_bytes()).hexdigest(), HOOK_SHA256)

    def test_request_copy_and_order(self):
        records=[dict(case_id=i, requested_rewrite={'target_new':{'str':str(i)}}) for i in [4,2]]
        requests=requests_from_records(records)
        self.assertEqual([r['case_id'] for r in requests],[4,2])
        requests[0]['target_new']['str']='changed'
        self.assertEqual(records[0]['requested_rewrite']['target_new']['str'],'4')

    def test_nested_weight_duplicate_and_length_invariance(self):
        # Request A: two unique inputs, first repeated twice. Request B: one
        # longer input. Each request has .5 total irrespective of lengths.
        caches=[cache([[1.,0.],[0.,1.]]),cache([[1.,0.]]),cache([[2.,0.],[0.,2.],[2.,2.]])]
        identities=[identity([1,2]), identity([1]), identity([3,4,5])]
        rows=[row('a',0,'a0'),row('a',0,'a0dup'),row('a',1,'a1'),row('b',2,'b0')]
        out=weighted_columns(caches,rows,identities,['a','b'])
        self.assertAlmostEqual(float(out['weights'].sum()),1.)
        for receipt in out['manifest']['request_weights']:
            self.assertAlmostEqual(receipt['total_weight'],.5)
        # Shared prefix first key receives .125+.25; duplicate row adds no mass.
        self.assertAlmostEqual(float(out['weights'][0]),.375)
        reduced=weighted_columns(caches,[rows[0],rows[2],rows[3]],identities,['a','b'])
        self.assertTrue(torch.equal(out['K'],reduced['K']))
        self.assertTrue(torch.allclose(out['weights'],reduced['weights'],atol=1e-16,rtol=0))

    def test_byte_variants_preserved_with_first_representative(self):
        caches=[cache([[1.,0.]]),cache([[1.+2**-20,0.]])]
        out=weighted_columns(caches,[row(1,0,'x'),row(1,1,'y')],
                             [identity([1]),identity([1])],[1])
        self.assertEqual(out['K'].shape,(2,2))
        self.assertEqual(out['representative_indices'].tolist(),[0,0])
        self.assertEqual(out['manifest']['byte_variant_columns'],1)

    def test_signed_zero_counts_as_distinct_bytes(self):
        caches=[cache([[0.,1.]]),cache([[-0.,1.]])]
        out=weighted_columns(caches,[row(1,0,'x'),row(1,1,'y')],
                             [identity([1]),identity([1])],[1])
        self.assertEqual(out['K'].shape[1],2)

    def test_policy_disambiguates_full_inputs(self):
        p=dict(input_ids=torch.tensor([[1,2]]),attention_mask=torch.ones((1,2),dtype=torch.long),
               position_ids=torch.tensor([[0,1]]))
        rows=[row(1,0,'x'),dict(row(1,0,'y'),kind='canonical')]
        packs,newrows,ids=complete_packs([p],rows,dict(native={'bos':False},canonical={'bos':True}))
        self.assertEqual(len(packs),2)
        self.assertNotEqual(ids[0]['sha256'],ids[1]['sha256'])
        self.assertEqual([r['cache'] for r in newrows],[0,1])

    def test_missing_request_rejected(self):
        with self.assertRaisesRegex(ValueError,'MISSING_CURRENT_REQUEST'):
            weighted_columns([cache([[1.,0.]])],[row('a',0,'x')],[identity([1])],['a','b'])

    def test_nonfinite_keys_rejected(self):
        with self.assertRaisesRegex(ValueError,'FINITE_FP32'):
            weighted_columns([cache([[float('nan'),0.]])],[row('a',0,'x')],[identity([1])],['a'])


if __name__=='__main__':
    unittest.main()
