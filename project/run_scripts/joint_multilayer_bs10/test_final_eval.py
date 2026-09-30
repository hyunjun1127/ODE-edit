import unittest
from .final_eval import rows_for

class Tok:
    bos_token_id=1;unk_token_id=0
    def __call__(self,text,add_special_tokens=True):return {'input_ids':[3,4]}
    def encode(self,text,add_special_tokens=False):return [5]

class TestEval(unittest.TestCase):
    def test_inventory(self):
        r=dict(case_id=1,requested_rewrite=dict(prompt='{} is',subject='x',target_new={'str':'new'},target_true={'str':'old'}),
            paraphrase_prompts=['p0','p1'],neighborhood_prompts=['n'+str(i) for i in range(10)])
        rows=rows_for(Tok(),r)
        self.assertEqual(len(rows),26)
        self.assertEqual({k:sum(x['kind']==k for x in rows) for k in ('R','P','N')},{'R':2,'P':4,'N':20})
        self.assertEqual(len({x['pair_id'] for x in rows}),13)
        self.assertTrue(all(x['positions']==[1] for x in rows))
    def test_missing_neighborhood_blocks(self):
        r=dict(paraphrase_prompts=['p0','p1'],neighborhood_prompts=[],requested_rewrite={})
        with self.assertRaises(RuntimeError):rows_for(Tok(),r)

if __name__=='__main__':unittest.main()
