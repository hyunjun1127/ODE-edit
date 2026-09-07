"""Small CPU checks: metric direction, sample ordering, state, native observer."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from .integrity import content, digest, restore, save, signature
from .evaluation import reduce
from .observer import observe


class Focused(unittest.TestCase):
    def test_metric_pair_direction_ties(self):
        def row(n):
            return dict(case_id=1,prompt_index=0,prompt='p',target='t',nll=n,all_tokens_correct=False,token_correct=[False])
        for tag in ['rewrite','rephrase','locality']:
            for new,true,expect in [(1,2,tag!='locality'),(2,1,tag=='locality'),(1,1,False)]:
                r=reduce({tag+'_target_new':[row(new)],tag+'_target_true':[row(true)]})
                self.assertEqual(next(iter(r.values()))['numerator'],int(expect))
    def test_pair_order_fail(self):
        a=dict(case_id=1,prompt_index=0,prompt='a',nll=1)
        with self.assertRaises(AssertionError):reduce({'rewrite_target_new':[a],'rewrite_target_true':[dict(a,prompt='b')]})
    def test_atomic_restore(self):
        w={'w':torch.ones(2,3)}; c=torch.zeros(2,3,3); before=signature(w,c); s={'w':w['w'].clone()}; m=c.clone()
        w['w'].add_(2);c.add_(4); restore(w,s,c,m)
        self.assertEqual(content(before),content(signature(w,c)))
        self.assertEqual(before['weights']['w']['pointer'],w['w'].data_ptr())
    def test_create_once(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';save(p,{'value':1})
            with self.assertRaises(FileExistsError):save(p,{'value':2})
    def test_native_calls_and_finally(self):
        z=lambda *a:torch.ones(3)
        k=lambda *a:torch.ones(1,3)
        m=SimpleNamespace(compute_z=z,compute_ks=k);hp=SimpleNamespace(layers=[4,8]);rr=[{'case_id':1}]
        with observe(m,hp,{},None,rr) as rec:
            for l in hp.layers:m.compute_z(None,None,rr[0],hp,l,None);m.compute_ks(None,None,rr,hp,l,None)
            for l in hp.layers:m.compute_ks(None,None,rr,hp,l,None)
        self.assertEqual(rec['compute_z'],2);self.assertIs(m.compute_z,z);self.assertIs(m.compute_ks,k)
        with self.assertRaises(ValueError):
            with observe(m,hp,{},None,rr):raise ValueError()
        self.assertIs(m.compute_z,z)


if __name__=='__main__':unittest.main()
