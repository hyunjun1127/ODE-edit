"""CPU production-boundary regressions; synthetic Llama is not pretrained PASS."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from .profile import LlamaAdapter
from .inputs import CounterFactAdapter
from .entry import prepare_entry
from .causal_builder import build
from .subject import evaluate,COEF
from .qualification import fixed,one,singletons,compare,full_masked,actual_commit_probe
from .terminal import observe as terminal_observe
from .writer import Transaction,commit
from .common import state,write,serial
from .optimize import fit
from .allocation import Root
from .observe import reduce_rows,active_flags,observe

torch.set_num_threads(1)


class Tokens:
    bos_token_id=1;unk_token_id=0;eos_token_id=2;pad_token_id=2;padding_side='right'
    def encode(self,text,add_special_tokens=True):return ([1] if add_special_tokens else [])+[3+ord(c)%90 for c in text]
    def decode(self,ids):return ''.join(chr(int(x)-3+90 if int(x)<35 else int(x)-3) for x in ids)
    def __call__(self,text,return_tensors=None,padding=False,add_special_tokens=True):
        many=isinstance(text,list);rows=[self.encode(x,add_special_tokens) for x in (text if many else [text])]
        if return_tensors:
            width=max(map(len,rows));return {'input_ids':torch.tensor([x+[2]*(width-len(x)) for x in rows]),'attention_mask':torch.tensor([[1]*len(x)+[0]*(width-len(x)) for x in rows])}
        return dict(input_ids=rows if many else rows[0])


def record(i):
    return dict(case_id=i,requested_rewrite=dict(subject=['Joe','Anna','Li'][i%3],relation_id='r',prompt='{} was',
                target_new=dict(str=' B',id='b'),target_true=dict(str=' C',id='c')),
                paraphrase_prompts=['Where was '+str(i),'Thing '+str(i)],neighborhood_prompts=['Other '+str(i)])


def fixture(folder,B=2):
    torch.manual_seed(913)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=96,hidden_size=16,intermediate_size=32,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,attention_dropout=0.,_attn_implementation='eager'))
    a=LlamaAdapter(model,dict(eligible_layers=[0,1,2],nll_layer=3,lambda_C=15000.,kl_factor=.0625))
    bench=CounterFactAdapter(Tokens(),[['{}'],['The {}','A {}']])
    h={l:torch.zeros(32,32) for l in a.sites};stats={}
    for l in a.sites:
        p=folder/f'stat{l}.npz';np.savez(p,**{'mom2.mom2':np.eye(32,dtype=np.float32),'mom2.count':np.array(3)})
        stats[str(l)]=str(p)
    entry=prepare_entry(a,bench,bench.prepare([record(i) for i in range(B)]),h,stats,2)
    return a,bench,h,entry,stats


class ProductionTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name);self.a,self.b,self.h,self.e,self.stats=fixture(self.path)
    def tearDown(self):self.tmp.cleanup()
    def test_dense_and_staged_total_gradient(self):
        left=one(self.a,self.e);self.a.checkpoint_enabled=False;right=one(self.a,self.e,'dense',True)
        self.assertLess(abs(left['loss']-right['loss']),1e-5)
        self.assertTrue(all(x['passed'] for x in compare(left['gradient'],right['gradient']).values()))
    def test_microbatch_permutation_whole_B(self):
        left=one(self.a,self.e);right=one(self.a,singletons(self.e))
        self.assertTrue(all(x['passed'] for x in compare(left['gradient'],right['gradient']).values()))
    def test_full_block_native_hook_parity(self):
        left=one(self.a,self.e)
        with full_masked(self.a):right=one(self.a,self.e)
        self.assertTrue(all(x['passed'] for x in compare(left['gradient'],right['gradient']).values()))
    def test_zero_subgradient_finite(self):
        R=fixed(self.a,self.e,0.);built=build(self.a,self.e,R,1);r=evaluate(self.a,self.e,R,built,'B',components=True)
        for x in r['components']['norm'].values():self.assertEqual(float(x.norm()),0.)
        for x in r['components']['allocation'].values():self.assertEqual(float(x.norm()),0.)
    def test_components_sum(self):
        R=fixed(self.a,self.e);built=build(self.a,self.e,R,2);r=evaluate(self.a,self.e,R,built,'A',components=True)
        for l in R:torch.testing.assert_close(sum(COEF[k]*g[l] for k,g in r['components'].items()),r['gradient'][l],atol=1e-6,rtol=1e-3)
    def test_actual_KL_and_rewrite_mapping(self):
        R=fixed(self.a,self.e);built=build(self.a,self.e,R,2)
        self.assertEqual(len(built['v'][0]),8);self.assertEqual(built['geometry'][0]['K'].shape,(32,2))
        self.assertEqual(sum(r['kind']=='kl' for r in built['rows']),2)
    def test_geometry_negative_control(self):
        # Synthetic random Llama keys are tiny: use a dimensionless toy prior
        # to resolve the stopped-path signal. Production lambda_C is unchanged.
        from .geometry import prior
        self.e['factors']={l:prior(self.stats[str(l)],self.h[l],'cpu',.001)[0] for l in self.a.sites}
        self.e['first_geometry']={}
        left=one(self.a,self.e);right=one(self.a,self.e,stop_geometry=True)
        self.assertGreater(sum(float((left['gradient'][l]-right['gradient'][l]).norm()) for l in left['gradient']),0.)
    def test_transaction_H_once_and_rollback(self):
        before=state(self.a,self.h);meta={'ledger':[]};R=fixed(self.a,self.e)
        with torch.no_grad():built=build(self.a,self.e,R,25);payload,diag=terminal_observe(self.a,self.e,R,built)
        contexts=[['original']]
        with Transaction(self.a,self.h,meta,contexts) as tx:
            applied=commit(self.a,self.h,self.e,payload);meta['ledger'].append(1)
            contexts[0].append('injected')
            for l in self.h:torch.testing.assert_close(self.h[l],payload['keys'][l]@payload['keys'][l].T,atol=0,rtol=0)
            with self.assertRaisesRegex(RuntimeError,'DUPLICATE'):commit(self.a,self.h,self.e,payload)
            actual_commit_probe(self.a,self.e,payload,self.path/'probe')
        self.assertTrue(tx.rollback_verified);self.assertEqual(state(self.a,self.h),before);self.assertEqual(meta,{'ledger':[]})
        self.assertEqual(contexts,[['original']])
    def test_observer_no_mutation_and_metrics(self):
        result=observe(self.a,self.b,[record(0)],[record(0)],self.h,0,self.path/'obs',2)
        self.assertTrue(result['no_mutation']);self.assertEqual(result['summary']['P']['denominator'],2)
    def test_25_candidates_terminal_backward_24updates_noCP(self):
        payload,result=fit(self.a,self.e,'B',self.path/'fit',1)
        self.assertEqual(result['candidates'],25);self.assertEqual(result['Adam_updates'],24)
        last=json.loads((self.path/'fit/candidate-25.json').read_text());self.assertTrue(last['terminal_backward']);self.assertFalse(last['terminal_update'])
        self.assertEqual(len(list((self.path/'fit').glob('candidate-*.json'))),25);self.assertFalse(list(self.path.rglob('*.pt')))
    def test_partial_B1_B3(self):
        for B in (1,3):
            a,b,h,e,stats=fixture(self.path,B);r=one(a,e);self.assertTrue(all(torch.isfinite(g).all() for g in r['gradient'].values()))
    def test_nonfinite_receipt_rejected(self):
        with self.assertRaises(ValueError):write(self.path/'bad.json',dict(x=float('nan')))
        self.assertFalse((self.path/'bad.json').exists())
    def test_atomic_conflict_preserves_old(self):
        p=self.path/'once.json';write(p,dict(a=1));old=p.read_bytes()
        with self.assertRaises(RuntimeError):write(p,dict(a=2))
        self.assertEqual(p.read_bytes(),old)
    def test_active_overwrite(self):
        x=record(0);y=record(0);y['case_id']=1;y['requested_rewrite']['target_new']={'str':' D','id':'d'}
        self.assertEqual(active_flags([x,y]),{0:False,1:True})


if __name__=='__main__':unittest.main()
