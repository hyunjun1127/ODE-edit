"""Small randomly initialized CPU model fixtures, never actual model PASS."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from transformers import AutoTokenizer,LlamaConfig,LlamaForCausalLM
from .common import state
from .inputs import CounterFactAdapter
from .physical import LlamaAdapter
from .entry import prepare_entry,prepare_reference
from .oracle import Oracle
from .writer import Transaction,commit
from .memory import NativeMemory

class IntegrationTests(unittest.TestCase):
    def test_entry_dense_direct_commit_memory_next_entry(self):
        torch.set_num_threads(2);torch.manual_seed(11)
        tok=AutoTokenizer.from_pretrained('/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2',local_files_only=True)
        tok.pad_token=tok.eos_token;tok.padding_side='right'
        config=LlamaConfig(vocab_size=len(tok),hidden_size=16,intermediate_size=24,num_hidden_layers=3,
            num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=256,attention_dropout=0.)
        config._attn_implementation='eager'
        a=LlamaAdapter(LlamaForCausalLM(config).float().eval(),dict(eligible_layers=[1,2],nll_layer=2,
            lambda_C=15000.,kl_factor=.0625,preservation_K=.0625,preservation_E=1.,norm_factor=.5,clamp_factor=.75))
        bench=CounterFactAdapter(tok,[['{}'],['Therefore {}','It is known that {}']])
        records=json.loads(Path('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json').read_text())[2000:2004]
        h={l:torch.zeros(24,24) for l in a.sites};memory=NativeMemory();before=state(a,h)
        with tempfile.TemporaryDirectory() as tmp:
            stats={}
            for l in a.sites:
                p=Path(tmp)/f'{l}.npz';np.savez(p,**{'mom2.mom2':np.eye(24,dtype=np.float32)*2,'mom2.count':np.array(2)})
                stats[str(l)]=str(p)
            e=prepare_entry(a,bench,bench.prepare(records[:2]),h,stats,2)
            v={l:torch.randn(16,2)*.01 for l in a.sites}
            d=Oracle(a,e,[],[],0,'dense',False)(v,True)
            c=Oracle(a,e,[],[],0,'direct',True)(v,True)
            self.assertLess(abs(c['smooth']-d['smooth']),1e-4)
            for l in a.sites:torch.testing.assert_close(c['grad'][l],d['grad'][l],atol=1e-6,rtol=1e-3)
            with Transaction(a,h,memory) as tx:
                memory.sample([],2);r=commit(a,h,memory,e,c['payload'],records[:2]);tx.finish()
            self.assertEqual(len(memory),2)
            for x in h.values():self.assertTrue(torch.equal(x,x.T));self.assertGreater(float(x.norm()),0)
            self.assertNotEqual(state(a,h),before)
            e2=prepare_entry(a,bench,bench.prepare(records[2:]),h,stats,1)
            refs=memory.sample([],2);rg=prepare_reference(a,bench,refs,1)
            v2={l:torch.zeros(16,2) for l in a.sites}
            b=Oracle(a,e2,refs,rg,1,'direct',True)(v2,True)
            self.assertTrue(all(torch.isfinite(g).all() for g in b['grad'].values()))
            before2=state(a,h);mbefore=memory.summary()
            with Transaction(a,h,memory):
                with torch.no_grad():a.weights[1].add_(1);h[1].add_(1)
            self.assertEqual(state(a,h),before2);self.assertEqual(memory.summary(),mbefore)

if __name__=='__main__':unittest.main()
