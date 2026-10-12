"""Real native function + tiny random CPU Qwen, no pretrained/GPU evidence."""
import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import Qwen2Config,Qwen2ForCausalLM
from official.diagnostics.fe_native.sequential_z import current_batch_targets
from official.diagnostics.fe_qwen_current import failure_probe as p
from official.experiments.checkpoint import rng_snapshot,rng_restore


class Batch(dict):
    def to(self,d):return Batch({k:v.to(d) for k,v in self.items()})


class Tokenizer:
    bos_token_id=30;unk_token_id=31
    def encode(self,s):return [{'is':7,'a':8}.get(w,int(w) if w.isdigit() else 9) for w in s.split()]
    def __call__(self,s,**kwargs):
        ids=[self.encode(t) for t in ([s] if isinstance(s,str) else s)];n=max(map(len,ids))
        return Batch(input_ids=torch.tensor([i+[0]*(n-len(i)) for i in ids]),
            attention_mask=torch.tensor([[1]*len(i)+[0]*(n-len(i)) for i in ids]))
    def decode(self,ids):return ' '.join(str(int(i)) for i in ids)


class ProbeTests(unittest.TestCase):
    def test_native_seed_flags_then_checkpoint_rng(self):
        import random
        root=Path(os.environ.get('FE_NATIVE_AUTHOR_ROOT','/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author'))
        if not root.exists():self.skipTest('pinned native fixture absent')
        tree=ast.parse((root/'algs/memit/FE-memit_main.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='set_random_seed')
        ns=dict(random=random,np=np,torch=torch,os=os)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'NATIVE_SEED','exec'),ns)
        state=rng_snapshot();before=p.state_digest(state)
        flags=(torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)
        old_hash_seed=os.environ.get('PYTHONHASHSEED')
        try:
            torch.backends.cudnn.benchmark=True;torch.backends.cudnn.deterministic=False
            p.restore_native_execution_state(ns['set_random_seed'],0,state)
            self.assertFalse(torch.backends.cudnn.benchmark);self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertEqual(p.state_digest(rng_snapshot()),before)
        finally:
            torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic=flags
            if old_hash_seed is None:os.environ.pop('PYTHONHASHSEED',None)
            else:os.environ['PYTHONHASHSEED']=old_hash_seed
            rng_restore(state)

    def test_actual_native_fit_supplier_probe_parity(self):
        root=Path(os.environ.get('FE_NATIVE_AUTHOR_ROOT','/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author'))
        if not root.exists():self.skipTest('pinned native fixture absent')
        spec=importlib.util.spec_from_file_location('probe_native_nethook',root/'util/nethook.py')
        nh=importlib.util.module_from_spec(spec);spec.loader.exec_module(nh)
        fn=next(n for n in ast.parse((root/'algs/memit/compute_z.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compute_z')
        fn.returns=None
        for a in fn.args.args:a.annotation=None
        tok=Tokenizer()
        def lookup(prompt,subject,tok,*a,**kw):return tok.encode(prompt.format(subject)).index(int(subject))
        ns=dict(torch=torch,np=np,nethook=nh,find_fact_lookup_idx=lookup)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'PINNED_NATIVE_FIT','exec'),ns)
        torch.set_num_threads(1);torch.manual_seed(11)
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=8,intermediate_size=16,
            num_hidden_layers=28,num_attention_heads=2,num_key_value_heads=2,use_cache=False)).to(torch.bfloat16).eval()
        cfg=OmegaConf.create(dict(gpu=0,lti=False,negetive_prompt_test=False,bs=3,llms=dict(
            layers=[4,5,6,7,8],lm_head_module='lm_head',ln_f_module='model.norm',midlayers=[15],
            layer_module_tmp='model.layers.{}',rewrite_module_tmp='model.layers.{}.mlp.down_proj',
            constr_pos='subject',fact_token='subject_last',v_loss_layer=27,v_lr=.01,
            v_num_grad_steps=3,kl_factor=.0625,v_weight_decay=.001,clamp_norm_factor=1)))
        rows=[dict(case_id=i,prompt=prompt,subject=subject,target_new=' 4 5 6')
            for i,prompt,subject in [(1,'{} 3 ','2'),(2,'1 {} 3 ','2'),(3,'1 2 {} ','3')]]
        kwargs=dict(compute_z=ns['compute_z'],find_fact_lookup_idx=lookup,TraceDict=nh.TraceDict)
        weights={n:v.clone() for n,v in model.named_parameters()};rng=rng_snapshot()
        with contextlib.redirect_stdout(io.StringIO()):
            expected=current_batch_targets(model,tok,rows,cfg,[['{}'],['1 {}']],**kwargs)
        rng_restore(rng)
        with contextlib.redirect_stdout(io.StringIO()):
            actual,receipt=p.probe_targets(model,tok,rows,cfg,[['{}'],['1 {}']],supplier=current_batch_targets,
                compute_z=ns['compute_z'],lookup=lookup,TraceDict=nh.TraceDict)
        for layer in expected:self.assertTrue(torch.equal(expected[layer],actual[layer]))
        for n,v in model.named_parameters():self.assertTrue(torch.equal(weights[n],v))
        self.assertEqual(receipt['extra_forwards'],6)
        self.assertTrue(receipt['rng_unchanged']);self.assertTrue(receipt['hooks_unchanged'])
        self.assertEqual(len(receipt['fits']),3)
        self.assertTrue(all(len(x['nll_each'])==2 for x in receipt['fits']))
        self.assertTrue(all(x['fit_effective_vs_return_cast_maxabs']==0 for x in receipt['fits']))
        self.assertTrue(all('L5_key' in r and len(r['L5_key']['top_indices'])<=8 for r in receipt['replays']))

    def test_missing_binding_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as d:
            from official.experiments.prepare import write_new
            root=Path(d);write_new(root/'binding.json',{})
            with self.assertRaises(KeyError):p.run(root/'binding.json',root/'new')
            self.assertFalse((root/'new').exists())


if __name__=='__main__':unittest.main()
