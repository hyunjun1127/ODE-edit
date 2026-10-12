"""CPU passive-observation parity; no pretrained or historical result claims."""
import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import unittest
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import Qwen2Config,Qwen2ForCausalLM
from official.diagnostics.fe_native.sequential_z import current_batch_targets
from official.diagnostics.fe_qwen_current import failure_case_probe as p
from official.diagnostics.fe_qwen_current.test_failure_probe import Tokenizer
from official.experiments.checkpoint import rng_snapshot,rng_restore


class CaseProbeTests(unittest.TestCase):
    def test_norm_comparison_preserves_exact_and_bounded_roundoff(self):
        value=torch.tensor(55.,dtype=torch.float32);nextvalue=torch.nextafter(value,torch.tensor(float('inf')))
        checked=p.norm_check(float(nextvalue),float(value))
        self.assertFalse(checked['exact']);self.assertTrue(checked['within_reduction_roundoff'])
        self.assertFalse(p.norm_check(55.01,55.)['within_reduction_roundoff'])

    def test_linear_decomposition_and_direction(self):
        w0=torch.eye(3,dtype=torch.bfloat16);w5=torch.diag(torch.tensor([2.,3.,-1.],dtype=torch.bfloat16))
        key=torch.tensor([1.,2.,3.],dtype=torch.bfloat16);native=w5@key
        r=p.linear_attribution(w0,w5,key,native)
        self.assertEqual(r['decomposition']['maxabs'],0)
        self.assertEqual(r['native_vs_explicit_BF16']['maxabs'],0)
        self.assertAlmostEqual(r['W0_projection_fraction']+r['delta_projection_fraction'],1.)
        self.assertAlmostEqual(sum(c['B5_column_projection_fraction'] for c in r['columns']),1.)
        self.assertAlmostEqual(r['B5_k_norm'],float((w5.double()@key.double()).norm()))

    def test_actual_native_tiny_bf16_passive_parity(self):
        root=Path(os.environ.get('FE_NATIVE_AUTHOR_ROOT','/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author'))
        if not root.exists():self.skipTest('native fixture absent')
        spec=importlib.util.spec_from_file_location('case_native_nethook',root/'util/nethook.py')
        nh=importlib.util.module_from_spec(spec);spec.loader.exec_module(nh)
        fn=next(n for n in ast.parse((root/'algs/memit/compute_z.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compute_z')
        fn.returns=None
        for arg in fn.args.args:arg.annotation=None
        tok=Tokenizer()
        def lookup(prompt,subject,tok,*a,**kw):return tok.encode(prompt.format(subject)).index(int(subject))
        ns=dict(torch=torch,np=np,nethook=nh,find_fact_lookup_idx=lookup)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'PINNED_NATIVE_CASE_FIT','exec'),ns)
        torch.set_num_threads(1);torch.manual_seed(17)
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=8,intermediate_size=16,
            num_hidden_layers=28,num_attention_heads=2,num_key_value_heads=2,use_cache=False)).to(torch.bfloat16).eval()
        w0=model.model.layers[5].mlp.down_proj.weight.detach().clone()
        with torch.no_grad():model.model.layers[5].mlp.down_proj.weight.add_(.01)
        cfg=OmegaConf.create(dict(gpu=0,lti=False,negetive_prompt_test=False,bs=100,llms=dict(
            layers=[4,5,6,7,8],lm_head_module='lm_head',ln_f_module='model.norm',midlayers=[15],
            layer_module_tmp='model.layers.{}',rewrite_module_tmp='model.layers.{}.mlp.down_proj',
            constr_pos='subject',fact_token='subject_last',v_loss_layer=27,v_lr=.01,
            v_num_grad_steps=3,kl_factor=.0625,v_weight_decay=.001,clamp_norm_factor=1)))
        rows=[dict(case_id=i,prompt=prompt,subject=subject,target_new=' 4 5 6')
            for i,prompt,subject in [(7231,'{} 3 ','2'),(761,'1 {} 3 ','2'),(20473,'1 2 {} ','3')]]
        contexts=[['{}'],['1 {}']];rng=rng_snapshot();weights={n:v.detach().clone() for n,v in model.named_parameters()}
        with contextlib.redirect_stdout(io.StringIO()):
            expected=current_batch_targets(model,tok,rows,cfg,contexts,compute_z=ns['compute_z'],find_fact_lookup_idx=lookup,TraceDict=nh.TraceDict)
        rng_restore(rng)
        with contextlib.redirect_stdout(io.StringIO()):
            actual,result=p.observe(model,tok,rows,cfg,contexts,supplier=current_batch_targets,
                compute_z=ns['compute_z'],lookup=lookup,TraceDict=nh.TraceDict,w0=w0)
        for layer in expected:self.assertTrue(torch.equal(expected[layer],actual[layer]))
        for n,v in model.named_parameters():self.assertTrue(torch.equal(weights[n],v))
        self.assertEqual(result['extra_forwards'],6);self.assertTrue(result['weights_rng_hooks_unchanged'])
        self.assertEqual([r['local_index'] for r in result['rows']],[2,31,93])
        for r in result['rows']:
            self.assertEqual(len(r['fit_rows']),3);self.assertEqual(r['fit_rows'][-1]['kind'],'KL')
            self.assertEqual(len(r['extra']),2);self.assertEqual(set(r['bare_fit_vs_replay']),{'key','mlp','block'})
            self.assertLess(r['replay_linear']['decomposition']['maxabs'],1e-10)


if __name__=='__main__':unittest.main()
