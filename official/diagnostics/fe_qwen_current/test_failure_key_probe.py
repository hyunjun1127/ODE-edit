"""Exact pinned author key aggregation/writer AST with tiny BF16 CPU inputs."""
import ast
import contextlib
import copy
import io
import os
from pathlib import Path
from types import SimpleNamespace as N
import time
import unittest
import numpy as np
import torch
from official.diagnostics.fe_qwen_current.failure_key_probe import capture_keys,passive_write,transfer_rows


def load_function(path,names,env):
    tree=ast.parse(path.read_text());body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    for n in body:
        n.returns=None
        for a in n.args.args:a.annotation=None
    if len(body)!=len(names):raise AssertionError('author function missing')
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),env);return env


class Tests(unittest.TestCase):
    def setUp(self):
        self.root=Path(os.environ.get('FE_AUTHOR_ROOT','/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author'))/'algs/memit'
        if not self.root.is_dir():self.skipTest('FE_AUTHOR_ROOT missing')

    def test_actual_keys_and_writer_on_off(self):
        torch.manual_seed(31);raw=torch.randn(12,4,dtype=torch.bfloat16)
        contexts=[['{}'],['A {}','B {}','C {}','D {}','E {}']]
        cfg=N(llms=N(layers=[4,5],rewrite_module_tmp='layers.{}',layer_module_tmp='layers.{}',fact_token='subject_last',mom2_update_weight=[15000]*2),
            algs=N(L2=10,add_old_keys=True),negetive_prompt_test=False)
        req=[dict(case_id=i,sample_idx=i,prompt='{} is',subject=str(i)) for i in range(2)]
        calls=[]
        def extract(*a,**kw):calls.append(kw);return raw,torch.zeros(12,3)
        kg=load_function(self.root/'compute_ks.py',{'compute_ks'},dict(torch=torch,np=np,get_module_input_output_at_words=extract));key=kg['compute_ks']
        a=torch.nn.Module();a.layers=torch.nn.ModuleDict({str(l):torch.nn.Linear(4,3,bias=False,dtype=torch.bfloat16) for l in (4,5)})
        b=copy.deepcopy(a);ha=torch.zeros(2,4,4);hb=ha.clone()
        K=key(a,None,req,cfg,4,contexts)
        returned,obs=capture_keys(key,a,None,req,cfg,4,contexts,lambda p,s:len(p))
        self.assertTrue(torch.equal(returned,K));self.assertTrue(torch.equal(obs['bare'],raw[[0,6]]))
        self.assertEqual(obs['positions'],[[len(ctx.format(r['prompt'])) for g in contexts for ctx in g] for r in req])
        self.assertIs(kg['get_module_input_output_at_words'],extract)
        env=load_function(self.root/'FE-memit_main.py',{'batch_edit','upd_matrix_match_shape'},dict(torch=torch,time=time,covs=[torch.eye(4)]*2,
            nethook=N(get_parameter=lambda m,n:dict(m.named_parameters())[n]),get_context_templates=lambda *a:contexts,compute_ks=key,
            get_module_input_output_at_words=lambda m,t,l,**kw:(None,K.float()@dict(m.named_parameters())[f'layers.{l}.weight'].float().T)))
        writer=env['batch_edit'];rng=torch.get_rng_state().clone();calls.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            for batch in range(1,4):
                z={l:torch.full((3,2),float(batch+l)) for l in (4,5)}
                writer(cfg,a,None,req,'cpu',ha,z)
                result=passive_write(writer,cfg,b,None,req,'cpu',hb,z,batch,lambda p,s:len(p),channels=(0,3))
                self.assertTrue(torch.equal(ha,hb));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(x,y))
                self.assertTrue(all(r['aggregation_exact'] for r in result['layers']))
                self.assertIs(env['compute_ks'],key)
        self.assertEqual(len(calls),12) # 3 batches * 2 layers * (bare + observed), no extra extraction

    def test_rectangular_linear_predictions(self):
        D=torch.tensor([[1.,2.,3.,4.],[-1.,2.,0.,1.]],dtype=torch.float64)
        bar=torch.tensor([[1.],[0.],[2.],[1.]],dtype=torch.float64);bare=torch.tensor([[0.,1.,1.,2.]],dtype=torch.bfloat16)
        R=D@bar
        r=transfer_rows(D,bar,bare,R,[dict(case_id=7)],[[0,1,2,3,4,5]],(0,3))[0]
        self.assertEqual(r['D_Kbar_vs_R']['error_norm'],0)
        self.assertAlmostEqual(r['D_Kbare_vs_R']['error_norm'],float((D@bare.double().T-R).norm()))
        self.assertEqual(r['selected_channels']['3']['bare'],2.)


if __name__=='__main__':unittest.main()
