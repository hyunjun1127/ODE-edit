"""Actual author writer AST with rectangular tiny BF16 state, no model download."""
import ast
import contextlib
import copy
import io
import os
from pathlib import Path
from types import SimpleNamespace as N
import time
import unittest
import torch
from official.diagnostics.fe_qwen_current.failure_zero_probe import masked_write,MASK


class Tests(unittest.TestCase):
    def test_native_zero_columns_and_history(self):
        path=Path(os.environ.get('FE_AUTHOR_ROOT','/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author'))/'algs/memit/FE-memit_main.py'
        if not path.is_file():self.skipTest('FE_AUTHOR_ROOT missing')
        tree=ast.parse(path.read_text());body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'batch_edit','upd_matrix_match_shape'}]
        self.assertEqual(len(body),2)
        K=torch.tensor([[1.,2.,.5,-1.],[.2,-.4,1.,2.],[1.,.5,2.,.1],[.3,.7,.1,2.]])
        cfg=N(llms=N(layers=[4,5],rewrite_module_tmp='layers.{}',layer_module_tmp='layers.{}',fact_token='subject_last',mom2_update_weight=[15000]*2),
            algs=N(L2=10,add_old_keys=True),negetive_prompt_test=False)
        def make(key):
            env=dict(torch=torch,time=time,covs=[torch.eye(4),torch.eye(4)],
                nethook=N(get_parameter=lambda m,n:dict(m.named_parameters())[n]),get_context_templates=lambda *a:[['{}']],
                compute_ks=key,get_module_input_output_at_words=lambda m,t,l,**kw:(None,K@dict(m.named_parameters())[f'layers.{l}.weight'].float().T))
            exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),env);return env['batch_edit']
        torch.manual_seed(9);a=torch.nn.Module();a.layers=torch.nn.ModuleDict({str(l):torch.nn.Linear(4,3,bias=False,dtype=torch.bfloat16) for l in (4,5)})
        b=copy.deepcopy(a);ha=torch.zeros(2,4,4);hb=ha.clone();rng=torch.get_rng_state().clone()
        original=lambda *args:K
        writer=make(original)
        with contextlib.redirect_stdout(io.StringIO()):
            for batch in range(1,6):
                ids=sorted(MASK.get(batch,set()));ids+=list(range(90000,90000+4-len(ids)))
                req=[dict(case_id=c,sample_idx=i,prompt='{}',subject='s') for i,c in enumerate(ids)]
                # Every request retains a target, including masked cases.
                zs={l:torch.full((3,4),float(batch+l)) for l in (4,5)}
                direct=K.clone();direct[[i for i,c in enumerate(ids) if c in MASK.get(batch,set())]]=0
                reference=make(lambda *args:direct)
                reference(cfg,a,None,req,'cpu',ha,zs)
                report=masked_write(writer,cfg,b,None,req,'cpu',hb,zs,batch,channels=(0,3),lookup=lambda r:0)
                self.assertEqual(report['case_ids'],ids);self.assertEqual(len(req),4)
                self.assertIs(writer.__globals__['compute_ks'],original)
                self.assertTrue(torch.equal(ha,hb));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(x,y))
                for row in report['layers']:
                    for i,c in enumerate(ids):
                        if c in MASK.get(batch,set()):self.assertEqual(row['key_norms'][i],0)

    def test_wrong_case_members_rejected(self):
        def writer(*args):raise AssertionError('must not execute')
        writer.__globals__['compute_ks']=lambda *a:None
        with self.assertRaisesRegex(ValueError,'EXACT_BATCH_MASK_MEMBERS'):
            masked_write(writer,None,None,None,[dict(case_id=21007)],None,None,None,1)


if __name__=='__main__':unittest.main()
