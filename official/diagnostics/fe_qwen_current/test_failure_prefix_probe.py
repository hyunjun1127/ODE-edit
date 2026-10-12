"""Actual pinned author writer AST; tiny CPU instrumentation parity, not GPU proof."""
import ast
import contextlib
import copy
import io
import os
from pathlib import Path
from types import SimpleNamespace as N
import time
import unittest
from unittest.mock import patch
import torch
from official.diagnostics.fe_qwen_current.failure_prefix_probe import observe_write,dual_columns,check_callback_binding


class Tests(unittest.TestCase):
    def test_actual_native_five_batches(self):
        path=Path(os.environ.get('FE_AUTHOR_ROOT','/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author'))/'algs/memit/FE-memit_main.py'
        if not path.is_file():self.skipTest('Pinned author source unavailable; set FE_AUTHOR_ROOT')
        tree=ast.parse(path.read_text());names={'batch_edit','upd_matrix_match_shape'}
        from official.diagnostics.fe_qwen_current.patch import native_source
        self.assertTrue(check_callback_binding(native_source(path.read_text())))
        selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        self.assertEqual(len(selected),2)
        K=torch.tensor([[1.,2.,.5,-1.],[.2,-.4,1.,2.]])
        env=dict(torch=torch,time=time,covs=[torch.eye(4),torch.eye(4)],
            nethook=N(get_parameter=lambda m,n:dict(m.named_parameters())[n]),
            get_context_templates=lambda *a:[['{}']],compute_ks=lambda *a:K,
            get_module_input_output_at_words=lambda m,t,l,**kw:(None,K@dict(m.named_parameters())[f'layers.{l}.weight'].float().T))
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),env)
        writer=env['batch_edit'];torch.manual_seed(42)
        a=torch.nn.Module();a.layers=torch.nn.ModuleDict({str(l):torch.nn.Linear(4,3,bias=False,dtype=torch.bfloat16) for l in (4,5)})
        b=copy.deepcopy(a);ha=torch.zeros(2,4,4);hb=ha.clone()
        cfg=N(llms=N(layers=[4,5],rewrite_module_tmp='layers.{}',layer_module_tmp='layers.{}',fact_token='subject_last',mom2_update_weight=[15000,15000]),
            algs=N(L2=10,add_old_keys=True),negetive_prompt_test=False)
        req=[dict(sample_idx=i,prompt='{}',subject='x') for i in range(2)]
        rng=torch.get_rng_state().clone();calls=[];solve=torch.linalg.solve
        def counted(A,B):calls.append(B.shape[1]);return solve(A,B)
        with contextlib.redirect_stdout(io.StringIO()),patch.object(torch.linalg,'solve',counted):
            for batch in range(1,6):
                targets={l:torch.full((3,2),float(batch+l)) for l in (4,5)}
                writer(cfg,a,None,req,'cpu',ha,targets)
                row=observe_write(writer,cfg,b,None,req,'cpu',hb,targets,batch,channels=(0,3))
                for p,q in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(p,q))
                self.assertTrue(torch.equal(ha,hb));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                l5=row['layers'][1]
                for c in l5['update_contributions']:
                    self.assertEqual(c['interpretation'],'NUMERICALLY_VERIFIED_UPDATE_CONTRIBUTION')
                    self.assertAlmostEqual(sum(c['per_request_signed_projection']),1.,places=10)
                for c in l5['columns']:self.assertLess(c['rhs_relative_reconstruction_error'],1e-10)
        self.assertEqual(len(calls),25) # bare10 + instrumented10 + dual5
        self.assertEqual(calls.count(2),5) # only diagnostics have two RHS
        self.assertIs(env['torch'],torch)

    def test_nonsymmetric_transpose(self):
        A=torch.tensor([[4.,1.,0.],[0.,3.,.5],[.2,0.,2.]],dtype=torch.float64)
        K=torch.tensor([[1.,2.],[3.,-.5],[.2,1.]],dtype=torch.float64);R=torch.tensor([[1.,2.],[2.,-1.]],dtype=torch.float64)
        U=torch.linalg.solve(A,K@R.T)
        rows=dual_columns(A,K,R,U,(0,2),torch.linalg.solve)
        for r in rows:self.assertLess(r['reconstruction_relative_error'],1e-12)


if __name__=='__main__':unittest.main()
