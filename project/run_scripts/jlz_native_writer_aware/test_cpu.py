"""CPU toy Llama production checks; NOT actual pretrained GPU qualification."""
import ast,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from project.run_scripts.jlz_realized_subject.entry import prepare_entry
from project.run_scripts.jlz_realized_subject.qualification import fixed,compare
from project.run_scripts.jlz_realization.writer import Transaction
from project.run_scripts.jlz_shared_budget.optimizer import EfficiencyAdam,project,norm_gradient
from .physical import Adapter,materialize,linear
from .builder import build,reverse
from .subject import evaluate
from .qualification import check
from .optimize import stop,fit
from .telemetry import commit_measure
from .collect import validate_fit,paired
from .common import ROOT,state

def fixture(root,B=3,mb=2):
    torch.manual_seed(14)
    model=LlamaForCausalLM(LlamaConfig(hidden_size=8,intermediate_size=16,num_hidden_layers=4,
        num_attention_heads=2,num_key_value_heads=2,vocab_size=31,max_position_embeddings=64,_attn_implementation='eager'))
    a=Adapter(model,dict(eligible_layers=[0,2],anchor_layer=2,nll_layer=3,kl_factor=.0625,lambda_C=15000.))
    bench=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    rows=3*B
    pack=dict(identity='CPU_TINY',n_requests=B,n_rw=2,record_ids=list(range(B)),canonical_rows=list(range(0,rows,3)),
        context_group_slices=[(0,1),(1,2)],lookup=[1]*rows,row_kind=['rewrite','rewrite','kl']*B,
        row_request=[i//3 for i in range(rows)],tokens=dict(input_ids=torch.arange(rows*5).reshape(rows,5)%29+1,
        attention_mask=torch.ones(rows,5,dtype=torch.long)),targets=torch.full((rows,5),-100))
    for i in range(rows):
        if pack['row_kind'][i]=='rewrite':pack['targets'][i,3:]=torch.tensor([2,3])
    H={l:torch.zeros(16,16) for l in a.sites};stats={}
    for l in a.sites:
        path=root/f'stats{l}.npz';np.savez(path,**{'mom2.mom2':np.eye(16,dtype=np.float32),'mom2.count':np.array(1)})
        stats[str(l)]=str(path)
    entry=prepare_entry(a,bench,pack,H,stats,mb)
    return a,entry,H

class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_production_streamed_dense_fullnative_shapes(self):
        for B in (1,2,3):
            with tempfile.TemporaryDirectory() as t:
                root=Path(t);a,entry,H=fixture(root,B)
                r=check(a,entry,root/'check',zero=B==1)
                self.assertTrue(r['native_loss_pass'])
    def test_microbatch_gradient_and_sumscale(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a,e,H=fixture(root,3,2);b,f,_=fixture(root,3,1)
            results=[]
            for adapter,entry in ((a,e),(b,f)):
                R=fixed(adapter,entry);built=build(adapter,entry,R,0);value=evaluate(adapter,entry,built,True)
                grad,_=reverse(adapter,entry,R,built,value['adjoint']);results.append((value,grad))
            for x in compare(results[0][1],results[1][1]).values():self.assertTrue(x['passed'])
            torch.testing.assert_close(results[0][0]['F'],results[1][0]['F'],atol=1e-5,rtol=1e-4)
            # Reverse is linear in all row adjoints; no hidden 1/B or B multiplier.
            R=fixed(a,e);built=build(a,e,R,0);value=evaluate(a,e,built,True)
            g,_=reverse(a,e,R,built,{l:3*x for l,x in value['adjoint'].items()})
            for l in a.sites:torch.testing.assert_close(g[l],3*results[0][1][l],atol=1e-6,rtol=1e-3)
    def test_synchronous_stop(self):
        self.assertFalse(stop(torch.tensor([.01,.06]),0))
        self.assertTrue(stop(torch.tensor([.01,.02]),0))
        self.assertTrue(stop(torch.tensor([1.,2.]),24))
    def test_efficiency_projection_and_reentry(self):
        blocks=[torch.zeros(5),torch.zeros(3)];opt=EfficiencyAdam(blocks,torch.tensor(2.))
        new,r=opt.step(blocks,[torch.ones(5),torch.zeros(3)])
        self.assertEqual(opt.t,1);self.assertEqual(float(new[1].norm()),0.)
        again,r=opt.step(new,[torch.zeros(5),torch.ones(3)])
        self.assertGreater(float(again[1].norm()),0.);self.assertLessEqual(r['stored_budget'],.750001)
        self.assertEqual(float(norm_gradient(torch.zeros(3),1.).norm()),0.)
    def test_complete_fit_commit_H_restore_and_collector(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a,e,H=fixture(root,2);before=state(a,H)
            tx=Transaction(a,H)
            with tx:
                plan=fit(a,e,root/'fit')
                receipt=commit_measure(a,e,H,plan,root/'writer')
                self.assertEqual(receipt['history_appends'],2);self.assertEqual(receipt['postfit_commits'],1)
                checked=validate_fit(root/'fit');self.assertEqual(checked['updates'],24)
                for l in a.sites:self.assertTrue(torch.equal(a.weights[l],plan['built']['weights'][l]))
            self.assertTrue(tx.rollback_verified);self.assertEqual(state(a,H),before)
    def test_coupled_key_gradient_negativecontrol(self):
        with tempfile.TemporaryDirectory() as t:
            a,e,H=fixture(Path(t),3)
            # CPU geometry strengthened only within this test fixture.
            for prior in e['factors'].values():
                prior['A']=torch.eye(16,dtype=torch.float64)*.0001;prior['L']=torch.linalg.cholesky(prior['A'])
            R=fixed(a,e,.05);built=build(a,e,R,0);v=evaluate(a,e,built,True)
            g,_=reverse(a,e,R,built,v['adjoint']);stopped,_=reverse(a,e,R,built,v['adjoint'],stop_solve=True)
            self.assertGreater(float((g[a.first]-stopped[a.first]).norm()),1e-9)
    def test_scope_noCP(self):
        folder=ROOT/'project/run_scripts/jlz_native_writer_aware'
        for p in folder.glob('*.py'):
            if p.name.startswith('test_'):continue
            calls=[ast.unparse(n.func) for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.Call)]
            self.assertNotIn('torch.save',calls);self.assertNotIn('np.save',calls)
        calls=[n.func.id for n in ast.walk(ast.parse((folder/'run.py').read_text())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertEqual(calls.count('fit'),1);self.assertEqual(calls.count('commit_measure'),1)
    def test_metric_pair_strict_direction(self):
        before=[];after=[]
        for kind in ('R','P','N'):
            r=dict(identity=kind,kind=kind,new_nll=2.,true_nll=1.,new_strict=False,true_strict=True,new_token_identity='a',true_token_identity='b')
            before.append(r);after.append(r|dict(new_nll=.5,new_strict=True,true_strict=False))
        p=paired(before,after);self.assertEqual(p['R']['preference']['gained'],1);self.assertEqual(p['N']['preference']['lost'],1)

if __name__=='__main__':unittest.main()
