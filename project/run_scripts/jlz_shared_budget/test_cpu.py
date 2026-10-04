"""Production regression. Toy tests are explicitly not actual-Llama evidence."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
import numpy as np
from .optimizer import EfficiencyAdam,project,norm_gradient
from .optimize import fit,stop_reason
from .telemetry import Events,validate_relations
from .common import selection

class Tests(unittest.TestCase):
    def test_projection(self):
        for radius in (0.,.75,100.):
            x=[torch.tensor([3.,4.]),torch.tensor([0.,2.,0.]),torch.zeros(1)]
            y,r=project(x,radius)
            self.assertLessEqual(sum(float(v.double().norm()) for v in y),radius+1e-6)
            for old,new in zip(x,y):
                self.assertAlmostEqual(float(new.norm()),max(0.,float(old.norm())-r['tau']),places=6)
        self.assertTrue(torch.equal(norm_gradient(torch.zeros(4),2),torch.zeros(4)))

    def test_scaled_native_adam(self):
        for anchor in (.3,5.,100.):
            for scale in (1.,1e-9):
                d=torch.zeros(5,requires_grad=True);u=[torch.zeros(5)]
                native=torch.optim.Adam([d],lr=.1,eps=1e-8,foreach=False)
                relative=EfficiencyAdam(u,torch.tensor(anchor))
                for t in range(24):
                    g=torch.arange(1,6,dtype=torch.float32)*scale*(-1 if t%3==0 else 1)
                    d.grad=g.clone();native.step()
                    with torch.no_grad():
                        if d.norm()>.75*anchor:d.mul_(.75*anchor/d.norm())
                    u,_=relative.step(u,[anchor*g])
                    torch.testing.assert_close(u[0]*anchor,d,atol=2e-5,rtol=2e-4)
                self.assertEqual(relative.t,24)

    def test_reentry_and_independence(self):
        u=[torch.zeros(2),torch.zeros(3)];opt=EfficiencyAdam(u,torch.tensor(1.))
        for _ in range(12):u,_=opt.step(u,[torch.ones(2)*10,torch.zeros(3)])
        self.assertEqual(float(u[1].norm()),0.)
        old=opt.m[0].clone();u,r=opt.step(u,[torch.zeros(2),torch.ones(3)*100])
        self.assertGreater(float(u[1].norm()),0.)
        torch.testing.assert_close(opt.m[0],old*.9)
        z,r=EfficiencyAdam([torch.zeros(1),torch.zeros(7)],torch.tensor(3.)).step([torch.zeros(1),torch.zeros(7)],[torch.zeros(1),torch.zeros(7)])
        self.assertEqual(r['gamma'],[1.,1.])

    def test_stop_horizon(self):
        self.assertEqual(stop_reason(0.,0),'ZERO_STEP')
        self.assertEqual(stop_reason(.049,9),'OBJECTIVE_THRESHOLD')
        self.assertIsNone(stop_reason(.05,23))
        self.assertEqual(stop_reason(1.,24),'EVALUATION_BUDGET')
        rows=list(range(2000))
        for b in range(1,21):
            cur,obs=selection(rows,b);self.assertEqual(len(cur),100)
            self.assertEqual(len(obs),100*b if b in (5,10,15,20) else 100)
        with self.assertRaises(Exception):selection(rows,21)

    def test_request_terminal_schema(self):
        class Adapter:
            device='cpu';sites=(4,5);dims={4:(2,3),5:(3,4)};profile={'anchor_layer':5,'kl_factor':.0625}
        entry=dict(pack=dict(n_requests=3,record_ids=[11,12,13]),anchors={4:torch.ones(3),5:torch.ones(3)},
            groups=[dict(rows=[dict(request=r)]) for r in range(3)])
        seen={r:[] for r in range(3)}
        def fake(a,e,g,u):
            r=g['rows'][0]['request'];c=len(seen[r]);seen[r].append(c)
            base=0. if r==0 or (r==1 and c==2) else 1.
            loss=base+sum(v[:,r].square().sum()*.001 for v in u.values())
            return {r:(loss,loss*0)}, {r:{l:torch.full((a.dims[l][0],),float(c)) for l in a.sites}}
        with tempfile.TemporaryDirectory() as t,patch('project.run_scripts.jlz_shared_budget.optimize.forward',fake):
            out=Path(t);events=Events(out/'events.jsonl','CPU',1)
            plan,result=fit(Adapter(),entry,events,out)
            self.assertEqual([plan['terminal'][r]['candidate'] for r in range(3)],[0,2,24])
            self.assertEqual(result['request_evaluations'],29)
            self.assertEqual(result['request_updates'],26)
            self.assertEqual([len(seen[r]) for r in range(3)],[1,3,25])
            for l in Adapter.sites:self.assertEqual(plan['z'][l][0].tolist(),[0.,2.,24.])
            rows=[json.loads(x) for x in events.path.read_text().splitlines()]
            self.assertTrue(all(r['payload']['gradient']['availability']=='NO_BACKWARD_TERMINAL'
                for r in rows if r['event']=='candidate_layer' and r['candidate_index']==plan['terminal'][[11,12,13].index(int(r['request_id']))]['candidate']))

    def test_tiny_model_entry_fit_write_transaction(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        from .entry import Adapter,prepare_entry
        from .writer import apply,capture
        from .events import metadata,batch_entry
        from .common import state
        from project.run_scripts.jlz_realization.writer import Transaction
        from types import SimpleNamespace
        torch.set_num_threads(2);torch.manual_seed(71)
        model=LlamaForCausalLM(LlamaConfig(hidden_size=8,intermediate_size=12,num_hidden_layers=3,
            num_attention_heads=2,num_key_value_heads=2,vocab_size=31,max_position_embeddings=64,
            _attn_implementation='eager'))
        profile=dict(eligible_layers=[0,1],anchor_layer=1,nll_layer=2,kl_factor=.0625)
        a=Adapter(model,profile);bench=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
        pack=dict(n_requests=3,n_rw=2,record_ids=[11,12,13],canonical_rows=[0,3,6],
            context_group_slices=[(0,1),(1,2)],lookup=[1]*9,row_kind=['rewrite','rewrite','kl']*3,
            row_request=[i//3 for i in range(9)],tokens=dict(input_ids=torch.arange(45).reshape(9,5)%29+1,
            attention_mask=torch.ones(9,5,dtype=torch.long)),targets=torch.full((9,5),-100))
        for i in range(9):
            if pack['row_kind'][i]=='rewrite':pack['targets'][i,3:]=torch.tensor([2,3])
        H={l:torch.zeros(12,12) for l in a.sites};before=state(a,H)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);stats={}
            for l in a.sites:
                path=root/f'stats{l}.npz';np.savez(path,**{'mom2.mom2':np.eye(12,dtype=np.float32),'mom2.count':np.array(1)})
                stats[str(l)]=str(path)
            with Transaction(a,H) as tr:
                entry=prepare_entry(a,bench,pack,H,stats,requests_per_group=2)
                ev=Events(root/'events.jsonl','TINY',1);batch_entry(ev,entry,before)
                # The real production 25-evaluation loop, including partial group.
                plan,result=fit(a,entry,ev,root/'fit')
                self.assertLessEqual(result['request_evaluations'],75)
                rec=apply(a,entry,plan,H,ev,root)
                self.assertEqual(rec['history_appends'],2)
                final=capture(a,entry,a.sites)
                for l in a.sites:
                    k=final['mean'][l];torch.testing.assert_close(H[l],k@k.T,atol=0,rtol=0)
                # No finish deliberately: verify whole transaction rollback.
            self.assertTrue(tr.rollback_verified);self.assertEqual(state(a,H),before)

    def test_partial_collector_and_margin(self):
        from .collect import reduce_chain,describe,paired
        with tempfile.TemporaryDirectory() as t:
            result=reduce_chain(Path(t),dict(packing=[]),{},[])
            self.assertEqual(result['status'],'PARTIAL');self.assertEqual(result['commits'],0)
        row=dict(identity='x',kind='N',true_nll=1.,new_nll=2.,true_token_count=2,true_token_correct=2,
            true_strict=True,new_token_count=3,new_token_correct=1,new_strict=False)
        result=describe([row])['N'];self.assertEqual(result['rate'],1.)
        self.assertEqual(result['margin_quantiles']['0.5'],-1.)
        changed=row|dict(new_nll=.5)
        self.assertEqual(paired([row],[changed])['N']['lost'],1)

if __name__=='__main__':unittest.main()
