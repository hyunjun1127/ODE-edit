"""Bounded production CPU tests; tiny tensors/Llama are not actual LM PASS."""
import ast
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from .geometry import mean_keys,ridge,exact
from .allocation import loss,Root
from .physical_linear import materialize,linear
from .optimize import fit,scales,clamp,partitions
from .profile import LlamaAdapter
from .causal_builder import build
from .subject import row_logprobs
from .qualification import full_native_reference
from .writer import Transaction,commit
from .common import state,write,sha
from .collect import collect
from .submit import arguments,dependencies,dependency_members,expected_dependencies

def prior(A):
    return dict(A=A,L=torch.linalg.cholesky(A),LU=None,pivots=None,SPD=True,asymmetry_max=0.)

def tiny(B=2):
    cfg=LlamaConfig(hidden_size=8,intermediate_size=12,num_hidden_layers=4,num_attention_heads=2,
        num_key_value_heads=2,vocab_size=32,max_position_embeddings=64,attention_dropout=0.)
    cfg._attn_implementation='eager'
    a=LlamaAdapter(LlamaForCausalLM(cfg),dict(eligible_layers=[0,1,2],nll_layer=3,kl_factor=.0625))
    rows=[]
    for r in range(B):
        for c in range(4):
            target=torch.full((4,),-100);target[3]=7+r
            rows.append(dict(kind='rewrite' if c<3 else 'kl',request=r,lookup=1,global_row=r*4+c,target=target,
                tokens=dict(input_ids=torch.tensor([1,2+r,3+c,4]),attention_mask=torch.ones(4,dtype=torch.long))))
    tokens={key:torch.stack([r['tokens'][key] for r in rows]) for key in ('input_ids','attention_mask')}
    group=dict(rows=rows,tokens=tokens,cache=a.prefix(tokens))
    pack=dict(n_requests=B,n_rw=3,context_group_slices=[(0,1),(1,3)],context_group_lens=[1,2],
              key_request=[r for r in range(B) for _ in range(3)])
    e=dict(groups=[group],pack=pack,factors={l:prior(torch.eye(12,dtype=torch.float64)) for l in a.sites},
        entry_weights={l:w.detach().clone() for l,w in a.weights.items()},first_geometry={},
        anchors={l:torch.ones(B) for l in a.sites})
    with torch.no_grad():
        zero={l:torch.zeros(8,B) for l in a.sites};nh,fh,_=a.native(group,zero)
        e['teachers']={r:a.head(fh[r*4+3,1]).log_softmax(-1).cpu() for r in range(B)}
        b=build(a,e,zero,0);e['mean_keys']={l:g['K'].float() for l,g in b['geometry'].items()}
    return a,e,{l:torch.zeros(12,12) for l in a.sites}

class Core(unittest.TestCase):
    def setUp(self):torch.manual_seed(20261003);torch.set_num_threads(2)
    def test_nested_mean_native_exact_and_permutation(self):
        for B in (1,3):
            raw=torch.randn(5,B*6);rows=[dict(global_row=r*7+c) for r in range(B) for c in range(6)]
            pack=dict(n_requests=B,n_rw=6,context_group_slices=[(0,1),(1,6)])
            want=torch.stack([torch.stack([raw[:,r*6:r*6+1].T.contiguous().mean(0),raw[:,r*6+1:(r+1)*6].T.contiguous().mean(0)]).mean(0) for r in range(B)]).T.contiguous()
            self.assertTrue(torch.equal(mean_keys(raw,rows,pack),want))
            self.assertTrue(torch.equal(mean_keys(raw.flip(1),list(reversed(rows)),pack),want))
            self.assertFalse(torch.equal(want,raw.reshape(5,B,6).mean(-1)))
    def test_native_mean_gradient(self):
        raw=torch.randn(5,6,requires_grad=True);pack=dict(n_requests=1,n_rw=6,context_group_slices=[(0,1),(1,6)])
        out=mean_keys(raw,[dict(global_row=c) for c in range(6)],pack);out.sum().backward()
        torch.testing.assert_close(raw.grad,torch.tensor([.5,.1,.1,.1,.1,.1]).expand(5,6),atol=0,rtol=0)
    def test_ridge_value_and_full_key_gradient(self):
        A=torch.randn(7,7,dtype=torch.float64);A=A@A.T+torch.eye(7,dtype=torch.float64)
        K=torch.randn(7,3,dtype=torch.float64,requires_grad=True);D=torch.randn(4,3,dtype=torch.float64,requires_grad=True)
        g=ridge(K,prior(A));p=torch.linalg.solve(A+K@K.T,K);M=p.T@K;E=(M-torch.eye(3,dtype=torch.float64));ref=dict(P=p,K=K,M=M,G=p.T@A@p,E=E@E.T)
        torch.testing.assert_close(g['P'],p,atol=1e-9,rtol=1e-8)
        for arm in ('A','B'):
            f=loss({0:D},{0:g},{0:torch.ones(3)},arm)[0];r=loss({0:D},{0:ref},{0:torch.ones(3)},arm)[0]
            fg=torch.autograd.grad(f,(D,K),retain_graph=True);rg=torch.autograd.grad(r,(D,K),retain_graph=True)
            for x,y in zip(fg,rg):torch.testing.assert_close(x,y,atol=1e-9,rtol=1e-8)
    def test_direct_D_P_input(self):
        x=torch.randn(2,3,7,requires_grad=True);D=torch.randn(5,2,requires_grad=True);P=torch.randn(7,2,dtype=torch.float64,requires_grad=True);w=torch.randn(5,7)
        W=materialize(w,D,P);f=linear(x,D,P,W.detach());ref=torch.nn.functional.linear(x,W)
        self.assertTrue(torch.equal(f,ref));up=torch.randn_like(f)
        a=torch.autograd.grad((f*up).sum(),(x,D,P),retain_graph=True);b=torch.autograd.grad((ref*up).sum(),(x,D,P))
        for v,r in zip(a,b):torch.testing.assert_close(v,r,atol=2e-6,rtol=2e-6)
    def test_merged_not_split_zero(self):
        D={0:torch.zeros(3,2,requires_grad=True),1:torch.zeros(3,2,requires_grad=True)}
        g={l:ridge(torch.randn(4,2,dtype=torch.float64),prior(torch.eye(4,dtype=torch.float64))) for l in D}
        for arm in ('A','B'):
            f=loss(D,g,{l:torch.ones(2) for l in D},arm)[0];gr=torch.autograd.grad(f,tuple(D.values()))
            self.assertEqual(float(f),0.);self.assertTrue(all(torch.equal(v,torch.zeros_like(v)) for v in gr))
    def test_q_bridge_first_step_and_clamp_moments(self):
        anchors={l:torch.tensor([2.,3.]) for l in range(5)};dims={l:(32,16) for l in anchors};s=scales(anchors,dims)
        q={l:torch.zeros(32,2,requires_grad=True) for l in anchors};opt=torch.optim.Adam(list(q.values()),lr=.1,foreach=False)
        for l in q:q[l].grad=s[l]*torch.ones_like(q[l])
        opt.step();rho=torch.stack([((s[l]*q[l]).double().norm(dim=0)/anchors[l]).square() for l in q]).sum(0)
        self.assertTrue(bool((rho<=.01+1.1e-7).all()))
        moments={l:opt.state[v]['exp_avg'].clone() for l,v in q.items()}
        with torch.no_grad():
            for v in q.values():v.mul_(10000)
        clamp(q,s,anchors)
        for l in q:
            self.assertTrue(torch.equal(moments[l],opt.state[q[l]]['exp_avg']))
            self.assertTrue(bool(((s[l]*q[l]).norm(dim=0)<=.75*anchors[l]+1e-6).all()))
    def test_exact_fullrank_and_unsupported(self):
        A=torch.eye(6,dtype=torch.float64);K=torch.randn(6,2,dtype=torch.float64);D=torch.randn(3,2)*.01
        op,r=exact(K,prior(A),D,torch.zeros(3,6));self.assertEqual(r['status'],'QUALIFIED')
        torch.testing.assert_close(D.double()@op['P'].T@K,D.double(),atol=1e-9,rtol=1e-8)
        _,r=exact(K[:,0:1].expand(6,2),prior(A),D,torch.zeros(3,6));self.assertEqual(r['status'],'rank_unsupported')
    def test_adapter_native_and_causal(self):
        a,e,h=tiny();D={l:(torch.randn(8,2)*.01).requires_grad_(True) for l in a.sites}
        left=a.native(e['groups'][0],D,True)
        with full_native_reference(a):right=a.native(e['groups'][0],D,True)
        torch.testing.assert_close(left[0],right[0],atol=1e-6,rtol=1e-5)
        b=build(a,e,D,1);n,_,_,k=a.actual(e['groups'][0],D,b['P'],b['weights'],capture=True)
        with a.install(D,b['P'],b['weights'],route='dense'):ref,_=a.full(e['groups'][0]['tokens'])
        torch.testing.assert_close(n,ref,atol=1e-6,rtol=1e-5)
        objective=loss(D,b['geometry'],e['anchors'],'B')[0]+n.square().mean()
        grads=torch.autograd.grad(objective,tuple(D.values()));self.assertTrue(all(bool(torch.isfinite(v).all()) for v in grads))
        self.assertTrue(b['P'][1].requires_grad)
    def test_complete_tiny_fit_commit_H_and_rollback(self):
        a,e,h=tiny();before=state(a,h)
        with tempfile.TemporaryDirectory() as path:
            with Transaction(a,h) as tx:
                payload,summary=fit(a,e,'A',Path(path),1,20261002,'fixture')
                self.assertEqual(summary['Adam_updates'],24)
                expected={l:payload['keys'][l]@payload['keys'][l].T for l in a.sites}
                commit(a,h,e,payload)
                for l in h:self.assertTrue(torch.equal(h[l],expected[l]))
                self.assertEqual(len(list(Path(path).glob('candidate-*.json'))),25)
                self.assertFalse(json.loads((Path(path)/'candidate-25.json').read_text())['gradient_measured'])
                last=json.loads((Path(path)/'terminal-actual.json').read_text())
                self.assertIn('actual_target_kl_from_virtual',last)
                for item in last['decomposition']:
                    for layer in item.values():self.assertIn('ideal_direction',layer);self.assertIn('canonical',layer)
                second=json.loads((Path(path)/'candidate-02.json').read_text())
                self.assertIn('proposal_radial',second);self.assertIn('projection_discarded_norm',second)
                for layer in second['layer'].values():self.assertIn('realized_rho',layer);self.assertIn('self_direction',layer)
                self.assertEqual(len(list((Path(path)/'diagnostics').glob('*.npz'))),6)
            self.assertTrue(tx.rollback_verified);self.assertEqual(state(a,h),before)
    def test_partition_budget_resources_parse(self):
        for n in (0,1,2,3,100):self.assertEqual(sorted(sum(partitions(n,1,'x',1),[])),list(range(n)))
        for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
        dep=dependencies(('afterok',['1','2']));self.assertEqual(dependency_members('afterok:1(unfulfilled):2(unfulfilled)'),expected_dependencies(dep))
        r=dict(host_mib=60416,collector_host_mib=24576,qualification_wall='24:00:00',wall='7-00:00:00',collector_wall='04:00:00')
        for name in ('prep-A','prep-B','main-A','main-B','collector'):
            args=arguments(name,dep,Path('/tmp/scoped'),r);self.assertIn('--hold',args);self.assertIn('--export=NONE',args)
            self.assertEqual('--gres=gpu:1' in args,name!='collector')
    def test_Q2_frozen_upper_failure_does_not_copy_to_causal(self):
        from .exact_probe import probe
        from .subject import native
        from .physical_aux import actual
        a,e,h=tiny();D={l:torch.randn(8,2)*.01 for l in a.sites};before=state(a,h)
        with torch.no_grad():
            n=native(a,e,D,False,range(2));b=build(a,e,D,25)
            payload=actual(a,e,b,D,n['teachers'],range(2),False,terminal=True)['payload']
        calls=[]
        def forced_upper(K,prior,D,W,**kwargs):
            calls.append(1)
            if len(calls)==2:return None,dict(status='rank_unsupported',reason='test_frozen_upper_only')
            return exact(K,prior,D,W,**kwargs)
        with tempfile.TemporaryDirectory() as path:
            with patch('project.run_scripts.jlz_realization.exact_probe.exact',side_effect=forced_upper),patch('project.run_scripts.jlz_realization.exact_probe.observe'):
                probe(a,None,h,e,D,b,payload,n['teachers'],[{},{}],Path(path),2)
            r=json.loads((Path(path)/'receipt.json').read_text())
            self.assertTrue(r['restore_verified']);self.assertTrue(r['exact_qualified']);self.assertEqual(r['history_appends'],0)
            self.assertEqual(state(a,h),before);self.assertNotIn('exact',e['first_geometry'])
    def test_q_bridge_matches_full_autograd_SUM(self):
        a={0:torch.tensor([2.,3.]),1:torch.tensor([4.,5.])};dims={0:(3,4),1:(3,4)};s=scales(a,dims)
        q={l:torch.randn(3,2,requires_grad=True) for l in a};D={l:s[l]*q[l] for l in q}
        def f(ds):return 2*sum((d.double()@torch.tensor([[1.,.3],[.3,2.]],dtype=torch.float64)).square().mean() for d in ds.values())
        full=torch.autograd.grad(f(D),tuple(q.values()));leaves={l:d.detach().requires_grad_(True) for l,d in D.items()}
        partial=torch.autograd.grad(f(leaves),tuple(leaves.values()))
        for l,x,y in zip(q,partial,full):self.assertTrue(torch.equal(s[l]*x,y))
    def test_partial_collect(self):
        with tempfile.TemporaryDirectory() as path:
            p=Path(path);write(p/'w0.json',dict(rows=[]));write(p/'bridge.json',dict(observations=dict(path=str(p/'w0.json'),sha256=sha(p/'w0.json'))))
            write(p/'config.json',dict(w0_reuse=dict(receipt=dict(path=str(p/'bridge.json')))));write(p/'execution.lock.json',dict(source_commit='fixture'))
            collect(p,p/'report');self.assertEqual(json.loads((p/'report/terminal.json').read_text())['status'],'PARTIAL_OR_TECHNICAL_FAILED')

if __name__=='__main__':unittest.main()
