"""Production-path bounded CPU regression; not actual-model qualification."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from .geometry import root_cost,stable_squared,allocation
from .optimize import coordinates,clamp,stop,fit
from .subject import native
from .common import selection,MILESTONES,ROOT,write
from .collect import paired,reduce_rows

class CPU(unittest.TestCase):
    def test_spd_proxy_all_B(self):
        for B in (1,3,7):
            torch.manual_seed(B);d=11
            A=torch.randn(d,d,dtype=torch.float64);A=A@A.T+2*torch.eye(d,dtype=torch.float64)
            K=torch.randn(d,B,dtype=torch.float64);D=torch.randn(4,B,dtype=torch.float64,requires_grad=True)
            P=torch.linalg.solve(A+K@K.T,K);G=P.T@A@P;E=(P.T@K-torch.eye(B)).square() # matrix below is full, not elementwise E
            R=P.T@K-torch.eye(B);Q=G+R@R.T
            T=torch.linalg.cholesky(torch.eye(B)+K.T@torch.linalg.solve(A,K))
            direct=((D.T@D)*Q.T).sum();stable=stable_squared(D,T)
            self.assertTrue(torch.allclose(direct,stable,atol=1e-10,rtol=1e-10))
            self.assertTrue(torch.autograd.gradcheck(lambda x:root_cost(x,T),(D,),eps=1e-6,atol=1e-7,rtol=1e-5))
            z=torch.zeros_like(D,requires_grad=True);root_cost(z,T).backward();self.assertTrue(torch.equal(z.grad,torch.zeros_like(z)))
    def test_offdiagonal_not_discarded(self):
        T=torch.tensor([[2.,0.],[1.,1.]],dtype=torch.float64);D=torch.tensor([[1.,2.]],dtype=torch.float64)
        self.assertNotEqual(float(stable_squared(D,T)),float(stable_squared(D,torch.diag(T.diag()))))
    def test_q_sum_bridge_once(self):
        for B in (1,3):
            anchors={l:torch.ones(B)*(l+2) for l in range(3)};dims={l:(4,9) for l in anchors}
            s=coordinates(anchors,dims);q={l:torch.randn(4,B,requires_grad=True) for l in anchors}
            loss=sum(((s[l]*q[l])**2).sum() for l in q)/B
            expected=torch.autograd.grad(B*loss,tuple(q.values()))
            D={l:(s[l]*q[l]).detach().requires_grad_(True) for l in q}
            mean=sum((d*d).sum() for d in D.values())/B;g=torch.autograd.grad(mean,tuple(D.values()))
            for l,want,got in zip(q,expected,g):self.assertTrue(torch.allclose(B*s[l]*got,want,atol=1e-6))
    def test_clamp_preserves_moments(self):
        q={0:torch.full((4,3),100.,requires_grad=True)};s={0:torch.ones(3)};a={0:torch.ones(3)}
        opt=torch.optim.Adam(list(q.values()),lr=.1,foreach=False)
        q[0].grad=torch.ones_like(q[0]);opt.step();old={k:v.clone() for k,v in opt.state[q[0]].items()}
        clamp(q,s,a)
        self.assertTrue(all(torch.equal(v,opt.state[q[0]][k]) for k,v in old.items()))
        self.assertTrue(bool((q[0].norm(dim=0)<=.750005).all()))
    def test_earlystop_and_budget(self):
        self.assertTrue(stop(.049,1));self.assertFalse(stop(.05,1));self.assertTrue(stop(1.,25))
        for means,want in [([.03],(1,0)),([.2,.03],(2,1)),([1.]*25,(25,24))]:
            calls=[0]
            def toy(adapter,entry,D,backward=False,components=False):
                n=calls[0]
                if not backward:calls[0]+=1
                for d in D.values():
                    if backward:d.grad=torch.ones_like(d)
                return dict(mean=means[min(n,len(means)-1)] if not backward else 1.,components={},
                    gradients={name:{l:torch.zeros_like(d) for l,d in D.items()} for name in ('nll','kl','norm')} if components else None,
                    forward_groups=1,backward_calls=int(backward),seconds=0.)
            a=SimpleNamespace(dims={0:(2,3)},sites=(0,),device='cpu',native_route='cached')
            e=dict(pack=dict(n_requests=1),anchors={0:torch.ones(1)})
            f={0:dict(T=torch.eye(1,dtype=torch.float64),sigma=torch.tensor(1.))}
            with tempfile.TemporaryDirectory() as folder,patch('project.run_scripts.jlz_native_increment.optimize.native',toy):
                _,receipt=fit(a,e,f,'NOALLOC',Path(folder))
                self.assertEqual((receipt['candidates'],receipt['updates']),want)
                self.assertEqual(receipt['diagnostic_candidates'],sorted(set([n for n in (2,9) if n<=want[0]]+[want[0]])))
    def test_schedule_pre_post_allseen(self):
        records=list(range(2000));pairs=2000*13
        for _ in range(3):
            for n in range(1,21):
                current,seen,selected=selection(records,n)
                self.assertEqual(len(current),100);self.assertEqual(len(seen),n*100)
                self.assertEqual(len(selected),n*100 if n in (5,10,15,20) else 100)
                pairs+=(100+len(selected))*13
        self.assertEqual(pairs,361400)
        with self.assertRaises(RuntimeError):selection(records,21)
    def test_science_exclusions_source(self):
        folder=Path(__file__).parent
        for name in ('optimize.py','geometry.py','writer.py','subject.py'):
            tree=ast.parse((folder/name).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    self.assertNotIn(node.func.attr,('save','savez','pinv','inverse'))
        self.assertNotIn('causal_builder',(folder/'run.py').read_text())
        self.assertNotIn('physical_aux',(folder/'run.py').read_text())
    def test_request_mean_native_sum_grad(self):
        class Toy:
            device='cpu'
            def native(self,group,D,capture=False):
                rows=group['rows'];h=torch.stack([D[0][:,r['request']]+.2*D[1][:,r['request']] for r in rows])[:,None,:]
                return h,h,{}
            def head(self,x):return x
        for B in (1,3):
            a=Toy();rows=[]
            for r in range(B):
                for c in range(3):rows.append(dict(request=r,kind='rewrite' if c<2 else 'kl',lookup=0,target=torch.tensor([1 if c<2 else -100])))
            e=dict(pack=dict(n_requests=B,n_rw=2),anchors={0:torch.ones(B)*2,1:torch.ones(B)*3},
                teachers={r:torch.log_softmax(torch.zeros(4),0) for r in range(B)},groups=[])
            for start in range(0,len(rows),2):
                part=rows[start:start+2];e['groups'].append(dict(rows=part,tokens={'input_ids':torch.ones(len(part),1,dtype=torch.long),'attention_mask':torch.ones(len(part),1,dtype=torch.long)}))
            torch.manual_seed(B);D={l:torch.randn(4,B,requires_grad=True) for l in (0,1)}
            result=native(a,e,D,True,True);g={l:d.grad.clone() for l,d in D.items()}
            x=D[0]+.2*D[1];lp=x.T.log_softmax(-1)
            mean=-lp[:,1].mean()+.0625*(lp.exp()*(lp+torch.log(torch.tensor(4.)))).sum()/B
            mean+=.5*sum((d.norm(dim=0)/e['anchors'][l].square()).sum() for l,d in D.items())/B
            reference=torch.autograd.grad(B*mean,tuple(D.values()))
            self.assertAlmostEqual(result['mean'],float(mean.detach()),places=5)
            for l,v in zip(D,reference):self.assertTrue(torch.allclose(g[l],v,atol=1e-6,rtol=1e-6))
    def test_terminal_writer_increment_upper_refresh_H_once(self):
        from .writer import apply
        weights={l:torch.eye(2) for l in (0,1)}
        class Toy:
            sites=(0,1);device='cpu'
            def native(self,g,D,capture):return None,None,{l:D[l].T for l in self.sites}
        a=Toy();a.weights=weights
        K=torch.tensor([[1.],[2.]])
        entry=dict(pack=dict(n_requests=1),groups=[dict(rows=[dict(kind='rewrite',lookup=0)])],
            entry_weights={l:w.clone() for l,w in weights.items()},mean_keys={l:K.clone() for l in weights},
            entry_hidden={l:w@K for l,w in weights.items()},factors={})
        for l in weights:
            A=torch.eye(2,dtype=torch.float64)*2
            entry['factors'][l]=dict(A=A,L=torch.linalg.cholesky(A),LU=None,pivots=None,device='cpu',SPD=True)
        calls=[]
        def cap(adapter,e,layers,hidden_out=None):
            keys={l:K+(weights[0]-torch.eye(2)).sum() if l else K.clone() for l in layers}
            calls.append(tuple(layers))
            if hidden_out is not None:hidden_out.update({l:dict(keys=k,hidden=weights[l]@k) for l,k in keys.items()})
            return keys,dict(nll_mean=1,KL_mean=0)
        D={0:torch.tensor([[.1],[.2]]),1:torch.zeros(2,1)};H={l:torch.zeros(2,2) for l in weights}
        with tempfile.TemporaryDirectory() as folder,patch('project.run_scripts.jlz_native_increment.writer.capture',cap):
            result=apply(a,entry,D,H,Path(folder))
        self.assertEqual(result['history_appends'],2);self.assertEqual(calls,[(0,),(1,),(0,1)])
        self.assertTrue(torch.equal(weights[1],torch.eye(2))) # no inherited-gap compensation
        self.assertGreater(result['layers'][1]['key_drift_norm'],0)
        self.assertTrue(torch.equal(H[0],K@K.T))
        upper=K+(weights[0]-torch.eye(2)).sum();self.assertTrue(torch.equal(H[1],upper@upper.T))
    def test_RAM_transaction_exception_restore(self):
        from project.run_scripts.jlz_realization.writer import Transaction
        a=SimpleNamespace(weights={0:torch.ones(2,3)},guard=lambda:dict(nonselected=1))
        H={0:torch.ones(3,3)};transaction=Transaction(a,H)
        with self.assertRaisesRegex(ValueError,'planned'):
            with transaction:
                a.weights[0].zero_();H[0].zero_();raise ValueError('planned')
        self.assertTrue(transaction.rollback_verified);self.assertTrue(torch.equal(a.weights[0],torch.ones(2,3)))
        self.assertTrue(torch.equal(H[0],torch.ones(3,3)))
    def test_native_container_value_gradient(self):
        from project.run_scripts.jlz_two_arm.baseline_pilot import _native_container_callback
        d=torch.tensor(2.,requires_grad=True);x=torch.ones(2,3)
        def callback(out,layer):return (out[0]+d,)
        y=_native_container_callback(callback)(x,'layer');y.sum().backward()
        self.assertTrue(torch.equal(y,torch.full((2,3),3.)));self.assertEqual(float(d.grad),6.)
    def test_reducer_ties_nonfinite_and_pairs(self):
        def row(identity,n,t):return dict(identity=identity,kind='N',new_nll=n,true_nll=t,
            new_token_count=1,true_token_count=1,new_token_correct=1,true_token_correct=0,new_strict=True,true_strict=False)
        before=[row('a',1.,1.),row('b',2.,1.)];after=[row('a',2.,1.),row('b',0.,1.)]
        self.assertEqual(reduce_rows(before)['N']['numerator'],1)
        self.assertEqual(paired(before,after)['N'],dict(denominator=2,before=1,after=1,lost=1,gained=1))
        with self.assertRaises(RuntimeError):reduce_rows([row('bad',float('nan'),1.)])
        with self.assertRaises(RuntimeError):paired(before,after[:1])
    def test_partial_collector(self):
        from .collect import reduce_chain
        with tempfile.TemporaryDirectory() as folder:
            r=reduce_chain(Path(folder),dict(packing=[]),{},[])
            self.assertEqual(r['status'],'PARTIAL');self.assertEqual(r['commits'],0)

if __name__=='__main__':unittest.main()
