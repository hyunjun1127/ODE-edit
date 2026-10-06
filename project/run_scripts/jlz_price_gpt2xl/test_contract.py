"""CPU production-function regressions; no model/pilot/fit or online run."""
import unittest,types,torch
from unittest.mock import patch
from .adapter import Adapter,materialize
from .controller import RequestController
from .optimizer import EfficiencyAdamAbs
from .prepare import native_hparams
from .profile import arm_profile
from .submit import resource_order
from .common import CELLS
from .tracking import contract_ready,batch_values,TrackedEvents

class ContractTests(unittest.TestCase):
    def test_native_parser_and_profile(self):
        for w in ('memit','alphaedit'):
            hp,_=native_hparams(w)
            for arm in ('CAP075','CAP100','FREE100'):
                p=arm_profile(hp,arm,w)
                self.assertEqual((p['lr'],p['K_eval'],p['max_updates'],p['terminal_candidate']),(.5,20,19,19))
                self.assertEqual(p['lambda_C'],20000 if w=='memit' else 0)
                if w=='alphaedit':self.assertEqual(p['lambda_alpha'],10)
    def test_conv1d_payload_and_bias(self):
        # Pure operator fixture, not a tiny-model fit or target-model parity.
        W=torch.arange(12,dtype=torch.float32).reshape(4,3)/100
        R=torch.arange(6,dtype=torch.float32).reshape(3,2)/10
        Q=torch.arange(8,dtype=torch.float64).reshape(4,2)/10
        payload=materialize(W,R,Q)
        self.assertTrue(torch.equal(payload,W+(R.double()@Q.T).T.float()))
        bias=torch.tensor([1.,2.,3.]);key=torch.ones(2,5,4)
        a=object.__new__(Adapter);a.projection=lambda _:types.SimpleNamespace(bias=bias)
        self.assertTrue(torch.equal(a.local_linear(13,key,payload),
            torch.addmm(bias,key.reshape(-1,4),payload).reshape(2,5,3)))
    def test_controller_twenty_terminal(self):
        c=RequestController(torch.ones(5,2),torch.ones(2),range(13,18),torch.ones(5,2))
        for k in range(20):
            active,terminal=c.observe(torch.ones(2),k)
            self.assertEqual(terminal,k==19)
            if not terminal:c.before_update(torch.ones(2));c.record_update(active)
        self.assertEqual(c.t.tolist(),[19,19])
        with self.assertRaises(RuntimeError):c.record_update(active)
    def test_optimizer_nineteen_guard(self):
        opt=EfficiencyAdamAbs({13:torch.zeros(3,2)})
        self.assertEqual(opt.lr,.5)
        opt.t[:]=19
        with self.assertRaisesRegex(RuntimeError,'ADAM_UPDATE_BUDGET'):
            opt.step({13:torch.zeros(3,2)},{13:torch.zeros(3,2)},[True,True],None,torch.ones(1,2),torch.ones(2))
    def test_dag(self):
        d=resource_order(2)
        self.assertEqual(d['ALPHAEDIT_CAP075'],['MEMIT_CAP075'])
        self.assertEqual(d['MEMIT_CAP100'],['MEMIT_CAP075'])
        self.assertEqual(d['collector'],list(CELLS))
        for parallel in (1,2):
            for r,parents in resource_order(parallel).items():
                if r!='collector':self.assertTrue(all(CELLS.index(p)<CELLS.index(r) for p in parents))
    def test_tracking_capability_and_axis(self):
        contract_ready()
        class E:
            batch=2
            def emit(self,*args):pass
        class T:
            def log(self,v):self.value=v;return True
        t=T();e=TrackedEvents(E(),t)
        row=dict(candidate=0,F=[1]*100,J_mean=1,nll=[[1]*6]*100,KL=[0]*100,norm=[0]*100,
                 controller=dict(update_counts=[0]*100))
        e.emit('candidate',row)
        self.assertEqual(t.value['fit/global_candidate'],20)

    def test_serial_gpt2_residual(self):
        a=object.__new__(Adapter)
        attn=lambda hidden_states,**kw:(hidden_states*2,None)
        block=types.SimpleNamespace(ln_1=lambda x:x+1,attn=attn,ln_2=lambda x:x-3,
            mlp=types.SimpleNamespace(c_fc=lambda x:x*4,act=lambda x:x*x))
        a.blocks={13:block}
        x=torch.tensor([[[.25,-.5]]])
        key,residual=a.pre_projection(13,x,{})
        expected=(x+1)*2+x
        self.assertTrue(torch.equal(residual,expected))
        self.assertTrue(torch.equal(key,((expected-3)*4).square()))

    def test_alpha_ten_operator_and_LOO(self):
        from .alpha_geometry import ridge
        from .alpha_price import leave_one_out
        N=torch.diag(torch.tensor([1.,0.,1.]))
        H=torch.tensor([[2.,.3,.1],[.3,1.,.2],[.1,.2,3.]])
        K=torch.tensor([[.2,.4],[.1,-.1],[.3,.5]],dtype=torch.float64)
        A0=10*torch.eye(3,dtype=torch.float64)+N.double()@H.double()
        LU,piv=torch.linalg.lu_factor(A0)
        factor=dict(N=N,H=H,LU=LU,pivots=piv,metadata={},device='cpu')
        got=ridge(K,factor)
        expected=torch.linalg.solve(A0+N.double()@K@K.T,N.double()@K)
        torch.testing.assert_close(got['P'],expected,atol=1e-12,rtol=1e-12)
        a=types.SimpleNamespace(sites=(13,))
        b=dict(P={13:got['P']},K={13:K},mean_M={13:got['M']})
        result=leave_one_out(a,dict(pack={'n_requests':2},factors={13:factor}),b)
        self.assertEqual(len(result['pairs']),2)
        self.assertTrue(all(r['residual_relative']<=1e-8 for r in result['pairs']))

    def test_memit_explicit_coefficient(self):
        from .geometry import prior
        a=types.SimpleNamespace(profile={'writer':'memit','lambda_C':20000},device='cpu')
        with patch('project.run_scripts.jlz_price_gpt2xl.geometry.memit.prior',return_value='bound') as fn:
            self.assertEqual(prior(a,'readonly.npz','H',13),'bound')
            fn.assert_called_once_with('readonly.npz','H','cpu',20000)

    def test_method_only_execution_block(self):
        from .submit import submit
        from .run import main
        with patch('subprocess.run',side_effect=AssertionError('NO_COMMAND_ALLOWED')):
            with self.assertRaisesRegex(RuntimeError,'METHOD_ONLY_USER_DIRECTED'):
                submit(None,None)
            with self.assertRaisesRegex(RuntimeError,'METHOD_ONLY_USER_DIRECTED'):
                main()

if __name__=='__main__':unittest.main()
