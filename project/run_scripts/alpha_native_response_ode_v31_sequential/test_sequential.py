"""Only the new support/state/telemetry/stream boundaries; no legacy matrix."""
import ast,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from project.run_scripts.native_response_ode_v31.algebra import nnls_response,FrozenNormalization
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_set_sha256,tensor_sha256
from .telemetry import restricted_l8,initial_history_gram,actual_delta_rows,same_state_shadows
from .state import commit_checked,check_entry,checkpoint
from .contracts import LAYERS,H,N,T,CHAINS,SCIENCE
from .evaluation import public_full,public_rewrite,join_panels
from .accounting import Accounting


class FakeNative:
    def __init__(self):self.cache_c=torch.zeros(5,1)
    def compute_ks(self):return torch.ones(1)
    def execute_AlphaEdit(self):
        cache_c=self.cache_c
        for i in range(5):
            keys=self.compute_ks()
            cache_c[i]+=keys


class Tests(unittest.TestCase):
    def test_profiling_preserves_values_and_counts_history(self):
        module=FakeNative();model=torch.nn.Linear(1,1);ledger=Accounting(model)
        try:
            ledger.bind_native(module);module.execute_AlphaEdit()
            self.assertTrue(torch.equal(module.cache_c,torch.ones(5,1)))
            self.assertEqual(ledger.counts['history_key_captures'],5)
            self.assertEqual(ledger.counts['native_keys'],5)
            self.assertGreaterEqual(ledger.finish_history()['seconds'],0)
            torch.testing.assert_close(torch.linalg.solve(torch.eye(2),torch.ones(2)),torch.ones(2))
            self.assertEqual(ledger.counts['linalg_solve'],1)
        finally:ledger.close()
    def test_l8_closed_form_support_and_mapping(self):
        gen=torch.Generator().manual_seed(31)
        p=torch.randn(9,5,generator=gen,dtype=torch.float64);e=torch.randn(9,generator=gen,dtype=torch.float64)
        c=restricted_l8(e,p,list(LAYERS));ref=nnls_response(e,p[:,4:],torch.eye(1),.1)
        torch.testing.assert_close(c[4:],ref.coefficients,rtol=1e-13,atol=1e-13)
        self.assertEqual(c[:4].count_nonzero(),0);self.assertEqual(LAYERS,(4,5,6,7,8))
        self.assertEqual(N*H,T);self.assertEqual(len(CHAINS),6)

    def test_whitening_single_h(self):
        q=torch.tensor([2.,7.],dtype=torch.float64);c=torch.tensor([3.,4.],dtype=torch.float64)
        directions=torch.diag(q.sqrt());actual=H*(directions@(c/q.sqrt()))
        torch.testing.assert_close(actual,H*c)
        self.assertEqual(SCIENCE['normalization'],'SOURCE_EXACT_N0')

    def test_initial_history_gram_is_not_identity(self):
        d=SimpleNamespace(family=SimpleNamespace(hparams=SimpleNamespace(L2=2.)),qref=10.,raw=lambda b,**kw:b)
        g=initial_history_gram(d,[3.,5.],torch.tensor([2.,4.]))
        torch.testing.assert_close(g,torch.diag(torch.tensor([.3,.25],dtype=torch.float64)))
        self.assertFalse(torch.equal(g,torch.eye(2)))

    def test_shadows_are_observation_only(self):
        e=torch.tensor([1.,-.5],dtype=torch.float64);p=torch.tensor([[1.,2.],[-.4,.3]],dtype=torch.float64)
        g=torch.eye(2,dtype=torch.float64);q=torch.ones(2,dtype=torch.float64)
        c=nnls_response(e,p,g).coefficients;copy=c.clone()
        out=same_state_shadows(e,p,g,g,q,[4,8],c,2*g)
        self.assertTrue(torch.equal(c,copy));self.assertEqual(out['extra_jvp_count'],0)
        self.assertEqual(out['history_cost']['G_initial'],[[2.,0.],[0.,2.]])

    def test_normalized_error_space_and_physical_delta(self):
        z=torch.tensor([[2.,10.],[0.,0.]],dtype=torch.float32);entry=torch.zeros_like(z)
        norm=FrozenNormalization.capture(z,entry,'entry')
        self.assertNotEqual(float(norm.weight(z).norm()),float(z.norm()))
        rows=actual_delta_rows({'a':torch.ones(2,2)},{'a':torch.zeros(2,2)},{'a':torch.zeros(2,2)},{8:'a'})
        self.assertEqual(rows[0]['actual_step_DeltaW_squared'],4.)

    def test_transaction_commit_once_and_continuity(self):
        weights={'a':torch.zeros(2,2)};cache=torch.zeros(2,2);target=torch.ones(2,2)
        f=SimpleNamespace(parameters=weights,w0_sha256=tensor_set_sha256(weights),
            _prepared_method_state_identity='M0',method_state_identity=lambda:'M0',
            _captured_endpoint_method_state=target,module=SimpleNamespace(cache_c=cache))
        expected=tensor_set_sha256({'a':target})
        calls=[]
        def commit(**kw):
            calls.append(kw);weights['a'].copy_(target);cache.copy_(target)
            return dict(committed_weight_sha256=expected,writer_recompute_count=0,fixed_z_recompute_count=0,model_forward_count=0,evaluator_count=0)
        f.commit_captured_endpoint=commit
        receipt=commit_checked(f,dict(history_append_count=1,selected_weight_endpoint_sha256=expected))
        self.assertEqual(len(calls),1);f.w0_sha256=expected;check_entry(f,receipt)
        f.w0_sha256='wrong'
        with self.assertRaisesRegex(RuntimeError,'WEIGHT_CONTINUITY'):check_entry(f,receipt)
        with self.assertRaisesRegex(RuntimeError,'APPEND_ONCE'):commit_checked(f,dict(history_append_count=2))

    def test_checkpoint_actual_tensors_reload(self):
        w={'a':torch.tensor([[1.,2.]],dtype=torch.float32)};m=torch.eye(2,dtype=torch.float32)
        f=SimpleNamespace(parameters=w,module=SimpleNamespace(cache_c=m,cache_c_new=True))
        c=dict(committed_weight_sha256=tensor_set_sha256(w),committed_M_content_sha256=tensor_sha256(m))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'checkpoint.pt';r=checkpoint(path,f,c,{'batch':1})
            self.assertEqual(r['reload_tensor_identity'],'PASS')
            with self.assertRaises(FileExistsError):checkpoint(path,f,c,{})

    def test_evaluation_contract_and_denominator(self):
        row=dict(case_id=1,requested_rewrite=dict(prompt='{} is',subject='S',target_new={'str':' N'},target_true={'str':' T'}))
        raw={}
        for kind,n in [('rewrite_target_new',1),('rewrite_target_true',1),('rephrase_target_new',2),('rephrase_target_true',2),('locality_target_new',10),('locality_target_true',10)]:
            raw[kind]=[dict(case_id=1,kind=kind,prompt_index=i,prompt=f'{kind.split("_target")[0]}{i}',target=kind.split('_')[-1],
                nll=1. if kind.endswith('new') else 2.,target_token_ids=[1],token_predictions=[1],token_correct=[True],all_tokens_correct=True) for i in range(n)]
        public=public_full(raw,[row]);self.assertEqual(public['request_count'],1)
        self.assertIn('kind_summaries',public);self.assertIn('canonical_ns',public['locality'])
        r=public_rewrite(raw,'W1',1);self.assertEqual(len(r),1);self.assertTrue(r[0]['success'])
        self.assertEqual(len(join_panels([raw,raw])['rewrite_target_new']),2)

    def test_runtime_no_old_ray_dispatch_or_extra_sweep(self):
        root=Path(__file__).parent
        source=(root/'trajectory.py').read_text();tree=ast.parse(source)
        calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertNotIn('ray_solution',calls);self.assertNotIn('diagnostic_normalizations',calls)
        self.assertIn('restricted_l8',calls)
        launcher=(root/'run.sbatch').read_text()
        self.assertEqual(launcher.count('#SBATCH --mem='),1)
        self.assertIn('--mem=60416M',launcher)


if __name__=='__main__':unittest.main()
