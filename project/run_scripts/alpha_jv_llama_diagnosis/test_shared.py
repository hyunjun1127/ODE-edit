"""Small CPU fixtures exercise the real overlay/JVP/lifecycle, never load an LLM."""
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch
from project.run_scripts.native_response_ode_v31.runtime import ObservedFamily
from project.run_scripts.native_response_ode_v31.native_binding import NativeDictionary
from project.run_scripts.native_response_ode_v31.algebra import FrozenNormalization
from project.run_scripts.ordered_response_barrier_ode.adapters import FixedZArtifact,LayerBuild
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_sha256,tensor_set_sha256
from project.run_scripts.ordered_response_barrier_ode.semantic import SemanticObservation
from .contracts import TrajectoryConfig,BindingBoundary,primary_residual
from .fixture import EntrySnapshot,RNGSnapshot
from .normalization_views import NormalizationView
from .trajectory import run_joint,solve_current
from .observations import same_state_shadows


class CPUFamily(ObservedFamily):
    def __init__(self):
        self.model=torch.nn.Module()
        self.model.layers=torch.nn.ModuleList([torch.nn.Linear(2,2,bias=False) for _ in range(9)])
        with torch.no_grad():
            for layer in self.model.layers:layer.weight.copy_(torch.eye(2)*.8)
        self.model.requires_grad_(False)
        self.parameters={f'layers.{i}.weight':self.model.layers[i].weight for i in range(4,9)}
        self.w0={n:p.detach().clone() for n,p in self.parameters.items()}
        self.w0_sha256=tensor_set_sha256(self.w0);self.pointer_identity={n:p.data_ptr() for n,p in self.parameters.items()}
        self.hparams=SimpleNamespace(rewrite_module_tmp='layers.{}',L2=1.)
        self.family='AlphaEdit';self.tokenizer=SimpleNamespace(padding_side='right');self.device=torch.device('cpu')
        self.module=SimpleNamespace(cache_c=torch.zeros(5,2,2),cache_c_new=True,
            compute_ks=lambda *args:torch.eye(2))
        self._alpha_cache_entry=self.module.cache_c.clone();self._alpha_cache_entry_is_zero=True
        self._prepared_method_state_identity=self.method_state_identity()
        self._capture_persistent_endpoint=True;self._captured_endpoint_weights=None
        self._captured_endpoint_method_state=None;self._captured_endpoint_sha256=None
        self.requests=({},{});self.contexts=(('{}',),);self.last_terminal=None
        self.x=torch.tensor([[.6,.3],[.2,.5]],dtype=torch.float32)
        self.fixed_z=None;target=self.terminal()+.2
        self.fixed_z=FixedZArtifact(target,tensor_sha256(target),'a'*64,'b'*64)
        self.evaluation_seconds=0.;self.last_physical_action=None;self.last_old_evaluation=None

    def terminal_graph(self):
        x=self.x
        for i in range(4,9):x=torch.tanh(self.model.layers[i](x))
        return x.T

    def observe_semantic(self):
        return SemanticObservation(False,0,2,(False,False),0,0.,0.,0.,0)

    def evaluate_endpoint(self):
        self.last_terminal=self.terminal()
        return dict(active_weight_sha256=tensor_set_sha256(self.parameters),request_count=2)


class CPUDictionary(NativeDictionary):
    def build(self,terminal,version,**kwargs):
        self.build_count+=5;self.solve_count+=5
        return [LayerBuild(i,f'layers.{i}.weight',(self.family.fixed_z.values-terminal).clone(),
            torch.eye(2),version,1,'c'*64,'d'*64,'e'*64,0.) for i in range(4,9)]


def fixture():
    f=CPUFamily();d=CPUDictionary(f);entry=f.terminal()
    norm=NormalizationView.from_source(FrozenNormalization.capture(f.fixed_z.values,entry,f.w0_sha256))
    d.capture_reference(d.build(entry,0))
    return f,d,norm


class SharedTests(unittest.TestCase):
    def test_immutable_config_and_source_residual(self):
        c=TrajectoryConfig(.1,4.,8)
        with self.assertRaises(FrozenInstanceError):c.T=2
        self.assertEqual(c.endpoint_clock(4)['effective_T'],2)
        self.assertEqual(c.receipt()['actual_clock'],4)
        a=torch.tensor([[2**24]],dtype=torch.float32);b=torch.tensor([[.75]],dtype=torch.float32)
        self.assertFalse(torch.equal(primary_residual(a,b).double(),a.double()-b.double()))
        with self.assertRaises(BindingBoundary):primary_residual(a.double(),b)

    def test_restore_precedes_family_snapshot(self):
        f,d,n=fixture();entry=EntrySnapshot.capture(f.parameters,f.module)
        with torch.no_grad():
            f.model.layers[4].weight.add_(1);f.module.cache_c.add_(2)
        def factory(model,module):
            self.assertEqual(tensor_set_sha256({n:model.get_parameter(n) for n in entry.weights}),entry.W_sha256)
            self.assertTrue(torch.equal(module.cache_c,entry.alpha_cache))
            return SimpleNamespace(w0_sha256=entry.W_sha256,bind_existing_method_state=lambda:None)
        entry.make_family(factory,model=f.model,module=f.module)
        entry.weights['layers.4.weight'].add_(1)
        with self.assertRaises(BindingBoundary):entry.restore(f.model,f.module)

    def test_lambda_binds_solve_and_all_shadows(self):
        f,d,n=fixture();terminal=f.terminal();target=f.fixed_z.values
        responses=[torch.ones_like(target),torch.tensor([[1.,0.],[0.,1.]])]
        q=torch.ones(2,dtype=torch.float64)
        for lam in (.01,.03162277660168379,.1,.31622776601683794,1.):
            c=TrajectoryConfig(lam,2,4)
            e,psi,g,sol,fact=solve_current(target,terminal,responses,q,n,c)
            out=same_state_shadows(e,psi,g,g,q,[4,8],sol.coefficients,c,g)
            self.assertEqual(out['lambda_response'],lam)
            self.assertEqual(out['history_cost']['lambda_response'],lam)
            self.assertLess(abs(fact['dissipation_residual']),1e-10)

    def test_real_overlay_trajectory_prefix_no_influence(self):
        config=TrajectoryConfig(.1,2.,4)
        with tempfile.TemporaryDirectory() as path,patch('project.run_scripts.ordered_response_barrier_ode.runtime._sync'):
            f,d,n=fixture();before=RNGSnapshot();qref=d.qref
            a=run_joint(f,d,n,config,output=Path(path)/'a',prefixes={2:'SHORT'})
            self.assertTrue(before.matches());self.assertEqual(d.qref,qref)
            self.assertEqual(tensor_set_sha256(f.parameters),f.w0_sha256)
            self.assertEqual(a['endpoints']['SHORT']['history_append_count'],0)
            self.assertEqual(a['endpoints']['SHORT']['evaluation']['active_weight_sha256'],a['endpoints']['SHORT']['selected_weight_endpoint_sha256'])
            self.assertEqual(a['endpoints']['JV_NATIVE']['history_append_count'],1)
            self.assertEqual(a['overlay']['physical_terminal_write_count'],1)
            b=run_joint(f,d,n,config,output=Path(path)/'b')
            self.assertEqual(a['endpoints']['JV_NATIVE']['selected_weight_endpoint_sha256'],b['endpoints']['JV_NATIVE']['selected_weight_endpoint_sha256'])
            short=run_joint(f,d,n,TrajectoryConfig(.1,1.,2),output=Path(path)/'short')
            self.assertEqual(a['endpoints']['SHORT']['selected_weight_endpoint_sha256'],short['endpoints']['JV_NATIVE']['selected_weight_endpoint_sha256'])
            self.assertNotEqual(a['nodes'][0]['q_layers'],a['nodes'][1]['q_layers'])

    def test_failure_is_persisted_after_restore(self):
        with tempfile.TemporaryDirectory() as path,patch('project.run_scripts.ordered_response_barrier_ode.runtime._sync'):
            f,d,n=fixture()
            def broken(*args):raise ValueError('forced_observation_failure')
            with self.assertRaisesRegex(ValueError,'forced_observation_failure'):
                run_joint(f,d,n,TrajectoryConfig(.1,2,4),output=Path(path),raw_sink=broken)
            import json
            record=json.loads((Path(path)/'failure-boundary.json').read_text())
            self.assertTrue(record['entry_restore']);self.assertEqual(record['completed_nodes'],0)
            self.assertEqual(record['stage'],'NODE_0')

    def test_prefix_rng_mutation_fails_and_restores(self):
        with tempfile.TemporaryDirectory() as path,patch('project.run_scripts.ordered_response_barrier_ode.runtime._sync'):
            f,d,n=fixture();before=RNGSnapshot();original=f.evaluate_endpoint
            def impure():
                torch.rand(1);return original()
            f.evaluate_endpoint=impure
            with self.assertRaisesRegex(BindingBoundary,'PREFIX_OBSERVER_INTERVENTION'):
                run_joint(f,d,n,TrajectoryConfig(.1,2,4),output=Path(path),prefixes={2:'SHORT'})
            self.assertTrue(before.matches());self.assertEqual(tensor_set_sha256(f.parameters),f.w0_sha256)

    def test_all_sinks_rng_mutation_rejected(self):
        for sink in ('raw','prefix','terminal'):
            with self.subTest(sink=sink),tempfile.TemporaryDirectory() as path,patch('project.run_scripts.ordered_response_barrier_ode.runtime._sync'):
                f,d,n=fixture();before=RNGSnapshot()
                def impure(*args):torch.rand(1)
                kwargs={'raw_sink':impure} if sink=='raw' else {'endpoint_sink':impure}
                if sink=='prefix':kwargs['prefixes']={2:'SHORT'}
                with self.assertRaisesRegex(BindingBoundary,'OBSERVATION_CALLBACK_INTERVENTION'):
                    run_joint(f,d,n,TrajectoryConfig(.1,2,4),output=Path(path),**kwargs)
                self.assertTrue(before.matches())
                self.assertEqual(tensor_set_sha256(f.parameters),f.w0_sha256)


if __name__=='__main__':unittest.main()
