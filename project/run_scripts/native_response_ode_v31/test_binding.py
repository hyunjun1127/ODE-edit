import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from project.run_scripts.ordered_response_barrier_ode.adapters import LayerBuild
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import GroupedFP32Overlay
from project.run_scripts.ordered_response_barrier_ode.terminal_jvp import TerminalResponseObserver
from .provenance import save


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first=torch.nn.Linear(3,4,bias=False)
        self.second=torch.nn.Linear(4,2,bias=False)

    def forward(self,x):
        return self.second(torch.tanh(self.first(x)))


class BindingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260906)
        self.model=Toy();self.x=torch.randn(2,3)
        self.names={4:'first.weight',5:'second.weight'}
        self.builds=[LayerBuild(layer,name,torch.randn(dict(self.model.named_parameters())[name].shape[0],2),
                    torch.randn(dict(self.model.named_parameters())[name].shape[1],2),0,1,'a'*64,'b'*64,'cpu')
                    for layer,name in self.names.items()]

    def test_joint_state_forward_and_restore(self):
        clone=copy.deepcopy(self.model)
        snapshots={k:(v.data_ptr(),v._version,v.detach().clone()) for k,v in self.model.named_parameters()}
        with GroupedFP32Overlay(self.model,self.names) as overlay,torch.no_grad():
            token=overlay.seal_sweep_entry(0)
            for b in self.builds:
                overlay.append(b.overlay_delta(.5),sweep_token=token)
                dict(clone.named_parameters())[b.weight_name].addmm_(b.left,b.right.T,alpha=.5)
            overlay.close_sweep(token)
            torch.testing.assert_close(self.model(self.x),clone(self.x),atol=2e-6,rtol=2e-6)
            with self.assertRaises(Exception):overlay.append(self.builds[0].overlay_delta(.5))
        for key,p in self.model.named_parameters():
            ptr,version,value=snapshots[key]
            self.assertEqual(p.data_ptr(),ptr);self.assertEqual(p._version,version)
            self.assertTrue(torch.equal(p,value))

    def test_raw_direction_jvp_fd_at_joint_state(self):
        with GroupedFP32Overlay(self.model,self.names) as overlay,torch.no_grad():
            observer=TerminalResponseObserver(model=self.model,overlay=overlay,capture_terminal_graph=lambda:self.model(self.x).T)
            for b in self.builds:
                response=observer.observe(b,expected_state_version=0)
                fun=observer._function(b);eps=2**-9
                fd=(fun(torch.tensor(eps))-fun(torch.tensor(-eps)))/(2*eps)
                torch.testing.assert_close(response.response,fd,atol=2e-4,rtol=2e-3)

    def test_create_once_and_symlink_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'record.json';save(path,{'test':1})
            with self.assertRaises(FileExistsError):save(path,{'test':2})
            (Path(folder)/'link').symlink_to(Path(folder),target_is_directory=True)
            with self.assertRaises(RuntimeError):save(Path(folder)/'link'/'escape.json',{})

    def test_derived_prefix_cannot_replace_primary_old_edit(self):
        from .runtime import ObservedFamily,old
        f=object.__new__(ObservedFamily)
        f.last_terminal='PRIMARY';f.last_old_evaluation={'primary':1}
        def observe(**kwargs):
            f.last_terminal='PREFIX';f.last_old_evaluation={'prefix':1}
            return {'status':'derived'}
        with patch.object(old.FamilyRuntime,'finalize',side_effect=observe):
            result=f.finalize(derived_observation_only=True)
        self.assertEqual(result['status'],'derived')
        self.assertEqual(f.last_terminal,'PRIMARY')
        self.assertEqual(f.last_old_evaluation,{'primary':1})


if __name__=='__main__':unittest.main()
