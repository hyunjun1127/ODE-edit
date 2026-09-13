import copy
import types
import unittest
import torch
from project.run_scripts.multilayer_joint_compensation.observations import JointView,pack,current_callback,teacher,bind_teacher,StateBoundary

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(11,4)
        self.first=torch.nn.Linear(4,4,bias=False);self.second=torch.nn.Linear(4,11,bias=False)
    def forward(self,input_ids,attention_mask,use_cache=False):
        x=self.embedding(input_ids).cumsum(1)
        return types.SimpleNamespace(logits=self.second(torch.tanh(self.first(x))))

def rows():
    return [dict(input_ids=[1,2,3],positions=[1,2],target_ids=[3,4],token_mean_weight=.5,
                 context_weight=.2,identity='a'),
            dict(input_ids=[1,5],positions=[1],target_ids=[7],token_mean_weight=1.,context_weight=.3,identity='b'),
            dict(input_ids=[2,4,6,1],positions=[2,3],target_ids=[1,2],token_mean_weight=.5,context_weight=.5,identity='c')]

class Observations(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13);self.model=Toy().eval().requires_grad_(False)
        self.view=JointView(self.model,('first.weight','second.weight'))

    def test_joint_dense_clone_parity_and_no_live_mutation(self):
        delta=tuple(torch.randn_like(w)*.1 for w in self.view.entry)
        weights=self.view.weights_for_delta(delta);clone=copy.deepcopy(self.model)
        with torch.no_grad():
            for n,w in zip(self.view.names,weights):dict(clone.named_parameters())[n].copy_(w)
        for batch in pack(self.view,rows(),0,2):
            out=clone(batch.input_ids,batch.attention_mask).logits[batch.positions]
            self.assertTrue(torch.equal(batch.logits(weights),out))
        self.view.assert_live(bytes_check=True)

    def test_global_microbatch_gradient(self):
        d=tuple(torch.randn_like(w)*.03 for w in self.view.entry)
        ref=current_callback(self.view,rows(),0,1)(d)
        for micro in (2,4):
            got=current_callback(self.view,rows(),0,micro)(d)
            self.assertAlmostEqual(got[0],ref[0],places=6)
            for a,b in zip(got[1],ref[1]):self.assertTrue(torch.allclose(a,b,atol=1e-7,rtol=1e-5))

    def test_joint_jvp_symmetric_fd(self):
        batch=next(pack(self.view,rows(),0,4));w=tuple(x+.07 for x in self.view.entry)
        d=tuple(torch.randn_like(x) for x in w)
        _,j=torch.func.jvp(batch.logits,(w,),(d,))
        e=2**-9;fd=(batch.logits(tuple(a+e*b for a,b in zip(w,d)))-batch.logits(tuple(a-e*b for a,b in zip(w,d))))/(2*e)
        self.assertTrue(torch.allclose(j,fd,atol=3e-4,rtol=3e-3))
        self.view.assert_live(bytes_check=True)

    def test_teacher_repacking_and_target_identity(self):
        saved=teacher(self.view,self.view.entry,rows(),0,1)
        for b in pack(self.view,rows(),0,2):
            t=bind_teacher(b,saved)
            self.assertTrue(torch.allclose(t['teacher_logp'],b.logits(self.view.entry).log_softmax(-1),atol=1e-6))
        saved[0]['target_ids'][0]=9
        with self.assertRaises(StateBoundary):bind_teacher(next(pack(self.view,rows(),0,2)),saved)

    def test_unselected_parameter_mutation_rejected(self):
        with torch.no_grad():self.model.embedding.weight.add_(.1)
        with self.assertRaises(StateBoundary):self.view.assert_live()

if __name__=='__main__':unittest.main()
