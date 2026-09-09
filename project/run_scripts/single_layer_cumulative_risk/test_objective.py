import types
import unittest
import torch
from .objective import DirectObjective,WEIGHT
from .records import Ledger

class Tokenizer:
    padding_side='right';pad_token_id=0;bos_token_id=-1;unk_token_id=-2
    def __call__(self,text,return_tensors=None):
        ids=[ord(x)%8+1 for x in text]
        return {'input_ids':torch.tensor([ids]) if return_tensors else ids}
    def decode(self,ids):return ''.join(chr(64+x) for x in ids)

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embed=torch.nn.Embedding(10,4)
        self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([torch.nn.Module() for _ in range(5)])
        self.model.layers[4].mlp=torch.nn.Module();self.model.layers[4].mlp.down_proj=torch.nn.Linear(4,3,bias=False)
        self.out=torch.nn.Linear(3,10,bias=False)
    def forward(self,input_ids,attention_mask=None,use_cache=False):
        x=self.embed(input_ids);x=self.model.layers[4].mlp.down_proj(x)
        return types.SimpleNamespace(logits=self.out(torch.tanh(x)))

class ObjectiveTests(unittest.TestCase):
    def test_actual_objective_gradient_fd_and_group_average(self):
        torch.manual_seed(20260910);model=Toy().double().requires_grad_(False)
        w=dict(model.named_parameters())[WEIGHT].detach().clone();u,_=torch.linalg.qr(torch.randn(4,2,dtype=torch.float64))
        requests=[dict(prompt='{} lives in',subject=s,target_new={'str':' xy'}) for s in ['Ab','Cd']]
        objective=DirectObjective(model,Tokenizer(),requests,[['{}','In summary: {}']],lambda *args,**kw:0,
                                 w,u,torch.eye(2,dtype=torch.float64),2.,Ledger(),microbatch=1)
        a=(torch.randn(3,2,dtype=torch.float64)*.02).requires_grad_();direction=torch.randn_like(a)
        value=objective.evaluate(a,True);gradient=a.grad.clone();epsilon=1e-5
        objective.accumulate_weight_gradient=True
        accumulated=objective.evaluate(a,True)
        torch.testing.assert_close(a.grad,gradient,atol=1e-12,rtol=1e-12)
        self.assertEqual(value,accumulated)
        plus=objective.evaluate((a.detach()+epsilon*direction).requires_grad_(),False)['objective']
        minus=objective.evaluate((a.detach()-epsilon*direction).requires_grad_(),False)['objective']
        self.assertAlmostEqual((plus-minus)/(2*epsilon),float((gradient*direction).sum()),places=8)
        individual,_=objective.group_edit_gradients(a,[[0],[1]])
        combined,_=objective.group_edit_gradients(a,[[0,1]])
        torch.testing.assert_close(individual.mean(0),combined[0],atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(dict(model.named_parameters())[WEIGHT],w,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
