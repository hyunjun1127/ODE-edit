"""Small CPU functional/JVP fixtures; not an actual Llama numerical PASS."""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from .response import RepairResponse, ResponseError


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(9,4)
        self.repair = nn.Linear(4,4,bias=False)
        self.head = nn.Linear(4,9,bias=False)
    def forward(self,input_ids,attention_mask,use_cache=False):
        x = self.emb(input_ids)*attention_mask[...,None]
        x = x.cumsum(1)/attention_mask.cumsum(1).clamp_min(1)[...,None]
        return SimpleNamespace(logits=self.head(torch.tanh(self.repair(x))))


class TinyTok:
    bos_token_id=0;unk_token_id=8;pad_token_id=8;eos_token_id=8
    def encode(self,text,add_special_tokens=False):
        return ([0] if add_special_tokens else [])+[1+ord(c)%7 for c in text.strip()]
    def __call__(self,text,add_special_tokens=True):
        return {'input_ids':self.encode(text,add_special_tokens=add_special_tokens)}


class TinyTeacher:
    def __init__(self,model):
        self.ids = torch.arange(257)[None,:]%9
        with torch.no_grad():
            self.logp = model(self.ids,torch.ones_like(self.ids)).logits[:,128:256,:].log_softmax(-1).clone()
    def indices(self,role):
        if role not in ('S64','Dev128'): raise ValueError(role)
        return [0,1]
    def document(self,index,device):
        return self.ids.to(device),self.logp.to(device),str(index)


class ResponseTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91);torch.set_num_threads(1)
        self.model=TinyLM().eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.teacher=TinyTeacher(self.model)
        with torch.no_grad():self.model.repair.weight.add_(torch.randn_like(self.model.repair.weight)*.1)
        self.adapter=RepairResponse(self.model,TinyTok(),self.teacher,weight_name='repair.weight',expected_vocab=9,microbatch=2)
        self.records=[dict(case_id=i,requested_rewrite=dict(subject='x',prompt='{} '+('z'*i),
            target_new={'str':new},target_true={'str':old})) for i,(new,old) in enumerate((('ab','cd'),('a','c'),('cde','f')))]
    def direction(self):
        q=torch.randn_like(self.model.repair.weight);return q/q.norm()
    def test_functional_physical_and_no_mutation(self):
        weight=self.model.repair.weight.clone()+self.direction()*.03
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        result,_=self.adapter.panel(self.records,weight=weight)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in self.model.state_dict().items()))
        with torch.no_grad():self.model.repair.weight.copy_(weight)
        physical,_=self.adapter.panel(self.records)
        self.assertEqual(result['rows'],physical['rows'])
        self.assertEqual(result['E'],physical['E'])
    def test_panel_gradient_and_request_mass(self):
        panel,g=self.adapter.panel(self.records,gradient=True)
        individual=[];grads=[]
        for record in self.records:
            result,grad=self.adapter.panel([record],gradient=True)
            individual.append(result['E']);grads.append(grad)
        self.assertAlmostEqual(panel['E'],sum(individual)/3,places=6)
        torch.testing.assert_close(g,sum(grads)/3,rtol=2e-5,atol=2e-7)
        self.assertEqual(panel['counts']['backwards'],2)
        q=self.direction();h=.01
        plus,_=self.adapter.panel(self.records,weight=self.model.repair.weight+h*q)
        minus,_=self.adapter.panel(self.records,weight=self.model.repair.weight-h*q)
        self.assertAlmostEqual(float((g*q).sum()),(plus['E']-minus['E'])/(2*h),delta=3e-5)
    def test_base_gradient_jvp_and_fisher(self):
        base,g=self.adapter.base(gradient=True)
        q=self.direction()
        response=self.adapter.response([q],[],[],None,None,include_guards=False)
        self.assertAlmostEqual(float((g*q).sum()),float(response['b'][0]),delta=3e-7)
        self.assertGreaterEqual(float(response['H'][0,0]),0.)
        self.assertEqual(response['A'].shape,(0,1))
        self.assertEqual(response['counts']['guard_jvp_forwards'],0)
        ids=self.teacher.ids
        def logits(w):
            return torch.func.functional_call(self.model,{'repair.weight':w},(),dict(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)).logits[:,128:256,:]
        z,j=torch.func.jvp(logits,(self.model.repair.weight,),(q,))
        p=z.softmax(-1).double();j=j.double()
        expected=((p*j*j).sum(-1)-(p*j).sum(-1)**2).mean()
        self.assertAlmostEqual(float(expected),float(response['H'][0,0]),places=11)
        self.assertEqual(base['counts']['scored_tokens'],256)
    def test_guard_mean_preference_and_fd(self):
        panel,_=self.adapter.panel(self.records)
        q=self.direction()
        response=self.adapter.response([q],self.records,[],panel,None)
        self.assertEqual(response['guard_rows'][0]['kind'],'mean_nll')
        self.assertEqual(float(response['s'][0]),0.)
        _,g=self.adapter.panel(self.records,gradient=True)
        self.assertAlmostEqual(float(response['A'][0,0]),float((g*q).sum()),delta=3e-7)
        h=.01
        plus,_=self.adapter.panel(self.records,weight=self.model.repair.weight+h*q)
        minus,_=self.adapter.panel(self.records,weight=self.model.repair.weight-h*q)
        plus={r['case_id']:r for r in plus['rows']};minus={r['case_id']:r for r in minus['rows']}
        for index,row in enumerate(response['guard_rows']):
            if row['kind']=='new_old_nll_preference':
                cid=row['case_id'];fd=-(plus[cid]['preference_margin']-minus[cid]['preference_margin'])/(2*h)
                self.assertAlmostEqual(float(response['A'][index,0]),fd,delta=3e-5)
    def test_missing_old_and_empty_past(self):
        records=copy.deepcopy(self.records)
        for r in records:del r['requested_rewrite']['target_true']
        p,_=self.adapter.panel(records)
        self.assertEqual(p['preference_status'],'NOT_AVAILABLE')
        self.assertFalse(p['preference_ids'])
        empty,g=self.adapter.panel([],gradient=True)
        self.assertIsNone(g);self.assertIsNone(empty['E'])
    def test_strict_margin_competitor_fixed(self):
        # Select a one-token target that is the actual prediction, then verify
        # its fixed non-target competitor and signed JVP inequality row.
        record=copy.deepcopy(self.records[0]);record['requested_rewrite']['target_new']['str']='a'
        for char in 'abcdefg':
            record['requested_rewrite']['target_new']['str']=char
            panel,_=self.adapter.panel([record])
            if panel['strict_ids']:break
        if not panel['strict_ids']:
            self.skipTest('Tiny random fixture has no representable strict token')
        q=self.direction();r=self.adapter.response([q],[record],[],panel,None)
        rows=[(i,row) for i,row in enumerate(r['guard_rows']) if row['kind']=='strict_token_margin']
        self.assertEqual(len(rows),1)
        idx,row=rows[0]
        self.assertNotEqual(row['target_token_id'],row['competitor_id'])
        self.assertGreaterEqual(float(r['s'][idx]),0.)
    def test_nonfinite_and_observer_gradient_rejected(self):
        with self.assertRaises(ResponseError):self.adapter.base('Dev128',gradient=True)
        bad=self.model.repair.weight.clone();bad[0,0]=float('nan')
        with self.assertRaises(ResponseError):self.adapter.panel(self.records,weight=bad)
        with self.assertRaises(ResponseError):self.adapter.panel(self.records+[self.records[0]])


if __name__=='__main__':unittest.main()
