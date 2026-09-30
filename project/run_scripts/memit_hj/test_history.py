import copy,types,unittest
from unittest.mock import patch
import torch
from .history import History
class HistoryTests(unittest.TestCase):
 def test_context_recipe_matches_native_and_reference(self):
  raw=torch.arange(18,dtype=torch.float32).reshape(6,3)+1
  m=types.SimpleNamespace(CONTEXT_TEMPLATES_CACHE=[['{}'],['{}']*5],get_module_input_output_at_words=lambda *a,**kw:(raw,None))
  rows=[dict(requested_rewrite=dict(prompt='{} x',subject='subject'))]
  h=History(m,None,None,types.SimpleNamespace(rewrite_module_tmp='x{}',fact_token='subject_last'),None,rows)
  k,d=h.keys([0],4,True)
  torch.testing.assert_close(k[:,0],(raw[0]+raw[1:].mean(0))/2)
  center=raw.mean(0);expected=((raw-center).square().sum(1).mean()).sqrt()/center.norm()
  torch.testing.assert_close(d[0],expected)
 def test_rebuild_all_occurrences_and_reset_reference(self):
  H=torch.zeros(5,3,3);k=torch.tensor([[1.,2.,1.,3.],[2.,1.,2.,1.],[1.,1.,1.,0.]])
  h=History(None,None,None,None,H,None)
  h.keys=lambda ids,l,context=False:(k[:,ids],torch.full((len(ids),),.25)) if context else k[:,ids]
  samples=[dict(ordinal=i,origin={l:torch.zeros(3) for l in range(4,9)},reference={l:torch.zeros(3) for l in range(4,9)},dispersion={l:None for l in range(4,9)}) for i in range(4)]
  meta=dict(cursor=4,ledger=list(range(4)),samples=samples)
  original=torch.zeros
  def zeros(*a,**kw):
   if kw.get('device')=='cuda':kw['device']='cpu'
   return original(*a,**kw)
  with patch.object(torch.Tensor,'cuda',lambda t,*a,**kw:t),patch('torch.zeros',zeros):r=h.rebuild(meta,[5,6,7,8])
  for i in range(1,5):torch.testing.assert_close(H[i],k@k.T)
  self.assertTrue(torch.equal(H[0],torch.zeros(3,3)))
  self.assertEqual(r['occurrences'],4);self.assertEqual(meta['ledger'],[0,1,2,3])
  for s in samples:
   self.assertTrue(torch.equal(s['origin'][5],torch.zeros(3)));torch.testing.assert_close(s['reference'][5],k[:,s['ordinal']])
  layers,obs=h.trigger(meta);self.assertEqual(layers,[]);self.assertEqual(obs['4']['status'],'TRIGGER_UNDEFINED')
if __name__=='__main__':unittest.main()
