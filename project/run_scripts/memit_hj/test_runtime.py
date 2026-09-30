"""CPU regressions for persistence, routing and the real pinned AST path."""
import contextlib,copy,importlib,io,json,os,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from .state import States,Checkpoints,metadata_hash
from .spg import calibrate
from .reducer import verify_metric,transitions
from .plan import cells

class Runtime(unittest.TestCase):
 def test_state_and_checkpoint(self):
  weights={'w':torch.arange(6,dtype=torch.float32).reshape(2,3)};H=torch.eye(3)[None]
  m=types.SimpleNamespace(CONTEXT_TEMPLATES_CACHE=[['{}']],COV_CACHE={'x':torch.eye(3)})
  with patch('torch.cuda.synchronize'),patch('torch.cuda.get_rng_state_all',return_value=[]),patch('torch.cuda.set_rng_state_all'):
   s=States(weights,H,m,None,None,None);s.probe=lambda:metadata_hash(weights)['w']['sha256'];meta=dict(cursor=1000,ledger=[1,1,2],samples=[{'k':torch.ones(2)}])
   snap=s.snapshot(meta);weights['w'].add_(2);H.mul_(2);meta['samples'][0]['k'].zero_()
   restored=s.restore(snap);self.assertEqual(restored['samples'][0]['k'].tolist(),[1,1]);self.assertEqual(s.identity(restored),snap['identity'])
   with tempfile.TemporaryDirectory() as td:
    cp=Checkpoints(td,s,dict(source='frozen'));a=cp.write('000',1000,restored);cp.pin(a['path'],1,'pending')
    b=cp.write('000',2000,restored);c=cp.write('000',3000,restored)
    self.assertTrue(Path(a['path']).exists());self.assertFalse(Path(b['path']).with_suffix('.tombstone.json').exists())
    with self.assertRaises(AssertionError):cp.delete(a['path'],'illegal_pinned')
    cp.pin(a['path'],-1,'done');cp.delete(a['path'],'released');self.assertFalse(Path(a['path']).exists())
    with self.assertRaises(ValueError):cp.write('010',1000,restored)
    weights['w'].zero_();cp.load(c['path'],c['sha256']);self.assertEqual(s.identity(restored),snap['identity'])
 def test_calibration_fail_closed(self):
  p=[dict(anchor=a,normalized_error=1e-7) for a in [0,1000] for _ in range(48)]
  r=[dict(anchor=a,calls=25,status='CONVERGED') for a in [0,1000] for _ in range(16)]
  self.assertEqual(calibrate(p,r)['cap'],32)
  r[0].update(status='NOT_CONVERGED',calls=400);c=calibrate(p,r);self.assertEqual(c['status'],'BLOCKED');self.assertIn('CENSORED_OR_FAILED',c['reasons'])
  r=[dict(anchor=a,calls=350,status='CONVERGED') for a in [0,1000] for _ in range(16)]
  self.assertGreater(calibrate(p,r)['diagnostic_proposed_cap'],400);self.assertIsNone(calibrate(p,r)['cap'])
 def test_cells_and_reference_hashes(self):
  root=Path(__file__).resolve().parents[3];c=cells(root/'plans/global/2026-09-30-memit-hj-experiment-design-v1/cells.csv')
  self.assertEqual(len([x for x in c if x['family']!='main']),20)
  self.assertTrue(all(x['batch_size']=='10' and x['total_batches']=='100' for x in c if x['family']!='main'))
 def test_metric_ties_and_target_direction(self):
  from project.run_scripts.memit_history_lifelong.metrics import summarize
  r=dict(identity='a',case_id=1,new_nll=2.,true_nll=1.,margin=-1.,success=True,new_token_correct=0,new_token_count=2,new_strict=False,true_token_correct=2,true_token_count=2,true_strict=True)
  m=dict(rows=[r],**summarize([r],'NS'));self.assertEqual(verify_metric(m,'NS')['tf_strict'],1.)
  bad=copy.deepcopy(m);bad['rows'][0]['success']=False
  with self.assertRaises(AssertionError):verify_metric(bad,'NS')
  r2=dict(r,success=False);self.assertEqual(transitions([r],[r2])['lost'],['a'])
 def test_actual_pinned_adapter_history_timing(self):
  # Tiny CPU model with the real pinned BLUE execute/apply function bodies.
  # Fake keys depend on prior writes, so a pre-write history append fails.
  import sys
  from .writer import Adapter
  blue='/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/blue-upstream'
  cwd=os.getcwd();sys.path.insert(0,blue)
  try:
   os.chdir(blue);native=importlib.import_module('memit.memit_seq_main')
  finally:os.chdir(cwd)
  m=types.ModuleType('fixture');m.__dict__.update(native.__dict__)
  hp=types.SimpleNamespace(blue=False,layers=list(range(4,9)),rewrite_module_tmp='layer.{}',layer_module_tmp='layer.{}',fact_token='subject_last',mom2_update_weight=15000,mom2_dataset='x',mom2_n_samples=1,mom2_dtype='float32')
  ws={f'layer.{l}.weight':torch.zeros(3,4) for l in hp.layers};H=torch.zeros(5,4,4)
  m.nethook=types.SimpleNamespace(get_parameter=lambda model,n:ws[n]);m.get_context_templates=lambda *args:[['{}']]
  m.get_cov=lambda *args,**kw:torch.eye(4)
  def keys(model,tok,requests,hp,l,ctx):return torch.tensor([[1.,2.,3.,4.],[2.,1.,0.,1.]])+sum(float(w.sum()) for w in ws.values())*.01
  m.compute_ks=keys;m.get_module_input_output_at_words=lambda model,tok,l,**kw:(torch.zeros(2,4),torch.stack([sum(ws.values()).sum(1)]*2))
  a=Adapter.__new__(Adapter);a.module=m;a.model=types.SimpleNamespace();a.tok=None;a.hp=hp;a.H=H;a.zlock=None;a.anchor_H=None
  a.native_z=lambda *args:torch.ones(3)
  a.function=a.compile();ns=dict(m.__dict__,execute_memit=a.function)
  f=m.apply_memit_seq_to_model;a.apply=types.FunctionType(f.__code__,ns,f.__name__,f.__defaults__,f.__closure__)
  req=[dict(case_id=i,prompt='{} is',subject='x',target_new={'str':' y'}) for i in [1,2]]
  original_to=torch.Tensor.to
  def cpu_to(t,*args,**kw):
   if args and isinstance(args[0],str) and args[0].startswith('cuda'):return t
   return original_to(t,*args,**kw)
  with patch.object(torch.Tensor,'cuda',lambda t,*args,**kw:t),patch.object(torch.Tensor,'to',cpu_to),patch('torch.cuda.synchronize'),patch('torch.cuda.empty_cache'),contextlib.redirect_stdout(io.StringIO()):
   rec=a.run(req);post=keys(None,None,req,hp,4,None).T
   for h in H:torch.testing.assert_close(h,post@post.T)
   self.assertEqual([r['layer'] for r in rec['append']],hp.layers)
   prior=H.clone();a.run(req);self.assertTrue(torch.all(H.diagonal(dim1=1,dim2=2)>prior.diagonal(dim1=1,dim2=2)))
   from .engine import Engine
   entry=dict(weights={k:w.clone() for k,w in ws.items()},H=H.clone())
   def restore(s):
    for k,w in ws.items():w.copy_(s['weights'][k])
    H.copy_(s['H'])
   e=types.SimpleNamespace(adapter=a,restore=restore,model=a.model,tok=None,module=m,hp=hp,weights=ws,H=H,total={'shadow_requests':0})
   energy=Engine.energy_write(e,req,entry,0);post=keys(None,None,req,hp,4,None).T
   for i,h in enumerate(H):torch.testing.assert_close(h,entry['H'][i]+post@post.T)
   self.assertEqual(e.total['shadow_requests'],4)
   self.assertEqual(energy['history_appends'],5);self.assertGreater(energy['energy_scale'],0)
if __name__=='__main__':unittest.main()
