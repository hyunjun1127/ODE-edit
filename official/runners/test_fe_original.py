import tempfile,unittest,ast,threading,time
from unittest.mock import patch
from pathlib import Path
import torch
from official.runners import fe_original as f
from official.runners.fe_original_compat import check
ROOT=Path('/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author')
class Tests(unittest.TestCase):
 def test_bfloat16_hash_bytes(self):
  import hashlib
  x=torch.tensor([1,2,3],dtype=torch.bfloat16)
  self.assertEqual(f.tensor_sha(x),hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest())
  self.assertNotEqual(f.tensor_sha(x),f.tensor_sha(x.float()))
 def test_author_default_attention(self):
  tree=ast.parse(Path(f.__file__).read_text())
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='from_pretrained']
  self.assertTrue(calls)
  self.assertTrue(all('attn_implementation' not in [k.arg for k in n.keywords] for n in calls))
 def test_payload_authoritative_after_metadata_crash(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);identity={'run':1}
   f.save_latest(root,{'w':torch.ones(2,dtype=torch.bfloat16)},torch.zeros(1),[],{},5,identity,{'edits':500})
   old=(root/'latest.json').read_bytes();replace=f.os.replace
   def crash(src,dst):
    if Path(dst).name=='latest.json':raise OSError('metadata crash')
    return replace(src,dst)
   with patch.object(f.os,'replace',side_effect=crash):
    with self.assertRaises(OSError):f.save_latest(root,{'w':torch.ones(2)*2},torch.zeros(1),[],{},10,identity,{'edits':1000})
   self.assertEqual((root/'latest.json').read_bytes(),old)
   self.assertEqual(f.load_latest(root,identity)['batch'],10)
   self.assertFalse(list(root.glob('*.partial')))
   (root/'latest.json').unlink()
   self.assertEqual(f.load_latest(root,identity)['batch'],10)
   payload=torch.load(root/'latest.pt',weights_only=False);payload['weights']['w'][0]=99;torch.save(payload,root/'latest.pt')
   with self.assertRaises(AssertionError):f.load_latest(root,identity)
 def test_resume_observations_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);f.write_new(root/'commits/batch-06.json',{'elapsed':1})
   first=f.observation_root(root,True);second=f.observation_root(root,True)
   self.assertNotEqual(first,second)
   f.write_new(first/'commits/batch-06.json',{'elapsed':2})
   self.assertEqual(f.read(root/'commits/batch-06.json'),{'elapsed':1})
   self.assertEqual(f.read(first/'commits/batch-06.json'),{'elapsed':2})
 def test_pinned_native_profiles(self):
  for model,clamp,steps in [('llama3',.75,35),('gptj',.75,25),('qwen25',1,35)]:
   c=f.config_native(ROOT,model,'cf');self.assertEqual(c.llms.clamp_norm_factor,clamp);self.assertEqual(c.llms.v_num_grad_steps,steps);self.assertEqual(c.model_dtype,'bfloat16')
 def test_exact_diff(self):self.assertTrue(check(ROOT))
 def test_author_entrypoints(self):
  p,n=f.modules(ROOT);self.assertIn(str(ROOT),n.batch_edit.__code__.co_filename);self.assertIn(str(ROOT),p.compute_z_batch_first_forward.__code__.co_filename)
 def test_fixed_z_schedule_source(self):
  source=Path(f.__file__).read_text();self.assertEqual(source.count('pre.compute_z_batch_first_forward('),1)
  self.assertLess(source.index('pre.compute_z_batch_first_forward('),source.index('for batch in range(start+1,21)'))
  self.assertNotIn('memit_fe_history',source);self.assertIn("'current/post'",source);self.assertIn("'all_seen/post'",source)
 def test_latest_resume(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d);w={'w':torch.ones(2,3,dtype=torch.bfloat16)};h=torch.eye(3)[None];identity={'test':1}
   for b in [5,10,15,20]:
    w['w'].fill_(b);f.save_latest(path,w,h,[['{}']],{'test':'z'},b,identity,{'edits':b*100})
    v=f.load_latest(path,identity);self.assertTrue(torch.equal(v['weights']['w'],w['w']));self.assertTrue(torch.equal(v['cache_c'],h));self.assertEqual(v['batch'],b)
    self.assertEqual(len(list(path.glob('*.pt'))),1);self.assertFalse(list(path.glob('*.partial')))
   with self.assertRaises(AssertionError):f.load_latest(path,{'wrong':1})
 def test_no_W0_checkpoint(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(AssertionError):f.save_latest(Path(d),{},torch.zeros(1),[],{},0,{}, {})
 def test_shared_lock_serializes(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);active=[];peaks=[];real=torch.save;errors=[]
   def slow(*args,**kwargs):
    active.append(1);peaks.append(len(active));time.sleep(.03)
    try:return real(*args,**kwargs)
    finally:active.pop()
   def worker(n):
    try:f.save_latest(root/str(n),{'w':torch.ones(2)},torch.zeros(1),[],{},5,{'run':n},{},lock_path=root/'host.lock')
    except BaseException as e:errors.append(e)
   with patch.object(torch,'save',slow):
    threads=[threading.Thread(target=worker,args=(n,)) for n in (1,2)]
    for t in threads:t.start()
    for t in threads:t.join()
   self.assertEqual(errors,[]);self.assertEqual(max(peaks),1)
   self.assertTrue((root/'1/latest.pt').exists() and (root/'2/latest.pt').exists())
 def test_tmp_failure_cleanup(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   with patch.object(torch,'save',side_effect=OSError('fixture')):
    with self.assertRaises(OSError):f.save_latest(root,{'w':torch.ones(2)},torch.zeros(1),[],{},5,{}, {})
   self.assertFalse((root/'latest.pt.partial').exists())
   f.save_latest(root,{'w':torch.ones(2)},torch.zeros(1),[],{},5,{}, {})
if __name__=='__main__':unittest.main()
