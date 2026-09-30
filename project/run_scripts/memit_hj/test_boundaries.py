"""Failure injection and actual library dtype / scheduler boundary fixtures."""
import json,os,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from .precision import fp64_reference
from .oracle import Oracle
from .state import States,Checkpoints
from . import submit
from project.run_scripts.memit_history_lifelong.io import save,file_sha

class Boundaries(unittest.TestCase):
 def test_actual_library_FP64_and_finally(self):
  from transformers.models.llama.modeling_llama import LlamaRMSNorm
  model=torch.nn.Sequential(LlamaRMSNorm(3));original=model[0].forward.__func__
  oracle=types.SimpleNamespace(prefix={'hidden':torch.ones(1,3)},initial=torch.ones(1,3),kl=torch.ones(1,2),precision='float32')
  def dtype(d):
   oracle.prefix={'hidden':oracle.prefix['hidden'].to(d)};oracle.initial=oracle.initial.to(d);oracle.kl=oracle.kl.to(d)
  oracle.dtype=dtype;before=model[0].weight.detach().clone()
  with patch('torch.cuda.empty_cache'):
   with self.assertRaisesRegex(RuntimeError,'injected'):
    with fp64_reference(model,oracle) as observed:
     x=torch.tensor([[1.,2.,3.]],dtype=torch.float64,requires_grad=True);y=model(x)
     self.assertEqual(y.dtype,torch.float64);self.assertEqual(torch.autograd.grad(y.sum(),x)[0].dtype,torch.float64)
     self.assertEqual(observed[0]['FP32_constants_promoted'],1)
     raise RuntimeError('injected')
  self.assertIs(model[0].forward.__func__,original);self.assertTrue(torch.equal(before,model[0].weight));self.assertEqual(model[0].weight.dtype,torch.float32);self.assertEqual(oracle.prefix['hidden'].dtype,torch.float32)
 def test_matched_adam_final_call_budget(self):
  class Quadratic(Oracle):
   def __init__(self):self.initial=torch.ones(1,2);self.hp=types.SimpleNamespace(v_lr=.1);self.radius=10.;self.n=0
   def __call__(self,x):self.n+=1;return float(((x-1)**2).sum().detach()),2*(x-1),{}
  q=Quadratic();x,r=q.adam(2);self.assertTrue(torch.all(x>0));self.assertEqual(q.n,2);self.assertEqual(r['calls'],2)
 def test_cp_failed_atomic_write_and_scalar_metadata(self):
  with tempfile.TemporaryDirectory() as td,patch('torch.cuda.synchronize'),patch('torch.cuda.get_rng_state_all',return_value=[]),patch('torch.cuda.set_rng_state_all'):
   s=States({'x':torch.zeros(1)},torch.zeros(1,1,1),types.SimpleNamespace(CONTEXT_TEMPLATES_CACHE=[],COV_CACHE={}),None,None,None);s.probe=lambda:'probe'
   cp=Checkpoints(td,s,{'version':str(torch.__version__)})
   with patch('torch.save',side_effect=OSError('injected_disk_error')):
    with self.assertRaisesRegex(OSError,'injected_disk_error'):cp.write('000',1000,{'ledger':[0],'cursor':1})
   self.assertFalse(list(Path(td).glob('*.pt')));self.assertFalse(cp.paths)
   save(Path(td)/'version.json',{'version':torch.__version__});self.assertEqual(json.loads((Path(td)/'version.json').read_text())['version'],str(torch.__version__))
 def test_production_submit_full_DAG_no_monitoring(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);src=root/'src/project/run_scripts/memit_hj';src.mkdir(parents=True)
   for name in ['run.sbatch','collect.sbatch']:(src/name).write_text('# fixture\n')
   lock=dict(resource=dict(cap=1,memory_mib=121856),instruction='fixture',source_commit='frozen',source_root=str(root/'src'),
    launchers={p.name:file_sha(p) for p in src.iterdir()},plan=dict(groups={g:[] for g in 'PABCD'}))
   save(root/'lock.json',lock);registered={};events=[]
   def fake(argv):
    events.append(argv)
    if argv[0]=='id':return 'janghj\n'
    if argv[0]=='squeue':return '42|odeedit_prior|RUNNING|gres/gpu:1|ubuntu\n'
    if argv[0]=='sbatch':
     job=str(100+len(registered));registered[job]=argv;return job+'\n'
    if argv[:2]==['scontrol','release']:return ''
    if argv[:3]==['scontrol','show','job']:
     job=argv[3];a=registered[job];group=next(x for x in a if x.startswith('--job-name=')).split('=')[1].split('_')[-2]
     dep=next((x.split('=',1)[1] for x in a if x.startswith('--dependency=')),'(null)')
     return ' '.join([f'JobId={job}','UserId=janghj(1000)',f'JobName=odeedit_memit_hj_{group}_s3','JobState=PENDING','Priority=0','ReqNodeList=ubuntu','Partition=gpu','NumCPUs=8','CPUs/Task=8',
      'ReqTRES=cpu=8,mem=119G,gres/gpu=1' if group!='CPU' else 'ReqTRES=cpu=8,mem=32G',
      'TresPerNode=gres/gpu:1' if group!='CPU' else 'TresPerNode=(null)',
      'MinMemoryNode=119G' if group!='CPU' else 'MinMemoryNode=32G','Requeue=0',
      'TimeLimit=30-00:00:00' if group!='CPU' else 'TimeLimit=04:00:00',f'Command={a[-4]}',f'Dependency={dep}','SubmitLine='+' '.join(a)])
    raise AssertionError(argv)
   import contextlib,io
   with patch.object(submit,'run',fake),contextlib.redirect_stdout(io.StringIO()):submit.submit(root/'lock.json')
   self.assertEqual(len(registered),6)
   self.assertEqual([a[2] for a in events if a[:2]==['scontrol','release']],['105','104','103','102','101','100'])
   self.assertEqual(submit.dependency('B',{'P':'100','A':'101'},[]),'afterok:100,afterany:101')
   self.assertEqual(submit.dependency('CPU',{g:str(i) for i,g in enumerate('PABCD')},[]),'afterany:0:1:2:3:4')
   with patch.object(submit,'run',fake):
    with self.assertRaises(FileExistsError):submit.submit(root/'lock.json')
   self.assertEqual(len(registered),6)
 def test_recovery_rejects_changed_source_and_corruption(self):
  from . import recovery
  with tempfile.TemporaryDirectory() as td,patch('torch.cuda.synchronize'),patch('torch.cuda.get_rng_state_all',return_value=[]),patch('torch.cuda.set_rng_state_all'):
   root=Path(td);s=States({'x':torch.ones(2)},torch.zeros(1,2,2),types.SimpleNamespace(CONTEXT_TEMPLATES_CACHE=[],COV_CACHE={}),None,None,None);s.probe=lambda:'p'
   cp=Checkpoints(root/'attempt/output/temporary-checkpoints',s,{'source':'fixed'})
   rec=cp.write('000',1000,dict(cell='main_000',cursor=1000,ledger=list(range(1000))))
   with patch.object(recovery,'TASK_ROOT',root):
    snap,m=recovery.payload(rec['path'],{'source':'fixed'});self.assertEqual(snap['meta']['cursor'],1000)
    with self.assertRaises(AssertionError):recovery.payload(rec['path'],{'source':'changed'})
    with open(rec['path'],'ab') as f:f.write(b'corrupt')
    with self.assertRaises(AssertionError):recovery.payload(rec['path'],{'source':'fixed'})
 def test_Z_block_before_model_preserves_nonZ(self):
  from .runner import pre_model_z_gate
  with tempfile.TemporaryDirectory() as td:
   out=Path(td);binding={'source':'fixed'};save(out/'calibration/lock.json',dict(status='BLOCKED',bindings=binding,reasons=['CENSORED_OR_FAILED']))
   lock={'bindings':binding}
   self.assertFalse(pre_model_z_gate(lock,'B',out,out/'groups/B'))
   self.assertTrue(pre_model_z_gate(lock,'C',out,out/'groups/C'))
   self.assertFalse((out/'cells/main_100').exists())
   r=json.loads((out/'groups/C/terminal.json').read_text());self.assertFalse(r['model_loaded']);self.assertEqual(len(r['coverage']),2)
 def test_independent_NLL_and_input_identity(self):
  from .reducer import expected_identities,verify_identities,verify_metric
  from project.run_scripts.memit_history_lifelong.metrics import summarize
  from project.run_scripts.memit_history_lifelong.io import digest
  row=dict(case_id=17,requested_rewrite=dict(subject='a',prompt='{} is',target_new={'str':'b'},target_true={'str':'c'}),paraphrase_prompts=['P','P2'],neighborhood_prompts=['N']*10)
  expected=expected_identities([row]);identity=digest([17,0,'a is','b','c'])
  item=dict(case_id=17,prompt_index=0,identity=identity,new_nll=1.,true_nll=2.,margin=1.,success=True,new_token_correct=1,new_token_count=1,new_strict=True,true_token_correct=0,true_token_count=1,true_strict=False)
  m=dict(rows=[item],**summarize([item],'RS'));verify_identities({'metrics':{'RS':m}},expected);verify_metric(m,'RS')
  m['new_nll']=2.
  with self.assertRaisesRegex(AssertionError,'NLL_REDUCER'):verify_metric(m,'RS')
  item['identity']='wrong'
  with self.assertRaisesRegex(AssertionError,'RAW_INPUT_IDENTITY'):verify_identities({'metrics':{'RS':m}},expected)
if __name__=='__main__':unittest.main()
