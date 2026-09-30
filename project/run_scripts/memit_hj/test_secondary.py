import json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from .secondary import endpoint_at,factorial,compare
from project.run_scripts.memit_history_lifelong.io import save,file_sha
from project.run_scripts.memit_history_lifelong.metrics import summarize

class Secondary(unittest.TestCase):
 def test_long_alias_preserves_requested_2k_endpoint(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);save(root/'cells/main_111/terminal.json',dict(status='NOT_FIRED',parent='main_110',endpoint=10000))
   save(root/'cells/main_110/terminal.json',dict(status='COMPLETED',cursor=10000))
   expected=dict(requests=2000,metrics={k:{'denominator':n} for k,n in [('RS',2000),('PS',4000),('NS',20000)]})
   save(root/'cells/main_110/C02000/all-seen.json',expected)
   result,origin=endpoint_at(root,'111',2000);self.assertEqual(result,expected);self.assertEqual(origin['endpoint'],2000);self.assertFalse(origin['independent_replication'])
 def test_factorial_all8_and_blocked_not_zero(self):
  from . import secondary
  def rows(tag,win):
   true=1. if tag=='NS' else 2.;new=2. if tag=='NS' else 1.
   if not win:true,new=new,true
   return [dict(identity=f'{tag}:{i}',case_id=i,prompt_index=0,new_nll=new,true_nll=true,margin=true-new,success=win,new_token_correct=int(win),new_token_count=1,new_strict=win,true_token_correct=int(win),true_token_count=1,true_strict=win) for i in range(2)]
  def endpoint(output,arm,n):
   if arm in ['010','011','110','111']:return None,{'status':'BLOCKED_Z_CALIBRATION'}
   ms={tag:rows(tag,arm[0]=='1') for tag in ['RS','PS','NS']}
   return {'metrics':{tag:dict(rows=rr,**summarize(rr,tag)) for tag,rr in ms.items()}},{'status':'OBSERVED','artifact':arm}
  with tempfile.TemporaryDirectory() as td,patch.object(secondary,'endpoint_at',endpoint):
   table=factorial(td,Path(td));self.assertEqual(len(table),24)
   self.assertTrue(all(r['preference'] is None for r in table if r['arm']=='010'))
   raw=json.loads((Path(td)/'factorial-2k.json').read_text());self.assertEqual(len(raw['contrasts']),12)
   self.assertEqual(raw['contrasts']['100-000']['metrics']['PS']['preference_delta_pp'],100.)
   self.assertEqual(raw['contrasts']['110-010']['status'],'BLOCKED')
 def test_collector_secondary_failure_never_completed(self):
  from . import collector
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);out=root/'output';bindings={'runtime':'frozen'};lock={'source_commit':'runtime','output':str(out),'bindings':bindings,'cells':'fixture','dataset_root':str(root)}
   save(root/'lock.json',lock);save(root/'mapping.json',dict(source_commit='runtime',lock_sha256=file_sha(root/'lock.json'),jobs=[dict(job_id='1',group='P')]))
   save(root/'counterfact.json',[])
   for g in 'PABCD':save(out/'groups'/g/'terminal.json',dict(status='COMPLETED_GROUP',bindings=bindings))
   save(out/'cells/c/terminal.json',dict(status='COMPLETED'))
   with patch.object(collector,'cells',return_value=[{'cell_id':'c'}]),patch.object(collector,'reduce',return_value={}),patch.object(collector.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='',stderr='')),patch('project.run_scripts.memit_hj.figures.make'),patch('project.run_scripts.memit_hj.secondary.make',side_effect=OSError('injected_report_failure')):
    collector.collect(root/'lock.json',root/'mapping.json')
   r=json.loads((out/'terminal.json').read_text());self.assertEqual(r['status'],'TECHNICAL_INCOMPLETE')
   self.assertTrue((out/'collection/report-ko.md').is_file());self.assertTrue((out/'collection/manifest.json').is_file());self.assertIn('injected_report_failure',r['reducer_error']['error'])
if __name__=='__main__':unittest.main()
