import unittest
from .fe_qwen_move import append_graph,MOVE,read,validate,storage
from .submit import graph_width
from official.tracking.schema import config

class Tests(unittest.TestCase):
 def test_legacy_overcap_no_new_overlap(self):
  rows=[dict(key=str(i),gpus=1,parents=[]) for i in (1,2,3)]
  graph,legacy,width=append_graph(dict(jobs=rows,allocated_gpus=3,admitted_DAG_width=3))
  self.assertTrue(legacy);self.assertEqual(width,3)
  for deps in graph.values():self.assertEqual(set(deps),{'1','2','3'})
 def test_two_tails(self):
  rows=[dict(key=str(i),gpus=1,parents=[]) for i in (1,2)]
  graph,legacy,width=append_graph(dict(jobs=rows,allocated_gpus=2,admitted_DAG_width=2))
  self.assertFalse(legacy);self.assertEqual(width,2);self.assertEqual(graph,{'qwen25-cf':['1'],'qwen25-zsre':['2']})
 def test_actual_Qwen_configs_schema(self):
  ready=read(MOVE/'preparation/READY.json');self.assertEqual(len(ready['configs']),2)
  for m in ready['configs']:
   c=read(m['path']);validate(c);self.assertEqual(c['server'],'server1');self.assertEqual(c['model'],'qwen25')
   v=dict(server='server1',task_id='fe-original-w0-2k-20261011',model='qwen25',model_family='qwen2',writer='fe_author_w0_fixed',baseline='FE-author-repo-W0-fixed-z-sequential',role='scientific',arm=c['arm'],attempt=c['attempt'],source_sha='a'*40,config_sha=c['config_sha256'],dataset=c['dataset'],metric_schema='official-baselines-scalar-v1',instruction_id='USER-OFFICIAL-BASELINES-20261008-R1')
   if c['dataset']=='cf':v['generation_schedule']='DEFERRED_CHECKPOINT_EVALUATION'
   config(v)
 def test_storage_six_latest_one_tmp(self):
  s=storage();self.assertEqual(s['six_latest_bytes'],2*sum(s['latest_estimates'].values()))
  self.assertEqual(s['one_tmp_bytes'],max(s['latest_estimates'].values()));self.assertEqual(s['reserve_bytes'],64*1024**3)
if __name__=='__main__':unittest.main()
