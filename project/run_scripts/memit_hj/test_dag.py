"""Run the actual orchestration against an isolated state/receipt fixture."""
import copy,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from .orchestrator import Orchestrator
from .plan import cells
from project.run_scripts.memit_history_lifelong.io import save

class EngineFixture:
 def __init__(self,root,fired):
  self.output=Path(root);self.rows=[dict(case_id=i,requested_rewrite=dict(id=i)) for i in range(10000)]
  self.meta=dict(cursor=0,ledger=[],samples=[],triggers=[],commits=[],cell='W0');self.completed={};self.count=0;self.writes=0;self.refreshes=[];self.fired=fired
  self.adapter=types.SimpleNamespace(anchor_H=None,zlock=None);self.lock=dict(resource={'wall_hours':720},bindings={'source':'test'})
  self.states=types.SimpleNamespace(probe=lambda:self.meta['cursor'],identity=lambda m:copy.deepcopy(m))
  self.cp=types.SimpleNamespace(write=lambda arm,n,m:{'path':str(self.output/f'{arm}-{n}.pt')},pin=lambda *a:None,complete=lambda *a:None)
  self.w0=self.snapshot();self.visits=[]
 def snapshot(self):return dict(meta=copy.deepcopy(self.meta),H=[0]*5,identity=copy.deepcopy(self.meta))
 def restore(self,s,output_parity=None):
  self.meta=copy.deepcopy(s['meta'])
  if output_parity is not None:assert self.states.probe()==output_parity
 def observe(self,*a,**kw):pass
 def refresh(self,layers,path):self.refreshes.append((self.meta['cell'],self.meta['cursor'],layers))
 def trigger(self,path):return [5,6,7,8] if self.fired else []
 def step(self,c,bs,*args):
  self.visits.append((c['cell_id'],self.meta['cursor'],bs));self.meta['cursor']+=bs;self.count+=bs;self.writes+=1
  self.meta['ledger']=list(range(self.meta['cursor']))
 def complete(self,c,*args):self.completed[c['cell_id']]=dict(status='COMPLETED')

class DAG(unittest.TestCase):
 def run_plan(self,fired,z=True):
  plan=cells(Path(__file__).resolve().parents[3]/'plans/global/2026-09-30-memit-hj-experiment-design-v1/cells.csv')
  with tempfile.TemporaryDirectory() as td:
   e=EngineFixture(td,fired);o=Orchestrator(e,plan)
   def finish(w0,a):
    lock=dict(status='PASS' if z else 'BLOCKED',tol=1e-5,cap=32,bindings=e.lock['bindings']);save(e.output/'calibration/lock.json',lock);return lock
   o.probe=lambda n:None;o.cal.finish=finish;o.freeze_cost=lambda:None
   allcells={};counts=[]
   for g in 'ABCD':
    e.completed={};before=e.count;o.run_group(g);allcells.update(e.completed);counts.append(e.count-before)
   self.assertEqual(len(allcells),28)
   self.assertEqual(len(set(e.visits)),len(e.visits),'DUPLICATE_PHYSICAL_CELL_CURSOR')
   self.assertTrue(all(bs==10 for name,n,bs in e.visits if not name.startswith('main_')))
   self.assertTrue(all(cursor<10000 for _,cursor,_ in e.refreshes),'TERMINAL_REFRESH_FORBIDDEN')
   for arm in ['001','101','011']:
    visits=[n for name,n,bs in e.visits if name=='main_'+arm]
    if visits:self.assertEqual(visits,list(range(1000,2000,100)))
   return e.count,e.writes,counts,allcells
 def test_all_trigger_maximum_physical(self):
  n,w,c,_=self.run_plan(True);self.assertEqual((n,w),(64000,2440));self.assertEqual(c,[31000,11000,3000,19000])
 def test_no_trigger_alias_minimum(self):
  n,w,c,allcells=self.run_plan(False);self.assertEqual((n,w),(52000,2320))
  self.assertEqual(sum(v['status']=='NOT_FIRED' for v in allcells.values()),4)
 def test_z_fail_closed_nonZ_kept(self):
  n,w,c,allcells=self.run_plan(True,False);self.assertEqual(c,[31000,11000,0,0]);self.assertEqual(sum(v['status']=='BLOCKED_Z_CALIBRATION' for v in allcells.values()),4)
 def test_resume_does_not_restart_frozen_prefix(self):
  plan=cells(Path(__file__).resolve().parents[3]/'plans/global/2026-09-30-memit-hj-experiment-design-v1/cells.csv')
  with tempfile.TemporaryDirectory() as td:
   e=EngineFixture(td,False);o=Orchestrator(e,plan)
   for c in plan:
    if c['family']!='main' and int(c['anchor_requests'])<5000:e.completed[c['cell_id']]={'status':'COMPLETED'}
   e.meta.update(cursor=5000,ledger=list(range(5000)),cell='main_000',boundary_pending=True,boundary_layers=[],child_fired=False)
   s=e.snapshot();save(e.output/'cells/main_000/lineage.json',dict(prefix_requests=0))
   o.run_main('000',from_snapshot=s,resume=True)
   self.assertTrue(all(n>=5000 for name,n,bs in e.visits if name=='main_000'))
   self.assertEqual(e.count,13000);self.assertEqual(e.writes,850)
if __name__=='__main__':unittest.main()
