import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .common import *
from .metrics import success,reduce_eval
from .details import transitions
from .plots import generate

class Focused(unittest.TestCase):
 def test_direction_tie(self):
  self.assertTrue(success(1,2,'RS'));self.assertTrue(success(1,2,'PS'));self.assertFalse(success(1,2,'NS'))
  for t in ['RS','PS','NS']:self.assertFalse(success(1,1,t))
 def test_quantiles(self):
  s=stats([1,2,3,10]);self.assertEqual(s['median'],2.5);self.assertAlmostEqual(s['p90'],7.9)
 def test_prompt_strict_cardinality(self):
  rows=[dict(case_id=1,prompt_index=i,identity=str(i),new_nll=a,true_nll=2,success=a<2,new_strict=a<2,true_strict=False,new_token_correct=int(a<2),new_token_count=1,true_token_correct=0,true_token_count=1) for i,a in enumerate([1,3])]
  d=reduce_eval(dict(metrics={'PS':dict(rows=rows)}),'toy',1,'toy',[1]);self.assertEqual(d['PS_num'],1);self.assertEqual(d['PS_strict_num'],0)
  with self.assertRaises(AssertionError):reduce_eval(dict(metrics={'PS':dict(rows=rows[::-1])}),'toy',1,'toy',[1])
 def test_nonfinite(self):
  with self.assertRaises(AssertionError):stats([float('nan')])
 def test_transition_conservation(self):
  a=[dict(identity=str(i),case_id=i,prompt_index=0,success=b) for i,b in enumerate([True,True,False])]
  b=[dict(r,success=v) for r,v in zip(a,[True,False,True])]
  t=transitions(a,b,'A','B','NS',10);self.assertEqual(t['success_to_loss'],1);self.assertEqual(t['failure_to_recovery'],1)
 def test_rehash_detects_change(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';p.write_text('sealed');m=dict(path='x',sha256=sha(p),bytes=p.stat().st_size)
   self.assertEqual(len(verify_members(d,[m],'toy')),1);p.write_text('changed')
   with self.assertRaises(AssertionError):verify_members(d,[m],'toy')
 def test_create_once(self):
  with tempfile.TemporaryDirectory() as d:
   save(Path(d)/'r.json',{'a':1})
   with self.assertRaises(FileExistsError):save(Path(d)/'r.json',{'a':2})
 def test_synthetic_plots_order_title_no_reference(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);rows=[]
   for a in ARMS:
    r=dict(arm=a,batch=1,cohort=1,current_success=99,canonical_denominator=100,process_seconds=100,target_seconds=50,layer=4,batch_net_norm=1)
    for t in ['RS','PS','NS']:r[t+'_rate']=.9
    for c in ['rewrite','rephrase','locality']:
     for k in ['mean','median','p90','max']:
      r[c+'_margin_'+k]=1
      for s in ['new','true']:r[c+'_target_'+s+'_nll_'+k]=1
    rows.append(r)
   for n in ['final_metrics','current_batch','seen_prefix','allseen_rewrite','retention_cohort','layer_action','compute']:csvwrite(p/(n+'.csv'),rows)
   from matplotlib.figure import Figure
   original=Figure.savefig;captures=[]
   def capture(fig,*args,**kwargs):
    if fig._suptitle is not None and fig._suptitle.get_text()=='Layer-wise Update Magnitude':
     captures.append([ax.get_title() for ax in fig.axes]);self.assertEqual(sum(len(ax.lines) for ax in fig.axes),0)
    return original(fig,*args,**kwargs)
   with patch.object(Figure,'savefig',capture):ledger=generate(p,p/'png')
   self.assertEqual(captures,[ARMS]);self.assertEqual(len(ledger),9);self.assertTrue(all((p/'png'/x['path']).is_file() for x in ledger))

if __name__=='__main__':unittest.main()
