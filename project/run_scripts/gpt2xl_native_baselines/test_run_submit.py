"""CPU production transaction/launcher/frontier regression; no native fitting."""
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from .run import NativeTransaction
from .submit import arguments,launcher,resource_frontiers
from .common import TASK

class View:
    def __init__(self):
        self.model=torch.nn.Module();self.model.register_parameter('weight',torch.nn.Parameter(torch.zeros(2,2)))
        self.model.register_parameter('other',torch.nn.Parameter(torch.zeros(1)));self.weights={13:self.model.weight}
    def hook_signature(self):return ()

class Engine:
    def __init__(self,H):self.H=H
    def history(self):return self.H
    def reset_history_to_uninitialized(self):self.H={}

class ProductionTests(unittest.TestCase):
    def test_observer_error_rolls_back_actual_weights_and_existing_native_H(self):
        v=View();e=Engine({l:torch.zeros(2,2) for l in (13,14,15,16,17)});bench=SimpleNamespace(contexts=[['{}']]);tx=NativeTransaction(v,e,bench)
        with self.assertRaisesRegex(ValueError,'observer'):
            with tx:
                with torch.no_grad():v.weights[13].add_(1);e.H[13].add_(2)
                raise ValueError('observer')
        self.assertTrue(tx.rollback_verified);self.assertEqual(float(v.weights[13].detach().sum()),0);self.assertEqual(float(e.H[13].sum()),0)
    def test_first_native_alpha_partial_initialization_rolls_back_to_uninitialized(self):
        v=View();e=Engine({});bench=SimpleNamespace(contexts=[['{}']]);tx=NativeTransaction(v,e,bench)
        with self.assertRaises(ValueError):
            with tx:e.H={13:torch.ones(2,2)};raise ValueError('native')
        self.assertEqual(e.history(),{});self.assertTrue(tx.rollback_verified)
    def test_success_retains_actual_native_history_once_no_caller_append(self):
        v=View();e=Engine({});bench=SimpleNamespace(contexts=[['{}']])
        with NativeTransaction(v,e,bench) as tx:
            with torch.no_grad():v.weights[13].add_(1)
            e.H={13:torch.ones(2,2)};tx.finish()
        self.assertEqual(float(v.weights[13].detach().sum()),4);self.assertEqual(float(e.H[13].sum()),4)
    def test_frontiers_extend_corresponding_ours_lanes(self):
        ours={'jobs':dict(MEMIT_CAP075='1',MEMIT_CAP100='2',MEMIT_FREE100='3',ALPHAEDIT_CAP075='4',ALPHAEDIT_CAP100='5',ALPHAEDIT_FREE100='6')}
        inventory={'jobs':[dict(job=str(i)) for i in range(1,7)]}
        deps,_=resource_frontiers(inventory,ours)
        self.assertEqual(deps,dict(BASE_MEMIT=['3'],BASE_ALPHAEDIT=['6']))
        inventory['jobs'].append(dict(job='9'));deps,_=resource_frontiers(inventory,ours)
        self.assertEqual(deps,dict(BASE_MEMIT=['3','9'],BASE_ALPHAEDIT=['6','9']))
    def test_cancelled_tail_is_not_used_as_resource_barrier(self):
        ours={'jobs':dict(MEMIT_CAP075='1',MEMIT_CAP100='2',MEMIT_FREE100='3',ALPHAEDIT_CAP075='4',ALPHAEDIT_CAP100='5',ALPHAEDIT_FREE100='6')}
        deps,_=resource_frontiers({'jobs':[dict(job='1'),dict(job='2')]},ours)
        self.assertEqual(deps,dict(BASE_MEMIT=['1','2'],BASE_ALPHAEDIT=[]))
    def test_launcher_resources_and_no_checkpoint_or_z_cache_options(self):
        r=dict(cpu=8,host_mib=65536,wall='2-00:00:00',collector_cpu=8,collector_host_mib=24576,collector_wall='02:00:00')
        path=Path('/specific/task/attempt');args=arguments('BASE_MEMIT',['123'],path,r)
        for field in ('--hold','--export=NONE','--no-requeue','--gres=gpu:1','--mem=65536M','--dependency=afterany:123'):self.assertIn(field,args)
        cpu=arguments('collector',['123','124'],path,r);self.assertFalse(any('gres=' in x for x in cpu))
        script=launcher(path/'source','a'*40,'BASE_ALPHAEDIT',path)
        self.assertIn('GPT2_NATIVE_BASELINE_SOURCE_COMMIT',script);self.assertNotIn('WANDB_API_KEY',script)
        self.assertIn('--arm BASE_ALPHAEDIT',script);self.assertIn(TASK,args[args.index('--job-name='+TASK+'-BASE_MEMIT')])
if __name__=='__main__':unittest.main()
