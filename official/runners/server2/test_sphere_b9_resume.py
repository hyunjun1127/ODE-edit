"""CPU-only narrow ancestor restore/continuity controls, no model qualification."""
import ast
import copy
import inspect
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from official.experiments import checkpoint
from official.experiments.prepare import digest,write_new
from official.runners.server2 import sphere_b9_resume as r
from official.runners.server4.qwen_native_state import NativeState

class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.records=[dict(occurrence_index=n,case_id=n+20) for n in range(1,2001)]
        self.identity={k:'fixture-'+k for k in checkpoint.IDENTITY_FIELDS}
        self.identity.update(code_commit=r.SOURCE,config_sha256=r.CONFIG)
        self.contexts=[['{}'],['A {}']]
        self.binding=dict(parent_identity=self.identity,context_sha256=digest(self.contexts),binding_sha256='fixture')
        self.payload=dict(schema='official-baseline-checkpoint-v1',batch=9,method='SPHERE',identity=self.identity,
            weights={'model.layers.4.mlp.down_proj.weight':torch.arange(6,dtype=torch.float32).reshape(2,3)},
            cache_c={'4':torch.eye(3)*9},contexts=self.contexts,rng=checkpoint.rng_snapshot(),
            evaluation_cursor=dict(completed_batch=9,evaluated_endpoints=[0,5],per_request_records=self.records[:900]))
        self.payload['rng']['torch_cuda']=[torch.zeros(12,dtype=torch.uint8)]
    def validate(self,p=None):
        return r.validate_payload(p or self.payload,self.binding,self.records,width=3,hidden=2,layers=[4])
    def test_exact_boundary_and_prefix(self):
        p=self.validate();self.assertEqual(p['next_batch'],10);self.assertEqual(p['remaining_batches'],list(range(10,21)))
        self.assertEqual(p['remaining_requests'],1100)
    def test_tampered_payload_rejected(self):
        for mutate in [lambda p:p.update(batch=10),lambda p:p['identity'].update(code_commit='relabel'),
                       lambda p:p['cache_c']['4'].fill_(float('nan')),
                       lambda p:p['evaluation_cursor']['per_request_records'].reverse(),
                       lambda p:p.update(contexts=[['changed']])]:
            p=copy.deepcopy(self.payload);mutate(p)
            with self.assertRaises(ValueError):self.validate(p)
    def test_actual_native_restore_not_apply(self):
        native=NativeState.__new__(NativeState)
        native.method='SPHERE';native.layers=[4];native._width=3
        native._names=list(self.payload['weights'])
        weight=torch.nn.Parameter(torch.zeros(2,3))
        native.model=SimpleNamespace(get_parameter=lambda name:weight)
        native.module=SimpleNamespace(CONTEXT_TEMPLATES_CACHE=None,cache_c=None)
        native.native_apply=lambda *a,**k: self.fail('restore must not fit or append')
        p=copy.deepcopy(self.payload);p['rng']['torch_cuda']=None
        saved=checkpoint.rng_snapshot()
        try:
            cursor=native.restore_from_checkpoint(p)
            self.assertEqual(cursor['completed_batch'],9)
            self.assertTrue(torch.equal(weight,p['weights'][native._names[0]]))
            self.assertTrue(r.equal_state(native.cache_for_checkpoint(),p['cache_c']))
            self.assertEqual(native.context_snapshot(),p['contexts'])
            self.assertTrue(r.equal_state(checkpoint.rng_snapshot(),p['rng']))
            self.assertEqual(float(native.cache_for_checkpoint()['4'].trace()),27.)
        finally:checkpoint.rng_restore(saved)
    def test_child_bootstrap_then_contiguous_stock_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent=root/'parent.json';write_new(parent,{'fixture':True})
            identity=dict(self.identity,code_commit='new-child-source')
            cursor=dict(completed_batch=10,resume_parent_binding_sha256='fixture',per_request_records=self.records[:1000])
            kwargs=dict(batch=10,weights=self.payload['weights'],cache_c=self.payload['cache_c'],
                        contexts=self.contexts,evaluation_cursor=cursor,identity=identity,method='SPHERE',evaluation_complete=True)
            with patch.object(r,'validate_binding',return_value=self.binding):
                ck=r.save_descendant(root/'child',parent_binding=parent,**kwargs)
                self.assertEqual(ck['batch'],10)
                loaded=checkpoint.load(root/'child',identity);self.assertEqual(loaded['resume_ancestor']['path'],str(parent))
                kwargs.update(batch=11,evaluation_cursor=dict(cursor,completed_batch=11,per_request_records=self.records[:1100]))
                self.assertEqual(r.save_descendant(root/'child',parent_binding=parent,**kwargs)['batch'],11)
                kwargs['batch']=13
                with self.assertRaisesRegex(ValueError,'NONCONTIGUOUS'):r.save_descendant(root/'child',parent_binding=parent,**kwargs)
                self.assertEqual(r.read(parent),{'fixture':True})
    def test_wrong_first_child_and_parent_digest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'bad.json'
            write_new(p,dict(instruction_id=r.INSTRUCTION,parent_job_id='62087',binding_sha256='bad'))
            with self.assertRaisesRegex(ValueError,'RESUME_BINDING_DIGEST'):r.validate_binding(p)
            with patch.object(r,'validate_binding',return_value=self.binding):
                with self.assertRaisesRegex(ValueError,'FIRST_CHILD'):
                    r.save_descendant(root/'child',parent_binding=p,batch=9,method='SPHERE',evaluation_complete=True)
    def test_actual_caller_branch_does_not_write_w0_or_replay_ancestors(self):
        from official.runners.server2.qwen_run import execute
        source=inspect.getsource(execute);tree=ast.parse(source)
        self.assertLess(source.index('out.exists()'),source.index('historical-w0-consumer-binding.json'))
        branches=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='resume_parent is not None']
        restore=next(n for n in branches if any(isinstance(x,ast.Call) and ast.unparse(x.func)=='restore_parent' for a in n.body for x in ast.walk(a)))
        calls=[ast.unparse(x.func) for a in restore.body for x in ast.walk(a) if isinstance(x,ast.Call)]
        self.assertNotIn('checkpoint.save',calls);self.assertNotIn('native.apply',calls)
        self.assertIn('range(start + 1, max_batch + 1)',source)
        self.assertIn('start!=9',source)
    def test_resumed_tracker_uses_new_identity_and_real_parent_hash(self):
        from official.runners.server2.qwen_run import _tracker
        from official.tracking.schema import config
        with patch.dict(os.environ,ODEEDIT_WANDB_ENV_FILE='/fixture/env',ODEEDIT_ATTEMPT_ID='resume-fixture'),patch('official.tracking.init') as init:
            _tracker('/fixture/out',arm='qwen25-zsre-sphere-resume-b9',writer='SPHERE',dataset='zsre',
                     source_sha='a'*40,config_sha=r.CONFIG,assets={},
                     resume_binding={'parent_checkpoint':{'sha256':r.CP_SHA}})
            values=init.call_args.kwargs['config'];config(values)
            self.assertEqual(values['checkpoint_sha256'],r.CP_SHA)
            self.assertEqual(values['source_run_id'],'job62087-b9')
            self.assertNotEqual(values['source_sha'],r.SOURCE)
            self.assertFalse(any(k.startswith('generation') for k in values))

if __name__=='__main__':unittest.main()
