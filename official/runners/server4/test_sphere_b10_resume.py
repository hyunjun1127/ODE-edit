import copy
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from official.experiments import checkpoint as cp
from official.experiments.prepare import digest
from official.runners.server4 import sphere_b10_resume as r
from official.tracking.schema import config, official_zsre_metrics


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.identity={k:'fixture-'+k for k in cp.IDENTITY_FIELDS}
        self.records=[dict(case_id=n,occurrence_index=n+1) for n in range(2000)]
        self.context=[['{}']];pending={'checkpoint_identity':self.identity,'batch':10}
        self.binding=dict(parent_identity=self.identity,context_sha256=digest(self.context),parent_pending=pending,b9_binding_sha256='b9')
        self.payload=dict(schema='official-baseline-checkpoint-v1',method='SPHERE',batch=10,identity=self.identity,
            weights={'model.layers.4.mlp.down_proj.weight':torch.arange(6,dtype=torch.float32).reshape(2,3)},
            cache_c={'4':torch.eye(3)},contexts=self.context,rng=cp.rng_snapshot(),evaluation_cursor=dict(completed_batch=10,
            evaluated_endpoints=[0,5,10],per_request_records=self.records[:1000],pending_receipt_sha256=digest(pending),
            ancestor_prefix_edits=900,resume_parent_binding_sha256='b9'))
        self.payload['rng']['torch_cuda']=[torch.zeros(4,dtype=torch.uint8)]
    def validate(self,payload):r.validate_parent(payload,self.binding,self.records,layers=[4],hidden=2,width=3)
    def test_b10_and_ancestry_validation(self):
        self.validate(self.payload)
        mutations=[lambda p:p.update(batch=9),lambda p:p['identity'].update(code_commit='different'),
            lambda p:p['evaluation_cursor'].update(ancestor_prefix_edits=1000),
            lambda p:p['evaluation_cursor']['per_request_records'].reverse(),
            lambda p:p['cache_c']['4'].fill_(float('nan')),lambda p:p.update(contexts=[['bad']])]
        for change in mutations:
            p=copy.deepcopy(self.payload);change(p)
            with self.assertRaises(ValueError):self.validate(p)
    def slots(self,root):
        for i in (0,1):
            with (root/f'slot-{i}.pt').open('xb') as f:os.posix_fallocate(f.fileno(),0,256*1024)
    def kwargs(self,batch):
        return dict(batch=batch,identity=self.identity,weights=self.payload['weights'],cache_c=self.payload['cache_c'],
            contexts=self.context,cursor={'completed_batch':batch},ancestor={'sha256':'parent-B10'})
    def test_two_slots_keep_standard_checkpoint_compatible_and_contiguous(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.slots(root)
            for n in range(11,21):
                self.payload['weights']['model.layers.4.mlp.down_proj.weight'].add_(1)
                ref=r.save_reserved(root,**self.kwargs(n));saved=cp.load(root,self.identity)
                self.assertEqual(saved['batch'],n);self.assertEqual(ref['final_W20'],n==20)
                self.assertTrue(torch.equal(saved['weights']['model.layers.4.mlp.down_proj.weight'],self.payload['weights']['model.layers.4.mlp.down_proj.weight']))
            self.assertEqual(len(list(root.glob('*.pt'))),2)
            with self.assertRaisesRegex(ValueError,'CHILD_BATCH_RANGE'):r.save_reserved(root,**self.kwargs(21))
            with self.assertRaisesRegex(ValueError,'IDENTITY_MISMATCH'):cp.load(root,dict(self.identity,code_commit='wrong'))
    def test_failed_inactive_write_or_pointer_publication_preserves_active(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.slots(root);r.save_reserved(root,**self.kwargs(11))
            old=(root/'latest.json').read_bytes()
            def bad_write(payload,stream):stream.write(b'partial');raise OSError('disk full')
            with patch.object(r.torch,'save',side_effect=bad_write):
                with self.assertRaises(OSError):r.save_reserved(root,**self.kwargs(12))
            self.assertEqual(cp.load(root,self.identity)['batch'],11)
            with patch.object(r.os,'replace',side_effect=OSError('pointer failure')):
                with self.assertRaises(OSError):r.save_reserved(root,**self.kwargs(12))
            self.assertEqual((root/'latest.json').read_bytes(),old)
            self.assertEqual(cp.load(root,self.identity)['batch'],11)
            r.save_reserved(root,**self.kwargs(12));self.assertEqual(cp.load(root,self.identity)['batch'],12)
    def test_saved_rng_history_and_weights_resume_exactly(self):
        torch.manual_seed(3);random.seed(3);np.random.seed(3)
        def step(W,H):
            W['W'].add_(torch.rand(2,3)+random.random()+np.random.random());H['4'].add_(W['W'].T@W['W'])
        W={'W':torch.zeros(2,3)};H={'4':torch.zeros(3,3)}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.slots(root)
            for b in range(11,15):
                step(W,H);r.save_reserved(root,batch=b,identity=self.identity,weights=W,cache_c=H,contexts=[],cursor={},ancestor={})
            saved=cp.load(root,self.identity)
            for _ in range(6):step(W,H)
            cp.rng_restore(saved['rng'])
            for _ in range(6):step(saved['weights'],saved['cache_c'])
            self.assertTrue(torch.equal(W['W'],saved['weights']['W']));self.assertTrue(torch.equal(H['4'],saved['cache_c']['4']))
    def test_first_child_requires_11_and_preallocated_slot(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaisesRegex(ValueError,'FIRST_CHILD'):r.save_reserved(root,**self.kwargs(12))
            with self.assertRaisesRegex(ValueError,'PREALLOCATED'):r.save_reserved(root,**self.kwargs(11))
            self.slots(root);r.save_reserved(root,**self.kwargs(11))
            with self.assertRaisesRegex(ValueError,'NONCONTIGUOUS'):r.save_reserved(root,**self.kwargs(13))
    def test_server4_registration_preserves_two_lanes(self):
        from official.runners.server4.sphere_b10_submit import parents
        from official.runners.server1.submit import graph_width
        for n in (0,1,2,3):
            jobs=[dict(key=str(i),gpus=1,parents=[]) for i in range(n)]
            existing=dict(jobs=jobs,admitted_DAG_width=n)
            before=copy.deepcopy(existing);deps,width=parents(existing)
            self.assertEqual(existing,before)
            self.assertLessEqual(width,max(2,n))
            self.assertEqual(graph_width(jobs+[dict(key='new',gpus=1,parents=deps)]),width)
            if n>2:self.assertEqual(set(deps),{str(i) for i in range(n)})

    def test_child_tracking_has_server4_parent_b10_identity(self):
        c=dict(server='server4',task_id=r.TASK,arm='qwen25-zsre-sphere-resume-b10',attempt='r1',source_sha='a'*40,
            config_sha='b'*64,model='qwen25',model_family='qwen2',writer='SPHERE',baseline='SPHERE',role='scientific',
            metric_schema='official-baselines-scalar-v1',instruction_id='USER-OFFICIAL-BASELINES-20261008-R1',dataset='zsre',
            checkpoint_sha256=r.PARENT_SHA,source_run_id='job62538-b10')
        config(c);m=official_zsre_metrics(dict(Efficacy=90.,Generalization=85.,Specificity=30.,Specificity_loc_ans=30.,requests=2000),
            config_values=c,endpoint='all_seen/post',edits=2000,post_state_edits=2000)
        self.assertEqual(m['zsre/all_seen/post/Specificity'],30.)


if __name__=='__main__':unittest.main()
