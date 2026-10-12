"""Tiny native writer/evaluation/storage tests, no pretrained/GPU evidence."""
import contextlib
import copy
import io
import os
from pathlib import Path
from types import SimpleNamespace as N
import tempfile
import time
import unittest
import numpy as np
import torch
from official.diagnostics.fe_qwen_current.failure_eot_prefix import all_write,preserved,tracking_values
from official.diagnostics.fe_qwen_current.failure_eot_probe import InputAdapter
from official.diagnostics.fe_qwen_current.test_failure_eot_probe import TinyTokenizer
from official.diagnostics.fe_qwen_current.test_failure_key_probe import load_function
from official.diagnostics.fe_qwen_current.storage import save_latest,state_digest
from official.experiments.checkpoint import rng_snapshot


class Tests(unittest.TestCase):
    def test_tracking_actual_rent_identity_and_diagnostic_group(self):
        from official.tracking.schema import bind_job_identity,metrics
        b=dict(execution=dict(server='rent',attempt='eot-prefix-r1',source_commit='a'*40,job_id='121200'))
        values=tracking_values(b,dict(stream=dict(sha256='b'*64)))
        bound=bind_job_identity(values,{'ODEEDIT_RENT_JOB_ID':'121200'})
        self.assertEqual(bound['job_id'],b['execution']['job_id'])
        self.assertEqual(bound['task_id'],'fe-qwen-basic-eot-prefix-20261012')
        self.assertEqual(bound['evaluation_profile'],'FE_PLUS_OURS_BASIC_POSITION0_EOT_INPUT_PREPROCESSING')
        metrics(dict(edits=500,batch=5,post_state_edits=500,**{'official/all_seen/post/requests':500,'official/all_seen/post/Efficacy':95.}),scientific=True,config_values=bound)

    def test_native_all_request_writer_and_unaffected_parity(self):
        root=Path(os.environ.get('FE_AUTHOR_ROOT','/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author'))
        if not root.exists():self.skipTest('FE author unavailable')
        ns=load_function(root/'locate_edit_utils/repr_tools.py',{'get_words_idxs_in_templates'},dict(torch=torch))
        lookup=load_function(root/'algs/memit/compute_z.py',{'find_fact_lookup_idx'},ns)['find_fact_lookup_idx']
        tok=TinyTokenizer();adapter=InputAdapter(tok,lookup);calls=[];contexts=[['{}'],['1 {}','3 {}','4 {}','5 {}','6 {}']]
        def extracted(m,t,l,**kw):
            calls.append(kw['context_templates']);n=len(kw['words'])
            return torch.arange(n*4,dtype=torch.float32).reshape(n,4).add(1),torch.zeros(n,3)
        key=load_function(root/'algs/memit/compute_ks.py',{'compute_ks'},dict(torch=torch,np=np,get_module_input_output_at_words=extracted))['compute_ks']
        env=load_function(root/'algs/memit/FE-memit_main.py',{'batch_edit','upd_matrix_match_shape'},dict(torch=torch,time=time,covs=[torch.eye(4)]*2,
            nethook=N(get_parameter=lambda m,n:dict(m.named_parameters())[n]),get_context_templates=lambda *a:contexts,compute_ks=key,get_module_input_output_at_words=extracted))
        native=env['batch_edit'];changed=adapter.writer(native,key)
        cfg=N(llms=N(layers=[4,5],rewrite_module_tmp='layers.{}',layer_module_tmp='layers.{}',fact_token='subject_last',mom2_update_weight=[15000]*2),algs=N(L2=10,add_old_keys=True),negetive_prompt_test=False)
        torch.manual_seed(9);a=torch.nn.Module();a.layers=torch.nn.ModuleDict({str(l):torch.nn.Linear(4,3,bias=False,dtype=torch.bfloat16) for l in (4,5)})
        b=copy.deepcopy(a);ha=torch.zeros(2,4,4);hb=ha.clone();rng=torch.get_rng_state().clone()
        req=[dict(case_id=i,sample_idx=i,prompt='{} is a',subject='2 3') for i in range(2)];targets={l:torch.ones(3,2)*3 for l in (4,5)}
        with contextlib.redirect_stdout(io.StringIO()):
            native(cfg,a,tok,req,'cpu',ha,targets)
            result=all_write(changed,cfg,b,tok,req,'cpu',hb,targets,1,channels=(0,3))
            self.assertTrue(torch.equal(ha,hb));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(x,y))
            calls.clear();req[0]['subject']='2'
            result=all_write(changed,cfg,b,tok,req,'cpu',hb,targets,2,channels=(0,3))
        self.assertEqual(result['case_ids'],[0,1]);self.assertEqual(result['masked_requests'],0)
        for row in result['layers']:self.assertTrue(all(k>0 for k in row['key_norms']))
        self.assertTrue(any(p[0].startswith('<|endoftext|>') for p in calls))
        self.assertIs(env['compute_ks'],key)

    def test_evaluation_rng_modes_hooks_and_weights(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.Dropout(.2));model.train();model[0].eval();H=torch.zeros(2,3,3)
        before=state_digest(rng_snapshot());modes=[m.training for m in model.modules()];versions=[p._version for p in model.parameters()]
        def evaluate():
            self.assertTrue(all(not m.training for m in model.modules()));torch.rand(7);return {'requests':100}
        self.assertEqual(preserved(model,H,evaluate),{'requests':100})
        self.assertEqual(before,state_digest(rng_snapshot()));self.assertEqual(modes,[m.training for m in model.modules()]);self.assertEqual(versions,[p._version for p in model.parameters()])
        def bad():
            with torch.no_grad():model[0].weight.add_(1)
        with self.assertRaisesRegex(ValueError,'EVALUATION_MUTATED_EDITING_STATE'):preserved(model,H,bad)

    def test_one_b5_checkpoint_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);W={'w':torch.ones(3,4,dtype=torch.bfloat16)};H=torch.ones(2,4,4);rng=state_digest(rng_snapshot())
            receipt=save_latest(path/'cp',W,H,[['{}']],{'policy':'EOT'},5,{'input_policy':'OURS_BASIC_SUBJECT_LAST_ZERO_EOT'},
                {'edits':500},lock_path=path/'lock',reserve_bytes=0)
            payload=torch.load(receipt['path'],weights_only=False,mmap=True);self.assertTrue(torch.equal(payload['cache_c'],H));self.assertEqual(payload['batch'],5)
            self.assertFalse(receipt['final_W20']);self.assertEqual(rng,state_digest(payload['rng']))
            self.assertEqual(sorted(p.name for p in (path/'cp').iterdir()),['latest.json','latest.pt'])


if __name__=='__main__':unittest.main()
