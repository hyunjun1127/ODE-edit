"""CPU-only resume contract and actual pinned native tiny-BF16 parity."""
import contextlib
import copy
import importlib.util
import io
import os
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import Qwen2Config,Qwen2ForCausalLM
from official.diagnostics.fe_qwen_current import author_basic_eot_continue as a
from official.diagnostics.fe_qwen_current.failure_eot_probe import InputAdapter
from official.diagnostics.fe_qwen_current.test_failure_eot_probe import TinyTokenizer
from official.diagnostics.fe_qwen_current.test_failure_key_probe import load_function
from official.diagnostics.fe_native.sequential_z import current_batch_targets
from official.diagnostics.fe_qwen_current.storage import save_latest,tensor_sha,state_digest
from official.experiments.checkpoint import rng_snapshot
from official.runners.server1.common import member
from official.tracking.schema import config as validate_config


class Tests(unittest.TestCase):
    def test_exact_tail_and_coverage(self):
        data=[dict(case_id=i,target_new='X') for i in range(2000)]
        for b in range(16,21):
            rows=a.requests_for(data,b)
            self.assertEqual([r['case_id'] for r in rows],list(range((b-1)*100,b*100)))
            self.assertEqual([r['sample_idx'] for r in rows],list(range(100)))
            self.assertEqual(rows[0]['target_new'],' X')
        self.assertNotIn('sample_idx',data[1500])
        for b in (15,21):
            with self.assertRaises(ValueError):a.requests_for(data,b)
        events=[dict(batch=0,endpoint='W0_first2000',requests=2000)]
        events += [dict(batch=b,endpoint='current/post',requests=100) for b in range(1,21)]
        events += [dict(batch=b,endpoint='all_seen/post',requests=b*100) for b in (5,10,15,20)]
        p=[e for e in events if e['batch']<=15];c=[e for e in events if e['batch']>15]
        a.combined_coverage(p,c)
        bad=copy.deepcopy(p);bad[0]['requests']=500
        with self.assertRaises(ValueError):a.combined_coverage(bad,c)
        with self.assertRaises(ValueError):a.combined_coverage(p,c[:-1]+c[:1])

    def test_tracking_and_unchanged_callbacks(self):
        b=dict(dataset='zsre',execution=dict(server='rent',attempt='resume-r1',source_commit='a'*40,job_id='121400',source_run_id=a.PARENT_WANDB),checkpoint=dict(sha256='b'*64))
        values=a.tracking(b,dict(main_stream=dict(sha256='c'*64)),{},dict(ODEEDIT_RENT_JOB_ID='121400'))
        validate_config(values)
        self.assertEqual(values['source_run_id'],a.PARENT_WANDB)
        self.assertNotIn('generation_schedule',values)
        for name in ('evaluate','targets','write','snapshot'):
            self.assertIs(getattr(a.Continuation,name),getattr(a.author.AuthorObserver,name))
        import inspect
        self.assertIn("self.b['dataset']=='cf'",inspect.getsource(a.Continuation.snapshot))

    def test_low_zsre_metrics_do_not_stop_and_eval_preserves_state(self):
        model=torch.nn.Linear(3,2);H=torch.zeros(1,3,3)
        before=state_digest(rng_snapshot())
        a.parent.preserved(model,H,lambda:torch.rand(7))
        self.assertEqual(before,state_digest(rng_snapshot()));self.assertTrue(model.training)
        with tempfile.TemporaryDirectory() as d:
            obs=a.Continuation.__new__(a.Continuation);obs.out=Path(d);obs.batch=16;obs.b={'dataset':'zsre'}
            obs.names=set();obs.nonedited={n:(id(p),p._version) for n,p in model.named_parameters()}
            obs.stream=list(range(2000));obs.raw=[];obs.identity={'test':True}
            def evaluate(*args):
                obs.raw.append({'summary':{'Efficacy':0.,'Generalization':0.,'Specificity':0.}})
                return {'test_raw':True}
            obs.evaluate=evaluate
            with patch.object(a.author.common,'quality_stop',side_effect=AssertionError('CF_GATE_ON_ZSRE')):
                obs.snapshot(None,model,None,H,[],16,[])
            self.assertTrue((obs.out/'commits/B16.json').exists())
            self.assertFalse((obs.out/'QUALITY_EARLY_STOP.json').exists())

    def test_native_b15_private_resume_tail(self):
        root=Path(os.environ.get('FE_AUTHOR_ROOT','/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author'))
        if not root.exists():self.skipTest('native author missing')
        spec=importlib.util.spec_from_file_location('author_resume_nethook',root/'util/nethook.py');nh=importlib.util.module_from_spec(spec);spec.loader.exec_module(nh)
        tok=TinyTokenizer();ns=load_function(root/'locate_edit_utils/repr_tools.py',{'get_words_idxs_in_templates'},dict(torch=torch))
        lookup=load_function(root/'algs/memit/compute_z.py',{'find_fact_lookup_idx'},ns)['find_fact_lookup_idx']
        fit=load_function(root/'algs/memit/compute_z.py',{'compute_z'},dict(torch=torch,np=np,nethook=nh,find_fact_lookup_idx=lookup))['compute_z']
        seed=load_function(root/'algs/memit/FE-memit_main.py',{'set_random_seed'},dict(torch=torch,np=np,random=random,os=os))['set_random_seed']
        torch.set_num_threads(1);torch.manual_seed(51)
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=8,intermediate_size=16,num_hidden_layers=28,num_attention_heads=2,num_key_value_heads=2,use_cache=False)).to(torch.bfloat16).eval()
        resumed=copy.deepcopy(model)
        # Tiny fixture dimensions/optimizer iterations only; production all-field
        # YAML checks are exercised separately by test_author_basic_eot_benchmark.
        cfg=OmegaConf.create(dict(seed=0,gpu=0,lti=False,negetive_prompt_test=False,bs=100,num_edits=2000,algs=dict(L2=0,add_old_keys=True),llms=dict(
            layers=[4,5,6,7,8],mom2_update_weight=[15000]*5,lm_head_module='lm_head',ln_f_module='model.norm',midlayers=[15],layer_module_tmp='model.layers.{}',rewrite_module_tmp='model.layers.{}.mlp.down_proj',
            constr_pos='subject',fact_token='subject_last',v_loss_layer=27,v_lr=.01,v_num_grad_steps=2,kl_factor=.0625,v_weight_decay=.001,clamp_norm_factor=1)))
        contexts=[['{}'],['1 {}','3 {}','4 {}','5 {}','6 {}']];adapter=InputAdapter(tok,lookup)
        def extraction(m,t,l,**kw):
            key=torch.arange(1,17,dtype=torch.float32).repeat(len(kw['words']),1)/16
            W=dict(m.named_parameters())[cfg.llms.rewrite_module_tmp.format(l)+'.weight']
            return key,key@W.float().T
        key=load_function(root/'algs/memit/compute_ks.py',{'compute_ks'},dict(torch=torch,np=np,get_module_input_output_at_words=extraction))['compute_ks']
        native=load_function(root/'algs/memit/FE-memit_main.py',{'batch_edit','upd_matrix_match_shape'},dict(torch=torch,time=time,covs=[torch.eye(16)]*5,nethook=nh,
            get_context_templates=lambda *args:contexts,compute_ks=key,get_module_input_output_at_words=extraction))['batch_edit']
        writer=adapter.writer(native,key);fit=adapter.fit(fit);supplier=adapter.supplier(current_batch_targets);H=torch.zeros(5,16,16)
        names={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers}
        def step(m,h,b):
            req=[dict(case_id=b,prompt='{} is a',subject='2' if b%2 else '2 3',target_new=' 4 5',sample_idx=0)]
            target=supplier(m,tok,req,cfg,contexts,compute_z=fit,find_fact_lookup_idx=adapter.lookup,TraceDict=nh.TraceDict)
            writer(cfg,m,tok,req,'cpu',h,target)
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            out=Path(d)
            for b in range(1,21):
                step(model,H,b)
                if b==15:
                    cp=save_latest(out/'cp',{n:p for n,p in model.named_parameters() if n in names},H,contexts,{},15,{},dict(native_config=OmegaConf.to_container(cfg)),lock_path=out/'lock',reserve_bytes=0)
            expected_rng=state_digest(rng_snapshot());before=member(cp['path'])
            payload=torch.load(cp['path'],weights_only=False,mmap=True)
            H2,ctx,receipt=a.restore(resumed,payload,N(set_random_seed=seed,CONTEXT_TEMPLATES_CACHE=None))
            self.assertEqual(ctx,contexts)
            for b in range(16,21):step(resumed,H2,b)
            self.assertTrue(torch.equal(H,H2));self.assertEqual(expected_rng,state_digest(rng_snapshot()))
            for (n,p),(n2,p2) in zip(model.named_parameters(),resumed.named_parameters()):
                self.assertEqual(n,n2);self.assertTrue(torch.equal(p,p2),n)
            self.assertEqual(member(cp['path']),before)
            fresh=torch.load(cp['path'],weights_only=False,mmap=True)
            self.assertEqual(tensor_sha(fresh['cache_c']),receipt['history'])


if __name__=='__main__':unittest.main()
