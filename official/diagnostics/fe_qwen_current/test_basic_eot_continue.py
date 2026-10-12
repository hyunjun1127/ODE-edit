"""Tiny actual native fit/supplier/writer resume parity; no pretrained/GPU."""
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
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import Qwen2Config,Qwen2ForCausalLM
from official.diagnostics.fe_qwen_current.basic_eot_continue import requests_for,restore,validate_checkpoint
from official.diagnostics.fe_qwen_current.failure_eot_probe import InputAdapter
from official.diagnostics.fe_qwen_current.test_failure_eot_probe import TinyTokenizer
from official.diagnostics.fe_qwen_current.test_failure_key_probe import load_function
from official.diagnostics.fe_native.sequential_z import current_batch_targets
from official.diagnostics.fe_qwen_current.storage import save_latest,tensor_sha,state_digest
from official.diagnostics.fe_qwen_current import failure_eot_probe as eot
from official.runners.server1.common import member
from official.experiments.checkpoint import rng_snapshot
from official.diagnostics.fe_qwen_current.failure_eot_prefix import preserved


class Tests(unittest.TestCase):
    def test_future_slices_and_labels(self):
        rows=[dict(case_id=i,target_new=' X') for i in range(2000)]
        for b in (6,10,15,20):
            part=requests_for(rows,b);self.assertEqual([x['case_id'] for x in part],list(range((b-1)*100,b*100)))
            self.assertEqual([x['sample_idx'] for x in part],list(range(100)));self.assertEqual(part[0]['target_new'],'  X')
        self.assertNotIn('sample_idx',rows[500])
        with self.assertRaises(ValueError):requests_for(rows,5)

    def test_actual_native_targets_writer_b5_resume(self):
        root=Path(os.environ.get('FE_AUTHOR_ROOT','/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author'))
        if not root.exists():self.skipTest('native author missing')
        spec=importlib.util.spec_from_file_location('continue_nethook',root/'util/nethook.py');nh=importlib.util.module_from_spec(spec);spec.loader.exec_module(nh)
        tok=TinyTokenizer();ns=load_function(root/'locate_edit_utils/repr_tools.py',{'get_words_idxs_in_templates'},dict(torch=torch))
        lookup=load_function(root/'algs/memit/compute_z.py',{'find_fact_lookup_idx'},ns)['find_fact_lookup_idx']
        fit=load_function(root/'algs/memit/compute_z.py',{'compute_z'},dict(torch=torch,np=np,nethook=nh,find_fact_lookup_idx=lookup))['compute_z']
        seed=load_function(root/'algs/memit/FE-memit_main.py',{'set_random_seed'},dict(torch=torch,np=np,random=random,os=os))['set_random_seed']
        torch.set_num_threads(1);torch.manual_seed(51)
        base=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=8,intermediate_size=16,num_hidden_layers=28,num_attention_heads=2,num_key_value_heads=2,use_cache=False)).to(torch.bfloat16).eval()
        model=copy.deepcopy(base);resumed=copy.deepcopy(base)
        cfg=OmegaConf.create(dict(seed=0,gpu=0,lti=False,negetive_prompt_test=False,bs=100,num_edits=2000,algs=dict(L2=10,add_old_keys=True),llms=dict(
            layers=[4,5,6,7,8],mom2_update_weight=[15000]*5,lm_head_module='lm_head',ln_f_module='model.norm',midlayers=[15],layer_module_tmp='model.layers.{}',rewrite_module_tmp='model.layers.{}.mlp.down_proj',
            constr_pos='subject',fact_token='subject_last',v_loss_layer=27,v_lr=.01,v_num_grad_steps=2,kl_factor=.0625,v_weight_decay=.001,clamp_norm_factor=1)))
        contexts=[['{}'],['1 {}','3 {}','4 {}','5 {}','6 {}']];adapter=InputAdapter(tok,lookup)
        # Actual writer AST with deterministic tiny representation fixture;
        # actual native compute_z and replay run the tiny Qwen model.
        def extraction(m,t,l,**kw):
            n=len(kw['words']);key=torch.arange(1,17,dtype=torch.float32).repeat(n,1)/16
            W=dict(m.named_parameters())[cfg.llms.rewrite_module_tmp.format(l)+'.weight']
            return key,key@W.float().T
        key=load_function(root/'algs/memit/compute_ks.py',{'compute_ks'},dict(torch=torch,np=np,get_module_input_output_at_words=extraction))['compute_ks']
        native=load_function(root/'algs/memit/FE-memit_main.py',{'batch_edit','upd_matrix_match_shape'},dict(torch=torch,time=time,covs=[torch.eye(16)]*5,nethook=nh,
            get_context_templates=lambda *a:contexts,compute_ks=key,get_module_input_output_at_words=extraction))['batch_edit']
        writer=adapter.writer(native,key);fit=adapter.fit(fit);supplier=adapter.supplier(current_batch_targets);H=torch.zeros(5,16,16)
        names={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers}
        def step(m,h,b):
            req=[dict(case_id=b,prompt='{} is a',subject='2' if b%2 else '2 3',target_new=' 4 5',sample_idx=0)]
            target=supplier(m,tok,req,cfg,contexts,compute_z=fit,find_fact_lookup_idx=adapter.lookup,TraceDict=nh.TraceDict)
            writer(cfg,m,tok,req,'cpu',h,target)
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            rootout=Path(d);pointer=None
            for b in range(1,8):
                step(model,H,b)
                if b==5:
                    ident=dict(input_policy='OURS_BASIC_SUBJECT_LAST_ZERO_EOT',L2=10,clamp_norm_factor=1.,no_mask=True)
                    cursor=dict(edits=500,case_ids=list(range(500)),native_config=OmegaConf.to_container(cfg),input_adapter=member(eot.__file__))
                    pointer=save_latest(rootout/'cp',{n:p for n,p in model.named_parameters() if n in names},H,contexts,{},5,ident,cursor,lock_path=rootout/'lock',reserve_bytes=0)
            expected_rng=state_digest(rng_snapshot());payload=torch.load(pointer['path'],weights_only=False,mmap=True)
            proof=dict(identity=pointer['identity'],adapter=member(eot.__file__));stream=[dict(case_id=i) for i in range(2000)]
            validate_checkpoint(payload,pointer,proof,OmegaConf.to_container(cfg),stream)
            H2,ctx,receipt=restore(resumed,payload,N(set_random_seed=seed,CONTEXT_TEMPLATES_CACHE=None))
            for b in (6,7):step(resumed,H2,b)
            self.assertTrue(torch.equal(H,H2));self.assertEqual(expected_rng,state_digest(rng_snapshot()))
            for (n,a),(m,b) in zip(model.named_parameters(),resumed.named_parameters()):self.assertEqual(n,m);self.assertTrue(torch.equal(a,b),n)
            # MAP_PRIVATE updates do not mutate checkpoint's saved H.
            fresh=torch.load(pointer['path'],weights_only=False,mmap=True)
            self.assertEqual(tensor_sha(fresh['cache_c']),receipt['history'])
            bad=copy.copy(payload);bad['cache_c']=payload['cache_c'].double()
            with self.assertRaises(ValueError):validate_checkpoint(bad,pointer,proof,OmegaConf.to_container(cfg),stream)
            wrong=copy.deepcopy(proof);wrong['adapter']['sha256']='0'*64
            with self.assertRaises(ValueError):validate_checkpoint(fresh,pointer,wrong,OmegaConf.to_container(cfg),stream)

    def test_evaluation_preserved(self):
        m=torch.nn.Linear(3,2);H=torch.zeros(2,3,3);before=state_digest(rng_snapshot())
        preserved(m,H,lambda:torch.rand(5));self.assertEqual(before,state_digest(rng_snapshot()));self.assertTrue(m.training)


if __name__=='__main__':unittest.main()
