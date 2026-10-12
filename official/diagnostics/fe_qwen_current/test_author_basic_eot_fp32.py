"""CPU fixtures only; no pretrained/GPU claim."""
import copy
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import Qwen2Config,Qwen2ForCausalLM
from official.diagnostics.fe_qwen_current import author_basic_eot_fp32 as a
from official.diagnostics.fe_qwen_current.test_author_basic_eot_benchmark import Tests as AuthorFixtures
from official.diagnostics.fe_qwen_current.test_failure_eot_probe import TinyTokenizer
from official.diagnostics.fe_qwen_current.test_failure_key_probe import load_function
from official.diagnostics.fe_qwen_current.storage import save_latest,state_digest,tensor_sha
from official.experiments.checkpoint import rng_snapshot
from official.runners.server1.common import member
from official.tracking.schema import config as validate_tracking
from official.diagnostics.fe_native.patch import function_text
from official.diagnostics.fe_qwen_current.storage import admit_remaining


class Tests(unittest.TestCase):
    def test_author_all_fields_dtype_only(self):
        f=AuthorFixtures();f.setUp()
        cfg,_=a.author.execution_config(f.historical,f.b);cfg.model_dtype='float32';cfg.diagnostic_z_reference='batch_entry';cfg.gpu=0
        actual=a.audit_settings(cfg,f.profiles)
        self.assertEqual(actual['llms'],f.profiles['llms']);self.assertEqual(actual['algs'],f.profiles['algs'])
        for group,key,value in [('algs','L2',10),('llms','clamp_norm_factor',.5),('llms','mom2_dtype','float64')]:
            bad=copy.deepcopy(cfg);bad[group][key]=value
            with self.assertRaises(ValueError):a.audit_settings(bad,f.profiles)
        cfg.model_dtype='bfloat16'
        with self.assertRaises(ValueError):a.audit_settings(cfg,f.profiles)

    def test_hosts_tracking_storage_bytes(self):
        for server,dataset,env,job in [('server4','cf',{'SLURM_JOB_ID':'63199'},'63199'),('rent','zsre',{'ODEEDIT_RENT_JOB_ID':'121500'},'121500')]:
            b=dict(dataset=dataset,main_stream=dict(sha256=a.STREAMS[dataset][1]),execution=dict(server=server,attempt='fp32-r1',source_commit='a'*40,job_id=job,source_run_id=a.REFERENCES[dataset]))
            values=a.tracking(b,env);validate_tracking(values)
            self.assertEqual(values['evaluation_profile'],a.PROFILE);self.assertEqual(values['job_id'],job)
        bound=a.payload_bound(dict(hidden_size=3584,intermediate_size=18944))
        self.assertEqual(bound,5*(3584*18944*4+18944**2*4)+16*1024**2)

    def test_storage_credit_and_real_shortage(self):
        from official.experiments.prepare import write_new
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'checkpoints').mkdir();p=out/'checkpoints/latest.pt'
            torch.save({'tiny':torch.ones(100)},p)
            write_new(out/'checkpoints/latest.json',dict(identity={'config_sha256':'a'*64},bytes=p.stat().st_size))
            cfg=dict(planned_latest_count=2,checkpoint_payload_bound=10000,raw_margin_bytes=1000,reserve_bytes=20000)
            rows=[dict(output=str(out),config_sha256='a'*64)]
            budget=admit_remaining(cfg,rows,51000)
            self.assertGreater(budget['credited_existing_latest_bytes'],0)
            with self.assertRaises(ValueError):admit_remaining(cfg,rows,budget['required_free_bytes']-1)
            wrong=admit_remaining(cfg,[dict(output=str(out),config_sha256='b'*64)],51000)
            self.assertEqual(wrong['credited_existing_latest_bytes'],0)

    def test_actual_native_fp32_target_writer_observation_parity(self):
        root=Path(os.environ.get('FE_AUTHOR_ROOT','/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author'))
        if not root.exists():self.skipTest('pinned author unavailable')
        original=(root/'algs/memit/FE-memit_main.py').read_text();patched=a.native_source(original)
        for name in ('batch_edit','load_cov','get_context_templates','upd_matrix_match_shape'):
            self.assertEqual(function_text(original,name),function_text(patched,name))
        spec=importlib.util.spec_from_file_location('fp32_nethook',root/'util/nethook.py');nh=importlib.util.module_from_spec(spec);spec.loader.exec_module(nh)
        tok=TinyTokenizer();ns=load_function(root/'locate_edit_utils/repr_tools.py',{'get_words_idxs_in_templates'},dict(torch=torch))
        lookup=load_function(root/'algs/memit/compute_z.py',{'find_fact_lookup_idx'},ns)['find_fact_lookup_idx']
        fit=load_function(root/'algs/memit/compute_z.py',{'compute_z'},dict(torch=torch,np=np,nethook=nh,find_fact_lookup_idx=lookup))['compute_z']
        torch.set_num_threads(1);torch.manual_seed(51)
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=8,intermediate_size=16,num_hidden_layers=28,num_attention_heads=2,num_key_value_heads=2,use_cache=False)).float().eval()
        cfg=OmegaConf.create(dict(seed=0,gpu=0,lti=False,negetive_prompt_test=False,bs=100,num_edits=2000,algs=dict(L2=0,add_old_keys=True),llms=dict(
            layers=[4,5,6,7,8],mom2_update_weight=[15000]*5,lm_head_module='lm_head',ln_f_module='model.norm',midlayers=[15],layer_module_tmp='model.layers.{}',rewrite_module_tmp='model.layers.{}.mlp.down_proj',
            constr_pos='subject',fact_token='subject_last',v_loss_layer=27,v_lr=.01,v_num_grad_steps=2,kl_factor=.0625,v_weight_decay=.001,clamp_norm_factor=1)))
        contexts=[['{}'],['1 {}','3 {}','4 {}','5 {}','6 {}']];adapter=a.eot.InputAdapter(tok,lookup)
        def extraction(m,t,l,**kw):
            key=torch.arange(1,17,dtype=torch.float32).repeat(len(kw['words']),1)/16
            W=dict(m.named_parameters())[cfg.llms.rewrite_module_tmp.format(l)+'.weight']
            return key,key@W.T
        key=load_function(root/'algs/memit/compute_ks.py',{'compute_ks'},dict(torch=torch,np=np,get_module_input_output_at_words=extraction))['compute_ks']
        native=load_function(root/'algs/memit/FE-memit_main.py',{'batch_edit','upd_matrix_match_shape'},dict(torch=torch,time=time,covs=[torch.eye(16)]*5,nethook=nh,
            get_context_templates=lambda *args:contexts,compute_ks=key,get_module_input_output_at_words=extraction))['batch_edit']
        writer=adapter.writer(native,key);wrappedfit=adapter.fit(fit);supplier=adapter.supplier(a.sequential_z.current_batch_targets)
        req=[dict(case_id=1,prompt='{} is a',subject='2',target_new=' 4 5',sample_idx=0)]
        with contextlib.redirect_stdout(io.StringIO()):
            weights_before={n:tensor_sha(p) for n,p in model.named_parameters()};rng_before=state_digest(rng_snapshot())
            target=supplier(model,tok,req,cfg,contexts,compute_z=wrappedfit,find_fact_lookup_idx=adapter.lookup,TraceDict=nh.TraceDict)
            self.assertEqual(weights_before,{n:tensor_sha(p) for n,p in model.named_parameters()})
            self.assertEqual(rng_before,state_digest(rng_snapshot()))
            self.assertTrue(all(t.dtype==torch.float32 for t in target.values()))
            m2=copy.deepcopy(model);H=torch.zeros(5,16,16);H2=H.clone();before=state_digest(rng_snapshot())
            writer(cfg,model,tok,req,'cpu',H,target)
            observed=a.observed_write(writer,cfg,m2,tok,req,'cpu',H2,target,1)
        self.assertEqual(state_digest(rng_snapshot()),before);self.assertTrue(torch.equal(H,H2))
        self.assertEqual(len(observed['layers']),5)
        for (n,p),(n2,p2) in zip(model.named_parameters(),m2.named_parameters()):self.assertEqual(n,n2);self.assertTrue(torch.equal(p,p2))
        # Same FP32 state atomic save/mmap roundtrip; then actual collector rejection.
        cfg.diagnostic_z_reference='batch_entry';cfg.model_dtype='float32';resolved=OmegaConf.to_container(cfg)
        names={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers}
        stream=[dict(case_id=i) for i in range(2000)];identity={'test':True}
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);cursor=dict(edits=2000,case_ids=list(range(2000)),native_config=resolved,input_adapter=member(a.eot.__file__))
            cp=save_latest(folder/'cp',{n:p for n,p in model.named_parameters() if n in names},H,contexts,{},20,identity,cursor,lock_path=folder/'lock',reserve_bytes=0)
            proof=dict(checkpoint=cp,pointer=member(folder/'cp/latest.json'),identity=identity)
            a.audit_checkpoint(proof,stream,resolved)
            payload=torch.load(cp['path'],weights_only=False,mmap=True)
            self.assertTrue(torch.equal(H,payload['cache_c']))
            for n,w in payload['weights'].items():self.assertEqual(tensor_sha(w),tensor_sha(dict(model.named_parameters())[n]))
            # Fresh separate BF16 checkpoint must never pass the FP32 collector.
            bf=save_latest(folder/'bf',{n:p.bfloat16() for n,p in model.named_parameters() if n in names},H,contexts,{},20,identity,cursor,lock_path=folder/'lock',reserve_bytes=0)
            with self.assertRaises(ValueError):a.audit_checkpoint(dict(checkpoint=bf,pointer=member(folder/'bf/latest.json'),identity=identity),stream,resolved)
            bad=dict(proof);bad['checkpoint']=dict(cp,batch=15)
            with self.assertRaises(ValueError):a.audit_checkpoint(bad,stream,resolved)


if __name__=='__main__':unittest.main()
