"""Native AST/real local tokenizer/tiny BF16 CPU checks; no pretrained forward."""
import contextlib
import copy
import importlib.util
import io
import os
import time
from types import SimpleNamespace as N
from pathlib import Path
import unittest
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer,Qwen2Config,Qwen2ForCausalLM
from official.diagnostics.fe_qwen_current.failure_eot_probe import InputAdapter,observe_fit
from official.diagnostics.fe_qwen_current.test_failure_key_probe import load_function
from official.diagnostics.fe_qwen_current.test_failure_probe import Batch
from official.diagnostics.fe_native.sequential_z import current_batch_targets


class TinyTokenizer:
    bos_token_id=30;unk_token_id=31
    def encode(self,s,**kw):
        prefix=s.startswith('<|endoftext|>');s=s.removeprefix('<|endoftext|>')
        return ([151643] if prefix else [])+[{'is':7,'a':8}.get(w,int(w) if w.isdigit() else 9) for w in s.split()]
    def __call__(self,s,**kw):
        ids=[self.encode(t) for t in ([s] if isinstance(s,str) else s)];n=max(map(len,ids))
        # EOT maps to tiny model token29 only in this synthetic model fixture.
        ids=[[29 if x==151643 else x for x in row] for row in ids]
        return Batch(input_ids=torch.tensor([r+[29]*(n-len(r)) for r in ids]),attention_mask=torch.tensor([[1]*len(r)+[0]*(n-len(r)) for r in ids]))
    def decode(self,ids):return ' '.join(str(int(x)) for x in ids)


class Tests(unittest.TestCase):
    def setUp(self):
        self.root=Path(os.environ.get('FE_AUTHOR_ROOT','/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author'))
        if not self.root.is_dir():self.skipTest('native author unavailable')
    def lookup(self):
        ns=load_function(self.root/'locate_edit_utils/repr_tools.py',{'get_words_idxs_in_templates'},dict(torch=torch))
        return load_function(self.root/'algs/memit/compute_z.py',{'find_fact_lookup_idx'},ns)['find_fact_lookup_idx']
    def test_real_tokenizers_all_routes(self):
        root=Path(os.environ.get('OURS_TOKENIZER_CACHE','/mnt/raid5/janghj/.cache/huggingface/hub'))
        for family,name,token in [('qwen2','Qwen--Qwen2.5-7B-Instruct',151643),('gptj','EleutherAI--gpt-j-6b',50256)]:
            paths=sorted((root/('models--'+name)/'snapshots').glob('*'))
            if not paths:self.skipTest('local tokenizer missing '+family)
            tok=AutoTokenizer.from_pretrained(paths[0],local_files_only=True);tok.pad_token='<|endoftext|>';tok.padding_side='right'
            adapter=InputAdapter(tok,self.lookup(),family)
            for subject,prompt in [('Paris','{} is in'),('New York','{} is in'),('Paris','Yesterday {} is in')]:
                original=tok.encode(prompt.format(subject));pos=self.lookup()(prompt,subject,tok,'subject_last',False)
                changed=adapter.prompt(prompt,subject);ids=tok.encode(changed.format(subject))
                self.assertEqual(ids,[token]+original if pos==0 else original)
                self.assertEqual(adapter.prompt(changed,subject),changed)
                self.assertEqual(adapter.lookup(prompt,subject,tok,'subject_last',verbose=False),1 if pos==0 else pos)
            prompts=['{} is in','Yesterday {} is in'];seen=[]
            def extract(m,t,l,**kw):seen.extend(kw['context_templates']);return None
            adapter.extraction(extract)(None,tok,4,context_templates=prompts,words=['Paris']*2,fact_token_strategy='subject_last')
            self.assertEqual(seen,[adapter.prompt(p,'Paris') for p in prompts])
            batch=tok([p.format('Paris') for p in seen],padding=True,return_tensors='pt')
            self.assertEqual(int(batch['input_ids'][0,0]),token);self.assertEqual(int(batch['attention_mask'][0,0]),1)

    def test_actual_bf16_fit_supplier_and_label_alignment(self):
        spec=importlib.util.spec_from_file_location('eot_native_nethook',self.root/'util/nethook.py');nh=importlib.util.module_from_spec(spec);spec.loader.exec_module(nh)
        tok=TinyTokenizer();lookup=self.lookup();adapter=InputAdapter(tok,lookup)
        ns=load_function(self.root/'algs/memit/compute_z.py',{'compute_z'},dict(torch=torch,np=np,nethook=nh,find_fact_lookup_idx=lookup))
        fit=ns['compute_z'];newfit=adapter.fit(fit);newsupplier=adapter.supplier(current_batch_targets)
        torch.set_num_threads(1);torch.manual_seed(17)
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=8,intermediate_size=16,num_hidden_layers=28,num_attention_heads=2,num_key_value_heads=2,use_cache=False)).to(torch.bfloat16).eval()
        cfg=OmegaConf.create(dict(gpu=0,lti=False,negetive_prompt_test=False,bs=100,llms=dict(layers=[4,5,6,7,8],lm_head_module='lm_head',ln_f_module='model.norm',midlayers=[15],layer_module_tmp='model.layers.{}',rewrite_module_tmp='model.layers.{}.mlp.down_proj',constr_pos='subject',fact_token='subject_last',v_loss_layer=27,v_lr=.01,v_num_grad_steps=3,kl_factor=.0625,v_weight_decay=.001,clamp_norm_factor=1)))
        contexts=[['{}'],['1 {}','3 {}','5 {}','6 {}','7 {}']]
        weights={n:p.detach().clone() for n,p in model.named_parameters()};rng=torch.get_rng_state().clone()
        with contextlib.redirect_stdout(io.StringIO()):
            for subject in ('2 3','2'):
                req=dict(case_id=1,prompt='{} is a',subject=subject,target_new=' 4 5 6');old=[];new=[]
                a=current_batch_targets(model,tok,[req],cfg,contexts,compute_z=observe_fit(fit,old),find_fact_lookup_idx=lookup,TraceDict=nh.TraceDict)
                b=newsupplier(model,tok,[req],cfg,contexts,compute_z=observe_fit(newfit,new),find_fact_lookup_idx=adapter.lookup,TraceDict=nh.TraceDict)
                for x,y in zip(old[0]['input_rows'],new[0]['input_rows']):
                    change=x['lookup']==0;self.assertEqual(y['ids'],[29]+x['ids'] if change else x['ids'])
                    self.assertEqual(y['lookup'],1 if change else x['lookup']);self.assertTrue(all(y['mask']))
                    if x['labels'] is not None:self.assertEqual(y['labels'],[-100]+x['labels'] if change else x['labels'])
                if subject=='2 3':
                    for l in a:self.assertTrue(torch.equal(a[l],b[l]))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for n,p in model.named_parameters():self.assertTrue(torch.equal(p,weights[n]))
        self.assertTrue(all(p['non_input_ast_exact'] for p in adapter.proofs.values()))

    def test_rectangular_native_writer_routes_and_unaffected_parity(self):
        tok=TinyTokenizer();adapter=InputAdapter(tok,self.lookup());calls=[]
        contexts=[['{}'],['1 {}','3 {}','5 {}','6 {}','7 {}']]
        def extraction(m,t,l,**kw):
            calls.append(kw['context_templates']);n=len(kw['words'])
            return torch.ones(n,4,dtype=torch.bfloat16),torch.ones(n,3,dtype=torch.bfloat16)
        key=load_function(self.root/'algs/memit/compute_ks.py',{'compute_ks'},dict(torch=torch,np=np,get_module_input_output_at_words=extraction))['compute_ks']
        env=load_function(self.root/'algs/memit/FE-memit_main.py',{'batch_edit','upd_matrix_match_shape'},dict(torch=torch,time=time,covs=[torch.eye(4)]*2,
            nethook=N(get_parameter=lambda m,n:dict(m.named_parameters())[n]),get_context_templates=lambda *a:contexts,compute_ks=key,get_module_input_output_at_words=extraction))
        original=env['batch_edit'];changed=adapter.writer(original,key)
        cfg=N(llms=N(layers=[4,5],rewrite_module_tmp='layers.{}',layer_module_tmp='layers.{}',fact_token='subject_last',mom2_update_weight=[15000]*2),algs=N(L2=10,add_old_keys=True),negetive_prompt_test=False)
        torch.manual_seed(1);a=torch.nn.Module();a.layers=torch.nn.ModuleDict({str(l):torch.nn.Linear(4,3,bias=False,dtype=torch.bfloat16) for l in (4,5)})
        b=copy.deepcopy(a);ha=torch.zeros(2,4,4);hb=ha.clone();req=[dict(case_id=1,sample_idx=0,prompt='{} is a',subject='2 3')];zs={l:torch.ones(3,1)*3 for l in (4,5)}
        with contextlib.redirect_stdout(io.StringIO()):
            original(cfg,a,tok,req,'cpu',ha,zs);changed(cfg,b,tok,req,'cpu',hb,zs)
            self.assertTrue(torch.equal(ha,hb))
            for p,q in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(p,q))
            calls.clear();req[0]['subject']='2';changed(cfg,b,tok,req,'cpu',hb,zs)
        for prompts in calls:
            self.assertTrue(prompts[0].startswith('<|endoftext|>'))
            if len(prompts)>1:self.assertTrue(all(not p.startswith('<|endoftext|>') for p in prompts[1:]))
        self.assertIs(env['compute_ks'],key);self.assertIs(env['get_module_input_output_at_words'],extraction)


if __name__=='__main__':unittest.main()
