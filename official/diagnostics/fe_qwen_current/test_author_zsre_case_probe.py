"""Actual pinned native functions, random tiny BF16 CPU Qwen. No pretrained load."""
import ast
import contextlib
import copy
import importlib.util
import inspect
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import Qwen2Config, Qwen2ForCausalLM
from official.diagnostics.fe_native.sequential_z import current_batch_targets
from official.diagnostics.fe_qwen_current import author_zsre_case_probe as p
from official.diagnostics.fe_qwen_current.test_failure_eot_probe import TinyTokenizer
from official.diagnostics.fe_qwen_current.test_failure_key_probe import load_function
from official.experiments.checkpoint import rng_snapshot, rng_restore


class Tests(unittest.TestCase):
    def setUp(self):
        self.root = Path(os.environ.get('FE_NATIVE_AUTHOR_ROOT', '/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author'))
        if not self.root.is_dir(): self.skipTest('pinned native fixture unavailable')
        self.yaml = {'author_yaml': {k: p.member(self.root / path) for k, path in
            [('algs', 'configs/algs/memit.yaml'), ('llms', 'configs/llms/qwen2.5-7b.yaml')]}}
        self.profiles = p.author.author_profiles(self.yaml)

    def parent_fixture(self):
        cfg = dict(model_dtype='bfloat16', bs=100, num_edits=2000, seed=0, gpu=0,
            cov_mode='default', lti=False, negetive_prompt_test=False, diagnostic_z_reference='batch_entry', **copy.deepcopy(self.profiles))
        parent = dict(dataset='zsre', execution=dict(source_commit=p.SOURCE, job_id='121243'),
            main_stream=dict(path='/original/main-zsre.json', bytes=1490821, sha256=p.STREAM_SHA))
        audit = dict(resolved=cfg, resolved_sha256=p.digest(cfg), profiles_sha256=p.digest(self.profiles),
            status='AUTHOR_SETTINGS_CPU_VERIFIED', author_commit=p.author.AUTHOR)
        identity = dict(stage='AUTHOR_BASIC_EOT_MAIN', dataset='zsre', source=p.SOURCE, job_id='121243',
            config_sha256=p.digest(parent), author_settings_sha256=audit['resolved_sha256'], author_commit=p.author.AUTHOR,
            stream=parent['main_stream'], adapter=dict(sha256=p.ADAPTER_SHA), L2=0, clamp_norm_factor=1.,
            no_mask=True, z_reference='batch_entry', input_policy=p.author.common.POLICY)
        runtime = dict(native_config=cfg, author_settings=audit, identity=identity, attention_backend='sdpa')
        cp = dict(path='/original/latest.pt', sha256=p.CP_SHA, bytes=p.CP_BYTES)
        pointer = dict(cp, batch=15, identity=identity, final_W20=False)
        commit = dict(batch=15, edits=1500, identity=identity, checkpoint=pointer, raw=[])
        return parent, runtime, audit, commit, pointer, cp

    def test_real_author_parent_accept_and_old_tuned_rejection(self):
        args = self.parent_fixture(); cfg = p.validate_parent(*args, self.profiles)
        self.assertEqual(cfg.algs.L2, 0); self.assertEqual(cfg.llms.v_num_grad_steps, 35)
        for group, key, value in [('algs', 'L2', 10), ('llms', 'clamp_norm_factor', .5), ('llms', 'v_lr', .1)]:
            bad = copy.deepcopy(args); bad[1]['native_config'][group][key] = value
            with self.assertRaises(ValueError): p.validate_parent(*bad, self.profiles)
        for field, value in [('source', 'a'*40), ('job_id', '121242'), ('dataset', 'cf')]:
            bad = copy.deepcopy(args); bad[1]['identity'][field] = value
            with self.assertRaises(ValueError): p.validate_parent(*bad, self.profiles)
        bad = copy.deepcopy(args); bad[5]['sha256'] = '0'*64
        with self.assertRaises(ValueError): p.validate_parent(*bad, self.profiles)
        bad = copy.deepcopy(args); bad[3]['batch'] = 16
        with self.assertRaises(ValueError): p.validate_parent(*bad, self.profiles)

    def test_weights_restore_does_not_touch_H_or_unselected(self):
        model = torch.nn.Module(); model.layers = torch.nn.ModuleDict({str(l): torch.nn.Linear(5, 3, bias=False, dtype=torch.bfloat16) for l in range(9)})
        cfg = OmegaConf.create(dict(llms=dict(layers=[4,5,6,7,8], rewrite_module_tmp='layers.{}')))
        stream = [dict(case_id=i) for i in range(2000)]; commit = dict(raw=[dict(sha256='a'*64)])
        class NoHistory(dict):
            def __getitem__(self, key):
                if key == 'cache_c': raise AssertionError('H accessed')
                return super().__getitem__(key)
            def items(self): raise AssertionError('whole payload traversal')
        payload = NoHistory(identity=dict(test=1), batch=15, cursor=dict(native_config=OmegaConf.to_container(cfg),
            edits=1500, case_ids=list(range(1500)), raw=commit['raw'], input_adapter=dict(sha256=p.ADAPTER_SHA)),
            weights={f'layers.{l}.weight': torch.ones(3,5,dtype=torch.bfloat16)*l for l in range(4,9)},
            contexts=[['{}'],['1 {}','2 {}','3 {}','4 {}','5 {}']], rng=rng_snapshot(), cache_c=object())
        other = {n: x.detach().clone() for n, x in model.named_parameters() if int(n.split('.')[1]) < 4}
        contexts, rng, hashes = p.restore_weights_only(model, payload, cfg, dict(test=1), commit, stream)
        self.assertEqual(len(hashes),5); self.assertEqual(contexts,payload['contexts'])
        for n, x in model.named_parameters():
            if n in other: self.assertTrue(torch.equal(x,other[n]))
            else: self.assertTrue(torch.equal(x,payload['weights'][n]))
        self.assertIs(rng,payload['rng'])

    def test_native_context_aggregation_exact_KL_excluded(self):
        torch.manual_seed(9); raw = torch.randn(7,16).to(torch.bfloat16)
        native = load_function(self.root/'algs/memit/compute_ks.py', {'compute_ks'}, dict(torch=torch,np=np,
            get_module_input_output_at_words=lambda *args,**kw: (raw[:6],torch.zeros(6,8))))['compute_ks']
        cfg = OmegaConf.create(dict(negetive_prompt_test=False,llms=dict(rewrite_module_tmp='layers.{}',fact_token='subject_last')))
        req = [dict(prompt='{} is a',subject='2 3')]
        with contextlib.redirect_stdout(io.StringIO()):
            expected = native(None,None,req,cfg,5,[['{}'],['1 {}','2 {}','3 {}','4 {}','5 {}']])
        actual = p.key_geometry(raw)
        self.assertEqual(actual['context_average'],p.base.scalar_vector(expected[0]))
        raw[6].fill_(999)
        self.assertEqual(actual,p.key_geometry(raw))

    def test_actual_native_bf16_observer_on_off_parity_and_no_write(self):
        spec=importlib.util.spec_from_file_location('author_zsre_native_nethook',self.root/'util/nethook.py')
        nh=importlib.util.module_from_spec(spec);spec.loader.exec_module(nh)
        tools=load_function(self.root/'locate_edit_utils/repr_tools.py',{'get_words_idxs_in_templates'},dict(torch=torch))
        lookup=load_function(self.root/'algs/memit/compute_z.py',{'find_fact_lookup_idx'},tools)['find_fact_lookup_idx']
        compute=load_function(self.root/'algs/memit/compute_z.py',{'compute_z'},dict(torch=torch,np=np,nethook=nh,find_fact_lookup_idx=lookup))['compute_z']
        tok=TinyTokenizer();torch.set_num_threads(1);torch.manual_seed(53)
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=8,intermediate_size=16,
            num_hidden_layers=28,num_attention_heads=2,num_key_value_heads=2,use_cache=False)).to(torch.bfloat16).eval()
        # Exact native compute_z and supplier; tiny dimensions/3 CPU iterations
        # are fixture-only, never the production author-settings validator.
        cfg=OmegaConf.create(dict(gpu=0,lti=False,negetive_prompt_test=False,bs=100,algs=dict(L2=0),diagnostic_z_reference='batch_entry',llms=dict(
            layers=[4,5,6,7,8],lm_head_module='lm_head',ln_f_module='model.norm',midlayers=[15],layer_module_tmp='model.layers.{}',
            rewrite_module_tmp='model.layers.{}.mlp.down_proj',constr_pos='subject',fact_token='subject_last',v_loss_layer=27,
            v_lr=.5,v_num_grad_steps=3,kl_factor=.0625,v_weight_decay=.001,clamp_norm_factor=1)))
        requests=[dict(case_id=cid,sample_idx=i,prompt='1 {} 4 ',subject='2 3',target_new=' 5 6') for i,cid in enumerate(p.CASES)]
        contexts=[['{}'],['1 {}','3 {}','5 {}','6 {}','7 {}']]
        w0=model.model.layers[5].mlp.down_proj.weight.detach().clone()
        with torch.no_grad():model.model.layers[5].mlp.down_proj.weight.add_(.01)
        before={n:v.detach().clone() for n,v in model.named_parameters()};grads={n:v.requires_grad for n,v in model.named_parameters()}
        state=rng_snapshot();hooks=p.hooks_state(model)
        with contextlib.redirect_stdout(io.StringIO()):
            expected=current_batch_targets(model,tok,requests,cfg,contexts,compute_z=compute,find_fact_lookup_idx=lookup,TraceDict=nh.TraceDict)
        for n,v in model.named_parameters():v.requires_grad_(grads[n])
        rng_restore(state)
        with patch.object(torch.linalg,'solve',side_effect=AssertionError('SOLVE_FORBIDDEN')), contextlib.redirect_stdout(io.StringIO()):
            actual,receipt=p.observe(model,tok,requests,cfg,contexts,compute_z=compute,supplier=current_batch_targets,lookup=lookup,TraceDict=nh.TraceDict,w0=w0)
        for l in expected:self.assertTrue(torch.equal(expected[l],actual[l]))
        for n,v in model.named_parameters():self.assertTrue(torch.equal(v,before[n]));self.assertEqual(v.requires_grad,grads[n])
        self.assertEqual(p.state_digest(state),p.state_digest(rng_snapshot()));self.assertEqual(hooks,p.hooks_state(model))
        self.assertEqual(receipt['fits'],3);self.assertEqual(receipt['supplier_replays'],3);self.assertEqual(receipt['extra_bare_forwards'],6)
        self.assertFalse(receipt['history_accessed']);self.assertEqual(receipt['writer_calls'],0)
        self.assertTrue(receipt['weights_rng_hooks_modes_unchanged'])
        for row in receipt['rows']:
            self.assertEqual(len(row['input_rows']),7);self.assertEqual(len(row['fit_probabilities']),6)
            self.assertTrue(row['first_fit_delta_zero_key_geometry']['KL_excluded'])
            self.assertEqual(row['final_fit_vs_return_cast']['maxabs'],0)
            self.assertLess(row['replay_linear']['decomposition']['maxabs'],1e-10)
            self.assertEqual(len(row['extra']),2);self.assertNotIn('B5_k_norm',row['replay_linear'])
        for change in ('cases','L2','reference'):
            rows=copy.deepcopy(requests);bad=copy.deepcopy(cfg)
            if change=='cases':rows.reverse()
            elif change=='L2':bad.algs.L2=10
            else:bad.diagnostic_z_reference='initial_model_on_arrival'
            with self.assertRaises(ValueError):p.observe(model,tok,rows,bad,contexts,compute_z=compute,supplier=current_batch_targets,lookup=lookup,TraceDict=nh.TraceDict,w0=w0)

    def test_no_writer_history_or_evaluator_in_runtime_route(self):
        tree=ast.parse(inspect.getsource(p))
        calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(calls & {'batch_edit','apply_memit_to_model','solve','evaluate','save','save_latest','get_context_templates'})
        indexes={n.slice.value for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant)}
        self.assertNotIn('cache_c',indexes)
        self.assertEqual(p.member(p.eot.__file__)['sha256'],p.ADAPTER_SHA)


if __name__ == '__main__':unittest.main()
