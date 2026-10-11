import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from official.diagnostics.fe_qwen_current.contract import SPEC, select, split_rows, tracking_config
from official.diagnostics.fe_qwen_current.patch import native_source, main_source, function_text
from official.diagnostics.fe_qwen_current.observer import preserved_evaluation
from official.experiments.prepare import digest
from official.tracking.schema import config as validate_tracking
from official.runners.fe_original import save_latest, load_latest, cf_payload
from official.evaluation.factual import evaluate_counterfact, _plan, validate_retained_observation
from official.tests.test_factual import CharacterTokenizer,TransitionLM,cf
from official.tracking.schema import metrics
from official.tracking.schema import bind_job_identity
from official.tests.test_fe_native import Model, config, requests, tokenize
from official.diagnostics.fe_qwen_current.prepare import validate_bindings
from official.diagnostics.fe_qwen_current import observer
from official.diagnostics.fe_qwen_current import storage
from official.runners.fe_original import state_digest as old_digest

def result(e=96,g=90,s=80):
    return dict(Efficacy=e,Generalization=g,Specificity=s,Score=3/(1/e+1/g+1/s),
                requests=2000,committed_batches=20)

class ContractTests(unittest.TestCase):
    def test_rent_config_and_job_backend(self):
        c=dict(server='rent',L2=1,attempt='r1',config_sha256='a'*64,
            selection_contract_sha256=digest(SPEC),stream=dict(sha256='b'*64))
        cfg=bind_job_identity(tracking_config(c,'c'*40),{'ODEEDIT_RENT_JOB_ID':'120101'})
        self.assertEqual(cfg['execution_backend'],'kubernetes')
        self.assertEqual(cfg['server'],'rent')
    def test_later_sibling_admission_no_double_reserve_and_real_shortage(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            first=Path(tmp)/'first';(first/'checkpoints').mkdir(parents=True)
            payload=first/'checkpoints/latest.pt';payload.write_bytes(b'x'*2048)
            c=dict(output=str(first),config_sha256='a'*64,checkpoint_payload_bound=4096,
                planned_latest_count=6,raw_margin_bytes=1024,reserve_bytes=8192)
            baseline=storage.remaining_storage(c,[c])['required_free_bytes']
            meta=dict(identity=dict(config_sha256=c['config_sha256']),bytes=2048)
            (first/'checkpoints/latest.json').write_text(json.dumps(meta))
            after=storage.remaining_storage(c,[c])
            self.assertEqual(after['required_free_bytes'],baseline-2048)
            storage.admit_remaining(c,[c],baseline-2048)
            with self.assertRaisesRegex(ValueError,'STORAGE_REQUIRED'):
                storage.admit_remaining(c,[c],baseline-2049)
            self.assertEqual(after['reserve_bytes'],8192)
            # Unknown or mismatched identity never credits a path.
            self.assertEqual(storage.remaining_storage(dict(c),[dict(c,config_sha256='b'*64)])['required_free_bytes'],baseline)
    def test_chunked_hash_exact_bytes_and_small_checkpoint_compatible(self):
        import numpy as np
        data=dict(bf16=torch.arange(20,dtype=torch.bfloat16).reshape(4,5),
            float32=torch.arange(100,dtype=torch.float32).reshape(10,10),
            numpy=np.arange(20,dtype=np.uint32),scalar=np.int64(4),tuple=(1,2))
        self.assertEqual(storage.state_digest(data),old_digest(data))
        self.assertTrue(storage.finite_tensor(data['float32']))
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'run';ident=dict(source='a');w={'weight':data['bf16']};h=torch.ones(1,2,2)
            for batch in (5,10):
                storage.save_latest(folder,w,h,[['{}']],{},batch,ident,dict(edits=batch*100),
                    lock_path=Path(tmp)/'lock',reserve_bytes=0)
            self.assertEqual(load_latest(folder,ident)['batch'],10)
            self.assertEqual(sorted(p.name for p in folder.iterdir()),['latest.json','latest.pt'])
    def test_host_binding_keeps_s2_reserve_and_threads(self):
        b=dict(server='server2',workspace='/repo',python='/venv/python',author_clone='/author',
            tracking_env_file='/env',checkpoint_lock='/local/lock',hf_cache='/hf',cpus=6,
            reserve_bytes=32*1024**3,planned_latest_count=6,raw_margin_bytes=4*1024**3,
            runtime=dict(torch='2.9.1+cu128',transformers='4.51.3'),C0=[dict(layer=l) for l in SPEC['layers']])
        self.assertEqual(validate_bindings(b)['reserve_bytes'],32*1024**3)
        with self.assertRaises(ValueError):validate_bindings(dict(b,reserve_bytes=31*1024**3))
        with self.assertRaises(ValueError):validate_bindings(dict(b,server='server1'))
    def test_control_never_promoted_and_tie_lower_l2(self):
        rows={l:result() for l in SPEC['L2_arms']}
        self.assertEqual(select(rows)['selected_L2'],.1)
        rows[0]=result(99,99,99)
        self.assertEqual(select(rows)['status'],'NO_PROMOTION')
    def test_no_promotion_below_gate(self):
        rows={l:result(s=60) for l in SPEC['L2_arms']}
        self.assertEqual(select(rows)['status'],'NO_PROMOTION')
    def test_failed_arm_does_not_bypass_control_veto(self):
        rows={l:result() for l in SPEC['L2_arms']};rows[0]=result(99,99,99)
        rows[10]=dict(status='NUMERICAL_FAILURE',verified_failure=True,failure_receipt_sha256='a'*64)
        self.assertEqual(select(rows)['status'],'NO_PROMOTION')
        rows[0]=dict(status='NUMERICAL_FAILURE',verified_failure=True,failure_receipt_sha256='b'*64)
        self.assertEqual(select(rows)['selected_L2'],.1)
        rows[10]=dict(status='RESOURCE_FAILURE')
        with self.assertRaises(ValueError):select(rows)
    def test_partial_and_wrong_score_rejected(self):
        rows={l:result() for l in SPEC['L2_arms']}
        rows[10]['committed_batches']=19
        with self.assertRaises(ValueError):select(rows)
        rows[10]=result();rows[10]['Score']=100
        with self.assertRaises(ValueError):select(rows)
    def test_split_order_disjoint(self):
        rows=[dict(case_id=i) for i in range(10000)]
        chosen,proof=split_rows(rows,rows[:2000],rows[2000:4000])
        self.assertEqual(chosen,rows[4000:6000]);self.assertTrue(proof['pairwise_case_id_disjoint'])
        rows[4000]['case_id']=0
        with self.assertRaises(ValueError):split_rows(rows,rows[:2000],rows[2000:4000])
    def test_tracking_heldout_group_and_baseline_not_main(self):
        c=dict(server='server2',L2=.1,attempt='r1',config_sha256='a'*64,selection_contract_sha256=digest(SPEC),stream=dict(sha256='b'*64))
        values=validate_tracking(tracking_config(c,'c'*40))
        self.assertIn('heldout',values['task_id']);self.assertIn('tuned',values['baseline'])
        self.assertEqual(values['evaluation_profile'],'heldout-selection-not-main')
        self.assertIn('batch_entry',values['writer'])
        self.assertEqual(values['server'],'server2')
    def test_eval_restores_rng_and_rejects_mutation(self):
        m=Model();h=torch.zeros(2,3,3);rng=torch.get_rng_state().clone()
        preserved_evaluation(m,h,lambda:torch.rand(4))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        with self.assertRaisesRegex(ValueError,'MUTATED'):
            preserved_evaluation(m,h,lambda:h.add_(1))
    def test_actual_current_and_all_seen_payload_and_raw_tamper(self):
        model=TransitionLM();tok=CharacterTokenizer();rows=[cf(i) for i in range(1,101)]
        actual=evaluate_counterfact(model,tok,rows,batch_size=16,device='cpu')
        plan=_plan(rows,'cf',tok)[2]
        self.assertEqual(validate_retained_observation(actual,'cf',input_signature=plan)['requests'],100)
        for endpoint,count in (('current/post',100),('all_seen/post',500)):
            measured=actual if count==100 else evaluate_counterfact(model,tok,[cf(i) for i in range(1,501)],batch_size=16,device='cpu')
            payload=cf_payload(measured,endpoint,count)
            c=dict(server='server2',L2=.1,attempt='r1',config_sha256='a'*64,
                selection_contract_sha256=digest(SPEC),stream=dict(sha256='b'*64))
            metrics(payload,scientific=True,config_values=tracking_config(c,'c'*40))
            self.assertEqual(payload['official/'+endpoint+'/requests'],count)
            self.assertEqual(payload[endpoint+'/R/count'],count)
            self.assertEqual(payload[endpoint+'/P/count'],count)
            self.assertEqual(payload[endpoint+'/N/count'],count*2)
        actual['summary']['Score']=0
        with self.assertRaises(ValueError):validate_retained_observation(actual,'cf',input_signature=plan)
    def test_bf16_latest_overwrite_no_W0_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'run';ident=dict(source='a');w={'weight':torch.ones(2,2,dtype=torch.bfloat16)}
            for batch in (5,10,15,20):
                save_latest(folder,w,torch.zeros(1,2,2),[['{}']],{},batch,ident,dict(edits=batch*100),lock_path=Path(tmp)/'lock')
            self.assertEqual(load_latest(folder,ident)['batch'],20)
            self.assertEqual(sorted(p.name for p in folder.iterdir()),['latest.json','latest.pt'])
    @unittest.skipUnless(os.environ.get('FE_NATIVE_AUTHOR_ROOT'),'pinned native source required')
    def test_actual_native_L2_consumption_and_patch_invariance(self):
        root=Path(os.environ['FE_NATIVE_AUTHOR_ROOT'])
        original=(root/'algs/memit/FE-memit_main.py').read_text()
        changed=native_source(original)
        self.assertEqual(function_text(original,'batch_edit'),function_text(changed,'batch_edit'))
        self.assertIn('observed_write(batch_edit',changed)
        main=(root/'main.py').read_text();modified=main_source(main)
        self.assertNotIn('save_model(edited_model,cfg)',modified)
        self.assertEqual(main.replace('            save_model(edited_model,cfg)',''),
                         modified[:modified.index('            # Storage only:')]+modified[modified.index('            finish(cfg)')+len('            finish(cfg)'):])
        K=torch.tensor([[1.,.2,.4],[.1,1.,.3]])
        ns=dict(torch=torch,nethook=SimpleNamespace(get_parameter=lambda m,n:m.get_parameter(n)),
                get_context_templates=lambda *a:[['{}']],compute_ks=lambda *a:K.clone(),
                time=__import__('time'),covs=[torch.eye(3),torch.eye(3)],
                get_module_input_output_at_words=lambda m,t,l,**kw:(None,torch.zeros(2,3)))
        exec(function_text(original,'upd_matrix_match_shape')+'\n'+function_text(original,'batch_edit'),ns)
        norms=[]
        for l2 in SPEC['L2_arms']:
            m=Model();cfg=config();cfg.algs.L2=l2;H=torch.zeros(2,3,3)
            rs=[dict(r,sample_idx=i) for i,r in enumerate(requests()[:2])]
            zs={l:torch.ones(3,2) for l in range(2)}
            before=m.layers[0].weight.detach().clone()
            with contextlib.redirect_stdout(io.StringIO()):ns['batch_edit'](cfg,m,tokenize,rs,'cpu',H,zs)
            expected=torch.linalg.solve(K.double().T@K.double()+2*torch.eye(3,dtype=torch.double)+l2*torch.eye(3,dtype=torch.double),K.double().T@torch.ones(2,3,dtype=torch.double))
            self.assertTrue(torch.allclose((m.layers[0].weight-before).double(),expected,atol=1e-7,rtol=1e-6))
            self.assertTrue(torch.equal(H[0],(K.double().T@K.double()).float()))
            norms.append(float(torch.linalg.norm(m.layers[0].weight.detach()-before)))
        self.assertTrue(all(a>b for a,b in zip(norms,norms[1:])))
    @unittest.skipUnless(os.environ.get('FE_NATIVE_AUTHOR_ROOT'),'pinned native source required')
    def test_layer_trace_has_identical_native_weights_history_rng(self):
        original=(Path(os.environ['FE_NATIVE_AUTHOR_ROOT'])/'algs/memit/FE-memit_main.py').read_text()
        K=torch.arange(300,dtype=torch.float32).reshape(100,3)/300
        ns=dict(torch=torch,nethook=SimpleNamespace(get_parameter=lambda m,n:m.get_parameter(n)),
            get_context_templates=lambda *a:[['{}']],compute_ks=lambda *a:K.clone(),
            time=__import__('time'),covs=[torch.eye(3) for _ in range(5)],
            get_module_input_output_at_words=lambda m,t,l,**kw:(None,torch.zeros(100,3)))
        exec(function_text(original,'upd_matrix_match_shape')+'\n'+function_text(original,'batch_edit'),ns)
        def make_model():
            m=torch.nn.Module();m.layers=torch.nn.ModuleList([torch.nn.Linear(3,3,bias=False) for _ in range(9)])
            with torch.no_grad():
                for p in m.parameters():p.copy_(torch.eye(3))
            return m
        a=make_model();b=make_model();cfg=config();cfg.bs=100
        cfg.llms.layers=SPEC['layers'];cfg.llms.mom2_update_weight=[2.]*5;cfg.algs.L2=.1
        ra=[dict(case_id=i,subject='x',prompt='{}',sample_idx=i) for i in range(100)]
        zs={l:torch.ones(3,100) for l in SPEC['layers']};ha=torch.zeros(5,3,3);hb=ha.clone()
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            ns['batch_edit'](cfg,a,tokenize,ra,'cpu',ha,zs)
            rng=torch.get_rng_state().clone()
            with patch.object(observer,'_STATE',dict(out=Path(tmp),last_batch=0)):
                observer.observed_write(ns['batch_edit'],cfg,b,tokenize,ra,'cpu',hb,zs,1)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(torch.equal(ha,hb))
            for pa,pb in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(pa,pb))

if __name__=='__main__':unittest.main()
