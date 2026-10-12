"""CPU fixtures only. No real final proof/checkpoint or pretrained/GPU claim."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from official.diagnostics.fe_qwen_current import flucon as f,flucon_prepare as prep
from official.experiments.prepare import digest,write_new
from official.runners.server1.common import member,read
from official.tests.test_fe_cake_flucon import Tiny,HP,edited
from official.evaluation.generation.native_observer import NativeGenerationObserver,read_observed
from official.evaluation.generation.test_native_generator import Model,Tokenizer,prompt
from official.evaluation.generation.test_observer import FakeAssets,record
from official.evaluation.generation.generator import rng_snapshot,rng_equal
from official.evaluation.generation.common import GenerationError
from official.evaluation.generation.metrics import generation_payload,PUBLIC_REASONS
from official.evaluation.generation.paper_display import paper_generation
from official.tracking.schema import bind_job_identity,metrics


class Fixture:
    def __init__(self,root):
        self.root=Path(root);self.ids=list(range(2000));self.weights=edited(torch.bfloat16)
        self.identity=dict(stage='final_benchmark',dataset='cf',split='original_cf_first2000',
            selection_contract_sha256=digest(f.SPEC),L2=100,z_reference='batch_entry',dtype='bfloat16')
        write_new(self.root/'B20.json',dict(raw=[{'fixture':'current'},{'fixture':'allseen'}]))
        self.payload=dict(weights=self.weights,cache_c=torch.zeros(2,4,4),contexts=[['{}']],
            z_member=dict(policy='batch_entry_current100',durable_target_cache=False),batch=20,
            identity=self.identity,rng={'fixture_not_restored':True},integrity_sha256='a'*64,
            cursor=dict(edits=2000,case_ids=self.ids,raw=read(self.root/'B20.json')['raw'],
                native_config=dict(bs=100,num_edits=2000,seed=0,model_dtype='bfloat16',
                    algs=dict(L2=100,add_old_keys=True),llms=dict(HP,clamp_norm_factor=1,
                        v_num_grad_steps=35,v_lr=.5,v_weight_decay=.001,kl_factor=.0625,
                        v_loss_layer=27,mom2_update_weight=[15000]*5))))
        self.original=dict(identity=self.identity,commit=member(self.root/'B20.json'),hparams=HP,
            payload_tensor_sha256={n:f.tensor_sha(v) for n,v in self.weights.items()},payload_integrity_sha256='a'*64)
    def save(self):
        torch.save(self.payload,self.root/'latest.pt')
        cp=member(self.root/'latest.pt')
        # Re-signing is ONLY a corruption fixture, not production API.
        (self.root/'latest.json').write_text(json.dumps(dict(cp,batch=20,final_W20=True,identity=self.identity)))
        self.original.update(checkpoint=cp,pointer=member(self.root/'latest.json'))
        return self.original


class FluconTests(unittest.TestCase):
    def test_missing_or_incomplete_final_proof_blocks_before_preparation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_new(root/'incomplete.json',dict(both_W20_complete=False))
            write_new(root/'bindings.json',dict(final_result=member(root/'incomplete.json')))
            with patch.object(torch,'load',side_effect=AssertionError('must not mmap')):
                with self.assertRaisesRegex(ValueError,'BOTH_FINAL_W20_REQUIRED'):
                    prep.prepare(root/'new',root/'bindings.json')
            self.assertFalse((root/'new').exists())
    def test_old_selection_or_no_selection_never_bypasses_rescue_reaudit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'selection-result.json';write_new(p,dict(selection=dict(status='SELECTED',selected_L2=10)))
            with self.assertRaises((FileNotFoundError,ValueError,KeyError)):f.validated_selection(member(p))
    def test_real_mmap_bf16_restore_exact_other_parameters_and_rng_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            x=Fixture(d);original=x.save();model=Tiny()
            untouched={n:p.detach().clone() for n,p in model.named_parameters() if n not in x.weights}
            rng=torch.get_rng_state().clone();before=member(Path(d)/'latest.pt');load=torch.load
            with patch.object(f,'HP',HP),patch.object(f,'SELECTED_SHAPE',(3,4)),patch.object(f,'ORDER',digest(x.ids)),\
                 patch.object(torch,'load',wraps=load) as loading,\
                 patch('official.diagnostics.fe_qwen_current.storage.state_digest',side_effect=AssertionError('H traversal forbidden')):
                result=f.restore(model,original)
            self.assertTrue(loading.call_args.kwargs['mmap']);self.assertEqual(loading.call_args.kwargs['map_location'],'cpu')
            self.assertFalse(result['history_contexts_rng_consumed'])
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(before,member(Path(d)/'latest.pt'))
            for n,p in model.named_parameters():
                self.assertTrue(torch.equal(p,x.weights[n].float() if n in x.weights else untouched[n]))
                if n in x.weights:self.assertTrue(torch.equal(p.bfloat16(),x.weights[n]))
    def test_re_signed_wrong_checkpoint_identity_cursor_hparams_rejected(self):
        for kind in ('heldout','dataset','batch','cursor','order','dtype','z','L2','clamp','weight_dtype','weight_shape','nonfinite','integrity'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                x=Fixture(d);p=x.payload
                if kind=='heldout':p['identity']['stage']='heldout_selection'
                if kind=='dataset':p['identity']['dataset']='zsre'
                if kind=='batch':p['batch']=19
                if kind=='cursor':p['cursor']['edits']=1900
                if kind=='order':p['cursor']['case_ids']=list(reversed(x.ids))
                if kind=='dtype':p['identity']['dtype']='float32'
                if kind=='z':p['z_member']['policy']='W0'
                if kind=='L2':p['identity']['L2']=10
                if kind=='clamp':p['cursor']['native_config']['llms']['clamp_norm_factor']=.5
                first=next(iter(p['weights']))
                if kind=='weight_dtype':p['weights'][first]=p['weights'][first].float()
                if kind=='weight_shape':p['weights'][first]=p['weights'][first][:1]
                if kind=='nonfinite':p['weights'][first][0,0]=float('nan')
                if kind=='integrity':p['integrity_sha256']='b'*64
                original=x.save()
                with patch.object(f,'HP',HP),patch.object(f,'SELECTED_SHAPE',(3,4)),patch.object(f,'ORDER',digest(x.ids)),self.assertRaises(ValueError):
                    f.payload_weights(original)
    def test_payload_sha_checked_before_mmap_and_weight_hash_before_write(self):
        with tempfile.TemporaryDirectory() as d:
            x=Fixture(d);original=x.save();model=Tiny();before={n:p.clone() for n,p in model.named_parameters()}
            original['payload_tensor_sha256'][next(iter(x.weights))]='f'*64
            with patch.object(f,'HP',HP),patch.object(f,'SELECTED_SHAPE',(3,4)),patch.object(f,'ORDER',digest(x.ids)),self.assertRaisesRegex(ValueError,'TENSOR_SHA'):
                f.restore(model,original)
            for n,p in model.named_parameters():self.assertTrue(torch.equal(p,before[n]))
            with (Path(d)/'latest.pt').open('ab') as out:out.write(b'corrupt')
            with patch.object(torch,'load',side_effect=AssertionError('must verify first')),self.assertRaisesRegex(ValueError,'MEMBER_CHANGED'):
                f.payload_weights(original)
    def test_original_reference_identity_cannot_be_relabelled(self):
        manifest=Path('/mnt/raid5/janghj/ODE-edit/local/baseline-generation-eval-assets/20261007/reference-ready-r1/manifest.json')
        if not manifest.exists():self.skipTest('original small manifest absent')
        m=read(manifest);m['versions']['numpy']='changed'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fake.json';write_new(p,m)
            with self.assertRaises((GenerationError,ValueError,RuntimeError)):
                f.load_reference(p)
    def config(self):
        return dict(key='qwen25-final-generation',model='qwen25',server='rent',config_sha256='a'*64,
            original=dict(job_id='120900',method='FE_NATIVE_SELECTED_CURRENT_Z',checkpoint=dict(sha256='b'*64),identity={}),
            model_identity=dict(model='qwen25',model_revision='d'*40,tokenizer_sha256='c'*64),
            evaluator=dict(sha256='e'*64),stream=dict(sha256='f'*64),reference_identity=f.REFERENCE,
            final_result=dict(sha256='0'*64))
    def test_actual_shared_generation_query_unicode_identity_and_independent_scores(self):
        c=self.config();lock=dict(source_commit='1'*40,official_tree='2'*40)
        assets=FakeAssets();assets.sha=f.REFERENCE
        rows=[record(1,prompts=[prompt(98),prompt(99)]),record(2,501,prompts=[prompt(99)])]
        # Non-ASCII reference identity tests the generation digest, not ASCII config digest.
        rows[1]['requested_rewrite']['relation_id']='référence'
        with tempfile.TemporaryDirectory() as d:
            model=Model(stochastic=True);tok=Tokenizer();before=rng_snapshot()
            observer=NativeGenerationObserver(model,tok,assets,f.generation_config(c,lock),Path(d)/'raw',lambda:{'W':'unchanged'})
            actual=observer.observe(rows,'W20','first2000',f.endpoint_binding(c,lock))
            checked=read_observed(actual['rows_path'],assets=assets)
            with patch.object(f,'COUNT',2):f.verify_generation(checked,rows,c,lock)
            self.assertTrue(rng_equal(before));self.assertEqual(actual['summary'],checked['summary'])
            self.assertEqual(len(model.calls),3);self.assertEqual(checked['summary']['fluency_count'],2)
            self.assertEqual(checked['summary']['consistency_count'],0)
            self.assertNotIn('reference_score',checked['summary'])
            self.assertIn('Flu_paper_x100',paper_generation(checked['summary']))
            for wrong in (list(reversed(rows)),[dict(rows[0],generation_prompts=[prompt(99)]),rows[1]]):
                with patch.object(f,'COUNT',2),self.assertRaisesRegex(ValueError,'EXACT_QUERY'):
                    f.verify_generation(checked,wrong,c,lock)
            with patch.object(f,'COUNT',2),self.assertRaisesRegex(ValueError,'EXACT_QUERY'):
                f.verify_generation(checked,rows,c,dict(lock,source_commit='3'*40))
    def test_tracking_actual_rent_identity_generation_only_axes(self):
        with patch.dict(os.environ,{},clear=True):
            values=bind_job_identity(f.tracking_values(self.config(),'1'*40),{'ODEEDIT_RENT_JOB_ID':'120901'})
        self.assertEqual(values['role'],'eval_only');self.assertEqual(values['task_id'],f.TASK)
        self.assertEqual(values['source_run_id'],'120900');self.assertEqual(values['job_id'],'120901')
        self.assertEqual(values['generation_eval_seed'],20261007)
        payload={'all_seen/post/generation/planned_count':2000,'edits':2000,'post_state_edits':2000}
        # Use the common actual metric mapping, not guessed namespace.
        summary=dict(planned_count=2000,fluency_count=2000,consistency_count=2000,
            ngram_entropy=5.,reference_score=.2,fluency_unit='bits',consistency_unit='cosine_0_to_1',
            generation_prompt_count=20000,generated_token_count=1000000,
            missing_reason_counts={k:0 for k in PUBLIC_REASONS})
        payload=generation_payload('all_seen/post',summary);payload.update(edits=2000,post_state_edits=2000)
        metrics(payload,scientific=True,config_values=values)
        for bad in ({'fit/loss':1.},{'official/all_seen/post/Efficacy':90.},dict(payload,edits=0)):
            with self.assertRaises(ValueError):metrics(bad,scientific=True,config_values=values)
    def test_storage_no_reserve_reduction(self):
        with tempfile.TemporaryDirectory() as d,self.assertRaisesRegex(ValueError,'32GIB'):
            prep.storage_admission(d,1024,4*1024**3)


if __name__=='__main__':unittest.main()
