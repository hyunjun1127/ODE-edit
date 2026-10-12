"""Small CPU fixtures only; no real final checkpoint or performance claim."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from official.diagnostics.fe_qwen_current import author_basic_eot_flucon as a
from official.diagnostics.fe_qwen_current import author_basic_eot_benchmark as author
from official.diagnostics.fe_qwen_current import flucon as base,test_flucon as old
from official.experiments.prepare import digest,write_new
from official.runners.server1.common import member,read


class Tests(unittest.TestCase):
    def fixture(self,folder):
        x=old.Fixture(folder);root=Path(folder)
        yaml_root=Path('/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author/configs')
        if not yaml_root.exists():self.skipTest('pinned author YAML unavailable')
        yamls=dict(algs=member(yaml_root/'algs/memit.yaml'),llms=member(yaml_root/'llms/qwen2.5-7b.yaml'))
        profiles=author.author_profiles(dict(author_yaml=yamls))
        historical=dict(model_dtype='bfloat16',bs=100,num_edits=2000,seed=0,cov_mode='default',lti=False,negetive_prompt_test=False,**copy.deepcopy(profiles));historical['algs']['L2']=10
        write_new(root/'old-runtime.json',dict(resolved=historical))
        b=dict(runtime=member(root/'old-runtime.json'),author_yaml=yamls,main_stream=dict(sha256=base.STREAM))
        _,audit=author.execution_config(historical,b);write_new(root/'bindings.json',b)
        x.identity.clear();x.identity.update(stage='AUTHOR_BASIC_EOT_MAIN',dataset='cf',L2=0,clamp_norm_factor=1.,no_mask=True,
            input_policy=author.common.POLICY,z_reference='batch_entry',author_commit=author.AUTHOR,
            author_settings_sha256=audit['resolved_sha256'],config_sha256=digest(b),stream=b['main_stream'],adapter=member(a.eot.__file__))
        x.payload['cursor'].update(native_config=audit['resolved'],input_adapter=member(a.eot.__file__))
        x.payload['z_member']=dict(policy='BASIC_EOT_CURRENT_ENTRY',durable_target_cache=False)
        x.original['config']=member(root/'bindings.json')
        return x

    def test_missing_partial_old_proofs_block_before_model_or_mmap(self):
        for status,complete in [('BOTH_AUTHOR_BASIC_EOT_W20_CPU_AUDITED',False),('BOTH_FINAL_W20_CPU_AUDITED',True),('QUALITY_EARLY_STOP',False)]:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);write_new(root/'final.json',dict(status=status,both_W20_complete=complete))
                write_new(root/'inputs.json',dict(final_result=member(root/'final.json')))
                with patch.object(torch,'load',side_effect=AssertionError('must not mmap')):
                    with self.assertRaisesRegex(ValueError,'BOTH_AUTHOR_FINAL_W20_REQUIRED'):a.prepare(root/'new',root/'inputs.json')
                self.assertFalse((root/'new').exists())

    def test_selected_only_mmap_exact_restore_rng_unselected_and_old_gate(self):
        original_validator=base.validate_payload;original_audit=base.audit_final
        with tempfile.TemporaryDirectory() as d:
            x=self.fixture(d);o=x.save();model=old.Tiny();before={n:p.detach().clone() for n,p in model.named_parameters()}
            rng=torch.get_rng_state().clone();cp=member(Path(d)/'latest.pt')
            with a.bindings(),patch.object(base,'HP',old.HP),patch.object(base,'SELECTED_SHAPE',(3,4)),patch.object(base,'ORDER',digest(x.ids)),\
                 patch('official.diagnostics.fe_qwen_current.storage.state_digest',side_effect=AssertionError('generation must not hash H')):
                receipt=base.restore(model,o)
            for n,p in model.named_parameters():
                self.assertTrue(torch.equal(p,x.weights[n].float() if n in x.weights else before[n]))
                if n in x.weights:self.assertTrue(torch.equal(p.bfloat16(),x.weights[n]))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(cp,member(Path(d)/'latest.pt'))
            self.assertFalse(receipt['history_contexts_rng_consumed'])
        self.assertIs(base.validate_payload,original_validator);self.assertIs(base.audit_final,original_audit)

    def test_resigned_wrong_author_cp_rejected(self):
        for kind in ('L210','clamp','heldout','zsre','partial','config','order','policy','dtype','adapter'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                x=self.fixture(d);p=x.payload
                if kind=='L210':x.identity['L2']=10
                if kind=='clamp':x.identity['clamp_norm_factor']=.5
                if kind=='heldout':x.identity['stage']='BASIC_EOT_HELDOUT_B5_CONTINUATION'
                if kind=='zsre':x.identity['dataset']='zsre'
                if kind=='partial':p['batch']=15
                if kind=='config':p['cursor']['native_config']['algs']['L2']=10
                if kind=='order':p['cursor']['case_ids']=list(reversed(x.ids))
                if kind=='policy':p['z_member']['policy']='W0'
                if kind=='dtype':p['weights'][next(iter(x.weights))]=torch.zeros(3,4)
                if kind=='adapter':p['cursor']['input_adapter']['sha256']='0'*64
                o=x.save()
                with a.bindings(),patch.object(base,'HP',old.HP),patch.object(base,'SELECTED_SHAPE',(3,4)),patch.object(base,'ORDER',digest(x.ids)):
                    with self.assertRaises(ValueError):base.payload_weights(o)

    def test_actual_author_collector_is_called_and_summary_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=[]
            for dataset in ('cf','zsre'):
                p=root/f'{dataset}.json';write_new(p,dict(fixture=True));rows.append(dict(dataset=dataset,result=member(p)))
            final=dict(status='BOTH_AUTHOR_BASIC_EOT_W20_CPU_AUDITED',both_W20_complete=True,results=rows)
            write_new(root/'final.json',final)
            def collect(cf,zsre,out):
                self.assertEqual(Path(cf),root/'cf.json');self.assertEqual(Path(zsre),root/'zsre.json')
                write_new(out,dict(final,results=[]));return member(out)
            with patch.object(author,'collect',side_effect=collect) as called:
                with self.assertRaisesRegex(ValueError,'INDEPENDENT_REAUDIT'):a.audit_final(member(root/'final.json'))
                self.assertEqual(called.call_count,1)

    def test_exact_reference_manifest_rejection_before_loader(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'wrong.json';write_new(p,dict(schema='not-original'))
            with patch.object(a,'_load_reference',side_effect=AssertionError('must not load')):
                with self.assertRaisesRegex(ValueError,'REFERENCE_MANIFEST'):a.load_reference(p)

    def test_shared_canonical_generation_reduction_schema_and_scope_restore(self):
        original_task=base.TASK
        with a.bindings():
            t=old.FluconTests();t.test_actual_shared_generation_query_unicode_identity_and_independent_scores()
            t.test_tracking_actual_rent_identity_generation_only_axes()
            values=base.tracking_values(t.config(),'1'*40)
            self.assertEqual(values['task_id'],a.TASK);self.assertIn('L2-0',values['arm'])
            self.assertEqual(values['role'],'eval_only');self.assertEqual(values['generation_eval_seed'],20261007)
        self.assertEqual(base.TASK,original_task)


if __name__=='__main__':unittest.main()
