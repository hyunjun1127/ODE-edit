"""Tiny CPU fixtures; no actual final proof, model forward or GPU evidence."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from official.experiments.prepare import digest,write_new
from official.runners.server1.common import member
from official.diagnostics.fe_qwen_current import flucon_clamp as c,flucon as f
from official.diagnostics.fe_qwen_current import test_flucon as old


class ClampFluconTests(unittest.TestCase):
    def fixture(self,d):
        x=old.Fixture(d)
        x.identity.update(selection_contract_sha256=digest(c.SPEC),clamp_norm_factor=.5,
            author_hparam_profile=c.SPEC['protocol'])
        x.payload['cursor']['native_config']['llms']['clamp_norm_factor']=.5
        return x

    def test_missing_both_proof_fails_before_mmap_or_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_new(root/'result.json',dict(both_W20_complete=False))
            write_new(root/'bindings.json',dict(final_result=member(root/'result.json')))
            with patch.object(torch,'load',side_effect=AssertionError('no mmap')):
                with self.assertRaisesRegex(ValueError,'BOTH_FINAL_W20_REQUIRED'):
                    c.prepare(root/'new',root/'bindings.json')
            self.assertFalse((root/'new').exists())

    def test_new_mmap_restore_exact_unselected_rng_and_old_gate_unchanged(self):
        before_spec=copy.deepcopy(f.SPEC);validator=f.validate_payload
        with tempfile.TemporaryDirectory() as d:
            x=self.fixture(d);original=x.save();model=old.Tiny()
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            rng=torch.get_rng_state().clone();cp=member(Path(d)/'latest.pt')
            with c.bindings(),patch.object(f,'HP',old.HP),patch.object(f,'SELECTED_SHAPE',(3,4)),patch.object(f,'ORDER',digest(x.ids)),\
                 patch('official.diagnostics.fe_qwen_current.storage.state_digest',side_effect=AssertionError('no H hash')):
                result=f.restore(model,original)
            for n,p in model.named_parameters():
                self.assertTrue(torch.equal(p,x.weights[n].float() if n in x.weights else before[n]))
                if n in x.weights:self.assertTrue(torch.equal(p.bfloat16(),x.weights[n]))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(cp,member(Path(d)/'latest.pt'))
            self.assertFalse(result['history_contexts_rng_consumed'])
            with patch.object(f,'HP',old.HP),patch.object(f,'ORDER',digest(x.ids)),self.assertRaises(ValueError):
                f.validate_payload(x.payload,original)
        self.assertEqual(f.SPEC,before_spec);self.assertIs(f.validate_payload,validator)

    def test_resigned_wrong_new_payloads_rejected(self):
        for kind in ('old_contract','old_clamp','identity_clamp','L2','partial','heldout','dtype','order','weight'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                x=self.fixture(d);p=x.payload
                if kind=='old_contract':x.identity['selection_contract_sha256']=digest(f.SPEC)
                if kind=='old_clamp':p['cursor']['native_config']['llms']['clamp_norm_factor']=1
                if kind=='identity_clamp':x.identity['clamp_norm_factor']=1
                if kind=='L2':p['cursor']['native_config']['algs']['L2']=10
                if kind=='partial':p['batch']=19
                if kind=='heldout':x.identity['stage']='heldout_selection'
                if kind=='dtype':p['cursor']['native_config']['model_dtype']='float32'
                if kind=='order':p['cursor']['case_ids']=list(reversed(x.ids))
                if kind=='weight':p['weights'][next(iter(x.weights))]=torch.zeros(3,4)
                original=x.save()
                with c.bindings(),patch.object(f,'HP',old.HP),patch.object(f,'SELECTED_SHAPE',(3,4)),patch.object(f,'ORDER',digest(x.ids)),self.assertRaises(ValueError):
                    f.payload_weights(original)

    def test_selected_proof_uses_new_independent_collector(self):
        with c.bindings():
            self.assertIs(f.validated_selection,c.rescue.validated_selection)
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'selection-result.json'
                write_new(path,dict(selection=dict(status='SELECTED',selected_L2=100,selected_clamp_norm_factor=1)))
                with self.assertRaises((ValueError,KeyError,FileNotFoundError)):
                    f.validated_selection(member(path))

    def test_shared_generation_query_reduction_and_schema(self):
        # Actual canonical tiny observer + scorer, not another implementation.
        with c.bindings():
            t=old.FluconTests()
            t.test_actual_shared_generation_query_unicode_identity_and_independent_scores()
            t.test_tracking_actual_rent_identity_generation_only_axes()
            values=f.tracking_values(t.config(),'1'*40)
            self.assertEqual(values['task_id'],c.TASK)
            self.assertIn('clamp-0p5',values['baseline'])

    def test_both_final_runtime_cp_clamps_and_failed_receipt(self):
        # Isolate the new metadata layer; unchanged base raw auditor tested above.
        for bad in (None,'cf','zsre','failed'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as d:
                root=Path(d);rows=[]
                for dataset in ('cf','zsre'):
                    out=root/dataset;out.mkdir()
                    identity=dict(L2=100,clamp_norm_factor=.5,author_hparam_profile=c.SPEC['protocol'],
                        selection_contract_sha256=digest(c.SPEC))
                    native=dict(llms=dict(clamp_norm_factor=1 if bad==dataset else .5),algs=dict(L2=100))
                    config=dict(identity,output=str(out))
                    write_new(out/'config.json',config)
                    write_new(out/'COMPLETE.json',dict(identity=identity))
                    write_new(out/'runtime.json',dict(identity=identity,resolved=native))
                    torch.save(dict(identity=identity,batch=20,cursor=dict(edits=2000,native_config=native)),out/'latest.pt')
                    if bad=='failed' and dataset=='zsre':write_new(out/'QUALITY_EARLY_STOP.json',dict(status='QUALITY_EARLY_STOP'))
                    rows.append(dict(config=member(out/'config.json'),checkpoint=member(out/'latest.pt')))
                write_new(root/'final-result.json',dict(rows=rows))
                with patch.object(c,'_audit',return_value={'already_independently_audited':True}):
                    if bad is None:self.assertTrue(c.audit_final(member(root/'final-result.json')))
                    else:
                        with self.assertRaises(ValueError):c.audit_final(member(root/'final-result.json'))


if __name__=='__main__':unittest.main()
