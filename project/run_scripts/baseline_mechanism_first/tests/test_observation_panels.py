import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from project.run_scripts.baseline_mechanism_first.contracts import digest
from project.run_scripts.baseline_mechanism_first.evaluation import bind_evaluation_sources,diagnostic_layout
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng
from project.run_scripts.baseline_mechanism_first.observation_panels import (
    evaluate_general,per_position_energy,position_tags,run_observations,signed_inventory,temporary_weight)
from project.run_scripts.baseline_mechanism_first.tests.test_evaluation import HISTORY,Toy,Tokenizer,record


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): bind_evaluation_sources(HISTORY)

    def test_tagging_masks_last_prefix_not_continuation(self):
        batch=diagnostic_layout(Tokenizer(),record())
        tags=position_tags(batch)
        first=int(batch['target_prediction_mask'][0].nonzero()[0])
        self.assertEqual(tags[0][first],'PREFIX_SUBJECT_UNRESOLVED')
        self.assertEqual(tags[0][first+1],'CONTINUATION_INPUT')
        self.assertEqual(tags[1][0],'PADDING')

    def test_dense_factor_position_energy_oracle(self):
        q=torch.arange(24,dtype=torch.float32).reshape(2,3,4)/10
        k=torch.arange(8,dtype=torch.float32).reshape(4,2)/7
        f=k/3; delta=torch.arange(20,dtype=torch.float32).reshape(5,4)/8
        out=per_position_energy(q,k,f,delta)
        oracle=float((delta.double()@q[1,2].double()).square().sum())
        gamma=q.shape[-1]*torch.finfo(torch.float32).eps
        self.assertLessEqual(abs(out['actual_delta_q_sq'][1][2]-oracle),2*gamma*oracle)
        self.assertEqual(len(out['raw_Ktq_sq']),2)

    def test_temporary_storage_restores_pointer_version_on_exception(self):
        model=Toy();weight=model.write.weight
        ptr,version=weight.data_ptr(),weight._version
        before=weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError,'forced'):
            with temporary_weight(weight,before+1):
                self.assertFalse(torch.equal(weight,before))
                raise RuntimeError('forced')
        self.assertEqual((weight.data_ptr(),weight._version),(ptr,version))
        self.assertTrue(torch.equal(weight,before))

    def test_full_observation_raw_rows_rng_and_flags(self):
        model=Toy();tok=Tokenizer();weight=model.write.weight
        weight.requires_grad_(False)
        before={n:(p.data_ptr(),p._version,p.requires_grad,p.grad,p.detach().clone()) for n,p in model.named_parameters()}
        rng=capture_rng()
        rows=[dict(ordinal=0,input_ids=[1,4,5],predicted_tokens=2)]
        manifest=dict(rows=rows,row_identity=digest(rows))
        entry=weight.detach().clone();endpoint=entry+0.02
        delta=endpoint-entry;k=torch.ones(5,2);f=k/3
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'observation'
            receipt=run_observations(model,tok,weight,'write',[record()],[0],[],manifest,
                       entry,endpoint,delta,k,f,output,dict(layer=4,entry_n=0),w0_weight=entry)
            self.assertEqual(receipt['status'],'OBSERVATIONS_FINITE')
            self.assertEqual(receipt['forward_pairs'],14)
            self.assertEqual(receipt['general_forward_sequences'],4)
            self.assertEqual(receipt['selected_version_change'],0)
            self.assertTrue(receipt['own_input_stability']['equal'])
            with gzip.open(output/'query-positions.jsonl.gz','rt') as handle:
                observations=[json.loads(line) for line in handle]
            self.assertEqual(len(observations),receipt['raw_position_rows'])
            self.assertEqual({r['category'] for r in observations},{'R','P','N','GENERAL'})
            self.assertEqual({r['side'] for r in observations},{'new','true','text'})
            self.assertEqual(json.loads((output/'general-nll.json').read_text())['W0'][0]['state'],'W0')
        self.assertEqual(rng,capture_rng())
        for name,p in model.named_parameters():
            ptr,version,flag,grad,value=before[name]
            self.assertEqual((p.data_ptr(),p._version,p.requires_grad),(ptr,version,flag))
            self.assertIs(p.grad,grad);self.assertTrue(torch.equal(p,value))

    def test_hash_signed_inventory_unchanged_by_outcomes(self):
        records=[record(i) for i in range(228)]
        rows=[dict(ordinal=i,input_ids=[1,4]) for i in range(128)]
        first=signed_inventory(records,list(range(100)),list(range(100,228)),rows)
        for rec in records: rec['outcome']=-999
        second=signed_inventory(records,list(range(100)),list(range(100,228)),rows)
        self.assertEqual(first,second)
        self.assertEqual(len(first['pairs']),160)
        self.assertEqual(len(first['general_ordinals']),16)

    def test_signed_dense_factor_and_failure_restore(self):
        model=Toy();tok=Tokenizer();weight=model.write.weight
        before=weight.detach().clone();delta=torch.full_like(before,.01)
        records=[record(i) for i in range(64)]
        rows=[dict(ordinal=i,input_ids=[1,4,5]) for i in range(16)]
        manifest=dict(rows=rows,row_identity=digest(rows))
        k=torch.ones(5,2);f=k/3;R=torch.ones(5,2)*.015
        rng=capture_rng()
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'signed'
            # Keep full signed selection, but omit query-only multiplication cost.
            result=run_observations(model,tok,weight,'write',records,list(range(32)),list(range(32,64)),
                      manifest,before,before+delta,delta,k,f,output,
                      dict(layer=4,entry_n=5000,signed_enabled=True,
                           signed_panel_identity=digest(signed_inventory(records,list(range(32)),list(range(32,64)),rows))),diagnostic_R=R)
            self.assertEqual(result['signed_backward_pairs'],160)
            self.assertEqual(result['general_backward_sequences'],16)
            with gzip.open(output/'signed-response.jsonl.gz','rt') as handle:
                signed=[json.loads(line) for line in handle]
            self.assertTrue(all('diagnostic_factor_derivative' in row for row in signed))
            self.assertTrue(all(row['direction_source']=='ACTUAL_ENDPOINT_DIFFERENCE_FP32' for row in signed))
            self.assertEqual(len(signed),176)
            failed=Path(folder)/'failed'
            with patch.object(model,'forward',side_effect=RuntimeError('forced forward')):
                with self.assertRaisesRegex(RuntimeError,'forced forward'):
                    run_observations(model,tok,weight,'write',[record()],[0],[],manifest,
                        before,before+delta,delta,k,f,failed,dict(layer=4,entry_n=0))
            receipt=json.loads((failed/'observation-receipt.json').read_text())
            self.assertEqual(receipt['status'],'TECHNICAL_EXCEPTION')
            self.assertTrue(receipt['selected_bytes_restored'])
            self.assertTrue(receipt['rng_restored'])
        self.assertTrue(torch.equal(weight,before))
        self.assertEqual(rng,capture_rng())


if __name__=='__main__':unittest.main()
