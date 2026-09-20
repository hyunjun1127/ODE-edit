"""CPU integration of real tiny-Llama history with a known reference block."""
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from project.run_scripts.en_adapt_gss_history import factors
from project.run_scripts.en_adapt_gss_history.objective import Objective
from project.run_scripts.en_adapt_gss_history.tests.test_factors import (
    tiny_model, canonical_contract, versions,
)
from project.run_scripts.en_adaptive_nullspace.json_io import save


class KnownReference:
    """A finite quadratic with exact gradient; model evaluation stays read-only."""
    def __init__(self, model, *unused):
        self.model = model
        weight = dict(model.named_parameters())['model.layers.4.mlp.down_proj.weight']
        generator = torch.Generator(device='cpu').manual_seed(701)
        self.anchor = weight.detach().cpu().double()-torch.randn(weight.shape,generator=generator).double()*.05
        self.store = types.SimpleNamespace(receipt=dict(manifest_sha256='fixture-reference'))
        self.counts = dict(reference_gradient=0,reference_candidate=0)
        self.rebinds = 0

    def rebind(self, expected):
        weight = dict(self.model.named_parameters())['model.layers.4.mlp.down_proj.weight']
        if not torch.equal(weight.detach().cpu(),expected.detach().cpu()):
            raise ValueError('EXPECTED_NATIVE_WEIGHT_DIFFERS')
        self.rebinds += 1

    def evaluate(self, weight, *, gradient=False, kind='candidate'):
        difference = weight.detach().cpu().double()-self.anchor
        value = float(.5*difference.square().sum())
        self.counts['reference_gradient' if gradient else 'reference_candidate'] += 1
        return value, difference.clone() if gradient else None, dict(
            L_R=value,L_H=0.,J=value,kind=kind,gradient=bool(gradient),
            reference_documents=512,reference_positions=130235,
            fixture=True,precision='NOT_ESTABLISHED')


class ObjectiveIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_objective(self, model, root):
        with patch('project.run_scripts.en_adapt_gss_history.objective.ReferenceObjective',KnownReference):
            return Objective(model,object(),'unused-reference','unused-inputs',root,
                             np.eye(16,dtype=np.float64))

    def test_combined_blocks_and_cross_terms_use_actual_sum(self):
        model = tiny_model()
        weight = dict(model.named_parameters())['model.layers.4.mlp.down_proj.weight']
        with tempfile.TemporaryDirectory() as temporary, patch.object(factors,'_token_contracts',canonical_contract):
            objective = self.make_objective(model,temporary)
            rows = versions(4)
            captured = objective.capture(rows,weight.detach())
            rows = [dict(row,teacher_binding=captured['bindings'][row['version_id']]) for row in rows]
            with torch.no_grad():
                weight.add_(torch.randn_like(weight)*.035)
            objective.rebind(weight.detach())
            native = weight.detach().cpu().clone()
            combined,receipt,bank = objective.prepare(native,rows,arm='EN_ADAPT_H_RES',batch=4)
            reference_gradient = native.double()-objective.reference.anchor
            history_gradient = bank['gradient']
            torch.testing.assert_close(combined,reference_gradient+history_gradient,rtol=0,atol=0)
            self.assertAlmostEqual(receipt['J'],receipt['L_R']+receipt['L_H'],places=15)
            cross = float(2*(reference_gradient*history_gradient).sum())
            self.assertGreater(abs(cross),1e-12)
            self.assertAlmostEqual(float(combined.square().sum()),
                float(reference_gradient.square().sum()+history_gradient.square().sum())+cross,places=14)
            self.assertEqual(objective.reference.counts['reference_gradient'],1)
            self.assertEqual(bank['receipt']['NLL_VJPs'],0)
            self.assertEqual(len(bank['selected_ids']),4)
            save(Path(temporary)/'native-combined.json',receipt)
            save(Path(temporary)/'captured-teachers.json',captured)
            self.assertEqual(json.loads((Path(temporary)/'native-combined.json').read_text())['bank_identity'],objective.bank_identity)

    def test_two_cached_candidates_keep_native_physical_weight_and_bank(self):
        model = tiny_model()
        parameter = dict(model.named_parameters())['model.layers.4.mlp.down_proj.weight']
        with tempfile.TemporaryDirectory() as temporary, patch.object(factors,'_token_contracts',canonical_contract):
            objective = self.make_objective(model,temporary)
            rows = versions(3)
            objective.capture(rows,parameter.detach())
            with torch.no_grad():
                parameter.add_(torch.randn_like(parameter)*.025)
            objective.rebind(parameter.detach())
            native = parameter.detach().cpu().clone()
            version = parameter._version
            with patch.object(factors,'CAPACITY',2),patch.object(factors,'POOL_MAX',3):
                _,native_receipt,bank = objective.prepare(native,rows,arm='EN_ADAPT_H_GSS_REC',batch=4)
                self.assertEqual(bank['receipt']['NLL_VJPs'],3)
                selected = tuple(objective.selected_ids)
                weights = tuple(objective.weights)
                bank_identity = objective.bank_identity
                counts = dict(objective.replay.counts)
                for index,change in enumerate((.004,-.006)):
                    candidate = native+change
                    result = objective.evaluate(candidate)
                    self.assertEqual(result['bank_identity'],bank_identity)
                    self.assertEqual(tuple(result['history_ids']),selected)
                    self.assertEqual(tuple(result['history_weights']),weights)
                    self.assertEqual(result['history']['candidate_fact_forwards'],2)
                    self.assertAlmostEqual(result['J'],result['L_R']+result['L_H'],places=15)
                    self.assertTrue(torch.equal(parameter.detach().cpu(),native))
                    self.assertEqual(parameter._version,version)
                    save(Path(temporary)/f'candidate-{index}.json',result)
                self.assertEqual(objective.reference.counts['reference_gradient'],1)
                self.assertEqual(objective.reference.counts['reference_candidate'],2)
                self.assertEqual(objective.replay.counts['NLL_VJPs'],counts['NLL_VJPs'])
                self.assertEqual(objective.replay.counts['KL_VJPs'],counts['KL_VJPs'])
                self.assertEqual(objective.replay.counts['pool_forwards'],counts['pool_forwards'])
                self.assertEqual(objective.replay.counts['candidate_forwards']-counts['candidate_forwards'],4)
                self.assertEqual(len(objective.history_sweeps),3)
                save(Path(temporary)/'all-history-sweeps.json',objective.history_sweeps)
                save(Path(temporary)/'native-overflow.json',native_receipt)

    def test_empty_b1_matches_reference_without_history_work(self):
        model = tiny_model()
        parameter = dict(model.named_parameters())['model.layers.4.mlp.down_proj.weight']
        with tempfile.TemporaryDirectory() as temporary:
            objective = self.make_objective(model,temporary)
            native = parameter.detach().cpu().clone()
            gradient,receipt,bank = objective.prepare(native,[],arm='EN_ADAPT_H_RES',batch=1)
            torch.testing.assert_close(gradient,native.double()-objective.reference.anchor,atol=0,rtol=0)
            self.assertEqual(bank['gradient'].shape,gradient.shape)
            self.assertEqual(bank['gradient'].dtype,torch.float64)
            self.assertEqual(receipt['L_H'],0.)
            self.assertEqual(receipt['J'],receipt['L_R'])
            self.assertEqual(bank['receipt']['dense_gradient_D2H'],0)
            candidate = objective.evaluate(native+.001)
            self.assertEqual(candidate['L_H'],0.)
            self.assertEqual(candidate['J'],candidate['L_R'])
            for key in ('pool_forwards','NLL_VJPs','KL_VJPs','candidate_forwards','gradient_D2H'):
                self.assertEqual(objective.replay.counts[key],0)
            self.assertIsNone(objective.replay.oracle)
            self.assertIsNone(objective.replay.maps)
            save(Path(temporary)/'B1-native.json',receipt)
            save(Path(temporary)/'B1-candidate.json',candidate)

    def test_presealed_map_logical_bytes_precede_publication(self):
        model = tiny_model()
        basis = np.eye(16,dtype=np.float64)
        maps = factors.make_maps(12,16,basis)
        arrays = {f'replica_{i}_{kind}':array for i,triples in enumerate(maps)
                  for kind,array in zip(('output','input','Pstar_input'),triples)}
        seal = dict(arrays={name:dict(shape=list(array.shape),dtype=str(array.dtype),
                    sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())
                    for name,array in arrays.items()})
        with tempfile.TemporaryDirectory() as temporary:
            good = factors.HistoryReplay(model,object(),Path(temporary)/'good',basis,
                                         expected_map_seal=seal)
            good._initialize_maps()
            self.assertEqual(good.map_receipt['presealed_logical_verification'],'PASS')
            self.assertEqual(good.map_receipt['logical_arrays'],seal['arrays'])
            bad_seal = json.loads(json.dumps(seal))
            bad_seal['arrays']['replica_0_output']['sha256'] = '0'*64
            bad = factors.HistoryReplay(model,object(),Path(temporary)/'bad',basis,
                                        expected_map_seal=bad_seal)
            with self.assertRaisesRegex(ValueError,'PRESEALED_FIXED_MAP_LOGICAL_IDENTITY_MISMATCH'):
                bad._initialize_maps()
            self.assertFalse((Path(temporary)/'bad/fixed-maps.npz').exists())
            self.assertFalse((Path(temporary)/'bad/fixed-maps.json').exists())
            self.assertIsNone(bad.maps)


if __name__ == '__main__':
    unittest.main()
