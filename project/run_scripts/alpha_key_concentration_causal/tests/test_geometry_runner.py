"""CPU runtime fixtures; no model/Slurm/GPU activity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from project.run_scripts.alpha_key_concentration_causal.geometry import GeometryError
from project.run_scripts.alpha_key_concentration_causal.geometry_runner import (
    COHORTS, _Output, _gate_up, _restore_lower, _upper_control, project_keys, run_geometry, validate_panels,
)


def panels_fixture():
    panels, byid = {'cohorts': {}}, {}
    for index, name in enumerate(COHORTS):
        ids = list(range(index * 1000, (index + 1) * 1000))
        panels['cohorts'][name] = {'case_ids': ids, 'calibration_case_ids': ids[:128], 'assessment_case_ids': ids[128:]}
        byid.update({case_id: {'case_id': case_id} for case_id in ids})
    return panels, byid


class FakeRuntime:
    def __init__(self, break_upper=False):
        self.weights = {layer: torch.zeros((2, 2)) for layer in range(4, 9)}
        self.base_weights = {layer: torch.zeros((2, 2)) for layer in range(4, 9)}
        self.P = torch.eye(2).repeat(5, 1, 1)
        self.states, self.captures = [], []
        self.break_upper = break_upper

    def set_state(self, state):
        self.states.append(state)
        for layer in self.weights:
            self.weights[layer].fill_(float(state))

    def capture(self, records, *, contexts, features, full):
        assert contexts and not full
        self.captures.append([r['case_id'] for r in records])
        keys, means, result_features = {}, {}, {}
        for layer in self.weights:
            upstream = sum(float(self.weights[i][0, 0]) for i in range(4, layer))
            if self.break_upper and layer == 6:
                upstream += float(self.weights[8][0, 0])
            arr = torch.ones((len(records), 6, 2)) * (upstream + 1)
            keys[layer] = arr
            means[layer] = arr[:, 0].clone()
            if features:
                result_features[f'L{layer}/gate'] = arr.clone()
                result_features[f'L{layer}/up'] = arr.clone()
        return {'keys': keys, 'means': means, 'features': result_features,
                'token_receipt': {'fixture': True, 'case_ids': [r['case_id'] for r in records]}, 'seconds': 0.0}


class RunnerTests(unittest.TestCase):
    def test_actual_runtime_feature_names(self):
        gate = np.ones((2, 6, 3), dtype=np.float32)
        features = {'L4/gate_preactivation': gate, 'L4/up_projection': gate * 2}
        np.testing.assert_array_equal(_gate_up(features, 4, 'gate'), gate)
        np.testing.assert_array_equal(_gate_up(features, 4, 'up'), gate * 2)
        features['L4/gate'] = gate * 3
        with self.assertRaises(GeometryError):
            _gate_up(features, 4, 'gate')

    def test_panel_integrity(self):
        panels, byid = panels_fixture()
        result = validate_panels(panels, byid)
        self.assertEqual(len(result['calibration_ids']), 512)
        panels['cohorts']['early']['assessment_case_ids'][0] = 0
        with self.assertRaises(GeometryError):
            validate_panels(panels, byid)

    def test_projection_matches_direct(self):
        rng = np.random.default_rng(3)
        keys = rng.normal(size=(7, 6, 5)).astype(np.float32)
        P = rng.normal(size=(5, 5)).astype(np.float32)
        actual = project_keys(keys, P, block_rows=4)
        np.testing.assert_allclose(actual, (keys.reshape(-1, 5) @ P.T).reshape(keys.shape), rtol=1e-6)
        with self.assertRaises(GeometryError):
            project_keys(keys.astype(np.float64), P)

    def test_subset_restores_only_allowed_lower_layers(self):
        rt = FakeRuntime()
        rt.set_state(80)
        _restore_lower(rt, (0, 1, 0))
        self.assertEqual(float(rt.weights[4][0, 0]), 0)
        self.assertEqual(float(rt.weights[5][0, 0]), 80)
        self.assertEqual(float(rt.weights[6][0, 0]), 0)
        self.assertEqual(float(rt.weights[7][0, 0]), 80)

    def test_actual_analyzer_serialization_and_paired_reload(self):
        _, byid = panels_fixture()
        ids = [0, 1, 2]
        rt = FakeRuntime()
        with tempfile.TemporaryDirectory() as temp:
            output = _Output(rt, temp)
            rt.set_state(0)
            initial = rt.capture([byid[i] for i in ids], contexts=True, features=True, full=False)
            output.analyze(initial, ids, phase='E1', state=0, condition='native', panel='fixture', save_context=True, features=True)
            rt.set_state(10)
            updated = rt.capture([byid[i] for i in ids], contexts=True, features=False, full=False)
            output.analyze(updated, ids, phase='E1', state=10, condition='native', panel='fixture', save_context=True)
            saved = torch.load(Path(temp) / 'E1/W000/native/fixture/L4-keys.pt', weights_only=True)
            self.assertEqual(saved['case_ids'], ids)
            self.assertEqual(saved['context_keys'].shape, (3, 6, 2))
            paired = json.loads((Path(temp) / 'E1/W010/native/fixture/L5-paired-W0.json').read_text())
            self.assertEqual(paired['raw']['writer_mean']['identity'], 'EXACT_ORDERED_IDS')
            self.assertEqual(len(output.rows), 2 * 5 * 7 * 2)
            self.assertFalse(any('weights' in torch.load(Path(temp) / 'E1/W000/native/fixture' / f'L{l}-keys.pt', weights_only=True) for l in range(4, 9)))
            receipt = output.close()
            self.assertEqual(receipt['rows'], 2 * 5 * 7 * 2 * 3)
            if receipt['format_status'] == 'AVAILABLE':
                import pyarrow.parquet as pq
                table = pq.read_table(receipt['path'])
                self.assertEqual(table.num_rows, receipt['rows'])
                self.assertTrue(set(table['case_id'].to_pylist()) == set(ids))

    def test_upper_control_detects_wrong_dependency(self):
        _, byid = panels_fixture()
        for broken in (False, True):
            rt = FakeRuntime(break_upper=broken)
            rt.set_state(80)
            baseline = rt.capture([byid[0]], contexts=True, features=False, full=False)
            with tempfile.TemporaryDirectory() as temp:
                output = _Output(rt, temp)
                if broken:
                    with self.assertRaises(GeometryError):
                        _upper_control(rt, baseline, [0], byid, output, 80)
                else:
                    _upper_control(rt, baseline, [0], byid, output, 80)
                output.close()

    def test_all_scoped_cells_and_final_restore(self):
        panels, byid = panels_fixture()
        rt = FakeRuntime()
        # Test orchestration while spectrum algebra is separately tested. No
        # tiny-subset flag is added to the scientific runner.
        def fake_analyze(output, capture, ids, **meta):
            output.cells.append(dict(meta, n=len(ids)))
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(_Output, 'analyze', fake_analyze), \
             patch.object(_Output, '_features'), \
             patch('project.run_scripts.alpha_key_concentration_causal.geometry_runner._cross_gate_up'):
            result = run_geometry(rt, panels, byid, Path(temp) / 'output')
            self.assertEqual(result['status'], 'COMPLETED')
            self.assertEqual(result['completed_panel_observations'], 28 + 18 * 5)
            self.assertEqual(len(rt.captures), 28 + 1 + 18 * 5)
            self.assertEqual(rt.states[-1], 0)
            self.assertEqual(len(rt.states), 7 + 1 + 16 + 2 + 1)
            self.assertFalse(result['new_checkpoint_saved'])
            self.assertTrue((Path(temp) / 'output' / 'terminal.json').exists())

    def test_failure_restores_without_fabricating_completion(self):
        panels, byid = panels_fixture()
        rt = FakeRuntime()
        with tempfile.TemporaryDirectory() as temp, patch.object(rt, 'capture', side_effect=RuntimeError('fixture capture failure')):
            target = Path(temp) / 'out'
            with self.assertRaisesRegex(RuntimeError, 'fixture capture failure'):
                run_geometry(rt, panels, byid, target)
            self.assertEqual(rt.states[-1], 0)
            self.assertFalse((target / 'terminal.json').exists())
            self.assertEqual(json.loads((target / 'failure.json').read_text())['type'], 'RuntimeError')


if __name__ == '__main__':
    unittest.main()
