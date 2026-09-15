"""CPU pure reductions and tiny fake-adapter I/O; not actual Llama/GPU FD."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from .repair_checks import (RepairCheckError, RepairNumerics, check_direct_route,
                            make_directions, reduce_fd_grid, run_directional_grid,
                            run_objective_fd)


def grid(analytic=1., baseline=1., slopes=None):
    rows = []
    for k in range(10):
        h = .1 * 2. ** -k
        slope = analytic if slopes is None else slopes[k]
        rows.append(dict(k=k, h=h, plus=baseline+h*slope, minus=baseline-h*slope,
            raw_sha256='raw', plus_weight_sha256=f'plus{k}', minus_weight_sha256=f'minus{k}',
            plus_actual_norm=h, minus_actual_norm=h,
            plus_actual_direction_dot=h, minus_actual_direction_dot=-h))
    return rows


class TinyAdapter:
    """Linear scalar model, only exercising save/guard/FD orchestration."""
    def __init__(self, fail_at=None):
        self.raw = torch.eye(2)*10
        self.fixed_a = torch.eye(2)
        self.device = self.raw.device
        self.calls = 0
        self.fail_at = fail_at

    def _assert_episode(self):
        pass

    def current(self, records, correction):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('SYNTHETIC_FORWARD_FAILURE')
        value = 1. + float(correction.sum())
        rows = [dict(case_id=i, kind='rewrite_target_new', prompt_index=0, prompt=f'q{i}',
                     target='t', target_token_ids=[1], nll=value) for i in range(100)]
        return dict(E=value, denominator=100, rows=rows,
            counts=dict(forwards=7, backwards=0, scored_tokens=100),
            parameter_hook_rng_nonmutation=True)

    def generic(self, role, correction):
        self.calls += 1
        value = 1. + float(correction.sum())
        rows = [dict(index=i, role='S64', source_row_id=f'row{i}', kl=value) for i in range(64)]
        return dict(D=value, denominator=64, rows=rows,
            counts=dict(forwards=64, backwards=0, scored_tokens=8192),
            parameter_hook_rng_nonmutation=True)


class RepairFixtures(unittest.TestCase):
    def reduce(self, rows, analytic=1., baseline=(1., 1., 1.)):
        return reduce_fd_grid(rows, analytic=analytic, baseline_values=list(baseline))

    def test_full_grid_selects_coarsest_allowed_adjacent_pair(self):
        result = self.reduce(grid())
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['selected_window']['k_start'], 1)
        self.assertEqual(result['selected_window']['k_stop'], 2)
        self.assertEqual(len(result['rows']), 10)
        self.assertEqual(result['signed_grid_observations'], 20)

    def test_coarse_failure_retained_later_local_window_valid(self):
        result = self.reduce(grid(slopes=[9.5, 8.2, 1., 1., 1., 1., 1., 1., 1., 1.]))
        self.assertEqual(result['status'], 'PASS')
        self.assertFalse(result['rows'][0]['derivative_pass'])
        self.assertFalse(result['rows'][1]['derivative_pass'])
        self.assertEqual(result['selected_window']['k_start'], 2)

    def test_isolated_matching_point_never_passes(self):
        slopes = [3.] * 10
        slopes[3] = 1.
        result = self.reduce(grid(slopes=slopes))
        self.assertEqual(result['status'], 'UNRESOLVED')

    def test_only_old_coarse_pair_not_a_new_local_window(self):
        result = self.reduce(grid(slopes=[1., 1.] + [3.] * 8))
        self.assertEqual(result['status'], 'UNRESOLVED')

    def test_convergence_denominator_is_ad_not_fd(self):
        result = self.reduce(grid(slopes=[5., 1.14, .86] + [5.] * 7))
        self.assertTrue(result['rows'][1]['derivative_pass'])
        self.assertTrue(result['rows'][2]['derivative_pass'])
        self.assertAlmostEqual(result['windows'][0]['convergence_relative'], .28)
        self.assertEqual(result['windows'][0]['convergence_denominator'], 1.)
        self.assertFalse(result['windows'][0]['valid'])

    def test_jitter_span_can_remove_apparent_signal_resolution(self):
        result = self.reduce(grid(), baseline=(.95, 1., 1.05))
        self.assertEqual(result['status'], 'UNRESOLVED')
        self.assertAlmostEqual(result['baseline_span'], .1)
        self.assertTrue(all(not x['signal_resolved'] for x in result['rows']))

    def test_baseline_three_observations_are_mandatory(self):
        with self.assertRaisesRegex(RepairCheckError, 'THREE_BASELINE'):
            reduce_fd_grid(grid(), analytic=1., baseline_values=[1., 1.])

    def test_near_zero_analytic_is_unresolved_not_automatic_pass(self):
        result = self.reduce(grid(analytic=1e-8, baseline=1e-4), analytic=1e-8,
                             baseline=(1e-4, 1e-4, 1e-4))
        self.assertEqual(result['status'], 'UNRESOLVED')
        self.assertEqual(result['reason'], 'NEAR_ZERO_AD_UNRESOLVED')

    def test_taylor_remainders_retain_sign_and_anchor(self):
        rows = grid()
        rows[1]['plus'] += .01
        rows[1]['minus'] += .02
        result = self.reduce(rows)
        self.assertAlmostEqual(result['rows'][1]['taylor_plus'], .01)
        self.assertAlmostEqual(result['rows'][1]['taylor_minus'], .02)

    def test_actual_duplicate_amplitude_not_new_resolution(self):
        rows = grid()
        for row in rows[1:]:
            row['plus_weight_sha256'], row['minus_weight_sha256'] = 'plus0', 'minus0'
        result = self.reduce(rows)
        self.assertEqual(result['status'], 'UNRESOLVED')
        self.assertFalse(result['rows'][1]['actual_amplitude_not_duplicate'])

    def test_actual_action_raw_alias_or_wrong_sign_unresolved(self):
        for modify in ('raw_alias', 'zero_action', 'wrong_sign', 'missing_raw'):
            rows = grid()
            for row in rows:
                if modify == 'raw_alias': row['plus_weight_sha256'] = 'raw'
                if modify == 'zero_action': row['minus_actual_norm'] = 0.
                if modify == 'wrong_sign': row['minus_actual_direction_dot'] = 1.
                if modify == 'missing_raw': del row['raw_sha256']
            self.assertEqual(self.reduce(rows)['status'], 'UNRESOLVED')

    def test_full_grid_and_interval_contract_cannot_be_relaxed(self):
        with self.assertRaisesRegex(RepairCheckError, 'TEN_SCALE'):
            self.reduce(grid()[:2])
        rows = grid(); rows[5]['h'] *= .75
        with self.assertRaisesRegex(RepairCheckError, 'UNLOCKED_FD_INTERVAL'):
            self.reduce(rows)
        for kwargs in ({'derivative_relative_tolerance': .2}, {'convergence_relative_tolerance': .2},
                       {'first_window_k': 0}, {'extra_rechecks': 1}, {'signed_grid_observations_max': 82}):
            with self.assertRaises(RepairCheckError): RepairNumerics(**kwargs)

    def test_independent_direction_is_seeded_not_gradient_chosen(self):
        self.assertEqual(RepairNumerics().diagnostic_seed, 2026091503)
        g1 = torch.arange(1., 13.).reshape(3, 4)
        g2 = -g1 * 100
        cpu_rng = torch.random.get_rng_state().clone()
        first = make_directions(g1, objective='E')
        second = make_directions(g2, objective='E')
        other = make_directions(g1, objective='D')
        self.assertTrue(torch.equal(first['independent']['direction'], second['independent']['direction']))
        self.assertFalse(torch.equal(first['independent']['direction'], other['independent']['direction']))
        self.assertTrue(torch.equal(cpu_rng, torch.random.get_rng_state()))
        self.assertAlmostEqual(float(first['independent']['direction'].double().norm()), 1., places=6)
        self.assertEqual(first['independent']['seed'], 2026091501)

    def test_direct_weight_chain_rule_and_bilinear(self):
        gw = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        a = torch.tensor([[1., 0., 1.], [0., 1., 1.]])
        gc = gw @ a.T
        directions = {'self_gradient': torch.ones_like(gc)/2,
                      'independent': torch.tensor([[1., 0.], [0., 0.]])}
        result = check_direct_route(gc, gw, a, directions)
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['matrix_pass'])
        self.assertEqual(result['difference_norm'], 0.)
        self.assertFalse(result['full_neural_backward_independently_proven'])
        failed = check_direct_route(gc*2, gw, a, directions)
        self.assertEqual(failed['status'], 'FAIL_OR_UNRESOLVED')

    def test_direct_nearzero_bilinear_is_unresolved(self):
        z, a = torch.zeros(2, 2), torch.eye(2)
        result = check_direct_route(z, z, a, {'independent': torch.ones(2, 2)/2})
        self.assertTrue(result['matrix_pass'])
        self.assertEqual(result['status'], 'FAIL_OR_UNRESOLVED')
        self.assertEqual(result['bilinear'][0]['status'], 'UNRESOLVED_NEAR_ZERO_BILINEAR')

    def test_probe_outputs_saved_early_and_no_fullweight_payload(self):
        adapter = TinyAdapter()
        baseline = adapter.current([], torch.zeros(2, 2))
        adapter.calls = 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'E-self'
            result = run_directional_grid(adapter, [], torch.ones(2, 2), objective='E',
                direction_id='self_gradient', direction=torch.ones(2, 2)/2,
                baseline_observations=[baseline]*3, output=root)
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(adapter.calls, 20)
            self.assertEqual(result['forward_groups_or_documents'], 140)
            self.assertEqual(len(list(root.glob('k*/plus/observation.json'))), 10)
            self.assertEqual(len(list(root.glob('k*/minus/observation.json'))), 10)
            payload = torch.load(root/'k00/plus/correction.pt', weights_only=True)
            self.assertEqual(set(payload), {'C'})
            inp = json.loads((root/'k00/plus/input.json').read_text())
            self.assertFalse(inp['materialization']['full_weight_per_probe_saved'])
            self.assertTrue((root/'k00/plus/scalar-diagnostics.json').is_file())
            self.assertTrue((root/'k00/plus/paired-rows.json').is_file())
            with self.assertRaises(FileExistsError):
                run_directional_grid(adapter, [], torch.ones(2, 2), objective='E',
                    direction_id='self_gradient', direction=torch.ones(2, 2)/2,
                    baseline_observations=[baseline]*3, output=root)

    def test_forward_failure_preserves_prior_probes_and_failing_input(self):
        adapter = TinyAdapter()
        baseline = adapter.current([], torch.zeros(2, 2))
        adapter.calls, adapter.fail_at = 0, 3
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'failed'
            with self.assertRaisesRegex(RuntimeError, 'SYNTHETIC_FORWARD_FAILURE'):
                run_directional_grid(adapter, [], torch.ones(2, 2), objective='E',
                    direction_id='self_gradient', direction=torch.ones(2, 2)/2,
                    baseline_observations=[baseline]*3, output=root)
            failure = json.loads((root/'failure.json').read_text())
            self.assertEqual(failure['completed_signed_observations'], 2)
            self.assertTrue((root/'k00/plus/observation.json').is_file())
            self.assertTrue((root/'k00/minus/observation.json').is_file())
            self.assertTrue((root/'k01/plus/input.json').is_file())

    def test_D_grid_fail_closed_before_E_both_directions_pass(self):
        adapter = TinyAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RepairCheckError, 'E_BOTH_DIRECTIONS'):
                run_objective_fd(adapter, [], torch.ones(2, 2), objective='D',
                    baseline_observations=[], output=Path(tmp)/'D')
            self.assertEqual(adapter.calls, 0)
            self.assertEqual(json.loads((Path(tmp)/'D/failure.json').read_text())['status'], 'NOT_RUN_BLOCKED')


if __name__ == '__main__':
    unittest.main()
