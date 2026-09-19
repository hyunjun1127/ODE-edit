"""CPU controller fixtures only; not actual Llama/T0 or independent red evidence."""
from copy import deepcopy
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .basis import BasisResult
from .controller import (ControllerFailure, native_result, optimize_decision, optimize_kl)
from .decision import DecisionObservation
from .history import HistoryObservation
from .solver import CoefficientResult
from project.run_scripts.single_layer_edit_preserving_correction.optimizer import Check, tensor_sha256


def good_current(weight, ideal, trial):
    return dict(quality_pass=True, reasons=[], rows=[], invariant={'pass': True}, **{'pass': True})


def bad_current(weight, ideal, trial):
    return dict(quality_pass=False, reasons=['CURRENT_NLL'], rows=[], **{'pass': False})


def kl_rows(loss):
    return [dict(index=i, role='R512', source_row_id=f'doc-{i}', scored_positions=1,
                 loss=float(loss)) for i in range(512)]


def ref(mu=-1., *, correct=None, endpoint='native', input_id='same-input'):
    correct = mu > 0 if correct is None else correct
    rows = [dict(index=i, source_row_id=f'doc-{i}', capsule_sha256=f'capsule-{i}',
                 positions=[128], labels=[4], correct=[correct], scored_positions=1,
                 mu=float(mu)) for i in range(512)]
    return DecisionObservation(endpoint, input_id, 'R512', rows, max(0., -mu)**2,
                               0 if correct else 512, None,
                               dict(complete=True, documents=512, positions=512,
                                    expected_positions=512), {})


def hist(slack=0., *, passed=True, case_id='past-1', endpoint='native', empty=False):
    rows = [] if empty else [dict(case_id=case_id, slack=float(slack))]
    return HistoryObservation(endpoint, 'same-past', 'entry', rows,
        0. if empty else max(0., -slack)**2, passed, None,
        dict(complete=True, active_requests=len(rows)), {})


class Quadratic:
    """Full-coverage scalar fixture L=1/2*(w-2)^2 at WN=0."""
    def __init__(self):
        self.gradient_calls = self.value_calls = 0

    def __call__(self, weight, gradient=False):
        if gradient:
            self.gradient_calls += 1
        else:
            self.value_calls += 1
        difference = weight.double() - 2.
        loss = float(torch.sum(difference**2)/2)
        return loss, difference if gradient else None, kl_rows(loss)


class KLTests(unittest.TestCase):
    def setUp(self):
        self.W = torch.zeros((1, 1), dtype=torch.float32)

    def run_kl(self, **overrides):
        kwargs = dict(objective=Quadratic(), project=lambda g: g,
                      current_check=good_current, proposal_check=lambda d: Check(True))
        kwargs.update(overrides)
        return optimize_kl(self.W, **kwargs)

    def test_first_accept_one_gradient_and_receipt(self):
        objective = Quadratic()
        result = self.run_kl(objective=objective)
        self.assertTrue(result.summary['accepted'])
        self.assertEqual((objective.gradient_calls, objective.value_calls), (1, 1))
        self.assertEqual(result.stop_reason, 'ACCEPTED_KL')
        self.assertEqual(float(result.weight[0, 0]), 1.)
        self.assertTrue(torch.equal(self.W, torch.zeros_like(self.W)))
        self.assertIsNone(result.summary['reference_pass'])  # no DEC choice guard added
        self.assertEqual(result.receipt()['history_appends_in_controller'], 0)
        json.dumps(result.receipt(), allow_nan=False)

    def test_current_rejection_halves_without_new_gradient(self):
        objective = Quadratic()
        def current(w, d, trial):
            return bad_current(w, d, trial) if trial == 1 else good_current(w, d, trial)
        result = self.run_kl(objective=objective, current_check=current)
        self.assertEqual(len(result.trials), 2)
        self.assertEqual(result.trials[-1]['scale'], .5)
        self.assertEqual((objective.gradient_calls, objective.value_calls), (1, 2))

    def test_four_trials_no_legacy_eight(self):
        objective = Quadratic()
        result = self.run_kl(objective=objective, current_check=bad_current)
        self.assertEqual(len(result.trials), 4)
        self.assertEqual([r['scale'] for r in result.trials], [1., .5, .25, .125])
        self.assertEqual(objective.gradient_calls, 1)
        self.assertEqual(result.stop_reason, 'NO_RESOLVED_STEP')
        self.assertTrue(torch.equal(result.weight, self.W))

    def test_armijo_rejection_does_not_run_current(self):
        def objective(w, gradient=False):
            return 2., torch.ones_like(w) if gradient else None, kl_rows(2.)
        result = self.run_kl(objective=objective,
                             current_check=lambda *a: self.fail('guard before Armijo'))
        self.assertEqual(result.counters['current_checks'], 0)
        self.assertEqual(result.counters['objective_value_sweeps'], 4)

    def test_actual_p_uses_fp32_delta(self):
        result = self.run_kl()
        self.assertEqual(result.trials[0]['actual_p'], -2.)
        self.assertEqual(result.trials[0]['actual_armijo_rhs'], 2.-2e-4)

    def test_zero_gradient(self):
        result = self.run_kl(project=lambda g: torch.zeros_like(g))
        self.assertEqual(result.stop_reason, 'ZERO_GRADIENT')
        self.assertEqual(len(result.trials), 0)

    def test_numerical_floor(self):
        result = self.run_kl(objective=lambda w, gradient: (1e-7, torch.ones_like(w), kl_rows(1e-7)))
        self.assertEqual(result.stop_reason, 'NUMERICAL_FLOOR')

    def test_space_unresolved_does_not_run_objective_or_project(self):
        for status in ('RANK_UNRESOLVED', 'REPAIR_SPACE_EMPTY'):
            result = self.run_kl(space_status=status,
                objective=lambda *a, **k: self.fail('unresolved objective'),
                project=lambda *a: self.fail('unresolved projection'))
            self.assertEqual(result.stop_reason, status)
            self.assertEqual(result.counters['gradient_sweeps'], 0)

    def test_negative_kl_is_technical(self):
        with self.assertRaisesRegex(ControllerFailure, 'KL_BELOW_ROUNDOFF'):
            self.run_kl(objective=lambda w, gradient: (-.01, torch.ones_like(w), kl_rows(-.01)))

    def test_nonfinite_is_technical_not_fallback(self):
        with self.assertRaisesRegex(ControllerFailure, 'NONFINITE_KL'):
            self.run_kl(objective=lambda w, gradient: (float('nan'), torch.ones_like(w), kl_rows(0)))

    def test_failed_projection_is_technical(self):
        with self.assertRaisesRegex(ControllerFailure, 'IDEAL_PROJECTION_CHECK_FAILED'):
            self.run_kl(proposal_check=lambda d: Check(False, 'DK'))

    def test_incomplete_bank_is_technical(self):
        with self.assertRaisesRegex(ControllerFailure, 'KL_FULL512_REQUIRED'):
            self.run_kl(objective=lambda w, gradient: (1., torch.ones_like(w), kl_rows(1.)[:4]))

    def test_candidate_document_identity_change_is_technical(self):
        objective = Quadratic()
        def changed(w, gradient=False):
            loss, grad, rows = objective(w, gradient)
            if not gradient:
                rows[-1]['source_row_id'] = 'wrong'
            return loss, grad, rows
        with self.assertRaisesRegex(ControllerFailure, 'KL_TRIAL_INPUT_COVERAGE_CHANGED'):
            self.run_kl(objective=changed)

    def test_endpoint_mutation_is_technical_and_caller_unchanged(self):
        def corrupt(w, gradient=False):
            w.add_(1)
            return 1., torch.ones_like(w), kl_rows(1.)
        with self.assertRaisesRegex(ControllerFailure, 'CALLBACK_ENDPOINT_MUTATION'):
            self.run_kl(objective=corrupt)
        self.assertTrue(torch.equal(self.W, torch.zeros_like(self.W)))

    def test_oom_and_evidence_failure_are_not_native_success(self):
        def oom(*a, **k):
            raise RuntimeError('CUDA out of memory')
        with self.assertRaisesRegex(ControllerFailure, 'out of memory') as cm:
            self.run_kl(objective=oom)
        self.assertEqual(cm.exception.events[-1]['event'], 'TECHNICAL_FAILURE')
        with self.assertRaisesRegex(ControllerFailure, 'EVIDENCE_WRITE_FAILURE'):
            self.run_kl(event=lambda *a: (_ for _ in ()).throw(OSError('ENOSPC')))

    def test_initial_reuse_requires_exact_identity(self):
        objective = Quadratic()
        loss, gradient, rows = objective(self.W, True)
        initial = dict(loss=loss, gradient=gradient, rows=rows, objective_id='R512',
                       weight_sha256=tensor_sha256(self.W))
        result = self.run_kl(initial_observation=initial)
        self.assertEqual(result.counters['gradient_sweeps'], 0)
        initial['weight_sha256'] = 'wrong'
        with self.assertRaisesRegex(ControllerFailure, 'INITIAL_KL_REUSE_IDENTITY'):
            self.run_kl(initial_observation=initial)

    def test_past_violation_fallback_and_no_b1_history_evaluation(self):
        native = hist(-.1, passed=False)
        result = self.run_kl(native_history=native, history_check=lambda w, trial: native)
        self.assertEqual(result.stop_reason, 'FALLBACK_WITH_PAST_VIOLATION')
        self.assertFalse(result.summary['history_pass'])
        self.run_kl(native_history=hist(empty=True),
                    history_check=lambda *a: self.fail('B1 empty history must not forward'))


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.W = torch.zeros((1, 1), dtype=torch.float32)
        self.basis = BasisResult((np.ones((1, 1), dtype=np.float64),), dict(shape=[1, 1], rank=1))
        self.solution = CoefficientResult(np.ones(1), 'LOCAL_TWO_PHASE_SOLVED', {'fixture': True})

    def call(self, **overrides):
        args = dict(native_reference=ref(), raw_gradient=torch.ones((1, 1)),
                    projected_gradient=torch.ones((1, 1)), basis=self.basis,
                    reference_J=torch.ones((512, 1), dtype=torch.float64),
                    evaluate_reference=lambda w, trial: ref(-.5, endpoint=f'candidate-{trial}'),
                    current_check=good_current, proposal_check=lambda d: Check(True))
        args.update(overrides)
        arm = args.pop('arm', 'DEC_MODES_CUM')
        with patch('project.run_scripts.single_layer_mechanism_first.controller.solver.solve_coefficients',
                   return_value=self.solution) as mocked:
            result = optimize_decision(arm, self.W, **args)
        return result, mocked

    def test_margin_only_first_accept_and_gate_summary(self):
        result, called = self.call()
        self.assertEqual(result.stop_reason, 'MARGIN_ONLY')
        self.assertEqual(called.call_count, 1)
        self.assertEqual(result.counters['reference_candidate_sweeps'], 1)
        self.assertEqual(result.counters['gradient_sweeps'], 0)  # supplied own-center derivative
        self.assertTrue(result.summary['full512'])
        self.assertEqual(result.summary['phi_reference_gain'], .75)
        self.assertEqual(result.summary['repaired_choices'], 0)
        json.dumps(result.receipt(), allow_nan=False)

    def test_choice_repair(self):
        result, _ = self.call(evaluate_reference=lambda w, trial: ref(.2, endpoint='repair'))
        self.assertEqual(result.stop_reason, 'CHOICE_REPAIRED')
        self.assertEqual(result.summary['repaired_choices'], 512)

    def test_line_uses_same_solver_and_acceptance(self):
        result, called = self.call(arm='DEC_LINE')
        self.assertEqual(result.stop_reason, 'MARGIN_ONLY')
        self.assertEqual(called.call_count, 1)
        self.assertEqual(called.call_args.args[2], 1.)

    def test_four_current_rejections_native_fallback(self):
        result, _ = self.call(current_check=bad_current)
        self.assertEqual(len(result.trials), 4)
        self.assertEqual(result.stop_reason, 'FINITE_SEARCH_UNRESOLVED')
        self.assertEqual(result.counters['reference_candidate_sweeps'], 4)
        self.assertTrue(torch.equal(result.weight, self.W))

    def test_new_flip_inside_unsafe_document_rejected(self):
        native = ref(-1)
        candidate = ref(-.5, endpoint='candidate')
        for obs, flags in ((native, [False, True]), (candidate, [False, False])):
            for row in obs.rows:
                row.update(positions=[128, 129], labels=[4, 4], correct=flags, scored_positions=2)
            obs.coverage.update(positions=1024, expected_positions=1024)
            obs.mismatches = 512 if obs is native else 1024
        result, _ = self.call(native_reference=native, evaluate_reference=lambda w, t: candidate)
        self.assertEqual(result.stop_reason, 'FINITE_SEARCH_UNRESOLVED')
        self.assertIn('REFERENCE_NEW_FLIP', result.trials[0]['acceptance']['reasons'])

    def test_zero_risk_and_tie_only_distinct(self):
        for native, reason in ((ref(1), 'ZERO_RISK_NO_OP'),
                               (ref(0, correct=False), 'TIE_ONLY_NO_DIRECTION')):
            result, called = self.call(native_reference=native,
                evaluate_reference=lambda *a: self.fail('no-op model call'))
            self.assertEqual(result.stop_reason, reason)
            self.assertEqual(called.call_count, 0)

    def test_small_risk_and_projected_zero(self):
        result, _ = self.call(native_reference=ref(-1e-6))
        self.assertEqual(result.stop_reason, 'BELOW_RISK_RESOLUTION')
        result, _ = self.call(projected_gradient=torch.zeros((1, 1)))
        self.assertEqual(result.stop_reason, 'NO_PROJECTED_DIRECTION')

    def test_space_unresolved_is_not_zero_gradient_diagnosis(self):
        result, called = self.call(space_status='RANK_UNRESOLVED')
        self.assertEqual(result.stop_reason, 'RANK_UNRESOLVED')
        self.assertEqual(called.call_count, 0)

    def test_reference_coverage_and_nan_technical(self):
        bad = ref()
        bad.coverage['complete'] = False
        with self.assertRaisesRegex(ControllerFailure, 'FULL512_REQUIRED'):
            self.call(native_reference=bad)
        bad = ref(-.5, endpoint='candidate')
        bad.phi_reference = float('nan')
        with self.assertRaisesRegex(ControllerFailure, 'NONFINITE_REFERENCE_RISK'):
            self.call(evaluate_reference=lambda w, t: bad)

    def test_changed_reference_identity_technical(self):
        with self.assertRaisesRegex(ControllerFailure, 'ACCEPTANCE_REFERENCE_IDENTITY'):
            self.call(evaluate_reference=lambda w, t: ref(-.5, input_id='changed'))

    def test_finite_solver_failure_not_infeasibility(self):
        self.solution = CoefficientResult(None, 'FINITE_SOLVER_UNRESOLVED', {})
        result, _ = self.call()
        self.assertEqual(result.stop_reason, 'FINITE_SOLVER_UNRESOLVED')
        self.assertEqual(result.counters['materializations'], 0)

    def test_infeasibility_requires_certificate(self):
        self.solution = CoefficientResult(None, 'LOCAL_HISTORY_CONSTRAINT_INFEASIBLE', {})
        with self.assertRaisesRegex(ControllerFailure, 'CERTIFICATE_REQUIRED'):
            self.call()
        self.solution.receipt['infeasibility_certificate'] = {'verified': True}
        result, _ = self.call()
        self.assertEqual(result.stop_reason, 'LOCAL_HISTORY_CONSTRAINT_INFEASIBLE')

    def test_native_past_violation_never_protection_success(self):
        native = hist(-.75, passed=False)
        result, _ = self.call(native_history=native,
                              history_J=np.ones((1, 1)),
                              history_check=lambda w, t: native)
        self.assertEqual(result.stop_reason, 'FALLBACK_WITH_PAST_VIOLATION')
        self.assertFalse(result.summary['history_pass'])
        self.assertEqual(result.counters['linear_history_skips'], 3)
        self.assertEqual(result.counters['reference_candidate_sweeps'], 1)

    def test_past_repair_and_entry_identity_binding(self):
        native = hist(-.75, passed=False)
        result, _ = self.call(native_history=native, history_J=np.ones((1, 1)),
                              history_check=lambda w, t: hist(0., endpoint='candidate'))
        self.assertTrue(result.summary['history_pass'])
        self.assertTrue(result.summary['accepted'])
        with self.assertRaisesRegex(ControllerFailure, 'HISTORY_TRIAL_INPUT_OR_ENTRY_CHANGED'):
            self.call(native_history=native, history_J=np.ones((1, 1)),
                history_check=lambda w, t: hist(0., case_id='another-request'))

    def test_nonempty_history_cannot_be_silently_skipped(self):
        with self.assertRaisesRegex(ControllerFailure, 'ACTIVE_HISTORY_CALLBACK_REQUIRED'):
            self.call(native_history=hist(-.1, passed=False), history_J=np.ones((1, 1)))

    def test_line_rank_above_one_technical(self):
        basis = BasisResult((np.ones((1, 1)), np.ones((1, 1))), dict(shape=[1, 1]))
        with self.assertRaisesRegex(ControllerFailure, 'BASIS_ARM_RANK'):
            self.call(arm='DEC_LINE', basis=basis)

    def test_native_result_has_no_fit_or_commit(self):
        result = native_result(self.W, native_history=hist(empty=True))
        self.assertEqual(result.stop_reason, 'NATIVE_BASELINE')
        self.assertEqual(result.receipt()['history_appends_in_controller'], 0)
        self.assertEqual(result.counters['materializations'], 0)
        broken = native_result(self.W, native_history=hist(-1, passed=False))
        self.assertEqual(broken.stop_reason, 'FALLBACK_WITH_PAST_VIOLATION')

    def test_real_one_dimensional_two_phase_solver_connection(self):
        result = optimize_decision('DEC_LINE', self.W,
            native_reference=ref(), raw_gradient=torch.ones((1, 1)),
            projected_gradient=torch.ones((1, 1)), basis=self.basis,
            reference_J=np.ones((512, 1)),
            evaluate_reference=lambda w, trial: ref(-1.+float(w[0, 0]), endpoint='candidate'),
            current_check=good_current, proposal_check=lambda d: Check(True))
        self.assertTrue(result.summary['accepted'])
        self.assertEqual(result.details['solver_status'], 'LOCAL_TWO_PHASE_SOLVED')
        self.assertEqual(result.counters['solver_calls'], 1)

    def test_event_payloads_are_json_no_checkpoint_tensor(self):
        events = []
        result, _ = self.call(event=events.append)
        self.assertTrue(events)
        json.dumps(events, allow_nan=False)
        self.assertTrue(all('weight' not in r for r in events))
        self.assertEqual(result.counters['accepted_rounds'], 1)


if __name__ == '__main__':
    unittest.main()
