"""실제 모델 없이 캐시·예산·품질·commit 계약을 검사하는 CPU fixture."""
import math
import unittest

from controller_reference import BudgetExceeded, ContractError, Controller, Limits, Scores, run_cobyla

try:
    import scipy
    LEGACY_SCIPY_AVAILABLE = scipy.__version__ == "1.15.3"
except ImportError:
    LEGACY_SCIPY_AVAILABLE = False


def good_scores(base=.1, **changes):
    defaults = dict(base_kl=base, training_e=.01, current_strict=frozenset({1, 2}),
                    current_pair=frozenset({1}), past_h=.02, past_strict=frozenset({3}),
                    past_pair=frozenset({3}), canonical_e=.01,
                    current_token_margins={1: .2, 2: .3}, current_pair_margins={1: .1},
                    past_token_margins={3: .2}, past_pair_margins={3: .1})
    defaults.update(changes)
    return Scores(**defaults)


class ToyBackend:
    def __init__(self, layers=(4, 8), score_fn=None, delta_fn=None):
        self.weights = {l: 0. for l in layers}
        self.history = {l: 0 for l in layers}
        self.fit_calls, self.score_calls = [], 0
        self.score_fn = score_fn or (lambda w: good_scores((w[4] - .6) ** 2 + sum(v*v for l, v in w.items() if l != 4)))
        self.delta_fn = delta_fn or (lambda l, w: 1. if l == 4 else .1 + sum(w[j] for j in w if j < l))

    def snapshot(self):
        return (tuple(sorted(self.weights.items())), tuple(sorted(self.history.items())))

    def restore(self, snapshot):
        self.weights, self.history = dict(snapshot[0]), dict(snapshot[1])

    def state_token(self):
        return repr(tuple((l, w.hex()) for l, w in sorted(self.weights.items())))

    def history_token(self):
        return repr(tuple(sorted(self.history.items())))

    def fit_native(self, layer):
        self.fit_calls.append((layer, self.state_token()))
        self.weights[layer] += self.delta_fn(layer, self.weights)
        return 0

    def apply_gate(self, layer, before, native, gate):
        start, end = dict(before[0])[layer], dict(native[0])[layer]
        self.weights[layer] = end if gate == 1 else start + gate * (end - start)

    def score(self):
        self.score_calls += 1
        return self.score_fn(self.weights)

    def changed_layers(self, before, after):
        left, right = dict(before[0]), dict(after[0])
        return tuple(l for l in sorted(left) if left[l] != right[l])

    def action_norm(self, before, after):
        left, right = dict(before[0]), dict(after[0])
        return math.sqrt(sum((left[l] - right[l]) ** 2 for l in left))

    def finalize_history(self, layers):
        for layer in layers:
            self.history[layer] += 1


class ControllerTests(unittest.TestCase):
    def make(self, backend=None, layers=(4, 8), limits=Limits()):
        backend = backend or ToyBackend(layers)
        return backend, Controller(backend, layers, namespace="fixed-toy-capsule", limits=limits)

    def test_exact_prefix_reuse_and_upstream_invalidation(self):
        b, c = self.make()
        c.evaluate((.6, .2))
        c.evaluate((.6, .8))  # own gate changes do not change the native fit anchor
        self.assertEqual(c.counts["suffix_fits"], 1)
        c.evaluate((.7, .8))
        self.assertEqual(c.counts["suffix_fits"], 2)
        self.assertEqual(c.counts["l4_fits"], 1)
        self.assertNotEqual(b.fit_calls[1][1], b.fit_calls[2][1])
        self.assertEqual(b.state_token(), c.entry_token)

    def test_zero_layer_skips_fit_but_final_history_appends_once(self):
        b, c = self.make()
        c.evaluate((.6, 0.))
        self.assertEqual(c.counts["suffix_fits"], 0)
        self.assertEqual(b.history, {4: 0, 8: 0})
        chosen = c.finalize()
        self.assertEqual(chosen.gates, (.6, 0.))
        self.assertEqual(b.history, {4: 1, 8: 1})
        with self.assertRaises(ContractError):
            c.finalize()
        with self.assertRaises(ContractError):
            c.evaluate((.7, .5))

    def test_objective_constraints_share_cache_and_exact_endpoint_alias(self):
        b = ToyBackend(delta_fn=lambda layer, w: 1. if layer == 4 else 0.)
        _, c = self.make(b)
        first = c.objective_and_constraints((1., 1.))
        second = c.objective_and_constraints((1., 1.))
        c.evaluate((1., .5))
        self.assertEqual(first, second)
        self.assertEqual(b.score_calls, 1)  # all endpoints equal the N4 baseline
        self.assertEqual(c.counts["endpoints"], 0)
        self.assertEqual(c.counts["suffix_fits"], 1)

    def test_bounds_rejection_is_analytical_without_clipping_or_model_call(self):
        b, c = self.make()
        f, slack = c.objective_and_constraints((1.000001, .5))
        self.assertTrue(math.isfinite(f))
        self.assertLess(min(slack), 0)
        self.assertEqual(len(b.fit_calls), 1)
        self.assertEqual(b.score_calls, 1)
        self.assertTrue(c.best().is_n4)

    def test_reserve_is_available_only_to_pruning(self):
        limits = Limits(max_suffix_fits=2, max_endpoints=2,
                        reserve_suffix_fits_for_pruning=1, reserve_endpoints_for_pruning=1)
        b, c = self.make(limits=limits)
        c.evaluate((.6, .1))
        with self.assertRaises(BudgetExceeded):
            c.evaluate((.7, .1))
        c.evaluate((.7, .1), phase="pruning")
        self.assertEqual(c.counts["suffix_fits"], 2)
        self.assertEqual(c.counts["endpoints"], 2)
        self.assertEqual(b.state_token(), c.entry_token)

    def test_partial_path_budget_exhaustion_retains_cost_and_rolls_back(self):
        layers = (4, 5, 6, 7, 8)
        b, c = self.make(layers=layers, limits=Limits(max_suffix_fits=1, reserve_suffix_fits_for_pruning=0))
        with self.assertRaises(BudgetExceeded):
            c.evaluate((.6, 1., 1., 1., 1.))
        self.assertEqual(c.counts["suffix_fits"], 1)
        self.assertEqual(c.counts["endpoints"], 0)
        self.assertEqual(b.history, {l: 0 for l in layers})
        self.assertEqual(b.state_token(), c.entry_token)
        self.assertTrue(c.best().is_n4)

    def test_quality_ids_and_pair_failures_keep_n4(self):
        def score(w):
            return good_scores(.1) if w[4] == 1 else good_scores(.01, current_strict=frozenset({1}), current_pair=None)
        _, c = self.make(ToyBackend(score_fn=score))
        row = c.evaluate((.6, 0.))
        self.assertIn("CURRENT_STRICT_IDS", row.reasons)
        self.assertIn("CURRENT_PAIR_IDS", row.reasons)
        self.assertTrue(c.best().is_n4)

    def test_no_point05_plateau_and_past_mean_protection(self):
        def score(w):
            return good_scores(.1) if w[4] == 1 else good_scores(.01, training_e=.011, past_h=.021)
        _, c = self.make(ToyBackend(score_fn=score))
        row = c.evaluate((.6, 0.))
        self.assertIn("CURRENT_TRAINING_MEAN", row.reasons)
        self.assertIn("PAST_MEAN", row.reasons)
        self.assertTrue(c.best().is_n4)

    def test_intermediate_infeasibility_is_not_pruned(self):
        def score(w):
            bad = w[4] < .9 and w[8] == 0
            return good_scores(.01 if w[4] < .9 else .1, training_e=.2 if bad else .01)
        _, c = self.make(ToyBackend(score_fn=score))
        partial = c.evaluate((.6, 0.))
        final = c.evaluate((.6, 1.))
        self.assertFalse(partial.feasible)
        self.assertTrue(final.feasible)
        self.assertEqual(c.best().gates, (.6, 1.))

    def test_pruning_cannot_accumulate_epsilon_degradation(self):
        eps = 1e-6
        def score(w):
            if w[4] == 1:
                return good_scores(.2)
            absent = int(w[5] == 0) + int(w[6] == 0)
            return good_scores(.1 + absent * .75 * eps)
        layers = (4, 5, 6)
        b = ToyBackend(layers, score_fn=score, delta_fn=lambda l, w: 1.)
        _, c = self.make(b, layers=layers)
        c.evaluate((.6, 1., 1.))
        selected = c.prune()
        self.assertEqual(len(selected.active_layers), 2)
        self.assertLessEqual(selected.scores.base_kl, .1 + eps)
        self.assertTrue(any(r.scores.base_kl > .1 + eps for r in c.records if not r.is_n4))

    def test_n4_wins_global_epsilon_tie(self):
        def score(w):
            return good_scores(.1 if w[4] == 1 else .1 - .5e-6)
        _, c = self.make(ToyBackend(score_fn=score))
        c.evaluate((.6, 0.))
        self.assertTrue(c.best().is_n4)

    def test_nonfinite_score_is_technical_error_with_rollback(self):
        def score(w):
            return good_scores(.1) if w[4] == 1 else good_scores(float("nan"))
        b, c = self.make(ToyBackend(score_fn=score))
        with self.assertRaisesRegex(ContractError, "NONFINITE_SCORE"):
            c.evaluate((.6, 0.))
        self.assertEqual(b.state_token(), c.entry_token)
        self.assertEqual(c.counts["endpoints"], 1)

    def test_raw_search_proposal_cap_counts_unique_points(self):
        _, c = self.make(limits=Limits(max_search_proposals=1))
        c.evaluate((.6, 0.))
        c.evaluate((.6, 0.))
        with self.assertRaisesRegex(BudgetExceeded, "SEARCH_PROPOSAL_CAP"):
            c.evaluate((.7, 0.))
        self.assertEqual(c.counts["search_proposals"], 1)

    def test_missing_margin_is_not_invented(self):
        def score(w):
            return good_scores(.1) if w[4] == 1 else good_scores(.01, current_token_margins={1: .2})
        _, c = self.make(ToyBackend(score_fn=score))
        with self.assertRaisesRegex(ContractError, "MISSING_PROTECTED_MARGIN"):
            c.objective_and_constraints((.6, 0.))

    def test_actual_adam_charge_releases_unused_reserve(self):
        class SteppedBackend(ToyBackend):
            def fit_native(self, layer):
                super().fit_native(layer)
                return 2400 if layer == 4 else 600
        b, c = self.make(SteppedBackend(), limits=Limits(max_extra_adam=3000))
        c.evaluate((.6, .1))
        c.evaluate((.7, .1))  # 2400 reserve가 남으므로 두 번째 fit 가능
        self.assertEqual(c.counts["extra_adam"], 1200)
        self.assertEqual(c.counts["l4_adam"], 2400)  # extra cap에서 제외
        c.evaluate((.7, .2))  # 캐시 재사용은 Adam reserve가 필요 없다.
        with self.assertRaisesRegex(BudgetExceeded, "EXTRA_ADAM_RESERVE"):
            c.evaluate((.8, .1))
        self.assertEqual(c.counts["suffix_fits"], 2)
        self.assertEqual(c.counts["extra_adam"], 1200)
        self.assertEqual(b.state_token(), c.entry_token)

    def test_search_subset_keeps_all_five_histories(self):
        all_layers = (4, 5, 6, 7, 8)
        b = ToyBackend(all_layers)
        c = Controller(b, (4, 8), namespace="V2_SHARED_HISTORY", history_layers=all_layers)
        c.evaluate((.6, 0.))
        c.finalize()
        self.assertEqual(b.history, {l: 1 for l in all_layers})
        self.assertEqual([l for l, _ in b.fit_calls], [4])

    def test_empty_protected_sets_have_inactive_margin_rows(self):
        score = lambda w: good_scores(.1, current_strict=frozenset(), current_pair=None,
                                     past_h=None, past_strict=frozenset(), past_pair=None)
        _, c = self.make(ToyBackend(score_fn=score))
        _, slack = c.objective_and_constraints((1., 0.))
        self.assertEqual(slack[-4:], (1., 1., 1., 1.))

    def test_raw_reduction_violation_cannot_round_into_legal_gate(self):
        b, c = self.make()
        self.assertEqual(1. - (-1e-18), 1.)  # gate 변환만 검사하면 사라지는 위반
        _, slack = c.measure_reductions((-1e-18, 1.))
        self.assertLess(min(slack), 0.)
        self.assertEqual(c.counts["bound_rejections"], 1)
        self.assertEqual(b.score_calls, 1)
        self.assertTrue(c.best().is_n4)
        self.assertTrue(c.evaluate((1., 0.)).feasible)  # 정상 N4 캐시를 오염시키지 않음

    def test_incomplete_pruning_is_explicit(self):
        limits = Limits(max_endpoints=1, reserve_endpoints_for_pruning=0)
        _, c = self.make(limits=limits)
        c.evaluate((.6, .1))
        selected = c.prune()
        self.assertEqual(selected.gates, (.6, .1))
        self.assertFalse(c.pruning_status["complete"])
        self.assertEqual(c.pruning_status["attempted"], 1)
        self.assertEqual(c.pruning_status["completed"], 0)
        self.assertEqual(c.pruning_status["budget_reason"], "ENDPOINT_CAP")

    def test_search_coverage_excludes_incomplete_and_pruning_candidates(self):
        b, c = self.make()
        c.evaluate((.6, .1))
        self.assertFalse(c.coverage()["adequate_by_count_proxy"])
        c.evaluate((.7, .1))
        c.evaluate((.8, .1))
        c.evaluate((.9, .1))
        coverage = c.coverage()
        self.assertTrue(coverage["adequate_by_count_proxy"])
        self.assertEqual(coverage["completed_search_gate_vectors"], 4)
        c.prune()
        self.assertEqual(c.coverage()["completed_search_gate_vectors"], 4)
        self.assertEqual(b.state_token(), c.entry_token)


@unittest.skipUnless(LEGACY_SCIPY_AVAILABLE, "실제 solver 검사는 SciPy 1.15.3 필요")
class LegacyCobylaTests(unittest.TestCase):
    @staticmethod
    def coupled_score(w):
        strength = w[4] + w[8]
        return good_scores((w[4] - .613)**2 + .1*w[8]**2,
                           training_e=.01 + .1*(1-strength)**2,
                           current_strict=frozenset({1, 2}) if strength >= .98 else frozenset(),
                           current_pair=frozenset({1}) if strength > .9 else frozenset(),
                           current_token_margins={1: strength-.98, 2: strength-.98},
                           current_pair_margins={1: strength-.9})

    @staticmethod
    def fresh_delta(layer, weights):
        if layer == 4:
            return 1.
        if layer == 8:
            return 1. - weights[4]
        return .2

    def controller(self, layers=(4, 8), limits=Limits()):
        b = ToyBackend(layers, score_fn=self.coupled_score, delta_fn=self.fresh_delta)
        return b, Controller(b, layers, namespace="ACTUAL_SCIPY_CPU_TOY", limits=limits)

    def test_real_solver_finds_continuous_coupled_candidate(self):
        b, c = self.controller()
        selected, stop = run_cobyla(c)
        self.assertEqual(stop["reason"], "SOLVER_RETURN")
        self.assertTrue(selected.feasible)
        self.assertLess(selected.scores.base_kl, c.reference.base_kl - 1e-6)
        self.assertTrue(any(all(abs(a-v) > 1e-5 for v in (0., .5, .75, 1.)) for a in selected.gates))
        self.assertFalse(c.evaluate((selected.gates[0], 0.)).feasible)
        self.assertEqual(b.history, {4: 0, 8: 0})

    def test_real_five_dimensional_search_budget_then_deletion(self):
        layers = (4, 5, 6, 7, 8)
        b, c = self.controller(layers)
        run_cobyla(c)
        selected = c.prune()
        self.assertTrue(selected.feasible)
        self.assertEqual(selected.gates[1:4], (0., 0., 0.))
        self.assertLess(selected.scores.base_kl, c.reference.base_kl - 1e-6)
        self.assertLessEqual(c.counts["suffix_fits"], 40)
        self.assertLessEqual(c.counts["endpoints"], 28)
        self.assertEqual(b.history, {l: 0 for l in layers})
        c.finalize()
        self.assertEqual(b.history, {l: 1 for l in layers})

    def test_real_fortran_callback_budget_exception_returns_n4(self):
        b, c = self.controller(limits=Limits(max_suffix_fits=1,
                                             reserve_suffix_fits_for_pruning=0))
        selected, stop = run_cobyla(c)
        self.assertEqual(stop["reason"], "SUFFIX_FIT_CAP")
        self.assertTrue(selected.is_n4)
        self.assertEqual(c.counts["suffix_fits"], 1)
        self.assertEqual(b.state_token(), c.entry_token)
        self.assertEqual(c.counts["commits"], 0)
        self.assertFalse(c.search_coverage["adequate_by_count_proxy"])


if __name__ == "__main__":
    unittest.main()
