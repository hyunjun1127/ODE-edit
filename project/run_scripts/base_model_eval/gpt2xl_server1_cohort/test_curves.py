"""CPU-only saved-row fixtures: no model, SDK, tensor or scheduler operation."""
import copy
import math
import unittest

from .curves import (DEFAULT_CONFIG, FIELDS, KINDS, PANELS, attach_occurrence_ordinals,
    build_curves, curve_coverage, metric_values, reduce_rows, validate_curve_payload, validate_rows)


def fixture_rows():
    actual, expected = [], []
    for ordinal in range(2000):
        # Repeated and non-monotonic IDs deliberately make case-set slicing wrong.
        case = (9001, 7, 9001, 444)[ordinal % 4]
        for kind, index in PANELS:
            row = dict(occurrence_ordinal=ordinal, case_id=case, kind=kind, prompt_index=index,
                identity=f'legacy-{case}-{kind}-{index}', endpoint='W0',
                new_token_identity=f'new-{case}-{kind}-{index}',
                true_token_identity=f'true-{case}-{kind}-{index}',
                new_nll=1. + (ordinal % 23) / 10., true_nll=2.,
                new_token_count=1 if ordinal % 2 == 0 else 4,
                new_token_correct=0 if ordinal % 2 == 0 else 4,
                new_strict=ordinal % 2 != 0,
                true_token_count=2 if ordinal % 2 == 0 else 6,
                true_token_correct=2 if ordinal % 2 == 0 else 0,
                true_strict=ordinal % 2 == 0)
            row['margin_true_minus_new'] = row['true_nll'] - row['new_nll']
            row['margin_new_minus_true'] = -row['margin_true_minus_new']
            actual.append(row)
            expected.append({key:row[key] for key in (
                'occurrence_ordinal','case_id','kind','prompt_index','identity',
                'new_token_identity','true_token_identity')})
    return actual, expected


class CohortCurvesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.expected = fixture_rows()
        cls.payloads = build_curves(cls.rows, cls.expected)

    def test_full_schedule_occurrences_not_case_ids(self):
        self.assertEqual(len(self.payloads), 21)
        self.assertEqual([payload['edits'] for payload in self.payloads], list(range(0,2001,100)))
        self.assertEqual(len(set(row['case_id'] for row in self.rows)), 3)
        self.assertLess(len(set(row['identity'] for row in self.rows)), len(self.rows))
        coverage = curve_coverage(self.payloads, DEFAULT_CONFIG)
        self.assertEqual(coverage['current_coverage'], 20)
        self.assertEqual(coverage['all_seen_points'], [500,1000,1500,2000])
        self.assertEqual(coverage['new_forward_calls_by_reducer'], 0)
        for payload in self.payloads:
            self.assertEqual(payload['actual_model_edits'], 0)
            self.assertEqual(payload['actual_applied_edits'], 0)
            self.assertEqual(payload['pre_state_edits'], 0)
            self.assertEqual(payload['post_state_edits'], 0)
            self.assertIs(payload['reference_only'], True)
            self.assertEqual(payload['evaluation_model_state'], 'W0')

    def test_first_last_exact_slices_and_prefix_denominators(self):
        for batch in (1,20):
            payload = self.payloads[batch]
            start, stop = 100*(batch-1), 100*batch
            subset = self.rows[start*13:stop*13]
            group = reduce_rows(subset)
            for kind, count in (('R',100),('P',200),('N',1000)):
                self.assertEqual(payload[f'current/post/{kind}/count'], count)
                self.assertEqual(payload[f'current/post/{kind}/success_count'], group[kind]['numerator'])
                self.assertEqual(payload[f'current/post/{kind}/new_nll'], group[kind]['new_nll_mean'])
        self.assertNotEqual(self.payloads[1]['current/post/R/new_nll'],
                            self.payloads[20]['current/post/R/new_nll'])
        self.assertEqual([self.payloads[0][f'W0_first2000/{kind}/count'] for kind in KINDS],
                         [2000,4000,20000])
        for batch in (5,10,15,20):
            self.assertEqual([self.payloads[batch][f'all_seen/post/{kind}/count'] for kind in KINDS],
                             [100*batch,200*batch,1000*batch])
        self.assertFalse(any(key.startswith('all_seen/') for key in self.payloads[4]))

    def test_exact_token_micro_not_prompt_or_chunk_average_and_N_desired_true(self):
        payload = self.payloads[1]
        # R: desired new: 50*0+50*4 / (50*1+50*4) = .8, not macro .5.
        self.assertEqual(payload['current/post/R/token_acc_pct'], 80.)
        self.assertEqual(payload['current/post/R/prompt_acc_pct'], 50.)
        # N: desired true: 50*2+50*0 / (50*2+50*6) = .25, not new .8.
        self.assertEqual(payload['current/post/N/token_acc_pct'], 25.)
        self.assertEqual(payload['current/post/N/prompt_acc_pct'], 50.)
        self.assertEqual(payload['current/post/N/strict_acc_pct'], 50.)
        for field in FIELDS:
            self.assertEqual(payload['w0/current/N/'+field], payload['current/post/N/'+field])
            self.assertEqual(self.payloads[5]['w0/all_seen/N/'+field],
                             self.payloads[5]['all_seen/post/N/'+field])

    def test_attachment_preserves_raw_and_existing_ordinal_cannot_be_repaired(self):
        raw = [{key:value for key,value in row.items() if key != 'occurrence_ordinal'}
               for row in self.rows[:26]]
        metadata = [{key:value for key,value in row.items() if key != 'occurrence_ordinal'}
                    for row in self.expected[:26]]
        before = copy.deepcopy(raw)
        annotated, identities = attach_occurrence_ordinals(raw, metadata)
        self.assertEqual(raw, before)
        self.assertEqual([row['occurrence_ordinal'] for row in annotated], [0]*13+[1]*13)
        self.assertEqual([row['occurrence_ordinal'] for row in identities], [0]*13+[1]*13)
        annotated[0]['occurrence_ordinal'] = 100
        with self.assertRaisesRegex(ValueError, 'ORDINAL_CONFLICT'):
            attach_occurrence_ordinals(annotated)

    def test_ordinal_token_order_missing_and_duplicate_panel_block(self):
        for field,value in (('occurrence_ordinal',1),('case_id',-17),('new_token_identity','changed')):
            broken = list(self.rows)
            broken[0] = dict(broken[0], **{field:value})
            with self.assertRaises(ValueError):
                build_curves(broken,self.expected)
        reversed_occurrences = self.rows[13:26]+self.rows[:13]+self.rows[26:]
        with self.assertRaises(ValueError):
            build_curves(reversed_occurrences,self.expected)
        missing_ordinal = list(self.expected)
        missing_ordinal[0] = {key:value for key,value in missing_ordinal[0].items() if key!='occurrence_ordinal'}
        with self.assertRaisesRegex(ValueError,'EXPECTED_ORDINAL'):
            build_curves(self.rows,missing_ordinal)
        broken = list(self.rows)
        broken[1] = dict(broken[0])
        with self.assertRaises(ValueError):
            validate_rows(broken,self.expected,require_full=True)
        with self.assertRaisesRegex(ValueError,'26000'):
            build_curves(self.rows[:-13],self.expected[:-13])

    def test_zero_measured_vs_missing_not_zero(self):
        zero = copy.deepcopy(self.rows[:1300])
        for row in zero:
            if row['kind']=='N':
                row.update(new_nll=1.,true_nll=2.,margin_true_minus_new=1.,margin_new_minus_true=-1.)
        groups = reduce_rows(zero)
        payload = metric_values('current/post',groups,100)
        self.assertEqual(payload['current/post/N/success_count'],0)
        self.assertEqual(payload['current/post/success_harmonic_pct'],0.)
        subset = metric_values('w0/current',{'N':groups['N']},100)
        self.assertNotIn('w0/current/R/count',subset)
        self.assertNotIn('w0/current/success_harmonic_pct',subset)
        self.assertEqual(metric_values('missing',{},100),{})
        self.assertEqual(reduce_rows([]),{})
        broken = dict(self.payloads[1])
        del broken['current/post/N/count']
        with self.assertRaisesRegex(ValueError,'MISSING_METRIC'):
            validate_curve_payload(broken,DEFAULT_CONFIG)

    def test_truthful_payload_accepted_only_by_task_profile_shared_axis_still_rejects(self):
        from project.run_scripts.experiment_tracking.method import validate
        payload = self.payloads[1]
        self.assertEqual(validate_curve_payload(payload,DEFAULT_CONFIG),payload)
        with self.assertRaisesRegex(ValueError,'POST_STATE_AXIS'):
            validate(payload,scientific=True)

    def test_state_axis_identity_privacy_lies_block(self):
        changes = dict(post_state_edits=100,actual_model_edits=100,actual_applied_edits=100,
            reference_only=False,evaluation_model_state='W1',edits_axis_semantics='actual_model_edits',
            reference_cohort_edits=0,edits=101,**{'fit/global_candidate':1,'raw_prompt':'forbidden'})
        for key,value in changes.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_curve_payload(dict(self.payloads[1],**{key:value}),DEFAULT_CONFIG)
        for key,value in dict(writer='ours',role='derived_comparison_snapshot',
            reference_only=False,w0_reference_schema='wrong',actual_model_edits=100).items():
            with self.subTest(config=key), self.assertRaises(ValueError):
                validate_curve_payload(self.payloads[1],dict(DEFAULT_CONFIG,**{key:value}))

    def test_optional_pre_is_same_raw_alias_not_state_or_forward(self):
        curves = build_curves(self.rows,self.expected,include_current_pre=True)
        config = dict(DEFAULT_CONFIG,include_current_pre=True)
        curve_coverage(curves,config)
        for key,value in curves[1].items():
            if key.startswith('current/post/'):
                self.assertEqual(curves[1][key.replace('current/post','current/pre',1)],value)
        self.assertEqual(curves[1]['pre_state_edits'],0)
        with self.assertRaises(ValueError):
            validate_curve_payload(curves[1],DEFAULT_CONFIG)

    def test_coverage_partial_never_fills_missing_or_claims_complete(self):
        partial = curve_coverage(self.payloads[:2],DEFAULT_CONFIG,require_complete=False)
        self.assertEqual(partial['status'],'INCOMPLETE')
        self.assertEqual(partial['missing_points'],list(range(200,2001,100)))
        self.assertEqual(partial['current_coverage'],1)
        self.assertEqual(partial['all_seen_coverage'],0)
        with self.assertRaisesRegex(ValueError,'INCOMPLETE'):
            curve_coverage(self.payloads[:2],DEFAULT_CONFIG)
        for invalid in (self.payloads+[self.payloads[20]],list(reversed(self.payloads))):
            with self.assertRaisesRegex(ValueError,'MONOTONIC_UNIQUE'):
                curve_coverage(invalid,DEFAULT_CONFIG)

    def test_finite_counts_units_nine_fields_and_success_harmonic(self):
        payload = self.payloads[5]
        for prefix in ('current/post','all_seen/post'):
            success = [payload[prefix+'/'+kind+'/success_pct'] for kind in KINDS]
            expected = 0. if any(value==0 for value in success) else 3/math.fsum(1/value for value in success)
            self.assertAlmostEqual(payload[prefix+'/success_harmonic_pct'],expected)
            for kind in KINDS:
                self.assertEqual(len([key for key in payload if key.startswith(prefix+'/'+kind+'/')]),9)
                self.assertEqual(payload[prefix+'/'+kind+'/margin_true_minus_new'],
                    payload[prefix+'/'+kind+'/true_nll']-payload[prefix+'/'+kind+'/new_nll'])
        for key,value in (('current/post/N/count',100),('current/post/N/token_acc_pct',101),
                          ('current/post/R/true_nll',float('nan')),
                          ('current/post/R/true_nll',-1)):
            with self.assertRaises(ValueError):
                validate_curve_payload(dict(payload,**{key:value}),DEFAULT_CONFIG)


if __name__=='__main__':
    unittest.main()
