"""Synthetic scalar tests only; no model/production reducer imports."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('cpu_review', Path(__file__).with_name('review_completed.py'))
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def row(identity='r1', kind='R', case=1, new=.2, true=1., correct=1, count=1):
    return dict(identity=identity, case_id=case, kind=kind, prompt_index=0, endpoint='W1',
        new_token_identity=identity + '-new', true_token_identity=identity + '-true',
        new_nll=new, true_nll=true, margin_true_minus_new=true - new,
        new_token_count=count, new_token_correct=correct, new_strict=count == correct,
        true_token_count=count, true_token_correct=correct, true_strict=count == correct,
        active_at_endpoint=True)


class ReviewTests(unittest.TestCase):
    def test_ties_are_failure(self):
        items = [row('r', 'R', new=1, true=1), row('p', 'P', new=1, true=1),
                 row('n', 'N', new=1, true=1)]
        result = review.reduce_rows(items)
        self.assertTrue(all(result[k]['numerator'] == 0 for k in ('R', 'P', 'N')))
        self.assertEqual(review.harmonic(result), 0)

    def test_harmonic_zero_and_partial_not_zero(self):
        self.assertIsNone(review.harmonic(review.reduce_rows([row()])))
        self.assertEqual(review.reduce_rows([]), {})
        result = review.reduce_rows([row('r', 'R'), row('p', 'P'), row('n', 'N', new=1, true=.2)])
        self.assertEqual(review.harmonic(result), 1)

    def test_nonfinite_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaisesRegex(ValueError, 'NONFINITE'):
                review.reduce_rows([row(new=value)])

    def test_duplicate_missing_and_wrong_order_rejected(self):
        a, b = row('a'), row('b', case=2)
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            review.validate_rows([a, a])
        with self.assertRaisesRegex(ValueError, 'CARDINALITY'):
            review.validate_rows([a], [a, b])
        with self.assertRaisesRegex(ValueError, 'ORDER'):
            review.validate_rows([b, a], [a, b])

    def test_token_identity_and_count_relations(self):
        a = row()
        b = copy.deepcopy(a)
        b['true_token_identity'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'TOKEN_IDENTITY'):
            review.validate_rows([b], [a])
        b = copy.deepcopy(a)
        b['new_token_correct'] = 2
        with self.assertRaisesRegex(ValueError, 'COUNT'):
            review.reduce_rows([b])
        b = copy.deepcopy(a)
        b['new_strict'] = False
        with self.assertRaisesRegex(ValueError, 'STRICT'):
            review.reduce_rows([b])

    def test_margin_sign(self):
        a = row()
        a['margin_true_minus_new'] *= -1
        with self.assertRaisesRegex(ValueError, 'MARGIN'):
            review.reduce_rows([a])

    def test_paired_identity_token_and_conservation(self):
        before = [row('a', new=.2, true=1), row('b', case=2, new=1, true=.2)]
        after = [row('a', new=1, true=.2), row('b', case=2, new=.2, true=1)]
        pair = review.paired(before, after)['R']['preference']
        self.assertEqual((pair['lost'], pair['gained'], pair['before'], pair['after']), (1, 1, 1, 1))
        with self.assertRaisesRegex(ValueError, 'SET'):
            review.paired(before, after[:1])
        after[0]['new_token_count'] = 2
        after[0]['new_strict'] = False
        with self.assertRaisesRegex(ValueError, 'PAIRED_TOKEN'):
            review.paired(before, after)

    def test_quantiles_interpolation_and_sign(self):
        self.assertEqual(review.quantiles([0, 10])['p50'], 5)
        result = review.reduce_rows([row()])['R']
        self.assertEqual(result['true_minus_new_mean'], -result['new_minus_true_mean'])

    def test_source_independent_no_model_or_scheduler_import(self):
        source = Path(review.__file__).read_text()
        self.assertNotIn('import torch', source)
        self.assertNotIn('import subprocess', source)
        self.assertNotIn('from project.', source)

    def test_chunk_state_and_order_hash(self):
        items = [row('r', 'R'), row('p0', 'P'), row('p1', 'P')]
        items += [row('n' + str(i), 'N') for i in range(10)]
        for i, r in enumerate(items):
            r['prompt_index'] = i
        expected = [{k: r[k] for k in ('identity', 'case_id', 'kind', 'prompt_index',
                                      'new_token_identity', 'true_token_identity')} for r in items]
        records = [dict(case_id=1, requested_rewrite=dict(subject='a', relation_id='r',
                        target_new=dict(id='new', str='new')))]
        summary = review.reduce_rows(items)
        stored = {k: {field: value for field, value in row.items() if not field.endswith('_quantiles')}
                  for k, row in summary.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            chunk = dict(rows=items, state={'W': 'x'}, optimizer_feedback=False)
            (path / 'chunk-0000.json').write_text(json.dumps(chunk))
            receipt = dict(state={'W': 'x'}, endpoint='W1', requests=1, row_count=13,
                           row_order=review.digest([r['identity'] for r in items]),
                           no_mutation=True, optimizer_feedback=False, replay=False, summary=stored)
            (path / 'summary.json').write_text(json.dumps(receipt))
            rows, _ = review.read_endpoint(review.Reader(), path, expected, [1], 'W1', {'W': 'x'}, records)
            self.assertEqual(len(rows), 13)
            with self.assertRaisesRegex(ValueError, 'CHUNK_STATE'):
                review.read_endpoint(review.Reader(), path, expected, [1], 'W1', {'W': 'wrong'}, records)

    def test_reused_W0_subset_schema_and_exact_scalar_values(self):
        items = [row('r', 'R'), row('p0', 'P'), row('p1', 'P')]
        items += [row('n' + str(i), 'N') for i in range(10)]
        for i, r in enumerate(items):
            r.update(prompt_index=i, endpoint='W0')
        expected = [{k: r[k] for k in ('identity', 'case_id', 'kind', 'prompt_index',
                                      'new_token_identity', 'true_token_identity')} for r in items]
        records = [dict(case_id=1, requested_rewrite=dict(subject='a', relation_id='r',
                        target_new=dict(id='new', str='new')))]
        with tempfile.TemporaryDirectory() as directory:
            arm = Path(directory)
            path = arm / 'batch-01/pre'
            path.mkdir(parents=True)
            reused = [dict(r, endpoint='B1_PRE') for r in items]
            (path / 'chunk-0000.json').write_text(json.dumps(dict(rows=reused, state={'W': 'x'},
                                                                 optimizer_feedback=False)))
            receipt = dict(state={'W': 'x'}, endpoint='B1_PRE', requests=1, new_forwards=0,
                           reused_from=str(arm / 'W0'), no_mutation=True, optimizer_feedback=False,
                           summary=review.reduce_rows(reused))
            (path / 'summary.json').write_text(json.dumps(receipt))
            actual, _ = review.read_endpoint(review.Reader(), path, expected, [1], 'B1_PRE',
                                             {'W': 'x'}, records, reused_rows=items)
            self.assertEqual(actual, reused)
            changed = copy.deepcopy(items)
            changed[0]['new_nll'] += .1
            with self.assertRaisesRegex(ValueError, 'EXACT_RAW_VALUES'):
                review.read_endpoint(review.Reader(), path, expected, [1], 'B1_PRE',
                                     {'W': 'x'}, records, reused_rows=changed)


if __name__ == '__main__':
    unittest.main()
