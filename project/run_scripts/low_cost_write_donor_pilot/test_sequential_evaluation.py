"""CPU-only fixture tests: no model, no evaluator forward, no GPU."""
import copy
import unittest
from unittest.mock import patch

from project.run_scripts.baseline_mechanism_first.contracts import ContractBoundary, digest
from project.run_scripts.baseline_mechanism_first.case_population import source_digest
from project.run_scripts.baseline_mechanism_first.performance_schema import MULTIPLICITY, from_rows
from . import sequential_evaluation as se


def record(i):
    return dict(case_id=i, requested_rewrite=dict(prompt='{} lives', subject=str(i),
        relation_id='r', target_new={'str': 'new'}, target_true={'str': 'true'}),
        paraphrase_prompts=[f'{i} p{j}' for j in range(2)],
        neighborhood_prompts=[f'{i} n{j}' for j in range(10)])


def fake_counterfact(model, tokenizer, records, *, panel, annotations=None):
    groups = {}
    for tag in MULTIPLICITY:
        rows = []
        for r in records:
            w = r['requested_rewrite']
            prompts = ([w['prompt'].format(w['subject'])] if tag == 'RS' else
                       r['paraphrase_prompts' if tag == 'PS' else 'neighborhood_prompts'])
            for i, prompt in enumerate(prompts):
                new, true = (1., 1.) if r['case_id'] % 2 else (1., 2.)
                rows.append(dict(case_id=r['case_id'], prompt_index=i,
                    identity=source_digest([r['case_id'], i, prompt, 'new', 'true']),
                    new_nll=new, true_nll=true, margin=true-new,
                    desired_margin=new-true if tag == 'NS' else true-new,
                    success=(true < new if tag == 'NS' else new < true), panel=panel,
                    new_strict=False, true_strict=False))
        groups[tag] = {'rows': rows}
    return dict(from_rows([{'metrics': groups}], records), panel=panel,
                evaluator_layout='TEST_MB16')


class SequentialEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = [record(i) for i in range(10000)]

    def run_batch(self, batch):
        with patch.object(se.evaluation, 'counterfact', side_effect=fake_counterfact) as cf, \
             patch.object(se.evaluation, 'wiki', return_value={'denominator': 128}), \
             patch.object(se.evaluation, 'mmlu_alternative', return_value={'denominator': 32}):
            out = se.evaluate_batch(None, None, self.records, batch, list(range(128)), {},
                                    list(range(32)), {'W4': 'exact', 'M4': 'exact'})
            counts = [len(c.args[2]) for c in cf.call_args_list]
        return out, counts

    def test_terminal_full6000_single_forward_subsets(self):
        out, counts = self.run_batch(60)
        self.assertEqual(counts, [6000])
        self.assertEqual(out['fullseen']['metrics']['NS']['denominator'], 60000)
        self.assertEqual(out['suffix']['requests'], 1000)
        self.assertEqual(out['entry_old']['requests'], 5000)
        self.assertEqual(out['current']['metrics']['PS']['denominator'], 200)
        self.assertEqual(out['historical']['requests'], 128)
        self.assertEqual(out['current']['metrics']['RS']['numerator'], 50)
        self.assertEqual(out['fullseen']['metrics']['NS']['numerator'], 0)
        self.assertEqual(out['current']['metrics']['RS']['rows'][0]['case_id'], 5900)
        self.assertEqual(out['current']['metrics']['RS']['rows'][-1]['case_id'], 5999)
        self.assertTrue(all(r['added_forwards'] == 0 for r in out['reuse']))

    def test_b55_suffix_reuses_current(self):
        out, counts = self.run_batch(55)
        self.assertEqual(counts, [500, 128])
        self.assertEqual(out['current']['metrics']['RS']['rows'][0]['case_id'], 5400)
        self.assertNotIn('fullseen', out)

    def test_b51_current_and_history_only(self):
        out, counts = self.run_batch(51)
        self.assertEqual(counts, [100, 128])
        self.assertEqual(out['current']['metrics']['RS']['rows'][0]['case_id'], 5000)
        self.assertEqual(out['audit_evaluations'], 0)

    def test_subset_rejects_identity_state_duplicate_nonfinite(self):
        records = self.records[:2]
        doc = fake_counterfact(None, None, records, panel='source')
        state = {'W': 'exact'}
        doc['endpoint_state_sha256'] = digest(state)
        bad = copy.deepcopy(records[:1]); bad[0]['requested_rewrite']['target_new']['str'] = 'changed'
        for selected, ep in [(bad, state), ([records[0], records[0]], state), (records[:1], {'W': 'wrong'})]:
            with self.assertRaises(ContractBoundary):
                se.exact_subset(doc, records, selected, panel='dest', endpoint_state=ep)
        for field, value in [('new_nll', float('nan')), ('identity', 'wrong'), ('success', True)]:
            broken = copy.deepcopy(doc)
            broken['metrics']['NS']['rows'][0][field] = value
            with self.assertRaises(ContractBoundary):
                se.exact_subset(broken, records, records[:1], panel='dest', endpoint_state=state)

    def test_bad_batch_and_panel_fail_before_forward(self):
        with patch.object(se.evaluation, 'counterfact') as cf:
            for batch, panel in [(50, list(range(128))), (61, list(range(128))), (51, [0]*128)]:
                with self.assertRaises(ContractBoundary):
                    se.evaluate_batch(None, None, self.records, batch, panel, {}, list(range(32)), {})
            cf.assert_not_called()


if __name__ == '__main__':
    unittest.main()
