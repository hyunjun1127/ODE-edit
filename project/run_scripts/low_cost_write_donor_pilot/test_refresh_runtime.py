"""CPU fixtures only; not model-level fidelity or GPU continuation evidence."""
import copy
import unittest
from unittest.mock import patch

from . import refresh_evaluation as re
from .refresh_runtime import (BatchBarrier, POLICIES, assert_no_inner_history,
                              project_l4_state)
from .sequential_runtime import SequentialLedger, evaluate_nonmutating
from .test_sequential_evaluation import record, fake_counterfact


class RefreshRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = [record(i) for i in range(10000)]

    def test_policies_and_target_writer_budgets(self):
        self.assertEqual(list(POLICIES), ['FROZEN2','I2','FROZEN4','I4'])
        for policy,cfg in POLICIES.items():
            self.assertEqual(sum(cfg['caps']),24)
            self.assertEqual(cfg['gammas'][-1],1.)
            self.assertEqual(len(cfg['caps']),len(cfg['gammas']))
            barrier = BatchBarrier(100,len(cfg['caps']))
            for _ in cfg['caps']:
                for i in range(100):
                    barrier.target(i)
                barrier.write()
            barrier.finalize()
            with self.assertRaises(AssertionError):
                barrier.finalize()

    def test_no_b1_write_or_duplicate_or_early_finalize(self):
        barrier = BatchBarrier(100,2)
        barrier.target(0)
        for action in [barrier.write,barrier.finalize,lambda:barrier.target(0)]:
            with self.assertRaises(AssertionError):
                action()

    def test_l4_projection_and_inner_history_guard(self):
        state = dict(weights={'4':'w4','8':'w8'},M4='m4',M8='m8',P4='p4',P8='p8',contexts='ctx',rng='rng')
        selected = project_l4_state(state)
        self.assertEqual(set(selected),{'weights','M4','P4','contexts','rng'})
        self.assertEqual(selected['weights'],{'4':'w4'})
        after = copy.deepcopy(selected);after['weights']['4']='new'
        assert_no_inner_history(selected,after)
        for key in ['M4','P4','contexts']:
            bad = dict(after);bad[key]='changed'
            with self.assertRaises(AssertionError):
                assert_no_inner_history(selected,bad)

    def test_commit_next_entry_and_eval_mutation(self):
        entry = {'W':'w','M':'m','rng':'r'}
        ledger = SequentialLedger(entry,[4]);ledger.begin(51,entry)
        endpoint = {'W':'w1','M':'m1','rng':'r'}
        ledger.commit(endpoint,[dict(layer=4,history_append=1)])
        with self.assertRaises(AssertionError):
            ledger.begin(52,entry)
        ledger.begin(52,endpoint)
        mutable = dict(endpoint)
        with self.assertRaises(AssertionError):
            evaluate_nonmutating(lambda:dict(mutable),lambda:1,lambda:mutable.update(W='bad'))

    def evaluate(self,batch):
        with patch.object(re.evaluation,'counterfact',side_effect=fake_counterfact) as cf, \
             patch.object(re.evaluation,'wiki',return_value={'denominator':128}) as wiki, \
             patch.object(re.evaluation,'mmlu_alternative',return_value={'denominator':32}) as mmlu:
            out = re.evaluate_batch(None,None,self.records,batch,list(range(128)),{},list(range(32)),{'W':'w','M':'m'})
            return out,[len(c.args[2]) for c in cf.call_args_list],wiki.call_count,mmlu.call_count

    def test_terminal_one_fullseen_forward_and_subsets(self):
        out,counts,wiki,mmlu = self.evaluate(60)
        self.assertEqual(counts,[6000])
        self.assertEqual((wiki,mmlu),(1,1))
        self.assertEqual([r['case_id'] for r in out['first_suffix500']['metrics']['RS']['rows']],list(range(5000,5500)))
        self.assertEqual(out['first_suffix500']['metrics']['NS']['denominator'],5000)
        self.assertEqual(out['suffix']['metrics']['NS']['denominator'],10000)
        self.assertEqual(out['fullseen']['metrics']['NS']['denominator'],60000)
        self.assertEqual(out['strict']['current']['two_P_request']['denominator'],100)
        self.assertTrue(all(item['added_forwards']==0 for item in out['reuse']))

    def test_nonterminal_general_not_evaluated(self):
        for batch,expected in [(51,[100,128]),(55,[500,128])]:
            out,counts,wiki,mmlu = self.evaluate(batch)
            self.assertEqual(counts,expected)
            self.assertEqual((wiki,mmlu),(0,0))
            self.assertNotIn('wiki',out)
            self.assertEqual(out['audit_evaluations'],0)
            if batch == 55:
                self.assertEqual([r['case_id'] for r in out['first_suffix500']['metrics']['RS']['rows']],list(range(5000,5500)))

    def test_two_paraphrase_request_strict_not_prompt_pooling(self):
        doc = fake_counterfact(None,None,self.records[:2],panel='Current')
        for row in doc['metrics']['PS']['rows']:
            row['new_strict'] = row['case_id'] == 0 or row['prompt_index'] == 0
        result = re.strict_summary(doc)
        self.assertEqual(result['PS'],dict(numerator=3,denominator=4))
        self.assertEqual(result['two_P_request'],dict(numerator=1,denominator=2))


if __name__ == '__main__':
    unittest.main()
