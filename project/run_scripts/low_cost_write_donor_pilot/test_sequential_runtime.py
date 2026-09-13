"""Small CPU contracts for the new orchestration; no model/GPU smoke."""
import unittest
from .sequential_runtime import (POLICIES, batch_bounds, selected_layers,
                                 validate_batch_lock, assert_fit_history_unchanged,
                                 SequentialLedger, evaluate_nonmutating, restore_and_verify)
import copy
from project.run_scripts.baseline_mechanism_first.contracts import digest


class SequentialRuntimeTests(unittest.TestCase):
    def test_exact_suffix_boundaries(self):
        self.assertEqual([batch_bounds(b) for b in range(51,61)],
                         [(n,n+100) for n in range(5000,6000,100)])
        for wrong in (0,50,61,51.0):
            with self.assertRaises(ValueError):
                batch_bounds(wrong)

    def test_policy_support_and_counts(self):
        self.assertEqual(list(POLICIES),['N4','RES8','S875','S75','FULL8','REFIT4'])
        self.assertEqual(selected_layers('REFIT4'),[4])
        self.assertEqual(selected_layers('RES8'),[4,8])
        self.assertEqual(sum(10*(1+(second is not None)) for _,second in POLICIES.values()),90)
        self.assertEqual(sum(1000*(1+(second is not None)) for _,second in POLICIES.values()),9000)

    def test_order_target_mutation_fails(self):
        rows=[{'case_id':i,'requested_rewrite':{'target_new':str(i)}} for i in range(6000)]
        current=rows[5000:5100]
        ids=[r['case_id'] for r in current]
        lock=dict(batch=51,case_ids=ids,request_order_sha256=digest(ids),records_sha256=digest(current))
        self.assertEqual(validate_batch_lock(rows,51,lock),current)
        rows[5000]['requested_rewrite']['target_new']='changed'
        with self.assertRaisesRegex(AssertionError,'TARGET_SHA'):
            validate_batch_lock(rows,51,lock)

    def test_no_history_append_between_fits(self):
        state=dict(M4='4',M8='8',P4='p4',P8='p8',contexts='ctx')
        assert_fit_history_unchanged(state,state.copy())
        for key in state:
            with self.assertRaises(AssertionError):
                assert_fit_history_unchanged(state,dict(state,**{key:'bad'}))

    def test_production_transaction_two_batches_and_rollback(self):
        model_state={'W4':50,'W8':0,'M4':50,'M8':50,'RNG':0}
        read=lambda:copy.deepcopy(model_state)
        ledger=SequentialLedger(read(),[4,8])
        for batch in (51,52):
            entry=read()
            ledger.begin(batch,entry)
            model_state['W4']+=1
            model_state['W8']+=2
            # Two fits have no append; endpoint finalization appends once.
            self.assertEqual((model_state['M4'],model_state['M8']),(entry['M4'],entry['M8']))
            model_state['M4']+=1
            model_state['M8']+=1
            endpoint=read()
            self.assertEqual(evaluate_nonmutating(read,lambda:(1,2),lambda:{'n':100}),{'n':100})
            ledger.commit(endpoint,[{'layer':4,'history_append':1},{'layer':8,'history_append':1}])
        self.assertEqual(ledger.history_counts,{4:2,8:2})
        self.assertEqual(model_state['W4'],52)
        with self.assertRaisesRegex(AssertionError,'NEXT_ENTRY'):
            ledger.begin(53,dict(model_state,W4=50))
        endpoint=read()
        model_state['W4']=-1
        restore_and_verify(lambda s:(model_state.clear(),model_state.update(s)),endpoint,read,endpoint)
        self.assertEqual(read(),endpoint)

    def test_eval_mutation_and_duplicate_finalize_fail(self):
        state={'W':1}
        with self.assertRaisesRegex(AssertionError,'EVALUATION_MUTATION'):
            evaluate_nonmutating(lambda:dict(state),lambda:(0,0),lambda:state.update(W=2))
        ledger=SequentialLedger(state,[4])
        ledger.begin(51,state)
        with self.assertRaisesRegex(AssertionError,'FINALIZE_COUNTS'):
            ledger.commit(state,[{'layer':4,'history_append':1},{'layer':4,'history_append':1}])


if __name__=='__main__':
    unittest.main()
