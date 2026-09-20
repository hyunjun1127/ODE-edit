import copy
import unittest
from .review_server4 import validate_reduce,transitions,replay_controller


def fixture():
    rows=[]
    for family,count in [('RS',1),('PS',2),('NS',10)]:
        for case in [7,9]:
            for pos in range(count):
                row=dict(family=family,case_id=case,prompt_index=pos,identity=f'{family}:{case}:{pos}',token_identity='t',
                    new_nll=2. if family=='NS' else 1.,true_nll=1. if family=='NS' else 2.,success=True)
                for side in ['new','true']:row.update({side+'_token_correct':[True,False],side+'_token_count':2,side+'_strict':False})
                rows.append(row)
    return dict(request_ids=[7,9],rows=rows)


class ReviewTests(unittest.TestCase):
    def test_exact_denominators_and_token_accuracy(self):
        result,_=validate_reduce(fixture())
        self.assertEqual([r['denominator'] for r in result],[2,4,20])
        self.assertTrue(all(r['tf_token_micro']==.5 for r in result))
        self.assertTrue(all(r['strict_numerator']==0 for r in result))

    def test_ties_are_failure_and_stored_success_is_checked(self):
        a=fixture();r=a['rows'][0];r['new_nll']=r['true_nll']
        with self.assertRaisesRegex(ValueError,'STORED_PREFERENCE'):validate_reduce(a)
        r['success']=False;result,_=validate_reduce(a)
        self.assertEqual(result[0]['numerator'],1)

    def test_same_total_not_same_success_set(self):
        a=fixture();b=copy.deepcopy(a)
        a['rows'][0].update(new_nll=3.,success=False)
        b['rows'][1].update(new_nll=3.,success=False)
        changes=transitions(a,b,[7,9])
        self.assertEqual(sum(r['lost'] for r in changes),1)
        self.assertEqual(sum(r['gained'] for r in changes),1)

    def test_missing_duplicate_nonfinite_and_wrong_target_rejected(self):
        a=fixture();a['rows'].pop()
        with self.assertRaises(ValueError):validate_reduce(a)
        a=fixture();a['rows'].append(a['rows'][0])
        with self.assertRaises(ValueError):validate_reduce(a)
        a=fixture();a['rows'][0]['new_nll']=float('nan')
        with self.assertRaises(ValueError):validate_reduce(a)
        a=fixture();b=copy.deepcopy(a);b['rows'][0]['token_identity']='different'
        with self.assertRaises(ValueError):transitions(a,b,[7,9])

    def test_empty_active_past_is_not_filled_with_previous_cohort(self):
        a,_=validate_reduce(fixture(),[])
        self.assertTrue(all(r['denominator']==0 and r['percent'] is None for r in a))

    def test_selector_replay_rejects_mismatched_decision(self):
        c=dict(status='ACCEPTED',selected_trial='candidate1',ledger=[dict(trial='candidate1',
            objective={'J':.9},native_objective={'J':1.},actual_gradient_inner_product=-.1,
            accepted=True,geometry_checks={'finite':True},geometry={'actual_norm':.5})])
        self.assertEqual(replay_controller(c)['selected_trial'],'candidate1')
        c['ledger'][0]['objective']['J']=1.1
        with self.assertRaises(ValueError):replay_controller(c)


if __name__=='__main__':unittest.main()
