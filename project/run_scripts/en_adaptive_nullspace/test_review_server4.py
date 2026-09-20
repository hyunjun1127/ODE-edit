import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from .review_server4 import validate_reduce,transitions,replay_controller,replay_frontier,review


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

    def test_independent_frontier_arithmetic_and_negative(self):
        from .selector import select_arms
        s=dict(eigenvalues=[.01,.01,1.],mode_energies=[.3,.2,.4],exact_energy=.1,
            loss=.2,native_norm=2.,native_action=.8,group_ends=[2,3],numerical_released=2,exact_rank=3)
        payload=dict(spectrum=s,selection=select_arms(s))
        self.assertEqual(len(replay_frontier(payload)),3)
        payload['selection']['frontiers']['0.05'][1]['eta']+=.1
        with self.assertRaisesRegex(ValueError,'FRONTIER_ARITHMETIC'):replay_frontier(payload)

    def test_zero_loss_frontier_not_forced_to_correction(self):
        from .selector import select_arms
        s=dict(eigenvalues=[1.],mode_energies=[.4],exact_energy=.1,
            loss=-1e-8,native_norm=2.,native_action=.8,group_ends=[1],numerical_released=0,exact_rank=1)
        payload=dict(spectrum=s,selection=select_arms(s))
        self.assertTrue(all(r['selected_adaptive_modes']==0 for r in replay_frontier(payload)))

    def test_own_entry_link_and_atwrite_reduction(self):
        def obs(ids):
            template=[r for r in fixture()['rows'] if r['case_id']==7]
            return dict(request_ids=list(ids),rows=[dict(r,case_id=i,identity=f'{i}:{r["identity"]}') for i in ids for r in template])
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);out=root/'output';out.mkdir()
            def write(name,value):
                p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
            for b in (1,2):
                ids=list(range((b-1)*100,b*100))
                write(f'B{b}/W0-current.json',obs(ids));write(f'B{b}/N4-metrics.json',obs(range(b*100)))
                write(f'B{b}/N4-commit.json',dict(history_append=1,active_past_ids=list(range((b-1)*100)),
                    receipt=dict(case_ids=ids,native=[dict(layer=4,history_append=1,before_sha256=f'm{b-1}',after_sha256=f'm{b}',weight_sha256=f'w{b}')])) )
                write(f'B{b}/{"SHARED" if b==1 else "N4"}-native.json',dict(receipt=dict(history_append=0,history_sha256=f'm{b-1}',entry_weight_sha256=f'w{b-1}')))
            with contextlib.redirect_stdout(io.StringIO()):result=review(out,root/'report')
            self.assertEqual(result['adjacent_state_links'],1)
            self.assertEqual(result['history_appends'],2)
            write('B2/N4-native.json',dict(receipt=dict(history_append=0,history_sha256='m1',entry_weight_sha256='other-arm')))
            with self.assertRaisesRegex(ValueError,'OWN_NEXT_ENTRY_LINK'):review(out,root/'report2')


if __name__=='__main__':unittest.main()
