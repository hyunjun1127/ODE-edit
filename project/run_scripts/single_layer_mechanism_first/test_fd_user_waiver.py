import copy
import unittest
from unittest.mock import patch
from .reuse_t0_user_waiver import assess, FD_WAIVER_ID
from .test_b1_repair import b1_lock
from .test_program_integration import ProgramHarness


def fixture():
    names=('native_dense_replay','Current_Q_geometry','reference_repeat','panel_gradient_basis',
           'Current_affine_physical_invariant','cross_term_STEP_CUM','restore')
    results={key:dict(pass_=True) for key in names}
    results['pair_factor_AD_FD']=dict(pass_=False,rows=[dict(index=i,gradient_relative=0.,
        scalar_absolute_gap=1e-5,coefficient_relative_gap=1e-6,
        FD=dict(status='NO_RESOLVED_ADJACENT_WINDOW')) for i in (57,125,332,337)])
    return dict(status='NOT_ESTABLISHED',pass_=False,blocking=['pair_factor_AD_FD'],results=results)


POLICY=dict(id=FD_WAIVER_ID,user_quote='통과할테니 task 이어서 진행해')


class Tests(unittest.TestCase):
    def test_preserves_failure_and_original(self):
        original=fixture();before=copy.deepcopy(original)
        value=assess(original,POLICY)
        self.assertEqual(original,before)
        self.assertFalse(value['pass_']);self.assertTrue(value['user_authorized_B1'])
        self.assertEqual(value['full_numerical_validation'],'NOT_ESTABLISHED')
        self.assertFalse(value['method_guards_changed'])

    def test_narrow_authority_and_nonFD_finite_checks(self):
        with self.assertRaises(ValueError):assess(fixture(),{})
        for key in ('restore','native_dense_replay','Current_Q_geometry'):
            value=fixture();value['results'][key]['pass_']=False
            with self.assertRaises(ValueError):assess(value,POLICY)
        for key in ('gradient_relative','scalar_absolute_gap','coefficient_relative_gap'):
            value=fixture();value['results']['pair_factor_AD_FD']['rows'][0][key]=float('nan')
            with self.assertRaises(ValueError):assess(value,POLICY)

    def test_B1_runs_without_false_ready_or_sequential(self):
        h=ProgramHarness(b1=True);h.rt.lock=dict(b1_lock(),completed_T0_reuse={'job_id':'51057'})
        with patch('project.run_scripts.single_layer_mechanism_first.reuse_t0_user_waiver.reuse',
                   return_value=assess(fixture(),POLICY)):
            h.run()
        self.assertEqual(len(h.batch_calls),1)
        self.assertTrue(h.evidence.ending('T0_B1_USER_WAIVER.json'))
        self.assertFalse(h.evidence.ending('T0_READY.json'))
        terminal=h.evidence.ending('terminal.json')[0]
        self.assertEqual(terminal['full_numerical_validation'],'NOT_ESTABLISHED')
        self.assertFalse(terminal['sequential_authorized'])


if __name__=='__main__':unittest.main()
