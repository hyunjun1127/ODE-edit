import copy
from pathlib import Path
import unittest
from .gate import calibrated_budget, CalibrationMissing, scientific_admission

class GateTests(unittest.TestCase):
    def setUp(self):
        self.dispatch=Path(__file__).resolve().parents[3]/'plans/global/2026-09-15-bg1-c4-ours-first-dispatch-contract.json'
        self.c=dict(status='CALIBRATION10_FORWARD_VERIFIED',initial_model='pre-edit W0',policy='N4_L4_L2_1',
            editing_reruns=0,reference_role='S64',documents=64,teacher_dtype='float32',sample_prefix=[0,1000],
            reference_identity_sha256='a'*64,teacher_sha256='b'*64,numerical_floor=1e-6,fixed_budget=.9,
            endpoints=[dict(batch=i,ordinal_end=i*100,D64=i/10,actual_weight_sha256='c'*64,
                reference_identity_sha256='a'*64,teacher_sha256='b'*64,state_restored_and_forward_only=True) for i in range(1,11)])
    def test_no_partial_max_or_missing_state_fill(self):
        for ids in ([1,5,10],[],[1]*10):
            c=copy.deepcopy(self.c);c['endpoints']=[self.c['endpoints'][i-1] for i in ids]
            with self.assertRaises(CalibrationMissing):calibrated_budget(self.dispatch,c)
    def test_exact_ten_forward_binding(self):
        self.assertEqual(calibrated_budget(self.dispatch,self.c),.9)
        for key,value in [('initial_model','warm W50'),('editing_reruns',1),('fixed_budget',.3)]:
            c=copy.deepcopy(self.c);c[key]=value
            with self.assertRaises(AssertionError):calibrated_budget(self.dispatch,c)
    def test_teacher_and_math_ready_are_not_runner_or_G0(self):
        r=dict(policies=['BG-1'],chains=1,initial_model='pre-edit W0',ordinals=[0,1000],calibration=self.c,
               reference768_sealed=True,teacher192_sealed=True,model_adapter_validated=False)
        with self.assertRaises(AssertionError):scientific_admission(self.dispatch,r)
        for k in ['model_adapter_validated','native_relation_gpu_checked','persistent_runner_validated','resource_admitted']:r[k]=True
        self.assertFalse(scientific_admission(self.dispatch,r)['G0_PASS'])

if __name__=='__main__':unittest.main()
