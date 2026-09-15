"""Narrow CPU mode/finite/legacy-byte/routing tests. No Llama or CUDA."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import torch
from .policy import NumericalPolicy,PolicyError,build_correction,materialize_candidates
from .gate_skip import TASK,SWEEP_TASK,MODE,diagnostic
from .sweep_control import ARMS,DISPATCH,verify_arm

class CapTests(unittest.TestCase):
    def test_modes_and_null(self):
        for value in (1,10,100):NumericalPolicy(alpha_cap=value)
        self.assertIsNone(NumericalPolicy(alpha_cap_mode='disabled',alpha_cap=None).alpha_cap)
        for mode,value in [('bad',1),('bounded',None),('bounded',0),('bounded',float('inf')),
                           ('bounded',float('nan')),('bounded',True),('disabled',1),('disabled',float('inf'))]:
            with self.subTest(mode=mode,value=value),self.assertRaises(PolicyError):
                NumericalPolicy(alpha_cap_mode=mode,alpha_cap=value)

    def test_disabled_still_validates_numerics(self):
        for kw in ({'zeta':.5},{'epsilon_num':0},{'epsilon_num':float('nan')},{'trust_rtol':-1}):
            with self.assertRaises(PolicyError):NumericalPolicy(alpha_cap_mode='disabled',alpha_cap=None,**kw)

    @staticmethod
    def fixture(scale):
        return dict(Vp=torch.eye(2)*2,Wentry=torch.zeros(2,2),Zp=torch.zeros(2,2),
            anchors=torch.zeros(2,2),radii=torch.ones(2)*10,A=torch.eye(2),
            gE=torch.zeros(2,2),gD=-torch.ones(2,2)*scale)

    def test_legacy_exact_arithmetic_and_candidates(self):
        path=Path('/data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/gate-skip-r1/source-v1/project/run_scripts/bg_tw_reference/ep_tw/policy.py')
        spec=importlib.util.spec_from_file_location('ep_cap1_legacy_fixture',path)
        old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
        for scale in (.0001,.1,1.,10.):
            for cap in (1.,10.,100.):
                f=self.fixture(scale)
                a=old.build_correction(**f,config=old.NumericalPolicy(alpha_cap=cap))
                before=torch.random.get_rng_state().clone()
                b=build_correction(**f,config=NumericalPolicy(alpha_cap=cap))
                self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
                self.assertTrue(torch.equal(a.C,b.C));self.assertEqual(a.diagnostics['alpha'],b.diagnostics['alpha'])
                self.assertEqual(a.status,b.status)
                ca=old.materialize_candidates(f['Vp'],a.C,f['A'],native_delta_norm=a.diagnostics['actual_native_delta_norm'])
                cb=materialize_candidates(f['Vp'],b.C,f['A'],native_delta_norm=b.diagnostics['actual_native_delta_norm'])
                self.assertEqual([x.sha256 for x in ca],[x.sha256 for x in cb])

    def test_disabled_norm_and_trust(self):
        f=self.fixture(.0001);p=NumericalPolicy(alpha_cap_mode='disabled',alpha_cap=None)
        b=build_correction(**f,config=p)
        self.assertEqual(b.diagnostics['alpha'],b.diagnostics['alpha_norm'])
        self.assertGreater(b.diagnostics['alpha'],100)
        self.assertFalse(b.diagnostics['alpha_cap_active'])
        self.assertLessEqual(b.diagnostics['mapped_correction_norm'],b.diagnostics['native_trust_limit']*(1+p.trust_rtol))
        json.dumps(b.to_dict(),allow_nan=False)
        f['Vp'][0,0]=float('nan')
        with self.assertRaises(PolicyError):build_correction(**f,config=p)

    def test_single_arm_dispatch_scope(self):
        root=Path(__file__).resolve().parents[4]
        for arm in ARMS:
            d=json.loads((root/(DISPATCH+'/arms/'+arm+'.json')).read_text());verify_arm(d)
            d['new_scientific_chains']=3
            with self.assertRaises(PolicyError):verify_arm(d)

    def test_sweep_waiver_callback_never_called(self):
        lock=dict(instruction_id=SWEEP_TASK,validation_mode=MODE,numerical_validation='NOT_ESTABLISHED',
            validation_override=dict(instruction_id=TASK,inherited_by=SWEEP_TASK))
        def forbidden():raise RuntimeError('FORBIDDEN_DIAGNOSTIC')
        self.assertEqual(diagnostic(lock,'FD_ULP_SELFKL',forbidden)['calls'],0)
        lock['validation_override']['inherited_by']='wrong'
        with self.assertRaises(AssertionError):diagnostic(lock,'FD',forbidden)

if __name__=='__main__':unittest.main()
