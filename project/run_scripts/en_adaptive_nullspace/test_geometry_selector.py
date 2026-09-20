"""Independent small-matrix algebra checks; these do not assert model T0 PASS."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from .geometry import build_geometry, nested_weights, first_representatives, request_alias_weights
from .selector import select_arms, frontier, choose
from .controller import run_controller


class GeometrySelectorTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(9)
        v = np.linalg.qr(rng.normal(size=(7, 5)))[0]
        k = v @ np.diag([3., 2., .2, 0., 0.]) @ rng.normal(size=(5, 9))
        w = np.arange(1., 10.); w /= w.sum()
        geo = build_geometry(k, w, np.arange(9), v, block_columns=4)
        g = rng.normal(size=(3, 7)); delta = rng.normal(size=(3, 7))
        return geo, g, delta

    def test_weighted_spectrum_vs_direct_svd(self):
        geo, _, _ = self.fixture()
        direct = np.linalg.svd(geo.basis.T @ (geo.keys*np.sqrt(geo.weights)), compute_uv=False)
        np.testing.assert_allclose(geo.singular, direct, atol=1e-14, rtol=1e-12)
        self.assertEqual(geo.rank, 3)
        self.assertEqual(geo.diagnostic['svd_calls'], 1)

    def test_energy_and_action_match_explicit_projection(self):
        geo, g, delta = self.fixture()
        spectrum = geo.gradient_spectrum(g, delta, .3)
        rows = frontier(**{k:spectrum[k] for k in ('eigenvalues','mode_energies','exact_energy','loss','native_norm','native_action','group_ends')}, epsilon=.05)
        for row in rows:
            h = geo.direction(g, row)
            self.assertAlmostEqual(np.linalg.norm(h)**2, row['gradient_energy'], places=10)
            self.assertAlmostEqual(geo.action_norm(h)**2, row['unit_coefficient_action_squared'], places=10)
        self.assertLess(abs(spectrum['energy_reconstruction_gap']), 1e-10)

    def test_nested_weights_duplicate_sequence_invariance(self):
        w, _ = nested_weights([dict(sequences=[dict(identity='a',columns=[0,1]),dict(identity='a',columns=[2,3]),dict(identity='b',columns=[4])]),dict(sequences=[dict(identity='c',columns=[5,6])])],7)
        np.testing.assert_allclose(w, [.0625,.0625,.0625,.0625,.25,.25,.25])
        self.assertEqual(first_representatives(['a','b','a']).tolist(), [0,1,0])

    def test_direct_exact_energy_preserves_small_residual(self):
        k = np.array([[1., 2.], [0., 0.], [0., 0.]])
        geo = build_geometry(k, [.5,.5], [0,1], np.eye(3))
        g = np.array([[1., 1e-10, 0.]])
        spectrum = geo.gradient_spectrum(g, np.ones_like(g), 1.)
        self.assertAlmostEqual(spectrum['exact_energy']/1e-20, 1.)
        comparison = geo.compare_exact_space(np.array([[1.], [0.], [0.]]))
        self.assertLess(comparison['projector_difference_frobenius'], 1e-14)
        self.assertEqual(comparison['extra_svd_calls'], 0)

    def test_request_response_uses_own_alias_weights(self):
        # Same two columns shared by both requests, different own mass.
        manifest = dict(case_ids=['a','b'], aliases=[
            dict(case_id='a',actual_key_column=0,weight=.45),
            dict(case_id='a',actual_key_column=1,weight=.05),
            dict(case_id='b',actual_key_column=0,weight=.05),
            dict(case_id='b',actual_key_column=1,weight=.45)])
        maps = request_alias_weights(manifest)
        geo = build_geometry(np.eye(2), [.5,.5], [0,1], np.eye(2))
        d = np.array([[1.,3.]])
        diagnostic = geo.diagnostics(d,d,request_columns=maps)
        self.assertAlmostEqual(diagnostic['request_response'][0]['conditional_RMS'], np.sqrt(1.8))
        self.assertAlmostEqual(diagnostic['request_response'][1]['conditional_RMS'], np.sqrt(8.2))
        self.assertAlmostEqual(diagnostic['actual_response']**2, 5.)
        self.assertNotEqual(diagnostic['request_response'][0]['conditional_RMS'], diagnostic['request_response'][1]['conditional_RMS'])

    def test_negative_signed_loss_is_preserved_no_signal(self):
        rows = frontier([1.],[1.],exact_energy=1.,loss=-1e-8,native_norm=1.,native_action=1.,epsilon=.05)
        self.assertTrue(all(row['status']=='NO_SIGNAL' and row['eta']==0 for row in rows))
        self.assertTrue(all(row['native_signed_loss']==-1e-8 and not row['loss_was_clamped'] for row in rows))

    def test_numerical_duplicate_witness(self):
        k = np.array([[1.,1.],[0.,1e-7],[0.,0.]])
        geo = build_geometry(k, [.5,.5], [0,0], np.eye(3))
        self.assertAlmostEqual(geo.diagnostic['duplicate_difference_frobenius'], 1e-7/np.sqrt(2))
        self.assertEqual(geo.numerical_released, 1)
        self.assertEqual(geo.keys.shape[1], 2)

    def test_reference_scalar_agreement(self):
        root = Path(__file__).resolve().parents[3]
        path = root/'audits/global/2026-09-20-en-adaptive-nullspace-design-v1/selector_reference.py'
        spec = importlib.util.spec_from_file_location('sealed_reference', path)
        ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
        args = dict(eigenvalues=[.01,.01,.2],mode_energies=[1.,3.,2.],exact_energy=.4,loss=.2,native_norm=2.,native_action=3.,epsilon=.05)
        ours, expected = frontier(**args), ref.frontier(**args)
        self.assertEqual([r['released_modes'] for r in ours], [0,2,3])
        for row, other in zip(ours, expected):
            for field in ('eta','predicted_decrease','correction_norm','response_norm'):
                self.assertEqual(row[field],other[field])
        self.assertEqual(choose(ours)['released_modes'], ref.choose(expected)['released_modes'])

    def test_zero_response_and_native_action(self):
        rows = frontier([1.],[10.],exact_energy=1.,loss=.2,native_norm=1.,native_action=0.,epsilon=.05)
        self.assertGreater(rows[0]['eta'],0.)
        self.assertEqual(rows[1]['eta'],0.)
        with self.assertRaises(ValueError):
            frontier([1.,1.],[1.,1.],exact_energy=0.,loss=1.,native_norm=1.,native_action=1.,epsilon=.05,group_ends=[1,2])

    def test_controller_quadratic_actual_rounding_and_immutable_center(self):
        geo = build_geometry(np.eye(2), [.5,.5], [0,1], np.eye(2))
        wn = np.array([[.1,.2]], dtype=np.float32); original = wn.copy()
        g = np.array([[1.,0.]])
        calls=[]
        def objective(w):
            x=float(np.asarray(w)[0,0]-wn[0,0]); calls.append(x)
            return dict(J=1.+x+2*x*x, L_R=1.+x+2*x*x, L_H=0.)
        result=run_controller(wn,g,np.array([[-1.,0.]]),dict(J=1.),objective,geo,
                              native_norm=2.,native_action=100.,input_identity='fixed',epsilon=.05)
        np.testing.assert_array_equal(wn,original)
        self.assertEqual(result['status'],'ACCEPTED')
        self.assertEqual(result['evaluations'],2)
        self.assertAlmostEqual(result['ledger'][1]['scale'],.25,places=6)
        self.assertIn('rounding_response',result['ledger'][1]['geometry'])
        self.assertEqual(result['selected_trial'],'candidate2')

    def test_controller_cache_requires_signed_zero_byte_identity(self):
        geo = build_geometry(np.eye(2), [.5,.5], [0,1], np.eye(2))
        wn=np.zeros((1,2),np.float32); g=np.array([[1.,0.]])
        cached=np.array([[-.1,-0.]],np.float32)
        cache=[dict(weight=cached,input_identity='x',objective={'J':.9},label='signed-zero-other')]
        calls=[]
        def objective(w): calls.append(w.copy()); return dict(J=.9)
        result=run_controller(wn,g,-g*.1,1.,objective,geo,native_norm=1.,native_action=100.,input_identity='x',objective_cache=cache)
        self.assertEqual(len(calls),1)
        self.assertIsNone(result['ledger'][0]['alias'])

    def test_controller_native_fallback_and_cross_arm_exact_alias(self):
        geo = build_geometry(np.eye(2), [.5,.5], [0,1], np.eye(2))
        wn=np.zeros((1,2),np.float32); g=np.array([[1.,0.]]); d=-g*.1; cache=[]
        def bad(w): return dict(J=2.)
        a=run_controller(wn,g,d,1.,bad,geo,native_norm=1.,native_action=100.,input_identity='x',objective_cache=cache)
        b=run_controller(wn,g,d,1.,bad,geo,native_norm=1.,native_action=100.,input_identity='x',objective_cache=cache)
        self.assertEqual(a['status'],'SEARCH_LIMIT_NO_ACCEPTED_CANDIDATE')
        self.assertEqual(b['evaluations'],0)
        self.assertTrue(all(row['alias'] for row in b['ledger']))


if __name__ == '__main__':
    unittest.main()
