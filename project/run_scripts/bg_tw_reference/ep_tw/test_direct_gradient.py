"""New CPU FP64/FP32 repair fixtures, not real Llama/GPU validation."""
from unittest import mock
import unittest

import torch

from .model_adapter import EpisodeAdapter, ModelBoundary, _RawAnchoredWeight, tensor_sha
from .test_model_adapter import TinyCausal, TinyTeacher, TinyTokenizer, records


class CustomNodeDoubleTests(unittest.TestCase):
    def test_same_custom_node_FP64_gradcheck_at_zero(self):
        torch.manual_seed(875)
        raw = torch.randn(3, 5, dtype=torch.float64)
        fixed_a = torch.randn(2, 5, dtype=torch.float64)
        correction = torch.zeros(3, 2, dtype=torch.float64, requires_grad=True)
        # Test the actual production custom Function.apply, not a new affine
        # replacement. The scientific wrapper remains FP32-only and unchanged.
        function = lambda value: _RawAnchoredWeight.apply(value, raw, fixed_a)
        self.assertTrue(torch.autograd.gradcheck(function, (correction,),
                                                eps=1e-6, atol=1e-5, rtol=1e-3))
        self.assertEqual(tensor_sha(function(correction)), tensor_sha(raw))

    def test_same_custom_node_FP64_gradcheck_nonzero(self):
        torch.manual_seed(876)
        raw = torch.randn(3, 5, dtype=torch.float64)
        fixed_a = torch.randn(2, 5, dtype=torch.float64)
        correction = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda value: _RawAnchoredWeight.apply(value, raw, fixed_a),
            (correction,), eps=1e-6, atol=1e-5, rtol=1e-3))
        cotangent = torch.randn_like(raw)
        actual = torch.autograd.grad(
            (_RawAnchoredWeight.apply(correction, raw, fixed_a)*cotangent).sum(), correction)[0]
        self.assertTrue(torch.equal(actual, cotangent @ fixed_a.T))


class DirectWeightTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(345)
        self.model = TinyCausal().float().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.teacher = TinyTeacher(self.model)
        self.adapter = EpisodeAdapter(self.model, TinyTokenizer(), 'down_proj.weight', self.teacher)
        self.records = records()
        with torch.no_grad():
            self.model.down_proj.weight.add_(.07)
        self.raw = self.model.down_proj.weight.detach().clone()
        self.a = torch.randn(len(self.records), 5) * .15
        self.adapter.set_episode(self.raw, self.a)

    def test_direct_route_bypasses_custom_node(self):
        residual = self.adapter.gradient_sweeps(self.records)
        with mock.patch.object(_RawAnchoredWeight, 'apply', side_effect=AssertionError('CUSTOM_NODE_FORBIDDEN')):
            direct = self.adapter.direct_weight_gradients(self.records)
        for g_c, g_w in [('gE', 'gWE'), ('gD', 'gWD')]:
            contracted = direct[g_w] @ self.a.T
            self.assertTrue(torch.allclose(residual[g_c], contracted, rtol=3e-5, atol=2e-7))
        self.assertEqual(direct['current']['rows'], residual['current']['rows'])
        self.assertEqual(direct['generic']['rows'], residual['generic']['rows'])
        self.assertEqual(direct['receipt']['custom_affine_node_calls'], 0)

    def test_direct_contraction_same_independent_direction(self):
        residual = self.adapter.gradient_sweeps(self.records)
        direct = self.adapter.direct_weight_gradients(self.records)
        generator = torch.Generator().manual_seed(20260915)
        direction = torch.randn(residual['gE'].shape, generator=generator)
        direction /= direction.norm()
        for c_name, w_name in [('gE','gWE'), ('gD','gWD')]:
            lhs = (residual[c_name].double() * direction.double()).sum()
            rhs = (direct[w_name].double() * (direction @ self.a).double()).sum()
            self.assertTrue(torch.allclose(lhs, rhs, rtol=3e-5, atol=2e-7))

    def test_direct_one_objective_at_a_time(self):
        before = self.adapter._sweeps
        current = self.adapter.direct_weight_gradients(self.records, objectives=('E',))
        generic = self.adapter.direct_weight_gradients(self.records, objectives=('D',))
        self.assertIn('gWE', current)
        self.assertNotIn('gWD', current)
        self.assertIn('gWD', generic)
        self.assertNotIn('gWE', generic)
        self.assertEqual(self.adapter._sweeps, before)
        self.assertEqual(current['current']['counts']['backwards'], 2)
        self.assertEqual(generic['generic']['counts']['backwards'], 3)
        with self.assertRaisesRegex(ModelBoundary, 'ALREADY_CONSUMED'):
            self.adapter.direct_weight_gradients(self.records, objectives=('E',))

    def test_direct_D_first_rejected(self):
        with self.assertRaisesRegex(ModelBoundary, 'E_MUST_PRECEDE'):
            self.adapter.direct_weight_gradients(self.records, objectives=('D',))

    def test_residual_recorder_order_and_payload(self):
        events = []
        original_document = self.teacher.document

        def document(*args):
            self.assertTrue(events, 'Current must be saved before generic starts')
            self.assertEqual(events[0][0], 'residual_current_gradient')
            return original_document(*args)

        def record(stage, payload):
            events.append((stage, payload))
            self.assertEqual(payload['gradient_sha256'], tensor_sha(payload['gradient']))
            self.assertIn('torch_cpu', payload['rng'])
            self.assertFalse(payload['technical_only'])

        with mock.patch.object(self.teacher, 'document', side_effect=document):
            result = self.adapter.gradient_sweeps(self.records, recorder=record)
        self.assertEqual([e[0] for e in events], ['residual_current_gradient','residual_generic_gradient'])
        self.assertTrue(torch.equal(events[0][1]['gradient'], result['gE']))
        self.assertTrue(torch.equal(events[1][1]['gradient'], result['gD']))

    def test_record_failure_stops_before_next_sweep(self):
        with mock.patch.object(self.teacher, 'document') as document:
            def failed_record(stage, payload):
                raise OSError('create-once recording failed')
            with self.assertRaisesRegex(OSError, 'recording failed'):
                self.adapter.gradient_sweeps(self.records, recorder=failed_record)
            document.assert_not_called()
        with self.assertRaisesRegex(ModelBoundary, 'ALREADY_CONSUMED'):
            self.adapter.gradient_sweeps(self.records)

    def test_direct_recorder_before_next_sweep_and_failure(self):
        def failed_record(stage, payload):
            self.assertEqual(stage, 'direct_current_weight_gradient')
            self.assertTrue(payload['technical_only'])
            self.assertEqual(payload['route'], 'DIRECT_SELECTED_WEIGHT_LEAF')
            self.assertEqual(payload['gradient_shape'], list(self.raw.shape))
            raise OSError('disk failure')
        with mock.patch.object(self.teacher, 'document') as document:
            with self.assertRaisesRegex(OSError, 'disk failure'):
                self.adapter.direct_weight_gradients(self.records, recorder=failed_record)
            document.assert_not_called()
        with self.assertRaisesRegex(ModelBoundary, 'E_MUST_PRECEDE'):
            self.adapter.direct_weight_gradients(self.records, objectives=('D',))

    def test_recorder_mutation_is_a_failure(self):
        def mutating(stage, payload):
            payload['gradient'].add_(1)
        with self.assertRaisesRegex(ModelBoundary, 'RECORDER_MUTATION'):
            self.adapter.gradient_sweeps(self.records, recorder=mutating)

    def test_direct_no_nonselected_parameter_or_rng_side_effect(self):
        versions = [(p,p.data_ptr(),p._version) for p in self.model.parameters()]
        rng = torch.random.get_rng_state().clone()
        self.adapter.direct_weight_gradients(self.records)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(p.data_ptr()==ptr and p._version==version and p.grad is None
                            for p,ptr,version in versions))
        self.assertTrue(torch.equal(self.model.down_proj.weight,self.raw))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
