"""Current-batch target scope, author history semantics and rollback."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from official.baselines import fe_author_sequential as adapter
from official.baselines.fe_observation import observe_batch, summarize


class AuthorSequentialTests(unittest.TestCase):
    def fixture(self):
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Linear(3, 2, bias=False) for _ in range(2)])
        with torch.no_grad():
            for w in model.parameters():
                w.fill_(.1)
        hp = SimpleNamespace(layers=[0, 1], rewrite_module_tmp='layers.{}')
        history = {str(l): torch.zeros(3, 3) for l in hp.layers}
        requests = [dict(case_id=i, prompt='A subject lives', subject='subject', target_new='B') for i in (301, 302)]
        captured = []

        def target(model, tok, request, hp, layers, contexts):
            captured.append((request['case_id'], float(model.layers[0].weight[0, 0].detach())))
            self.assertEqual(request['prompt'], 'A {} lives')
            self.assertEqual(request['target_new'], ' B')
            return {l: model.layers[l].weight[:, 0].detach().clone()+1 for l in layers}

        author = SimpleNamespace(torch=torch)
        def batch_edit(cfg, model, tok, requests, device, H, zs):
            self.assertEqual([r['sample_idx'] for r in requests], [0, 1])
            self.assertTrue(all(z.shape == (2, 2) for z in zs.values()))
            for l in hp.layers:
                K = torch.tensor([[1., 2.], [.3, .4], [2., 1.]], dtype=torch.float64)
                R = zs[l].double()
                update = author.torch.linalg.solve(K@K.T + H[l].double()+torch.eye(3), K@R.T)
                H[l] += K@K.T
                with torch.no_grad():
                    model.layers[l].weight.copy_(model.layers[l].weight+update.T)
        author.batch_edit = batch_edit
        return model, hp, history, requests, captured, target, author

    def test_next_batch_targets_see_committed_model_and_history_accumulates(self):
        model, hp, H, requests, captured, target, author = self.fixture()
        original = deepcopy(requests)
        with patch.object(adapter.target_native, 'compute_z', target), patch.object(adapter.target_native, 'get_context_templates', return_value=[['{}']]):
            adapter.apply(model, None, requests, hp, history=H, author=author, config=None)
            first_H = {l: x.clone() for l, x in H.items()}
            entry = float(model.layers[0].weight[0, 0].detach())
            second = [dict(r, case_id=r['case_id']+100) for r in requests]
            adapter.apply(model, None, second, hp, history=H, author=author, config=None)
        self.assertEqual([r[0] for r in captured], [301, 302, 401, 402])
        self.assertEqual(captured[2][1], entry)
        self.assertNotEqual(captured[0][1], entry)
        self.assertTrue(all(torch.equal(H[l], 2*first_H[l]) for l in H))
        self.assertEqual(requests, original)

    def test_late_author_failure_rolls_back_weights_and_history(self):
        model, hp, H, requests, _, target, author = self.fixture()
        weights = {k: p.clone() for k, p in model.named_parameters()}
        old_H = dict(H)
        real = author.batch_edit
        def fail(*args):
            real(*args)
            raise RuntimeError('late failure')
        author.batch_edit = fail
        with patch.object(adapter.target_native, 'compute_z', target), patch.object(adapter.target_native, 'get_context_templates', return_value=[['{}']]):
            with self.assertRaisesRegex(RuntimeError, 'late failure'):
                adapter.apply(model, None, requests, hp, history=H, author=author, config=None)
        self.assertTrue(all(torch.equal(p, weights[k]) for k, p in model.named_parameters()))
        self.assertTrue(all(H[l] is old_H[l] and torch.count_nonzero(H[l]) == 0 for l in H))
        self.assertIs(author.torch, torch)

    def test_cross_module_observer_captures_author_tuple_output(self):
        target = SimpleNamespace(compute_z=lambda *a: {0: torch.ones(2)})
        writer = SimpleNamespace(compute_ks=lambda *a: torch.ones(1, 3),
                                 get_module_input_output_at_words=lambda *a, **kw: (torch.ones(1, 3), torch.zeros(1, 2)))
        originals = dict(vars(writer))
        rows = [dict(case_id=2001)]
        hp = SimpleNamespace(layers=[0])
        with observe_batch(target, rows, hp, {}, writer=writer) as observation:
            target.compute_z(None, None, rows[0])
            writer.compute_ks(None, None, rows, hp, 0)
            writer.get_module_input_output_at_words(None, None, 0)
        self.assertEqual(vars(writer), originals)
        self.assertEqual(summarize(observation)['layers']['0']['residual_norm'], [2**.5])


if __name__ == '__main__':
    unittest.main()
