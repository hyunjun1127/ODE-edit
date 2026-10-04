"""Controller+observer+raw reducer fixtures; random tiny CPU, not LM validation."""
import copy, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from project.run_scripts.jlz_realized_writer.capture import Adapter
from .common import *
from .run import drive
from .collect import reduce_arm

class TinyBench:
    contexts = [['{}'], ['p{}']]
    tokenizer = SimpleNamespace(pad_token_id=0)

    def prepare(self, records):
        B = len(records); ids = [r['case_id'] for r in records]
        pack = dict(identity=digest(ids), n_requests=B, n_rw=2, record_ids=ids, canonical_rows=list(range(0, 3*B, 3)),
            context_group_slices=[(0, 1), (1, 2)], lookup=[1]*(3*B), row_kind=['rewrite', 'rewrite', 'kl']*B,
            row_request=[i//3 for i in range(3*B)], tokens=dict(input_ids=(torch.arange(15*B).reshape(3*B, 5)+ids[0])%29+1,
            attention_mask=torch.ones(3*B, 5, dtype=torch.long)), targets=torch.full((3*B, 5), -100))
        for i in range(3*B):
            if pack['row_kind'][i] == 'rewrite': pack['targets'][i, 3:] = torch.tensor([2, 3])
        return pack

    def panels(self, record):
        return {kind: [f'{record["case_id"]}:{kind}'] for kind in ('R', 'P', 'N')}

    def evaluation_ids(self, prompt, target):
        return [1, 4, int(prompt.split(':')[0]) + 5], [2 if target == 'new' else 3]

class ControllerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27); torch.set_num_threads(2)

    def fixture(self, root):
        model = LlamaForCausalLM(LlamaConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=3,
            num_attention_heads=2, num_key_value_heads=2, vocab_size=31, max_position_embeddings=64, _attn_implementation='eager')).eval()
        a = Adapter(model, dict(eligible_layers=[0, 1], anchor_layer=1, nll_layer=2, kl_factor=.0625))
        H = {l: torch.zeros(16, 16) for l in a.sites}; bench = TinyBench()
        records = [dict(case_id=i, requested_rewrite=dict(subject='x' + str(i), relation_id='r',
            target_new=dict(str='new'), target_true=dict(str='true'))) for i in range(1, 5)]
        expected = []
        for record in records:
            for kind, prompts in bench.panels(record).items():
                prompt = prompts[0]
                expected.append(dict(identity=digest([record['case_id'], kind, 0, prompt, 'new', 'true']),
                    case_id=record['case_id'], kind=kind, prompt_index=0,
                    new_token_identity=digest(bench.evaluation_ids(prompt, 'new')),
                    true_token_identity=digest(bench.evaluation_ids(prompt, 'true'))))
        identity = root / 'identity.json'; write(identity, dict(rows=expected))
        stats = {}
        for l in a.sites:
            path = root / f'C{l}.npz'; np.savez(path, **{'mom2.mom2': np.eye(16, dtype=np.float32), 'mom2.count': np.array(1)})
            stats[str(l)] = str(path)
        c = dict(settings=dict(B=2, batches=2, requests=4, observer_microbatch=2, fit_requests_per_group=1),
            packs=[dict(identity=bench.prepare(batch)['identity'], ids=[r['case_id'] for r in batch]) for _, batch, _ in batches(records, 2)],
            observer_identity=member(identity), stats=stats, profile=dict(eligible_layers=[0, 1]),
            qualification_reuse=dict(cold_W0_H0=state(a, H)))
        return a, bench, records, H, c, expected

    def test_end_to_end_two_commit_controller_and_partial_reducer(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp); root = attempt / 'main-MD'; root.mkdir()
            a, bench, records, H, c, identities = self.fixture(attempt)
            commits = drive(a, bench, records, H, c, 'MD', root, 'CPU_SOURCE')
            self.assertEqual(len(commits), 2); self.assertEqual(commits[1]['before'], commits[0]['after'])
            self.assertEqual(sum(r['history_appends'] for r in commits), 4)
            self.assertTrue((root / 'initial.json').exists()); self.assertFalse((root / 'batch-03').exists())
            out = attempt / 'cpu-report'; out.mkdir()
            result, raw = reduce_arm(attempt, 'MD', c, dict(source_commit='CPU_SOURCE'), identities, records, out)
            self.assertEqual(result['commits'], 2); self.assertEqual(result['next_entry_links'], 1)
            self.assertEqual(result['history_appends'], 4); self.assertEqual(result['warnings'], [])
            self.assertEqual(result['status'], 'PARTIAL_OR_TECHNICAL_BLOCKED')  # not a production W20 claim
            self.assertEqual(a.last_virtual, {})

    def test_controller_second_write_failure_rolls_back_to_first_commit(self):
        from project.run_scripts.jlz_realized_writer.writer import apply_sequential
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp); root = attempt / 'main-CD'; root.mkdir()
            a, bench, records, H, c, _ = self.fixture(attempt)
            calls = 0
            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    args[0].weights[args[0].first].add_(1)
                    args[5][args[0].first].add_(1)
                    raise RuntimeError('INJECTED_SECOND_BATCH_WRITER_IO_ERROR')
                return apply_sequential(*args, **kwargs)
            with patch('project.run_scripts.jlz_realized_writer_sequential.run.apply_sequential', fail_second), torch.no_grad():
                # The fit needs autograd even though only the deliberate writer
                # fault is an in-place no-grad mutation.
                with torch.enable_grad():
                    with self.assertRaisesRegex(RuntimeError, 'INJECTED_SECOND_BATCH_WRITER_IO_ERROR'):
                        drive(a, bench, records, H, c, 'CD', root, 'CPU_SOURCE')
            first = json.loads((root / 'batch-01/commit.json').read_text())
            self.assertEqual(state(a, H), first['after'])
            rollback = json.loads((root / 'batch-02/rollback.json').read_text())
            self.assertTrue(rollback['verified']); self.assertEqual(rollback['earlier_committed_prefix'], 1)
            self.assertFalse((root / 'batch-02/commit.json').exists()); self.assertFalse((root / 'batch-03').exists())

if __name__ == '__main__':
    unittest.main()
