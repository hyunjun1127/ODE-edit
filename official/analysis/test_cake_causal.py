"""CPU checks of the intervention, cohort and raw-probability definition."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from official.analysis.cake_causal import (
    aggregate, collect_knowns, intervention, known_record, load_knowns,
    parameter_signature, subject_embedding_std, trace_case, validate,
)
from official.tracking.schema import config as tracking_config, metrics

CONFIG = json.loads((Path(__file__).resolve().parents[1] /
                    'hparams/CAKE/causal-qwen25.json').read_text())


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.Tanh())

    def forward(self, value):
        # Causal cross-token dependence makes the subject-state restoration
        # affect the final prediction; all layers still have different values.
        return value + self.mlp(value.cumsum(dim=1) / 3)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(123)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(12, 4)
        self.model.layers = nn.ModuleList([Block(4) for _ in range(9)])
        self.lm_head = nn.Linear(4, 12)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return SimpleNamespace(logits=self.lm_head(x))


class SelectionTests(unittest.TestCase):
    class Tokenizer:
        def __call__(self, text, **kwargs):
            offsets = [m.span() for m in re.finditer(r'\S+', text)]
            return dict(input_ids=list(range(1, len(offsets) + 1)), offset_mapping=offsets)

        def decode(self, ids):
            return ' Paris'

    def test_author_exact_next_token_match(self):
        tok = self.Tokenizer()
        record = dict(known_id=7, subject='Alice', prompt='Alice lives in France',
                      attribute='Paris', template='DO NOT USE THIS TEMPLATE')
        row, status = known_record(tok, record, 3)
        self.assertEqual(status, 'ELIGIBLE')
        self.assertEqual(row['prompt'], record['prompt'])
        self.assertEqual(row['case_id'], record['known_id'])
        self.assertEqual(row['known_id'], record['known_id'])
        self.assertEqual(row['input_ids'], [1, 2, 3, 4])
        self.assertEqual(row['subject_range'], [0, 1])
        for attribute in ['London', 'paris', 'Paris France', 'Par']:
            record['attribute'] = attribute
            self.assertEqual(known_record(tok, record, 3), (None, 'INCORRECT_NEXT_TOKEN'))

    def test_all_eligible_facts_keep_public_order_without_sampling_or_generation(self):
        tok = self.Tokenizer()
        # More than 1000 correct predictions must all survive. Toy has no
        # generate method, so this also rejects accidentally reintroduced generation.
        records = [dict(known_id=i, subject='Alice', prompt='Alice lives in France',
                        attribute='London' if i == 10 else 'Paris') for i in range(1209)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            progress = []
            selected = collect_knowns(Toy().eval(), tok, records, CONFIG, output, progress.append)
            self.assertEqual([r['known_id'] for r in selected], [i for i in range(1209) if i != 10])
            report = json.loads((output / 'knowns.json').read_text())
            self.assertEqual(report['scanned'], 1209)
            self.assertEqual(report['eligible'], 1208)
            self.assertFalse(report['resampled'])
            journal = [json.loads(line) for line in (output / 'selection.jsonl').read_text().splitlines()]
            self.assertEqual(len(journal), 1209)
            self.assertEqual(journal[10]['status'], 'INCORRECT_NEXT_TOKEN')
            self.assertEqual(progress[-1]['causal/eligible_facts'], 1208)

    def test_empty_eligible_cohort_fails_with_exclusion_record(self):
        record = dict(known_id=7, subject='Alice', prompt='Alice lives in France', attribute='London')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'NO_CORRECT_PUBLIC_KNOWN_FACTS'):
                collect_knowns(Toy().eval(), self.Tokenizer(), [record], CONFIG, Path(directory), lambda _: None)
            report = json.loads((Path(directory) / 'knowns.json').read_text())
            self.assertEqual(report['counts'], {'INCORRECT_NEXT_TOKEN': 1})

    def test_input_checksum_and_public_ids_are_required(self):
        records = [dict(known_id=i, subject='Alice', prompt='Alice lives in France', attribute='Paris')
                   for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'known_1000.json'
            path.write_text(json.dumps(records))
            config = dict(input_facts=2, knowns_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(load_knowns(path, config), records)
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'SHA256_MISMATCH'):
                load_knowns(path, config)
            records[-1]['known_id'] = 0
            path.write_text(json.dumps(records))
            config['knowns_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'COHORT_MISMATCH'):
                load_knowns(path, config)

    def test_real_qwen_token_boundary(self):
        from transformers import AutoTokenizer
        path = Path('/mnt/raid5/janghj/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28')
        if not path.is_dir():
            self.skipTest('local tokenizer not available; no download')
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        prompt = 'The mother tongue of Danielle Darrieux is'
        base = tok.encode(prompt, add_special_tokens=False)
        answer = tok.encode(' French', add_special_tokens=False)[0]
        record = dict(known_id=1, subject='Danielle Darrieux', attribute='French', prompt=prompt)
        row, status = known_record(tok, record, answer)
        self.assertEqual(status, 'ELIGIBLE')
        self.assertEqual(row['input_ids'], base)
        self.assertEqual(row['object_token_id'], answer)
        self.assertEqual(row['prompt'], prompt)
        self.assertEqual(tok.decode(base[row['subject_range'][0]:row['subject_range'][1]]).strip(),
                         'Danielle Darrieux')


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.model = Toy().eval()

    def test_exact_single_module_and_subject_last_patch(self):
        seen, ids = {}, torch.tensor([[1, 2, 3, 4]] * 11)
        hooks = []
        # Register observation after the actual intervention hooks.
        with intervention(self.model, 'model.embed_tokens', 'model.layers.5.mlp',
                          (1, 3), torch.ones(10, 2, 4)):
            for name in ['model.embed_tokens', 'model.layers.4.mlp', 'model.layers.5.mlp', 'model.layers.6.mlp']:
                hooks.append(self.model.get_submodule(name).register_forward_hook(
                    lambda m, a, o, key=name: seen.update({key: o.detach().clone()})))
            self.model(ids)
        for handle in hooks:
            handle.remove()
        self.assertTrue(torch.equal(seen['model.embed_tokens'][0, 0], seen['model.embed_tokens'][1, 0]))
        self.assertTrue(torch.equal(seen['model.embed_tokens'][0, 3], seen['model.embed_tokens'][1, 3]))
        self.assertTrue(torch.allclose(seen['model.embed_tokens'][1, 1:3] -
                                       seen['model.embed_tokens'][0, 1:3], torch.ones(2, 4)))
        self.assertTrue(torch.equal(seen['model.layers.5.mlp'][0, 2], seen['model.layers.5.mlp'][1, 2]))
        for layer, token in [(4, 2), (6, 2), (5, 1), (5, 3)]:
            self.assertFalse(torch.equal(seen[f'model.layers.{layer}.mlp'][0, token],
                                         seen[f'model.layers.{layer}.mlp'][1, token]))

    def test_cleanup_after_exception(self):
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with intervention(self.model, 'model.embed_tokens', 'model.layers.5.mlp',
                              (0, 1), torch.zeros(10, 1, 4)):
                raise RuntimeError('injected')
        self.assertFalse(any(m._forward_hooks for m in self.model.modules()))

    def test_zero_noise_zero_effect_and_weights_unchanged(self):
        ids = [1, 2, 3, 4]
        target = self.model(torch.tensor([ids])).logits[0, -1].argmax().item()
        row = dict(case_id=1, input_ids=ids, subject_range=[0, 2], object_token_id=target)
        before = {k: p.clone() for k, p in self.model.named_parameters()}
        signature = parameter_signature(self.model)
        result = trace_case(self.model, row, CONFIG, 0)
        self.assertEqual(result['indirect_effect'], {str(l): 0 for l in CONFIG['layers']})
        self.assertEqual(signature, parameter_signature(self.model))
        self.assertTrue(all(torch.equal(before[k], p) for k, p in self.model.named_parameters()))

    def test_paired_noise_repeatability_and_mean_subtraction(self):
        ids = [1, 2, 3, 4]
        target = self.model(torch.tensor([ids])).logits[0, -1].argmax().item()
        row = dict(case_id=77, input_ids=ids, subject_range=[1, 3], object_token_id=target)
        a, b = [trace_case(self.model, row, CONFIG, .2) for _ in range(2)]
        self.assertEqual(a, b)
        self.assertEqual(len(a['corrupted']), 10)
        self.assertGreater(len(set(a['corrupted'])), 1)
        for layer in CONFIG['layers']:
            expected = sum(a['restored'][str(layer)]) / 10 - sum(a['corrupted']) / 10
            self.assertAlmostEqual(a['indirect_effect'][str(layer)], expected, places=14)

    def test_std_matches_hooked_embedding_definition(self):
        tok = SimpleNamespace(encode=lambda s, **kw: {'a': [1, 2], 'b': [3]}[s])
        measured, size = subject_embedding_std(self.model, tok, [dict(subject='a'), dict(subject='b')])
        outputs = []
        hook = self.model.model.embed_tokens.register_forward_hook(lambda m, a, o: outputs.append(o[0].detach()))
        for ids in [[1, 2], [3]]:
            self.model(torch.tensor([ids]))
        hook.remove()
        self.assertEqual(measured, torch.cat(outputs).std().item())
        self.assertEqual(size, 12)


class AggregateTests(unittest.TestCase):
    def rows(self, count=37):
        return [dict(case_id=i, corrupted=[.4] * 10,
                     restored={str(l): [.3 + .01 * l] * 10 for l in CONFIG['layers']})
                for i in range(count)]

    def test_aie_is_difference_and_preserves_negative_values(self):
        result = aggregate(self.rows(), CONFIG, list(range(37)))
        self.assertEqual(result['requests'], 37)
        self.assertAlmostEqual(result['causal_scores']['0'], -.06)
        self.assertAlmostEqual(result['physical_layer_scores']['8'], -.02)
        self.assertAlmostEqual(sum(result['layer_weights'].values()), 1)
        self.assertLess(result['layer_weights']['4'], result['layer_weights']['8'])

    def test_reject_missing_or_duplicate_facts(self):
        with self.assertRaises(ValueError):
            aggregate(self.rows()[:-1], CONFIG, list(range(37)))
        rows = self.rows()
        rows[-1]['case_id'] = rows[0]['case_id']
        with self.assertRaises(ValueError):
            aggregate(rows, CONFIG, list(range(37)))
        with self.assertRaises(ValueError):
            aggregate([], CONFIG, [])

    def test_actual_denominator_includes_every_fact_above_1000(self):
        rows = self.rows(1209)
        rows[-1]['restored']['4'] = [.9] * 10
        result = aggregate(rows, CONFIG, list(range(1209)))
        self.assertEqual(result['requests'], 1209)
        self.assertAlmostEqual(result['physical_layer_scores']['4'], (1208 * -.06 + .5) / 1209)

    def test_reject_missing_noise_or_nan(self):
        rows = self.rows()
        rows[0]['corrupted'] = [0] * 9
        with self.assertRaises(ValueError):
            aggregate(rows, CONFIG, list(range(37)))
        rows[0]['corrupted'] = [float('nan')] * 10
        with self.assertRaises(ValueError):
            aggregate(rows, CONFIG, list(range(37)))

    def test_reject_window_or_no_noise_repeats_change(self):
        self.assertEqual(validate(CONFIG), CONFIG)
        for key, value in [('restore_window', 10), ('noise_samples', 1), ('input_facts', 1000),
                           ('knowns_sha256', '0' * 64), ('cohort_rule', 'sample_1000'),
                           ('noise_subjects', 'only_eligible')]:
            wrong = deepcopy(CONFIG)
            wrong[key] = value
            with self.assertRaises(ValueError):
                validate(wrong)

    def test_causal_metrics_cannot_be_editing_scores(self):
        cfg = tracking_config(dict(server='server3', task_id='cake-qwen25-causal-score-20261011',
            arm='qwen25-cake-causal-score', attempt='run-r1', source_sha='a' * 40))
        metrics({'causal/AIE': -.2, 'causal/layer': 4}, config_values=cfg)
        with self.assertRaises(ValueError):
            metrics({'causal/AIE': .2}, scientific=True, config_values=cfg)
        with self.assertRaises(ValueError):
            metrics({'causal/AIE': .2, 'eval/RS': 90}, config_values=cfg)
        with self.assertRaises(ValueError):
            metrics({'causal/AIE': .2}, config_values={'arm': 'other'})


if __name__ == '__main__':
    unittest.main()
