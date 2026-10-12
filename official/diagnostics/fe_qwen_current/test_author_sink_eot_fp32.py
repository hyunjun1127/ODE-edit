import importlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from official.diagnostics.fe_qwen_current import author_basic_eot_fp32 as base
from official.diagnostics.fe_qwen_current import author_sink_eot_fp32 as v
from official.diagnostics.fe_qwen_current import failure_eot_probe as eot
from official.ours import pos0

SNAPSHOT = Path('/mnt/raid5/janghj/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/'
                'a09a35458c702b33eeacc393d103063234e8bc28')
AUTHOR = Path('/mnt/raid5/janghj/ODE-edit/local/fe-collapse-repair-20261011/native-r2/author')
CF = Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1/streams/cf-stream.json')
ZSRE = Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1/streams/zsre-stream.json')
CONTEXTS = [['{}'], ['The following is a single-choice question from a Chinese. {}',
                     'Therefore, the final answer is \\(\\boxed \\. {}', 'Because I am new to this, I would appreciate. {}',
                     'I am trying to create a virtual environment for my. {}', 'You are given an array of strings words and a. {}']]
W0_MISSED_SINKS = {303: 'The Experiment', 13152: 'The Possible', 20814: 'The Assault'}


class PolicyLabelsTest(unittest.TestCase):
    def test_policy_required(self):
        with self.assertRaises(Exception):
            v._patches(dict(eot_policy='always'))
        with self.assertRaises(Exception):
            v.prefix_rule('basic')

    def test_patch_scope_and_labels(self):
        b = dict(eot_policy='sink_region', dataset='cf')
        with patch.object(base, 'tracking', lambda bb, env=None: dict(task_id='x', arm='y')), \
             patch.object(base, 'identity', lambda bb, r: dict(stage='old', input_policy='OURS_BASIC_SUBJECT_LAST_ZERO_EOT', L2=0)):
            original = eot.repair_prompt
            with v._patches(b)[0], v._patches(b)[2], v._patches(b)[3]:
                self.assertIsNot(eot.repair_prompt, original)
                t = base.tracking(b)
                i = base.identity(b, {})
            self.assertIs(eot.repair_prompt, original)
        self.assertEqual(t['arm'], 'AUTHOR-cf-FP32-L2-0-clamp1-sink_regionEOT')
        self.assertEqual(i['stage'], v.PROFILE)
        self.assertEqual(i['input_policy'], v.POLICIES['sink_region'])
        self.assertEqual(i['L2'], 0)


@unittest.skipUnless(SNAPSHOT.is_dir() and AUTHOR.is_dir() and CF.is_file() and ZSRE.is_file(), 'local Qwen tokenizer/author/streams')
class QwenPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import transformers
        cls.tok = transformers.AutoTokenizer.from_pretrained(str(SNAPSHOT), local_files_only=True)
        sys.path.insert(0, str(AUTHOR))
        cls.z = importlib.import_module('algs.memit.compute_z')

    def rows(self, path):
        for r in json.loads(path.read_text()):
            rr = r['requested_rewrite']
            prompts = [c.format(rr['prompt']) for group in CONTEXTS for c in group] + ['{} is a']
            yield r['case_id'], rr['subject'], prompts

    def adapted(self, policy, prompt, subject):
        adapter = eot.InputAdapter(self.tok, self.z.find_fact_lookup_idx)
        if policy == 'basic':
            return adapter.prompt(prompt, subject), adapter.lookup(prompt, subject, self.tok, 'subject_last', verbose=False)
        with patch.object(eot, 'repair_prompt', v.prefix_rule(policy)):
            return adapter.prompt(prompt, subject), adapter.lookup(prompt, subject, self.tok, 'subject_last', verbose=False)

    def test_policies_contain_basic_shift_only_and_cover_w0_sinks(self):
        counts = {}
        for name, path in (('cf', CF), ('zsre', ZSRE)):
            changed = {p: 0 for p in ('basic', 'sink_region', 'subject_start')}
            for case, subject, prompts in self.rows(path):
                for prompt in prompts:
                    plain = self.z.find_fact_lookup_idx(prompt, subject, self.tok, 'subject_last', verbose=False)
                    results = {p: self.adapted(p, prompt, subject) for p in changed}
                    for p, (new, lookup) in results.items():
                        if new == prompt:
                            self.assertEqual(lookup, plain)
                            continue
                        changed[p] += 1
                        self.assertEqual(new, pos0.PREFIX + prompt)
                        self.assertEqual(lookup, plain + 1)
                        self.assertEqual(self.tok(new.format(subject))['input_ids'],
                                         [151643] + self.tok(prompt.format(subject))['input_ids'])
                        self.assertEqual(self.adapted(p, new, subject)[0], new)  # idempotent
                    basic, sink, start = (results[p][0] != prompt for p in ('basic', 'sink_region', 'subject_start'))
                    self.assertTrue(sink or not basic)
                    self.assertTrue(start or not sink)
                    self.assertEqual(sink, prompt.startswith('{}') and plain <= 1)
                    if case in W0_MISSED_SINKS and prompt.startswith('{}') and name == 'cf':
                        self.assertEqual(subject, W0_MISSED_SINKS[case])
                        self.assertFalse(basic)
                        self.assertTrue(sink)
            counts[name] = changed
        # Rows over 6 contexts + KL, 2000 requests each.
        self.assertEqual(counts['cf']['basic'], 21)
        self.assertEqual(counts['cf']['sink_region'], 501)
        self.assertEqual(counts['zsre']['basic'], 0)
        self.assertEqual(counts['zsre']['sink_region'], 158)


if __name__ == '__main__':
    unittest.main()
