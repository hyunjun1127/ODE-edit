"""Narrow CPU control/path checks, no scientific or scheduler execution."""
import copy
import unittest
from pathlib import Path

from .common import LOCAL, NONCE, TASK
from .register_resume import assert_rebinding, rebound_config, SCIENTIFIC_SOURCE
from .submit import launcher


class RegistrationPathTests(unittest.TestCase):
    def setUp(self):
        self.parent = dict(instruction_id=NONCE, task_id=TASK,
            attempt=str(LOCAL / 'attempt-r1'),
            generation=dict(W0_ready_path=str(LOCAL / 'attempt-r1/shared-generation-W0-ready.json'),
                            source_sha='unchanged', assets_sha256='unchanged'),
            run_instance=dict(attempt='20261007-r1'),
            native=dict(MEMIT={'v_lr': .1}), science=dict(batch=100, dtype='float32'))
        self.target = LOCAL / 'attempt-r2'

    def test_exact_three_changes_and_parent_immutable(self):
        old = copy.deepcopy(self.parent)
        child = rebound_config(self.parent, self.target)
        self.assertEqual(self.parent, old)
        self.assertEqual(child['attempt'], str(self.target))
        self.assertEqual(child['generation']['W0_ready_path'],
                         str(self.target / 'shared-generation-W0-ready.json'))
        self.assertEqual(child['run_instance']['attempt'], '20261007-r2')
        self.assertEqual(child['instruction_id'], NONCE)
        assert_rebinding(self.parent, child, self.target)

    def test_science_or_reference_change_rejected(self):
        child = rebound_config(self.parent, self.target)
        for field in ('science', 'native'):
            changed = copy.deepcopy(child); changed[field] = {'unapproved': True}
            with self.assertRaises(RuntimeError): assert_rebinding(self.parent, changed, self.target)
        changed = copy.deepcopy(child); changed['generation']['source_sha'] = 'different'
        with self.assertRaises(RuntimeError): assert_rebinding(self.parent, changed, self.target)

    def test_parent_nonce_and_path_guard(self):
        for bad in (LOCAL / 'attempt-r1', Path('/tmp/attempt-r2')):
            with self.assertRaises(RuntimeError): rebound_config(self.parent, bad)
        wrong = dict(self.parent, instruction_id='resume-not-runtime-parent')
        with self.assertRaises(RuntimeError): rebound_config(wrong, self.target)

    def test_launcher_new_path_original_scientific_source(self):
        r = dict(cpu=8, collector_cpu=8)
        value = launcher(self.target / 'source', SCIENTIFIC_SOURCE, 'MEMIT', self.target,
                         r, {'nltk_data': '/private-public-nltk'})
        self.assertIn('LLAMA3_NATIVE_BASELINES_SOURCE_COMMIT=' + SCIENTIFIC_SOURCE, value)
        self.assertIn('--attempt ' + str(self.target), value)
        self.assertNotIn('attempt-r1', value)
        self.assertIn('HF_HUB_OFFLINE=1', value)
        self.assertNotIn('API_KEY', value)


if __name__ == '__main__':
    unittest.main()
