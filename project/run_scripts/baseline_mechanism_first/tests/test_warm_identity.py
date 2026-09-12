"""Historical request identities retain their original JSON byte convention."""
import hashlib
import json
import unittest

from project.run_scripts.baseline_mechanism_first.case_population import source_digest
from project.run_scripts.baseline_mechanism_first.contracts import digest


class WarmIdentityTests(unittest.TestCase):
    def test_unicode_original_blue_not_local_manifest(self):
        row = {'subject': 'Montréal', 'target_new': {'str': 'Québec'}}
        original = json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
        self.assertEqual(source_digest(row), hashlib.sha256(original).hexdigest())
        self.assertNotEqual(source_digest(row), digest(row))

    def test_ascii_unchanged(self):
        row = [['{}'], ['The {}']]
        self.assertEqual(source_digest(row), digest(row))


if __name__ == '__main__':
    unittest.main()
