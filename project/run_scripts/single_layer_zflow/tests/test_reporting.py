import json
from pathlib import Path
import tempfile
import unittest

from project.run_scripts.single_layer_zflow.reporting import publish, table


class ReportingTests(unittest.TestCase):
    def test_exact_numden_is_not_converted_to_percent(self):
        result = table([{'value': '998/1000', 'delta': -1.25, 'missing': None}],
                       [('value', 'n/d'), ('delta', 'pp'), ('missing', 'missing')])
        self.assertIn('998/1000', result)
        self.assertIn('-1.25', result)
        self.assertIn('NOT_RECORDED', result)

    def test_bar_character_is_escaped(self):
        self.assertIn('a\\|b', table([{'x':'a|b'}], [('x','value')]))

    def test_partial_and_fixture_results_cannot_be_published_as_full(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            for status, checkpoint in [('INCOMPLETE', 'FULL_FILE_TENSOR_RNG_SHA256'),
                                       ('CPU_VERIFIED_SEQ1000', 'INJECTED_FIXTURE_LOADER_NOT_A_PRODUCTION_RECEIPT')]:
                (root/'verification.json').write_text(json.dumps(dict(status=status, checkpoint_verification=checkpoint)))
                with self.assertRaisesRegex(ValueError, 'verified full ten'):
                    publish(root, root/'figures', root/'output', root/'technical', root/'reuse', root/'jobs')


if __name__ == '__main__': unittest.main()
