import json
from pathlib import Path
import tempfile
import unittest

from project.run_scripts.single_layer_zflow.reporting import publish, table, technical_binding
from project.run_scripts.single_layer_zflow.analysis import member


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

    def test_actual_technical_binding_rejects_changed_threshold_or_input(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            tolerance = {'cost_relative': .001, 'max_logit_abs': .0001}
            (root/'phase1-complete.json').write_text('{}')
            (root/'parity-tolerance.json').write_text(json.dumps(tolerance))
            gate = dict(status='ACTUAL_LLAMA_TECHNICAL_VALID', source_input_lock_sha256='input',
                phase1_sha256=member(root/'phase1-complete.json')['sha256'],
                tolerance_sha256=member(root/'parity-tolerance.json')['sha256'])
            (root/'TECHNICAL_VALID.json').write_text(json.dumps(gate))
            verification = dict(execution_source={'input_lock_sha256':'input'},
                parity_tolerance=dict(tolerance), cost_relative_tolerance=.001)
            self.assertEqual(technical_binding(root, verification)[0], gate)
            verification['cost_relative_tolerance'] = .002
            with self.assertRaisesRegex(ValueError, 'tolerance differs'):
                technical_binding(root, verification)
            verification['cost_relative_tolerance'] = .001
            verification['execution_source']['input_lock_sha256'] = 'different'
            with self.assertRaisesRegex(ValueError, 'input binding mismatch'):
                technical_binding(root, verification)
            verification['execution_source']['input_lock_sha256'] = 'input'
            (root/'phase1-complete.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'evidence SHA mismatch'):
                technical_binding(root, verification)


if __name__ == '__main__': unittest.main()
