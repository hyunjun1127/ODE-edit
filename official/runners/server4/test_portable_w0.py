"""CPU caller wiring only; mocks are not complete W0/GPU READY evidence."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import portable_w0 as caller


class CallerTests(unittest.TestCase):
    def test_explicit_consumer_locks_and_member_map_passthrough(self):
        reader = Mock()
        fingerprint = {'consumer': 'CPU fixture, not producer locks'}
        reader.computational_fingerprint.return_value = fingerprint
        execution = {'actual_consumer': 'fixture'}
        content = {'explicit_consumed_content': 'fixture'}
        paths = {'source.factual.py': '/existing/fixture'}
        with patch.object(caller, 'verify_api', return_value=reader):
            result = caller.borrow('/missing/READY.json', consumer_execution_identity=execution,
                                   consumed_content=content, member_paths=paths)
        reader.computational_fingerprint.assert_called_once_with(content)
        reader.read_ready.assert_called_once_with('/missing/READY.json',
            consumer_execution_identity=execution, consumer_fingerprint=fingerprint, member_paths=paths)
        self.assertIs(result, reader.read_ready.return_value)

    def test_shared_input_error_propagates_no_fallback(self):
        reader = Mock()
        failure = ValueError('REFERENCE_INPUT_MISSING:fixture')
        reader.read_ready.side_effect = failure
        with patch.object(caller, 'verify_api', return_value=reader):
            with self.assertRaises(ValueError) as caught:
                caller.borrow('/missing', consumer_execution_identity={}, consumed_content={})
        self.assertIs(caught.exception, failure)
        reader.read_ready.assert_called_once()

    def test_source_absent_blocks(self):
        with patch.object(caller.importlib, 'import_module', side_effect=ModuleNotFoundError):
            with self.assertRaisesRegex(ValueError, 'SOURCE_INPUT_PENDING'):
                caller.verify_api()

    def test_source_change_requires_explicit_rebinding(self):
        with patch.object(caller.importlib, 'import_module', return_value=SimpleNamespace(__file__='/fixture')), \
                patch.object(caller, 'file_sha', return_value='wrong'):
            with self.assertRaisesRegex(ValueError, 'REVIEWED_SOURCE_REBIND_REQUIRED'):
                caller.verify_api()

    def test_original_observations_and_portable_view_not_relabelled(self):
        class BorrowedFixture:
            pass
        borrowed = BorrowedFixture()
        borrowed.values = {'cf_factual': {'producer': 'cf'},
                           'zsre_reference': {'evaluation': {'producer': 'zsre'}}}
        borrowed.zsre = Mock(return_value=object())
        reader = SimpleNamespace(BorrowedW0=BorrowedFixture)
        external = {'actual_consumer': 'different execution'}
        with patch.object(caller, 'verify_api', return_value=reader):
            cf, reference = caller.observer_inputs(borrowed, dataset='cf', consumer_external_identity=external)
            self.assertIs(cf, borrowed.values['cf_factual'])
            self.assertIsNone(reference)
            zsre, reference = caller.observer_inputs(borrowed, dataset='zsre', consumer_external_identity=external)
            self.assertIs(zsre, borrowed.values['zsre_reference']['evaluation'])
            self.assertIs(reference, borrowed.zsre.return_value)
            borrowed.zsre.assert_called_once_with(consumer_external_identity=external)
            with self.assertRaisesRegex(ValueError, 'VERIFIED_BORROWED_INPUT_REQUIRED'):
                caller.observer_inputs({}, dataset='cf', consumer_external_identity=external)
            with self.assertRaisesRegex(ValueError, 'DATASET'):
                caller.observer_inputs(borrowed, dataset='other', consumer_external_identity=external)


if __name__ == '__main__':
    unittest.main()
