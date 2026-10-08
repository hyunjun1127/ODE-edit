"""Prospective SH4 caller for SH1's reader; no reader fork or W0 fallback.

This adapter is CPU-prepared against the published SH1 branch. Production
adoption still requires reviewed main source and actual complete W0 inputs.
"""
import importlib
import inspect

from official.experiments.prepare import file_sha


SOURCE_COMMIT = '100f49d733649143ce0ec435cb89b3babcd219e5'
SOURCE_HASHES = {
    'official.evaluation.w0_reference': 'eeb098c827fda07eff20aaad7e18cd765c9dfc32a7168dd3c94442db2ca469c9',
    'official.evaluation.factual': '294480339e1742e1224d82f4faf1a23f3bbab4cda13dfbe21f1770299abfdaaa',
}


def verify_api():
    modules = {}
    for name, expected in SOURCE_HASHES.items():
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as error:
            raise ValueError('SOURCE_INPUT_PENDING_PORTABLE_W0_READER') from error
        if file_sha(module.__file__) != expected:
            raise ValueError('PORTABLE_W0_REVIEWED_SOURCE_REBIND_REQUIRED:' + name)
        modules[name] = module
    reader = modules['official.evaluation.w0_reference']
    signature = inspect.signature(reader.read_ready)
    if tuple(signature.parameters) != ('path', 'consumer_execution_identity',
                                      'consumer_fingerprint', 'member_paths'):
        raise ValueError('PORTABLE_W0_API_MISMATCH')
    for name in ('consumer_execution_identity', 'consumer_fingerprint', 'member_paths'):
        if signature.parameters[name].kind is not inspect.Parameter.KEYWORD_ONLY:
            raise ValueError('PORTABLE_W0_KEYWORD_API_MISMATCH')
    return reader


def borrow(path, *, consumer_execution_identity, consumed_content, member_paths=None):
    """Caller supplies its own measured locks, NEVER copies producer identity.

    Fingerprint derivation and complete input validation belong to the shared
    reader. Missing/incompatible inputs propagate without a forward or transfer.
    """
    reader = verify_api()
    fingerprint = reader.computational_fingerprint(consumed_content)
    return reader.read_ready(path, consumer_execution_identity=consumer_execution_identity,
                             consumer_fingerprint=fingerprint, member_paths=member_paths)


def observer_inputs(borrowed, *, dataset, consumer_external_identity):
    """Preserve original W0 observation and separate portable consumer binding."""
    reader = verify_api()
    if not isinstance(borrowed, reader.BorrowedW0):
        raise ValueError('PORTABLE_W0_VERIFIED_BORROWED_INPUT_REQUIRED')
    if dataset == 'zsre':
        reference = borrowed.zsre(consumer_external_identity=consumer_external_identity)
        return borrowed.values['zsre_reference']['evaluation'], reference
    if dataset == 'cf':
        return borrowed.values['cf_factual'], None
    raise ValueError('PORTABLE_W0_DATASET')
