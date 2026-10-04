"""Bounded owner source/CPU sequential regression. No GPU or scheduler calls."""
import argparse, io, json, platform, unittest
from .common import *

def main():
    p = argparse.ArgumentParser(); p.add_argument('--receipt', default='cpu-tests.json')
    args = p.parse_args(); require(Path(args.receipt).name == args.receipt and args.receipt.endswith('.json'), 'CPU_RECEIPT_BASENAME')
    stream = io.StringIO(); suite = unittest.TestSuite()
    for name in ('test_cpu', 'test_controller'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('project.run_scripts.jlz_realized_writer_sequential.' + name))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    receipt = dict(scope='new CPU tiny two-batch sequential production/state/reducer fixtures; NOT actual8B GPU',
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), passed=result.wasSuccessful(),
        python=platform.python_version(), source=[member(p) for p in sorted((ROOT / 'project/run_scripts/jlz_realized_writer_sequential').glob('*.py'))],
        owner_audit=True, independent_reviewer=False, actual_new_GPU='NOT_RUN', submissions=0,
        reused_frozen_B1=FROZEN, unchanged_reference_not_reexecuted=True)
    write(LOCAL / 'preparation-r1' / args.receipt, receipt)
    write(LOCAL / 'preparation-r1' / ('output-' + args.receipt), dict(output=stream.getvalue()))
    print(stream.getvalue()); print(json.dumps({k: v for k, v in receipt.items() if k != 'source'}))
    require(result.wasSuccessful(), 'CPU_PREFLIGHT')

if __name__ == '__main__':
    main()
