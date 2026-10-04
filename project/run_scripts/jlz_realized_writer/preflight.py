"""Bounded CPU regression receipt; no GPU or scheduler access."""
import argparse,io,json,platform,unittest
from .common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',default='cpu-tests.json');args=p.parse_args()
    require(Path(args.receipt).name==args.receipt and args.receipt.endswith('.json'),'RECEIPT_BASENAME')
    stream=io.StringIO();suite=unittest.TestSuite()
    for module in ('project.run_scripts.jlz_realized_writer.test_cpu','project.run_scripts.jlz_shared_budget.test_cpu'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(module))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    receipt=dict(scope='CPU production regressions and tiny random Llama; not actual 8B GPU',
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
        python=platform.python_version(),host=platform.node(),source=[member(p) for p in sorted((ROOT/'project/run_scripts/jlz_realized_writer').glob('*.py'))],
        reference16=member(LOCAL/'reference-cpu-r1/audit.json'),original_reference_unchanged=True,
        earlier_invocation_error='unittest discover without package context produced relative-import errors; corrected module-qualified invocation; no scientific run',
        independent_reviewer=False,owner_audit=True,GPU='NOT_RUN',submissions=0)
    write(LOCAL/'preparation-r1'/args.receipt,receipt)
    write(LOCAL/'preparation-r1'/('output-'+args.receipt),dict(output=stream.getvalue()))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('source',)}));require(result.wasSuccessful(),'CPU_PREFLIGHT')

if __name__=='__main__':main()
