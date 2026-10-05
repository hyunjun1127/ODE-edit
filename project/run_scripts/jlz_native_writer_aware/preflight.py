"""Owner narrow CPU regressions; exact canonical CPU15 receipt reused."""
import argparse,io,json,platform,unittest
from .common import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',default='cpu-tests-r1.json');args=p.parse_args()
    require(Path(args.receipt).name==args.receipt,'RECEIPT_BASENAME')
    reference=ROOT/DESIGN/'reference/runtime-audit.json'
    audit=json.loads(reference.read_text());require(audit['passed']==15 and audit['failed']==0,'CPU15_EXACT_REUSE')
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('project.run_scripts.jlz_native_writer_aware.test_r2')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    receipt=dict(scope='narrow CPU new projection/VJP/complete-owner/state fixtures; not actual pretrained GPU',
        tests_run=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
        source=[member(p) for p in sorted((ROOT/'project/run_scripts/jlz_native_writer_aware').glob('*.py'))],
        canonical15_reused=member(reference),original_reference_unchanged=True,python=platform.python_version(),host=platform.node(),
        independent_reviewer=False,owner_audit=True,GPU='NOT_RUN',submissions=0)
    write(LOCAL/'preparation-r1'/args.receipt,receipt);write(LOCAL/'preparation-r1'/('output-'+args.receipt),dict(output=stream.getvalue()))
    print(stream.getvalue());print(json.dumps({k:v for k,v in receipt.items() if k!='source'}));require(result.wasSuccessful(),'CPU_PREFLIGHT')
if __name__=='__main__':main()
