"""CPU/reference and ownership/source receipts before immutable Slurm source."""
import argparse,io,json,platform,unittest,subprocess,shutil
from .common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',default='cpu-tests-r1.json');args=p.parse_args()
    require(Path(args.receipt).name==args.receipt,'RECEIPT_BASENAME')
    reference=LOCAL/'reference-cpu-r1'
    if not reference.exists():
        reference.mkdir()
        for name in ('writer.py','test_ref.py','requirements.txt'):
            shutil.copyfile(ROOT/DESIGN/'reference'/name,reference/name)
        import sys
        proc=subprocess.run([sys.executable,'test_ref.py'],cwd=reference,capture_output=True,text=True)
        write(reference/'execution.json',dict(rc=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
        require(proc.returncode==0,'REFERENCE8')
    source=[member(p) for p in sorted((ROOT/'project/run_scripts/jlz_native_writer_aware').glob('*.py'))]
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('project.run_scripts.jlz_native_writer_aware.test_cpu')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    receipt=dict(scope='CPU tiny random Llama/FP64 reference only; NOT actual pretrained GPU qualification',
        tests_run=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
        source=source,reference8=member(reference/'audit.json'),original_reference_unchanged=True,
        python=platform.python_version(),host=platform.node(),independent_reviewer=False,owner_audit=True,
        GPU='NOT_RUN',submissions=0)
    write(LOCAL/'preparation-r1'/args.receipt,receipt)
    write(LOCAL/'preparation-r1'/('output-'+args.receipt),dict(output=stream.getvalue()))
    print(json.dumps({k:v for k,v in receipt.items() if k!='source'}));require(result.wasSuccessful(),'CPU_PREFLIGHT')

if __name__=='__main__':main()
