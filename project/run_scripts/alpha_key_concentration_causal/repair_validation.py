"""CPU-only regression receipt for the native-token witness and attempt repair."""
import argparse
import contextlib
import io
import os
from pathlib import Path
import time
import unittest
from .common import save,file_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent/'tests'))
    log=io.StringIO();started=time.monotonic()
    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'cpu-tests.log').open('x') as f:f.write(log.getvalue())
    import transformers,torch
    receipt=dict(status='PASS' if result.wasSuccessful() and not result.skipped else 'FAIL_OR_SKIPPED',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        seconds=time.monotonic()-started,transformers=transformers.__version__,torch=torch.__version__,
        model_loads=0,model_forwards=0,GPU_allocations=0,Slurm_calls=0,checkpoint_scans=0,
        native_tokenization_changed=False,scientific_thresholds_changed=False,
        actual_model_G1='NOT_RUN',log_sha256=file_sha(args.output/'cpu-tests.log'))
    save(args.output/'receipt.json',receipt)
    print(receipt)
    if not result.wasSuccessful() or result.skipped:
        print(log.getvalue());raise SystemExit(1)

if __name__=='__main__':main()
