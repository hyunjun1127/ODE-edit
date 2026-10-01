"""One bounded CPU suite receipt. No pretrained model, GPU or scheduler."""
import io
import json
import os
from pathlib import Path
import time
import unittest
from .common import ROOT,LOCAL,member,write

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU_ONLY_ENV_REQUIRED')
    start=time.monotonic();stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'project/run_scripts/jlz_two_arm'),pattern='test_*.py',top_level_dir=str(ROOT))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    out=LOCAL/'cpu-audit';out.mkdir(exist_ok=True)
    # Local raw unittest output only; tracked publication receives the summary.
    (out/'unittest.txt').write_text(stream.getvalue())
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
          failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),seconds=time.monotonic()-start,
          CPU_only=True,pretrained_model_loads=0,GPU_calls=0,
          source_members=[member(p) for p in sorted((ROOT/'project/run_scripts/jlz_two_arm').glob('*.py'))],
          log=member(out/'unittest.txt'))
    write(out/'receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_members','log')}))
    return 0 if result.wasSuccessful() else 2

if __name__=='__main__':raise SystemExit(main())
