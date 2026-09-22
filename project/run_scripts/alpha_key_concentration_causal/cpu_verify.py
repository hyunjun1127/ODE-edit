"""Generate scoped CPU receipt without touching authoritative design scripts."""
import argparse
import io
import json
from pathlib import Path
import platform
import subprocess
import unittest
from .common import save,file_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    here=Path(__file__).parent
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(here/'tests'))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    source=[dict(path=str(f.relative_to(here)),sha256=file_sha(f),bytes=f.stat().st_size) for f in sorted(here.rglob('*.py'))]
    import torch,numpy,scipy,pyarrow
    save(a.output,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        failures=[dict(test=str(t),trace=e) for t,e in result.failures],errors=[dict(test=str(t),trace=e) for t,e in result.errors],
        skipped=[dict(test=str(t),reason=e) for t,e in result.skipped],
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,pyarrow=pyarrow.__version__),
        source=source,new_GPU_calls=0,actual_model_gate='NOT_RUN',separate_red='bounded integration red worker fixtures; not GPU audit'))
    print(stream.getvalue());raise SystemExit(0 if result.wasSuccessful() else 1)

if __name__=='__main__':main()
