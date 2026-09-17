"""Bounded CPU-only preflight, with source-bound raw-free evidence."""
import ast
import importlib
import io
import os
from pathlib import Path
import subprocess
import unittest
from .common import ROOT,save,identity

def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU_CHECK_HIDE_GPU'
    here=Path(__file__).resolve().parent
    worktree=here.parents[2]
    files=sorted(here.glob('*.py'))
    for p in files:ast.parse(p.read_text())
    for name in ('runtime','engine','runner','technical','operations','control'):
        importlib.import_module(__package__+'.'+name)
    subprocess.run(['bash','-n',str(here/'run.sbatch')],check=True)
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(__package__+'.'+name)
        for name in ('test_qp','test_response','test_engine'))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
    record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        model_GPU_tests=0,actual_model_validation='NOT_RUN',syntax_files=len(files),shell_syntax='PASS',
        source_members=[dict(identity(p),relative=str(p.relative_to(worktree)))
            for p in sorted(here.iterdir()) if p.is_file()],
        independent_scope='QP/geometry and response CPU fixtures by two bounded workers; parent integration run',
        red='read-only actual source review, 3 findings repaired; no model-level PASS',
        checkpoint_override='NO_W_M_RNG_CHECKPOINT; RAM continuation only',output=stream.getvalue())
    save(ROOT/'cpu-checks-v1.json',record)
    print(record['status'],record['tests'],'CPU tests; actual Llama NOT_RUN')
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':run()
