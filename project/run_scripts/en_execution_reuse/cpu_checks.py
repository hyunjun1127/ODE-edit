"""Reproducible CPU-only component validation; never an actual-model PASS."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from .preparation import ROOT, create_json, member


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT.resolve()):raise ValueError('TASK_LOCAL_RECEIPT_ONLY')
    modules=['test_boundaries','test_endpoint_observation','test_current_observation',
             'test_generated_teacher','test_generated_reference']
    args=[sys.executable,'-B','-m','unittest',*['project.run_scripts.en_execution_reuse.'+m for m in modules],'-v']
    started=time.monotonic()
    result=subprocess.run(args,text=True,capture_output=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
                 OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8'))
    syntax=subprocess.run(['bash','-n',str(Path(__file__).with_name('run.sbatch'))],text=True,capture_output=True)
    receipt=dict(status='CPU_TESTS_PASS_ACTUAL_LLAMA_NOT_RUN' if result.returncode==syntax.returncode==0 else 'CPU_FAIL',
        args=args,exit=result.returncode,stdout=result.stdout,stderr=result.stderr,
        shell_syntax_exit=syntax.returncode,shell_stderr=syntax.stderr,seconds=time.monotonic()-started,
        model_loads=0,model_forwards=0,GPU_jobs=0,actual_numerical_validation='NOT_RUN',
        bounded_workers='endpoint/observation and generated teacher/reference implementation + own CPU fixtures; parent scope/parity tests',
        current_input_bound=dict(requests=100,sequence_rows=1400,distinct_inputs=624,total_input_positions=10416,
            prefix_unique_columns=4315,max_length=32,missing_old=0,
            source='CPU exact pinned-tokenizer protected_sequences before model load; actual FP32 key dedup not established'),
        sources=[member(p) for p in sorted(Path(__file__).parent.glob('*.py'))])
    create_json(a.output,receipt)
    print(result.stderr)
    if receipt['status']=='CPU_FAIL':raise SystemExit(1)


if __name__=='__main__':main()
