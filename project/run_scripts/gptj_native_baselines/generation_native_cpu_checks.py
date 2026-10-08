"""Narrow native-generation caller/control regression receipt, CPU fixtures only.

This script neither certifies GPT-J GPU generation nor submits anything. The
registered native runner performs its already-authorized first actual work.
Source, config and this create-once receipt are bound before submission.
"""
import argparse
import io
import platform
import resource
import sys
import time
import unittest
from pathlib import Path

from .generation_native_common import (
    NATIVE_LOCAL, PREPARATION, ROOT, TASK, NONCE, PROFILE, ROUTE,
    member, read, require, ready, write,
)

MODULES = ('test_generation_native_run', 'test_generation_native_bridge',
           'test_generation_native_collect', 'test_generation_native_control')
PRODUCTION = ('generation_native_common', 'generation_native_bind',
              'generation_native_bridge', 'generation_native_run',
              'generation_native_tracking', 'generation_native_collect',
              'generation_native_submit')
OUTPUT = NATIVE_LOCAL / 'cpu-integration-r1.json'


def check(out=OUTPUT):
    """Only four changed API/control modules; preserve older CPU evidence."""
    import torch
    started = time.monotonic()
    out = Path(out).absolute()
    require(out == OUTPUT and not out.exists() and not out.is_symlink(),
            'NATIVE_CPU_RECEIPT_EXACT_CREATE_ONCE')
    require(not torch.cuda.is_initialized(), 'NATIVE_CPU_NO_CUDA_BEFORE')
    torch.set_num_threads(1)
    path = PREPARATION / 'config.json'
    config = read(path)
    ready(config)
    config_member = member(path)
    own = ROOT / 'project/run_scripts/gptj_native_baselines'
    source_paths = [own / (name + '.py') for name in PRODUCTION + MODULES]
    source_paths.append(Path(__file__).resolve())
    require(len(source_paths) == len(set(source_paths)) == 12,
            'NATIVE_CPU_EXACT_SOURCE_SCOPE')
    source_members = [member(p) for p in source_paths]
    package = 'project.run_scripts.gptj_native_baselines.'
    suite = unittest.TestLoader().loadTestsFromNames([package + name for name in MODULES])
    capture = io.StringIO()
    result = unittest.TextTestRunner(stream=capture, verbosity=1).run(suite)
    require(member(path) == config_member and [member(p) for p in source_paths] == source_members,
            'NATIVE_CPU_SOURCE_CONFIG_NONMUTATION')
    cuda_initialized = torch.cuda.is_initialized()
    passed = result.wasSuccessful() and not cuda_initialized and torch.get_num_threads() == 1
    receipt = dict(
        task=TASK, instruction_id=NONCE,
        status='PASS_CPU_INTEGRATION' if passed else 'FAILED_CPU_INTEGRATION',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), modules=list(MODULES),
        source_members=source_members,
        production_source_count=len(PRODUCTION), test_source_count=len(MODULES),
        CPU_receipt_script_count=1,
        shared_source=config['generation']['shared_source_members'],
        shared_source_commit=config['generation']['source_sha'],
        shared_package_tree=config['generation']['package_tree'],
        config=config_member, config_sha256=config_member['sha256'],
        generation_profile=PROFILE, generation_route=ROUTE,
        generation_schedule='W20_ONLY_FIRST2000',
        runtime=dict(python=sys.executable, python_version=platform.python_version(),
                     torch_version=str(torch.__version__), CUDA_initialized=cuda_initialized),
        seconds=time.monotonic() - started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        CPU_threads=torch.get_num_threads(), CUDA_initialized=cuda_initialized,
        actual_GPU_qualification=False, pretrained_model_loads=0,
        actual_scientific_generation_forwards=0, actual_edit_commits=0,
        synthetic_CPU_fixtures_only=True, new_Slurm_jobs=0,
        checkpoint_saved=False, scientific_complete=False,
        source_config_nonmutation=True, shared_namespace_modified=False,
        historical_CPU_evidence_reused=True, historical_components_preserved=True,
        test_counts_overlap_not_additive=True,
        owner_review='좁은 변경 API/control CPU fixture 검산; actual GPU 인증 아님')
    write(out, receipt)
    print(capture.getvalue(), end='')
    print({key: receipt[key] for key in
           ('status', 'tests', 'failures', 'errors', 'seconds', 'CUDA_initialized')})
    require(passed, 'NATIVE_CPU_INTEGRATION_FAILED')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUTPUT)
    check(parser.parse_args().out)
