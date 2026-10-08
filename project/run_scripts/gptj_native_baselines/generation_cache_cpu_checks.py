"""Narrow owner API/control fixture checks, never actual GPU qualification."""
import argparse
import io
import resource
import time
import unittest
from pathlib import Path

from .generation_common import member, read, require, write
from .generation_cache_common import REPAIR_LOCAL, ROOT, ready, layout

MODULES = (
    'test_generation_cache_qualification',
    'test_generation_cache_reuse',
    'test_generation_cache_bridge',
    'test_generation_cache_tracking',
    'test_generation_tracking_reader',
    'test_generation_cache_profile',
    'test_generation_collect',
    'test_generation_run',
    'test_generation_tracking_shared',
)


def check(out, profile='r1'):
    import torch
    started = time.monotonic()
    require(not torch.cuda.is_initialized(), 'CPU_CHECK_MUST_NOT_INITIALIZE_CUDA')
    torch.set_num_threads(1)
    config_path = layout(profile)[0] / 'preparation-r1/config.json'
    config = read(config_path)
    ready(config)
    package = 'project.run_scripts.gptj_native_baselines.'
    suite = unittest.TestLoader().loadTestsFromNames([package + name for name in MODULES])
    capture = io.StringIO()
    result = unittest.TextTestRunner(stream=capture, verbosity=1).run(suite)
    source = [member(path) for path in sorted((ROOT / 'project/run_scripts/gptj_native_baselines').glob('generation_*.py'))]
    source += [member(path) for path in sorted((ROOT / 'project/run_scripts/gptj_native_baselines').glob('test_generation_cache_*.py'))]
    source += [member(ROOT / ('project/run_scripts/gptj_native_baselines/' + name + '.py'))
               for name in MODULES if not name.startswith('test_generation_cache_')]
    shared = config['generation']['shared_source_members']
    receipt = dict(status='PASS_CPU_INTEGRATION' if result.wasSuccessful() else 'FAILED_CPU_INTEGRATION',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        modules=list(MODULES), source=source, shared_source=shared,
        config=member(config_path), seconds=time.monotonic()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        CPU_threads=1, CUDA_initialized=torch.cuda.is_initialized(),
        actual_shared_API_binding='READ_BOUND_CPU_FIXTURE_NOT_PRETRAINED_PASS',
        production_config_bound=True, actual_GPU_qualification=False,
        pretrained_model_loads=0, actual_scientific_generation_forwards=0,
        synthetic_CPU_fixtures_only=True, new_Slurm_jobs=0,
        checkpoint_saved=False, scientific_complete=False,
        prior_CPU_components_preserved=True, overlapping_tests_not_summed=True,
        owner_review_only='separate bounded workers reviewed assigned seams; no actual GPU reviewer')
    write(out, receipt)
    print(capture.getvalue(), end='')
    print(dict(status=receipt['status'], tests=receipt['tests'], failures=receipt['failures'],
        errors=receipt['errors'], seconds=receipt['seconds'], CUDA_initialized=receipt['CUDA_initialized']))
    require(result.wasSuccessful() and not torch.cuda.is_initialized(), 'NARROW_CPU_API_CONTROL_FAILED')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', choices=('r1', 'r2'), default='r1')
    args = parser.parse_args()
    check(args.out, profile=args.profile)


if __name__ == '__main__':
    main()
