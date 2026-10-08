"""Narrow final schedule/control regressions, not model/GPU qualification."""
import argparse
import io
import resource
import time
import unittest
from pathlib import Path
from .generation_common import ROOT, member, read, require, write
from .generation_final_common import PREPARATION, ready

MODULES = ('test_generation_final_run', 'test_generation_final_bridge',
    'test_generation_final_collect', 'test_generation_final_tracking',
    'test_generation_final_profile', 'test_generation_final_submit',
    'test_generation_tracking_reader', 'test_generation_cache_tracking',
    'test_generation_collect', 'test_generation_run', 'test_generation_tracking_shared')


def check(out):
    import torch
    started = time.monotonic()
    require(not torch.cuda.is_initialized(), 'FINAL_CPU_NO_CUDA')
    torch.set_num_threads(1)
    path = PREPARATION / 'config.json'
    config = read(path)
    ready(config)
    package = 'project.run_scripts.gptj_native_baselines.'
    suite = unittest.TestLoader().loadTestsFromNames([package + name for name in MODULES])
    capture = io.StringIO()
    result = unittest.TextTestRunner(stream=capture, verbosity=1).run(suite)
    own = ROOT / 'project/run_scripts/gptj_native_baselines'
    source = [member(p) for p in sorted(own.glob('generation_*.py'))]
    source += [member(own / (name + '.py')) for name in MODULES]
    receipt = dict(status='PASS_CPU_INTEGRATION' if result.wasSuccessful() else 'FAILED_CPU_INTEGRATION',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        modules=list(MODULES), source=source, shared_source=config['generation']['shared_source_members'],
        config=member(path), seconds=time.monotonic() - started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        CPU_threads=1, CUDA_initialized=torch.cuda.is_initialized(),
        actual_GPU_qualification=False, pretrained_model_loads=0,
        actual_scientific_generation_forwards=0, synthetic_CPU_fixtures_only=True,
        new_Slurm_jobs=0, checkpoint_saved=False, scientific_complete=False,
        previous_CPU99_evidence_reused=True, historical_components_preserved=True,
        test_counts_overlap_not_additive=True,
        owner_review='bounded independent source seam review; no GPU reviewer')
    write(out, receipt)
    print(capture.getvalue(), end='')
    print({key: receipt[key] for key in ('status', 'tests', 'failures', 'errors', 'seconds', 'CUDA_initialized')})
    require(result.wasSuccessful() and not torch.cuda.is_initialized(), 'FINAL_CPU_INTEGRATION_FAILED')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    check(parser.parse_args().out)
