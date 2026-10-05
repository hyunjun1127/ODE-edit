"""Narrow production regressions and exact canonical receiver binding, CPU only."""
import json, os, platform, subprocess, sys, time
from pathlib import Path
from .common import *

def main():
    out = LOCAL / 'preparation-r1'; out.mkdir(parents=True, exist_ok=True)
    require(sha(ROOT / ENVELOPE) == '941f16ca95425086435e290288264a87666f8549eed560eb137949f310e64d40', 'ENVELOPE_SHA')
    manifest = ROOT / DESIGN / 'artifact-manifest.json'
    require(sha(manifest) == 'a67ab3941e21e32627aa06e3d502005bc7239111b2c6906fbd614f991a8205f0', 'MANIFEST_SHA')
    verified = []
    for row in json.loads(manifest.read_text())['files']:
        path = ROOT / row['path']
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'EXACT_CANONICAL:' + row['path'])
        verified.append(member(path))
    tests = ['project.run_scripts.jlz_cd_cumulative.test_geometry',
             'project.run_scripts.jlz_cd_cumulative.test_optimize',
             'project.run_scripts.jlz_cd_cumulative.test_collect',
             'project.run_scripts.jlz_cd_cumulative.test_controller']
    started = time.monotonic()
    result = subprocess.run([sys.executable, '-m', 'unittest', *tests, '-v'], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8'), capture_output=True, text=True)
    write(out / 'cpu-tests-r3.json', dict(passed=result.returncode == 0, returncode=result.returncode,
        tests=tests, seconds=time.monotonic() - started, stdout=result.stdout, stderr=result.stderr,
        source=[member(p) for p in sorted((ROOT / 'project/run_scripts/jlz_cd_cumulative').glob('*.py'))],
        runtime=sys.executable, Python=platform.python_version(), CUDA_visible='', actual_target_GPU=False,
        reused_design_CPU11=True, original_reference_not_rerun=True))
    require(result.returncode == 0, 'PRODUCTION_CPU_FAILED')
    write(out / 'receiver-full-read-r3.json', dict(nonce=NONCE, authority='d805e07e2ab0c346db31aa33ff0af89dd30d20a8',
        canonical_source='99a62de2635f4483514d69579dd4a5bd2e2cee53', members=verified,
        manifest=member(manifest), envelope=member(ROOT / ENVELOPE),
        independent_design_review=member(ROOT / 'audits/global/2026-10-05-jlz-cd-cumulative-2k-handoff/review.json'),
        GH_source_validation=member(ROOT / 'audits/global/2026-10-05-sh4-cd-cumulative-2k-dispatch/source-validation.json'),
        full_read='method/implementation/experiment/allTeXincludinghistorical/reference/validation/review/protocol',
        TeX_only_front_CD_Q_CD_C_operative=True, canonical_original_bytes_unchanged=True,
        old_V14_budget_inheritance=False, actual_target_GPU='NOT_RUN', submission='NOT_SUBMITTED',
        host='server4', session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd', worktree=str(ROOT)))
    print(json.dumps(dict(status='CPU_AND_RECEIVER_SHA_VERIFIED', receipt=str(out / 'cpu-tests-r3.json'))))

if __name__ == '__main__':
    main()
