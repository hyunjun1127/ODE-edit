"""CPU regressions; no pretrained model assets, GPU or scheduler execution."""
import argparse,contextlib,io,json,time,unittest
from pathlib import Path
from . import ROOT,NONCE,TASK,DESIGN,member,write,require

def run(out):
    out=Path(out);require(not out.exists(),'CREATE_ONCE_CPU_PREFLIGHT')
    folder=ROOT/'project/run_scripts/jlz_v12r';files=sorted(folder.glob('*.py'))
    for path in files:compile(path.read_text(),str(path),'exec')
    suite=unittest.defaultTestLoader.discover(str(folder),'test_*.py',str(ROOT))
    start=time.monotonic();capture=io.StringIO()
    with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):
        result=unittest.TextTestRunner(stream=capture,verbosity=2).run(suite)
    receipt=dict(instruction_id=NONCE,task_id=TASK,status='PASS' if result.wasSuccessful() else 'FAIL',
        passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        skipped=len(result.skipped),seconds=time.monotonic()-start,source=[member(p) for p in files],
        target_pretrained_model_load=0,target_pretrained_model_forward=0,
        CPU_random_model_fixture_and_forward=True,new_GPU=0,Slurm_write=0,actual_model_qualified=False,
        canonical_original_CPU_claim='UNVERIFIED_NOT_SUPPLIED',owner_tests_not_independent_red=True,
        evidence_scope='현재 생산 source CPU fixture만; actual MAIN4요청3fixedcandidate는 별도 GPU READY')
    write(out,receipt);Path(str(out)+'.txt').write_text(capture.getvalue())
    require(result.wasSuccessful(),'PRODUCTION_CPU_PREFLIGHT_FAILED');return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    print(json.dumps(run(p.parse_args().out)))
