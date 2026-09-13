"""CPU-only publication checks; no scheduler, model assets, or experiment raw."""
import ast,hashlib,importlib,json,os,subprocess,sys,types,unittest
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(2)
PACKAGE='project.run_scripts.multilayer_joint_compensation'
LEGACY=Path('/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-multilayer-damage-compensation-b-v1')
FROZEN='6f5e5456'

def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def main():
    # SH1 owns these dependencies. Reuse frozen source for tests without
    # publishing/modifying it. New SH2 modules always resolve from ROOT first.
    common=['__init__.py','contracts.py','observations.py','evaluation.py','history.py',
            'banks.py','fixture.py','track_a/native_geometry.py','tests/test_observations.py']
    dep=[]
    for rel in common:
        path='project/run_scripts/multilayer_joint_compensation/'+rel
        expected=git('show',FROZEN+':'+path)
        actual=(LEGACY/path).read_bytes()
        if actual!=expected:raise ValueError('FROZEN_COMMON_CHANGED:'+path)
        dep.append(dict(path=path,source_commit=git('rev-parse',FROZEN).decode().strip(),sha256=hashlib.sha256(actual).hexdigest(),owner='SH1',published_by_this_task=False))
    for suffix in ('','tests','track_a'):
        name=PACKAGE+('.'+suffix if suffix else '')
        module=types.ModuleType(name)
        rel=Path('project/run_scripts/multilayer_joint_compensation')/suffix
        module.__path__=[str(ROOT/rel),str(LEGACY/rel)]
        sys.modules[name]=module
        parent,attr=name.rsplit('.',1)
        setattr(importlib.import_module(parent),attr,module)
    names=['test_pcg','test_elastic','test_functional','test_b_protocol','test_native_baselines',
           'test_b_admission','test_b_partial_analysis','test_b_fd_refinement','test_b_runtime_gate','test_b_followup']
    loader=unittest.TestLoader()
    suite=unittest.TestSuite(loader.loadTestsFromName(PACKAGE+'.tests.'+n) for n in names)
    # 11 pure HA0 tests. The other two require Official imports; keep them
    # separate from this run and retain their historical source-bound receipt.
    ha=loader.loadTestsFromName('project.run_scripts.fzcb_hard_alpha_densec.tests.test_ha0_backend')
    def flatten(s):
        for x in s:
            if isinstance(x,unittest.TestSuite):yield from flatten(x)
            else:yield x
    excluded=[]
    for test in flatten(ha):
        if test._testMethodName in ('test_official_trace_kwargs_adapter_preserves_output','test_single_shape_adapter_matches_official'):
            excluded.append(test.id())
        else:suite.addTest(test)
    suite.addTests(loader.loadTestsFromName('project.run_scripts.fixed10k_preedit_eval.test_focused'))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        failures=[(t.id(),v) for t,v in result.failures],errors=[(t.id(),v) for t,v in result.errors],
        skipped=result.skipped,official_import_tests_not_rerun=excluded,
        dependency_mode='frozen SH1 common read-only; new SH2 source first in import paths',
        dependencies=dep,torch=torch.__version__,python=sys.version,
        GPU=0,Slurm=0,model_asset_load=0,experiment_raw=0)
    (OUT/'cpu-checks-r2.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
if __name__=='__main__':main()
