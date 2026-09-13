"""Run independent CPU suites; shared kernels are verified read-only dependencies."""
import importlib.util
import argparse
import json
from pathlib import Path
import subprocess
import time
import unittest
from unittest.mock import patch
import torch
import project.run_scripts.multilayer_joint_compensation as pkg

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--local-shared',action='store_true');args=parser.parse_args()
source=Path('project/run_scripts/multilayer_joint_compensation') if args.local_shared else Path('/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-multilayer-joint-edit-a-v1/project/run_scripts/multilayer_joint_compensation')
for f in ('functional.py','linear_solve.py','elastic_qp.py','track_b/native_adapter.py'):
    assert (source/f).read_bytes()==subprocess.check_output(['git','show','ba91f274:project/run_scripts/multilayer_joint_compensation/'+f])
if not args.local_shared:pkg.__path__.append(str(source))
a_files=subprocess.check_output(['git','ls-tree','-r','--name-only','b3b40db4','--','project/run_scripts/multilayer_joint_compensation/tests'],text=True).splitlines()
a_suite=unittest.TestSuite()
for f in a_files:
    if not Path(f).name.startswith('test_'):continue
    spec=importlib.util.spec_from_file_location(Path(f).stem,f)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    a_suite.addTests(unittest.TestLoader().loadTestsFromModule(module))
suites=[unittest.TestLoader().discover('project/run_scripts/baseline_mechanism_first/tests',top_level_dir='.'),
        a_suite]
mods=['test_p1r54_pdz_ablation','test_p1r55_rms_pdz_floored_rate','test_p1r52_c_writer_kstep','test_p1r52_c_writer_kstep_cache','test_p1r54_realization_reset','test_p1r52_target_timescale']
suites.append(unittest.TestSuite(unittest.TestLoader().loadTestsFromName('project.run_scripts.ode_bf.tests.'+m) for m in mods))
spec=importlib.util.spec_from_file_location('integration_tests',root/'focused_checks.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
suites.append(unittest.TestLoader().loadTestsFromModule(m))
counts=[s.countTestCases() for s in suites];start=time.monotonic()
with patch('project.run_scripts.ode_bf.p1r52_target_timescale_deployment.socket.gethostname',return_value='server4'):
    r=unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(suites))
assert not torch.cuda.is_initialized()
out={'status':'PASS' if r.wasSuccessful() else 'FAIL','test_count':r.testsRun,'suite_counts':counts,
     'errors':len(r.errors),'failures':len(r.failures),'skipped':len(r.skipped),'seconds':time.monotonic()-start,
     'cuda_initialized':False,'source_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
     'shared_dependency':('current clean integration local shared closure, ba91f274 exact kernel bytes' if args.local_shared else 'read-only ba91f274 SH2 kernel bytes; SH2 main integration separately owned'),
     'fixture_boundary':'CPU mock socket.gethostname=server4 for existing server4 capacity fixture; no real hostname/resource change',
     'initial_findings':['audit invocation PYTHONPATH binding corrected','audit-only wrong constant import corrected',
                         'T5 required original c0dc8e3f horizon patch, now integrated exactly',
                         'unmocked historical server4 fixture fails on devbox; unchanged source, mock recorded',
                         'combined discovery uses independent TestLoader per root to avoid cached top-level import mismatch']}
(root/('postmerge-focused-tests.json' if args.local_shared else 'focused-tests.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False))
raise SystemExit(not r.wasSuccessful())
