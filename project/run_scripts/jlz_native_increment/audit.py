"""Bounded owner CPU/source checklist, never an independent/GPU PASS claim."""
import argparse,importlib.util,io,json,unittest
from pathlib import Path
import torch
from .common import *
from . import test_cpu

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromModule(test_cpu)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    write(args.out/'production-cpu.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        pass_=result.wasSuccessful(),test_names=sorted(n for n in dir(test_cpu.CPU) if n.startswith('test_')),
        actual_GPU=False,independent_reviewer=False,owner_review=True))
    require(result.wasSuccessful(),'CPU_REGRESSION')
    path=ROOT/DESIGN/'math/validate_contract.py'
    spec=importlib.util.spec_from_file_location('v11_sealed_cpu_contract',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cases=[module.check_case(*c) for c in [(1,1,3,11),(2,2,4,22),(3,3,4,33),(2,3,3,44)]]
    proof=dict(cases=cases,coupling=module.coupling_and_writer_checks(),selection=module.selection_and_clamp_checks())
    write(args.out/'sealed-math-reproduction.json',dict(status='PASS',source=member(path),proof=proof,
        actual_model=False,scope='same four tiny synthetic dimensions; canonical source/results unchanged'))
    require(not torch.cuda.is_initialized(),'CPU_AUDIT_INITIALIZED_GPU')
    checklist={
        'authority_input':'17 canonical files and all2000 rows/fields,22 packs; original BLUE tokens/lookup verified in receiver receipt',
        'objective':'subject.native MEAN reporting/SUM gradients; allocation full-B once; q bridge scales once; CPU B1/B3 regression',
        'geometry':'frozen entry SPD factor; full offdiagonal; stable root zero subgradient; no jitter or symmetrization; native reference comparison runtime guard',
        'optimizer':'LR.1/warmup0/25eval24update/totalmean<.05 stop; final evaluated D; moments retained after physical clamp',
        'writer':'no candidate actual writer; terminal actual lower write precedes upper K; D not absolute residual; final key history once tested',
        'baseline':'pinned original BLUE residual/divisor/singleton z; fresh same-runtime20; process-local container adapter only',
        'pilot':'three independent cold BS2x2; fixed cached/full zero/nonzero and MB1 comparison; original native token/key/NLL alignment; no B100 extra fit',
        'observers':'outside fit; every pre/post; W5/10/15/20 allseen current subset; same shared freshW0 state/runtime/input; no quality feedback',
        'transaction':'W/H/RNG RAM rollback tested; observer state/RNG guards; initial B2 pack verified before receipt',
        'storage':'scalar/hash/metrics only; no persisted W/H/D/P/U/optimizer; no old tensor exception; noCP and noB21 tests',
        'resource':'cap1 serial DAG inside current USER project cap bounded by local/tracked policy; exact terminal reconciliation for explicit retry; existing other jobs untouched; all held inspection priorrelease',
        'collector':'model-free raw identity/count/NLL/TF/paired/cohort/partial checks; terminal after report+inventory',
        'limitations':'Actual LM native/matrix/fullgradient parity and B100 RAM remain NOT_OBSERVED until sealed pilot/MAIN execute; CPU is not GPU evidence'}
    write(args.out/'owner-red-checklist.json',dict(verdict='PASS_CPU_SOURCE_WITH_ACTUAL_GATES_PENDING',checks=checklist,
        source_members=[member(p) for p in sorted(Path(__file__).parent.glob('*.py'))],
        owner_audit=True,independent_red=False,actual_GPU='NOT_RUN',new_experiment_quality_gate=False))
    print(json.dumps(dict(CPU_tests=result.testsRun,synthetic_dimensions=4,actual_GPU='NOT_RUN',independent_red=False)))

if __name__=='__main__':main()
