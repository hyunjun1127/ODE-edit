"""Bounded CPU audit receipt; explicitly not a new GPU qualification."""
import json
import os
import subprocess
from pathlib import Path
from .common import *

def main():
    preparation=LOCAL/'preparation-r2'
    c=json.loads((preparation/'configuration.json').read_text())
    bridge=json.loads(verify(c['qualification_reuse']['receipt']).read_text())
    for row in bridge['source_members']:
        require(sha(ROOT/row['path'])==row['sha256'],'UNCHANGED_FROZEN_CLOSURE')
    for row in bridge['evidence']:verify(row)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    result=subprocess.run([c['runtime']['python'],'-B','-m','unittest',
        'project.run_scripts.jlz_realization_2k.test_pipeline','project.run_scripts.jlz_realization.test_core','-v'],
        cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
    log=LOCAL/'CPU-preflight-r1.log'
    with log.open('x') as f:f.write(result.stdout+'\n'+result.stderr)
    require(result.returncode==0 and 'Ran 22 tests' in result.stderr,'CPU_REGRESSION')
    report=dict(instruction_id=INSTRUCTION,status='CPU_PREFLIGHT_PASS',new_actual_GPU_PASS=False,
        owner_audit='SH4 source/controller/reducer/runtime audit',independent_reviewer='NOT_PERFORMED',
        CPU_tests=22,CPU_stdout=member(log),authority=AUTHORITY,frozen_core=CORE,
        frozen_core_files=25,frozen_import_closure_files=len(bridge['source_members']),
        actual_import_prefix_files=27,actual_import_prefix_evidence='CPU import and fresh SHA match; prior GPU import receipt bound separately',
        original_Q1_B_Slurm=bridge['prior_Q1_B_Slurm'],qualification_reuse=member(preparation/'qualification-reuse.json'),
        full_read=member(preparation/'full-read.json'),config=member(preparation/'configuration.json'),
        input_rows=2000,observer_identity_rows=26000,packing_rows=22,
        observer_identity=member(preparation/'observer-identity.json'),baseline_status=[(r['name'],r['status']) for r in c['baselines']],
        assertions=['20 batches/noB21','W5/W10/W20 all-seen and current reuse','old500 not complete2k',
            'strict commit/source/config/ownstate','partial and early error collector','no new Q2/noCP/no replay',
            'two independent1GPU held argv','unchanged frozen native/optimizer/allocation/geometry/causal core'],
        resources=c['resources'],storage='30GiB reserve; no deletion or old waiver',
        cost='prior measured500 plus finite2k estimate, wall not ETA',job_ids=[],status_at_audit='IMPLEMENTING_NOT_SUBMITTED',
        source_files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
            for p in sorted((ROOT/'project/run_scripts/jlz_realization_2k').glob('*.py'))])
    write(ROOT/f'audits/servers/server4/{TASK}/CPU-preflight.json',report)
    print(json.dumps(dict(status=report['status'],tests=22,job_ids=[])))

if __name__=='__main__':main()
