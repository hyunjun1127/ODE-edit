"""Outcome-free source/sample/asset seal; no GPU/model load."""
import json
import subprocess
from pathlib import Path
from .identity import *
from .banks import candidate_inventory
from scripts.fixed_counterfact import verify,load_prefix
from project.run_scripts.single_layer_cumulative_risk.panels import select,ENTRIES,ranked
from project.run_scripts.single_layer_cumulative_risk.binding import DATA,MODEL,PROJECTOR,COVARIANCE

def main():
    records=load_prefix(DATA,10000);data=verify(DATA);cases={}
    for entry in ENTRIES:
        for b in ([100,1,7] if entry=='Middle' else [100]):
            panel=select(records,entry);offset=ENTRIES[entry][1]
            current=list(range(offset,offset+b))
            inv=candidate_inventory(records,offset,current,panel,dict(entry=entry,prepared_sha=sha(PREPARED[entry])))
            panel['panels']['Current100']=inv['current_effective']
            panel['generation']=ranked(inv['current_effective'],'generation',lambda i:records[i]['case_id'])[:20]
            cases[f'{entry}-B{b}']=dict(panel=panel,inventory=inv,b_raw=b,b_effective=len(inv['current_effective']),
                native_reuse_allowed=b==100 and current==inv['current_effective'])
    assets=[]
    for entry,p in PREPARED.items():
        m=member(p);original=json.loads(p.with_name('prepared-receipt.json').read_text())
        if m['sha256']!=original['sha256']:raise ValueError('PREPARED_SHA_MISMATCH')
        assets.append(m)
    inputs=json.loads((ABC/'imports/manifest-v2.json').read_text())
    for m in inputs['members']:
        if sha(m['path'])!=m['sha256']:raise ValueError('IMPORTED_SOURCE_ASSET_MISMATCH')
    for p,expected in [(PROJECTOR,'6d356468c6408dca694c1907ffde31cddb99f7e67fa2e74502910d5afbede5ec'),(COVARIANCE,'7f5fc9b194d86ce289d607a23d02de5e7d7eb5a0833aaf6ef1698d814353585e')]:
        m=member(p)
        if m['sha256']!=expected:raise ValueError('P_OR_COV_ASSET_MISMATCH')
        assets.append(m)
    model=[dict(path=str(Path(MODEL)/n),sha256=sha(Path(MODEL)/n)) for n in ('config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json')]
    index=json.loads((Path(MODEL)/'model.safetensors.index.json').read_text())
    for n in set(index['weight_map'].values()):
        p=Path(MODEL)/n
        if not p.is_file():raise FileNotFoundError(p)
        model.append(dict(path=str(p),bytes=p.stat().st_size,identity='PINNED_SNAPSHOT_REVISION'))
    source=[member(p) for p in sorted((REPO/'project/run_scripts/l4_two_memory_conflict_routing').glob('*.py'))]
    lock=dict(instruction_id=INSTRUCTION,status='CPU_ASSETS_BOUND',cases=cases,dataset=data,assets=assets,
        model=model,source=source,source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        new_paths=16,native_b100_reuse=3,project_gpu_cap=2,memory_mib=182272,cpu=8,gpu_hour_cap=None,
        same_batch_latest_wins=True,projector_threshold=.5,past_tau=.01,base_reference='We',
        gamma=1,lambda_p=1,lambda_b=1,lambda_r='0.001*trace(CE+Cresp+Cp+Cb)/d+1e-8',
        kappa=2,eta=1,epsilon='.1*(qref+1e-4)',time_steps=dict(BF1=1,BF8=.125,FrozenBF8=.125),
        generation_selection='existing ABC hash rule min(20,effective B); no new panel',
        estimated_full_matrix_fp64_bytes=14336**2*8,estimated_live_full_model_bytes=8.03e9*4,
        cost_estimate='Measure full P*/bank/key/teacher/factor/inverse/gradient/eval at Middle; do not extrapolate from Ub100',
        scientific_promotion=False)
    lock['identity']=digest(lock);save(ROOT/'control/input.lock.json',lock)
    print(str(ROOT/'control/input.lock.json'),sha(ROOT/'control/input.lock.json'),flush=True)

if __name__=='__main__':main()
