"""CPU input seal; no independent regeneration of a common WN or teacher."""
import argparse
import json
from pathlib import Path
from .contracts import *
from . import banks

def cpu_seal(output):
    from scripts.fixed_counterfact import load_prefix
    from project.run_scripts.single_layer_cumulative_risk.panels import select
    records=load_prefix(DATA,10000);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    references=[]
    for name,(batch,offset,attempt) in ENTRIES.items():
        prepared=ABC/'A'/name/attempt/'prepared.pt'
        paths=[prepared,ABC/'imports/entries'/f'B{batch:03d}'/'W-method-state.pt',
          ABC/'imports/entries'/f'B{batch+1:03d}'/'native-targets.pt',
          ABC/'imports/entries'/f'B{batch+1:03d}'/'entry.json',ABC/'imports/config.json']
        panel=select(records,name);inv=banks.select(records,offset,range(offset,offset+100),panel)
        panel['panels']['Current100']=inv['current_effective']
        data=dict(entry=name,batch=batch,offset=offset,panel=panel,inventory=inv,members=[member(p) for p in paths],
          native_reuse_eligible=inv['current_effective']==inv['current_raw'],
          model_revision=MODEL.name,model_config=hf_member(MODEL/'config.json'),
          native_risk_calibration='PENDING_NEW_BANK_OBSERVATION',L8_history='PENDING_SAME_We_RECONSTRUCTION',
          teacher_and_forward_identity='PENDING_ACTUAL_MODEL',cross_server_VERIFIED=False)
        save(output/f'{name}-cpu-input.json',data);references.append(member(output/f'{name}-cpu-input.json'))
    save(output/'CPU_INPUT_READY.json',dict(status='CPU_INPUT_SEALED_NOT_GPU_COMMON_READY',members=references,
      members_root=digest(references),dataset=member(DATA/'counterfact.json'),
      historical_input_path_policy='server1 local only; no deleted server4 paths',scientific_promotion=False))
    print(json.dumps(dict(output=str(output),status='CPU_INPUT_SEALED_NOT_GPU_COMMON_READY')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();cpu_seal(a.output)
