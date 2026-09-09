"""Minimal CPU input binding and dry plan; no model load."""
import json
import platform
import subprocess
from pathlib import Path
import torch
from scripts.fixed_counterfact import verify,load_prefix
from .import_assets import ROOT,REPO,sha
from .binding import DATA,MODEL,PROJECTOR,COVARIANCE,load_entry,historical
from .panels import ENTRIES,select,writer_plan
from .records import save,digest

def main():
    receipt=verify(DATA);records=load_prefix(DATA,10000)
    inputs=json.loads((ROOT/'imports/manifest-v2.json').read_text())
    for m in inputs['members']:assert sha(m['path'])==m['sha256']
    cases={}
    for name in ENTRIES:
        cp,z,rows=load_entry(name,records)
        cases[name]=dict(checkpoint_weight=cp['metadata']['state']['weights'],history=cp['metadata']['state']['cache'],
                         covariance_checkpoint_empty=cp['metadata']['covariance']=={},native_cache_requests=len(z['values']),panels=select(records,name))
    assert sha(COVARIANCE)=='7f5fc9b194d86ce289d607a23d02de5e7d7eb5a0833aaf6ef1698d814353585e'
    assert sha(PROJECTOR)=='6d356468c6408dca694c1907ffde31cddb99f7e67fa2e74502910d5afbede5ec'
    p=torch.load(PROJECTOR,map_location='cpu',weights_only=True,mmap=True)
    assert historical().tensor_sha(p[0:1])=='8c50a474f01a28afbe52f3212e79da3dc4a22f5c753a53dedc3d31ed10c2ec68'
    assert torch.isfinite(p[0]).all()
    model_members=[]
    for name in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json']:
        f=Path(MODEL)/name;model_members.append(dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size))
    index=json.loads((Path(MODEL)/'model.safetensors.index.json').read_text())
    for f in sorted(set(index['weight_map'].values())):
        path=Path(MODEL)/f;assert path.is_file();model_members.append(dict(path=str(path),bytes=path.stat().st_size,sha256='INHERITED_SNAPSHOT_REVISION_NO_DUPLICATE_MULTIGB_HASH'))
    specs=[('/mnt/raid5/janghj/.codex/attachments/b6cc6d7a-25a0-403d-a509-d0a5cfe3a0a5/goal-objective.md','c54122b0cb96ffae4320aeff99ad938e63b7ec937f314a93c7e7cf1f7da8a117'),(str(REPO/'plans/global/2026-09-10-single-layer-cumulative-risk-diagnostic-design.md'),'1b9c0e35b0115fe79c0e376775070b81abff8f4420f32625bf1332752032ee5c'),('/mnt/raid5/janghj/.codex/attachments/0a4de462-9702-431b-b050-697f91f5f2aa/pasted-text.txt','15f62756dfec69a1d34fdbed16d1bab2b2af1b9abf5c1eadf3251627400f35e7')]
    for f,h in specs:assert sha(f)==h
    lock=dict(stage='A',status='CPU_INPUTS_BOUND',base_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              contracts=[dict(path=f,sha256=h,bytes=Path(f).stat().st_size) for f,h in specs],
              dataset=receipt,entries=cases,model_members=model_members,imports_manifest_sha=sha(ROOT/'imports/manifest-v2.json'),
              projector=dict(path=PROJECTOR,stack_sha=sha(PROJECTOR),index=0,physical_layer=4,regenerate=False),
              covariance=dict(path=COVARIANCE,sha256=sha(COVARIANCE)),writers=writer_plan(),direct_steps=320,native_scalings_extra=12,
              resource=dict(server='server1',node='devbox',gpu_per_process=1,cap=2,mem_mib=182272,cpu=8,gpu_hour_cap=None,
                            cost_estimate='First native/direct optimization and evaluation actual timings; no inherited budget'),
              full_pairs=3900,curve_pairs=1100,snapshots=[0,1,2,4,8,16,24,32],curve_steps=[4,8,16,24,32],
              request_microbatch=2,context_microbatch=1,lr_selection='finite W32; mean common objective W29..W32; PS/NS influence0',
              essence='KL(student||teacher_entry), source lookup {} is a, batchmean, coefficient0.0625 once',
              B_C_gpu_released=False,scientific_promotion=False)
    save(ROOT/'input.lock.json',lock)
    print(json.dumps(dict(status=lock['status'],path=str(ROOT/'input.lock.json'),sha256=sha(ROOT/'input.lock.json'))))

if __name__=='__main__':main()
