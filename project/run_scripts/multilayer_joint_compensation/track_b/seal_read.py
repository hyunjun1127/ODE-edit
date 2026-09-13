"""B task 원문·설계·참조 full-read와 실제 source identity를 create-once 봉인."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[4]
CONTROL=Path('/mnt/raid5/janghj/ODE-edit/local/multilayer-joint-compensation/20260911-v1/track_b/control-v1')

def main():
    members=[]
    names=['messages/head/2026-09-11-multilayer-damage-compensation-b-sh2.md',
           'messages/head/2026-09-11-multilayer-joint-compensation-common.md',
           'transfers/approvals/2026-09-11-multilayer-joint-compensation-input-sharing.md',
           'plans/global/2026-09-11-multilayer-joint-edit-and-compensation-design.md',
           'plans/global/2026-09-11-multilayer-joint-edit-and-compensation-design-math-check.json',
           'experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/independent-review-ko.md',
           'experiment-reports/global/l4-two-memory-v2-independent-audit-2026-09-11-v1/independent-review-ko.md',
           'PROTOCOL.md','servers/connection-inventory.md',
           'project/run_scripts/l4_two_memory_conflict_routing/controller_repair.py',
           'project/run_scripts/l4_two_memory_conflict_routing/native.py',
           'project/run_scripts/single_layer_cumulative_risk/binding.py']
    paths=[ROOT/n for n in names]
    paths += [Path('/mnt/raid5/janghj/ODE-edit/local/multilayer-joint-compensation/20260911-v1/imports/authority-v1/pasted-text.txt')]
    source=Path('/mnt/raid5/janghj/ODE-edit/local/blue-l4-progress-barrier/attempt-v1/imports/imports/blue-source/AlphaEdit')
    paths += [source/'AlphaEdit_main.py',source/'compute_z.py']
    authority=CONTROL/'authority';authority.mkdir(parents=True,mode=0o700,exist_ok=False)
    for i,p in enumerate(paths):
        assert p.is_file() and not p.is_symlink()
        data=p.read_bytes();s=hashlib.sha256(data).hexdigest()
        if p.name=='pasted-text.txt':assert (s,len(data),data.count(b'\n'))==('9094300705dca8f07b73dc9a6ba116c8016f660081f3dc9171a57544ece66400',8940,114)
        if p.name=='2026-09-11-multilayer-joint-edit-and-compensation-design.md':assert s=='a8f32937ddd2fc4f87cb3a2cfb9ef95de67d2fd39ce877142fa7b80a1884bf05'
        dest=authority/f'{i:02d}-{p.name}'
        fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as f:f.write(data)
        members.append(dict(path=str(p),copy=str(dest),bytes=len(data),lf_lines=data.count(b'\n'),sha256=s,full_read=True))
    result=dict(status='FULL_READ_PASS',instruction_id='ODEEDIT-S06-MULTILAYER-DAMAGE-COMPENSATION-B-SH2-V1',
      source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      source_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip(),members=members,
      current_scope='B8 endpoints + native6 same-entry + B-BF4 B51-B60 chain; shared kernels',
      gpu_cap=2,mem_mib=60416,submission=0,other_task_monitoring_resumed=False,
      common_fixture_status='AWAITING_SH1_SEAL_NO_INDEPENDENT_REGENERATION')
    with (CONTROL/'read-receipt.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print('FULL_READ_PASS',len(members),hashlib.sha256((CONTROL/'read-receipt.json').read_bytes()).hexdigest())

if __name__=='__main__':main()
