"""CPU-only content/resource plan preflight, never a model or scheduler call."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import hashlib
from .common import save,file_sha
from .runtime import Runtime
from .technical import validate_checkpoint_inventory

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--repo',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(args.root);design=root/'inputs/design';out=Path(args.output)
    old=json.loads((design/'evidence/audits/global/2026-09-22-alphaedit-native-criticality-audit/target-native-execution.lock.json').read_text())
    binding=json.loads((root/'receipts/native-input-binding-r1.json').read_text())
    path=next(x['resolved_path'] for x in binding['members'] if x['path']==old['fixed_dataset_binding']['loader'])
    spec=importlib.util.spec_from_file_location('alpha_cpu_dataset',path);loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    rows=loader.load_prefix(old['fixed_dataset_binding']['root'],10000)
    import torch
    torch.set_num_threads(2)
    first=torch.load(root/'inputs/checkpoints/BASE_ALPHAEDIT/B001/W-method-state.pt',weights_only=True,map_location='cpu',mmap=True)
    rt=SimpleNamespace(root=root,design=design,old=old,rows=rows,
        contract=json.loads((design/'contract.json').read_text()),contexts=first['metadata']['contexts'],
        cp_manifest=json.loads((design/'checkpoint-transfer-manifest.json').read_text()),
        cp_root=root/'inputs/checkpoints/BASE_ALPHAEDIT')
    rt.local_sidecar=lambda batch,name:Runtime.local_sidecar(rt,batch,name)
    contents=validate_checkpoint_inventory(rt,out/'checkpoint-content')
    f=os.statvfs(root)
    # GB payload accounting, not exclusive reservation. CP12 already present.
    planned_future={
        'E1_context_and_native_means':56_200_000_000,
        'E2_cal_context_assessment_means':36_500_000_000,
        'writer_24_branches_K_R_delta_coefficients':32_000_000_000,
        'writer_stage_geometry_history_banks':38_000_000_000,
        'components_KR_diagnostics_observers':12_000_000_000,
        'source_archives_atomic_partial_logs':8_000_000_000,
        'shared_volume_safety_margin':50_000_000_000}
    required=sum(planned_future.values())
    save(out/'resource-plan.json',dict(status='PASS' if f.f_bavail*f.f_frsize>=required else 'STORAGE_BLOCKED',
        free_bytes=f.f_bavail*f.f_frsize,available_inodes=f.f_favail,planned_future_bytes=planned_future,
        planned_future_total=required,CP12_existing_bytes=63_418_321_276,
        exclusive_reserved=False,storage_waiver=False,host_memory_MiB=60416,threads=8,
        task_gpu_cap=2,project_gpu_cap=2,preparation_gate_gpu=1,dependent_geometry_gpu=1,dependent_writers_gpu=1,
        gpu_peak='NOT_MEASURED; actual G1/microbatch memory evidence required inside allocated runner',
        host_estimate_peak_GiB=49,host_estimate_components='FP32 CPU mmap/P/M/entry RAM/native pre-states/features/BLAS scratch; no full model CPU copy retained',
        wall_seconds_operational_reservation=7*24*3600,
        actual_microbatch_target_solve_IO_timing='NOT_RUN; recorded by G1 and first native request/solve before remaining authorized entries',
        wall_is_scientific_budget=False,user_gpu_hour_hardcap=None))
    assert f.f_bavail*f.f_frsize>=required,'STORAGE_BLOCKED'
    save(out/'preflight.json',dict(status='PASS',checkpoint_content_receipt=contents['receipt'],
        actual_GPU_validation='NOT_RUN',input_counts={'CP':12,'design_members':23},
        source_binding_sha256=file_sha(root/'receipts/native-input-binding-r1.json'),
        design_transfer_sha256=file_sha(root/'receipts/design-transfer-r1.json'),
        checkpoint_transfer_sha256=file_sha(root/'receipts/checkpoint-transfer-r1.json'),
        missing_historical_nonexecutables=binding['unresolved']))

if __name__=='__main__':main()
