"""CPU source/input/resource freeze, preceding any model allocation."""
import argparse
import datetime
from pathlib import Path
import shutil
import subprocess
import sys
from .common import ROOT, PANELS, BASE, MODEL, DATA, DEPS, BATCHES, INSTRUCTION, sha, read, save, file_record, digest


def prepare():
    repo=Path(__file__).resolve().parents[3]
    received=repo/'transfers/verifications/2026-09-24-native-delayed-write-e3-sh4/design-received.json'
    panels=read(PANELS/'panel-manifest.json')
    input_rows=read(PANELS/'rows.json')
    assert panels['status']=='SEALED_BEFORE_OUTCOMES' and panels['new_model_calls']==0
    members=[file_record(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.sh')]
    members += [file_record(p) for p in sorted((ROOT/'inputs/design').rglob('*')) if p.is_file()]
    members += [file_record(p) for p in sorted(PANELS.glob('*.json'))]
    members += [file_record(DATA),file_record(BASE/'execution.lock.json'),file_record(BASE/'science-policy.lock.json')]
    native=Path('/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/blue-source')
    for rel in ('AlphaEdit/AlphaEdit_main.py','memit/memit_main.py','AlphaEdit/compute_z.py','memit/compute_z.py',
                'AlphaEdit/compute_ks.py','memit/compute_ks.py','hparams/AlphaEdit/Llama3-8B.json','hparams/MEMIT/Llama3-8B.json'):
        members.append(file_record(native/rel))
    for p in sorted((BASE/'native').glob('*.py')):members.append(file_record(p))
    legacy_eval=Path('/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/worktree/project/run_scripts')
    for rel in ('alphaedit_strength_neutral_barrier/evaluator.py','alphaedit_strength_neutral_barrier/contracts.py','blue_alphaedit_sequential_comparison/evaluation.py'):
        members.append(file_record(legacy_eval/rel))
    for name in ('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors.index.json'):
        members.append(file_record(MODEL/name))
    # Exact prior full-SHA model closure is reused, not a new 16GB rehash.
    native_receipt=Path('/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/receipts/native-input-binding-r2.json')
    members.append(file_record(native_receipt))
    model_evidence=[x for x in read(native_receipt)['members'] if str(MODEL) in x['resolved_path'] and x['resolved_path'].endswith('.safetensors')]
    assert len(model_evidence)==4
    for m in model_evidence:assert Path(m['resolved_path']).stat().st_size==m['bytes']
    checkpoints={}
    for family,cell in [('BASE_ALPHAEDIT',1),('BASE_MEMIT',2)]:
        original=read(BASE/f'output/main-cell-{cell}/terminal.json');assert original['status']=='TERMINAL_VALID'
        old={x['path']:x for x in original['manifest_members']}
        checkpoints[family]={}
        for b in BATCHES:
            rel=f'B{b:03d}/W-method-state.pt'
            if family=='BASE_ALPHAEDIT':p=Path('/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/inputs/checkpoints')/family/rel
            else:p=ROOT/'inputs/checkpoints'/family/rel
            st=p.stat();assert st.st_size==old[rel]['bytes']
            checkpoints[family][str(b)]=dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,
                sha256=old[rel]['sha256'],verification='prior verified fullSHA + current stat; MEMIT fresh receiver fullSHA',
                original_terminal=file_record(BASE/f'output/main-cell-{cell}/terminal.json'))
    memit=read(ROOT/'receipts/memit-transfer.json');assert memit['status']=='PASS' and memit['count']==12
    members.append(file_record(ROOT/'receipts/memit-transfer.json'))
    prior_alpha=Path('/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/receipts/checkpoint-transfer-r1.json')
    members.append(file_record(prior_alpha))
    # Hold target before freeze; a broad queue snapshot is resource-only, never raw.
    queue=subprocess.check_output(['squeue','-u','janghj','-w','server4','-h','-o','%i|%j|%T|%b|%R'],text=True)
    partitions=subprocess.check_output(['scontrol','show','partition','gpu'],text=True)
    free=shutil.disk_usage(ROOT).free
    reserve=4*2**30
    assert free>=reserve, ('BLOCKED_STORAGE',free,reserve)
    src=digest([dict(path=str(Path(m['path']).relative_to(repo)),sha256=m['sha256']) for m in members if m['path'].startswith(str(Path(__file__).parent))])
    cpu=subprocess.run([sys.executable,'-m','unittest','project.run_scripts.native_delayed_write_e3.test_contract','-v'],cwd=repo,text=True,capture_output=True)
    if cpu.returncode:raise RuntimeError(cpu.stdout+cpu.stderr)
    receipt=dict(instruction_id=INSTRUCTION,status='CPU_PREPARED_ACTUAL_GPU_NOT_RUN',source_sha256=src,
        tested=9,returncode=cpu.returncode,test_stdout=cpu.stdout,test_stderr=cpu.stderr,
        fixed_panels=panels['pair_counts'],extra_active_variants=panels['H_active_supplementary_variants'],
        current_free_bytes=free,output_temp_safety_reserve=reserve,previous_reserve=8*2**30,
        reserve_change='General W0 teacher 4202692608B is RAM-only in persistent process; no discarded output/evaluation axes',
        host_estimate_bytes=dict(selected_24_endpoint_mmap_pages=24*5*4096*14336*4,W0_selected=5*4096*14336*4,
            General_teacher=128*64*128256*4,other_peak_scratch_and_runtime=12*2**30),
        host_limit_MiB=60416,GPU_peak_estimate_GiB=52,host_peak_estimate_GiB=44,
        walltime='1-00:00:00',wall_estimate='4–16h estimate, not measured; GPUh hardcap not specified',
        queue_snapshot=queue,partition_snapshot=partitions,cpu_only_preflight=True,
        checkpoint_saved=False,exact_new_resume='NOT_AVAILABLE',new_native_fit=0,
        no_broadcast='NO_BROADCAST_NOT_REQUIRED: same-host outputs; exact missing input pull only',
        time=datetime.datetime.now(datetime.timezone.utc).isoformat())
    rec=save(ROOT/'receipts/preflight-r1.json',receipt)
    lock=dict(instruction_id=INSTRUCTION,attempt='attempt-v1',source_sha256=src,panel_sha256=sha(PANELS/'panel-manifest.json'),
        data_sha256=sha(DATA),ordered_root='5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729',
        case_order=[r['case_id'] for r in read(DATA)],panel_rows=len(input_rows),members=members,checkpoints=checkpoints,
        preflight_receipt=rec,design_receiver=file_record(received),model_prior_fullSHA=model_evidence,
        native_fitting=0,history_append=0,save_checkpoints=False,task_gpu_cap=1,project_gpu_cap=2,
        rotation_seeds=[2026092401,2026092402,2026092403],
        numerics=dict(nll_atol=32*2**-23,nll_rtol=256*2**-23,module_relative_ceiling=1e-4,
            note='pre-outcome FP32 criteria, historical relaxed gates not inherited; repeats also recorded; strict IDs exact, own-key invariance exact'),
        resources=dict(gpus=1,cpus=8,memory_MiB=60416,export='NONE',requeue=0,node='server4',partition='gpu',walltime='1-00:00:00'),
        allowed_stages=['G00','G10','G20','G21','G30','G31','G40','G50','G51','G60','G70'],
        NOT_AUTHORIZED=['E0_new_native_continuation','E2','E4','E5','E6','new_baseline','new_history_append'])
    print(save(ROOT/'attempt-v1/execution.lock.json',lock))


if __name__=='__main__':prepare()
