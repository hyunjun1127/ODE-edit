"""Cold 20-batch controller. All scientific batch calls use unchanged frozen v9."""
import argparse
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from project.run_scripts.jlz_realization.run import batch, seed
from project.run_scripts.jlz_realization.profile import LlamaAdapter
from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
from project.run_scripts.jlz_realization.observe import observe
from .common import INSTRUCTION, TASK, require, verify, write, member, sha, digest, state
from .schedule import BATCHES, selection, denominators, validate_config, verify_commit

def setup(config, out):
    require(config['instruction_id'] == INSTRUCTION and config['task_id'] == TASK, 'TASK_AUTHORITY')
    validate_config(config)
    require(torch.__version__ == config['runtime']['torch']
            and transformers.__version__ == config['runtime']['transformers'], 'RUNTIME_VERSION')
    for row in config['assets']:
        s = Path(row['path']).stat()
        require((s.st_size, s.st_ino, s.st_mtime_ns) == (row['bytes'], row['inode'], row['mtime_ns']), 'ASSET_CHANGED')
    torch.set_num_threads(8)
    seed(config)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(config['model'], local_files_only=True,
        dtype=torch.float32, attn_implementation='eager', low_cpu_mem_usage=True).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    adapter = LlamaAdapter(model, config['profile'])
    bench = CounterFactAdapter(tokenizer, json.loads(Path(config['contexts']).read_text()))
    history = {l: torch.zeros(shape[1], shape[1], dtype=torch.float32) for l, shape in adapter.dims.items()}
    records = json.loads(Path(config['stream']).read_text())
    write(out/'runtime.json', dict(instruction=INSTRUCTION, task=TASK, job_id=os.environ.get('SLURM_JOB_ID'),
        device=torch.cuda.get_device_name(), versions=config['runtime'], model=config['model'],
        profile=adapter.profile, checkpoint_saved=False, cold_W0_H0=True))
    imports = {name: member(m.__file__) for name, m in list(sys.modules.items())
        if getattr(m, '__file__', None) and Path(m.__file__).is_file()
        and name.startswith(('project.run_scripts.jlz_realization', 'project.run_scripts.jlz_writer_coupled',
                             'project.run_scripts.jlz_pilot.prompts', 'transformers.models.llama'))}
    write(out/'actual-imports.json', imports)
    expected = config['qualification_reuse']['actual_import_hashes']
    for name, value in expected.items():
        require(name in imports and imports[name]['sha256'] == value, 'REUSED_IMPORT_CHANGED:' + name)
    return adapter, bench, records, history

def drive(adapter, bench, records, history, config, arm, out, source):
    """Pure schedule plus fixed batch/observer calls; no exact callback or scheduler."""
    commits = []
    packing = {x['batch']: x for x in config['packing'] if x['phase'] == 'main'}
    for number in BATCHES:
        current, seen, selected = selection(records, number)
        if commits:
            require(state(adapter, history) == commits[-1]['after'], 'OWN_ENTRY_CONTINUITY')
        def callback(st, pack):
            require(pack['identity'] == packing[number]['identity']
                    and pack['record_ids'] == packing[number]['ids'], 'PACKING_LOCK')
            if number == 2:
                write(out/'initial.json', dict(instruction=INSTRUCTION, task=TASK, main_B1_commit=True,
                    all5_history=True, observer_restored=True, main_B2_ownentry=True, state=st,
                    input=pack['identity'], source=source, config=digest(config), representative_only=True))
        receipt = batch(adapter, bench, current, history, config, arm,
                        out/f'batch-{number:02d}', number, callback=callback, q2=False)
        verify_commit(receipt, number, [r['case_id'] for r in current], source, digest(config))
        commits.append(receipt)
        observation = observe(adapter, bench, seen, selected, history, number, out/f'observe-W{number:02d}',
            config['settings']['observer_microbatch'], [r['case_id'] for r in current])
        require({k:v['denominator'] for k,v in observation['summary'].items()} == denominators(number), 'EVAL_COUNTS')
        require({k:v['denominator'] for k,v in observation['current'].items()} == dict(R=100,P=200,N=1000), 'CURRENT_REUSE_COUNTS')
    return commits

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=('A','B'), required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--attempt', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    out = args.attempt/('main-' + args.arm)
    out.mkdir(parents=True, exist_ok=False)
    started, status = time.monotonic(), 'TECHNICAL_FAILED'
    try:
        lock = json.loads((args.attempt/'execution.lock.json').read_text())
        source = os.environ.get('ODEEDIT_SOURCE_COMMIT')
        require(source == lock['source_commit'] and sha(args.config) == lock['config_sha256'], 'SOURCE_CONFIG_LOCK')
        for row in lock['source_members'] + lock['runtime_sources'] + lock['native_reference']:
            verify(row)
        reuse = config['qualification_reuse']
        bridge = json.loads(verify(reuse['receipt']).read_text())
        require(bridge['status'] == 'REUSE_VERIFIED_CORE_RUNTIME_INPUT', 'Q1_REUSE_STATUS')
        for row in bridge['evidence']:
            verify(row)
        write(out/'upstream.json', dict(instruction=INSTRUCTION, explicit_reuse=reuse, new_Q1_fits=0, new_Q2=0))
        adapter, bench, records, history = setup(config, out)
        initial = state(adapter, history)
        write(out/'initial-state.json', initial)
        w0 = json.loads(verify(config['w0_reuse']['receipt']).read_text())
        require(initial == w0['state'], 'COLD_W0_IDENTITY')
        write(out/'W0-reuse.json', dict(new_forward=0, receipt=config['w0_reuse']['receipt'], actual_cold_state=True))
        drive(adapter, bench, records, history, config, args.arm, out, source)
        status = 'COMPLETED'
    except BaseException as exc:
        write(out/'first-error.json', dict(type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
            rollback_after_process_termination='NOT_VERIFIED', no_checkpoint=True))
        raise
    finally:
        write(out/'terminal.json', dict(status=status, phase='main', arm=args.arm, instruction=INSTRUCTION,
            source=os.environ.get('ODEEDIT_SOURCE_COMMIT'), config_sha256=sha(args.config),
            main_commits=len(list(out.glob('batch-*/commit.json'))), job_id=os.environ.get('SLURM_JOB_ID'),
            seconds=time.monotonic()-started, checkpoint_saved=False, no_B21=True, new_exact_probes=0,
            peak_host_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None))

if __name__ == '__main__':
    main()
