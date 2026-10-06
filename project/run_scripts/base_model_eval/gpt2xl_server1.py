"""One fresh cold GPT2-XL first2k W0 evaluation; no fitting or edited state."""
import argparse
import json
import os
import resource
import time
from pathlib import Path

from .gpt2xl_server1_common import *


def summary_payload(groups):
    from project.run_scripts.jlz_interference_l1.comparison_bridge import metric_row
    return dict(edits=0, pre_state_edits=0, post_state_edits=0,
                **metric_row('W0_first2000', groups, 2000))


def tracking_start(config, lock, output):
    from project.run_scripts.experiment_tracking import init
    cfg = dict(server='server1', task_id=TASK, arm='W0_BASE_MODEL', attempt=config['run_instance']['attempt'],
        source_sha=lock['source_commit'], config_sha=lock['config_sha256'], model='gpt2xl',
        model_family='gpt2', writer='none', role='scientific', metric_schema=SCHEMA,
        job_id=os.environ['SLURM_JOB_ID'])
    tracker = init(env_file=config['tracking']['env_file'], spool=output / 'tracking', config=cfg)
    write(output / 'tracking-identity.json', dict(run_id=tracker.run_id, url=tracker.startup.get('url'),
        config=tracker.config_values, startup=tracker.startup, scientific_complete=False,
        immutable_identity=True, startup_before_model_load=True))
    return tracker


def observe(view, bench, records, expected, output, tracker=None):
    from project.run_scripts.jlz_price_gpt2xl.scores import scores
    from project.run_scripts.jlz_realization.observe import active_flags, reduce_rows
    from project.run_scripts.jlz_realized_writer_sequential.review_completed import validate_rows
    before, selected_before, hooks = view.guard(), view.selected_state(), view.hooks()
    require(all(not keys for groups in hooks.values() for keys in groups), 'W0_NO_EXTRA_HOOKS')
    rows, flags, started = [], active_flags(records), time.monotonic()
    try:
        for start in range(0, len(records), 50):
            capacity(output)
            specs, pairs = pair_specs(records[start:start+50], bench)
            values = scores(view, bench, pairs, 2)
            require(len(values) == 2 * len(specs), 'W0_ACTUAL_SCORE_COUNT')
            for index, row in enumerate(specs):
                for label, value in zip(('new', 'true'), values[2*index:2*index+2]):
                    row.update({label + '_' + key: value for key, value in value.items()})
                row.update(active_at_endpoint=flags[row['case_id']],
                    margin_true_minus_new=row['true_nll']-row['new_nll'],
                    margin_new_minus_true=row['new_nll']-row['true_nll'])
            validate_rows(specs, expected[start*13:(start+len(records[start:start+50]))*13], 'W0')
            check_finite_rows(specs)
            check_guard(before, view.guard())
            require(hooks == view.hooks(), 'W0_HOOK_MUTATION')
            write(output / f'chunk-{start:04d}.json', dict(schema='jlz-observer-rows-v1',
                selected_W=selected_before, rows=specs, optimizer_feedback=False,
                fresh_W0=True, history_present=False, edits=0))
            rows.extend(specs)
            if tracker is not None:
                tracker.log(dict(step=start+len(records[start:start+50]), edits=0, phase_id=2,
                    **{'time/elapsed_seconds':time.monotonic()-started,
                       'memory/host_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))
    finally:
        check_guard(before, view.guard())
        require(hooks == view.hooks() and selected_before == view.selected_state(), 'W0_SELECTED_BYTE_MUTATION')
    summary = dict(endpoint='W0', requests=2000, summary=reduce_rows(rows), row_count=len(rows),
        candidate_rows=2*len(rows), row_order=digest([row['identity'] for row in rows]),
        selected_W=selected_before, no_mutation=True,
        nonselected_byte_certificate='NOT_ESTABLISHED; pointer/version/scope guard only',
        fresh_W0=True, reference_only=False, optimizer_feedback=False, seconds=time.monotonic()-started,
        physical_forward_calls=view.calls, physical_candidate_rows=view.physical_candidate_rows,
        edit_calls=0, fits=0, solves=0, history_appends=0, C0_P_loads=0, checkpoint_saved=False)
    require({kind:value['denominator'] for kind,value in summary['summary'].items()} == DENOMINATORS
            and len(rows) == 26000 and view.physical_candidate_rows == 52000, 'W0_FULL_DENOMINATORS')
    write(output / 'summary.json', summary)
    return summary


def run(config_path, lock_path):
    started = time.monotonic()
    config, lock = verify_config_lock(config_path, lock_path)
    require(os.environ.get('ODEEDIT_W0_SOURCE_COMMIT')==lock['source_commit'],'W0_EXPLICIT_RUNTIME_SOURCE')
    output = Path(config['attempt']) / 'W0'
    require(not output.exists(), 'W0_NO_REEVALUATION_OR_OVERWRITE')
    output.mkdir()
    tracker, terminal, stage, model = None, {}, 'TRACKING_STARTUP', None
    try:
        # Cheap online/auth/project/readback first. No model is loaded if blocked.
        tracker = tracking_start(config, lock, output)
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from scripts.fixed_counterfact import load_prefix
        from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
        require((str(torch.__version__), transformers.__version__) ==
                (config['runtime']['torch'], config['runtime']['transformers']), 'W0_RUNTIME_VERSIONS')
        torch.set_num_threads(8)
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        stage = 'COLD_MODEL_LOAD'
        t0 = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(config['model'], local_files_only=True,
            dtype=torch.float32, attn_implementation='eager', use_safetensors=True).cuda().eval()
        model.requires_grad_(False)
        require(all(bool(torch.isfinite(weight).all()) for weight in model.parameters()), 'W0_BASE_MODEL_NONFINITE')
        view = W0View(model)
        require(view.selected_state() == config['cold_selected_W'], 'W0_ACTUAL_COLD_WEIGHT_IDENTITY')
        tok = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
        tok.pad_token = tok.eos_token
        tok.padding_side = 'right'
        bench = CounterFactAdapter(tok, read(config['contexts']['path']))
        records = load_prefix(Path(config['stream']).parent, 2000)
        require(digest([row['case_id'] for row in records]) == config['ordered_ids_sha256'], 'W0_RUNTIME_FIRST2K')
        expected = read(config['observer_identity']['path'])['rows']
        require(token_identity(records, bench) == expected, 'W0_RUNTIME_FULL_TOKENS')
        write(output / 'runtime.json', dict(torch=str(torch.__version__), transformers=transformers.__version__,
            device=torch.cuda.get_device_name(), model=config['model'], revision=config['model_revision'],
            FP32=True, eager=True, autocast=False, TF32=False, model_eval=True,
            source=lock['source_commit'], config=lock['config_sha256'], selected_W=view.selected_state(),
            fresh_model_load=True, load_seconds=time.monotonic()-t0, C0_P_H_loaded=False))
        stage = 'FRESH_W0_OBSERVE'
        summary = observe(view, bench, records, expected, output, tracker)
        accepted = tracker.log(summary_payload(summary['summary']))
        write(output / 'tracking-summary-acceptance.json', dict(accepted=accepted,
            status='SDK_ASYNC_NOT_REMOTE_ACK' if accepted else 'LOGGING_DEGRADED', scientific_complete=True))
        terminal = dict(status='COMPLETED', requests=2000, prompt_pairs=26000, candidate_rows=52000,
            source=lock['source_commit'], config=lock['config_sha256'], fresh_W0=True,
            summary=member(output / 'summary.json'))
    except BaseException as error:
        terminal = dict(status='FAILED', stage=stage, error_type=type(error).__name__,
            error=str(error)[:1200], partial_chunks=len(list(output.glob('chunk-*.json'))),
            scientific_complete=False, automatic_retry=False)
        write(output / 'failure.json', terminal)
    finally:
        if tracker is not None:
            try:
                write(output / 'tracking-finish.json', tracker.finish(
                    exit_code=0 if terminal.get('status') == 'COMPLETED' else 1, timeout=45))
            except Exception as error:
                write(output / 'tracking-finish-error.json', dict(error_type=type(error).__name__,
                    logger_failure_does_not_restart_science=True))
        import torch
        terminal.update(program_seconds=time.monotonic()-started,
            source=lock['source_commit'],config=lock['config_sha256'],
            peak_host_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            checkpoint_saved=False, exact_resume='NOT_AVAILABLE', edit_calls=0, target_fits=0,
            solves=0, history_appends=0, stats_loads=0, projector_loads=0)
        write(output / 'terminal.json', terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'freeze', 'launchers', 'run', 'collect'))
    parser.add_argument('--attempt', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--source')
    parser.add_argument('--partition', default='gpu')
    parser.add_argument('--node', default='devbox')
    parser.add_argument('--qos', default='lab_gpu_s1')
    args = parser.parse_args()
    if args.command == 'prepare':
        from .gpt2xl_server1_prepare import prepare
        result = prepare(args.attempt, args.out)
        print(json.dumps(dict(status='CPU_PREPARED_NOT_SUBMITTED', config=result['attempt']+'/config.json')))
        return
    if args.command in ('freeze', 'launchers'):
        from .gpt2xl_server1_prepare import freeze, make_launchers
        result = freeze(args.config, args.source) if args.command == 'freeze' else make_launchers(
            args.config, args.lock, args.partition, args.node, args.qos)
    elif args.command == 'collect':
        from .gpt2xl_server1_collect import collect
        result = collect(args.config, args.lock)
    else:
        result = run(args.config, args.lock)
    print(json.dumps(result, ensure_ascii=False))
    if result.get('status') == 'FAILED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
