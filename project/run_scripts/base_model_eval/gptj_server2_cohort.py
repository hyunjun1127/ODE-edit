"""Fresh GPT-J W0 only: canonical PRICE scorer, no writer/history/statistics."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import resource
import time

from project.run_scripts.jlz_realization.common import digest, member, require, sha, write
from project.run_scripts.jlz_realization.observe import reduce_rows
from project.run_scripts.jlz_interference_l1 import validate_rows
from project.run_scripts.jlz_interference_l1.comparison_bridge import metric_row
from scripts.fixed_counterfact import load_prefix

TASK = 'base-model-gptj-w0-cohort-curves'
NONCE = 'USER-GH-SH1-SH2-W0-COHORT-CURVES-RERUN-20261007-R1-SERVER2'
REVISION = '47e169305d2e8376be1d31e765533382721b2cc1'
COUNTS = dict(R=2000, P=4000, N=20000)


def read(path):
    return json.loads(Path(path).read_text())


def guard(model):
    return [(n, p.data_ptr(), p._version, tuple(p.shape), str(p.dtype))
            for n, p in list(model.named_parameters()) + list(model.named_buffers())]


def payload(summary):
    return dict(edits=0, pre_state_edits=0, post_state_edits=0,
                **metric_row('W0_first2000', summary, 2000))


def specifications(bench, records):
    refs = []
    for ordinal, record in enumerate(records):
        rw = record['requested_rewrite']
        panels = bench.panels(record)
        require({k: len(v) for k, v in panels.items()} == dict(R=1, P=2, N=10), 'PANEL_COUNTS')
        for kind, prompts in panels.items():
            for index, prompt in enumerate(prompts):
                tokens = {label: bench.evaluation_ids(prompt, rw['target_' + label]['str'])
                          for label in ('new', 'true')}
                refs.append(dict(ordinal=ordinal, case_id=record['case_id'], kind=kind, prompt_index=index,
                    identity=digest([record['case_id'], kind, index, prompt,
                                     rw['target_new']['str'], rw['target_true']['str']]),
                    record_sha256=digest(record), prompt_sha256=digest(prompt),
                    new_token_identity=digest(tokens['new']), true_token_identity=digest(tokens['true']),
                    tokens=tokens))
    require(len(refs) == 26000 and len({r['identity'] for r in refs}) == 26000, 'ROW_IDENTITY')
    return refs


class W0Adapter:
    """Only readout path required by unchanged jlz_price_gptj.scores."""
    def __init__(self, model):
        c = model.config
        require(c.model_type == 'gptj' and (c.n_layer, c.n_embd, c.n_inner or 4*c.n_embd,
                c.vocab_size) == (28, 4096, 16384, 50400), 'GPTJ_ARCHITECTURE')
        require(c._attn_implementation == 'eager', 'EAGER_REQUIRED')
        self.model, self.device = model, next(model.parameters()).device

    def observer_hidden(self, **tokens):
        return self.model.transformer(**tokens, use_cache=False).last_hidden_state


def run(attempt):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
    from project.run_scripts.jlz_price_gptj.scores import scores
    from .gptj_server2_cohort_tracking import init
    from .gptj_server2_cohort_tracking.schema import EXTRA
    from project.run_scripts.experiment_tracking.schema import COMPARISON_SCHEMA

    c, lock = read(attempt/'config.json'), read(attempt/'execution.lock.json')
    require(c['instruction_id'] == NONCE and sha(attempt/'config.json') == lock['config_sha256'], 'LOCK')
    require(not (attempt/'terminal.json').exists() and not (attempt/'raw').exists(), 'NO_RERUN')
    require(torch.__version__ == c['runtime']['torch'] and transformers.__version__ == c['runtime']['transformers'], 'RUNTIME')
    for item in c['assets'] + c['runtime']['members']:
        st = Path(item['path']).stat()
        require((st.st_size, st.st_ino, st.st_mtime_ns) ==
                (item['bytes'], item['inode'], item['mtime_ns']), 'INPUT_STAT_CHANGED')
    for item in lock['source_members']:
        require(sha(attempt/'source'/item['relative']) == item['sha256'], 'SOURCE_CHANGED')
    torch.set_num_threads(c['resources']['cpu'])
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = load_prefix(Path(c['dataset']), 2000)
    require(digest([r['case_id'] for r in records]) == c['ordered_ids'], 'ORDER')
    require(sha(c['observer_manifest']['path']) == c['observer_manifest']['sha256'], 'OBSERVER_MANIFEST')
    refs = read(c['observer_manifest']['path'])
    tok = AutoTokenizer.from_pretrained(c['model'], local_files_only=True)
    tok.pad_token = tok.eos_token; tok.padding_side = 'right'
    bench = CounterFactAdapter(tok, [])  # evaluation_ids/panels never use generated contexts
    require(digest(specifications(bench, records)) == digest(refs), 'TOKEN_RECORD_BINDING')
    cfg = dict(server='server2', task_id=TASK, arm='W0_BASE_MODEL', attempt=attempt.name,
        source_sha=lock['source_commit'], config_sha=lock['config_sha256'], model='gptj',
        model_family='gptj', writer='none', role='scientific', metric_schema=COMPARISON_SCHEMA,
        observation_identity=c['observer_manifest']['sha256'], **EXTRA)
    tracker = None; started = time.monotonic(); result = None
    try:
        tracker = init(env_file=c['tracking_env'], spool=attempt/'tracking', config=cfg)
        startup_accepted=tracker.log({'phase_id': 0, 'step': 0, 'edits': 0})
        write(attempt/'tracking-start-log.json',dict(accepted=startup_accepted))
        load_start = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(c['model'], local_files_only=True,
            dtype=torch.float32, attn_implementation='eager', low_cpu_mem_usage=True,
            use_safetensors=False).to('cuda').eval()
        model.requires_grad_(False); model.config.use_cache = False
        adapter = W0Adapter(model)
        require(all(p.dtype == torch.float32 and not p.requires_grad for p in model.parameters()), 'FP32_FROZEN')
        with torch.no_grad():
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'NONFINITE_MODEL')
        before = guard(model)
        runtime = dict(torch=torch.__version__, transformers=transformers.__version__,
            GPU=torch.cuda.get_device_name(), total_VRAM_bytes=torch.cuda.get_device_properties(0).total_memory,
            model_load_seconds=time.monotonic()-load_start, dtype='FP32', eager=True,
            TF32=False, autocast=False, readout_block=27, no_C0_P_H=True)
        write(attempt/'runtime.json', runtime)
        rows = []; eval_start = time.monotonic()
        try:
            for start in range(0, 2000, 50):
                require(__import__('shutil').disk_usage(attempt).free >= c['reserve_bytes'], 'DISK_RESERVE')
                group = records[start:start+50]; pairs = []
                for record in group:
                    rw = record['requested_rewrite']
                    for prompts in bench.panels(record).values():
                        for prompt in prompts:
                            pairs.extend([(prompt, rw['target_new']['str']), (prompt, rw['target_true']['str'])])
                values = scores(adapter, bench, pairs, c['microbatch'])
                chunk_refs = refs[start*13:(start+50)*13]
                require(len(values) == len(chunk_refs)*2, 'CANDIDATE_COUNT')
                chunk = []
                for i, ref in enumerate(chunk_refs):
                    row = {k: ref[k] for k in ('case_id', 'kind', 'prompt_index', 'identity')}
                    row.update(endpoint='W0', ordinal=ref['ordinal'])
                    for label, value in zip(('new', 'true'), values[2*i:2*i+2]):
                        row.update({label+'_'+k: v for k,v in value.items()})
                    row['margin_true_minus_new'] = row['true_nll']-row['new_nll']
                    chunk.append(row)
                validate_rows(chunk, chunk_refs, [r['case_id'] for r in group], 'W0')
                require(guard(model) == before, 'MODEL_MUTATION')
                write(attempt/'raw'/f'chunk-{start:04d}.json', dict(rows=chunk, optimizer_feedback=False))
                rows.extend(chunk)
                progress_accepted=tracker.log({'phase_id': 1, 'step': start+50, 'edits': 0,
                             'time/elapsed_seconds': time.monotonic()-started})
                if not progress_accepted:
                    write(attempt/'tracking-rejected'/f'progress-{start}.json', dict(accepted=False,step=start+50))
        finally:
            require(guard(model) == before, 'MODEL_MUTATION')
        summary = validate_rows(rows, refs, [r['case_id'] for r in records], 'W0')
        require({k:v['denominator'] for k,v in summary.items()} == COUNTS, 'FINAL_COUNTS')
        from .gptj_server2_cohort_metrics import curves
        mapped_curves = curves(rows)
        mapped = mapped_curves[0]
        write(attempt/'cohort-curves.json', mapped_curves)
        result = dict(status='COMPLETED', summary=summary, scalar_payload=mapped,
            requests=2000, prompt_pairs=26000, candidates=52000, no_mutation=True,
            guard_sha256=digest(before), new_actual_evaluation=True, checkpoint_saved=False,
            edit_calls=0, fits=0, solves=0, history_appends=0,
            evaluation_seconds=time.monotonic()-eval_start, load_seconds=runtime['model_load_seconds'],
            elapsed_seconds=time.monotonic()-started, peak_VRAM_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), max_host_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            source=lock['source_commit'], config_sha256=lock['config_sha256'],
            raw=[member(p) for p in sorted((attempt/'raw').glob('*.json'))])
        write(attempt/'result.json', result)
        accepted=[tracker.log(p) for p in mapped_curves]
        write(attempt/'curve-transport.json', dict(accepted=accepted, schema_validated=True,
            SDK_async_not_remote_ACK=True, logging_degraded=not all(accepted)))
    except BaseException as error:
        write(attempt/'failure.json', dict(status='FAILED', error_type=type(error).__name__,
              error_code=str(error)[:240] if isinstance(error, (ValueError, RuntimeError)) and tracker is not None else 'REDACTED_OR_STARTUP_ERROR',
              elapsed_seconds=time.monotonic()-started, no_retry=True))
        raise
    finally:
        tracking = tracker.finish(exit_code=0 if result else 1) if tracker is not None else {'status':'STARTUP_NOT_ESTABLISHED'}
        write(attempt/'terminal.json', dict(status='COMPLETED' if result else 'FAILED',
            result_sha256=sha(attempt/'result.json') if result else None, tracking=tracking,
            elapsed_seconds=time.monotonic()-started, no_checkpoint=True))


def collect(attempt):
    """CPU-only independent arithmetic on the exact fresh stored pairs."""
    c = read(attempt/'config.json')
    terminal = read(attempt/'terminal.json') if (attempt/'terminal.json').exists() else {'status':'MISSING_TERMINAL'}
    rows = []; manifest = []
    for p in sorted((attempt/'raw').glob('chunk-*.json')):
        manifest.append(member(p)); rows.extend(read(p)['rows'])
    groups = reduce_rows(rows) if rows else {}
    independent = {}
    for kind in 'RPN':
        rs = [r for r in rows if r['kind']==kind]; desired = 'true' if kind=='N' else 'new'
        wins = sum(r['true_nll'] < r['new_nll'] if kind=='N' else r['new_nll'] < r['true_nll'] for r in rs)
        independent[kind] = dict(count=len(rs), success_count=wins,
            token_correct=sum(r[desired+'_token_correct'] for r in rs), token_count=sum(r[desired+'_token_count'] for r in rs))
        if rs:
            require(groups[kind]['numerator']==wins and groups[kind]['desired_token_correct']==independent[kind]['token_correct'], 'INDEPENDENT_REDUCTION')
    complete = terminal['status']=='COMPLETED'
    if complete:
        expected = read(c['observer_manifest']['path'])
        ids = [r['case_id'] for r in expected if r['kind']=='R']
        validate_rows(rows, expected, ids, 'W0')
        result = read(attempt/'result.json')
        require(sha(attempt/'result.json') == terminal['result_sha256'], 'RESULT_HASH')
        require(groups==result['summary'] and len(rows)==26000 and result['no_mutation'], 'RESULT_REDUCTION')
        require(manifest==result['raw'], 'RAW_HASH')
        from .gptj_server2_cohort_metrics import curves
        require(curves(rows)==read(attempt/'cohort-curves.json'), 'CURVE_REDUCTION')
    receipt = dict(status='COMPLETED' if complete else 'PARTIAL_OR_FAILED', rows=len(rows),
        independent=independent, summary=groups, terminal=terminal, raw=manifest,
        no_model_load=True, new_evaluation=False, no_checkpoint=True)
    write(attempt/'collection.json', receipt)
    text = '# GPT-J cold W0 평가\n\n상태: '+receipt['status']+'\n\n'
    text += '실제 pair rows: '+str(len(rows))+' / 26000. 신규 W0 관측; fit/edit/solve/H/CP=0.\n\n'
    text += '|종류|성공/분모|\n|---|---:|\n'
    for k, v in independent.items(): text += f"|{k}|{v['success_count']}/{v['count']}|\n"
    text += '\nPreference와 TF 정확도는 별개이며 N desired=true. 전체 raw와 token identity는 local-only.\n'
    with (attempt/'report-ko.md').open('x') as f: f.write(text)
    write(attempt/'collector-terminal.json', dict(status=receipt['status'], collection=member(attempt/'collection.json'), report=member(attempt/'report-ko.md')))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode', choices=('run','collect')); p.add_argument('--attempt', type=Path, required=True)
    args=p.parse_args(); (run if args.mode=='run' else collect)(args.attempt)
