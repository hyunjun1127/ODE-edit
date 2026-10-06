"""Task-local bounded IO; readonly native observer scoring/reductions unchanged."""
import time
from pathlib import Path
from project.run_scripts.jlz_realization.observe import scores,reduce_rows,active_flags
from . import digest,require,state
from .storage import write,guard,CHUNK_BYTES,ERROR_RESERVE_BYTES

def observe(adapter, bench, all_records, selected_records, history, endpoint, out,
            microbatch=2, current_ids=None):
    """Write scalar/hash-only rows and verify W/H/guard/hooks. Replay is absent.
    Completed chunks survive a later evaluation failure for partial reporting.
    """
    out = Path(out)
    guard(out,len(selected_records)//50*CHUNK_BYTES+ERROR_RESERVE_BYTES)
    before = state(adapter, history)
    before_guard, hooks = adapter.guard(), adapter.hook_signature()
    started, rows = time.monotonic(), []
    flags = active_flags(all_records)
    try:
        for start in range(0, len(selected_records), 50):
            records = selected_records[start:start + 50]
            specs, pairs = [], []
            for record in records:
                rewrite = record['requested_rewrite']
                for kind, prompts in bench.panels(record).items():
                    require(kind in {'R', 'P', 'N'}, 'UNKNOWN_OBSERVER_KIND')
                    for index, prompt in enumerate(prompts):
                        identity = digest([record['case_id'], kind, index, prompt,
                                           rewrite['target_new']['str'], rewrite['target_true']['str']])
                        specs.append(dict(case_id=record['case_id'], kind=kind, prompt_index=index,
                                          identity=identity, endpoint=endpoint,
                                          active_at_endpoint=flags[record['case_id']]))
                        pairs.extend([(prompt, rewrite['target_new']['str']),
                                      (prompt, rewrite['target_true']['str'])])
            values = scores(adapter, bench, pairs, microbatch)
            require(len(values) == 2 * len(specs), 'OBSERVER_SCORE_COUNT')
            for i, row in enumerate(specs):
                for label, value in zip(('new', 'true'), values[2 * i:2 * i + 2]):
                    row.update({label + '_' + k: v for k, v in value.items()})
                row['margin_true_minus_new'] = row['true_nll'] - row['new_nll']
                row['margin_new_minus_true'] = row['new_nll'] - row['true_nll']
            reduce_rows(specs)
            rows.extend(specs)
            write(out / f'chunk-{start:04d}.json', dict(schema='jlz-observer-rows-v1',
                  state=before, rows=specs, optimizer_feedback=False))
            print(dict(event='observer', endpoint=endpoint, requests_done=start + len(records)), flush=True)
    finally:
        require(state(adapter, history) == before and adapter.guard() == before_guard
                and adapter.hook_signature() == hooks, 'OBSERVER_MUTATION')
    current = set([r['case_id'] for r in selected_records] if current_ids is None else current_ids)
    result = dict(schema='jlz-observer-summary-v1', endpoint=endpoint, state=before,
                  requests=len(selected_records), summary=reduce_rows(rows),
                  current=reduce_rows([r for r in rows if r['case_id'] in current]),
                  row_count=len(rows), row_order=digest([r['identity'] for r in rows]),
                  seconds=time.monotonic() - started, no_mutation=True,
                  replay=False, optimizer_feedback=False)
    write(out / 'summary.json', result)
    return result
