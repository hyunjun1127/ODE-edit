"""Single pure controller/evaluator/collector horizon contract; no model import."""
from .common import require, digest

BATCHES = tuple(range(1, 21))
MILESTONES = (5, 10, 20)
B = 100

def selection(records, number):
    require(number in BATCHES and len(records) >= 2000, 'NO_B21_OR_SHORT_STREAM')
    current = records[(number - 1) * B:number * B]
    seen = records[:number * B]
    selected = seen if number in MILESTONES else current
    return current, seen, selected

def denominators(number):
    require(number in BATCHES, 'NO_B21')
    n = number * B if number in MILESTONES else B
    return dict(R=n, P=2*n, N=10*n)

def validate_config(c):
    e = c['experiment']
    require(e['main']['batches_per_arm'] == 20 and e['main']['batch_size'] == 100
            and e['main']['edits_per_arm'] == 2000 and e['main']['no_batch21'], 'EXACT_2K_SCOPE')
    require(c['evaluation_schedule']['all_seen_at'] == list(MILESTONES), 'MILESTONE_SCOPE')
    require(c['settings']['exact_probe'] == 'REUSE_HISTORICAL_NO_NEW_PROBE'
            and not c['settings']['save_checkpoints'] and c['baseline_new_runs'] == 0, 'NO_EXACT_BASELINE_CP')

def verify_commit(r, number, ids, source, config_sha):
    require(r['source'] == source and r['config'] == config_sha, 'COMMIT_SOURCE_CONFIG')
    require(r['batch'] == number and r['current_ids'] == ids and r['actual_B'] == B, 'COMMIT_CASE_ORDER')
    require(r['candidate_count'] == 25 and r['backward_count'] == r['Adam_updates'] == 24, 'COMMIT_BUDGET')
    require(r['history_appends'] == 5 and r['history']['accepted_weight_copy_exact'], 'WHOLE_HISTORY_COMMIT')
    require(r['exact_commit_sha'] == r['after']['W'], 'COMMIT_EVALUATED_WEIGHT')
    require(not r['Q2'] and not r['replay'] and not r['terminal_gradient_measured']
            and r['no_checkpoint'], 'RIDGE_ONLY_SCOPE')
