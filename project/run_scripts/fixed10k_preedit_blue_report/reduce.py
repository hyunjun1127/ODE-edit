"""Independent raw-pair reducer; no torch, tokenizer, or evaluator import/call."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

TASK = 'ODEEDIT-S06-FIXED10K-PREEDIT-BLUE-LIFELONG-REPORT-INTEGRATION-SH2-V1'
DATA_SHA = '3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'
ORDER_ROOT = '5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729'
LOCK_SHA = '7ea4991018fb18b1cb4dc520acf3e6adb386ad39112440fa2f57af9e33bdf167'
SOURCE_HEAD = '12ad4cbe417d9935b1e7e55600214d14ba0d6360'
SOURCE_TREE = 'be4ee8bede7e4e962fd13064814bd2fbd500b56a'
MULT = {'RS': 1, 'PS': 2, 'NS': 10}
KINDS = {'RS': 'rewrite', 'PS': 'rephrase', 'NS': 'locality'}


def require(ok, evidence):
    if not ok:
        raise ValueError(evidence)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(path, rows):
    require(bool(rows), 'EMPTY_TABLE')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)


def member(path):
    p = Path(path)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def verify_members(root, members, allow_symlink=False):
    seen = set()
    for m in members:
        p = Path(root) / m['path']
        require(str(p) not in seen, f'DUPLICATE_MEMBER {p}')
        seen.add(str(p))
        require(p.is_file() and (allow_symlink or not p.is_symlink()), f'FILE_TYPE {p}')
        require(p.stat().st_size == int(m['bytes']), f'BYTES {p}')
        require(sha(p) == m['sha256'], f'SHA256 {p}')
    return dict(files=len(seen), bytes=sum(int(m['bytes']) for m in members), status='PASS')


def preferred(new, true, metric):
    require(metric in MULT and math.isfinite(new) and math.isfinite(true), 'NONFINITE_OR_METRIC')
    return true < new if metric == 'NS' else new < true


def pair(new, true, expected, metric):
    case, index, prompt, target_new, target_true = expected
    for r, target, side in ((new, target_new, 'new'), (true, target_true, 'true')):
        require((r['case_id'], r['prompt_index'], r['prompt'], r['target']) == (case, index, prompt, target), 'RAW_PAIR_DATASET_IDENTITY')
        require(r['kind'] == KINDS[metric] + '_target_' + side, 'RAW_KIND')
        require(math.isfinite(r['nll']) and r['nll'] >= 0, 'NONFINITE_OR_NEGATIVE_NLL')
        ids, pred, bits = r['target_token_ids'], r['token_predictions'], r['token_correct']
        require(len(ids) > 0 and len(ids) == len(pred) == len(bits), 'TOKEN_LENGTH')
        require(bits == [a == b for a, b in zip(ids, pred)], 'TOKEN_CORRECTNESS')
        require(r['all_tokens_correct'] == all(bits), 'TF_STRICT')
    return dict(case_id=case, prompt_index=index, identity=digest(list(expected)),
                new_nll=new['nll'], true_nll=true['nll'], margin=true['nll']-new['nll'],
                success=preferred(new['nll'], true['nll'], metric),
                new_strict=all(new['token_correct']), true_strict=all(true['token_correct']),
                new_token_correct=sum(new['token_correct']), new_token_count=len(new['token_correct']),
                true_token_correct=sum(true['token_correct']), true_token_count=len(true['token_correct']))


def quantile(a, q):
    a = sorted(a); t = (len(a)-1)*q; i = int(t); j = min(i+1, len(a)-1)
    return a[i] + (a[j]-a[i])*(t-i)


def stats(a):
    return dict(n=len(a), mean=statistics.fmean(a), median=statistics.median(a),
                q25=quantile(a, .25), q75=quantile(a, .75), p90=quantile(a, .9), max=max(a))


def summarize(rows, metric):
    require(bool(rows), 'EMPTY_ROWS')
    require(len({r['identity'] for r in rows}) == len(rows), 'DUPLICATE_PAIR')
    groups = defaultdict(list)
    for r in rows:
        require(r['success'] == preferred(r['new_nll'], r['true_nll'], metric), 'SUCCESS')
        require(r['margin'] == r['true_nll'] - r['new_nll'], 'MARGIN')
        groups[r['case_id']].append(r)
    require(all(len(g) == MULT[metric] for g in groups.values()), 'REQUEST_CLUSTER_COUNT')
    out = dict(metric=metric, numerator=sum(r['success'] for r in rows), denominator=len(rows),
               rate=sum(r['success'] for r in rows)/len(rows), request_denominator=len(groups),
               pair_strict_num=sum(all(r['success'] for r in g) for g in groups.values()),
               ties=sum(r['new_nll'] == r['true_nll'] for r in rows),
               bit_order_sha256=digest([(r['identity'], r['success']) for r in rows]))
    for side in ('new', 'true'):
        out.update({side+'_nll_prompt_'+k:v for k,v in stats([r[side+'_nll'] for r in rows]).items()})
        out.update({side+'_nll_request_'+k:v for k,v in stats([statistics.fmean(r[side+'_nll'] for r in g) for g in groups.values()]).items()})
        out.update({side+'_strict_num':sum(r[side+'_strict'] for r in rows), side+'_strict_den':len(rows),
                    side+'_token_correct':sum(r[side+'_token_correct'] for r in rows),
                    side+'_token_den':sum(r[side+'_token_count'] for r in rows)})
    for unit, values in [('prompt', [r['margin'] for r in rows]),
                         ('request', [statistics.fmean(r['margin'] for r in g) for g in groups.values()])]:
        out.update({'margin_'+unit+'_'+k:v for k,v in stats(values).items()})
    return out


def reduce_run(task_root, dataset, out):
    require(not out.exists(), f'CREATE_ONCE {out}')
    lock_path = task_root/'execution.lock.json'
    require(sha(lock_path) == LOCK_SHA, 'EXECUTION_LOCK_SHA')
    lock = read(lock_path)
    require((lock['source_head'],lock['source_tree']) == (SOURCE_HEAD,SOURCE_TREE), 'EXECUTION_SOURCE')
    require(sha(dataset) == DATA_SHA, 'DATASET_SHA')
    records = read(dataset)
    require(len(records) == 10000 and len({r['case_id'] for r in records}) == 10000, 'DATASET_DENOMINATOR')
    output = task_root/'output'; terminal = read(output/'terminal.json'); runtime = read(output/'runtime.json')
    require(terminal['status'] == 'TERMINAL_VALID' and terminal['requests'] == 10000, 'TERMINAL')
    require(runtime['slurm_job'] == '42673' and runtime['lock_sha256'] == LOCK_SHA, 'JOB_BINDING')
    require(runtime['sample_root'] == ORDER_ROOT and lock['sample_root'] == ORDER_ROOT, 'SAMPLE_ROOT')
    for k in ('compute_z','writer','key','solve','history_append','optimizer','backward','parameter_gradient','model_update','failure','nonfinite','duplicate','imputation'):
        require(terminal[k] == 0, 'NONZERO_ACTION_'+k)
    for k in ('selected_W_pointer_version_bytes_exact','all_parameter_pointer_version_requiresgrad_exact'):
        require(terminal[k] is True, 'W0_GUARD_'+k)
    print('Verifying sealed output members', flush=True)
    output_check = verify_members(output, terminal['manifest_members'])
    require({str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} == {m['path'] for m in terminal['manifest_members']} | {'terminal.json'}, 'OUTPUT_CLOSURE')
    print('Verifying source/environment/asset lock members', flush=True)
    lock_check = verify_members(Path('/'), lock['members'], allow_symlink=True)
    dataset_check = verify_members(Path('/'), runtime['dataset']['members'])
    ids = [r['case_id'] for r in records]
    require(digest(ids) == runtime['dataset']['verification']['case_order_sha256'], 'CASE_ORDER')
    expected_weights = {k:v['sha256'] for k,v in runtime['selected_W0']['weights'].items()}
    all_rows = {k:[] for k in MULT}; part_stats = []
    for part in range(1,101):
        chunk = records[(part-1)*100:part*100]
        saved = read(output/f'part-{part:03d}.json')
        require(saved['before_after_exact'] is True and saved['weight_state'] == expected_weights, 'PART_W0')
        require(saved['evaluator_controller_influence'] == 0 and saved['requests'] == 100, 'PART_ACTION')
        for metric, kind in KINDS.items():
            new = read(output/f'raw-prompt-pairs/part-{part:03d}-{kind}_target_new.json')
            true = read(output/f'raw-prompt-pairs/part-{part:03d}-{kind}_target_true.json')
            expected = []
            for r in chunk:
                rr = r['requested_rewrite']
                prompts = [rr['prompt'].format(rr['subject'])] if metric=='RS' else r['paraphrase_prompts' if metric=='PS' else 'neighborhood_prompts']
                require(len(prompts) == MULT[metric], 'DATASET_PROMPT_INVENTORY')
                expected.extend((r['case_id'],i,p,rr['target_new']['str'],rr['target_true']['str']) for i,p in enumerate(prompts))
            require(len(new) == len(true) == len(expected), 'RAW_INVENTORY')
            rows = [pair(n,t,e,metric) for n,t,e in zip(new,true,expected)]
            require(rows == saved['metrics'][metric]['rows'], 'RAW_VS_SAVED_ROWS')
            s = summarize(rows,metric)
            for k in ('numerator','denominator','rate','bit_order_sha256'):
                require(s[k] == saved['metrics'][metric][k], 'PART_REDUCER_'+k)
            part_stats.append(dict(part=part,**s)); all_rows[metric].extend(rows)
    full = read(output/'preedit-full10000.json')
    require(full['request_order'] == digest(ids) and full['weight_state'] == expected_weights, 'FULL_ORDER_W0')
    summaries = []
    for n in (1000,3000,10000):
        for metric in MULT:
            rows = all_rows[metric][:n*MULT[metric]]
            s = summarize(rows,metric);summaries.append(dict(prefix_requests=n,scope='PRE_EDIT_W0_FIXED_PREFIX' if n<10000 else 'PRE_EDIT_W0_FULL10000',**s))
            if n==10000:
                require(rows == full['metrics'][metric]['rows'], 'FULL_ROW_EQUALITY')
                for k in ('numerator','denominator','rate','bit_order_sha256'):
                    require(s[k] == full['metrics'][metric][k], 'FULL_REDUCER_'+k)
    compute = read(output/'compute/part-100-cumulative.json')
    require(compute['calls']==600 and compute['raw_rows']==260000 and compute['original_state_guard_pass'], 'COMPUTE_LEDGER')
    out.mkdir(parents=True)
    write_csv(out/'w0-distributions.csv', summaries)
    write_csv(out/'w0-part-distributions.csv', part_stats)
    # Scalar identities/measurements are local-only, never Git-added.
    save(out/'w0-scalar-rows.local.json',all_rows)
    save(out/'terminal-copy.json',terminal)
    save(out/'runtime-copy.json',runtime)
    save(out/'compute-copy.json',compute)
    save(out/'observer-copy.json',read(output/'observer-source-runtime.json'))
    rec = dict(instruction_id=TASK,status='PASS',source_head=SOURCE_HEAD,source_tree=SOURCE_TREE,
               execution_lock=member(lock_path),terminal=member(output/'terminal.json'),dataset=member(dataset),
               order_root=ORDER_ROOT,case_order=digest(ids),source_asset_environment_rehash=lock_check,
               output_member_rehash=output_check,dataset_members_rehash=dataset_check,unique_requests=10000,
               paired_prompts={'RS':10000,'PS':20000,'NS':100000},raw_target_rows=260000,
               independent_raw_vs_part_vs_full_exact=True,part_W0_guard_pass=100,imputation=0,
               NLL_recomputed_from_logits=False,NLL_definition='recorded teacher-forced mean target-token negative log probability, nats/token; independent pair reduction only',
               nonselected_parameter_bytes='NOT_CLAIMED; all parameter pointers/versions/requires_grad guarded by execution',
               new_model_GPU_evaluator_edit_Slurm_actions=0,scientific_promotion=False)
    save(out/'independent-validation.json',rec)
    print(json.dumps([s for s in summaries if s['prefix_requests']==10000],ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--task-root',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();reduce_run(a.task_root,a.dataset,a.out)
