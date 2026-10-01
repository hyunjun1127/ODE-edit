"""CPU-only evidence for the efficiency design, not a timing benchmark."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import torch
    from transformers import AutoTokenizer
    from project.run_scripts.jlz_pilot.prompts import prepare
    torch.set_num_threads(1)
    base = Path('/mnt/raid5/janghj/ODE-edit')
    data_path = base/'local/datasets/counterfact-fixed-10k-v1/counterfact.json'
    contexts_path = base/'local/reviews/ep-tw1-completed-2026-09-15/baselines/1/contexts.json'
    log_path = base/'local/jlz-sequential/20261001-v1/attempt-r1/gpu-56684.out'
    model = '/mnt/raid5/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tok.padding_side = 'right'
    tok.pad_token = tok.eos_token
    records = json.loads(data_path.read_text())[:1000]
    contexts = json.loads(contexts_path.read_text())
    summaries = []
    for batch in range(10):
        requests = [r['requested_rewrite'] | {'case_id': r['case_id']}
                    for r in records[batch*100:(batch+1)*100]]
        spec = prepare(tok, requests, contexts, device='cpu')
        lengths = spec['tokens']['attention_mask'].sum(1).tolist()
        maximum = max(lengths)
        row = {'batch': batch+1, 'rows': len(lengths), 'max_length': maximum,
               'valid_tokens': sum(lengths), 'spec_identity': spec['identity'],
               'global_padded_tokens': len(lengths)*maximum,
               'global_attention_positions': len(lengths)*maximum*maximum, 'layouts': {}}
        for mb in (2,4,8):
            for order_name, order in [('original', list(range(len(lengths)))),
                                      ('stable_length', sorted(range(len(lengths)), key=lambda r: (lengths[r],r)))]:
                chunk_lengths = []
                tokens = attention = 0
                for start in range(0,len(order),mb):
                    selected = order[start:start+mb]
                    width = max(lengths[r] for r in selected)
                    tokens += len(selected)*width
                    attention += len(selected)*width*width
                    chunk_lengths.append(width)
                    # Crop preserves every nonpadding/target/lookup position.
                    for r in selected:
                        assert torch.equal(spec['tokens']['attention_mask'][r,width:],
                                           torch.zeros_like(spec['tokens']['attention_mask'][r,width:]))
                        assert bool((spec['targets'][r,width:] == -100).all())
                        assert spec['lookup'][r] < width
                row['layouts'][f'{order_name}_mb{mb}'] = {
                    'padded_tokens': tokens, 'attention_positions': attention,
                    'linear_work_remaining_fraction': tokens/row['global_padded_tokens'],
                    'attention_work_remaining_fraction': attention/row['global_attention_positions'],
                    'chunks':len(chunk_lengths)}
        summaries.append(row)
    selected_log = []
    with log_path.open() as stream:
        for line in stream:
            if "'event': 'oracle'" not in line:
                continue
            record = ast.literal_eval(line.strip())
            if record['calls'] in (10,20):
                selected_log.append(record)
            if len(selected_log) == 2:
                break
    assert [r['calls'] for r in selected_log] == [10,20]
    seconds = (selected_log[1]['seconds']-selected_log[0]['seconds'])/10
    aggregate = {}
    for layout in summaries[0]['layouts']:
        a = sum(row['layouts'][layout]['padded_tokens'] for row in summaries)
        b = sum(row['layouts'][layout]['attention_positions'] for row in summaries)
        aggregate[layout] = {'padded_tokens':a, 'attention_positions':b,
            'linear_work_remaining_fraction':a/sum(row['global_padded_tokens'] for row in summaries),
            'attention_work_remaining_fraction':b/sum(row['global_attention_positions'] for row in summaries)}
    paths = ['project/run_scripts/jlz_sequential/oracle.py',
             'project/run_scripts/jlz_sequential/run.py',
             'project/run_scripts/jlz_sequential/policy.py',
             'project/run_scripts/jlz_sequential/observation.py',
             'project/run_scripts/jlz_pilot/run.py',
             'project/run_scripts/jlz_pilot/prompts.py',
             'project/run_scripts/jlz_pilot/solver.py',
             'project/run_scripts/alphaedit_strength_neutral_barrier/evaluator.py',
             'project/run_scripts/single_layer_mechanism_first/z_hook.py',
             'project/run_scripts/memit_hj/writer.py']
    result = {'status':'CPU_LAYOUT_AND_EXISTING_TIMING_EVIDENCE_ONLY',
              'gpu_benchmark_performed':False, 'model_loaded':False,
              'torch':str(torch.__version__), 'source_root':str(args.source),
              'source_sha256':{p:sha(args.source/p) for p in paths},
              'data_sha256':sha(data_path), 'contexts_sha256':sha(contexts_path),
              'log_path':str(log_path), 'log_scope':'first B1 oracle10/20 records only; mutable log not hashed',
              'selected_log':selected_log, 'selected_log_sha256':hashlib.sha256(json.dumps(selected_log,sort_keys=True).encode()).hexdigest(),
              'seconds_per_oracle':seconds, 'optimization1200_hours':seconds/3,
              'fixed_timing_assumption':'all ten batches run at B1 calls10-20 throughput and use120 calls each',
              'padding_batches':summaries, 'padding_aggregate':aggregate,
              'caveat':'token and attention-size arithmetic, not measured FLOPs or runtime; stable_length is a proposal changing reduction order'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ['status','seconds_per_oracle','optimization1200_hours','padding_aggregate']},ensure_ascii=False))
    print(json.dumps({'B1':summaries[0]},ensure_ascii=False))


if __name__ == '__main__':
    main()
