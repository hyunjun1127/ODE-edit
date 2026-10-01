"""Small offline pretrained JLZ pilot with native-direction KL and real C0.

This is a bounded implementation pilot, not a calibrated performance experiment.
All edits live in this process; no pretrained files or checkpoints are written.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import prepare, native_loss
from .solver import solve


LAYERS = (4, 5, 6, 7, 8)
ROOT = Path(__file__).resolve().parents[3]
MODEL = Path('/mnt/raid5/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2')
DATA = ROOT / 'local/datasets/counterfact-fixed-10k-v1/counterfact.json'
CONTEXT = ROOT / 'local/reviews/ep-tw1-completed-2026-09-15/baselines/1/contexts.json'
STATS = Path('/mnt/raid5/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def detached(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: detached(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(detached(v) for v in value)
    if isinstance(value, list):
        return [detached(v) for v in value]
    return value


def serial(value):
    if isinstance(value, torch.Tensor):
        return serial(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


class PrefixStop(Exception):
    pass


def prefix_cache(model, tokens):
    cache = {}

    def hook(_module, args, kwargs):
        require(kwargs.get('past_key_values') is None and kwargs.get('past_key_value') is None,
                'KV_CACHE_FORBIDDEN')
        cache['args'], cache['kwargs'] = detached(args), detached(kwargs)
        raise PrefixStop()

    handle = model.model.layers[4].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            try:
                model.model(**tokens, use_cache=False)
            except PrefixStop:
                pass
    finally:
        handle.remove()
    require(bool(cache), 'PREFIX_NOT_CAPTURED')
    return cache


def replay(model, cache):
    args, kwargs = cache['args'], cache['kwargs']
    hidden = args[0] if args else kwargs['hidden_states']
    for layer in model.model.layers[4:]:
        local = dict(kwargs)
        if args:
            output = layer(hidden, *args[1:], **local)
        else:
            local['hidden_states'] = hidden
            output = layer(**local)
        hidden = output[0] if isinstance(output, (tuple, list)) else output
    return model.lm_head(model.model.norm(hidden))


@contextmanager
def functional_weights(model, effective):
    handles = []
    for l in LAYERS:
        def hook(module, args, _output, layer=l):
            return F.linear(args[0], effective[layer], module.bias)
        handles.append(model.model.layers[l].mlp.down_proj.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def chunks(spec, microbatch):
    count = spec['tokens']['input_ids'].shape[0]
    return [list(range(start, min(start + microbatch, count))) for start in range(0, count, microbatch)]


def token_subset(tokens, rows):
    return {name: value[rows] for name, value in tokens.items()}


@torch.no_grad()
def capture_keys(model, spec, contexts, microbatch):
    keys = {l: [] for l in LAYERS}
    tokens = spec['key_tokens']
    lookups = spec['key_lookup']
    for start in range(0, len(lookups), microbatch):
        rows = list(range(start, min(start + microbatch, len(lookups))))
        handles = []
        for l in LAYERS:
            def hook(_module, args, layer=l):
                local_rows = torch.arange(len(rows), device=args[0].device)
                cols = torch.tensor([lookups[r] for r in rows], device=args[0].device)
                keys[layer].append(args[0][local_rows, cols].detach().clone())
            handles.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook))
        try:
            model.model(**token_subset(tokens, rows), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
    n_context = sum(map(len, contexts))
    B = len(spec['specs'])
    # Match pinned compute_ks: average within each context group, then groups.
    answer = {}
    for l, parts in keys.items():
        raw = torch.cat(parts).reshape(B, n_context, -1)
        groups, start = [], 0
        for group in contexts:
            groups.append(raw[:, start:start + len(group)].mean(dim=1))
            start += len(group)
        answer[l] = torch.stack(groups).mean(dim=0).T.contiguous()
    return answer


@torch.no_grad()
def capture_entry(model, spec, microbatch):
    B = len(spec['specs'])
    anchors = {l: torch.zeros(model.config.hidden_size, B, device='cuda') for l in LAYERS}
    teacher = torch.empty(B, model.config.vocab_size, device='cuda')
    nll = torch.zeros(B, device='cuda')
    for rows in chunks(spec, microbatch):
        handles = []
        for l in LAYERS:
            def hook(_module, _args, output, layer=l):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                for request, item in enumerate(spec['specs']):
                    if item['offset'] in rows:
                        anchors[layer][:, request] = hidden[rows.index(item['offset']), item['lookup'][0]]
            handles.append(model.model.layers[l].register_forward_hook(hook))
        try:
            logits = model(**token_subset(spec['tokens'], rows), use_cache=False).logits
        finally:
            for handle in handles:
                handle.remove()
        for request, row in enumerate(spec['kl_rows']):
            if row in rows:
                teacher[request] = logits[rows.index(row), spec['kl_cols'][request]].log_softmax(-1)
        # Entry NLL is independent of teacher. Use no KL contribution here.
        _, partial_nll, _ = native_loss(logits, spec, None, torch.ones(B, dtype=torch.bool, device='cuda'), rows=rows)
        nll += partial_nll
    return anchors, teacher, nll


def geometry(keys, history, receipt, deadline):
    adj = {}
    for l in LAYERS:
        require(time.monotonic() < deadline, 'BUDGET_STOP_GEOMETRY')
        start = time.monotonic()
        path = STATS / f'model.layers.{l}.mlp.down_proj_float32_mom2_100000.npz'
        with np.load(path, allow_pickle=False) as data:
            count = int(data['mom2.count'])
            raw = torch.from_numpy(data['mom2.mom2'].copy())
        # SecondMoment.moment(): FP32 division BEFORE FP64 solve conversion.
        covariance = (raw / count).to(device='cuda', dtype=torch.float64)
        del raw
        key = keys[l].double()
        system = 15000.0 * covariance + history[l].to(device='cuda', dtype=torch.float64)
        del covariance
        system.add_(key @ key.T)
        adj[l] = torch.linalg.solve(system, key)
        residual = float((system @ adj[l] - key).norm() / key.norm().clamp_min(1.0))
        require(torch.isfinite(adj[l]).all().item() and residual < 1e-7, f'ADJ_SOLVE_FAILED_L{l}')
        receipt.append(dict(layer=l, covariance_file=str(path), count=count,
                            normalization='float32 raw/count then float64', solve_residual=residual,
                            self_realization_diagonal=(adj[l].T @ key).diag().cpu().tolist(),
                            seconds=time.monotonic() - start))
        print(json.dumps(dict(event='geometry', **receipt[-1])), flush=True)
        del system, key
        torch.cuda.empty_cache()
    return adj


class Oracle:
    def __init__(self, model, spec, teacher, keys, adj, active, microbatch):
        self.model, self.spec, self.teacher, self.adj, self.active = model, spec, teacher, adj, active
        self.B = len(spec['specs'])
        self.rows = chunks(spec, microbatch)
        self.prefixes = [prefix_cache(model, token_subset(spec['tokens'], rows)) for rows in self.rows]
        self.entry_w = {l: model.model.layers[l].mlp.down_proj.weight.detach().clone() for l in LAYERS}
        self.calls, self.forward_calls, self.backward_calls = 0, 0, 0
        self.initial_gradient = None

    def effective(self, x):
        return {l: self.entry_w[l] + (x[i*self.B:(i+1)*self.B].T.double() @ self.adj[l].T).float()
                for i, l in enumerate(LAYERS)}

    def __call__(self, x, route='suffix'):
        self.calls += 1
        variable = x.detach().clone().requires_grad_(True)
        total, gradient = 0.0, torch.zeros_like(variable)
        nll, kl = torch.zeros(self.B, device=x.device), torch.zeros(self.B, device=x.device)
        for chunk_index, (rows, cache) in enumerate(zip(self.rows, self.prefixes)):
            weights = self.effective(variable)
            with functional_weights(self.model, weights):
                logits = replay(self.model, cache) if route == 'suffix' else self.model(
                    **token_subset(self.spec['tokens'], rows), use_cache=False).logits
                loss, per_nll, per_kl = native_loss(logits, self.spec, self.teacher, self.active, rows=rows)
            grad = torch.autograd.grad(loss, variable)[0]
            gradient.add_(grad.detach())
            total += float(loss.detach())
            nll.add_(per_nll.detach())
            kl.add_(per_kl.detach())
            self.forward_calls += 1
            self.backward_calls += 1
            del logits, loss, grad, per_nll, per_kl
            if chunk_index != len(self.rows)-1:
                del weights
        payload = dict(nll=nll.cpu().tolist(), kl=kl.cpu().tolist(),
                       weights={l: weight.detach() for l, weight in weights.items()})
        if self.initial_gradient is None and bool((x == 0).all()):
            self.initial_gradient = gradient.detach().clone()
        if self.calls % 10 == 0:
            print(json.dumps(dict(event='oracle_progress', calls=self.calls, smooth=total,
                                  gradient_norm=float(gradient.norm()), nll=payload['nll'], kl=payload['kl'])),flush=True)
        return total, gradient, payload

    @torch.no_grad()
    def committed_losses(self):
        nll, kl = torch.zeros(self.B, device='cuda'), torch.zeros(self.B, device='cuda')
        for rows in self.rows:
            logits = self.model(**token_subset(self.spec['tokens'], rows), use_cache=False).logits
            _, nr, kr = native_loss(logits, self.spec, self.teacher, self.active, rows=rows)
            nll += nr
            kl += kr
        return nll, kl


def run_batch(model, tokenizer, requests, contexts, history, args, deadline, batch_number):
    start = time.monotonic()
    spec = prepare(tokenizer, requests, contexts, device='cuda')
    B = len(requests)
    anchors, teacher, nll_entry = capture_entry(model, spec, args.microbatch)
    active = nll_entry >= 0.05
    keys = capture_keys(model, spec, contexts, args.microbatch)
    row = dict(batch=batch_number, case_ids=[r['case_id'] for r in requests],
               input_identity=spec['identity'],
               nll_entry=nll_entry.cpu().tolist(), active=active.cpu().tolist(),
               entry_history_norm={str(l): float(history[l].norm()) for l in LAYERS}, geometry=[])
    adj = geometry(keys, history, row['geometry'], deadline)
    oracle = Oracle(model, spec, teacher, keys, adj, active, args.microbatch)
    x0 = torch.zeros(len(LAYERS)*B, model.config.hidden_size, device='cuda')
    norms = torch.cat([anchors[l].norm(dim=0) for l in LAYERS])
    c, rho, mask = 0.5 / norms.square(), 0.75 * norms, active.repeat(len(LAYERS))
    # Nonzero native loss/prefix gradient check; the oracle teacher remains fixed.
    generator = torch.Generator(device='cuda').manual_seed(20261001 + batch_number)
    probe = torch.randn(x0.shape, generator=generator, device='cuda')
    probe *= (0.02 * rho / probe.norm(dim=1)).unsqueeze(1)
    probe[~mask] = 0
    full, gf, pf = oracle(probe, route='full')
    del pf
    suffix, gs, ps = oracle(probe)
    del ps
    parity = dict(nonzero_loss_abs=abs(full-suffix),
                  nonzero_gradient_relative=float((gf-gs).norm()/gf.norm().clamp_min(1.0)))
    require(parity['nonzero_loss_abs'] <= 1e-3 and parity['nonzero_gradient_relative'] <= 1e-3,
            'NONZERO_FULL_SUFFIX_PARITY_FAILED')
    row['parity'] = parity
    del gf, gs, probe
    print(json.dumps(dict(event='entry_ready', batch=batch_number, nll=row['nll_entry'], parity=parity)), flush=True)
    if not active.any().item():
        value, gradient, payload = oracle(x0)
        result = dict(x=x0, gradient=gradient, status='POLICY_ZERO_STEP', calls=1,
                      value=value, smooth=value, decay=0.0, normalized_residual=0.0, final_payload=payload,
                      final_recomputed=True,commit_eligible=True)
    else:
        result = solve(oracle, x0, c, rho, mask, tol=args.tol, cap=args.cap,
                       max_seconds=max(0.0, min(args.batch_seconds, deadline-time.monotonic())))
    payload = result.pop('final_payload', None)
    x = result.pop('x')
    gradient = result.pop('gradient')
    row['solver'] = serial(result)
    row['oracle_total_calls_including_probes'] = oracle.calls
    row['oracle_prompt_chunk_forwards'] = oracle.forward_calls
    row['oracle_prompt_chunk_backwards'] = oracle.backward_calls
    row['block_norm_over_radius'] = (x.norm(dim=1)/rho).cpu().tolist()
    row['nonzero_blocks'] = int((x.norm(dim=1)>0).sum())
    row['initial_gradient_price_ratio'] = (oracle.initial_gradient.norm(dim=1)/c).cpu().tolist() if oracle.initial_gradient is not None else None
    row['commit'] = False
    row['fit'] = {k:v for k,v in (payload or {}).items() if k != 'weights'}
    row['finite'] = bool(torch.isfinite(x).all() and torch.isfinite(gradient).all())
    require(row['finite'] and bool((x.norm(dim=1) <= rho*(1+1e-6)).all()), 'RETURN_FEASIBILITY_OR_FINITE_FAILED')
    if (result['status'] in ('CONVERGED', 'POLICY_ZERO_STEP') and payload is not None
            and result.get('commit_eligible') and result.get('final_recomputed')):
        with torch.no_grad():
            for l in LAYERS:
                model.model.layers[l].mlp.down_proj.weight.copy_(payload['weights'][l])
        materialization = all(torch.equal(model.model.layers[l].mlp.down_proj.weight, payload['weights'][l]) for l in LAYERS)
        nr, kr = oracle.committed_losses()
        fit_nll = torch.tensor(payload['nll'], device='cuda')
        fit_kl = torch.tensor(payload['kl'], device='cuda')
        commit_error = float(torch.maximum((nr-fit_nll).abs().max(), (kr-fit_kl).abs().max()))
        post_keys = capture_keys(model, spec, contexts, args.microbatch)
        low_exact = torch.equal(keys[4], post_keys[4])
        if not (materialization and low_exact and commit_error <= 1e-3):
            with torch.no_grad():
                for l in LAYERS:
                    model.model.layers[l].mlp.down_proj.weight.copy_(oracle.entry_w[l])
            raise RuntimeError(f'COMMIT_GATE_FAILED error={commit_error} low_exact={low_exact}; W restored')
        for l in LAYERS:
            history[l].add_((post_keys[l] @ post_keys[l].T).cpu())
        row.update(commit=True, committed_nll=nr.cpu().tolist(), committed_kl=kr.cpu().tolist(),
                   commit_loss_max_abs=commit_error, materialization_bitwise=materialization,
                   lowest_layer_key_bitwise=low_exact,
                   post_history_norm={str(l):float(history[l].norm()) for l in LAYERS})
    row['layer_allocation'] = []
    for i,l in enumerate(LAYERS):
        R = x[i*B:(i+1)*B].T
        D = R.double() @ (adj[l].T @ keys[l].double())
        row['layer_allocation'].append(dict(layer=l, target_norm=float(R.norm()), realized_norm=float(D.norm()),
                                            active_blocks=int((R.norm(dim=0)>0).sum())))
    row['seconds'] = time.monotonic()-start
    del payload, oracle, x, gradient, adj, keys
    torch.cuda.empty_cache()
    return row


@torch.no_grad()
def observe(model, tokenizer, requests, contexts, microbatch):
    spec = prepare(tokenizer, requests, contexts, device='cuda')
    nll = torch.zeros(len(requests), device='cuda')
    strict = [None]*len(requests)
    for rows in chunks(spec, microbatch):
        logits = model(**token_subset(spec['tokens'], rows), use_cache=False).logits
        _, part, _ = native_loss(logits, spec, None, torch.ones(len(requests),dtype=torch.bool,device='cuda'), rows=rows)
        nll += part
        for r, item in enumerate(spec['specs']):
            if item['offset'] in rows:
                labels = spec['targets'][item['offset']]
                target_mask = labels != -100
                predicted = logits[rows.index(item['offset']),target_mask].argmax(-1)
                strict[r] = bool(torch.equal(predicted, labels[target_mask]))
    return dict(case_ids=[r['case_id'] for r in requests], native_context_mean_nll=nll.cpu().tolist(),
                canonical_teacher_forced_strict=strict, denominator=len(requests))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cap',type=int,default=120)
    parser.add_argument('--tol',type=float,default=1e-4)
    parser.add_argument('--microbatch',type=int,default=2)
    parser.add_argument('--batch-seconds',type=float,default=1200)
    parser.add_argument('--total-seconds',type=float,default=3300)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();deadline=start+args.total_seconds
    receipt=dict(status='STARTED',method='JLZ-native-v2',server3_touched=False,batches=[],
                 config=vars(args)|{'output_dir':str(args.output_dir)},
                 native_kl='KL(current||entry)',decay=.5,clamp=.75,kl_factor=.0625,lam=15000,
                 context_key_reduction='mean(context-group means)',context_loss_reduction='mean(all 6 contexts)',
                 calibrated_solver=False,performance_comparison=False)
    output=args.output_dir/'receipt.json'
    try:
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'ONE_ALLOCATED_GPU_REQUIRED')
        torch.set_num_threads(8);torch.manual_seed(20261001)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        require(file_sha(DATA)=='3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1','DATA_HASH_MISMATCH')
        require(file_sha(CONTEXT)=='33cec0eef9ec130f26c2f0e17f7c8be39e93c717ebb47eaa5e9f1c88b263524e','CONTEXT_HASH_MISMATCH')
        records=json.loads(DATA.read_text())[:4]
        requests=[r['requested_rewrite']|{'case_id':r['case_id']} for r in records]
        contexts=json.loads(CONTEXT.read_text())
        receipt['runtime']=dict(torch=torch.__version__,transformers=transformers.__version__,
                                gpu=torch.cuda.get_device_name(),job_id=os.getenv('SLURM_JOB_ID'),
                                node=os.getenv('SLURMD_NODENAME'),dtype='float32',TF32=False)
        receipt['source_sha256']={p.name:file_sha(p) for p in Path(__file__).parent.glob('*.py')}
        receipt['model']=str(MODEL);receipt['case_ids']=[r['case_id'] for r in requests]
        atomic_json(output,receipt)
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        tokenizer.padding_side='right';tokenizer.pad_token=tokenizer.eos_token
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32,
                                                  device_map={'':'cuda:0'},attn_implementation='eager').eval()
        model.config.use_cache=False
        for p in model.parameters():p.requires_grad_(False)
        receipt['load_seconds']=time.monotonic()-start
        print(json.dumps(dict(event='model_loaded',seconds=receipt['load_seconds'])),flush=True)
        history={l:torch.zeros(model.config.intermediate_size,model.config.intermediate_size) for l in LAYERS}
        receipt['W0_observation']=observe(model,tokenizer,requests,contexts,args.microbatch)
        for batch_no in range(2):
            require(time.monotonic()<deadline,'BUDGET_STOP_BEFORE_BATCH')
            row=run_batch(model,tokenizer,requests[2*batch_no:2*batch_no+2],contexts,history,args,deadline,batch_no)
            receipt['batches'].append(row)
            receipt['peak_gpu_GiB']=torch.cuda.max_memory_allocated()/1024**3
            if row['commit']:
                row['all_seen_observation']=observe(model,tokenizer,requests[:2*batch_no+2],contexts,args.microbatch)
            atomic_json(output,receipt)
            print(json.dumps(dict(event='batch_done',batch=batch_no,status=row['solver']['status'],
                                  calls=row['solver']['calls'],commit=row['commit'],seconds=row['seconds'])),flush=True)
            if not row['commit']:
                receipt['status']='STOPPED_'+row['solver']['status'];break
        else:
            receipt['status']='PILOT_COMPLETED'
        receipt['not_tested']=['pinned Transformers 4.44.2 bitwise parity','MEMIT-H comparative efficacy',
                               'long-run retention','full precision-floor calibration','full R/P/N evaluation']
    except Exception as exc:
        receipt['status']='TECHNICAL_FAILURE'
        receipt['error']=str(exc);receipt['traceback']=traceback.format_exc()
        print(receipt['traceback'],flush=True)
    finally:
        receipt['total_seconds']=time.monotonic()-start
        atomic_json(output,receipt)
    return 0 if receipt['status']=='PILOT_COMPLETED' else 2


if __name__=='__main__':
    raise SystemExit(main())
