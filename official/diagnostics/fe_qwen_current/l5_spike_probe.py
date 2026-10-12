"""Per-case L5 spike probe for one FE batch, record-only (no write, no history, no checkpoint, no W&B).

Restores the selected L4-L8 weights of a saved latest checkpoint (the state before batch B), recomputes the batch's
targets exactly as the run did (same adapter, prefix rule, native compute_z and firstforward supplier), and records
per case: L4 init/delta/target and fit probabilities, the firstforward targets z_l and the unedited outputs h_l at
L4-L8 (bare prompt, subject-last lookup), the residual z_l - h_l, its energy on the model's massive-activation
channels (measured at W0 on a plain sentence, position 0) and its top channels. The same requests are then recomputed
on the W0 weights. The batch Frobenius norm of the L5 targets is compared with the run's writer record.

python -m official.diagnostics.fe_qwen_current.l5_spike_probe BINDINGS CHECKPOINT BATCH NEW_OUTPUT [--writer RUN/writers/BXX.json]
"""
import argparse
import copy
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from official.diagnostics.fe_qwen_current import author_sink_eot_fp32 as sink
from official.diagnostics.fe_qwen_current import basic_eot_benchmark as common
from official.diagnostics.fe_qwen_current import failure_eot_prefix as parent
from official.diagnostics.fe_qwen_current import failure_eot_probe as eot
from official.diagnostics.fe_qwen_current.benchmark import native_records
from official.experiments.prepare import write_new
from official.runners.server1.common import read, verify, require


def massive_channels(model, tok, layers, factor=50.0):
    ids = tok('The capital of France is Paris.', return_tensors='pt').to(next(model.parameters()).device)
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states
    out = {}
    for l in layers:
        h = hs[l + 1][0, 0].float().abs()
        med = float(h.median())
        out[l] = sorted(int(i) for i in torch.nonzero(h > factor * med).flatten().tolist())
    return out


def bare_outputs(model, tok, adapter, cfg, request):
    prompt = adapter.prompt(request['prompt'], request['subject'])
    lookup = adapter.lookup(request['prompt'], request['subject'], tok, cfg.llms.fact_token, verbose=False)
    ids = tok(prompt.format(request['subject']), return_tensors='pt').to(next(model.parameters()).device)
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states
    return {l: hs[l + 1][0, lookup].float() for l in cfg.llms.layers}, lookup, ids['input_ids'][0].tolist()


def analyze(model, tok, adapter, fit, supply, z, cfg, requests, contexts, massive):
    fits = []
    targets = supply(model, tok, requests, cfg, contexts, compute_z=parent.fit_receipt(fit, fits),
                     find_fact_lookup_idx=adapter.lookup, TraceDict=z.nethook.TraceDict)
    require(len(fits) == len(requests), 'ALL_FITS')
    rows = []
    for i, r in enumerate(requests):
        h, lookup, ids = bare_outputs(model, tok, adapter, cfg, r)
        layers = {}
        for l in cfg.llms.layers:
            zl = targets[l][:, i].float();res = zl - h[l];energy = float(res.square().sum())
            m = massive[l];top = res.abs().topk(5).indices.tolist()
            layers[str(l)] = dict(z=float(zl.norm()), h=float(h[l].norm()), residual=float(res.norm()),
                massive_energy_fraction=float(res[m].square().sum()) / energy if m and energy > 0 else 0.0,
                z_massive_norm=float(zl[m].norm()) if m else 0.0, h_massive_norm=float(h[l][m].norm()) if m else 0.0,
                top_channels=top, top_values=[float(res[c]) for c in top])
        f = fits[i]
        rows.append(dict(case_id=r['case_id'], subject=r['subject'], prompt=r['prompt'], target=r['target_new'],
            lookup=lookup, input_ids=ids, native_triplet=f['native_triplet'], probabilities=f['probabilities'],
            mean_probability=sum(f['probabilities']) / len(f['probabilities']), layers=layers))
    batch = {str(l): float(targets[l].float().norm()) for l in cfg.llms.layers}
    return rows, batch


def run(bindings, checkpoint, batch, output, writer=None, policy='sink_region'):
    b = read(bindings);out = Path(output);require(not out.exists(), 'NEW_OUTPUT')
    import transformers
    cfg = OmegaConf.create(read(verify(b['native_config'])))
    rows = common.main_stream(b, []);requests = copy.deepcopy(native_records(rows)[(batch - 1) * 100:batch * 100])
    for local, r in enumerate(requests):
        r['target_new'] = ' ' + r['target_new'];r['sample_idx'] = local
    snap = b['model']['snapshot'];tok = transformers.AutoTokenizer.from_pretrained(snap, trust_remote_code=True, local_files_only=True)
    torch.backends.cuda.matmul.allow_tf32 = b['runtime']['allow_tf32'];torch.backends.cudnn.allow_tf32 = b['runtime']['cudnn_allow_tf32']
    model = transformers.AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.float32, trust_remote_code=True,
                                                              local_files_only=True).to('cuda:0')
    sys.path.insert(0, b['native']['path'])
    native = importlib.import_module('algs.memit.memit_main');z = importlib.import_module('algs.memit.compute_z')
    s = importlib.import_module('algs.memit.native_sequential_z')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    require(int(payload['batch']) == batch - 1, 'CHECKPOINT_IS_STATE_BEFORE_BATCH')
    contexts = payload['contexts'];native.CONTEXT_TEMPLATES_CACHE = contexts
    params = dict(model.named_parameters());names = sorted(payload['weights'])
    require(set(names) == {cfg.llms.rewrite_module_tmp.format(l) + '.weight' for l in cfg.llms.layers}, 'SELECTED_WEIGHTS')
    w0 = {n: params[n].detach().cpu().clone() for n in names}
    massive = massive_channels(model, tok, cfg.llms.layers)
    result = dict(status='L5_SPIKE_PROBE_RECORD_ONLY', batch=batch, checkpoint=str(checkpoint), checkpoint_batch=int(payload['batch']),
                  policy=policy, massive_channels={str(k): v for k, v in massive.items()}, writes=0, case_ids=[r['case_id'] for r in requests])
    with patch.object(eot, 'repair_prompt', sink.prefix_rule(policy)):
        adapter = eot.InputAdapter(tok, z.find_fact_lookup_idx);fit = adapter.fit(z.compute_z)
        supply = adapter.supplier(s.current_batch_targets)
        with torch.no_grad():
            for n in names:params[n].copy_(payload['weights'][n].to(params[n].device))
        result['state_before_batch'], result['batch_target_norms'] = analyze(model, tok, adapter, fit, supply, z, cfg, requests, contexts, massive)
        if writer:
            rec = {str(x['layer']): x['target_norm'] for x in read(writer)['layers']}
            result['run_writer_target_norms'] = rec
            result['reproduced_relative_difference'] = {l: abs(result['batch_target_norms'][l] - rec[l]) / rec[l] for l in rec}
        with torch.no_grad():
            for n in names:params[n].copy_(w0[n].to(params[n].device))
        result['W0'], result['W0_batch_target_norms'] = analyze(model, tok, adapter, fit, supply, z, cfg, requests, contexts, massive)
    key = lambda r: -r['layers']['5']['residual']
    result['top_L5_residual_cases'] = [dict(case_id=r['case_id'], subject=r['subject'], prompt=r['prompt'],
        L5_residual=r['layers']['5']['residual'], L5_massive_fraction=r['layers']['5']['massive_energy_fraction'],
        L4_delta=r['native_triplet']['delta'], mean_probability=r['mean_probability'],
        W0_L5_residual=next(w['layers']['5']['residual'] for w in result['W0'] if w['case_id'] == r['case_id']))
        for r in sorted(result['state_before_batch'], key=key)[:10]]
    out.mkdir(parents=True);write_new(out / 'l5-spike-probe.json', result)
    return dict(batch_target_norms=result['batch_target_norms'], W0_batch_target_norms=result['W0_batch_target_norms'],
                top=result['top_L5_residual_cases'][:5], reproduced=result.get('reproduced_relative_difference'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bindings');p.add_argument('checkpoint');p.add_argument('batch', type=int);p.add_argument('output')
    p.add_argument('--writer');p.add_argument('--policy', default='sink_region', choices=sorted(sink.POLICIES))
    a = p.parse_args()
    print(json.dumps(run(a.bindings, a.checkpoint, a.batch, a.output, a.writer, a.policy), indent=1))
