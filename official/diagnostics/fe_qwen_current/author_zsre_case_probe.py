"""POST-B15 fixed three-case no-write probe. See author_zsre_case_probe.md.

CLI: python -m official.diagnostics.fe_qwen_current.author_zsre_case_probe BINDINGS NEW_OUTPUT
No writer, history, solve, evaluator, checkpoint save, tracking or submission.
This is not a historical B13/B15 replay or an improved benchmark.
"""
import argparse
import copy
import importlib
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from official.diagnostics.fe_qwen_current import author_basic_eot_benchmark as author
from official.diagnostics.fe_qwen_current import failure_eot_probe as eot
from official.diagnostics.fe_qwen_current import failure_probe as base
from official.diagnostics.fe_qwen_current.failure_case_probe import compare, linear_attribution
from official.diagnostics.fe_qwen_current.benchmark import native_records
from official.diagnostics.fe_qwen_current.storage import state_digest, tensor_sha
from official.experiments.checkpoint import rng_snapshot
from official.experiments.prepare import digest, write_new
from official.runners.server1.common import read, verify, member, require

SCHEMA = 'author-zsre-post-B15-case-probe-v1'
CASES = [1417, 1425, 1457]
SOURCE = '2144a6feffb51f66ecc24de2f818f632d5eeb8a7'
CP_SHA = '1235764707e1128122c94d4da5c3926a5aaa8d02bfbf219f0e981d3f8e8eee43'
CP_BYTES = 7856481561
STREAM_SHA = 'f42ee4bc6e98b1133e48dc81102201ccd85ccc917a97160ee3c6a0f3a38f378c'
ADAPTER_SHA = '3b043010cd576fbd9843ca34ec2e4514eea45e2fd68dc09005ca1d7b24399755'
AUTHOR_MODULE_SHA = 'f7e35aa3119bbda75ff5e0d85a71f300c988a95f049dacb71673a5b8c3a0b972'


def validate_parent(parent, runtime, audit, commit, pointer, cp, profiles):
    """Small metadata-only guard, separately tested with the real author YAML."""
    cfg = OmegaConf.create(runtime['native_config'])
    resolved = author.audit_settings(cfg, profiles)
    require(runtime['author_settings'] == audit and audit['resolved'] == resolved
            and audit['resolved_sha256'] == digest(resolved)
            and audit['profiles_sha256'] == digest(profiles)
            and audit['status'] == 'AUTHOR_SETTINGS_CPU_VERIFIED'
            and audit['author_commit'] == author.AUTHOR, 'AUTHOR_SETTINGS_BINDING')
    identity = runtime['identity']; author.settings(identity)
    require(identity['dataset'] == parent['dataset'] == 'zsre'
            and identity['source'] == parent['execution']['source_commit'] == SOURCE
            and str(identity['job_id']) == str(parent['execution']['job_id']) == '121243'
            and identity['config_sha256'] == digest(parent)
            and identity['author_settings_sha256'] == audit['resolved_sha256']
            and identity['author_commit'] == author.AUTHOR, 'PARTIAL_AUTHOR_RUN_IDENTITY')
    require(identity['stream'] == parent['main_stream']
            and identity['stream']['sha256'] == STREAM_SHA
            and identity['adapter']['sha256'] == ADAPTER_SHA, 'STREAM_ADAPTER_IDENTITY')
    require(cp['sha256'] == CP_SHA and cp['bytes'] == CP_BYTES
            and all(pointer[k] == cp[k] for k in ('path', 'bytes', 'sha256'))
            and pointer['batch'] == commit['batch'] == 15 and commit['edits'] == 1500
            and pointer['final_W20'] is False and pointer['identity'] == commit['identity'] == identity
            and commit['checkpoint'] == pointer, 'EXACT_B15_COMMIT_POINTER')
    require(cfg.gpu == 0 and cfg.diagnostic_z_reference == 'batch_entry'
            and runtime['attention_backend'] == 'sdpa', 'POST_B15_NATIVE_SCOPE')
    return cfg


def validate_bindings(b):
    require(b['schema'] == SCHEMA and b['case_ids'] == CASES, 'FIXED_PROBE_SCOPE')
    execution = b['execution']
    require(len(execution['source_commit']) == 40
            and all(c in '0123456789abcdef' for c in execution['source_commit'])
            and str(execution['job_id']) == os.environ.get('ODEEDIT_RENT_JOB_ID')
            and str(execution['job_id']) != '121243' and execution['attempt'], 'NEW_ACTUAL_JOB')
    require(member(verify(b['probe_source']))['sha256'] == member(__file__)['sha256'], 'PROBE_SOURCE')
    require(member(verify(b['author_source']))['sha256'] == member(author.__file__)['sha256'] == AUTHOR_MODULE_SHA, 'FROZEN_AUTHOR_SOURCE')
    require(member(eot.__file__)['sha256'] == ADAPTER_SHA, 'C36_UNCHANGED')
    parent = read(verify(b['author_bindings']))
    # Reuse the original exact source/native/model references. No old checkpoint
    # tensor is loaded here and the historical L2=10 cfg is NEVER executed.
    old = base.validate_inputs(parent)
    runtime, audit, commit, pointer = [read(verify(b[k])) for k in ('runtime', 'author_settings', 'commit', 'pointer')]
    profiles = author.author_profiles(parent)
    cfg = validate_parent(parent, runtime, audit, commit, pointer, b['checkpoint'], profiles)
    _, expected_audit = author.execution_config(old['runtime']['resolved'], parent)
    require(audit == expected_audit and runtime['attention_backend'] == old['runtime']['attention_backend'], 'ORIGINAL_USED_CONFIG')
    verify(runtime['identity']['adapter'])
    for raw in commit['raw']: verify(raw)
    stream = author.common.main_stream(parent, read(verify(old['config']['stream'])))
    require([r['case_id'] for r in stream if r['case_id'] in CASES] == CASES, 'FIXED_STREAM_ORDER')
    verify(b['checkpoint'])
    return parent, old, runtime, audit, commit, pointer, cfg, stream


def restore_weights_only(model, payload, cfg, identity, commit, stream):
    """No access to cache_c/H, no full payload digest/clone or history restore."""
    require(payload['identity'] == identity and payload['batch'] == 15, 'PAYLOAD_B15_IDENTITY')
    cur = payload['cursor']
    require(cur['native_config'] == OmegaConf.to_container(cfg, resolve=True)
            and cur['edits'] == 1500 and cur['case_ids'] == [r['case_id'] for r in stream[:1500]]
            and cur['raw'] == commit['raw'] and cur['input_adapter']['sha256'] == ADAPTER_SHA, 'PAYLOAD_CURSOR')
    names = {cfg.llms.rewrite_module_tmp.format(l) + '.weight' for l in cfg.llms.layers}
    params = dict(model.named_parameters())
    require(set(payload['weights']) == names and names <= set(params), 'SELECTED_WEIGHTS_ONLY')
    untouched = {n: (id(p), p._version) for n, p in params.items() if n not in names}
    restored = {}
    with torch.no_grad():
        for n, w in payload['weights'].items():
            require(w.dtype == params[n].dtype == torch.bfloat16 and w.shape == params[n].shape, 'BF16_RESTORE')
            params[n].copy_(w); restored[n] = tensor_sha(w)
            require(tensor_sha(params[n]) == restored[n], 'EXACT_SELECTED_RESTORE')
    require(untouched == {n: (id(p), p._version) for n, p in params.items() if n not in names}, 'UNSELECTED_CHANGED')
    contexts = copy.deepcopy(payload['contexts']); rng = payload['rng']
    require(len(contexts) == 2 and contexts[0] == ['{}'] and len(contexts[1]) == 5, 'NATIVE_CONTEXT_GROUPS')
    return contexts, rng, restored


def key_geometry(keys):
    require(keys.ndim == 2 and keys.shape[0] == 7 and keys.dtype == torch.bfloat16, 'SIX_REWRITE_PLUS_KL')
    # Exact native group order / BF16 reduction: first group bare, second five
    # prefixes, equal group average. KL is intentionally excluded.
    bar = torch.stack([keys[:1].mean(0), keys[1:6].mean(0)]).mean(0)
    bare = keys[0]
    return dict(bare=base.scalar_vector(bare), context_average=base.scalar_vector(bar),
                cosine=float(torch.nn.functional.cosine_similarity(bare.float(), bar.float(), dim=0)),
                bare_vs_average=compare(bare, bar), KL_excluded=True,
                native_group_order=[1, 5], reduction_dtype=str(bar.dtype))


def hooks_state(model):
    return {n: (tuple(m._forward_hooks), tuple(m._forward_pre_hooks), tuple(m._backward_hooks))
            for n, m in model.named_modules()}


def observe(model, tok, requests, cfg, contexts, *, compute_z, supplier, lookup, TraceDict, w0):
    require([r['case_id'] for r in requests] == CASES and len(requests) == 3, 'FIXED_THREE_REQUESTS')
    require(list(cfg.llms.layers) == [4, 5, 6, 7, 8] and cfg.algs.L2 == 0
            and cfg.llms.clamp_norm_factor == 1 and cfg.diagnostic_z_reference == 'batch_entry', 'OBSERVATION_SCOPE')
    require(contexts[0] == ['{}'] and len(contexts) == 2 and len(contexts[1]) == 5, 'SIX_CONTEXTS')
    require(sys.getprofile() is None and all(not m.training for m in model.modules()), 'EVAL_NO_ACTIVE_PROFILE')
    adapter = eot.InputAdapter(tok, lookup)
    fit_fn = adapter.fit(compute_z); supply = adapter.supplier(supplier)
    for r in requests:
        for prompt in [p.format(r['prompt']) for group in contexts for p in group] + ['{} is a']:
            require(adapter.prompt(prompt, r['subject']) == prompt
                    and lookup(prompt, r['subject'], tok, cfg.llms.fact_token, verbose=False) > 0, 'EXPECTED_INERT_EOT')
    mods = dict(model.named_modules()); l4 = cfg.llms.layer_module_tmp.format(4)
    l5 = cfg.llms.layer_module_tmp.format(5); down = cfg.llms.rewrite_module_tmp.format(5)
    require(mods[down].bias is None and mods[down].weight.dtype == torch.bfloat16, 'NATIVE_BF16_BIAS_FREE')
    weights = {n: (id(p), p._version) for n, p in model.named_parameters()}
    requires_grad = {n: p.requires_grad for n, p in model.named_parameters()}
    modes = {n: m.training for n, m in model.named_modules()}; hooks = hooks_state(model)
    rng = state_digest(rng_snapshot()); cfg_hash = digest(OmegaConf.to_container(cfg, resolve=True))
    context_hash = state_digest(contexts); w0_hash = tensor_sha(w0)
    state = dict(phase='fit', frame=None, replay=-1)
    capture = {}; initial_capture = {}; fits = []; replays = []

    def positions():
        if state['phase'] == 'fit': return state['frame'].f_locals['lookup_idxs']
        r = requests[state['replay']]
        return [adapter.lookup(r['prompt'], r['subject'], tok, cfg.llms.fact_token, verbose=False)]

    def points(value):
        h = base.hidden(value)
        return torch.stack([h[i, int(p)].detach().clone() for i, p in enumerate(positions())])

    def l4_hook(module, args, value):
        if state['phase'] == 'replay':
            state['replay'] += 1; capture.clear()
            capture['pre_L4'] = points(value)

    def key_hook(module, args, value):
        capture['key'] = points(args[0]); capture['mlp'] = points(value)

    def l5_hook(module, args, value):
        capture['block'] = points(value)
        if state['phase'] == 'fit' and not initial_capture:
            require(torch.count_nonzero(state['frame'].f_locals['delta']).item() == 0, 'FIRST_FIT_DELTA_ZERO')
            initial_capture.update(capture)
        elif state['phase'] == 'replay': replays.append(dict(capture))

    handles = [mods[l4].register_forward_hook(l4_hook), mods[down].register_forward_hook(key_hook), mods[l5].register_forward_hook(l5_hook)]

    def fitted(*args, **kw):
        capture.clear(); initial_capture.clear(); receipt = {}
        def profile(frame, event, value):
            if frame.f_code is fit_fn.__code__:
                if event == 'call': state['frame'] = frame
                elif event == 'return' and value is not None:
                    v = frame.f_locals; tokens = v['input_tok']; rows = []
                    for i, pos in enumerate(v['lookup_idxs']):
                        length = int(tokens['attention_mask'][i].sum())
                        rows.append(dict(kind='rewrite' if i < 6 else 'KL', lookup=int(pos),
                            ids=tokens['input_ids'][i, :length].tolist(), mask=tokens['attention_mask'][i, :length].tolist(),
                            labels=v['rewriting_targets'][i, :length].tolist() if i < 6 else None))
                    require(len(rows) == 7 and all(r['lookup'] > 0 and all(r['mask']) for r in rows), 'ACTUAL_FIT_INPUTS')
                    actual = base.hidden(v['tr'][l4].output)[0, rows[0]['lookup']].detach().clone()
                    receipt.update(first=dict(initial_capture), final=dict(capture), rows=rows,
                        init=v['target_init'].detach().clone(), delta=v['delta'].detach().clone(), returned=value.detach().clone(),
                        effective_L4=actual, nll=v['nll_loss_each'].detach().float().cpu().tolist(),
                        probabilities=v['nll_loss_each'].detach().float().neg().exp().cpu().tolist(), iterations=int(v['it']) + 1)
        sys.setprofile(profile)
        try: result = fit_fn(*args, **kw)
        finally: sys.setprofile(None); state['frame'] = None
        require(bool(receipt), 'NATIVE_FIT_CAPTURE'); fits.append(receipt)
        if len(fits) == 3: state['phase'] = 'replay'
        return result

    try:
        try:
            targets = supply(model, tok, requests, cfg, contexts, compute_z=fitted,
                             find_fact_lookup_idx=adapter.lookup, TraceDict=TraceDict)
        finally:
            for h in handles: h.remove()
        require(len(fits) == len(replays) == 3, 'THREE_FITS_THREE_REPLAYS')
        rows = []
        for i, r in enumerate(requests):
            f = fits[i]; rp = replays[i]; w15 = mods[down].weight
            def linear(key, output):
                # Reuse tested chunked arithmetic, renaming the old helper's B5
                # labels explicitly to B15. It never changes a model parameter.
                value = linear_attribution(w0, w15, key, output)
                def rename(x):
                    if isinstance(x, dict): return {k.replace('B5', 'B15'): rename(v) for k, v in x.items()}
                    if isinstance(x, list): return [rename(v) for v in x]
                    return x
                return rename(value)
            fit_rows = []
            for j, inp in enumerate(f['rows']):
                fit_rows.append(dict(input=inp, first_delta_zero={k: base.scalar_vector(v[j]) for k, v in f['first'].items()},
                    final={k: base.scalar_vector(v[j]) for k, v in f['final'].items()},
                    linear=linear(f['final']['key'][j], f['final']['mlp'][j])))
            row = dict(case_id=r['case_id'], stream_index=r.get('original_stream_index'), sample_idx=i, input_rows=fit_rows,
                fit_nll=f['nll'], fit_probabilities=f['probabilities'], fit_iterations=f['iterations'],
                L4_init=base.scalar_vector(f['init']), L4_delta=base.scalar_vector(f['delta']),
                L4_returned=base.scalar_vector(f['returned']), L4_effective_final_fit=base.scalar_vector(f['effective_L4']),
                final_fit_vs_return_cast=compare(f['effective_L4'], f['returned'].to(f['effective_L4'].dtype)),
                returned_FP32_vs_BF16=compare(f['returned'], f['returned'].to(torch.bfloat16)),
                fit_init_vs_replay_pre_L4=compare(f['init'], rp['pre_L4'][0]),
                replay={k: base.scalar_vector(v[0]) for k, v in rp.items()},
                fit_bare_vs_replay={k: compare(f['final'][k][0], rp[k][0]) for k in ('key', 'mlp', 'block')},
                first_fit_delta_zero_key_geometry=key_geometry(f['first']['key']),
                final_fit_injected_key_geometry=key_geometry(f['final']['key']),
                key_scope='POST_B15_ENTRY_CAUSAL_SUBJECT_KEYS_NOT_HISTORICAL_WRITER_STAGE',
                replay_linear=linear(rp['key'][0], rp['mlp'][0]),
                targets={str(l): base.scalar_vector(z[:, i]) for l, z in targets.items()}, extra=[])
            pos = adapter.lookup(r['prompt'], r['subject'], tok, cfg.llms.fact_token, verbose=False)
            inputs = tok(adapter.prompt(r['prompt'], r['subject']).format(r['subject']), return_tensors='pt').to(next(model.parameters()).device)
            # Exactly two additional bare forwards per case, never six-context
            # extra fits or forwards. Keep only subject vectors in passive hooks.
            for injected in (False, True):
                extra = {}
                def inject(value, name):
                    if injected and name == l4: base.hidden(value)[0, pos] = targets[4][:, i]
                    return value
                def extra_key(module, args, value):
                    extra['key'] = args[0][0, pos].detach().clone(); extra['mlp'] = base.hidden(value)[0, pos].detach().clone()
                handle = mods[down].register_forward_hook(extra_key)
                try:
                    with torch.no_grad(), TraceDict(model, layers=[l4, l5], edit_output=inject) as trace: model(**inputs)
                    extra.update(L4=base.hidden(trace[l4].output)[0, pos].detach().clone(), block=base.hidden(trace[l5].output)[0, pos].detach().clone())
                finally: handle.remove()
                row['extra'].append(dict(injected=injected, lookup=pos, values={k: base.scalar_vector(v) for k, v in extra.items()},
                    first_fit_bare_vs_plain=None if injected else {k: compare(f['first'][k][0], extra[k]) for k in ('key', 'mlp', 'block')},
                    linear=linear(extra['key'], extra['mlp'])))
            rows.append(row)
    finally:
        for n, p in model.named_parameters(): p.requires_grad_(requires_grad[n])
    require(weights == {n: (id(p), p._version) for n, p in model.named_parameters()}, 'WEIGHTS_CHANGED')
    require(hooks == hooks_state(model) and modes == {n: m.training for n, m in model.named_modules()}, 'HOOK_MODE_CHANGED')
    require(rng == state_digest(rng_snapshot()) and tensor_sha(w0) == w0_hash, 'RNG_W0_COPY_CHANGED')
    require(cfg_hash == digest(OmegaConf.to_container(cfg, resolve=True)) and context_hash == state_digest(contexts), 'CONFIG_CONTEXT_CHANGED')
    return targets, dict(rows=rows, fits=3, supplier_replays=3, extra_bare_forwards=6,
        weights_rng_hooks_modes_unchanged=True, EOT_changed_rows=0, writer_calls=0, solve_calls=0,
        history_accessed=False, evaluation_calls=0, checkpoint_saves=0, W0_copy='L5_down_proj_only',
        overlay_proofs=adapter.proofs, historical_reproduction=False, scope='POST_B15_FIXED_STATE_ONLY')


def run(bindings, output):
    b = read(bindings); out = Path(output)
    require(not out.exists() and not out.is_symlink(), 'NEW_OUTPUT_ONLY')
    parent, old, runtime, audit, commit, pointer, cfg, stream = validate_bindings(b)
    import transformers
    import numpy
    import tokenizers
    require(torch.__version__ == old['lock']['runtime']['torch']
            and transformers.__version__ == old['lock']['runtime']['transformers']
            and numpy.__version__ == '1.26.4' and tokenizers.__version__ == '0.21.4', 'EXACT_RUNTIME')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'ROOT_GPU1_ONLY')
    snap = read(verify(old['config']['model_assets']))['model']['snapshot']
    model = transformers.AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True).to('cuda:0')
    require(model.config._attn_implementation == runtime['attention_backend'], 'ORIGINAL_BACKEND')
    tok = transformers.AutoTokenizer.from_pretrained(snap, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    require(tok.padding_side == 'right' and tok.pad_token_id == 151643, 'ORIGINAL_TOKENIZER')
    params = dict(model.named_parameters()); down = cfg.llms.rewrite_module_tmp.format(5) + '.weight'
    w0 = params[down].detach().to('cpu', copy=True)  # ONLY original L5; no H/model copy.
    payload = torch.load(b['checkpoint']['path'], map_location='cpu', mmap=True, weights_only=False)
    contexts, rng, restored = restore_weights_only(model, payload, cfg, runtime['identity'], commit, stream)
    del payload  # Never index, hash, copy or restore cache_c.
    sys.path.insert(0, old['lock']['author_path'])
    native = importlib.import_module('algs.memit.memit_main'); z = importlib.import_module('algs.memit.compute_z')
    supplier = importlib.import_module('algs.memit.native_sequential_z')
    for mod in (native, z, supplier):
        require(Path(mod.__file__).resolve().is_relative_to(Path(old['lock']['author_path']).resolve()), 'PINNED_NATIVE_IMPORT')
        require(any(m['sha256'] == member(mod.__file__)['sha256'] for m in old['lock']['author_members']), 'LOCKED_NATIVE_BYTES')
    requests = native_records([next(r for r in stream if r['case_id'] == case) for case in CASES])
    for i, r in enumerate(requests):
        r['sample_idx'] = i; r['target_new'] = ' ' + r['target_new']
        r['original_stream_index'] = next(j for j, row in enumerate(stream) if row['case_id'] == r['case_id'])
    base.restore_native_execution_state(native.set_random_seed, cfg.seed, rng)
    require(state_digest(rng_snapshot()) == state_digest(rng), 'EXACT_RESTORED_RNG')
    _, result = observe(model, tok, requests, cfg, contexts, compute_z=z.compute_z,
        supplier=supplier.current_batch_targets, lookup=z.find_fact_lookup_idx, TraceDict=z.nethook.TraceDict, w0=w0)
    require(restored == {n: tensor_sha(params[n]) for n in restored}, 'SELECTED_HASH_CHANGED')
    verify(b['checkpoint'])
    result.update(status='AUTHOR_ZSRE_POST_B15_CASE_PROBE_COMPLETE', schema=SCHEMA,
        bindings=member(bindings), source=member(__file__), execution=b['execution'], parent_identity=runtime['identity'],
        checkpoint=b['checkpoint'], restored_weight_sha256=restored, contexts_sha256=state_digest(contexts), RNG_sha256=state_digest(rng),
        source_checkpoint_fullSHA_verified=True, H_integrity_not_traversed=True, author_settings_sha256=audit['resolved_sha256'],
        recovery_claim=False, W20_complete=False, B13_or_pre_B15_reconstruction=False,
        runtime=dict(torch=torch.__version__, transformers=transformers.__version__, numpy=numpy.__version__, tokenizers=tokenizers.__version__, attention_backend=model.config._attn_implementation,
            gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
            matmul_TF32=torch.backends.cuda.matmul.allow_tf32, cudnn_TF32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
            tokenizer_pad_id=tok.pad_token_id, tokenizer_padding_side=tok.padding_side))
    out.mkdir(parents=True); write_new(out / 'case-probe.json', result)
    return member(out / 'case-probe.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bindings'); parser.add_argument('output'); args = parser.parse_args()
    print(run(args.bindings, args.output))
