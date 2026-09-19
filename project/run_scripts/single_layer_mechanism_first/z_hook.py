"""Own-entry native-z prefix/head reuse, without shared-module monkeypatches.

Numerical source: pinned BLUE AlphaEdit/compute_z.py (MIT, BLUE/LICENSE),
SHA a941a492d9e9e45f2aa7b88ab0137ae20910b423d7e59186b20c85bb285ff79f.
Prefix-cache/batch-freeze design reference: /data/janghj/tmp/dnm/hooking.py,
SHA feb3509940a40e8b8027ee7daae4c7486fc43ec80394383627699be4f34f072b.
That local reference has no separately located license; it is not imported or
modified. This adapter is restricted to the pinned Llama interface, FP32 and
right padding. Batched arithmetic is NOT asserted bitwise native equivalent.
Actual T0 must qualify the configured batch size before scientific use.

BLUE source license notice (MIT): Copyright (c) 2021 Sid Black.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import inspect
import json
import time
import types

import torch

from project.run_scripts.low_cost_write_donor_pilot.fitting import NativeSingletonFitter


NATIVE_Z_SHA = "a941a492d9e9e45f2aa7b88ab0137ae20910b423d7e59186b20c85bb285ff79f"


class ZHookBoundary(RuntimeError):
    pass


@dataclass(frozen=True)
class ZHookConfig:
    batch_size: int = 1
    record_vectors: bool = False

    def __post_init__(self):
        if self.batch_size not in (1, 16):
            raise ZHookBoundary("UNREGISTERED_Z_BATCH_SIZE")


def _hash_tensor(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _detach(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_detach(x) for x in value)
    if isinstance(value, list):
        return [_detach(x) for x in value]
    if isinstance(value, dict):
        return {k: _detach(v) for k, v in value.items()}
    return value


def normalize_requests(requests):
    """The exact native leading-space normalization; no sample selection."""
    result = copy.deepcopy(requests)
    for request in result:
        if not request["target_new"]["str"]:
            raise ZHookBoundary("EMPTY_NATIVE_TARGET")
        if request["target_new"]["str"][0] != " ":
            request["target_new"]["str"] = " " + request["target_new"]["str"]
    return result


def prepare_batch(tok, requests, contexts, hp, find_lookup, device):
    """Preserve native tokenization/decode/lookup and request/context order."""
    if tok.padding_side != "right":
        raise ZHookBoundary("RIGHT_PADDING_REQUIRED")
    specs, formatted = [], []
    for request in requests:
        target = tok(request["target_new"]["str"], return_tensors="pt")["input_ids"][0]
        if target.numel() and int(target[0]) in (tok.bos_token_id, tok.unk_token_id):
            target = target[1:]
        if not target.numel():
            raise ZHookBoundary("EMPTY_TARGET_TOKENS")
        rewrite = [context.format(request["prompt"]) + tok.decode(target[:-1])
                   for group in contexts for context in group]
        prompts = rewrite + ["{} is a"]
        lookup = [find_lookup(p, request["subject"], tok, hp.fact_token, verbose=False)
                  for p in prompts]
        specs.append(dict(target=target.to(device), lookup=lookup, n_rw=len(rewrite),
                          offset=len(formatted), case_id=request.get("case_id")))
        formatted.extend(p.format(request["subject"]) for p in prompts)
    if not specs or not specs[0]["n_rw"] or len({s["n_rw"] for s in specs}) != 1:
        raise ZHookBoundary("EMPTY_OR_VARIABLE_CONTEXT_INVENTORY")
    tokens = tok(formatted, return_tensors="pt", padding=True).to(device)
    width = tokens["input_ids"].shape[1]
    n_rw = specs[0]["n_rw"]
    targets = torch.full((len(specs) * n_rw, width), -100, dtype=torch.long, device=device)
    rw_rows, kl_rows, kl_cols, intervention_rows, intervention_cols = [], [], [], [], []
    for b, spec in enumerate(specs):
        for j in range(n_rw):
            row = spec["offset"] + j
            length = int(tokens["attention_mask"][row].sum())
            if length < spec["target"].numel():
                raise ZHookBoundary("NATIVE_TARGET_POSITION_UNDERFLOW")
            targets[b*n_rw+j, length-spec["target"].numel():length] = spec["target"]
            rw_rows.append(row)
        kl_rows.append(spec["offset"] + n_rw)
        kl_cols.append(spec["lookup"][-1])
        for j, position in enumerate(spec["lookup"]):
            # Negative 'last' indices retain native padded-tensor indexing.
            if position < -width or position >= width:
                raise ZHookBoundary("LOOKUP_OUT_OF_RANGE")
            intervention_rows.append(spec["offset"]+j)
            intervention_cols.append(position)
    return dict(tokens=tokens, specs=specs, targets=targets, n_rw=n_rw,
                rw_rows=rw_rows, kl_rows=kl_rows, kl_cols=kl_cols,
                intervention_rows=intervention_rows, intervention_cols=intervention_cols,
                identity=_digest(dict(ids=tokens["input_ids"].cpu().tolist(),
                                      mask=tokens["attention_mask"].cpu().tolist(),
                                      lookup=[s["lookup"] for s in specs],
                                      targets=[s["target"].cpu().tolist() for s in specs])))


class _PrefixComplete(Exception):
    pass


def capture_prefix(model, hp, layer, tokens):
    """Execute the real pinned backbone to capture mask/RoPE, stop after layer."""
    module = model.get_submodule(hp.layer_module_tmp.format(layer))
    captured = {}
    def before(_module, args, kwargs):
        if kwargs.get("past_key_value") is not None or kwargs.get("use_cache", False):
            raise ZHookBoundary("MUTABLE_KV_CACHE_FORBIDDEN")
        captured["args"] = _detach(args[1:])
        captured["kwargs"] = _detach(kwargs)
    def after(_module, _args, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.dtype != torch.float32 or hidden.ndim != 3:
            raise ZHookBoundary("PREFIX_FP32_BATCH_SEQUENCE_HIDDEN")
        captured["hidden"] = hidden.detach().clone()
        raise _PrefixComplete()
    handles = [module.register_forward_pre_hook(before, with_kwargs=True),
               module.register_forward_hook(after)]
    try:
        with torch.no_grad():
            try:
                model.model(**tokens, use_cache=False, output_attentions=False,
                            output_hidden_states=False, return_dict=True)
            except _PrefixComplete:
                pass
    finally:
        for handle in handles:
            handle.remove()
    if "hidden" not in captured:
        raise ZHookBoundary("PREFIX_NOT_CAPTURED")
    return captured


def suffix_hidden(model, hp, layer, prefix, delta, batch):
    hidden = prefix["hidden"].clone()
    # Native compute_z uses one sequential in-place add per context. A single
    # repeat_interleave/advanced-index add is mathematically equivalent, but
    # its delta VJP reduces the context axis using a different FP32 tree.
    # Preserve the native per-request AddBackward accumulation path instead.
    for b, spec in enumerate(batch["specs"]):
        request_delta = delta[b]
        for j, position in enumerate(spec["lookup"]):
            hidden[spec["offset"] + j, position, :] += request_delta
    loss_layer = max(hp.v_loss_layer, layer)
    loss_hidden = hidden if layer == loss_layer else None
    layers = model.get_submodule(hp.layer_module_tmp.rsplit(".", 1)[0])
    for index in range(layer + 1, len(layers)):
        kwargs = dict(prefix["kwargs"])
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hidden
            output = layers[index](*prefix["args"], **kwargs)
        else:
            output = layers[index](hidden, *prefix["args"], **kwargs)
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if index == loss_layer:
            loss_hidden = hidden
    if loss_hidden is None:
        raise ZHookBoundary("LOSS_LAYER_NOT_VISITED")
    return loss_hidden, hidden


def native_losses(model, hp, batch, loss_hidden, final_hidden, delta, initial, kl_initial):
    """Full-vocabulary heads, only native target and final-model KL positions."""
    norm = model.get_submodule(hp.ln_f_module)
    head = model.get_submodule(hp.lm_head_module)
    target_mask = batch["targets"] != -100
    # Keep native normalization on the full rewrite representation. Selecting
    # hidden rows before RMSNorm changes both the row layout and the backward
    # graph. Only the expensive full-vocabulary head is position-selected.
    rewrite_repr = loss_hidden[batch["rw_rows"]]
    selected = norm(rewrite_repr)[target_mask]
    # Rewrite path is precisely native ln_f(hidden) @ lm_w.T (+ bias).
    rewrite_logits = selected @ head.weight.T
    bias = head.bias if getattr(head, "bias", None) is not None else head.weight.new_zeros(head.weight.shape[0])
    rewrite_logits = rewrite_logits + bias
    lp = rewrite_logits.log_softmax(-1)
    selected_logp = lp.gather(1, batch["targets"][target_mask, None]).squeeze(1)
    # Native places the negation AFTER sequence-width summation, then divides
    # by target length and takes the scalar context mean for each request.
    # Keep that route (not a batched row reduction or mean of token losses).
    losses = torch.zeros_like(batch["targets"], dtype=torch.float32)
    losses[target_mask] = selected_logp
    nll = torch.stack([
        (-(losses[b*batch["n_rw"]:(b+1)*batch["n_rw"]] *
           target_mask[b*batch["n_rw"]:(b+1)*batch["n_rw"]].float()).sum(1) /
         spec["target"].numel()).mean()
        for b, spec in enumerate(batch["specs"])])
    # KL is FINAL-model hidden even when native rewrite loss_layer is earlier.
    kl_logits = head(norm(final_hidden)[batch["kl_rows"], batch["kl_cols"]])
    kl_lp = kl_logits.log_softmax(-1)
    if kl_initial is None:
        kl_initial = kl_lp.detach().clone()
    # Native KL has shape [one KL prompt, full vocabulary] and batchmean
    # reduction. reduction=none followed by sum(-1) uses a different FP32
    # reduction even for batch_size=1. Independent request batching must not
    # divide a request's KL by the number of other requests.
    kl = torch.stack([hp.kl_factor * torch.nn.functional.kl_div(
        kl_initial[b:b+1], kl_lp[b:b+1], log_target=True, reduction="batchmean")
        for b in range(len(batch["specs"]))])
    decay = torch.stack([hp.v_weight_decay * (torch.norm(delta[b]) /
        torch.norm(initial[b]) ** 2) for b in range(len(batch["specs"]))])
    total = torch.stack([nll[b] + kl[b].to(nll.device) + decay[b].to(nll.device)
                         for b in range(len(batch["specs"]))])
    if not all(torch.isfinite(x).all().item() for x in (total, nll, kl, decay)):
        raise FloatingPointError("NONFINITE_NATIVE_Z_LOSS")
    return total, nll, kl, decay, kl_initial


def freeze_and_clamp(delta, frozen, converged, max_norms):
    """Native clamp order per independent row, then undo frozen Adam momentum."""
    with torch.no_grad():
        for row in range(delta.shape[0]):
            if converged[row]:
                delta[row].copy_(frozen[row])
            elif delta[row].norm() > max_norms[row]:
                delta[row].copy_(delta[row] * max_norms[row] / delta[row].norm())


def compute_z_batch(model, tok, requests, hp, layer, contexts, find_lookup, *, config=ZHookConfig()):
    """Return [hidden,requests] and scalar/vector technical receipts.

    No disk cache, no cross-call prefix, no fit/write/history modification.
    Requests must already have native leading-space normalization.
    """
    if not requests or len(requests) > config.batch_size:
        raise ZHookBoundary("BATCH_SIZE_BOUNDARY")
    if model.training or getattr(model.config, "model_type", None) != "llama":
        raise ZHookBoundary("PINNED_LLAMA_EVAL_REQUIRED")
    if getattr(model.config, "_attn_implementation", "eager") != "eager":
        raise ZHookBoundary("EAGER_REQUIRED")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ZHookBoundary("TF32_MUST_BE_OFF")
    if any(p.dtype != torch.float32 or p.requires_grad for p in model.parameters()):
        raise ZHookBoundary("FP32_FROZEN_MODEL_REQUIRED")
    if (hp.v_num_grad_steps, hp.v_lr, hp.v_weight_decay, hp.clamp_norm_factor, hp.kl_factor) != (25, .1, .5, .75, .0625):
        raise ZHookBoundary("FIXED_NATIVE_HPARAMS")
    device = next(model.parameters()).device
    begin = time.monotonic()
    batch = prepare_batch(tok, requests, contexts, hp, find_lookup, device)
    prefix = capture_prefix(model, hp, layer, batch["tokens"])
    initial = torch.stack([prefix["hidden"][s["offset"], s["lookup"][0]].clone() for s in batch["specs"]])
    if not torch.isfinite(initial).all() or (initial.norm(dim=1) == 0).any():
        raise FloatingPointError("NONFINITE_OR_ZERO_NATIVE_ANCHOR")
    delta = torch.zeros_like(initial, requires_grad=True)
    frozen = delta.detach().clone()
    converged = torch.zeros(len(requests), dtype=torch.bool, device=device)
    optimizer = torch.optim.Adam([delta], lr=hp.v_lr)
    kl_initial = None
    traces = [[] for _ in requests]
    grad_traces = [[] for _ in requests]
    loss_steps = [0]*len(requests)
    adam_steps = [0]*len(requests)
    clamp_hits = [0]*len(requests)
    stop = [None]*len(requests)
    for iteration in range(hp.v_num_grad_steps):
        optimizer.zero_grad()
        lh, fh = suffix_hidden(model, hp, layer, prefix, delta, batch)
        total, nll, kl, decay, kl_initial = native_losses(model, hp, batch, lh, fh, delta, initial, kl_initial)
        for b in range(len(requests)):
            if not converged[b]:
                loss_steps[b] += 1
                traces[b].append(dict(iteration=iteration, loss=float(total[b].detach()), nll=float(nll[b].detach()),
                                      kl=float(kl[b].detach()), decay=float(decay[b].detach())))
        newly = (total.detach() < .05) & ~converged
        frozen[newly] = delta.detach()[newly]
        for b in range(len(requests)):
            if newly[b]: stop[b] = "NATIVE_TOTAL_LOSS_LT_0.05"
        converged |= newly
        if converged.all() or iteration == hp.v_num_grad_steps-1:
            break
        # The production singleton follows native scalar loss.backward().
        # Batching sums the still-active independent scalar losses only.
        (total[0] if len(requests) == 1 else total[~converged].sum()).backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise FloatingPointError("NONFINITE_NATIVE_Z_GRADIENT")
        for b in range(len(requests)):
            if not converged[b]:
                adam_steps[b] += 1
                if config.record_vectors:
                    grad_traces[b].append(delta.grad[b].detach().cpu().clone())
        optimizer.step()
        for b in range(len(requests)):
            if not converged[b] and delta[b].norm() > hp.clamp_norm_factor*initial[b].norm():
                clamp_hits[b] += 1
        freeze_and_clamp(delta, frozen, converged, hp.clamp_norm_factor*initial.norm(dim=1))
    targets = (initial+delta.detach()).T.contiguous()
    if not torch.isfinite(targets).all():
        raise FloatingPointError("NONFINITE_NATIVE_Z_TARGET")
    receipt = dict(batch_size=len(requests), configured_batch_size=config.batch_size,
                   input_identity=batch["identity"], case_ids=[s["case_id"] for s in batch["specs"]],
                   source_native_sha256=NATIVE_Z_SHA, prefix_calls=1,
                   arithmetic_route="native_context_add_and_per_request_reductions_v2",
                   head_positions="rewrite_target_and_final_KL_only_full_vocabulary",
                   suffix_sweeps=iteration+1, full_vocabulary=True, loss_layer=max(hp.v_loss_layer,layer),
                   KL_layer="final-model", dtype="float32", native_history_append=0,
                   seconds=time.monotonic()-begin, loss_steps=loss_steps, adam_steps=adam_steps,
                   clamp_hits=clamp_hits,
                   stop_reasons=[x or "MAX_25_LOSS_24_ADAM" for x in stop], losses=traces,
                   target_sha256=[_hash_tensor(targets[:,b]) for b in range(len(requests))],
                   cache_scope="one z call only; own entry/input/context/layer", bitwise_native_parity="NOT_ESTABLISHED")
    receipt["_target_observations"] = [dict(case_id=s["case_id"],anchor=initial[b].cpu().clone(),
        delta=delta[b].detach().cpu().clone(),target=targets[:,b].cpu().clone(),
        radius=float(hp.clamp_norm_factor*initial[b].norm()),adam_updates=adam_steps[b],
        loss_evaluations=loss_steps[b],clamp_hits=clamp_hits[b],lookup_indices=list(s["lookup"]),
        teacher_sha256=_hash_tensor(kl_initial[b]),loss_final=traces[b][-1]["loss"],
        source_code_filename=__file__) for b,s in enumerate(batch["specs"])]
    if config.record_vectors:
        receipt["gradients"] = grad_traces
        receipt["deltas"] = delta.detach().cpu().clone()
    return targets, receipt


class HookedNativeSingletonFitter(NativeSingletonFitter):
    """Inject only copied function globals; native solve/history stay intact."""
    def __init__(self, module, *, expected_source_sha256, contexts, config=ZHookConfig(), event=None):
        super().__init__(module, expected_source_sha256=expected_source_sha256, contexts=contexts)
        if hashlib.sha256(inspect.getsource(inspect.getmodule(module.compute_z)).encode()).hexdigest() != NATIVE_Z_SHA:
            raise ZHookBoundary("PINNED_COMPUTE_Z_SOURCE_SHA")
        self.config, self.event = config, event
        self._preloaded = None
        self.z_receipts = []
        self.target_observations = []

    def _record(self, receipt):
        self.target_observations.extend(receipt.pop("_target_observations"))
        self.z_receipts.append(receipt)
        if self.event: self.event(receipt)

    def fit(self, model, tok, hp, history, projector, requests, *, layer, capture=False):
        normalized = normalize_requests(requests)
        self._preloaded = None
        self.z_receipts = []
        self.target_observations = []
        started = time.monotonic()
        if self.config.batch_size > 1:
            # Native BLUE computes every z before any write. Preload at that same
            # entry; each request is still consumed once by the native loop.
            queue=[]
            for offset in range(0,len(normalized),self.config.batch_size):
                part=normalized[offset:offset+self.config.batch_size]
                targets, receipt=compute_z_batch(model,tok,part,hp,layer,self.contexts,
                                                 self.module.find_fact_lookup_idx,config=self.config)
                self._record(receipt)
                queue.extend((_digest(r),targets[:,i]) for i,r in enumerate(part))
            self._preloaded=iter(queue)
        try:
            result=super().fit(model,tok,hp,history,projector,requests,layer=layer,capture=capture)
            if self._preloaded is not None and next(self._preloaded,None) is not None:
                raise ZHookBoundary("UNCONSUMED_PRELOADED_TARGET")
            result["receipt"]["z_hook"] = dict(config=vars(self.config),batches=self.z_receipts,
                total_loss_steps=sum(sum(x["loss_steps"]) for x in self.z_receipts),
                total_adam_steps=sum(sum(x["adam_steps"]) for x in self.z_receipts),
                prefix_calls=len(self.z_receipts), numerical_parity="REQUIRES_T0_BINDING")
            result["receipt"]["seconds_including_z_preload"]=time.monotonic()-started
            return result
        finally:
            self._preloaded=None

    def _functions(self, counts, capture):
        fit, finalizer=super()._functions(counts,capture)
        ns=dict(fit.__globals__)
        def hooked(model,tok,request,hp,layer,contexts):
            begin=time.monotonic()
            if contexts!=self.contexts: raise ZHookBoundary("CONTEXT_CHANGED")
            if self._preloaded is not None:
                item=next(self._preloaded,None)
                if item is None or item[0]!=_digest(request): raise ZHookBoundary("PRELOAD_REQUEST_ORDER")
                target=item[1]
            else:
                z,receipt=compute_z_batch(model,tok,[request],hp,layer,contexts,
                                          self.module.find_fact_lookup_idx,config=self.config)
                target=z[:,0];self._record(receipt)
            counts["compute_z"]=counts.get("compute_z",0)+1
            counts["compute_z_seconds"]=counts.get("compute_z_seconds",0.)+time.monotonic()-begin
            if capture is not None: capture.setdefault("compute_z",[]).append(target.detach().cpu().clone())
            return target
        ns["compute_z"]=hooked
        return types.FunctionType(fit.__code__,ns,fit.__name__,fit.__defaults__,fit.__closure__),finalizer


def capture_hooked_native_fit(fitter,model,tok,hp,history,projector,requests,*,layer=4):
    """Return the existing native capsule schema from directly recorded z rows.

    The legacy capture_native_fit profiles the ORIGINAL compute_z code object
    and cannot observe this new adapter. This function makes that distinction
    explicit; it never labels the hook as source-unmodified native execution.
    """
    if not isinstance(fitter,HookedNativeSingletonFitter):raise ZHookBoundary("HOOKED_FITTER_REQUIRED")
    fit=fitter.fit(model,tok,hp,history,projector,requests,layer=layer,capture=True)
    rows=fitter.target_observations
    if [r['case_id'] for r in rows]!=[r['case_id'] for r in requests]:raise ZHookBoundary("HOOK_CAPTURE_REQUEST_ORDER")
    z=torch.stack(fit['captures']['compute_z'],dim=1)
    observed=torch.stack([r['target'] for r in rows],dim=1)
    if not torch.equal(z,observed):raise ZHookBoundary("HOOK_RETURN_CAPTURE_BRIDGE")
    fit.update(anchors=torch.stack([r['anchor'] for r in rows],dim=1),
               radii=torch.tensor([r['radius'] for r in rows],dtype=torch.float32),
               target=z,target_observations=rows)
    fit['receipt']['anchor_capture']=dict(method='TASK_LOCAL_Z_HOOK_DIRECT_RECEIPT',
        calls=len(rows),numeric_source_modified=True,native_equivalence='T0_QUALIFIED_SCOPE_ONLY',
        adam_updates=sum(r['adam_updates'] for r in rows),loss_evaluations=sum(r['loss_evaluations'] for r in rows),
        clamp_hits=sum(r['clamp_hits'] for r in rows),target_return_exact=True,extra_target_calls=0)
    return fit


def instrument_native_compute_z(native_function, sink):
    """Technical-only source-exact function clone with loss/gradient callbacks.

    No arithmetic is replaced. The sink receives local small vectors. Caller
    freezes source identity and uses this only on its fixed T0 request panel.
    """
    source=inspect.getsource(native_function)
    tree=ast.parse(source)
    fn=tree.body[0]
    if not isinstance(fn,ast.FunctionDef) or fn.name!="compute_z": raise ZHookBoundary("NATIVE_FUNCTION_LAYOUT")
    loss_added=gradient_added=0
    for node in ast.walk(fn):
        if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=="it":
            body=[]
            for statement in node.body:
                body.append(statement)
                if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="loss" for t in statement.targets):
                    # This loop's scalar total loss assignment is distinct from
                    # the gathered per-token loss assignment preceding it.
                    if isinstance(statement.value,ast.BinOp):
                        body.append(ast.parse("_z_sink('loss', it, delta, loss, nll_loss, kl_loss, weight_decay)").body[0]);loss_added+=1
                if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call):
                    call=statement.value
                    if isinstance(call.func,ast.Attribute) and call.func.attr=="backward":
                        body.append(ast.parse("_z_sink('gradient', it, delta, loss, nll_loss, kl_loss, weight_decay)").body[0]);gradient_added+=1
            node.body=body
    if (loss_added,gradient_added)!=(1,1): raise ZHookBoundary("NATIVE_INSTRUMENTATION_LAYOUT")
    ns=dict(native_function.__globals__);ns["_z_sink"]=sink
    exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(native_function)+":technical-observation","exec"),ns)
    return ns["compute_z"]
