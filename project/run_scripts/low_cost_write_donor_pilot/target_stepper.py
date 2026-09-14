"""Request-local carried Adam targets; no model writes or history mutation.

The loss/readout follows pinned BLUE AlphaEdit/compute_z.py. Unlike native fresh
compute_z, this adapter keeps one entry-offset leaf and Adam object across
chunks, rebasing the *hook* against the current unhooked first-context anchor.
The caller must finish all requests in a chunk before its B100 writer runs.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import time

import torch


class TargetBoundary(RuntimeError):
    pass


def _cpu(x):
    return x.detach().cpu().clone()


def _sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def native_regularizer(u, a0, decay):
    return decay * (torch.norm(u) / torch.norm(a0) ** 2)


def native_kl(teacher, current, factor):
    return factor * torch.nn.functional.kl_div(
        teacher, current, log_target=True, reduction="batchmean")


def rebased_hook_delta(u, a0, aj):
    """Preserve the native leaf graph when no exact coordinate shift is needed.

    Sharing a zero-offset AddBackward node across contexts changes the FP32
    accumulation order against the direct-leaf norm regularizer. This is an
    exact-state branch, never a tolerance/quality or optimizer-policy branch.
    """
    zero_offset = torch.equal(a0, aj)
    return (u if zero_offset else u + (a0 - aj)), zero_offset


@dataclass
class RequestTargetState:
    request_index: int
    request: dict
    input_tok: object
    target_ids: torch.Tensor
    rewriting_targets: torch.Tensor
    rewriting_count: int
    lookup_idxs: list
    u: torch.Tensor
    opt: object
    owner: object
    a0: object = None
    aj: object = None
    teacher: object = None
    chunks: int = 0
    actual_updates: int = 0
    loss_evaluations: int = 0
    clamp_hits: int = 0
    summaries: list = field(default_factory=list)

    def absolute_target(self):
        if self.a0 is None:
            raise TargetBoundary("UNINITIALIZED_TARGET")
        return (self.a0 + self.u).detach()


def snapshot_state(state):
    """Detached CPU payload. A snapshot is evidence, not a GPU replay claim."""
    if state.a0 is None or state.teacher is None:
        raise TargetBoundary("UNINITIALIZED_STATE_SNAPSHOT")
    adam = state.opt.state.get(state.u, {})
    zero = torch.zeros_like(state.u)
    step = adam.get("step", 0)
    return dict(
        schema="request-entry-offset-adam-v1", request_index=state.request_index,
        a0=_cpu(state.a0), aj=_cpu(state.aj), u=_cpu(state.u),
        Z=_cpu(state.absolute_target()), m=_cpu(adam.get("exp_avg", zero)),
        v=_cpu(adam.get("exp_avg_sq", zero)),
        t=int(step.item()) if torch.is_tensor(step) else int(step),
        teacher=_cpu(state.teacher), teacher_sha256=_sha(state.teacher),
        target_ids=_cpu(state.target_ids), input_ids=_cpu(state.input_tok["input_ids"]),
        attention_mask=_cpu(state.input_tok["attention_mask"]),
        rewriting_targets=_cpu(state.rewriting_targets), lookup_idxs=list(state.lookup_idxs),
        chunks=state.chunks, actual_adam_updates=state.actual_updates,
        target_loss_evaluations=state.loss_evaluations, clamp_hits=state.clamp_hits,
        optimizer_leaf="u_entry_offset", hook_delta="u+(a0-aj)",
        teacher_reference="request_batch_entry_native_zero_delta_first_loss",
        rebasing_does_not_replace_leaf_or_moments=True)


def advance_adam(u, opt, loss_fn, *, max_updates, max_norm_fn,
                 early_stop=0.05):
    """Native initial/post-loss order, reusable by model adapter and CPU tests.

    Each loss_fn returns (scalar loss, JSON-safe components). No unused quota is
    carried. Every invocation rechecks its initial loss, including zero quota.
    """
    if not isinstance(max_updates, int) or not 0 <= max_updates <= 24:
        raise TargetBoundary("UNREGISTERED_TARGET_QUOTA")
    losses, updates, clamps = [], 0, 0
    forward_seconds = backward_step_seconds = 0.0
    for it in range(max_updates + 1):
        opt.zero_grad()
        started = time.monotonic()
        loss, values = loss_fn()
        if loss.ndim or not torch.isfinite(loss).item():
            raise TargetBoundary("NONFINITE_OR_NONSCALE_TARGET_LOSS")
        losses.append(dict(values, total=float(loss.detach().item()), iteration=it))
        forward_seconds += time.monotonic() - started
        if loss < early_stop:
            stop = "LOSS_BELOW_0_05"
            break
        if it == max_updates:
            stop = "CHUNK_QUOTA_EXHAUSTED"
            break
        started = time.monotonic()
        loss.backward()
        if u.grad is None or not torch.isfinite(u.grad).all().item():
            raise TargetBoundary("NONFINITE_OR_MISSING_U_GRAD")
        opt.step()
        updates += 1
        max_norm = max_norm_fn()
        if u.norm() > max_norm:
            with torch.no_grad():
                u[...] = u * max_norm / u.norm()
            clamps += 1
        if not torch.isfinite(u).all().item():
            raise TargetBoundary("NONFINITE_TARGET_UPDATE")
        backward_step_seconds += time.monotonic() - started
    return dict(losses=losses, actual_adam_updates=updates,
                target_loss_evaluations=len(losses), target_forwards=len(losses),
                target_backwards=updates, clamp_hits=clamps, stop_reason=stop,
                unused_quota=max_updates-updates, quota_transferred=0,
                target_forward_loss_seconds=forward_seconds,
                target_backward_step_clamp_seconds=backward_step_seconds)


class NativeTargetStepper:
    def __init__(self, model, tok, hparams, layer, contexts, native_module):
        if layer != 4 or hparams.layers != [4] or not hparams.blue or hparams.L2 != 1:
            raise TargetBoundary("REFRESH_L4_SINGLETON_CONFIG")
        if hparams.v_num_grad_steps != 25 or tok.padding_side != "right" or model.training:
            raise TargetBoundary("NATIVE_BUDGET_PADDING_MODE")
        self.model, self.tok, self.hp = model, tok, hparams
        self.layer, self.contexts, self.native = layer, deepcopy(contexts), native_module
        self.nethook = native_module.nethook
        self.device = next(model.parameters()).device
        if next(model.parameters()).dtype != torch.float32:
            raise TargetBoundary("FP32_TARGET_REQUIRED")
        self.lm_w = self.nethook.get_module(model, hparams.lm_head_module).weight.T
        self.ln_f = self.nethook.get_module(model, hparams.ln_f_module)
        try:
            self.lm_b = self.nethook.get_parameter(model, hparams.lm_head_module + ".bias")
        except LookupError:
            self.lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    def create_state(self, request, request_index):
        hp, tok = self.hp, self.tok
        request = deepcopy(request)
        # Same normalization as native apply_AlphaEdit_to_model, not compute_z.
        if not request["target_new"]["str"]:
            raise TargetBoundary("EMPTY_TARGET")
        if request["target_new"]["str"][0] != " ":
            request["target_new"]["str"] = " " + request["target_new"]["str"]
        target_ids = tok(request["target_new"]["str"], return_tensors="pt").to(self.device)["input_ids"][0]
        if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
            target_ids = target_ids[1:]
        if len(target_ids) == 0:
            raise TargetBoundary("EMPTY_TOKENIZED_TARGET")
        rewrite = [context.format(request["prompt"]) + tok.decode(target_ids[:-1])
                   for group in self.contexts for context in group]
        prompts = rewrite + ["{} is a"]
        input_tok = tok([p.format(request["subject"]) for p in prompts],
                        return_tensors="pt", padding=True).to(self.device)
        targets = torch.tensor(-100, device=self.device).repeat(
            len(rewrite), *input_tok["input_ids"].shape[1:])
        for i in range(len(rewrite)):
            ex_len = input_tok["attention_mask"][i].sum()
            targets[i, ex_len-len(target_ids):ex_len] = target_ids
        lookup = [self.native.find_fact_lookup_idx(p, request["subject"], tok,
                  hp.fact_token, verbose=False) for p in prompts]
        hidden = getattr(self.model.config, "n_embd", None)
        if hidden is None:
            hidden = self.model.config.hidden_size
        u = torch.zeros((hidden,), requires_grad=True, device=self.device)
        opt = torch.optim.Adam([u], lr=hp.v_lr)
        return RequestTargetState(request_index, request, input_tok, target_ids,
                                  targets, len(rewrite), lookup, u, opt, self)

    def run_chunk(self, state, max_updates, chunk_index):
        if state.owner is not self or chunk_index != state.chunks:
            raise TargetBoundary("STATE_OWNER_OR_CHUNK_ORDER")
        if state.u.dtype != torch.float32 or not state.u.is_leaf:
            raise TargetBoundary("U_LEAF_FP32")
        hp, model = self.hp, self.model
        target_layer = hp.layer_module_tmp.format(self.layer)
        loss_layer = hp.layer_module_tmp.format(max(hp.v_loss_layer, self.layer))
        state.aj = None
        leaf_identity, opt_identity = id(state.u), id(state.opt)
        initial_adam = state.opt.state.get(state.u, {})
        moment_ptrs = {k: initial_adam[k].data_ptr() for k in ("exp_avg", "exp_avg_sq") if k in initial_adam}
        initial_opt = {k: _cpu(v) for k, v in initial_adam.items() if torch.is_tensor(v)}
        initial_u = _cpu(state.u)
        hook_zero_offset_native_leaf = None

        def edit_output(cur_out, cur_layer):
            nonlocal hook_zero_offset_native_leaf
            if cur_layer == target_layer:
                # Exact native clean-sentence indexing, before any intervention.
                if state.aj is None:
                    state.aj = cur_out[0][0, state.lookup_idxs[0]].detach().clone()
                    if state.a0 is None:
                        state.a0 = state.aj.detach().clone()
                delta, hook_zero_offset_native_leaf = rebased_hook_delta(state.u, state.a0, state.aj)
                for i, idx in enumerate(state.lookup_idxs):
                    if len(state.lookup_idxs) != len(cur_out[0]):
                        cur_out[0][idx, i, :] += delta
                    else:
                        cur_out[0][i, idx, :] += delta
            return cur_out

        def loss_fn():
            with self.nethook.TraceDict(module=model, layers=[loss_layer, target_layer],
                    retain_input=False, retain_output=True, edit_output=edit_output) as tr:
                logits = model(**state.input_tok).logits
                kl_logits = torch.stack([logits[-1, state.lookup_idxs[-1], :]], dim=0)
                kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
                if state.teacher is None:
                    state.teacher = kl_log_probs.detach().clone()
            output = tr[loss_layer].output[0]
            if output.shape[1] != state.rewriting_targets.shape[1]:
                output = torch.transpose(output, 0, 1)
            full_repr = output[:state.rewriting_count]
            log_probs = torch.log_softmax(self.ln_f(full_repr) @ self.lm_w.to(full_repr.device)
                         + self.lm_b.to(full_repr.device), dim=2)
            gathered = torch.gather(log_probs, 2, torch.where(state.rewriting_targets != -100,
                          state.rewriting_targets, 0).unsqueeze(2).to(log_probs.device)).squeeze(2)
            mask = (state.rewriting_targets != -100).float()
            nll_each = -(gathered * mask.to(gathered.device)).sum(1) / state.target_ids.size(0)
            nll = nll_each.mean()
            kl = native_kl(state.teacher, kl_log_probs, hp.kl_factor)
            reg = native_regularizer(state.u, state.a0, hp.v_weight_decay)
            total = nll + kl.to(nll.device) + reg.to(nll.device)
            return total, dict(nll=float(nll.detach().item()), kl=float(kl.detach().item()),
                               regularizer=float(reg.detach().item()),
                               nll_each=nll_each.detach().cpu().tolist())

        flags = [(p, p.requires_grad) for p in model.parameters()]
        begin = time.monotonic()
        try:
            self.nethook.set_requires_grad(False, model)
            result = advance_adam(state.u, state.opt, loss_fn, max_updates=max_updates,
                                 max_norm_fn=lambda: hp.clamp_norm_factor * state.a0.norm())
        finally:
            for p, flag in flags:
                p.requires_grad_(flag)
        if id(state.u) != leaf_identity or id(state.opt) != opt_identity:
            raise TargetBoundary("OPTIMIZER_REPLACED")
        for k, ptr in moment_ptrs.items():
            if state.opt.state[state.u][k].data_ptr() != ptr:
                raise TargetBoundary("ADAM_MOMENT_REPLACED")
        state.chunks += 1
        state.actual_updates += result["actual_adam_updates"]
        state.loss_evaluations += result["target_loss_evaluations"]
        state.clamp_hits += result["clamp_hits"]
        target = state.absolute_target()
        if not torch.isfinite(target).all().item():
            raise TargetBoundary("NONFINITE_ABSOLUTE_TARGET")
        summary = {k: v for k, v in result.items() if k != "losses"}
        summary.update(request_index=state.request_index, chunk_index=chunk_index,
                       max_updates=max_updates, seconds=time.monotonic()-begin,
                       optimizer_object_preserved=True, u_leaf_preserved=True,
                       hook_zero_offset_native_leaf=hook_zero_offset_native_leaf,
                       capture_anchor_extra_forwards=0, target_mode="entry-offset-carry",
                       clamp_max_norm=float((hp.clamp_norm_factor * state.a0.norm()).item()))
        state.summaries.append(deepcopy(summary))
        capture_begin = time.monotonic()
        evidence = snapshot_state(state)
        evidence.update(initial_u=initial_u, initial_adam=initial_opt,
                        losses=result["losses"], summary=deepcopy(summary))
        summary["capture_cpu_seconds"] = time.monotonic()-capture_begin
        return dict(target=target, evidence=evidence, summary=summary)
