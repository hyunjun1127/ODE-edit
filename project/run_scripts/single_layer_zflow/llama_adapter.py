"""Frozen Llama-4.44.2 all-token affine cut and selected-position full head.

No native z fitting or model/history update occurs in this module. The caller
owns the batch entry and must construct a NEW adapter after each physical write.
CPU-hosted caches hold every training token; only a microbatch is on the GPU.
"""
from dataclasses import dataclass
import time
import torch
import torch.nn.functional as F

WEIGHT = 'model.layers.4.mlp.down_proj.weight'


def model_guard(model):
    return tuple((n, p.data_ptr(), p._version, tuple(p.shape), str(p.dtype))
                 for n, p in model.named_parameters())


@dataclass
class LlamaCache:
    packed: dict
    h_entry: torch.Tensor
    a: torch.Tensor
    teacher_logp: torch.Tensor


class LlamaAffineOracle:
    """One logical global-weight sweep = fresh suffix F+B over all caches.

    X and model arithmetic are FP32. Scalar loss reductions and gradient
    accumulation across microbatches use FP64, then return the FP32 X gradient.
    The flow's analytic cost is deliberately excluded from this oracle.
    """
    def __init__(self, model, writer, packed_batches, *, beta=.0625):
        import transformers
        if transformers.__version__ != '4.44.2':
            raise ValueError('PINNED_TRANSFORMERS_4_44_2_REQUIRED')
        if model.config.model_type != 'llama' or model.config.pretraining_tp != 1:
            raise ValueError('LLAMA_SINGLE_TP_REQUIRED')
        if model.config._attn_implementation != 'eager' or model.training:
            raise ValueError('EAGER_EVAL_REQUIRED')
        if any(p.requires_grad or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('FROZEN_FULL_FP32_REQUIRED')
        if writer.dtype != torch.float32 or writer.ndim != 2:
            raise ValueError('FP32_WRITER_REQUIRED')
        self.model, self.decoder = model, model.model
        self.device = next(model.parameters()).device
        self.writer = writer.detach().to(self.device)
        self.shape = (model.config.hidden_size, writer.shape[0])
        self.beta = float(beta)
        self.guard = model_guard(model)
        self.caches, self.events = [], []
        self.work = dict(prefix_forward_microbatches=0, prefix_valid_tokens=0,
                         prefix_padded_tokens=0, teacher_suffix_microbatches=0,
                         teacher_suffix_tokens=0, teacher_head_positions=0,
                         oracle_calls=0, suffix_forward_microbatches=0,
                         suffix_backward_microbatches=0, suffix_valid_tokens=0,
                         suffix_padded_tokens=0, head_positions=0,
                         prefix_seconds=0.,teacher_suffix_head_seconds=0.,
                         cache_transfer_seconds=0.,suffix_forward_loss_seconds=0.,
                         suffix_backward_seconds=0.)
        started = time.perf_counter()
        for packed in packed_batches:
            self.caches.append(self._prepare(packed))
        self.preparation_seconds = time.perf_counter()-started
        edit_weight = sum(float(c.packed['edit_weights'].double().sum()) for c in self.caches)
        kl_weight = sum(float(c.packed['kl_weights'].double().sum()) for c in self.caches)
        if abs(edit_weight-1) > 1e-7 or abs(kl_weight-1) > 1e-7:
            raise ValueError('GLOBAL_LOGICAL_WEIGHTS_MUST_SUM_TO_ONE')
        self._guard()

    def _guard(self):
        if model_guard(self.model) != self.guard:
            raise RuntimeError('FROZEN_ENTRY_MODEL_MUTATED')
        if any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError('UNEXPECTED_PARAMETER_DENSE_GRADIENT')

    def _device_pack(self, packed):
        return {k: v.to(self.device) for k, v in packed.items()}

    def _sync(self):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)

    def _args(self, hidden, packed):
        cache_position = torch.arange(hidden.shape[1], device=self.device)
        pos = packed['position_ids']
        mask = self.decoder._update_causal_mask(packed['attention_mask'], hidden,
                                               cache_position, None, False)
        rotary = self.decoder.rotary_emb(hidden, pos)
        return dict(attention_mask=mask, position_ids=pos, past_key_value=None,
                    output_attentions=False, use_cache=False,
                    cache_position=cache_position, position_embeddings=rotary)

    def suffix_hidden(self, hidden, packed):
        args = self._args(hidden, packed)
        for layer in self.decoder.layers[5:]:
            hidden = layer(hidden, **args)[0]
        return self.decoder.norm(hidden)

    def selected_logits(self, hidden, packed):
        rows = torch.cat((packed['edit_rows'], packed['kl_rows']))
        cols = torch.cat((packed['edit_cols'], packed['kl_cols']))
        # FULL vocabulary, only sequence positions are selected. No vocabulary
        # truncation and no full [batch,sequence,vocabulary] tensor.
        return self.model.lm_head(hidden[rows, cols]).float()

    @torch.no_grad()
    def _prepare(self, source):
        self._sync();prefix_begin=time.perf_counter()
        packed = self._device_pack(source)
        hidden = self.decoder.embed_tokens(packed['input_ids'])
        args = self._args(hidden, packed)
        captured = []
        hook = self.decoder.layers[4].mlp.down_proj.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach()))
        try:
            for layer in self.decoder.layers[:5]:
                hidden = layer(hidden, **args)[0]
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError('ALL_TOKEN_KEY_CAPTURE_COUNT')
        a = captured.pop() @ self.writer.T
        self._sync();self.work['prefix_seconds']+=time.perf_counter()-prefix_begin
        teacher = torch.empty((0, self.model.config.vocab_size), device=self.device)
        if packed['kl_rows'].numel():
            teacher_begin=time.perf_counter()
            terminal = self.suffix_hidden(hidden, packed)
            teacher = self.model.lm_head(terminal[packed['kl_rows'], packed['kl_cols']]).float().log_softmax(-1)
            self.work['teacher_suffix_microbatches'] += 1
            self.work['teacher_suffix_tokens'] += hidden.shape[0]*hidden.shape[1]
            self.work['teacher_head_positions'] += len(teacher)
            self._sync();self.work['teacher_suffix_head_seconds']+=time.perf_counter()-teacher_begin
        self.work['prefix_forward_microbatches'] += 1
        self.work['prefix_valid_tokens'] += int(packed['attention_mask'].sum())
        self.work['prefix_padded_tokens'] += packed['input_ids'].numel()
        transfer_begin=time.perf_counter()
        result=LlamaCache({k: v.detach().cpu().clone() for k, v in source.items()},
                          hidden.detach().cpu(), a.detach().cpu(), teacher.detach().cpu())
        self._sync();self.work['cache_transfer_seconds']+=time.perf_counter()-transfer_begin
        return result

    def _loss(self, logits, packed, teacher):
        ne = len(packed['edit_labels'])
        edit = logits.new_zeros((), dtype=torch.float64)
        if ne:
            edit = (F.cross_entropy(logits[:ne], packed['edit_labels'], reduction='none').double()
                    * packed['edit_weights'].double()).sum()
        kl = logits.new_zeros((), dtype=torch.float64)
        if len(packed['kl_rows']):
            current = logits[ne:].log_softmax(-1)
            kl = (F.kl_div(teacher, current, log_target=True, reduction='none').double().sum(-1)
                  * packed['kl_weights'].double()).sum()
        return edit, kl

    def __call__(self, x):
        self._guard()
        if x.shape != self.shape or x.dtype != torch.float32 or x.device != self.device:
            raise ValueError('X_FP32_DEVICE_SHAPE')
        started = time.perf_counter()
        g = torch.zeros_like(x, dtype=torch.float64)
        forward_before=self.work['suffix_forward_loss_seconds']
        backward_before=self.work['suffix_backward_seconds']
        edit_sum = kl_sum = 0.
        coordinate = x.detach().requires_grad_(True)
        for cache in self.caches:
            self._sync();forward_begin=time.perf_counter()
            packed = self._device_pack(cache.packed)
            hidden = cache.h_entry.to(self.device) + cache.a.to(self.device) @ coordinate.T
            logits = self.selected_logits(self.suffix_hidden(hidden, packed), packed)
            edit, kl = self._loss(logits, packed, cache.teacher_logp.to(self.device))
            loss = edit + self.beta*kl
            if not torch.isfinite(loss):
                raise FloatingPointError('NONFINITE_ORACLE_LOSS')
            self._sync();self.work['suffix_forward_loss_seconds']+=time.perf_counter()-forward_begin
            backward_begin=time.perf_counter()
            grad, = torch.autograd.grad(loss, coordinate)
            self._sync();self.work['suffix_backward_seconds']+=time.perf_counter()-backward_begin
            g.add_(grad.double())
            edit_sum += float(edit.detach()); kl_sum += float(kl.detach())
            self.work['suffix_forward_microbatches'] += 1
            self.work['suffix_backward_microbatches'] += 1
            self.work['suffix_valid_tokens'] += int(packed['attention_mask'].sum())
            self.work['suffix_padded_tokens'] += packed['input_ids'].numel()
            self.work['head_positions'] += len(logits)
            del hidden, logits, edit, kl, loss, grad
        if not torch.isfinite(g).all():
            raise FloatingPointError('NONFINITE_ORACLE_GRADIENT')
        self.work['oracle_calls'] += 1
        self.events.append(dict(oracle=self.work['oracle_calls'], edit_nll=edit_sum,
                                essence_kl=kl_sum, L=edit_sum+self.beta*kl_sum,
                                seconds=time.perf_counter()-started,
                                forward_loss_seconds=self.work['suffix_forward_loss_seconds']-forward_before,
                                backward_seconds=self.work['suffix_backward_seconds']-backward_before,
                                x_norm=float(x.double().norm()), gradient_norm=float(g.norm())))
        self._guard()
        return edit_sum+self.beta*kl_sum, g.float()

    def parity(self, x, *, candidate_weight=None, backward=False, cache_indices=None):
        """Independent full decoder functional weight vs cached suffix.

        Exact pretrained Llama is required by the GPU technical runner. Tiny
        Llama fixtures only test wiring. No physical parameter is changed.
        Terminal uses the actually rounded FP32 candidate supplied by caller.
        """
        self._guard()
        names = range(len(self.caches)) if cache_indices is None else cache_indices
        entry = dict(self.model.named_parameters())[WEIGHT]
        stat = dict(max_logit_abs=0., logit_squared_error=0., logit_squared_reference=0.,
                    logits=0, cached_edit=0., actual_edit=0., cached_kl=0., actual_kl=0.)
        gc = torch.zeros_like(x, dtype=torch.float64)
        gf = torch.zeros_like(x, dtype=torch.float64)
        started = time.perf_counter()
        for index in names:
            cache = self.caches[index]; packed = self._device_pack(cache.packed)
            with torch.set_grad_enabled(backward):
                xc = x.detach().requires_grad_(backward)
                hc = cache.h_entry.to(self.device) + cache.a.to(self.device) @ xc.T
                lc = self.selected_logits(self.suffix_hidden(hc, packed), packed)
                ec, kc = self._loss(lc, packed, cache.teacher_logp.to(self.device))
                if backward:
                    v, = torch.autograd.grad(ec+self.beta*kc, xc); gc.add_(v.double())
                lc = lc.detach()
                xf = x.detach().requires_grad_(backward)
                wf = entry + xf @ self.writer if candidate_weight is None else candidate_weight.to(self.device)
                hf = torch.func.functional_call(self.decoder,
                     {'layers.4.mlp.down_proj.weight': wf}, (),
                     dict(input_ids=packed['input_ids'], attention_mask=packed['attention_mask'],
                          position_ids=packed['position_ids'], use_cache=False)).last_hidden_state
                lf = self.selected_logits(hf, packed)
                ef, kf = self._loss(lf, packed, cache.teacher_logp.to(self.device))
                if backward:
                    if candidate_weight is not None:
                        raise ValueError('BACKWARD_REQUIRES_DIFFERENTIABLE_FULL_WRITE')
                    v, = torch.autograd.grad(ef+self.beta*kf, xf); gf.add_(v.double())
                diff = (lf.detach()-lc).double()
                stat['max_logit_abs'] = max(stat['max_logit_abs'], float(diff.abs().max()))
                stat['logit_squared_error'] += float(diff.square().sum())
                stat['logit_squared_reference'] += float(lc.double().square().sum())
                stat['logits'] += lc.numel()
                for key, value in [('cached_edit',ec),('actual_edit',ef),('cached_kl',kc),('actual_kl',kf)]:
                    stat[key] += float(value.detach())
                del hc, lc, hf, lf, wf, ec, kc, ef, kf
        stat['logit_relative_l2'] = (stat['logit_squared_error']/max(stat['logit_squared_reference'],1e-300))**.5
        stat['edit_abs_error'] = abs(stat['cached_edit']-stat['actual_edit'])
        stat['kl_abs_error'] = abs(stat['cached_kl']-stat['actual_kl'])
        if backward:
            stat['gradient_relative_l2'] = float((gc-gf).norm()/gf.norm().clamp_min(1e-30))
            stat['gradient_max_abs'] = float((gc-gf).abs().max())
            stat['gradient_reference_norm'] = float(gf.norm())
        stat['seconds'] = time.perf_counter()-started
        self._guard()
        return stat

    @torch.no_grad()
    def physical_terminal_parity(self, x, candidate_weight):
        """Terminal-only actual Parameter copy, full forward, exact entry rollback.

        No history mutation. Unlike inner oracles this deliberately increments
        the owned selected parameter version; every other parameter is guarded.
        Cache authority is still the same byte-identical entry after rollback.
        """
        self._guard()
        parameter = dict(self.model.named_parameters())[WEIGHT]
        entry = parameter.detach().cpu().clone()
        before = self.guard
        out = dict(max_logit_abs=0., logit_squared_error=0., logit_squared_reference=0.,
                   logits=0, cached_edit=0., actual_edit=0., cached_kl=0., actual_kl=0.,
                   physical_parameter_copy=True, inner_history_append=0)
        started = time.perf_counter()
        try:
            parameter.copy_(candidate_weight.to(self.device))
            if not torch.equal(parameter.detach().cpu(), candidate_weight.cpu()):
                raise RuntimeError('PHYSICAL_CANDIDATE_BYTES_DIFFER')
            for cache in self.caches:
                packed = self._device_pack(cache.packed)
                h = cache.h_entry.to(self.device)+cache.a.to(self.device)@x.T
                expected = self.selected_logits(self.suffix_hidden(h,packed),packed)
                actual_h = self.decoder(input_ids=packed['input_ids'],
                         attention_mask=packed['attention_mask'],position_ids=packed['position_ids'],
                         use_cache=False).last_hidden_state
                actual = self.selected_logits(actual_h,packed)
                ec,kc = self._loss(expected,packed,cache.teacher_logp.to(self.device))
                ea,ka = self._loss(actual,packed,cache.teacher_logp.to(self.device))
                diff=(expected-actual).double()
                out['max_logit_abs']=max(out['max_logit_abs'],float(diff.abs().max()))
                out['logit_squared_error']+=float(diff.square().sum())
                out['logit_squared_reference']+=float(expected.double().square().sum())
                out['logits']+=expected.numel()
                for key,val in [('cached_edit',ec),('actual_edit',ea),('cached_kl',kc),('actual_kl',ka)]:
                    out[key]+=float(val)
        finally:
            parameter.copy_(entry.to(self.device))
            if not torch.equal(parameter.detach().cpu(),entry):
                raise RuntimeError('EXACT_ENTRY_ROLLBACK_FAILED')
            after=model_guard(self.model)
            if tuple(v for v in before if v[0]!=WEIGHT)!=tuple(v for v in after if v[0]!=WEIGHT):
                raise RuntimeError('NONSELECTED_PARAMETER_CHANGED')
            self.guard=after
        out.update(logit_relative_l2=(out['logit_squared_error']/max(out['logit_squared_reference'],1e-300))**.5,
                   edit_abs_error=abs(out['cached_edit']-out['actual_edit']),
                   kl_abs_error=abs(out['cached_kl']-out['actual_kl']),
                   entry_restored_exact=True,nonselected_pointer_version_unchanged=True,
                   seconds=time.perf_counter()-started)
        self._guard()
        return out
