"""Full-weight all-token L4 cut for ENFC, pinned to Llama/transformers 4.44.2.

This is NOT the SL-ZFlow ``writer @ K`` coordinate cache.  Each cache keeps raw
FP32 down-projection inputs and the exact residual immediately before the MLP
addition.  An *absolute, already rounded* weight is evaluated by F.linear(K,W)
and residual addition, in the same order as the physical decoder.  No teacher,
native fit, projector, target, history, optimizer or checkpoint is made here.

All caches reside on CPU.  Each document/sequence is a separate packed batch
of size one, including its padding.  Full-vocabulary logits are produced only
for a bounded position chunk.  The teacher callback is caller-owned and must
return fixed W0 FP32 log-probabilities; current endpoints never make teachers.

Pointer/version guards are runtime mutation checks, NOT whole-model byte
hashes.  Actual 8B stationarity/parity/derivative validation remains mandatory.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import time
from typing import Callable, Iterable, Mapping

import torch
import torch.nn.functional as F

WEIGHT = "model.layers.4.mlp.down_proj.weight"
PACK_FIELDS = ("input_ids", "attention_mask", "position_ids")


def tensor_sha256(value):
    """Header-and-bytes SHA (not interchangeable with a raw-bytes digest)."""
    value = value.detach().cpu().contiguous()
    header = f"{tuple(value.shape)}|{value.dtype}|".encode("ascii")
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def model_guard(model):
    """Cheap parameter/buffer/hook/mode identity; deliberately no byte claim."""
    tensors = tuple((kind, n, p.data_ptr(), p._version, tuple(p.shape),
                     str(p.dtype), str(p.device), bool(p.requires_grad))
                    for kind, items in (("parameter", model.named_parameters()),
                                        ("buffer", model.named_buffers()))
                    for n, p in items)
    modules = tuple((n, m.training,
                     tuple((k, id(v)) for k, v in m._forward_hooks.items()),
                     tuple((k, id(v)) for k, v in m._forward_pre_hooks.items()),
                     tuple((k, id(v)) for k, v in m._backward_hooks.items()))
                    for n, m in model.named_modules())
    return tensors, modules


def signed_forward_kl(logits, teacher_logp):
    """Full-vocab FP32 log-softmax, then signed FP64 sum/position mean.

    Neither the probability mass nor a negative roundoff result is clamped or
    re-normalized.  Loss-floor classification belongs to the caller's lock.
    """
    if logits.ndim != 2 or teacher_logp.shape != logits.shape:
        raise ValueError("FULL_VOCAB_TEACHER_SHAPE")
    if logits.dtype != torch.float32 or teacher_logp.dtype != torch.float32:
        raise ValueError("FP32_LOGITS_AND_TEACHER_REQUIRED")
    if not torch.isfinite(logits).all() or not torch.isfinite(teacher_logp).all():
        raise FloatingPointError("NONFINITE_LOGITS_OR_TEACHER")
    teacher = teacher_logp.detach().double()
    current = logits.log_softmax(-1).double()
    return (teacher.exp() * (teacher - current)).sum(-1).mean()


@dataclass(frozen=True)
class FullTokenCache:
    packed: dict
    keys: torch.Tensor                 # CPU [1, padded_length, intermediate]
    residual: torch.Tensor             # CPU [1, padded_length, hidden]
    input_identity: dict

    def valid_keys(self):
        """[intermediate, valid positions], preserving original token order."""
        return self.keys[self.packed["attention_mask"].bool()].T.contiguous()

    def valid_positions(self):
        return self.packed["attention_mask"][0].bool().nonzero().flatten()


class FullWeightLlamaOracle:
    """Absolute-W oracle, all other model parameters frozen.

    ``teacher_loader(index, cache)`` returns CPU FP32 [128, vocabulary] by
    default.  ``kl(..., gradient=True)`` returns (mean, FP64 CPU gradient, rows).
    ``route='physical'`` is an independent real selected-Parameter leaf/full
    decoder route, with exact-copy restoration even on failure.  It is for T,
    not a shortcut through the cache.  ``route='cached'`` differentiates W in
    F.linear(raw_keys,W), never low-dimensional writer coordinates.

    Non-default score_slice/require_reference_length are exposed for tiny CPU
    fixtures only; production callers must keep (128,256)/257 and lock them.
    """
    def __init__(self, model, packed_batches: Iterable[Mapping], *,
                 teacher_loader: Callable | None = None,
                 score_slice=(128, 256), require_reference_length=257,
                 head_chunk_positions=16):
        import transformers
        if transformers.__version__ != "4.44.2":
            raise ValueError("PINNED_TRANSFORMERS_4_44_2_REQUIRED")
        if model.config.model_type != "llama" or model.config.pretraining_tp != 1:
            raise ValueError("LLAMA_SINGLE_TP_REQUIRED")
        if model.config._attn_implementation != "eager" or model.training:
            raise ValueError("EAGER_EVAL_REQUIRED")
        if len(model.model.layers) < 5:
            raise ValueError("PHYSICAL_LAYER4_REQUIRED")
        if any(p.requires_grad or p.dtype != torch.float32
               for p in model.parameters()):
            raise ValueError("FROZEN_FULL_FP32_REQUIRED")
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise ValueError("TF32_MATMUL_AND_CUDNN_MUST_BE_OFF")
        if model.model.layers[4].mlp.down_proj.bias is not None:
            raise ValueError("LLAMA_BIASLESS_DOWN_PROJ_REQUIRED")
        self.model, self.decoder = model, model.model
        self.parameter = dict(model.named_parameters())[WEIGHT]
        self.device = self.parameter.device
        if any(p.device != self.device for p in model.parameters()):
            raise ValueError("SINGLE_DEVICE_MODEL_REQUIRED")
        self.shape = tuple(self.parameter.shape)
        self.teacher_loader = teacher_loader
        self.score_slice = tuple(score_slice)
        if not (len(self.score_slice) == 2 and
                0 <= self.score_slice[0] < self.score_slice[1]):
            raise ValueError("SCORE_SLICE")
        if head_chunk_positions < 1:
            raise ValueError("HEAD_CHUNK_POSITIONS_POSITIVE")
        self.require_reference_length = require_reference_length
        self.head_chunk_positions = int(head_chunk_positions)
        self.guard = model_guard(model)
        self.events = []
        self.work = {k: 0 for k in (
            "prefix_forwards", "prefix_input_tokens", "prefix_valid_tokens",
            "cached_suffix_forwards", "physical_full_forwards",
            "cached_backward_documents", "physical_backward_documents",
            "cached_input_tokens", "physical_input_tokens",
            "head_positions", "teacher_documents_read", "teacher_bytes_read",
            "kl_documents", "kl_scored_positions", "kl_input_tokens",
            "gradient_sweeps", "kl_sweeps", "physical_weight_installs",
            "physical_exact_restores", "comparison_logit_elements")}
        self.work.update(prefix_seconds=0., cache_transfer_seconds=0.,
                         cached_suffix_seconds=0., physical_full_seconds=0.,
                         backward_seconds=0., teacher_read_seconds=0.,
                         physical_copy_restore_seconds=0.)
        self.caches = [self._prepare(packed) for packed in packed_batches]
        if not self.caches:
            raise ValueError("EMPTY_CACHE_LIST")
        self.cache_guard = self._cache_guard()
        self._guard()

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _guard(self):
        if model_guard(self.model) != self.guard:
            raise RuntimeError("FROZEN_MODEL_PARAMETER_BUFFER_HOOK_OR_MODE_MUTATED")
        if any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("UNEXPECTED_MODEL_PARAMETER_GRAD")
        if hasattr(self, "cache_guard") and self._cache_guard() != self.cache_guard:
            raise RuntimeError("CPU_INPUT_KEY_OR_RESIDUAL_CACHE_MUTATED")

    def _cache_guard(self):
        return tuple((index, name, value.data_ptr(), value._version,
                      tuple(value.shape), str(value.dtype), str(value.device))
                     for index, cache in enumerate(self.caches)
                     for name, value in (*cache.packed.items(),
                                         ("keys", cache.keys),
                                         ("residual", cache.residual)))

    @staticmethod
    def _pack(source):
        if set(PACK_FIELDS) - set(source):
            raise ValueError("PACK_FIELDS_REQUIRED")
        packed = {k: source[k].detach().cpu().contiguous().clone()
                  for k in PACK_FIELDS}
        ids, mask, pos = (packed[k] for k in PACK_FIELDS)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.long:
            raise ValueError("PACK_ONE_SEQUENCE_INT64_IDS_REQUIRED")
        if ids.shape != mask.shape or pos.shape != ids.shape or pos.dtype != torch.long:
            raise ValueError("PACK_MASK_POSITION_SHAPE")
        if not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.any()):
            raise ValueError("PACK_BINARY_NONEMPTY_MASK_REQUIRED")
        if bool((pos < 0).any()) or bool((ids < 0).any()):
            raise ValueError("PACK_NEGATIVE_ID_OR_POSITION")
        return packed

    def _on_device(self, packed):
        return {k: v.to(self.device) for k, v in packed.items()}

    def _args(self, hidden, packed):
        cache_position = torch.arange(hidden.shape[1], device=self.device)
        mask = self.decoder._update_causal_mask(packed["attention_mask"], hidden,
                                               cache_position, None, False)
        rotary = self.decoder.rotary_emb(hidden, packed["position_ids"])
        return dict(attention_mask=mask, position_ids=packed["position_ids"],
                    past_key_value=None, output_attentions=False, use_cache=False,
                    cache_position=cache_position, position_embeddings=rotary)

    @torch.no_grad()
    def _prepare(self, source):
        self._guard()
        packed_cpu = self._pack(source)
        packed = self._on_device(packed_cpu)
        self._sync()
        started = time.perf_counter()
        hidden = self.decoder.embed_tokens(packed["input_ids"])
        args = self._args(hidden, packed)
        key, residual = [], []
        layer = self.decoder.layers[4]
        handles = [
            layer.mlp.down_proj.register_forward_pre_hook(
                lambda module, inputs: key.append(inputs[0].detach())),
            layer.post_attention_layernorm.register_forward_pre_hook(
                lambda module, inputs: residual.append(inputs[0].detach()))]
        try:
            for block in self.decoder.layers[:5]:
                hidden = block(hidden, **args)[0]
        finally:
            for handle in handles:
                handle.remove()
        if len(key) != 1 or len(residual) != 1:
            raise RuntimeError("ALL_TOKEN_KEY_RESIDUAL_CAPTURE_COUNT")
        self._sync()
        self.work["prefix_seconds"] += time.perf_counter() - started
        transfer = time.perf_counter()
        k = key[0].cpu().contiguous().clone()
        r = residual[0].cpu().contiguous().clone()
        if k.shape != (*packed_cpu["input_ids"].shape, self.shape[1]):
            raise RuntimeError("RAW_FULL_KEY_SHAPE")
        if r.shape != (*packed_cpu["input_ids"].shape, self.shape[0]):
            raise RuntimeError("PRE_MLP_RESIDUAL_SHAPE")
        if not torch.isfinite(k).all() or not torch.isfinite(r).all():
            raise FloatingPointError("NONFINITE_PREFIX_CACHE")
        identity = {k: tensor_sha256(v) for k, v in packed_cpu.items()}
        identity.update(keys_sha256=tensor_sha256(k), residual_sha256=tensor_sha256(r),
                        valid_tokens=int(packed_cpu["attention_mask"].sum()),
                        padded_tokens=packed_cpu["input_ids"].numel())
        self._sync()
        self.work["cache_transfer_seconds"] += time.perf_counter() - transfer
        self.work["prefix_forwards"] += 1
        self.work["prefix_input_tokens"] += packed_cpu["input_ids"].numel()
        self.work["prefix_valid_tokens"] += int(packed_cpu["attention_mask"].sum())
        self._guard()
        return FullTokenCache(packed_cpu, k, r, identity)

    def _weight(self, weight):
        if tuple(weight.shape) != self.shape or weight.dtype != torch.float32:
            raise ValueError("ABSOLUTE_FP32_FULL_WEIGHT_REQUIRED")
        if not torch.isfinite(weight).all():
            raise FloatingPointError("NONFINITE_CANDIDATE_WEIGHT")
        return weight.to(self.device)

    def capture_keys(self, index, *, valid_only=True):
        """Return an owned CPU tensor, never a writer-projected key."""
        cache = self.caches[index]
        return cache.valid_keys() if valid_only else cache.keys.clone()

    def acknowledge_selected_write(self, expected_weight):
        """Explicitly rebind after a caller-owned, intentional L4-only copy.

        Never automatically called when a guard fails.  The actual parameter
        must equal the caller's full expected FP32 tensor; all nonselected
        parameters/buffers/hooks/modes and CPU caches must still match.  Raw K
        and pre-MLP residual are independent of this same layer's down weight;
        this algebraic reuse does NOT assign a model-level stationarity PASS.
        """
        if expected_weight.dtype != torch.float32 or tuple(expected_weight.shape) != self.shape:
            raise ValueError("EXPECTED_SELECTED_FP32_WEIGHT_SHAPE")
        if not torch.isfinite(expected_weight).all():
            raise FloatingPointError("NONFINITE_EXPECTED_SELECTED_WEIGHT")
        after = model_guard(self.model)
        nonselected = lambda g: (tuple(x for x in g[0] if x[1] != WEIGHT), g[1])
        if nonselected(after) != nonselected(self.guard):
            raise RuntimeError("SELECTED_REBIND_NONSELECTED_MUTATION")
        if self.parameter.requires_grad or any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("SELECTED_REBIND_MODEL_NOT_FROZEN")
        if self._cache_guard() != self.cache_guard:
            raise RuntimeError("CPU_INPUT_KEY_OR_RESIDUAL_CACHE_MUTATED")
        if not torch.equal(self.parameter.detach().cpu(), expected_weight.detach().cpu()):
            raise RuntimeError("SELECTED_REBIND_EXPECTED_BYTES_DIFFER")
        self.guard = after
        receipt = dict(event="caller_owned_selected_write_rebind",
                       expected_weight_sha256=tensor_sha256(expected_weight),
                       nonselected_pointer_version_unchanged=True,
                       expected_selected_bytes_equal=True,
                       actual_stationarity_pass_not_assigned=True)
        self.events.append(receipt)
        self._guard()
        return receipt

    def suffix_hidden(self, index, weight):
        """Differentiable absolute-W path. Output is final normalized hidden."""
        self._guard()
        weight = self._weight(weight)
        cache = self.caches[index]
        packed = self._on_device(cache.packed)
        self._sync()
        started = time.perf_counter()
        hidden = cache.residual.to(self.device) + F.linear(cache.keys.to(self.device), weight)
        args = self._args(hidden, packed)
        for layer in self.decoder.layers[5:]:
            hidden = layer(hidden, **args)[0]
        hidden = self.decoder.norm(hidden)
        self._sync()
        self.work["cached_suffix_seconds"] += time.perf_counter() - started
        self.work["cached_suffix_forwards"] += 1
        self.work["cached_input_tokens"] += cache.packed["input_ids"].numel()
        return hidden

    def _physical_hidden(self, index):
        packed = self._on_device(self.caches[index].packed)
        self._sync()
        started = time.perf_counter()
        hidden = self.decoder(**packed, use_cache=False, output_attentions=False,
                              output_hidden_states=False, return_dict=True).last_hidden_state
        self._sync()
        self.work["physical_full_seconds"] += time.perf_counter() - started
        self.work["physical_full_forwards"] += 1
        self.work["physical_input_tokens"] += packed["input_ids"].numel()
        return hidden

    @contextmanager
    def physical_weight(self, weight, *, gradient=False):
        """Independent actual Parameter leaf; restore by copy, never +/-delta.

        A caller must finish any backward before leaving this context.  Only
        selected parameter versions change.  All other parameters, buffers,
        hooks, module training flags and requires_grad flags remain guarded.
        """
        self._guard()
        weight = self._weight(weight)
        before = self.guard
        saved = self.parameter.detach().cpu().clone()
        self._sync()
        started = time.perf_counter()
        try:
            with torch.no_grad():
                self.parameter.copy_(weight)
            if not torch.equal(self.parameter.detach(), weight):
                raise RuntimeError("PHYSICAL_WEIGHT_INSTALL_NOT_EXACT")
            self.parameter.requires_grad_(gradient)
            self.work["physical_weight_installs"] += 1
            self._sync()
            self.work["physical_copy_restore_seconds"] += time.perf_counter() - started
            yield self.parameter
        finally:
            self._sync()
            restore_started = time.perf_counter()
            self.parameter.requires_grad_(False)
            with torch.no_grad():
                self.parameter.copy_(saved.to(self.device))
            if not torch.equal(self.parameter.detach().cpu(), saved):
                raise RuntimeError("PHYSICAL_WEIGHT_RESTORE_NOT_EXACT")
            after = model_guard(self.model)
            nonselected = lambda g: (tuple(x for x in g[0] if x[1] != WEIGHT), g[1])
            if nonselected(after) != nonselected(before):
                raise RuntimeError("PHYSICAL_ROUTE_NONSELECTED_MUTATION")
            self.guard = after
            self.work["physical_exact_restores"] += 1
            self._sync()
            self.work["physical_copy_restore_seconds"] += time.perf_counter() - restore_started
            self._guard()

    def hidden(self, index, weight, *, route="cached"):
        """No-loss inference hidden. For gradients use suffix_hidden or kl."""
        with torch.no_grad():
            if route == "cached":
                return self.suffix_hidden(index, weight)
            if route == "physical":
                with self.physical_weight(weight):
                    return self._physical_hidden(index)
        raise ValueError("UNKNOWN_ROUTE")

    def _head(self, hidden, positions):
        positions = positions.to(self.device, dtype=torch.long)
        if positions.ndim != 1 or not positions.numel():
            raise ValueError("NONEMPTY_POSITION_VECTOR_REQUIRED")
        if bool(((positions < 0) | (positions >= hidden.shape[1])).any()):
            raise ValueError("LOGIT_POSITION_OUT_OF_BOUNDS")
        logits = self.model.lm_head(hidden[0, positions]).float()
        self.work["head_positions"] += positions.numel()
        return logits

    def logits_from_hidden(self, hidden, positions):
        """Reuse one suffix result for several caller-owned label panels.

        Returns full-vocabulary FP32 logits at explicit positions.  For all
        valid tokens use iter_logits so storage is position-chunk bounded.
        """
        return self._head(hidden, torch.as_tensor(positions, dtype=torch.long))

    def logits_at(self, index, weight, positions, *, route="cached"):
        """Inference helper: full-vocabulary logits at requested positions."""
        with torch.no_grad():
            return self.logits_from_hidden(self.hidden(index, weight, route=route), positions)

    def iter_logits(self, index, weight, *, positions=None, route="cached"):
        """Yield (CPU position IDs, full-vocab GPU logits), one bounded chunk.

        No entire document/full-vocab array is retained by the adapter.  Caller
        must consume/discard each chunk; iter_logits is inference only.
        """
        hidden = self.hidden(index, weight, route=route)
        if positions is None:
            positions = self.caches[index].valid_positions()
        with torch.no_grad():
            for part in positions.split(self.head_chunk_positions):
                yield part.cpu(), self._head(hidden, part)
        self._guard()

    def sequence_nll(self, index, weight, positions, labels, *, route="cached"):
        """Caller-supplied canonical/native TF labels; no target inference.

        positions are logit indices, not input label indices.  Mean is over
        exactly the supplied target tokens; request/context reduction belongs
        to the caller.  Native tie convention is torch.argmax (lowest ID).
        """
        positions = torch.as_tensor(positions, dtype=torch.long)
        labels = torch.as_tensor(labels, dtype=torch.long)
        if positions.shape != labels.shape or not labels.numel():
            raise ValueError("TARGET_POSITION_LABEL_SHAPE")
        if not bool(self.caches[index].packed["attention_mask"][0, positions].all()):
            raise ValueError("TARGET_LOGITS_INCLUDE_PADDING")
        hidden = self.hidden(index, weight, route=route)
        losses, correct, margins = [], [], []
        with torch.no_grad():
            for start in range(0, labels.numel(), self.head_chunk_positions):
                pos = positions[start:start+self.head_chunk_positions]
                target = labels[start:start+self.head_chunk_positions].to(self.device)
                logits = self._head(hidden, pos)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("NONFINITE_SEQUENCE_LOGITS")
                losses.extend(F.cross_entropy(logits, target, reduction="none").double().cpu().tolist())
                correct.extend(logits.argmax(-1).eq(target).cpu().tolist())
                chosen = logits.gather(1, target[:, None]).squeeze(1)
                alternatives = logits.clone()
                alternatives.scatter_(1, target[:, None], -torch.inf)
                margins.extend((chosen-alternatives.max(-1).values).double().cpu().tolist())
        self._guard()
        return dict(nll=sum(losses)/len(losses), token_nll=losses,
                    strict=all(correct), token_correct=correct, margins=margins,
                    tokens=len(losses), route=route)

    def kl(self, weight, *, gradient=False, route="cached", indices=None,
           teacher_loader=None):
        """Document-mean signed fixed-W0 KL; one full document graph at a time."""
        self._guard()
        if route not in ("cached", "physical"):
            raise ValueError("UNKNOWN_ROUTE")
        loader = teacher_loader or self.teacher_loader
        if loader is None:
            raise ValueError("FIXED_W0_TEACHER_CALLBACK_REQUIRED")
        selected = list(range(len(self.caches)) if indices is None else indices)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("NONEMPTY_UNIQUE_DOCUMENT_INDICES_REQUIRED")
        weight = self._weight(weight)
        accumulation = torch.zeros(self.shape, dtype=torch.float64) if gradient else None
        rows = []
        coordinate = weight.detach().requires_grad_(gradient)
        context = (self.physical_weight(weight, gradient=gradient)
                   if route == "physical" else nullcontext(coordinate))
        with context as leaf, torch.set_grad_enabled(gradient):
            for index in selected:
                cache = self.caches[index]
                count = cache.packed["input_ids"].numel()
                if self.require_reference_length is not None and count != self.require_reference_length:
                    raise ValueError("REFERENCE_DOCUMENT_LENGTH_MISMATCH")
                begin, end = self.score_slice
                if end > count or not bool(cache.packed["attention_mask"][0, begin:end].all()):
                    raise ValueError("REFERENCE_SCORED_RANGE_INVALID")
                started = time.perf_counter()
                teacher = loader(index, cache)
                self.work["teacher_read_seconds"] += time.perf_counter() - started
                if not isinstance(teacher, torch.Tensor) or teacher.dtype != torch.float32:
                    raise ValueError("TEACHER_CALLBACK_MUST_RETURN_FP32_TENSOR")
                if tuple(teacher.shape) != (end-begin, self.model.config.vocab_size):
                    raise ValueError("TEACHER_FULL_VOCAB_SCORED_SHAPE")
                teacher = teacher.detach().to(self.device)
                self.work["teacher_documents_read"] += 1
                self.work["teacher_bytes_read"] += teacher.numel()*teacher.element_size()
                hidden = (self.suffix_hidden(index, leaf) if route == "cached"
                          else self._physical_hidden(index))
                # Only128 positions, never full document logits.  Teacher and
                # all ephemeral FP64 KL arrays die before the next document.
                logits = self._head(hidden, torch.arange(begin, end))
                loss = signed_forward_kl(logits, teacher)
                if not torch.isfinite(loss):
                    raise FloatingPointError("NONFINITE_KL_LOSS")
                if gradient:
                    self._sync()
                    started = time.perf_counter()
                    grad, = torch.autograd.grad(loss, leaf)
                    if not torch.isfinite(grad).all():
                        raise FloatingPointError("NONFINITE_FULL_WEIGHT_GRADIENT")
                    accumulation.add_(grad.detach().double().cpu())
                    self._sync()
                    self.work["backward_seconds"] += time.perf_counter() - started
                    self.work[route+"_backward_documents"] += 1
                    del grad
                rows.append(dict(index=index, loss=float(loss.detach()),
                                 input_tokens=count, valid_tokens=int(cache.packed["attention_mask"].sum()),
                                 scored_positions=end-begin))
                self.work["kl_documents"] += 1
                self.work["kl_scored_positions"] += end-begin
                self.work["kl_input_tokens"] += count
                del hidden, logits, loss, teacher
        if gradient:
            accumulation.div_(len(selected))
            self.work["gradient_sweeps"] += 1
        self.work["kl_sweeps"] += 1
        mean = sum(row["loss"] for row in rows)/len(rows)
        self.events.append(dict(event="fixed_W0_forward_KL", route=route,
                                documents=len(rows), mean=mean, gradient=gradient,
                                gradient_norm=None if accumulation is None else float(accumulation.norm())))
        self._guard()
        return mean, accumulation, rows

    def compare_logits(self, index, left_weight, right_weight, *,
                       left_route="cached", right_route="physical"):
        """Streaming full-vocab comparison over EVERY valid input position."""
        left = self.hidden(index, left_weight, route=left_route)
        right = self.hidden(index, right_weight, route=right_route)
        stat = dict(max_abs=0., squared_error=0., squared_reference=0.,
                    logit_elements=0, valid_positions=0,
                    left_route=left_route, right_route=right_route)
        with torch.no_grad():
            for positions in self.caches[index].valid_positions().split(self.head_chunk_positions):
                l = self._head(left, positions)
                r = self._head(right, positions)
                if not torch.isfinite(l).all() or not torch.isfinite(r).all():
                    raise FloatingPointError("NONFINITE_COMPARISON_LOGITS")
                diff = r.double()-l.double()
                stat["max_abs"] = max(stat["max_abs"], float(diff.abs().max()))
                stat["squared_error"] += float(diff.square().sum())
                stat["squared_reference"] += float(l.double().square().sum())
                stat["logit_elements"] += diff.numel()
                stat["valid_positions"] += positions.numel()
        stat["rms"] = (stat["squared_error"]/stat["logit_elements"])**.5
        stat["relative_l2"] = (stat["squared_error"]/max(stat["squared_reference"], 1e-300))**.5
        self.work["comparison_logit_elements"] += stat["logit_elements"]
        self._guard()
        return stat

    @torch.no_grad()
    def key_stationarity(self, index, weight):
        """Re-capture physical keys at W; report observed error, never autoPASS."""
        captured = []
        with self.physical_weight(weight):
            handle = self.decoder.layers[4].mlp.down_proj.register_forward_pre_hook(
                lambda module, inputs: captured.append(inputs[0].detach().cpu().clone()))
            try:
                self._physical_hidden(index)
            finally:
                handle.remove()
        if len(captured) != 1:
            raise RuntimeError("STATIONARITY_CAPTURE_COUNT")
        old = self.caches[index].keys
        new = captured[0]
        diff = new.double()-old.double()
        return dict(max_abs=float(diff.abs().max()),
                    relative_l2=float(diff.norm()/old.double().norm().clamp_min(1e-300)),
                    byte_equal=torch.equal(new, old), keys_sha256=tensor_sha256(new),
                    scope="all padded and valid token keys for this fixed input",
                    technical_pass_not_assigned=True)
