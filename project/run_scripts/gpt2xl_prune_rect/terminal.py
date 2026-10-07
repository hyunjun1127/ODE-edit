"""Native PRUNE spectral map with the explicitly authorized cold-base repair.

No fitting, rank truncation, new cutoff or formula adaptation occurs here.
The native source used currentW + compressed(currentW-W0); this task writes
saved coldW0 + compressedD exactly once after the twentieth dense commit.
"""
import hashlib
import time

import torch


REPAIR = 'PRUNE_TERMINAL_BASE_FIX'
OLD_LINE = 'adjusted_weight = original_weight + upd_matrix[k]'
NEW_LINE = 'adjusted_weight = saved_cold_weight + upd_matrix[k]'


def require(condition, label):
    if not condition:
        raise RuntimeError(label)


def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    # Match the existing commit/state convention. No source is frozen and no
    # PRUNE/RECT job has run yet; these hashes can therefore bind directly to
    # cold_W/native_after/after without a second large CPU tensor copy.
    result = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    result.update(memoryview(value.numpy()).cast('B'))
    return result.hexdigest()


def compressed_delta(cold, dense):
    """Actual native torch.svd/where/log/operator order, stored Conv1D direction."""
    require(cold.ndim == dense.ndim == 2 and cold.shape == dense.shape
            and cold.dtype == dense.dtype == torch.float32
            and cold.device == dense.device
            and torch.isfinite(cold).all().item() and torch.isfinite(dense).all().item(),
            'PRUNE_NATIVE_INPUT_SHAPE_DTYPE_FINITE')
    update = dense - cold
    _, S_orig, _ = torch.svd(cold)
    max_sigma = S_orig.max().item()
    require(max_sigma > 0 and torch.isfinite(S_orig).all().item(),
            'PRUNE_ORIGINAL_SPECTRUM_POSITIVE_FINITE')
    U_upd, S_upd, V_upd = torch.svd(update)
    adjusted_S = torch.where(
        S_upd > max_sigma,
        torch.log(S_upd) - torch.log(torch.tensor(max_sigma, device=cold.device)) + max_sigma,
        S_upd,
    )
    compressed = torch.matmul(U_upd, torch.matmul(torch.diag(adjusted_S), V_upd.t()))
    require(tuple(U_upd.shape) == (cold.shape[0], min(cold.shape))
            and tuple(V_upd.shape) == (cold.shape[1], min(cold.shape))
            and torch.isfinite(S_upd).all().item() and torch.isfinite(adjusted_S).all().item()
            and torch.isfinite(compressed).all().item(), 'PRUNE_NATIVE_SVD_PAYLOAD_FINITE_SHAPE')
    return compressed, dict(max_sigma=max_sigma,
        singular_count=S_upd.numel(), transformed_singular_count=int((S_upd > max_sigma).sum().item()),
        delta_norm=float(torch.linalg.vector_norm(update).item()),
        compressed_delta_norm=float(torch.linalg.vector_norm(compressed).item()),
        largest_original_singular_value=max_sigma,
        largest_dense_update_singular_value=float(S_upd.max().item()),
        largest_compressed_update_singular_value=float(adjusted_S.max().item()),
        operator='native torch.svd, s>smax log(s)-log(smax)+smax, U@(diag(S)@V.T)',
        dtype=str(cold.dtype), stored_weight_shape=list(cold.shape))


def apply_terminal(weights, cold, *, expected_shape=(6400, 1600)):
    """Selected weights only; caller owns RAM transaction/nonselected guards."""
    require(set(weights) == set(cold) and len(weights) == 5, 'PRUNE_FIVE_COLD_SELECTED_WEIGHTS')
    updates, rows, started = {}, {}, time.monotonic()
    # Native CLI computes all five compressed updates before the write phase.
    # Preserve that ordering rather than partially committing during SVD.
    with torch.no_grad():
        for name, weight in weights.items():
            original = cold[name].to(weight.device)
            require(tuple(weight.shape) == tuple(original.shape) == tuple(expected_shape),
                    'PRUNE_STORED_NATIVE_CONV1D_SHAPE')
            before_hash = tensor_sha(weight)
            cold_hash = tensor_sha(original)
            update, row = compressed_delta(original, weight)
            updates[name] = update
            rows[name] = dict(row, dense_weight_sha256=before_hash,
                             cold_weight_sha256=cold_hash,
                             compressed_delta_sha256=tensor_sha(update),
                             dense_weight_norm=float(torch.linalg.vector_norm(weight).item()))
        for name, update in updates.items():
            weight = weights[name]
            saved_cold_weight = cold[name].to(weight.device)
            # The only scientific-result difference from the pinned native CLI:
            # use saved cold W0, not already updated current W20, as the base.
            adjusted_weight = saved_cold_weight + update
            require(torch.isfinite(adjusted_weight).all().item(), 'PRUNE_TERMINAL_WEIGHT_NONFINITE')
            weight.copy_(adjusted_weight)
            rows[name].update(final_weight_sha256=tensor_sha(weight),
                              final_weight_norm=float(torch.linalg.vector_norm(weight).item()),
                              exact_copy_verified=bool(torch.equal(weight, adjusted_weight)))
    updates.clear()
    return dict(status='PRUNE_TERMINAL_APPLIED', repair=REPAIR,
                upstream_bitwise_equivalence=False, repair_authorized=True,
                original_line=OLD_LINE, repaired_line=NEW_LINE,
                repair_diff='- ' + OLD_LINE + '\n+ ' + NEW_LINE,
                terminal_transforms=1, numerical_dtype='native FP32', layers=rows,
                seconds=time.monotonic() - started, checkpoint_saved=False,
                spectrum_formula_changed=False, final_base='SAVED_COLD_W0',
                tensor_hash_convention='sha256(str((tuple(shape),str(dtype))).encode()+contiguous raw bytes)')
