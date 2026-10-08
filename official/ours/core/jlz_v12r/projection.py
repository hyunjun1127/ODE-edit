"""Exact capped Euclidean energy projection, in absolute request coordinates."""
import torch


def _require(ok, label):
    if not bool(ok):
        raise RuntimeError(label)


def project_capped_energy(blocks, caps, radii):
    """Project d_l x B blocks onto local balls intersected with a shared L2 ball.

    A sorted breakpoint partitions capped and uncapped blocks. In each interval
    the energy equation is quadratic in a single common theta; no iterative
    radial/clamp approximation or post-cast shrink is used.
    """
    layers = tuple(blocks)
    _require(bool(layers), 'PROJECTION_NO_LAYERS')
    first = blocks[layers[0]]
    B = first.shape[1]
    device = first.device
    cap = torch.as_tensor(caps, dtype=torch.float64, device=device)
    rho = torch.as_tensor(radii, dtype=torch.float64, device=device)
    _require(cap.shape == (len(layers), B) and rho.shape == (B,), 'PROJECTION_SHAPE')
    _require(torch.isfinite(cap).all() and torch.isfinite(rho).all(), 'PROJECTION_NONFINITE_BUDGET')
    _require((cap >= 0).all() and (rho >= 0).all(), 'PROJECTION_NEGATIVE_BUDGET')
    for value in blocks.values():
        _require(value.ndim == 2 and value.shape[1] == B and torch.isfinite(value).all(), 'PROJECTION_INPUT')
    xnorm = torch.stack([torch.linalg.vector_norm(blocks[l].double(), dim=0) for l in layers])
    clipped_energy = torch.minimum(xnorm, cap).square().sum(0)
    theta = torch.ones(B, dtype=torch.float64, device=device)
    # L <= adapter's eligible count, B arbitrary. Request-specific sorted active
    # sets, with no per-coordinate loop and all arithmetic in FP64.
    for r in range(B):
        if clipped_energy[r] <= rho[r].square():
            continue
        n, c = xnorm[:, r], cap[:, r]
        positive = n > 0
        breakpoints = torch.where(positive, c / torch.where(positive, n, torch.ones_like(n)), torch.full_like(n, float('inf')))
        order = torch.argsort(breakpoints, stable=True)
        ns, cs, bp = n[order], c[order], breakpoints[order]
        cap_energy = torch.zeros((), dtype=torch.float64, device=device)
        free_energy = ns.square().sum()
        lower = torch.zeros((), dtype=torch.float64, device=device)
        chosen = None
        for q in range(len(layers) + 1):
            upper = bp[q] if q < len(layers) else torch.full_like(lower, float('inf'))
            if free_energy > 0:
                remaining = rho[r].square() - cap_energy
                if remaining >= 0:
                    candidate = torch.sqrt(remaining / free_energy)
                    if candidate >= lower and candidate <= upper:
                        chosen = candidate
                        break
            if q < len(layers):
                cap_energy = cap_energy + cs[q].square()
                free_energy = ns[q + 1:].square().sum()
                lower = bp[q]
        _require(chosen is not None, 'PROJECTION_ACTIVE_SET_UNRESOLVED')
        theta[r] = chosen
    out = {}
    projected_norm = torch.minimum(cap, theta[None, :] * xnorm)
    for i, l in enumerate(layers):
        denom = torch.where(xnorm[i] > 0, xnorm[i], torch.ones_like(xnorm[i]))
        scale = torch.where(xnorm[i] > 0, projected_norm[i] / denom, torch.zeros_like(denom))
        out[l] = (blocks[l].double() * scale[None, :]).float()
    stored_norm = torch.stack([torch.linalg.vector_norm(out[l].double(), dim=0) for l in layers])
    local_excess = (stored_norm - cap).clamp_min(0)
    radius_excess = (stored_norm.square().sum(0).sqrt() - rho).clamp_min(0)
    squared_excess = (stored_norm.square().sum(0) - rho.square()).clamp_min(0)
    _require(torch.isfinite(theta).all() and (theta >= 0).all() and (theta <= 1).all(), 'PROJECTION_THETA')
    _require(local_excess.max() <= 1e-6 and radius_excess.max() <= 1e-6, 'FP32_STORED_BUDGET')
    return out, {
        'theta': theta.detach().cpu().tolist(),
        'pre_norm': xnorm.detach().cpu().tolist(),
        'post_norm': stored_norm.detach().cpu().tolist(),
        'local_excess_max': float(local_excess.max()),
        'radius_excess_max': float(radius_excess.max()),
        'squared_energy_excess': squared_excess.detach().cpu().tolist(),
        'squared_energy_units': 'absolute_request_norm_squared',
        'postcast_shrink': False,
    }
