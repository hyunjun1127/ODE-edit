"""Synthetic FP64 causal MEMIT reference; no model or production executor.

Every candidate builds all lower updates before extracting an upper mean key.
The same candidate weights generate the task logits and can be committed.
Analytic JVPs include the solve's dependence on those upper keys.
"""

from dataclasses import dataclass

import numpy as np


def array(value, name, ndim=2):
    value = np.array(value, dtype=np.float64, copy=True)
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite {ndim}-D array")
    return value


def check_spd(value, size):
    value = array(value, "A")
    if value.shape != (size, size):
        raise ValueError("A: wrong shape")
    if not np.array_equal(value, value.T):
        raise ValueError("A: reference requires symmetric input; no silent symmetrization")
    np.linalg.cholesky(value)  # No jitter, fallback, or silent eigenvalue repair.
    return value


def native_mean_operator(owners, weights, request_count):
    """Row weights already specify native means; KL rows may have weight zero.

    This function validates sums rather than silently renormalizing them.
    """
    owners = np.asarray(owners)
    weights = array(weights, "mean weights", ndim=1)
    if owners.dtype.kind not in "iu" or owners.shape != weights.shape:
        raise ValueError("owners: wrong integer shape")
    if request_count < 1 or np.any(owners < 0) or np.any(owners >= request_count):
        raise ValueError("owners: out of range")
    if np.any(weights < 0):
        raise ValueError("mean weights: negative")
    M = np.zeros((len(owners), request_count), dtype=np.float64)
    M[np.arange(len(owners)), owners] = weights
    if not np.allclose(M.sum(axis=0), 1, atol=1e-12, rtol=0):
        raise ValueError("mean weights: each owner's supplied weights must sum to one")
    return M


def memit_writer(R, K, A):
    R, K = array(R, "R"), array(K, "K")
    A = check_spd(A, K.shape[0])
    if R.shape[1] != K.shape[1]:
        raise ValueError("R and K request counts differ")
    system = A + K @ K.T
    P = np.linalg.solve(system, K)
    U = R @ P.T
    F = P.T @ A @ P
    return {"R": R, "K": K, "A": A, "system": system,
            "P": P, "U": U, "F": F,
            "Q_dense": float(np.sum((U @ A) * U)),
            "Q_compact": float(np.sum((R @ F) * R)),
            "response": U @ K,
            "residual_energy": float(np.sum((R - U @ K) ** 2))}


def writer_jvp(writer, dR, dK):
    """Differentiate R K^T (A+KK^T)^-1 with fixed entry A."""
    R, K, A, P = (writer[x] for x in ("R", "K", "A", "P"))
    dR, dK = array(dR, "dR"), array(dK, "dK")
    if dR.shape != R.shape or dK.shape != K.shape:
        raise ValueError("writer tangent shape")
    dsystem = dK @ K.T + K @ dK.T
    dP = np.linalg.solve(writer["system"], dK - dsystem @ P)
    dU = dR @ P.T + R @ dP.T
    dF = dP.T @ A @ P + P.T @ A @ dP
    dense = 2.0 * float(np.sum((writer["U"] @ A) * dU))
    compact = (2.0 * float(np.sum((R @ writer["F"]) * dR))
               + float(np.sum((R @ dF) * R)))
    return {"dP": dP, "dU": dU, "dQ_dense": dense, "dQ_compact": compact}


@dataclass
class ToyModel:
    inputs: np.ndarray
    weights: list
    metrics: list
    head: np.ndarray
    mean_operator: np.ndarray
    owners: np.ndarray
    rewrite_weights: np.ndarray
    kl_weights: np.ndarray
    targets: np.ndarray
    anchors: list
    anchor_star: np.ndarray
    teacher_logprobs: np.ndarray


def logsoftmax(logits):
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=0, keepdims=True))


def committed_forward(model, updates):
    """Only apply candidate weights; do not re-solve any writer."""
    if len(updates) != len(model.weights):
        raise ValueError("all eligible layers must be supplied, including zero blocks")
    X = model.inputs
    hidden = []
    for index, (W, U) in enumerate(zip(model.weights, updates)):
        H = (W + U) @ X
        hidden.append(H)
        if index + 1 < len(updates):
            X = np.tanh(H)
    return model.head @ hidden[-1], hidden


def candidate(model, blocks):
    if len(blocks) != len(model.weights):
        raise ValueError("all eligible layers must remain in the candidate")
    X = model.inputs
    layers = []
    for index, (W, A, u, anchors) in enumerate(zip(model.weights, model.metrics, blocks, model.anchors)):
        u = array(u, "u")
        if u.shape != (W.shape[0], model.mean_operator.shape[1]):
            raise ValueError("candidate block shape")
        R = u * anchors[None, :]
        K = X @ model.mean_operator
        writer = memit_writer(R, K, A)
        H = (W + writer["U"]) @ X
        layers.append(dict(writer=writer, X=X, H=H))
        if index + 1 < len(model.weights):
            X = np.tanh(H)
    return {"layers": layers, "logits": model.head @ layers[-1]["H"],
            "Q_dense": sum(x["writer"]["Q_dense"] for x in layers),
            "Q_compact": sum(x["writer"]["Q_compact"] for x in layers)}


def task_loss(model, logits, selected_rows=None, kl_factor=0.0625):
    logp = logsoftmax(logits)
    probabilities = np.exp(logp)
    row_mask = np.ones(logits.shape[1], dtype=np.float64)
    if selected_rows is not None:
        row_mask[:] = 0
        row_mask[np.asarray(selected_rows)] = 1
    rewrite = model.rewrite_weights * row_mask
    kl_weights = model.kl_weights * row_mask
    chosen = logp[model.targets, np.arange(logits.shape[1])]
    nll = -float(np.sum(rewrite * chosen))
    ratio = logp - model.teacher_logprobs
    kl_rows = np.sum(probabilities * ratio, axis=0)
    kl = float(np.sum(kl_weights * kl_rows))
    grad = probabilities.copy()
    grad[model.targets, np.arange(logits.shape[1])] -= 1
    grad *= rewrite[None, :]
    grad += kl_factor * probabilities * (ratio - kl_rows[None, :]) * kl_weights[None, :]
    return {"native_sum": nll + kl_factor * kl, "nll_sum": nll,
            "kl_sum": kl, "logit_gradient": grad}


def norm_loss(model, blocks):
    """Preserved pre-writer requested-u norm; not a penalty on realized Y."""
    prices = 0.5 / model.anchor_star
    gradients, total = [], 0.0
    for u in blocks:
        norms = np.linalg.norm(u, axis=0)
        total += float(np.sum(prices * norms))
        gradients.append(u / np.where(norms > 0, norms, 1)[None, :] * prices[None, :])
    return total, gradients


def candidate_jvp(model, cached, directions, *, differentiate_upper_keys=True):
    dX = np.zeros_like(model.inputs)
    dQs, dQc, tangents = 0.0, 0.0, []
    for index, (W, layer, du, anchors) in enumerate(zip(model.weights, cached["layers"], directions, model.anchors)):
        dR = du * anchors[None, :]
        dK = dX @ model.mean_operator
        writer_dK = dK if differentiate_upper_keys else np.zeros_like(dK)
        tangent = writer_jvp(layer["writer"], dR, writer_dK)
        dH = tangent["dU"] @ layer["X"] + (W + layer["writer"]["U"]) @ dX
        dQs += tangent["dQ_dense"]
        dQc += tangent["dQ_compact"]
        tangents.append(dict(dK=dK, dU=tangent["dU"], dH=dH))
        if index + 1 < len(model.weights):
            dX = (1 - np.tanh(layer["H"]) ** 2) * dH
    return {"logits": model.head @ tangents[-1]["dH"], "Q_dense": dQs,
            "Q_compact": dQc, "layers": tangents}


def gradients(model, blocks, *, selected_rows=None, differentiate_upper_keys=True):
    """Small CPU gradient via analytic coordinate JVPs, not finite differences."""
    cached = candidate(model, blocks)
    task = task_loss(model, cached["logits"], selected_rows)
    task_grad = [np.zeros_like(u) for u in blocks]
    q_grad = [np.zeros_like(u) for u in blocks]
    compact_grad = [np.zeros_like(u) for u in blocks]
    directions = [np.zeros_like(u) for u in blocks]
    for layer, u in enumerate(blocks):
        for index in np.ndindex(u.shape):
            directions[layer][index] = 1.0
            derivative = candidate_jvp(model, cached, directions,
                                       differentiate_upper_keys=differentiate_upper_keys)
            task_grad[layer][index] = np.sum(task["logit_gradient"] * derivative["logits"])
            q_grad[layer][index] = derivative["Q_dense"]
            compact_grad[layer][index] = derivative["Q_compact"]
            directions[layer][index] = 0.0
    return {"cached": cached, "task": task, "native_gradient": task_grad,
            "Q_gradient": q_grad, "Q_compact_gradient": compact_grad}


def project_requested_budget(blocks, radius=0.75):
    """Euclidean group-L1 projection; every layer remains available afterward."""
    if not blocks or radius < 0:
        raise ValueError("projection input")
    projected = [array(u, "projection block") for u in blocks]
    for request in range(projected[0].shape[1]):
        norms = np.array([np.linalg.norm(u[:, request]) for u in projected])
        if norms.sum() <= radius:
            continue
        ordered = np.sort(norms)[::-1]
        thresholds = (np.cumsum(ordered) - radius) / np.arange(1, len(norms) + 1)
        active = ordered > thresholds
        tau = thresholds[np.flatnonzero(active)[-1]] if radius else ordered[0]
        scales = np.maximum(0, norms - tau) / np.where(norms > 0, norms, 1)
        for block, scale in zip(projected, scales):
            block[:, request] *= scale
    return projected


def gradient_norm(blocks):
    return float(np.sqrt(sum(np.sum(g * g) for g in blocks)))


def calibration_ratio(model, blocks):
    """One first-nonzero-candidate unit convention; no extra fitting loop."""
    derivative = gradients(model, blocks)
    if all(np.count_nonzero(x["writer"]["U"]) == 0 for x in derivative["cached"]["layers"]):
        return None
    _, norm_gradient = norm_loss(model, blocks)
    numerator, denominator = gradient_norm(norm_gradient), gradient_norm(derivative["Q_gradient"])
    if not np.isfinite(numerator + denominator) or numerator <= 0 or denominator <= 0:
        raise ValueError("nonzero candidate has invalid calibration gradient")
    return numerator / denominator
