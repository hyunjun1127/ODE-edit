"""FP64 reference algebra; no model, evaluator, or historical-prompt access."""
from dataclasses import dataclass
from itertools import combinations
import torch


class NativeMetricBoundary(RuntimeError):
    pass


def fp64(x):
    x = torch.as_tensor(x, dtype=torch.float64, device="cpu")
    if not torch.isfinite(x).all():
        raise NativeMetricBoundary("NONFINITE_CONTROLLER_INPUT")
    return x


@dataclass(frozen=True)
class Solution:
    coefficients: torch.Tensor
    gradient: torch.Tensor
    active: tuple
    objective: float
    stationarity: float
    complementarity: float


def nnls_response(e, psi, metric, lam=0.1):
    """Enumerate all faces, including columns with negative individual gain.

    The native metric is positive definite on included nonzero disjoint blocks.
    No damping, least-squares fallback, gain, or timestep enters this solve.
    """
    e, psi, metric = map(fp64, (e, psi, metric))
    if psi.ndim != 2 or psi.shape[0] != e.numel():
        raise ValueError("response shape")
    n = psi.shape[1]
    if n > 5 or metric.shape != (n, n) or lam <= 0:
        raise ValueError("native coordinate contract")
    if n == 0:
        empty = torch.empty(0, dtype=torch.float64)
        return Solution(empty, empty, (), 0.0, 0.0, 0.0)
    if not torch.allclose(metric, metric.T, atol=0, rtol=1e-13):
        raise NativeMetricBoundary("NATIVE_METRIC_UNRESOLVED: asymmetric")
    torch.linalg.cholesky(metric)  # unresolved native metric must not be repaired
    g = psi.T @ e.reshape(-1)
    a = psi.T @ psi + lam * metric
    scale = max(1.0, float(g.abs().max()), float(a.abs().max()))
    tol = 256 * torch.finfo(torch.float64).eps * max(1, n) * scale
    candidates = []
    for size in range(n + 1):
        for face in combinations(range(n), size):
            c = torch.zeros(n, dtype=torch.float64)
            if face:
                index = torch.tensor(face)
                c[index] = torch.linalg.solve(a[index][:, index], g[index])
            # Never clip a solution or discard a negative-g column beforehand.
            if bool((c < 0).any()):
                continue
            dual = a @ c - g
            inactive = [i for i in range(n) if i not in face]
            if inactive and bool((dual[inactive] < -tol).any()):
                continue
            stationarity = float(dual[list(face)].abs().max()) if face else 0.0
            if stationarity > tol:
                continue
            obj = float(0.5 * c @ a @ c - g @ c)
            candidates.append((obj, face, c, dual, stationarity))
    if not candidates:
        raise NativeMetricBoundary("NNLS_KKT_UNRESOLVED")
    obj, face, c, dual, stationarity = min(candidates, key=lambda row: (row[0], row[1]))
    return Solution(c, dual, face, obj, stationarity, float((c * dual).abs().max()))


def identities(e, psi, metric, c, lam=0.1):
    e, psi, metric, c = map(fp64, (e, psi, metric, c))
    prediction = psi @ c
    q = float(c @ metric @ c)
    v = float(e.square().sum() / 2)
    gain = float(e.reshape(-1) @ prediction)
    response_sq = float(prediction.square().sum())
    return dict(V=v, gain=gain, qN=q, response_sq=response_sq,
                dissipation_residual=gain-response_sq-lam*q,
                speed_bound=v/(2*lam), speed_excess=q-v/(2*lam),
                velocity_mismatch=float(torch.linalg.vector_norm(e.reshape(-1)-prediction)))


def ray_solution(e, psi, metric, q_layers, lam=0.1, *, diagonal=False):
    e, psi, metric, q_layers = map(fp64, (e, psi, metric, q_layers))
    g, h = psi.T @ e.reshape(-1), psi.T @ psi
    if diagonal:
        direction = g.clamp(min=0) / (h.diagonal() + lam)
    else:
        raw_gain = g * q_layers.sqrt()
        raw_response_sq = h.diagonal() * q_layers
        u = torch.zeros_like(g)
        valid = raw_response_sq > 0
        u[valid] = (raw_gain[valid] / raw_response_sq[valid]).clamp(0, 1)
        direction = u * q_layers.sqrt()
    gain = g @ direction
    if gain <= 0:
        return torch.zeros_like(g), direction
    denominator = direction @ (h + lam * metric) @ direction
    if not torch.isfinite(denominator) or denominator <= 0:
        raise NativeMetricBoundary("RAY_METRIC_UNRESOLVED")
    return (gain / denominator) * direction, direction


def turning(joint, ray, metric):
    joint, ray, metric = map(fp64, (joint, ray, metric))
    qj, qr = float(joint @ metric @ joint), float(ray @ metric @ ray)
    if qj == 0 or qr == 0:
        return dict(beta=None, Rturn=None, native_cosine=None, status="ZERO_FIELD_ANGLE_NA")
    cross = float(joint @ metric @ ray)
    beta = cross / qr
    residual = joint-beta*ray
    return dict(beta=beta, Rturn=float(residual@metric@residual)/qj,
                native_cosine=cross/(qj*qr)**0.5, status="FINITE")


def factor_inner(left_a, right_a, left_b, right_b, operator):
    """tr(A S B^T), with A=L_a R_a^T. Reduce in FP64, no dense update."""
    la, ra, lb, rb = map(fp64, (left_a, right_a, left_b, right_b))
    return float(((la.T @ lb) * (ra.T @ operator(rb))).sum())


@dataclass(frozen=True)
class FrozenNormalization:
    scales: torch.Tensor
    active: torch.Tensor
    entry_sha: str

    @classmethod
    def capture(cls, target, entry, entry_sha):
        # Preserve source N0 FP32 norm rounding before FP64 reductions.
        scales = torch.linalg.vector_norm(target.float()-entry.float(), dim=0).cpu()
        if not torch.isfinite(scales).all():
            raise NativeMetricBoundary("NONFINITE_NORMALIZATION")
        return cls(scales, scales > 0, entry_sha)

    def weight(self, tensor):
        value = fp64(tensor)
        count = int(self.active.sum())
        if count == 0:
            return value[:, :0].reshape(-1)
        return (value[:, self.active] / (self.scales[self.active].double()*count**0.5)).reshape(-1)
