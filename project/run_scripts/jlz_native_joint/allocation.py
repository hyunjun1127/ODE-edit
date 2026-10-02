"""FP64 whole-logical-batch ridge geometry. RAM factors, never disk."""
import time
import numpy as np
import torch
from .common import require

class Geometry:
    def __init__(self, adapter, history, stat_paths, coefficient):
        self.adapter, self.history = adapter, history
        self.paths, self.coefficient = stat_paths, coefficient
        self.factors, self.entry, self.records = {}, {}, []

    def system(self, layer):
        with np.load(self.paths[str(layer)], allow_pickle=False) as z:
            raw = torch.from_numpy(z['mom2.mom2'].copy())
            count = int(z['mom2.count'])
        require(raw.dtype == torch.float32 and count > 0, 'C0_NATIVE_SCHEMA')
        # Native SecondMoment.moment: FP32 division first, not raw sum.
        raw.div_(count)
        A = raw.to(self.adapter.device, dtype=torch.float64)
        A.mul_(self.coefficient).add_(self.history[layer].to(self.adapter.device, dtype=torch.float64))
        require(torch.isfinite(A).all(), 'NONFINITE_A')
        return A

    def factor(self, layer):
        if layer not in self.factors:
            start = time.monotonic()
            A = self.system(layer)
            L, info = torch.linalg.cholesky_ex(A)
            require(int(info) == 0, 'A_NOT_SPD_L' + str(layer))
            self.factors[layer] = L.cpu()
            self.records.append(dict(layer=layer, operation='A_Cholesky', seconds=time.monotonic()-start,
                                     storage='RAM_ONLY', history='batch_entry_native_FP32'))
        return self.factors[layer].to(self.adapter.device)

    def solve(self, layer, K, entry=False, reference_check=False):
        start = time.monotonic()
        key = K.double()
        L = self.factor(layer)
        T = torch.cholesky_solve(key, L)
        B = key.shape[1]
        # Dense dual for small B, exact primal/implicit operator for large B.
        if B <= min(key.shape[0], 2048):
            S = torch.eye(B, device=key.device, dtype=torch.float64) + key.T @ T
            C, info = torch.linalg.cholesky_ex(S)
            require(int(info) == 0, 'DUAL_NOT_SPD')
            P = torch.cholesky_solve(T.T, C).T
            E = torch.cholesky_solve(torch.eye(B, device=key.device, dtype=torch.float64), C)
            Q = key.T @ P
            Q = (Q + Q.T) * .5
            correction = float((E + Q - torch.eye(B, device=key.device)).abs().max())
            eig = torch.linalg.eigvalsh(Q)
            summary = dict(trace=float(Q.trace()), eigen_min=float(eig.min()), eigen_max=float(eig.max()),
                           E_plus_Q_error=correction, backend='dual_same_A')
        else:
            A = self.system(layer)
            for i in range(0, B, 256):
                A.add_(key[:, i:i+256] @ key[:, i:i+256].T)
            C, info = torch.linalg.cholesky_ex(A)
            require(int(info) == 0, 'PRIMAL_NOT_SPD')
            P = torch.cat([torch.cholesky_solve(key[:, i:i+256], C) for i in range(0, B, 256)], 1)
            E, Q = None, None
            summary = dict(trace='NOT_MATERIALIZED', eigen_min='NOT_MATERIALIZED',
                           eigen_max='NOT_MATERIALIZED', backend='primal_all_request_implicit_E')
        residual = L @ (L.T @ P) + key @ (key.T @ P) - key
        summary['scaled_residual'] = float(residual.norm() / key.norm().clamp_min(1e-30))
        require(torch.isfinite(P).all() and (E is None or torch.isfinite(E).all()), 'NONFINITE_GEOMETRY')
        result = dict(P=P, E=E, Q=Q, K=key, summary=summary)
        if reference_check:
            direct = torch.linalg.solve(self.system(layer) + key @ key.T, key)
            result['summary']['reference_P_relative'] = float((direct-P).norm()/direct.norm().clamp_min(1e-30))
            result['summary']['reference_E_maxabs'] = float((E-(torch.eye(B,device=key.device)-((key.T@direct)+(direct.T@key))*.5)).abs().max()) if E is not None else 'NOT_MATERIALIZED'
        if entry:
            self.entry[layer] = result
        self.records.append(dict(layer=layer, operation='entry_solve' if entry else 'current_solve',
                                 seconds=time.monotonic()-start, **summary))
        return result

def policy(D, geometry, scale):
    d = D.double()
    E, K, P = geometry['E'], geometry['K'], geometry['P']
    # Exact implicit action across all request columns; output tiles only.
    grad = d @ E if E is not None else torch.cat(
        [d[i:i+128] - (d[i:i+128] @ K.T) @ P for i in range(0, d.shape[0], 128)], 0)
    value = .5 * (d * grad).sum()
    missed = .5 * grad.square().sum()
    return value / scale.double(), grad / scale.double(), dict(
        raw_value=value, unrealized=missed, energy=value-missed)
