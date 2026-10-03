"""Fixed whole-B SPD proxy, never a candidate actual writer."""
import torch
from project.run_scripts.jlz_realization.geometry import inverse_apply, ridge
from .common import require,tensor_sha

def stable_squared(D, T):
    return torch.linalg.solve_triangular(T, D.double().T, upper=False).square().sum()

def root_cost(D, T):
    # Vector norm supplies the selected zero subgradient without smoothing.
    return torch.linalg.vector_norm(torch.linalg.solve_triangular(T, D.double().T, upper=False))

@torch.no_grad()
def freeze(entry, device):
    frozen, receipt = {}, {}
    for layer, factor in entry['factors'].items():
        require(factor['SPD'] and factor['L'] is not None and factor['LU'] is None, 'ENTRY_A_NOT_SPD')
        K = entry['mean_keys'][layer].to(device).double()
        Y = inverse_apply(factor, K)
        S = torch.eye(K.shape[1], device=device, dtype=torch.float64) + K.T @ Y
        T, info = torch.linalg.cholesky_ex(S)
        require(int(info) == 0 and torch.isfinite(T).all(), 'ENTRY_PROXY_NOT_SPD')
        native = ridge(K, factor)
        require(factor['L'] is not None, 'SPD_REFERENCE_MISMATCH_NO_SILENT_LU_PROXY')
        # Full-matrix check, not a favorable single D. No inverse is stored.
        identity = torch.eye(K.shape[1], device=device, dtype=torch.float64)
        L = torch.linalg.solve_triangular(T, identity, upper=False)
        cost_matrix = L.T @ L
        reference = native['G'] + native['E']
        error = (cost_matrix-reference).abs()
        require(bool((error <= 1e-8 + 1e-6*reference.abs()).all()), 'STABLE_GEOMETRY_REFERENCE')
        frozen[layer] = dict(T=T.detach(), sigma=entry['anchors'][layer].double().square().mean().sqrt(),
                             G=native['G'].detach(),E=native['E'].detach())
        receipt[layer] = dict(**native['metadata'], matrix_max_error=float(error.max()),
                              fixed_entry=True, whole_B=K.shape[1], gradient_into_keys=False,
                              K_hash=tensor_sha(K),A_hash=tensor_sha(factor['A']),P_hash=tensor_sha(native['P']),
                              capacity_diagonal=native['M'].diag().tolist(),
                              stable_cost_eigenvalues=torch.linalg.eigvalsh(cost_matrix).tolist(),
                              sigma=float(frozen[layer]['sigma']))
    return frozen, receipt

def allocation(D, frozen, coefficient):
    B = next(iter(D.values())).shape[1]
    costs = {l: root_cost(d, frozen[l]['T'])/(B**.5*frozen[l]['sigma']) for l,d in D.items()}
    return coefficient*sum(costs.values()), costs
