"""CPU toy qualification for T-prime. Not production JLZ/model validation."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(3102026)
B, C, L, D, T = 3, 6, 2, 4, 3
RW, ROWS, SUBJECT = B * C, B * (C + 1), 1
GROUP_W = torch.tensor([.5, .1, .1, .1, .1, .1])
X = torch.randn(ROWS, T, D) * .3
W = torch.randn(L, D, D) * .2
KEY = torch.randn(L, D, D) * .35
FCT = torch.randn(L, D, D)
A = FCT @ FCT.transpose(-1, -2) + 2 * torch.eye(D)
HEAD = torch.randn(5, D) * .3
ANCHOR = torch.rand(L, B) + 1.3
SCALE = ANCHOR[:, None, :] / (D * L) ** .5
MASK = torch.tensor([0., 1., 0.])[None, :, None]
TARGET = torch.arange(RW) % 5
# Arbitrary derivative-check fixture; .75 here is not an experiment coefficient.
# The v10 execution contract fixes the native norm coefficient at .5.
COEF = {'nll': 1., 'kl': .0625, 'norm': .75, 'alloc': .1}


def keys(x, layer):
    causal_mean = x.cumsum(1) / torch.arange(1, T + 1)[None, :, None]
    return torch.tanh(F.linear(x + .2 * causal_mean, KEY[layer]))


def builder(q, stop_geometry=False):
    x, increments, energies, actual_keys, updates = X, [], [], [], []
    for l in range(L):
        k = keys(x, l)
        ka = k[:, SUBJECT]
        mean = (ka[:RW].reshape(B, C, D) * GROUP_W[None, :, None]).sum(1).T
        solve_k = mean.detach() if stop_geometry else mean
        p = torch.linalg.solve(A[l] + solve_k @ solve_k.T, solve_k)
        u = (q[l] * SCALE[l]) @ p.T
        increments.append(F.linear(ka, u))  # includes native KL rows
        energies.append(torch.trace(u @ A[l] @ u.T))
        actual_keys.append(ka)
        updates.append(u)
        x = x + F.linear(k, W[l] + u)
    return increments, energies, actual_keys, updates


def fit(v, rows):
    x = X[rows]
    for l in range(L):
        x = x + F.linear(keys(x, l), W[l]) + MASK * v[l][rows, None, :]
    return F.linear(x[:, SUBJECT], HEAD).log_softmax(-1)


ENTRY = fit([torch.zeros(ROWS, D) for _ in range(L)], list(range(ROWS))).detach()


def terms(v, energies, rows):
    lp = fit(v, rows)
    zero = lp.sum() * 0
    nll, kl, norm = zero, zero, zero
    for i, row in enumerate(rows):
        if row < RW:
            req, ctx = divmod(row, C)
            nll = nll - lp[i, TARGET[row]] / (B * C)
            norm = norm + sum(GROUP_W[ctx] * v[l][row].norm() / ANCHOR[l, req] ** 2 / B for l in range(L))
        else:
            kl = kl + (lp[i].exp() * (lp[i] - ENTRY[row])).sum() / B
    alloc = sum(e.sqrt() / (B ** .5 * ANCHOR[l].square().mean().sqrt()) for l, e in enumerate(energies)) if energies else zero
    return dict(nll=nll, kl=kl, norm=norm, alloc=alloc)


def dense(q, stop_geometry=False):
    v, e, _, _ = builder(q, stop_geometry)
    return terms(v, e, list(range(ROWS)))


def staged(q, partition):
    v, e, _, _ = builder(q)
    leaves = [x.detach().requires_grad_() for x in v]
    for rows in partition:
        part = terms(leaves, [], rows)
        sum(COEF[k] * part[k] for k in ['nll', 'kl', 'norm']).backward()
    alloc = sum(ei.sqrt() / (B ** .5 * ANCHOR[l].square().mean().sqrt()) for l, ei in enumerate(e))
    torch.autograd.backward(v + [alloc], [x.grad for x in leaves] + [torch.tensor(COEF['alloc'])])
    return q.grad.detach()


def main():
    initial = torch.randn(L, D, B) * .4  # nonzero avoids nonsmooth norm origin
    q = initial.clone().requires_grad_()
    part = dense(q)
    loss = sum(COEF[k] * value for k, value in part.items())
    grad = torch.autograd.grad(loss, q)[0]
    errors = {}
    for size in [1, 4, ROWS]:
        order = list(range(ROWS - 1, -1, -1)) if size == 4 else list(range(ROWS))
        partition = [order[i:i + size] for i in range(0, ROWS, size)]
        result = staged(initial.clone().requires_grad_(), partition)
        errors[str(size)] = float((grad - result).abs().max())
    assert max(errors.values()) < 1e-11, errors

    finite_diff = {}
    for name in list(COEF) + ['total']:
        qq = initial.clone().requires_grad_()
        td = dense(qq)
        scalar = sum(COEF[k] * v for k, v in td.items()) if name == 'total' else td[name]
        exact = torch.autograd.grad(scalar, qq)[0]
        numerical = torch.zeros_like(initial)
        epsilon = 1e-5
        for i in range(initial.numel()):
            plus, minus = initial.clone(), initial.clone()
            plus.view(-1)[i] += epsilon
            minus.view(-1)[i] -= epsilon
            with torch.no_grad():
                p, m = dense(plus), dense(minus)
                pv = sum(COEF[k] * v for k, v in p.items()) if name == 'total' else p[name]
                mv = sum(COEF[k] * v for k, v in m.items()) if name == 'total' else m[name]
            numerical.view(-1)[i] = (pv - mv) / (2 * epsilon)
        finite_diff[name] = float((exact - numerical).abs().max())
    assert max(finite_diff.values()) < 1e-7, finite_diff

    wrong_q = initial.clone().requires_grad_()
    wrong = dense(wrong_q, stop_geometry=True)
    wrong_g = torch.autograd.grad(sum(COEF[k] * v for k, v in wrong.items()), wrong_q)[0]
    negative_control = float((wrong_g - grad).norm())
    assert negative_control > 1e-7, negative_control

    # T-prime uses the identical actual increment tensor, not U times masked key.
    v, e, ka, u = builder(initial)
    xs, xa = X.clone(), X.clone()
    gap_errors, local_errors = [], []
    for l in range(L):
        ks, kak = keys(xs, l), keys(xa, l)
        base_s = xs + F.linear(ks, W[l])
        base_a = xa + F.linear(kak, W[l])
        xs = base_s + MASK * v[l][:, None, :]
        xa = base_a + F.linear(kak, u[l])
        local_errors.append(float((v[l] - F.linear(ka[l], u[l])).abs().max()))
        gap_errors.append(float(((xa[:, SUBJECT] - xs[:, SUBJECT]) - (base_a[:, SUBJECT] - base_s[:, SUBJECT])).abs().max()))
    assert max(gap_errors + local_errors) < 1e-12

    # Finite-precision increment must be formed using materialized weights.
    k32, w32 = ka[-1].float(), W[-1].float()
    weff = w32 + u[-1].float()
    v32 = F.linear(k32, weff) - F.linear(k32, w32)
    same_input_reconstruction = float((F.linear(k32, w32) + v32 - F.linear(k32, weff)).abs().max())
    assert same_input_reconstruction < 2e-6
    report = {
        'kind': 'CPU_synthetic_two_layer_Tprime_autograd_not_production_model',
        'profile': dict(B=B, rewrite_contexts=C, KL_rows=B, layers=L, dim=D, dtype='float64'),
        'staged_vs_dense_gradient_max_abs_by_microbatch': errors,
        'full_K_P_path_finite_difference_max_abs': finite_diff,
        'stopped_geometry_negative_control_gradient_L2_error': negative_control,
        'local_increment_identity_max_abs': max(local_errors),
        'remaining_base_gap_identity_max_abs': max(gap_errors),
        'FP32_same_input_reconstruction_max_abs': same_input_reconstruction,
        'norm_context_weights': GROUP_W.tolist(),
        'NLL_context_weights': [1 / C] * C,
    }
    Path(__file__).with_name('tprime-adjoint-results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
