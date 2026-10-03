"""CPU-only algebra/graph validation of the proposed v11 contract.

This is a tiny nonlinear causal surrogate, not production Llama qualification.
No repository implementation is imported and no model weights are loaded.
"""
import json
import math
from pathlib import Path

import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
ALG_ATOL = 1e-10
FD_ATOL = 1e-7
FD_RTOL = 1e-5


def require(value, message):
    if not bool(value):
        raise AssertionError(message)


def error(x, y):
    return float((x - y).abs().max())


def clone_leaves(xs):
    return [x.detach().clone().requires_grad_(True) for x in xs]


def geometry(A, K):
    """Frozen whole-logical-B geometry; stable cost uses chol(I+K^T A^-1 K)."""
    chol_A = torch.linalg.cholesky(A)
    invA_K = torch.cholesky_solve(K, chol_A)
    S = K.T @ invA_K
    chol_T = torch.linalg.cholesky(torch.eye(K.shape[1]) + S)
    P = torch.cholesky_solve(K, torch.linalg.cholesky(A + K @ K.T))
    return dict(A=A, K=K, chol_A=chol_A, S=S, chol_T=chol_T, P=P)


def stable_factor(D, geo):
    # C=||L_T^-1 D^T||_F^2. No explicit subtraction I-M.
    return torch.linalg.solve_triangular(geo['chol_T'], D.T, upper=False)


def cost_squared(D, geo):
    return stable_factor(D, geo).square().sum()


def root_cost(D, geo, B, sigma2):
    # torch norm chooses zero subgradient at the origin; sqrt(sum(square)) does not.
    return torch.linalg.vector_norm(stable_factor(D, geo)) / torch.sqrt(B * sigma2)


def dense_cost(D, geo):
    U = D @ geo['P'].T
    Q = torch.linalg.vector_norm(U @ geo['chol_A']).square()
    E = torch.linalg.vector_norm(U @ geo['K'] - D).square()
    return Q + E, Q, E, U


def finite_difference(fn, leaves, eps=1e-6):
    numerical = []
    with torch.no_grad():
        for k, x in enumerate(leaves):
            grad = torch.zeros_like(x)
            for j in range(x.numel()):
                plus = [v.detach().clone() for v in leaves]
                minus = [v.detach().clone() for v in leaves]
                plus[k].reshape(-1)[j] += eps
                minus[k].reshape(-1)[j] -= eps
                grad.reshape(-1)[j] = (fn(plus) - fn(minus)) / (2 * eps)
            numerical.append(grad)
    return numerical


class TinyCausalModel:
    """Three contexts and three tokens; subject is token 1, readout is token 2."""
    def __init__(self, B, m, hidden, seed):
        g = torch.Generator().manual_seed(seed)
        self.B, self.m, self.hidden = B, m, hidden
        self.x = torch.randn(B, 3, 3, hidden, generator=g) * .35
        self.projections = [torch.randn(hidden + l + 1, hidden, generator=g) * .4 for l in range(m)]
        self.weights = [torch.randn(hidden, hidden + l + 1, generator=g) * .2 for l in range(m)]
        self.head = torch.randn(hidden, 7, generator=g) * .4
        self.suffix = torch.eye(hidden) + torch.randn(hidden, hidden, generator=g) * .1
        self.labels = torch.arange(B) % 7
        self.subject_mask = torch.tensor([0., 1., 0.]).view(1, 1, 3, 1)
        self.As = []
        for l in range(m):
            dim = hidden + l + 1
            z = torch.randn(dim, dim, generator=g)
            self.As.append(z @ z.T + .7 * torch.eye(dim))
        with torch.no_grad():
            logits, keys, states = self.virtual([torch.zeros(hidden, B) for _ in range(m)])
            self.teacher_logp = logits[:, 2].log_softmax(-1).detach()
            self.anchors = [h[:, 0, 1].norm(dim=-1).detach() for h in states]
            require(all((a > .01).all() for a in self.anchors), 'anchor unexpectedly tiny')
            self.geometries = [geometry(A, k[:, :2, 1].mean(1).T.detach()) for A, k in zip(self.As, keys)]
            self.sigma2 = [a.square().mean() for a in self.anchors]
            self.scales = [a / math.sqrt(hidden * m) for a in self.anchors]

    @staticmethod
    def mix(h):
        n = torch.arange(1, h.shape[2] + 1).view(1, 1, -1, 1)
        return h + .25 * h.cumsum(2) / n

    def logits(self, h):
        return torch.tanh(self.mix(h) @ self.suffix)[:, :, -1] @ self.head

    def virtual(self, D, indices=None):
        ix = torch.arange(self.B) if indices is None else torch.tensor(indices)
        h = self.x[ix]
        keys, states = [], []
        for l, (V, W) in enumerate(zip(self.projections, self.weights)):
            k = torch.tanh(self.mix(h) @ V.T)
            h = h + k @ W.T + D[l][:, ix].T[:, None, None, :] * self.subject_mask
            keys.append(k)
            states.append(h)
        return self.logits(h), keys, states

    def native_per_request(self, D, indices=None):
        ix = torch.arange(self.B) if indices is None else torch.tensor(indices)
        logits, _, _ = self.virtual(D, indices)
        lp = logits.log_softmax(-1)
        nll = -lp[:, :2].gather(-1, self.labels[ix, None, None].expand(-1, 2, 1)).squeeze(-1).mean(1)
        current = lp[:, 2]
        kl = (current.exp() * (current - self.teacher_logp[ix])).sum(-1)
        norm = sum(d[:, ix].norm(dim=0) / a[ix].square() for d, a in zip(D, self.anchors))
        return nll + .0625 * kl + .5 * norm

    def allocation(self, D):
        return sum(root_cost(d, geo, self.B, sigma) for d, geo, sigma in zip(D, self.geometries, self.sigma2))

    def from_q(self, q):
        return [v * s[None, :] for v, s in zip(q, self.scales)]

    def mean_loss(self, D):
        return self.native_per_request(D).mean() + .1 * self.allocation(D)

    def sum_loss_q(self, q):
        return self.B * self.mean_loss(self.from_q(q))

    def sequential_increment_write(self, D, stale_upper=False):
        h = self.x.clone()
        keys, states, Us, costs = [], [], [], []
        for l, (V, W, A) in enumerate(zip(self.projections, self.weights, self.As)):
            k = torch.tanh(self.mix(h) @ V.T)
            K = k[:, :2, 1].mean(1).T
            geo = self.geometries[l] if stale_upper and l > 0 else geometry(A, K)
            U = D[l] @ geo['P'].T
            h = h + k @ (W + U).T
            keys.append(K); states.append(h); Us.append(U)
            costs.append(cost_squared(D[l], geometry(A, K)))
        return dict(logits=self.logits(h), keys=keys, states=states, updates=Us, costs=costs)


def check_case(B, m, hidden, seed):
    model = TinyCausalModel(B, m, hidden, seed)
    g = torch.Generator().manual_seed(seed + 1)
    q = clone_leaves([torch.randn(hidden, B, generator=g) * .3 for _ in range(m)])
    D = model.from_q(q)
    algebra = []
    for l in range(m):
        c, Q, E, U = dense_cost(D[l], model.geometries[l])
        stable = cost_squared(D[l], model.geometries[l])
        dense_P = torch.linalg.solve(model.As[l] + model.geometries[l]['K'] @ model.geometries[l]['K'].T, model.geometries[l]['K'])
        P_error = error(dense_P, model.geometries[l]['P'])
        stationarity = U @ (model.As[l] + model.geometries[l]['K'] @ model.geometries[l]['K'].T) - D[l] @ model.geometries[l]['K'].T
        algebra.append(dict(layer=l, input_dim=model.As[l].shape[0], output_dim=hidden,
                            cost=float(stable.detach()), Q=float(Q.detach()), E=float(E.detach()),
                            cost_error=float((c - stable).abs().detach()), solve_residual=float(stationarity.abs().max().detach()),
                            SPD_vs_dense_solve_error=P_error))
        require(abs(c - stable) < ALG_ATOL, 'G+E != stable norm cost')
        require(P_error < ALG_ATOL, 'SPD vs dense ridge solve mismatch')
        require(stationarity.abs().max() < ALG_ATOL, 'ridge stationarity failed')
    dense_grad = torch.autograd.grad(model.sum_loss_q(q), q)
    # Split native task/norm losses; full-logical-B allocation appears ONCE.
    split_grad = [torch.zeros_like(v) for v in q]
    schedule = [[i] for i in reversed(range(B))]
    split_value = 0.
    for indices in schedule:
        value = model.native_per_request(model.from_q(q), indices).sum()
        split_value += float(value.detach())
        for acc, grad in zip(split_grad, torch.autograd.grad(value, q)):
            acc.add_(grad)
    once = B * .1 * model.allocation(model.from_q(q))
    split_value += float(once.detach())
    for acc, grad in zip(split_grad, torch.autograd.grad(once, q)):
        acc.add_(grad)
    split_error = max(error(a, b) for a, b in zip(dense_grad, split_grad))
    require(split_error < ALG_ATOL, 'microbatch split changed q gradient')
    require(abs(split_value - float(model.sum_loss_q(q).detach())) < ALG_ATOL, 'split loss changed')
    # Physical-D B*mean gradient, then exactly one D=sq bridge.
    direct_D = clone_leaves(model.from_q(q))
    gd = torch.autograd.grad(B * model.mean_loss(direct_D), direct_D)
    bridge = [v * s[None, :] for v, s in zip(gd, model.scales)]
    bridge_error = max(error(a, b) for a, b in zip(dense_grad, bridge))
    require(bridge_error < ALG_ATOL, 'Bsum/q bridge mismatch')
    fd = finite_difference(model.sum_loss_q, q)
    fd_error = max(error(a, b) for a, b in zip(dense_grad, fd))
    for a, b in zip(dense_grad, fd):
        require(torch.allclose(a, b, atol=FD_ATOL, rtol=FD_RTOL), 'nonlinear joint gradient finite difference failed')
    # Allocation-only FD and minimum-norm subgradient at every zero-D block.
    alloc_q = lambda qs: model.allocation(model.from_q(qs))
    alloc_grad = torch.autograd.grad(alloc_q(q), q)
    alloc_fd = finite_difference(alloc_q, q)
    require(all(torch.allclose(a, b, atol=FD_ATOL, rtol=FD_RTOL) for a, b in zip(alloc_grad, alloc_fd)), 'allocation FD failed')
    zeros = clone_leaves([torch.zeros(hidden, B) for _ in range(m)])
    zero_loss = sum(d.norm(dim=0).sum() for d in zeros) + model.allocation(zeros)
    zero_grad = torch.autograd.grad(zero_loss, zeros)
    require(all(torch.isfinite(v).all() and torch.count_nonzero(v) == 0 for v in zero_grad), 'zero norm subgradient unsafe')
    result = dict(logical_B=B, layers=m, output_dim=hidden, algebra=algebra,
                  split_gradient_error=split_error, Bsum_scale_bridge_error=bridge_error,
                  joint_finite_difference_max_abs=fd_error,
                  allocation_finite_difference_max_abs=max(error(a,b) for a,b in zip(alloc_grad,alloc_fd)),
                  zero_norm_chosen_subgradient='finite all-zero; nonsmooth point, not classical differentiability',
                  actual_B_normalizer=B,
                  nominal_batch_capacity_for_partial_example=max(B,3),
                  sigma2_error=max(float(abs(s-a.square().sum()/B)) for s,a in zip(model.sigma2,model.anchors)))
    if B == 1 and m == 1:
        d = model.from_q(q)[0][:, 0]
        native_norm = .5 * d.norm() / model.anchors[0][0].square()
        extended_norm = .5 * sum(z.norm(dim=0) / a.square() for z,a in zip(model.from_q(q),model.anchors)).mean()
        require(abs(native_norm-extended_norm) < ALG_ATOL, 'B1m1 native norm mismatch')
        result['B1_m1_native_norm_equivalence_error'] = float(abs(native_norm-extended_norm).detach())
        result['native_equivalence_scope'] = 'Only direct-delta norm form/dimensions; q optimizer and added allocation are not native compute_z equivalence.'
    return result


def coupling_and_writer_checks():
    model = TinyCausalModel(3, 3, 4, 150)
    g = torch.Generator().manual_seed(155)
    D = clone_leaves([torch.randn(4,3,generator=g)*.5 for _ in range(3)])
    grad_before = torch.autograd.grad(cost_squared(D[0], model.geometries[0]), D[0])[0]
    change = D[0].detach().clone();change[:, 1] += torch.tensor([.3,-.4,.2,.5]);change.requires_grad_(True)
    grad_after = torch.autograd.grad(cost_squared(change, model.geometries[0]), change)[0]
    coupling = float((grad_before[:,0]-grad_after[:,0]).norm())
    require(coupling > 1e-8, 'constructed batch has no observed offdiagonal coupling')
    column_zero = D[0].detach().clone();column_zero[:,0] = 0
    mixed = column_zero @ model.geometries[0]['P'].T @ model.geometries[0]['K']
    crossmix = float(mixed[:,0].norm())
    require(crossmix > 1e-8, 'zero request column crossmix not demonstrated')
    # Joint virtual layers affect one another through the forward graph.
    total = model.native_per_request(D).sum()
    joint_before = torch.autograd.grad(total, D)[0]
    changed = clone_leaves(D);changed[-1].data.add_(.2)
    joint_after = torch.autograd.grad(model.native_per_request(changed).sum(), changed)[0]
    joint_effect = float((joint_before-joint_after).norm())
    require(joint_effect > 1e-8, 'virtual layers not coupled')
    # Commit increments; actual upper K changes, but a zero entire layer has U=0.
    committed_D = [d.detach().clone() for d in D]
    committed_D[1].zero_()
    actual = model.sequential_increment_write(committed_D)
    stale = model.sequential_increment_write(committed_D, stale_upper=True)
    key_drifts = [float((K-geo['K']).norm()) for K,geo in zip(actual['keys'],model.geometries)]
    require(key_drifts[0] == 0 and max(key_drifts[1:]) > 1e-8, 'actual key refresh not observed')
    require(torch.count_nonzero(actual['updates'][1]) == 0, 'zero layer delta wrote nonzero weight')
    _, _, virtual = model.virtual(committed_D)
    local_gap = float((virtual[1][:,0,1]-actual['states'][1][:,0,1]).norm())
    require(local_gap > 1e-8, 'constructed virtual/actual gap absent')
    proxy = [float(cost_squared(d,geo)) for d,geo in zip(committed_D,model.geometries)]
    realized_costs = [float(v) for v in actual['costs']]
    difference = max(abs(a-b) for a,b in zip(proxy,realized_costs))
    require(difference > 1e-8, 'proxy/actual distinction absent')
    stale_diff = error(actual['logits'], stale['logits'])
    require(stale_diff > 1e-8, 'stale actual key caused no demonstrable difference')
    return dict(other_request_changes_gradient=coupling,
                zero_one_request_D_column_mean_key_realization_norm=crossmix,
                upper_delta_changes_lower_task_gradient_norm=joint_effect,
                actual_key_drift_from_entry=key_drifts,
                entire_layer_delta_zero_weight_update_norm=float(actual['updates'][1].norm()),
                zero_layer_virtual_actual_subject_gap=local_gap,
                entry_proxy_costs=proxy, actual_ridge_costs=realized_costs,
                proxy_actual_max_abs=difference, stale_vs_refreshed_logits_max_abs=stale_diff,
                interpretation='Zero entire layer => U=0. Zero individual request column need not imply zero own-key effect. No absolute target feedback; inherited hidden gaps can remain.')


def selection_and_clamp_checks():
    def schedule(values, cap=25):
        events=[];updates=0
        for i,value in enumerate(values[:cap],1):
            stop=(value < .05)
            update=not stop and i<cap
            events.append(dict(candidate=i,evaluated_after_updates=updates,joint_mean_full_loss=value,stop=stop,update_after=update))
            if update:updates+=1
            if stop or i==cap:break
        return dict(evaluations=len(events),updates=updates,terminal=events[-1],events=events)
    fixed=schedule([1.]*25);early=schedule([.8,.08,.049,.7])
    require(fixed['evaluations']==25 and fixed['updates']==24 and not fixed['terminal']['update_after'], 'unevaluated terminal step')
    require(early['evaluations']==3 and early['updates']==2, 'early exit count mismatch')
    request_losses=torch.tensor([.01,.20]);joint_mean=request_losses.mean()+.1*torch.tensor(.2)
    require(request_losses[0]<.05 and joint_mean>=.05, 'joint stop example failed')
    # The clamp changes q in place using physical D radii; Adam moments unchanged.
    q=torch.nn.Parameter(torch.tensor([[20.,-15.],[5.,12.]]))
    scales=torch.tensor([.3,.4]);anchors=torch.tensor([1.,2.])
    opt=torch.optim.Adam([q],lr=.1)
    q.square().sum().backward();opt.step()
    moment_before={k:v.clone() for k,v in opt.state[q].items() if torch.is_tensor(v)}
    with torch.no_grad():
        D=q*scales[None,:]
        factors=torch.minimum(torch.ones(2),.75*anchors/D.norm(dim=0))
        q.mul_(factors[None,:])
    moment_equal=all(torch.equal(v,opt.state[q][k]) for k,v in moment_before.items())
    rho=(q.detach()*scales[None,:]).norm(dim=0)/anchors
    require(moment_equal and bool((rho<=.75+1e-14).all()), 'clamp radius or moment preservation failed')
    return dict(fixed_budget=fixed,early_exit=early,
                stop_statistic='one joint mean full objective including allocation; no per-request freeze/stop',
                one_request_below_threshold_but_joint_continues=float(joint_mean),
                clamp_post_rho=rho.tolist(),adam_moments_bitwise_unchanged=moment_equal,
                scope='Contract execution examples only; not imported production solver tests.')


def main():
    cases=[check_case(1,1,3,11),check_case(2,2,4,22),check_case(3,3,4,33),check_case(2,3,3,44)]
    result=dict(status='PASS',dtype='torch.float64',device='cpu',torch_version=torch.__version__,
                tolerances=dict(algebra_abs=ALG_ATOL,gradient_FD_abs=FD_ATOL,gradient_FD_relative=FD_RTOL),
                stable_cost='T=I+K^T A^-1 K=L_T L_T^T; C=||L_T^-1 D^T||_F^2; root=norm/ sqrt(actual_B * mean(a^2))',
                dimensions_and_graph=cases,coupling_and_commit=coupling_and_writer_checks(),
                selection_and_clamp=selection_and_clamp_checks(),
                limitations=['Synthetic nonlinear causal model, not Llama/native implementation parity.',
                    'Entry geometry is frozen; no candidate K/P gradient is claimed.',
                    'Native norm equivalence only for B=1,m=1; optimizer/total objective can differ.',
                    'At D=0 the norm is nonsmooth; finite zero subgradient is an explicit convention.',
                    'Actual costs and virtual/actual fit gap are expected to differ; no closure claim.',
                    'No GPU experiments, production imports, model downloads, or performance estimates.'])
    Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'cases':len(cases),'max_FD_error':max(c['joint_finite_difference_max_abs'] for c in cases),'commit':result['coupling_and_commit']},indent=2))


if __name__ == '__main__':
    main()
