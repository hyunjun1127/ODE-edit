"""CPU-only illustrative VJP audit; no production imports or GPU use."""
import json
import math
import platform

import torch
from torch.autograd import Function


def emit(record):
    def portable(value):
        if isinstance(value, float) and not math.isfinite(value):
            return "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        return value
    print(json.dumps(portable(record), allow_nan=False), flush=True)


torch.set_num_threads(1)
emit({"kind": "environment", "python": platform.python_version(),
      "torch": torch.__version__, "device": "cpu", "threads": 1,
      "seed": 20261001})
gen = torch.Generator().manual_seed(20261001)
T, I, O, B, C = 31, 257, 127, 7, 5
A = torch.randn(I, B, generator=gen, dtype=torch.float64) / I**.5
Xs = [torch.randn(T, I, generator=gen) for _ in range(C)]
Gs = [torch.randn(T, O, generator=gen) for _ in range(C)]


def compare(name, inputs, upstream, adj):
    dense = torch.zeros(O, I)
    dense64 = torch.zeros(O, I, dtype=torch.float64)
    direct64 = torch.zeros(O, adj.shape[1], dtype=torch.float64)
    mixed64 = torch.zeros_like(direct64)
    for x, g in zip(inputs, upstream):
        dense.add_(g.T @ x)
        dense64.add_(g.double().T @ x.double())
        direct64.add_(g.double().T @ (x.double() @ adj))
        mixed64.add_((g.T @ (x @ adj.float())).double())
    ref = (dense.double() @ adj).float()
    high = dense64 @ adj
    direct, mixed = direct64.float(), mixed64.float()
    emit({"case": name, "dims": [len(inputs), T, I, O, int(adj.shape[1])],
          "reference_norm": float(ref.norm()),
          "reassociated_fp64_relative": float((high-direct64).norm()/high.norm().clamp_min(1e-30)),
          "direct_fp64_vs_current_relative": float((direct-ref).norm()/ref.norm().clamp_min(1e-30)),
          "direct_fp64_vs_current_maxabs": float((direct-ref).abs().max()),
          "fp32_projection_vs_current_relative": float((mixed-ref).norm()/ref.norm().clamp_min(1e-30)),
          "bitwise_equal": bool(torch.equal(direct, ref))})


compare("normal", Xs, Gs, A)
X, G = Xs[0], Gs[0]
H = torch.randn(T, O, generator=gen)
compare("cancellation", [X, X], [G, -G + 1e-5*H], A)

# A direct finite result does not establish original dense-gradient finiteness.
X = torch.tensor([[1e20, -1e20]], dtype=torch.float32)
G = torch.tensor([[1e20]], dtype=torch.float32)
A = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
dense = G.T @ X
old = (dense.double() @ A).float()
new = (G.double().T @ (X.double() @ A)).float()
emit({"case": "dense_overflow_domain", "dense_dW": dense.tolist(),
      "current_gR": old.tolist(), "direct_gR": new.tolist()})


class LinearR(Function):
    @staticmethod
    def forward(ctx, x, r64, weight, adj):
        ctx.save_for_backward(weight, x.double() @ adj)
        return torch.nn.functional.linear(x, weight)

    @staticmethod
    def backward(ctx, grad):
        weight, projected = ctx.saved_tensors
        dx = grad @ weight if ctx.needs_input_grad[0] else None
        dr = grad.double().T @ projected
        return dx, dr, None, None


# This resets the default generator independently of the contraction fixtures.
torch.manual_seed(20261001)
I, O, B, T = 17, 17, 3, 9
X = torch.randn(T, I)
Ws = [torch.randn(O, I)*.05 for _ in range(2)]
Rs = [torch.randn(O, B)*.01 for _ in range(2)]
As = [torch.randn(I, B, dtype=torch.float64)*.03 for _ in range(2)]
effective = [w + (r.double() @ a.T).float() for w, r, a in zip(Ws, Rs, As)]
leaves = [w.detach().requires_grad_() for w in effective]
y0 = torch.nn.functional.linear(
    torch.tanh(torch.nn.functional.linear(X, leaves[0])) + .05*X, leaves[1])
f0 = y0.square().sum()
dws = torch.autograd.grad(f0, leaves)
refs = [(g.double() @ a).float() for g, a in zip(dws, As)]
proxies = [r.double().detach().requires_grad_() for r in Rs]
y1 = LinearR.apply(
    torch.tanh(LinearR.apply(X, proxies[0], effective[0], As[0])) + .05*X,
    proxies[1], effective[1], As[1])
f1 = y1.square().sum()
drs = torch.autograd.grad(f1, proxies)
emit({"case": "two_layer_cross_effect",
      "two_layer_same_forward": torch.equal(y0, y1),
      "same_loss": torch.equal(f0, f1),
      "layer_grad_relative": [float((g.float()-r).norm()/r.norm()) for g, r in zip(drs, refs)],
      "layer_grad_maxabs": [float((g.float()-r).abs().max()) for g, r in zip(drs, refs)]})
