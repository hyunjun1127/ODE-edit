"""Strict native-endpoint actual radial price arithmetic and immutable lock.

No reference is fitted here.  Callers provide the declared one-layer native
endpoint and the actual all-token task/Q adjoints in u coordinates.  Only
compact scalar diagnostics and hashes are durable; the plan stays in RAM.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import torch

class CalibrationError(RuntimeError):
    def __init__(self, status, receipt):
        self.status, self.receipt = status, receipt
        super().__init__(status)


@dataclass
class CalibrationResult:
    lambda_Q: object
    receipt: dict

    @property
    def price(self):
        return self.lambda_Q

    def __getitem__(self, key):
        return getattr(self, key)


def _json_scalar(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _tensor_hash(value):
    value = value.detach().cpu().contiguous()
    header = _canonical(dict(shape=list(value.shape), dtype=str(value.dtype)))
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def _fail(status, stage, receipt):
    def safe(value):
        if isinstance(value, float):
            return _json_scalar(value)
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        return value
    sanitized = safe(receipt)
    receipt.clear(); receipt.update(sanitized)
    receipt.update(status=status, failure_stage=stage, lambda_Q=None,
                   price_locked=False, fallback_used=False)
    raise CalibrationError(status, receipt)


def calibrate(reference_u, anchors, task_gradient, Q_gradient, Q, identity,
              *, norm_coefficient=.5, metadata=None, genuine_noop=False,
              native_noop_requests=None):
    """lambda=(t-n)/d, never abs/epsilon/clip/default or a second reference.

    The fixed FP64 d=2Q check is atol1e-7+rtol1e-5*abs(2Q), exactly the
    canonical scale-aware FP64 gradient qualification tolerance.
    """
    receipt = dict(schema="CAUSAL_ALLOCATION_EDITING_RADIAL_PRICE",
                   identity=dict(identity), metadata=dict(metadata or {}),
                   norm_coefficient=norm_coefficient, precision="FP64_RADIAL_ARITHMETIC",
                   fallback_used=False, one_reference_only=True,
                   endpoint_post_expansion=False, price_locked=False,
                   d_two_Q_tolerance=dict(atol=1e-7, rtol=1e-5))
    keys = set(reference_u)
    if not keys or keys != set(anchors) or keys != set(task_gradient) or keys != set(Q_gradient):
        _fail("CALIBRATION_IDENTITY_ERROR", "layer_sets", receipt)
    B = next(iter(reference_u.values())).shape[1]
    if B == 0:
        _fail("CALIBRATION_IDENTITY_ERROR", "empty_reference", receipt)
    receipt["B"] = B
    for layer, u in reference_u.items():
        anchor = torch.as_tensor(anchors[layer])
        if (u.ndim != 2 or u.shape[1] != B or anchor.shape != (B,)
                or task_gradient[layer].shape != u.shape or Q_gradient[layer].shape != u.shape):
            _fail("CALIBRATION_IDENTITY_ERROR", f"shape:{layer}", receipt)
        if not bool(torch.isfinite(anchor).all()) or not bool((anchor > 0).all()):
            _fail("CALIBRATION_IDENTITY_ERROR", f"anchor:{layer}", receipt)
        if not all(bool(torch.isfinite(v).all()) for v in (u, task_gradient[layer], Q_gradient[layer])):
            receipt["nonfinite_layer"] = str(layer)
            receipt["nonfinite_counts"] = {
                name: int((~torch.isfinite(value)).sum()) for name, value in
                (("u", u), ("task_gradient", task_gradient[layer]), ("Q_gradient", Q_gradient[layer]))}
            _fail("CALIBRATION_NONFINITE", "channel_values", receipt)
    if not math.isfinite(norm_coefficient) or norm_coefficient < 0:
        _fail("CALIBRATION_NONFINITE", "norm_coefficient", receipt)
    receipt["reference_hashes"] = {str(l): _tensor_hash(v) for l, v in reference_u.items()}
    receipt["anchor_hashes"] = {str(l): _tensor_hash(torch.as_tensor(v)) for l, v in anchors.items()}
    receipt["task_gradient_hashes"] = {str(l): _tensor_hash(v) for l, v in task_gradient.items()}
    receipt["Q_gradient_hashes"] = {str(l): _tensor_hash(v) for l, v in Q_gradient.items()}
    nonzero_layers = [l for l, value in reference_u.items() if bool(torch.count_nonzero(value))]
    if not nonzero_layers:
        proof = native_noop_requests
        if genuine_noop and proof is not None and len(proof) == B and all(x is True for x in proof):
            receipt.update(status="PENDING_GENUINE_NATIVE_NOOP", lambda_Q=None,
                           genuine_native_noop=True, native_noop_requests=list(proof),
                           t=0., n=0., d=0., price_locked=False)
            return CalibrationResult(None, receipt)
        _fail("CALIBRATION_NO_VALID_ENERGY", "zero_reference_without_native_noop_proof", receipt)
    if len(nonzero_layers) != 1:
        _fail("CALIBRATION_IDENTITY_ERROR", "reference_not_single_native_layer", receipt)
    receipt["nonzero_reference_layer"] = str(nonzero_layers[0])
    t_per = torch.zeros(B, dtype=torch.float64)
    n_per = torch.zeros(B, dtype=torch.float64)
    d_per = torch.zeros(B, dtype=torch.float64)
    norm_g = {}; channel_norms = {}
    cap_contacts = {}
    for layer, u in reference_u.items():
        value = u.detach().cpu().double()
        g_task = task_gradient[layer].detach().cpu().double()
        g_Q = Q_gradient[layer].detach().cpu().double()
        beta = norm_coefficient / torch.as_tensor(anchors[layer]).detach().cpu().double()
        lengths = value.norm(dim=0)
        denominator = torch.where(lengths == 0, torch.ones_like(lengths), lengths)
        g_norm = value * (beta / denominator)[None, :]
        norm_g[layer] = g_norm
        t_per -= (g_task * value).sum(0)
        n_per += beta * lengths
        d_per += (g_Q * value).sum(0)
        channel_norms[str(layer)] = dict(task=_json_scalar(g_task.norm()), norm=_json_scalar(g_norm.norm()),
                                        Q=_json_scalar(g_Q.norm()))
        cap_contacts[str(layer)] = dict(relative_norms=lengths.tolist(),
                                        contacts=int((lengths >= .75).sum()),
                                        maximum=_json_scalar(lengths.max()))
    t, n, d = map(float, (t_per.sum(), n_per.sum(), d_per.sum()))
    Q = float(Q.detach().double()) if isinstance(Q, torch.Tensor) else float(Q)
    # Operands stay explicit even on failure; JSON nonfinites are named strings.
    receipt.update(t=_json_scalar(t), n=_json_scalar(n), d=_json_scalar(d),
                   Q=_json_scalar(Q), t_per_request=[_json_scalar(x) for x in t_per],
                   n_per_request=[_json_scalar(x) for x in n_per],
                   d_per_request=[_json_scalar(x) for x in d_per],
                   channel_gradient_norms=channel_norms, cap_contacts=cap_contacts)
    if not all(math.isfinite(v) for v in (t, n, d, Q)):
        _fail("CALIBRATION_NONFINITE", "radial_operands", receipt)
    difference = t - n
    receipt["t_minus_n"] = _json_scalar(difference)
    if not math.isfinite(difference):
        _fail("CALIBRATION_NONFINITE", "radial_subtraction", receipt)
    if d <= 0:
        _fail("CALIBRATION_NO_VALID_ENERGY", "d_nonpositive", receipt)
    if t <= n:
        _fail("CALIBRATION_NO_POSITIVE_PRICE", "t_not_above_n", receipt)
    two_Q = 2 * Q
    receipt["d_minus_two_Q"] = _json_scalar(d - two_Q)
    if not math.isfinite(two_Q):
        _fail("CALIBRATION_NONFINITE", "two_Q_overflow", receipt)
    tolerance = 1e-7 + 1e-5 * abs(two_Q)
    receipt["d_two_Q_absolute_tolerance"] = tolerance
    if abs(d - two_Q) > tolerance:
        _fail("CALIBRATION_RADIAL_Q_IDENTITY", "d_not_two_Q", receipt)
    price = difference / d
    receipt["lambda_candidate"] = _json_scalar(price)
    if not math.isfinite(price):
        _fail("CALIBRATION_NONFINITE", "price_division", receipt)
    if price <= 0:
        _fail("CALIBRATION_NO_POSITIVE_PRICE", "positive_price_underflow", receipt)
    radial = -t_per + n_per + price * d_per
    receipt.update(status="VALID_POSITIVE_RADIAL_PRICE", lambda_Q=price,
                   aggregate_radial_residual=float(radial.sum()),
                   radial_residual_per_request=radial.tolist(),
                   subtraction_relative_cancellation=abs(difference) / max(abs(t), abs(n)),
                   radial_energy_scale=d, genuine_native_noop=False)
    receipt["receipt_hash"] = _digest(receipt)
    return CalibrationResult(price, receipt)


class PriceLock:
    """Create-once compact JSON scalar lock, never a resume bundle."""
    def __init__(self, path):
        self.path = Path(path)

    def write(self, receipt):
        if receipt.get("status") != "VALID_POSITIVE_RADIAL_PRICE":
            raise CalibrationError("CALIBRATION_LOCK_INVALID", dict(receipt))
        price = receipt.get("lambda_Q")
        if not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            raise CalibrationError("CALIBRATION_LOCK_INVALID", dict(receipt))
        raw = dict(receipt); stored_hash = raw.pop("receipt_hash", None)
        if stored_hash != _digest(raw):
            raise CalibrationError("CALIBRATION_LOCK_HASH_IDENTITY", dict(receipt))
        data = _canonical(receipt) + b"\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if self.path.read_bytes() != data:
                raise CalibrationError("CALIBRATION_LOCK_IMMUTABLE", dict(receipt))
            return dict(receipt)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            # Do not replace/remove an interrupted lock; read detects corruption.
            raise
        return dict(receipt)

    def read(self, expected_identity=None):
        value = json.loads(self.path.read_text(encoding="utf-8"))
        raw = dict(value); stored_hash = raw.pop("receipt_hash", None)
        if (stored_hash != _digest(raw) or value.get("status") != "VALID_POSITIVE_RADIAL_PRICE"
                or not math.isfinite(value.get("lambda_Q", math.nan)) or value["lambda_Q"] <= 0):
            raise CalibrationError("CALIBRATION_LOCK_HASH_IDENTITY", value)
        if expected_identity is not None and value.get("identity") != dict(expected_identity):
            raise CalibrationError("CALIBRATION_LOCK_SOURCE_IDENTITY", value)
        return value
