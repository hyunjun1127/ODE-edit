"""One immutable cross-arm allocation unit receipt, never a strength sweep."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from project.run_scripts.jlz_realization.common import digest, tensor_sha, write
from project.run_scripts.jlz_shared_budget.optimizer import norm_gradient
from .geometry import requested_action, cost_gradient_u


class CalibrationError(RuntimeError):
    pass


class CalibrationLock:
    """Producer creates once; consumer verifies the same first cold candidate.

    A missing consumer lock fails immediately.  Dispatch must run CD_Q before
    CD_C at cap1; waiting on a queued producer would deadlock that allocation.
    Once verified, resolve only returns the sealed value for all later batches.
    """
    def __init__(self, path, *, role, identity):
        if role not in ('producer', 'consumer'):
            raise ValueError('calibration role must be producer or consumer')
        self.path = Path(path)
        self.role = role
        self.identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
        self.value = None
        self.receipt = None
        self.status = 'PENDING_FIRST_NONZERO_COLD_CANDIDATE'
        self.verified = False
        self._existing = self._read() if self.path.exists() else None

    def _read(self):
        receipt = json.loads(self.path.read_text())
        if receipt.get('schema') != 'JLZ_CD_CUMULATIVE_LAMBDA_LOCK_V1':
            raise CalibrationError('LAMBDA_LOCK_SCHEMA')
        if receipt.get('identity') != self.identity:
            raise CalibrationError('LAMBDA_LOCK_SOURCE_INPUT_W0_PLAN_IDENTITY')
        coefficient = receipt.get('lambda_alloc')
        if not isinstance(coefficient, (float, int)) or not math.isfinite(coefficient) or coefficient <= 0:
            raise CalibrationError('LAMBDA_LOCK_NONPOSITIVE_NONFINITE')
        core = {k: v for k, v in receipt.items() if k != 'receipt_hash'}
        if receipt.get('receipt_hash') != digest(core):
            raise CalibrationError('LAMBDA_LOCK_RECEIPT_HASH')
        return receipt

    @torch.no_grad()
    def resolve(self, *, candidate, geometries, blocks, anchors, prices, delta_zero):
        if self.verified:
            return self.value
        if not delta_zero or not all(g.receipt.get('delta_zero', False) for g in geometries.values()):
            raise CalibrationError('CALIBRATION_PENDING_REQUIRES_ACTUAL_ZERO_DISPLACEMENT')
        layers = list(blocks)
        if set(layers) != set(geometries) or set(layers) != set(anchors):
            raise CalibrationError('CALIBRATION_LAYER_COVERAGE')
        if not isinstance(candidate, int) or not 0 <= candidate <= 24:
            raise CalibrationError('CALIBRATION_CANDIDATE_RANGE')
        if not layers:
            raise CalibrationError('CALIBRATION_EMPTY_LAYERS')
        B = blocks[layers[0]].shape[1]
        if not isinstance(prices, torch.Tensor):
            prices = torch.tensor(prices, dtype=torch.float64)
        if prices.shape != (B,) or not bool(torch.isfinite(prices).all()) or not bool((prices > 0).all()):
            raise CalibrationError('CALIBRATION_NATIVE_NORM_PRICES')
        action_nonzero = False
        plan_nonzero = False
        norm_gradients = {}; q_gradients = {}; hashes = {}; operators = {}
        norm_square = torch.zeros((), dtype=torch.float64, device='cpu')
        q_square = torch.zeros((), dtype=torch.float64, device='cpu')
        for layer in layers:
            u = blocks[layer].detach(); a = anchors[layer].detach(); g = geometries[layer]
            D = requested_action(u, a)
            Y = D.double() @ g.S.to(D.device)
            plan_nonzero = plan_nonzero or bool(torch.count_nonzero(D))
            action_nonzero = action_nonzero or bool(torch.count_nonzero(Y))
            norm_g = torch.stack([norm_gradient(u[:, r], float(prices[r])) for r in range(B)], dim=1)
            q_g, _ = cost_gradient_u(u, a, g, alpha=0)
            if not bool(torch.isfinite(norm_g).all()) or not bool(torch.isfinite(q_g).all()):
                raise CalibrationError('CALIBRATION_NONFINITE_GRADIENT')
            norm_square += norm_g.double().square().sum().cpu()
            q_square += q_g.double().square().sum().cpu()
            norm_gradients[str(layer)] = tensor_sha(norm_g)
            q_gradients[str(layer)] = tensor_sha(q_g)
            hashes[str(layer)] = dict(u=tensor_sha(u), anchors=tensor_sha(a),
                                      D=tensor_sha(D), Y=tensor_sha(Y))
            operators[str(layer)] = g.operator_hash
        if not action_nonzero:
            self.status = 'PENDING_ZERO_PROJECTED_ACTION'
            return None
        norm_norm = float(norm_square.sqrt()); q_norm = float(q_square.sqrt())
        if not math.isfinite(norm_norm) or not math.isfinite(q_norm) or norm_norm <= 0 or q_norm <= 0:
            raise CalibrationError('CALIBRATION_NONZERO_ACTION_ZERO_OR_NONFINITE_GRADIENT')
        coefficient = norm_norm / q_norm
        if not math.isfinite(coefficient) or coefficient <= 0:
            raise CalibrationError('CALIBRATION_NONFINITE_RATIO')
        receipt = dict(schema='JLZ_CD_CUMULATIVE_LAMBDA_LOCK_V1', identity=self.identity,
                       candidate=candidate, requests=B, candidate_hashes=hashes,
                       plan_hash=digest(hashes), operator_hashes=operators,
                       norm_gradient_hashes=norm_gradients, Q_gradient_hashes=q_gradients,
                       prices_hash=tensor_sha(prices), norm_gradient_norm=norm_norm,
                       Q_gradient_norm=q_norm, lambda_alloc=coefficient,
                       gradient_reduction='FP64 Frobenius; unchanged FP32 stored-u cast chain',
                       target_gradient_ratio=1., alpha_in_calibration=0,
                       actual_delta_zero=True, requested_nonzero=plan_nonzero,
                       projected_action_nonzero=True, no_epsilon_clip_default=True,
                       shared_arms=['CD_Q', 'CD_C'], immutable=True, persisted_tensors=False)
        receipt['receipt_hash'] = digest(receipt)
        if self.role == 'producer':
            # write uses exclusive temp + hard-link publication.  An existing
            # receipt is accepted only byte-for-byte; it is never overwritten.
            try:
                write(self.path, receipt)
            except RuntimeError as error:
                raise CalibrationError('LAMBDA_LOCK_CREATE_ONCE_CONFLICT') from error
            self._existing = self._read()
        else:
            if not self.path.exists():
                raise CalibrationError('LAMBDA_LOCK_MISSING_RUN_PRODUCER_FIRST_NO_WAIT')
            self._existing = self._read()
        if self._existing != receipt:
            raise CalibrationError('LAMBDA_LOCK_COLD_CANDIDATE_OPERATOR_GRADIENT_MISMATCH')
        self.value = float(self._existing['lambda_alloc'])
        self.receipt = self._existing
        self.verified = True
        self.status = 'SEALED_PRODUCER' if self.role == 'producer' else 'SEALED_CONSUMER_VERIFIED'
        return self.value
