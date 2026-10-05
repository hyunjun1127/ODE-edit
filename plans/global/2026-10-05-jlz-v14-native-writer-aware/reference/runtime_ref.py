"""Small FP64 NumPy references for a semantics-preserving V14 runtime.

This is not a production runner. NumPy's dense solve may refactor B each time;
the tape proves primal-P reuse and the transpose-solve VJP, not GPU factor reuse.
The streaming scheduler models measured row values and masked-loss adjoints.
It never assumes that floating-point KL/NLL has a nonnegative lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable

import numpy as np


def fingerprint(*arrays: np.ndarray) -> str:
    digest = sha256()
    for array in arrays:
        x = np.ascontiguousarray(array)
        digest.update(str((x.shape, x.dtype.str)).encode())
        digest.update(x.tobytes())
    return digest.hexdigest()


def _frozen(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SolveTape:
    K: np.ndarray
    B: np.ndarray
    P: np.ndarray
    binding: str


def solve_forward(A: np.ndarray, K: np.ndarray, ledger: dict | None = None) -> SolveTape:
    """Cache the actual (possibly nonsymmetric) B and its already solved P."""
    A, K = np.asarray(A, dtype=np.float64), np.asarray(K, dtype=np.float64)
    if A.shape != (K.shape[0], K.shape[0]) or K.ndim != 2:
        raise ValueError("incompatible solve shapes")
    if not (np.isfinite(A).all() and np.isfinite(K).all()):
        raise ValueError("nonfinite solve input")
    B = A + K @ K.T
    P = np.linalg.solve(B, K)
    if not np.isfinite(P).all():
        raise ValueError("nonfinite primal solve")
    if ledger is not None:
        ledger["primal_solves"] = ledger.get("primal_solves", 0) + 1
    return SolveTape(_frozen(K), _frozen(B), _frozen(P), fingerprint(A, K))


def solve_vjp(tape: SolveTape, Gp: np.ndarray, *, need_key: bool = True,
              ledger: dict | None = None) -> np.ndarray | None:
    """dK = Z - (Z P^T + P Z^T) K, Z = B^{-T} Gp, with fixed A.

    No primal P solve is repeated here. B^{-1} Gp is incorrect for a
    nonsymmetric stored matrix. The final term is P Z^T K, not K P^T Z.
    """
    if not need_key:
        if ledger is not None:
            ledger["skipped_key_vjps"] = ledger.get("skipped_key_vjps", 0) + 1
        return None
    Gp = np.asarray(Gp, dtype=np.float64)
    if Gp.shape != tape.P.shape or not np.isfinite(Gp).all():
        raise ValueError("invalid P adjoint")
    Z = np.linalg.solve(tape.B.T, Gp)
    if ledger is not None:
        ledger["transpose_solves"] = ledger.get("transpose_solves", 0) + 1
    return Z - Z @ tape.P.T @ tape.K - tape.P @ Z.T @ tape.K


def solve_vjp_cached_A(A: np.ndarray, tape: SolveTape, Gp: np.ndarray, *,
                       A_backend_qualified: bool = True, need_key: bool = True,
                       ledger: dict | None = None) -> np.ndarray | None:
    """Optional fixed-A transpose backend, with native B fallback.

    A^T Z = Gp - K(P^T Gp), derived from A^T B^{-T}=I-KP^T.
    This identity uses the cached native P and holds for nonsymmetric A.
    A production backend must qualify its fixed-A solve/residual numerics;
    dense NumPy here proves only the algebra. An unqualified or exactly
    singular A uses the already defined native B transpose solve instead.
    """
    if not need_key:
        if ledger is not None:
            ledger["skipped_key_vjps"] = ledger.get("skipped_key_vjps", 0) + 1
        return None
    A, Gp = np.asarray(A, dtype=np.float64), np.asarray(Gp, dtype=np.float64)
    if fingerprint(A, tape.K) != tape.binding:
        raise ValueError("fixed A and cached candidate binding mismatch")
    if Gp.shape != tape.P.shape or not np.isfinite(Gp).all():
        raise ValueError("invalid P adjoint")
    if not A_backend_qualified:
        if ledger is not None:
            ledger["native_B_fallbacks"] = ledger.get("native_B_fallbacks", 0) + 1
        return solve_vjp(tape, Gp, ledger=ledger)
    residual = Gp - tape.K @ (tape.P.T @ Gp)
    try:
        Z = np.linalg.solve(A.T, residual)
    except np.linalg.LinAlgError:
        if ledger is not None:
            ledger["singular_A_B_fallbacks"] = ledger.get("singular_A_B_fallbacks", 0) + 1
        return solve_vjp(tape, Gp, ledger=ledger)
    if not np.isfinite(Z).all():
        raise FloatingPointError("nonfinite fixed-A transpose solve")
    if ledger is not None:
        ledger["fixed_A_transpose_solves"] = ledger.get("fixed_A_transpose_solves", 0) + 1
    return Z - Z @ tape.P.T @ tape.K - tape.P @ Z.T @ tape.K


def causal_two_layer(A1, A2, K1, R1, R2, link, skip, base_key, base_out, target,
                     *, backward=False, skip_fixed_first_key=True,
                     stop_upper_solve=False):
    """Smooth R1 -> actual upper K2 -> P2 -> subject output reference.

    Both direct action and upper mean-key/solve dependencies are present.
    Fixed K1's unused key adjoint may be skipped without skipping dR1.
    """
    ledger = {}
    first = solve_forward(A1, K1, ledger)
    U1 = R1 @ first.P.T
    S1 = U1 @ K1
    K2 = np.tanh(base_key + link @ S1)
    second = solve_forward(A2, K2, ledger)
    U2 = R2 @ second.P.T
    S2 = U2 @ K2
    output = np.tanh(base_out + skip @ S1 + S2)
    loss = .5 * np.sum((output - target) ** 2) + .15 * np.sum(S1 ** 2)
    result = dict(loss=float(loss), ledger=ledger, K2=K2, first=first, second=second)
    if not backward:
        return result
    G = (output - target) * (1 - output ** 2)
    Gu2 = G @ K2.T
    Gr2 = Gu2 @ second.P
    Gp2 = Gu2.T @ R2
    Gk2 = U2.T @ G
    if not stop_upper_solve:
        Gk2 += solve_vjp(second, Gp2, ledger=ledger)
    Gs1 = skip.T @ G + .3 * S1 + link.T @ (Gk2 * (1 - K2 ** 2))
    Gu1 = Gs1 @ K1.T
    Gr1 = Gu1 @ first.P
    Gp1 = Gu1.T @ R1
    indirect_first = solve_vjp(first, Gp1, need_key=not skip_fixed_first_key, ledger=ledger)
    result.update(R1_gradient=Gr1, R2_gradient=Gr2,
                  K1_gradient=None if indirect_first is None else U1.T @ Gs1 + indirect_first)
    return result


def owner_mean(X: np.ndarray, owners: np.ndarray, weights: np.ndarray, B: int) -> np.ndarray:
    """Native aggregation weights need not equal objective row weights."""
    K = np.zeros((X.shape[0], B), dtype=np.float64)
    for column, owner in enumerate(owners):
        K[:, owner] += X[:, column] * weights[column]
    return K


def whole_batch_rows(A, X, R, owners, aggregate_weights, target, chunks):
    """One whole-B writer; unequal owner row counts; partitioned SUM adjoint."""
    owners = np.asarray(owners, dtype=np.int64)
    B = R.shape[1]
    counts = np.bincount(owners, minlength=B)
    if counts.shape != (B,) or (counts <= 0).any():
        raise ValueError("each owner must have native rows")
    order = [column for chunk in chunks for column in chunk]
    if order != list(range(X.shape[1])):
        raise ValueError("partitions must preserve original row order")
    weights = 1. / counts[owners]
    K = owner_mean(X, owners, aggregate_weights, B)
    ledger = {}
    tape = solve_forward(A, K, ledger)
    U = R @ tape.P.T
    loss = np.float64(0.)
    Gu = np.zeros_like(U)
    Gx = np.zeros_like(X)
    per_owner = np.zeros(B)
    for chunk in chunks:
        index = np.asarray(chunk, dtype=np.int64)
        output = np.tanh(U @ X[:, index])
        error = output - target[:, index]
        G = error * (1 - output ** 2) * weights[index]
        # Preserve canonical row accumulation even at different chunk widths.
        for local, column in enumerate(index):
            value = np.float64(.5 * np.sum(error[:, local] ** 2) * weights[column])
            loss += value
            per_owner[owners[column]] += value
            Gu += np.outer(G[:, local], X[:, column])
            Gx[:, column] = U.T @ G[:, local]
    Gr = Gu @ tape.P
    Gp = Gu.T @ R
    Gk = solve_vjp(tape, Gp, ledger=ledger)
    for column, owner in enumerate(owners):
        Gx[:, column] += Gk[:, owner] * aggregate_weights[column]
    return dict(loss=float(loss), owner_losses=per_owner, R_gradient=Gr,
                X_gradient=Gx, ledger=ledger, tape=tape)


@dataclass(frozen=True)
class Row:
    owner: int
    component: str  # "nll" or "kl"; value already has native row normalization
    value: float
    gradient: np.ndarray  # derivative of this value w.r.t. a fixed toy variable


def _row_identity(rows) -> str:
    digest = sha256()
    for row in rows:
        digest.update(str((row.owner, row.component)).encode())
        digest.update(np.float64(row.value).tobytes())
    return digest.hexdigest()


def stream_candidate(groups, norm_values, norm_gradient, *, memory_groups=2,
                     final_cap=False, threshold=.05, kl_factor=.0625,
                     state_identity="same-candidate", replay_transform: Callable | None = None):
    """Exact measured-complete-owner witness for synchronous stop scheduling.

    An observed finite COMPLETE owner J >= threshold disproves all-owner stop.
    Until that witness, retain at most memory_groups prefix group graphs. The
    current live group is a separate working set, not part of this retention
    limit. Evicted prefix graphs are replayed only after a witness, in original
    backward order. Values are never replay-accumulated. A late invalid value or
    gradient aborts the entire update and discards accumulated adjoints.

    This toy has no optimizer/model and treats supplied row gradients as a
    graph payload. Production must also bind W/H/teacher/token/lookup/candidate
    identity and independently qualify its forward and adjoint implementations.
    """
    B = len(norm_values)
    norm_values = np.asarray(norm_values, dtype=np.float64)
    norm_gradient = np.asarray(norm_gradient, dtype=np.float64)
    if B == 0 or memory_groups < 0 or not groups:
        raise ValueError("invalid schedule shape")
    expected = np.zeros(B, dtype=np.int64)
    for group in groups:
        if not group:
            raise ValueError("empty row group")
        for row in group:
            if row.owner < 0 or row.owner >= B or row.component not in ("nll", "kl"):
                raise ValueError("invalid row identity")
            expected[row.owner] += 1
    if (expected == 0).any():
        raise ValueError("incomplete owner metadata")
    seen = np.zeros(B, dtype=np.int64)
    nll, kl = np.zeros(B), np.zeros(B)
    gradient = np.zeros_like(norm_gradient)
    graph_buffer, certificates = {}, {}
    forward_order, replay_order, backward_order, evicted = [], [], [], []
    witness = None
    peak_retained = 0
    norm_additions = 0
    aborted = None

    def forward(index, replay=False):
        rows = groups[index]
        if replay and replay_transform is not None:
            rows = replay_transform(index, rows)
        (replay_order if replay else forward_order).append(index)
        if not all(np.isfinite(row.value) for row in rows):
            raise FloatingPointError("NONFINITE_ROW_VALUE")
        return rows

    def backward(index, rows):
        if index != len(backward_order):
            raise AssertionError("noncanonical gradient order")
        for row in rows:
            if row.gradient.shape != gradient.shape or not np.isfinite(row.gradient).all():
                raise FloatingPointError("NONFINITE_OR_INVALID_ROW_GRADIENT")
            gradient[:] += row.gradient * (kl_factor if row.component == "kl" else 1.)
        backward_order.append(index)

    try:
        if not (np.isfinite(norm_values).all() and np.isfinite(norm_gradient).all()):
            raise FloatingPointError("NONFINITE_REQUESTED_NORM")
        for index in range(len(groups)):
            rows = forward(index)
            certificates[index] = (state_identity, _row_identity(rows))
            for row in rows:
                (nll if row.component == "nll" else kl)[row.owner] += row.value
                seen[row.owner] += 1
            J = nll + kl_factor * kl + norm_values
            if not np.isfinite(J).all():
                raise FloatingPointError("NONFINITE_PARTIAL_OBJECTIVE")
            if final_cap:
                continue  # Last candidate is always value-only, including a witness.
            if witness is not None:
                backward(index, rows)
                continue
            graph_buffer[index] = rows
            complete = np.flatnonzero(seen == expected)
            failing = [int(owner) for owner in complete if J[owner] >= threshold]
            if failing:
                owner = failing[0]
                witness = dict(group=index, owner=owner, J=float(J[owner]),
                               owner_rows=int(seen[owner]), expected_rows=int(expected[owner]))
                for previous in range(index + 1):
                    payload = graph_buffer.pop(previous, None)
                    if payload is None:
                        payload = forward(previous, replay=True)
                        if (state_identity, _row_identity(payload)) != certificates[previous]:
                            raise FloatingPointError("REPLAY_CERTIFICATE_MISMATCH")
                    backward(previous, payload)
            else:
                while len(graph_buffer) > memory_groups:
                    evicted.append(min(graph_buffer))
                    del graph_buffer[min(graph_buffer)]
                peak_retained = max(peak_retained, len(graph_buffer))
        if not np.array_equal(seen, expected):
            raise AssertionError("missing rows")
        J = nll + kl_factor * kl + norm_values
        stop_all = bool((J < threshold).all())
        if not final_cap and not stop_all:
            if witness is None or len(backward_order) != len(groups):
                raise AssertionError("missing stop witness or adjoint")
            gradient += norm_gradient
            norm_additions = 1
            if not np.isfinite(gradient).all():
                raise FloatingPointError("NONFINITE_TOTAL_GRADIENT")
        else:
            gradient[:] = 0
    except FloatingPointError as error:
        aborted = str(error)
        gradient[:] = 0
        norm_additions = 0
        stop_all = False
        J = nll + kl_factor * kl + norm_values
    allowed = aborted is None and not final_cap and not stop_all
    return dict(status="ABORTED" if aborted else "VALID", error=aborted,
                update_allowed=allowed, stop_all=stop_all, final_cap=final_cap,
                J=J, nll=nll, kl=kl, gradient=gradient, witness=witness,
                forward_order=forward_order, replay_order=replay_order,
                backward_order=backward_order, evicted_prefix_groups=evicted,
                peak_retained_groups=peak_retained, norm_gradient_additions=norm_additions,
                backward_calls=len(backward_order), forward_calls=len(forward_order)+len(replay_order))


def probe_candidate(groups, norm_values, norm_gradient, *, final_cap=False,
                    threshold=.05, kl_factor=.0625, state_identity="same-candidate",
                    replay_transform: Callable | None = None):
    """Default detached complete-owner probe, with exact full-value fallback.

    Choose the largest requested norm penalty; ties use original owner order.
    Evaluate the ORIGINAL row groups that contain all that owner's rows, in
    their original order. A measured COMPLETE J >= threshold proves no common
    stop. Then one full native F/B pass accumulates in original row-group order.
    If the witness fails, reuse detached probe-group values and evaluate only
    missing groups, reduce ALL values canonically, and decide the common stop.
    No probe is necessary for the final cap, and no positivity bound is used.
    """
    norm_values = np.asarray(norm_values, dtype=np.float64)
    norm_gradient = np.asarray(norm_gradient, dtype=np.float64)
    B = len(norm_values)
    if B == 0 or not groups:
        raise ValueError("invalid probe schedule shape")
    counts = np.zeros(B, dtype=np.int64)
    for rows in groups:
        if not rows:
            raise ValueError("empty group")
        for row in rows:
            if not 0 <= row.owner < B or row.component not in ("nll", "kl"):
                raise ValueError("invalid row metadata")
            counts[row.owner] += 1
    if (counts == 0).any():
        raise ValueError("missing owner rows")
    detached, identities = {}, {}
    value_order, training_order, backward_order = [], [], []
    gradient = np.zeros_like(norm_gradient)
    witness, error = None, None
    chosen, probe_groups = None, []
    nll, kl = np.zeros(B), np.zeros(B)
    J = norm_values.copy()
    fallback = False
    norm_additions = 0

    def forward(index, training=False):
        rows = groups[index]
        if training and replay_transform is not None:
            rows = replay_transform(index, rows)
        (training_order if training else value_order).append(index)
        if not all(np.isfinite(row.value) for row in rows):
            raise FloatingPointError("NONFINITE_ROW_VALUE")
        identity = (state_identity, _row_identity(rows))
        if index in identities and identity != identities[index]:
            raise FloatingPointError("REPLAY_CERTIFICATE_MISMATCH")
        identities[index] = identity
        return rows

    def reduce(values):
        loss_n, loss_k = np.zeros(B), np.zeros(B)
        for index in range(len(groups)):
            for row in values[index]:
                (loss_n if row.component == "nll" else loss_k)[row.owner] += row.value
        total = loss_n + kl_factor * loss_k + norm_values
        if not np.isfinite(total).all():
            raise FloatingPointError("NONFINITE_OBJECTIVE")
        return loss_n, loss_k, total

    try:
        if not (np.isfinite(norm_values).all() and np.isfinite(norm_gradient).all()):
            raise FloatingPointError("NONFINITE_REQUESTED_NORM")
        if final_cap:
            for index in range(len(groups)):
                detached[index] = forward(index)
            nll, kl, J = reduce(detached)
        else:
            chosen = int(np.argmax(norm_values))  # Stable first-owner tie break.
            probe_groups = [index for index, rows in enumerate(groups)
                            if any(row.owner == chosen for row in rows)]
            owner_nll = np.float64(0.)
            owner_kl = np.float64(0.)
            observed = 0
            for index in probe_groups:
                detached[index] = forward(index)
                for row in detached[index]:
                    if row.owner == chosen:
                        if row.component == "nll":
                            owner_nll += row.value
                        else:
                            owner_kl += row.value
                        observed += 1
            if observed != counts[chosen]:
                raise AssertionError("partial probe owner")
            owner_J = owner_nll + kl_factor * owner_kl + norm_values[chosen]
            if not np.isfinite(owner_J):
                raise FloatingPointError("NONFINITE_PROBE_OBJECTIVE")
            if owner_J >= threshold:
                witness = dict(owner=chosen, J=float(owner_J), owner_rows=int(observed),
                               expected_rows=int(counts[chosen]), source="COMPLETE_OWNER_PROBE")
            else:
                fallback = True
                for index in range(len(groups)):
                    if index not in detached:
                        detached[index] = forward(index)
                nll, kl, J = reduce(detached)
                failures = np.flatnonzero(J >= threshold)
                if len(failures):
                    owner = int(failures[0])
                    witness = dict(owner=owner, J=float(J[owner]), owner_rows=int(counts[owner]),
                                   expected_rows=int(counts[owner]), source="FULL_VALUE_FALLBACK")
            if witness is not None:
                # Values below model row-loss extraction are detached scalars.
                # This list is a toy payload; production retains one live graph.
                nll, kl = np.zeros(B), np.zeros(B)
                for index in range(len(groups)):
                    rows = forward(index, training=True)
                    for row in rows:
                        (nll if row.component == "nll" else kl)[row.owner] += row.value
                        if row.gradient.shape != gradient.shape or not np.isfinite(row.gradient).all():
                            raise FloatingPointError("NONFINITE_OR_INVALID_ROW_GRADIENT")
                        gradient += row.gradient * (kl_factor if row.component == "kl" else 1.)
                    backward_order.append(index)
                J = nll + kl_factor * kl + norm_values
                if not np.isfinite(J).all():
                    raise FloatingPointError("NONFINITE_OBJECTIVE")
                gradient += norm_gradient
                norm_additions = 1
                if not np.isfinite(gradient).all():
                    raise FloatingPointError("NONFINITE_TOTAL_GRADIENT")
    except FloatingPointError as problem:
        error = str(problem)
        gradient[:] = 0
        norm_additions = 0
    stop_all = error is None and bool((J < threshold).all())
    update = error is None and not final_cap and witness is not None
    if not update:
        gradient[:] = 0
    return dict(status="ABORTED" if error else "VALID", error=error,
                update_allowed=update, stop_all=stop_all, final_cap=final_cap,
                chosen_owner=chosen, probe_groups=probe_groups, witness=witness,
                fallback_full_value=fallback, J=J, nll=nll, kl=kl, gradient=gradient,
                value_forward_order=value_order, training_forward_order=training_order,
                backward_order=backward_order, backward_calls=len(backward_order),
                forward_calls=len(value_order)+len(training_order),
                norm_gradient_additions=norm_additions)
