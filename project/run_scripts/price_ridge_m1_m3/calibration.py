"""M3 CPU-only root search on supplied saved native B1 K/C0, H=0.

No tensor production/loading or fallback is performed by this function. A caller
must verify immutable saved key/stat identity before passing either tensor.
"""
import statistics
import torch


def calibrate(K, C0, native_lambda, *, key_identity_verified, stat_identity_verified):
    if not key_identity_verified or not stat_identity_verified:
        raise ValueError('M3_SAVED_ASSET_IDENTITY_REQUIRED')
    if K.device.type != 'cpu' or C0.device.type != 'cpu':
        raise ValueError('M3_CPU_SAVED_ONLY')
    if K.dtype != torch.float64 or C0.dtype != torch.float64:
        raise ValueError('M3_NATIVE_REDUCTION_THEN_FP64_REQUIRED')
    if K.ndim != 2 or K.shape[1] != 100 or C0.shape != (K.shape[0],K.shape[0]):
        raise ValueError('M3_B1_MEAN_KEY_SHAPE')
    if not bool(torch.isfinite(K).all() and torch.isfinite(C0).all()) or native_lambda <= 0:
        raise ValueError('M3_NONFINITE_OR_LAMBDA')
    gram=K@K.T; trace=[]
    def measure(value):
        # Raw nonsymmetric stored C0; no symmetrization, jitter or alternate solve.
        T=value*C0+gram
        P=torch.linalg.solve(T,K)
        if not bool(torch.isfinite(P).all()):raise ValueError('M3_SOLVE_NONFINITE')
        residual=float((T@P-K).norm()/K.norm())
        if not residual <= 1e-8:raise ValueError('M3_NATIVE_RIDGE_RESIDUAL')
        median=statistics.median((P.T@K).diagonal().tolist())
        trace.append(dict(lambda_C=value,median=median,residual=residual))
        return median
    high=float(native_lambda); m=measure(high)
    if m >= .5:return dict(status='NATIVE_UNCHANGED',lambda_C=high,median=m,trace=trace)
    low=high
    for _ in range(64):
        low/=2; m=measure(low)
        if abs(m-.5)<=.01:return dict(status='SAVED_B1_CALIBRATED',lambda_C=low,median=m,trace=trace)
        if m>.5:break
        high=low
    else:raise ValueError('M3_POSITIVE_BRACKET_NOT_FOUND')
    bracket=[low,high]
    for _ in range(64):
        mid=(low+high)/2;m=measure(mid)
        if abs(m-.5)<=.01:return dict(status='SAVED_B1_CALIBRATED',lambda_C=mid,median=m,trace=trace,bracket=bracket)
        if m>.5:low=mid
        else:high=mid
    raise ValueError('M3_BISECTION_NOT_CONVERGED')
