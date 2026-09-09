# C0 reduction engineering correction

Source commit cc094028 reads the immutable C0 NPZ, divides its FP32 numerator by
an int64 NumPy scalar, then converts to FP32. NumPy promotes this division to
FP64, while original BLUE SecondMoment.moment divides a Torch FP32 tensor by a
Python integer. A compact fixture reproduced a last-bit difference.

The next source uses the original Torch FP32 division. No NPZ bytes, count,
sampling, covariance definition, or threshold change. Current running A source
and results remain immutable. C0 is **observation-only in A**, absent from its
native writer, direct loss, gradient, and learning-rate selection. Thus no model
editing rerun is required for this diagnostic precision correction. When making
canonical covariance-risk tables, recompute the algebraic quantity from saved
weights with the original C0 reduction; preserve and label the original raw
diagnostic values. No extra model forward is needed. B's covariance-direction
probe will use the corrected source before its first execution.
