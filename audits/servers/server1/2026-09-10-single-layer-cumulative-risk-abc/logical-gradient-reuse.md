# Logical full-batch gradient reuse

The running cc094028 jobs remain unchanged. That implementation computes
`A @ U.T` and pulls each microbatch gradient back to A separately. For the full
stored P range (observed rank 14326), repeating these large GEMMs is unnecessary.

The next source computes the exact same `W = entry + A @ U.T` once per logical
batch, treats that functional weight as a differentiable leaf, accumulates
microbatch weight gradients in the same paired request/context order, and then
computes `grad_A = grad_W @ U` once. The native-action penalty gradient is added
in A coordinates as before. The two expressions are algebraically identical;
FP32 summation/pullback order is different, not claimed byte-identical. A compact
actual-objective fixture verifies both gradient forms and central differences.
No contexts, requests, steps, rank, normalization, learning rate, momentum,
essence objective or model forward dtype change. No model parameters are
persistently modified by this temporary functional representation.

This is compute-only reuse, not an outcome-driven method change. The original
Middle B execution source remains recorded separately in the run registry.
