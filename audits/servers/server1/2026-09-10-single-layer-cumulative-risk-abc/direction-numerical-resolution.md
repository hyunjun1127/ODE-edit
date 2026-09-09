# Direction resolution before B/C execution

The design says not to amplify zero/numerically tiny directions. No efficacy,
margin, risk-improvement, or magnitude-band acceptance threshold is introduced.
The implementation uses working-dtype/dimension arithmetic resolution:
`gamma_k = k*eps/(1-k*eps)` and `gamma_k * norm(abs(A) @ abs(B))_F` for a matrix
product. The coefficient-risk projection and final physical multiplication
budgets are recorded. COV includes its two product budgets; Q is the recorded
orthonormal range basis. These are arithmetic diagnostics, not statistical
noise estimates or global safety certificates.

A direction whose physical norm is zero or unresolved at that arithmetic scale
is recorded UNDEFINED_ZERO_OR_NUMERICALLY_UNRESOLVED_DIRECTION and is not
amplified. Its zero-action trial is retained separately, not silently replaced
with another direction, and it does not block other arms or C. Raw norm,
budget, applied norm and group leakage are published. No scalar is tuned using
NS/PS or the observed performance. Group gradients retain all ten groups; rows
unresolved relative to the largest row at FP32 precision remain explicit zeros.
