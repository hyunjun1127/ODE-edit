# FzCB-Edit initial MEMIT implementation

This package implements the 2026-08-31 FzCB-Edit proposal for pinned stock
MEMIT.  It captures each stock direct-z exactly once, treats the MEMIT writer as
a gauge-free metric-whitened operator, tracks the frozen target with full-model
JVP/VJP equalities, and applies one conditional-completion barrier.

Responsibilities are split across `target`, `geometry`, `linear`,
`controller`, `transaction`, `evaluation`, `data`, `preflight`, and `runtime`.
Endpoint evaluation is intentionally not imported by the controller.

The initial barrier sensitivity is a reduced-null symmetric finite-difference
correctness prototype.  Receipts do not label it as the production exact-HVP
method.  F2/q-KL/reference-fact barriers and prior K0 promotion logic are not
imported.

