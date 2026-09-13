# FzCB-Edit equation-to-asset mapping (server4 initial implementation)

## Authority and fixed boundary

- Method authority: `project/proposals/2026-08-31-fzcb-edit-method-pivot-proposal.md`
  (`47666dada9a95be2a4d4414dcb826990b94150c8dacea08a763ed063049a3853`).
- Execution authority: local create-once contract
  (`0ad22dad8ea63e65e116790c05cf3683db27cf7ea46b1d262d9a400705354073`).
- Base: `ddc178584ef14efd5d4e1271b3c324e3ebd3e443`, tree
  `83889fc3d2e32119dafdffc969482de761eedfe2`.
- First editor: pinned stock EasyEdit MEMIT at
  `/data/janghj/EasyEdit-stock-14cea824`, HEAD
  `14cea8245f06715684592ab55184939b99d70784`.
- Controller-visible state is restricted to registered activations, frozen
  direct-z, current MEMIT keys/factors, the full-model control operator, spent
  action, and the single conditional suffix value. Endpoint metrics never
  influence control or retry.

## Equation-to-code mapping

| Proposal object | Scientific source | Implementation responsibility | Reuse boundary |
|---|---|---|---|
| stock \(a^0,\delta^\star,z^\star\), exactly once/edit | stock `easyeditor.models.memit.memit_main.compute_z` | `fzcb_edit.target` captures per-edit calls and immutable hashes while executing Official MEMIT once | fixed-z Official hook pattern only; no old experiment controller |
| canonical \(\Phi,Z^\star\) and context-correct extension | stock fact lookup/tokenization | `fzcb_edit.target` and `fzcb_edit.geometry` build semantic-position-safe registered batches | fixed-z left-padding/evaluator helpers may be reused |
| MEMIT \(B_l=K_l^T(\lambda C_l^0+K_lK_l^T)^{-1}\) | stock `compute_ks`, `get_cov`, pinned hparams | `fzcb_edit.geometry.MEMITGeometryFactory` calls the stock functions and fixed solves directly | no duplicate key/covariance implementation |
| coefficient gauge removal and \(H=T^TM_0T\) | proposal §§3.4–3.5 | `fzcb_edit.geometry` rank-reveals the request-side Gram and exposes metric-whitened independent coordinates | no ridge-generated authority; only preregistered covariance precision floor |
| \(\mathcal C=J_\Phi T\), \(\mathcal C^T\) | proposal §4 | `fzcb_edit.geometry.FullModelControlOperator` uses `torch.func.jvp/vjp` and factorized MEMIT tangents | K0 operator idea selectively reimplemented; explicit Kronecker/Jacobian forbidden |
| equality-only minimum action and suffix KKT | proposal §§4.5, 6.3 | `fzcb_edit.linear` provides matrix-free Gram solves, hard range residuals, and suffix receipts | no pseudoinverse truncation as scientific success |
| \(h_{cc}=A_0-E-\widehat V_{suf}\), \(\kappa=0\) | proposal §5 | `fzcb_edit.controller` owns the only barrier state | q-KL, F2, facts, locality/history anchors are absent |
| reduced-null suffix sensitivity | proposal §12.3 | `fzcb_edit.controller` uses preregistered symmetric finite differences in a deterministic reduced equality-null basis | explicitly labeled correctness approximation, not exact HVP production |
| one-barrier scalar rectification | proposal §6.2 | `fzcb_edit.linear.scalar_rectification` implements the analytic solution and typed infeasibility | generic QP/multi-barrier solver absent |
| predictor/corrector and net action | proposal §7 | `fzcb_edit.controller` freezes entry \(T,H\), corrects the total coefficient, and accounts the final corrected action | no free corrector jump |
| virtual states, rollback, terminal commit | proposal §§7.7, 13 | `fzcb_edit.transaction` performs exact-copy snapshots and one terminal commit | K0 snapshot idea selectively reimplemented; old promotion logic absent |
| endpoint Eff/Gen/Loc/NLL | existing position-safe evaluator | `fzcb_edit.evaluation` evaluates only after an arm endpoint is sealed | controller/selection influence count is fixed to zero |
| sealed B1/B10 rows | Phase123 single canonical stream | `fzcb_edit.data` reads exact byte ranges and seals request/order identities | payload duplication/reordering forbidden |

## Initial arm semantics

- `OFFICIAL_MEMIT`: the one captured stock endpoint.
- `FROZEN_SPLIT`: four equal progress steps under entry MEMIT geometry; full
  nonlinear closure still checked.
- `REFRESHED_EQUALITY_ONLY`: geometry refresh at every accepted waypoint with
  the barrier disabled.
- `FZCB`: refreshed equality plus the single conditional-completion barrier.
- `STATIC_PATH`: one full-progress, entry-geometry nonlinear corrected solve;
  this is the initial strong static reachability comparator.

The preregistered initial progress grid is `0,.25,.5,.75,1`.  It is fixed
before GPU outcomes and is not presented as an ODE convergence study.

## Exact versus engineering approximation

Exact in the initial implementation: stock direct-z capture, stock MEMIT
key/covariance factors, gauge-free covariance action, matrix-free JVP/VJP
control action, equality/suffix KKT semantics, scalar rectification, entry-basis
corrector accounting, terminal-only commit and rollback.

Engineering approximation: the suffix-value weight sensitivity used by the
barrier is estimated by symmetric finite differences in a deterministic small
equality-null subspace.  It is receipt-labeled
`REDUCED_NULL_FINITE_DIFFERENCE_CORRECTNESS_PROTOTYPE`; no exact-HVP or scalable
production claim is made.
