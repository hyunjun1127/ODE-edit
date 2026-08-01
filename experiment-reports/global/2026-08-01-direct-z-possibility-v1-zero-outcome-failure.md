# Direct-z possibility v1: zero-outcome technical failure

## Disposition

`odeedit_dzf_pair_v1` job `15739` is a **technical failure**, not a completed
possibility diagnostic.  It produced no endpoint outcome records, so it cannot
support any scientific interpretation of direct-z fidelity, edit quality,
preservation, BF, DF, or the locked arms.

## Observed failure boundary

- The Slurm parent requested the locked pair envelope (`2` GPUs, `16` CPUs,
  `130000M`) and ended `FAILED (1:0)` after `00:01:46`.
- Both model checkpoints loaded.  The local trace identifies a `TypeError` in
  `run_direct_z_possibility` / `_run_event`; the available trace contains no
  message or deeper frame, so it is insufficient to attribute a source-level
  cause here.
- The failing child caused the paired launcher to terminate its sibling
  fail-fast.  Slurm records one child `FAILED (1:0)` and the other `CANCELLED
  by 0 (0:9)`; this report does not assign those child indices to models.
- The failure occurred before the required precommit artifacts and endpoint
  outcomes: the Llama summary records `receipt_before_outcome=false`,
  `direct_z_once_per_case=false`, `all_rollbacks_exact=false`, and
  `firewall_pass=false`.

## Exact artifact counts

| Run | Planned cases | Attempted | Pass | Failed | Outcome records | Event records | Summary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dzf_llama_p0_v1` | 8 | 1 | 0 | 8 | 0 | 1 failure event | present |
| `dzf_qwen_p0_v1` | 8 (manifest) | — | — | — | 0 | 0 | absent |

The locked arm count is six per case, hence the intended pair would have had
`2 × 8 × 6 = 96` outcomes.  Actual outcome count is exactly `0`.  The sole
Llama event is a `TypeError` failure event for one case; it is not an arm
outcome.  Qwen has a manifest but no summary or events because of sibling
termination.

## v2 repair conditions

1. Identify and eliminate the precommit `TypeError`, retaining its full local
   exception context only in local logs, not in scalar artifacts.
2. Before resubmission, add an `inspect.signature.bind`-based cross-module API
   contract test for the precommit call boundary, run the full CPU regression,
   and pass a clean v2 pair preflight.  A separate one-case GPU smoke is not a
   required audit.
3. Use fresh, exclusive v2 run IDs and preserve the locked models, case slice,
   arm order, caches, and two-GPU envelope.  Do not overwrite v1 artifacts.
4. Require terminal summaries for both children with eight attempted cases,
   six outcomes per case, and true direct-z-once, receipt-order, firewall, and
   rollback fields before any model or pair analysis is commissioned.

Until those conditions hold, v1 remains a zero-outcome technical incident.
