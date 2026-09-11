# L4 two-memory v2 execution and canonical provenance

The fixed scientific scope is the pinned `v2-base-preservation` design. This
package is not a lifelong runner and does not authorize additional experiments.

## Technical lineage

- `runtime.py`, `controller.py`, `run.sbatch` are immutable historical source
  `7e515f93`. **Do not launch them as a corrected full campaign.** The historical
  dual-domain check used a matrix-scaled tolerance on a differently dimensioned
  variable and admitted negative multipliers/slack in four iterative paths.
- `controller_repair.py` enforces the same nonnegative domain and uses residual
  bounds in residual units. The original step equations are reused in a private
  function-global namespace; original files/module globals are not changed.
- `repair_runtime.py` / `repair.sbatch` replay only the affected BF8/Frozen paths
  from exact saved geometry, teachers, native/OS state and calibration. N/OS/BF1
  stay unchanged. This is the corrected execution used by the final publication.
- `assemble_repair.py` makes a create-once, **read-only** regular-file hardlink
  view of valid original paths plus replacements. It is not a writer output
  directory. Never write, chmod, or run a writer into the view.
- `observations.py` / `observe.sbatch` consume saved endpoints only. They do not
  edit or select a model. Partial observations are reused only after exact
  bank/source/weight identity checks.

The original Middle-first and primary-three-entry reports are historical
milestones, superseded for BF8/Frozen and whole-campaign validity by the final
report and `technical-exclusions.json`. Their bytes remain preserved.

## Reproduction

Use the commands and exact local inputs in the final report/manifest. Analysis
is split into numeric aggregation, coverage/paired transitions, CPU geometry,
endpoint observations, allocation accounting, interpretation and code-only PNG
rendering. `publication.py` executes PNG rendering twice and requires identical
bytes. Raw tensors, teachers, generated text and full logs remain local-only.

Focused tests:

```bash
python -m unittest -q \
  project.run_scripts.l4_two_memory_conflict_routing.test_core \
  project.run_scripts.l4_two_memory_conflict_routing.test_observer \
  project.run_scripts.l4_two_memory_conflict_routing.test_edges \
  project.run_scripts.l4_two_memory_conflict_routing.test_analysis \
  project.run_scripts.l4_two_memory_conflict_routing.test_repair
```

Tests establish implementation identities, not efficacy/locality guarantees.
All outcomes and technical repair costs are retained; scientific promotion is
false. No additional arm, tuning, threshold relaxation or automatic follow-up
experiment is part of publication.
