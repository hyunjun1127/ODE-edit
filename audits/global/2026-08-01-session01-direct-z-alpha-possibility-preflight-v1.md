# Alpha direct-z paired replay v1 execution preflight

## Scope

- Atomic possibility/mechanism diagnostic only; no lifelong, sequential-collapse,
  or method-superiority claim.
- Fixed pair: `llama3-8b-inst` / `qwen2.5-7b-inst`, eight cases each.
- One simultaneous Slurm allocation on `server1/devbox`: two A6000 GPUs,
  `130000M`, twelve hours.  The local aggregate cap is four GPUs; the cap check
  returned `ALLOW` with two active plus two requested GPUs.
- Dedicated ignored session boundary:
  `servers/local/session-boundaries/direct-z-019fbb9a.env`.  The shared GH
  boundary was not modified.

## Read-only baseline and replay contract

- EasyEdit is a read-only dependency.  Its porcelain identity was unchanged
  before/after preflight:
  `f0570aea6efb0b989bff971b5af0dbd20737b808f0127e133e1169df48176f27`.
- Replay lock:
  `8ee67660c167233e95f77cc170011eccf1526d08bd1337f5fd6da612db641f6e`.
- Both models resolve exactly eight immutable artifacts from completed MEMIT-v2
  runs `dzf_llama_p0_v2` and `dzf_qwen_p0_v2`; source manifest, summary,
  artifact bytes, tensor identity, W0 lineage, target-token identity, and the
  per-case MEMIT BF endpoint C-energy are pinned.  Alpha computes direct-z zero
  times and loads it once per case.
- Existing Wikipedia covariance files and AlphaEdit projectors are loaded only;
  no stats/null-space recomputation or download path is used.
- Projector shapes validated from pinned mmap tensors:
  Llama `(5, 14336, 14336)`, Qwen `(5, 18944, 18944)`.
- Pinned Alpha solver/reference configuration IDs:
  Llama `8e665adcb6f7edbabd3cef9547af515c4d972ae301511c55ffefddcc5a44c40a`,
  Qwen `3f322115e91e9ba6414511180036cd65a34d64539b135ac5d7697f31d52e9b17`.

## Mathematical and experimental contract

- Genuine isolated Alpha uses `P` inside the first-edit normal equation via the
  exact low-rank Woodbury form; the post-hoc ablation uses the same identity-P
  Alpha base followed only by right projection.  Dense-equivalence and
  orientation tests pass.
- Natural genuine/post-hoc endpoints are reported separately.  Their matched
  arms and the evaluated four-hop genuine-Alpha BF endpoint use the exact
  paired MEMIT BF C-energy.  The synchronous cone is capped by that energy.
- Ordered Alpha has an exact temporary-write descendant receipt chain and
  byte-identical W0 rollback.  BF refresh uses four genuine Alpha directions.
- The receipt commits before held-out evaluation and directly binds the actual
  BF evaluation endpoint hash, raw path C-energy, normalization scale, and
  final C-energy.  An exact zero cone is retained as a valid no-signal outcome
  with zero projector-leak ratio rather than aborting the run.
- Outcomes separate direct-z fidelity, generated-context error, off-token
  spill, rewrite/paraphrase effects, held-out KL, C-energy, Frobenius norm,
  projector leakage, and compute.  The analyzer uses eight-case ITD and a
  lenient positive-mean plus 5/8 directional gate without model pooling.

## Verification

- `bash -n` passed for the child, pair, and one-shot submission wrappers.
- Full motivation suite: `265` tests passed.
- Real replay, Alpha reference, projector, and resource-cap preflights passed
  for both fixed models.
- Python compilation and `git diff --check` passed.
- The launch wrappers require clean `main == origin/main`, tracked-clean locked
  inputs, the exact verdict below, absent output directories, and a fresh
  one-shot marker before submission.  The helper records the exact submitted
  commit in that marker before `sbatch`; each child requires both live `HEAD`
  and `origin/main` to remain exactly equal to the submitted commit.

- 최종 판정: `PASS` — Alpha direct-z paired replay v1 1회에만 유효
