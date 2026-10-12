# Conditional clamp 0.5 final CF Flu/Con

New opt-in wrapper; no editing-source change, submission, or execution gate.
Old e87 Flu/Con and 59c531d8 editing sources remain byte-identical.

```sh
python -m official.diagnostics.fe_qwen_current.flucon_clamp prepare NEW_ROOT BINDINGS.json
PYTHONPATH=NEW_ROOT/source python -m official.diagnostics.fe_qwen_current.flucon_clamp run NEW_ROOT/config.json NEW_ROOT/lock.json
PYTHONPATH=NEW_ROOT/source python -m official.diagnostics.fe_qwen_current.flucon_clamp collect NEW_ROOT/config.json NEW_ROOT/lock.json --output NEW_REPORT.json
```

Bindings are identical to FLUCON.md: server=rent, actual final_result member,
original CF stream member, reference manifest member, absolute tracking_env_file,
reserve_bytes>=32GiB, raw_margin_bytes>=4GiB. Root supplies actual absolute paths.
Use the existing separate scoring runtime with exact locked reference packages.
No new download, reference fitting, CP copy, or rent lookup occurs in preparation.

Requires independently re-audited rescue_clamp selected L2=100/clamp=0.5 proof,
both final cold CF/zsRE W20 raw/query/20-commit proofs, and both native runtime/CP
identities and resolved configs with those exact knobs. Quality-stop, failed,
partial, old clamp1 or heldout CP cannot substitute for final CF W20.
No missing-proof fallback; no fabricated final result exists in this publication.

Canonical NativeGenerationObserver, profile/seed20261007, original CF first2000,
reference75e595c7f26ec334830e9bb9ca6028098c19ea84a9509a5713985847683f8ea6,
single generation shared by Flu/Con and independent raw scoring are unchanged.
Native BF16 selected weights are full-file-SHA verified, mmap read and exactly
cast/round-tripped into FP32 eager TF32-off evaluation. Unselected model parameters
and evaluation RNG remain guarded. H/context/checkpoint RNG are not consumed or
cloned. Editing precision and native solver are untouched.

Task/group: fe-qwen-current-z-clamp-0p5-final-cf-flucon-20261012. Standard shared
eval_only generation-only schema, actual job identity, edits2000, no factual/fit
or W0/current generation. Raw bits/cosine unchanged; paper display x100 half-up2.
The immutable config links original final run/CP identity and final-result proof.
CPU fixture success is not a real selected result, GPU generation or recovery.
