# AlphaEdit predictive q-KL projection hook

The current implementation is documented in
[`CACHE_AWARE_QKL_METHOD.md`](./CACHE_AWARE_QKL_METHOD.md).

It reuses an unmodified, identity-sealed stock EasyEdit checkout. The N=1 arm
is the upstream AlphaEdit entry point; N=2/4 split and projected arms are thin
ODE-edit writer bindings. Server-specific model, projector, dataset, and
runtime paths are supplied by launch adapters and are not scientific identity.

`build_qkl_b100_report.py` is the analysis-only create-once atomic-B100
report/manifest/rooted-receipt builder. It keeps PRE-EDIT, stock N=1, and
projected N=2/N=4 in the primary comparison and isolates split controls in an
implementation appendix.
