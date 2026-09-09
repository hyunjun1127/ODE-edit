# W0 panel reuse accounting deviation

The three native processes independently evaluate their full W0 panels.
Fixed100 is shared, and other W0 requests can overlap across historical-entry
panels. Therefore the current A execution does not achieve global cross-entry
W0 row reuse, despite avoiding full/curve repeated evaluation within an
endpoint. Do not report a global duplicate-evaluation count of zero.

This is redundant computation, not a new sample or independent replication.
The original per-entry measurements and all actual costs remain recorded.
Already valid jobs are not cancelled or rerun to hide this deviation. An
analysis-only identity audit must quantify repeated W0 rows and any numerical
variation; do not average/reconstruct a replacement baseline. B/C consume A's
existing W0/We/N measurements and do not launch new baseline evaluations.

The stage data remain usable with separate entry-conditioned denominators.
This deviation is disclosed in the report/requirements accounting and is not
used as a performance exclusion or a reason to invent additional experiments.
