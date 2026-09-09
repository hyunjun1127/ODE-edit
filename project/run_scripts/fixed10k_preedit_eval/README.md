# SH2 fixed10k PRE_EDIT evaluation

Instruction: ODEEDIT-S06-FIXED10K-PREEDIT-EVALUATION-SH2-V1.
Evaluation only: PRE_EDIT_W0_FULL10000, Llama revision
8afb486c1db24fe5011ec46dfbe5b5dccdb575c2. No edit/z/key/P/covariance/history,
optimizer/backward, additional model runs, or scientific promotion.

The original SH4 `native/preedit.py`, dataset verifier, metric reducer, canonical
per-prompt NLL kernel and tokenizer contract are reused unchanged from a sealed
local source archive/member closure. `prepare.py` relocates only paths and binds
the server2 implementation, dependency files, model snapshot and fixed dataset.
It deliberately excludes unused BLUE writer/projector/statistics assets.

`observe.py` delegates exactly once to the original evaluator with unchanged
100x100 request slicing and category order; native microbatch=16. It saves the
same returned raw prompt/target rows locally and counts existing root-model
forwards via a read-only hook. No rebatching or additional model forward occurs.
The hook's token-count reduction and filesystem/timing observations are included
in observed overhead, not claimed as zero-cost or bitwise cross-hardware parity.

RS/PS use new NLL < true NLL; NS uses true NLL < new NLL; ties fail. Token
top1/all-target-token secondary correctness remains distinct. Runtime preserves
the native eager/FP32 loading and tokenizer default BOS/right property with
manual left-padding in the exact kernel; actual TF32/CUDA flags are recorded,
not retuned. Selected five weight byte hashes and all-parameter pointer/version
guards are checked after each part. Full nonselected-weight byte hashing is
not claimed. No edit-state cache is initialized; the original empty signature
sentinel is not a native history covariance.

Raw model/cache/prompt/token rows stay outside Git under
`/mnt/raid5/janghj/ODE-edit/local/fixed10k-preedit-eval/attempt-v1/`.
The original SH4 preedit42656 was cancelled before running; receipt verified
before submission. SH4 base jobs and unrelated runs are not modified.

After first-valid.json and its runtime/schema/state checks, leave the job running
and enter MONITORING_PAUSED_AWAITING_GH. Do not add a monitor daemon or loop.
On authorized terminal review, derive full10k and prefix1k/3k tables from saved
prompt rows without new forwards, label W0 rather than current/at-write, and
report exact denominators, NLL summaries and source/input/evaluator identities.

Resource contract: server2 cap2, one GPU, 8 CPUs, 60416MiB, 48h maximum;
held-inspect-release after point-in-time allocation/resource recount.
