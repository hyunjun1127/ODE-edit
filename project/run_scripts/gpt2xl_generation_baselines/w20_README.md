# Three GPT2-XL native baselines: W20-only generation

New task/nonce only: AlphaEdit-BLUE, PRUNE and RECT each start independent cold
models and run the unchanged first2000 / BS100 × 20 native trajectory. RPN W0,
current pre/post and W5/10/15/20 measurements are unchanged. Fluency/consistency
generation is measured only for the actual W20 ordered first2000 state.
There is no generation W0 READY or cross-arm observation prerequisite.

Preparation reuses the exact prior model/token/native assets, generation
reference and prelocked ≤8-prompt qualification PLAN. Each new arm writes its
own actual route receipt inside its approved job; a source/CPU test is not GPU
qualification or online PASS. The original checkpoint/raw/source/cost remain
untouched. No model/C0/P downloads or recomputation occur.

The owner uses these task-private modules:

1. `w20_prepare --out <new CPU directory> --attempt <new local attempt> --source-attempt <prior cache-repair attempt> --transition-receipt <exact root reconciliation JSON>`
2. `w20_preflight --config <new CPU directory/config.json>`
3. Commit the reviewed scope, then `w20_submit --config <new CPU directory/config-sealed.json> --attempt <new local attempt>`.

Submission checks the exact new nonce before any registration. A started
registration pass, even without a returned job ID, is not blindly retried.
The three GPU jobs and target-only GPU0 collector are held, full-argv/source/
resource/dependency inspected, then released. All dependencies are `afterany`
resource barriers: cap2 permits BLUE + PRUNE, then RECT in the BLUE lane; a
stricter cap1 serializes all three. Existing current owner resource frontier is
retained; cancelled old target IDs are not used. Other users/servers/jobs are
never cancelled or preempted.

The local session receipt records App root CWD separately from this dedicated
non-main WT and the historical registry29e4 path. Source/config/input/PLAN/
reference/native/launcher/collector locks are immutable. Raw generated text and
tokens stay local; W&B contains approved scalar counts/progress/final metrics
only. CPU/static, actual runtime qualification, remote delivery and scientific
completion are separate states.
