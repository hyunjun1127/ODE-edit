# E01 explicit USER recall: bounded completed analysis and next warm window

Authority: ODEEDIT-GH-SH1-BASELINE-E01-RESUME-20260912-R1.
Other paused experiments are not recalled. Scientific promotion is false.

- Completed cold source: 4f86f4aa1cfc01539f68b83843c32387006363ce,
  tree06da9784b84f193a80fb9c7967558a8444f4d23b.
- Job45719 COMPLETED0:0, allocation1175 GPU-seconds, not initial-only885.32s.
  Cold native/evaluations are reused, never repeated.
- Cold report SHA ed1b05dfe19dfd6539d037474392a37b0f616ad9e5bb72db6d473c7b99f48b63.
- E1-A report SHA 476c1f5e823adf8fcf5e90bc35a4d57a3c7a7729b8e6a5b00c0e9348a6d0fa24.
  Same10,000 cases across ten arms, not100,000 independent cases.
- Warm entry is existing S1 exact L4 B010, SHA
  b88e96469463d4ef11596486583546bd705476a53f6f1abab9d02e44dab3dafc.
  CPU FP32/finite W/M, metadata hashes, original B011 W/M/context hashes PASS.
  Full RNG inventory exists; actual CUDA restore is a GPU initial gate, not a
  CPU claim. C0 was not restored from the empty checkpoint covariance dict.
- Input lock SHA e06d9d137a535c1b5e73fbfa2da176d0237b421ba88419de5c81c5a565064679.
  Original B020 comparison CP is not yet received. Original trajectory
  equivalence remains unverified, independently of finite native execution.
- Essential CPU suite: 56/56 PASS, unittest discover with `-t .`;
  compile, launch shell syntax, session and resource helper PASS.
  The first discovery invocation omitted `-t .`, producing two test-relative
  import errors; invocation-only correction, no implementation change.
- One technical preparation mismatch: original BLUE request hashing is
  ensure_ascii=True. New-manifest hashing is ensure_ascii=False. Original B011
  request index48 is Unicode and only that hash differed; original serialization
  matches100/100. Reuse source_digest, preserve both conventions, Unicode test.
  Input/raw/native source/sample change0; failed preparation GPU action0.
- Planned next job: L4 n1000 native B011–B020, first B011 shared with E1,
  1GPU/8CPU/182272MiB,8h Slurm limit. Cold measured806.54s native/batch implies
  about2.24h native for ten batches; total estimate2.5–4h including load,
  instrumented native/three panel states/I/O. Estimate is not a budget cap.
  Hourcap null; no old budget inherited. About25GB new raw/checkpoint allowance;
  available storage checked approximately980GB before submission.
- Fresh active plus admitted pending must remain<=2 at held submit/release.
  Next actual first-write gate then MONITORING_PAUSED_AWAITING_USER; no automatic
  terminal polling, follow-up submission, analysis or main integration.

E0/E1 whole campaign, general text, full query exposure, other19 E1 cells and
remaining E0 continuation are not complete. This is not a scientific final report.
