# SH4 → GH P4 Euler E0 / T_Z_CALIBRATION_PLAN

- instruction: `ODEEDIT-S05-P4-EULER-PROJECTED-SEMANTIC-ODE-V1`
- `FULL_READ_PASS`: regular non-symlink, mode `0600`, 21,620 bytes,
  989 newline records, SHA
  `1899bbfa3610ee2edc94044548475dbb39353ebb7aeafeea371c9a879e7d4398`;
  envelope/addendum conflict 없음.
- implementation commit/tree:
  `ae53e18235b605027944cb57a738c26c22550bb1` /
  `f18ccad77be37ad751b3d805940e87fd7c83867f`.
- E0: pure raw projected Euler integrator + thin semantic binding +
  receding-horizon/cache/K-target/Native-one-shot/FP32 runtime contracts 구현.
  Old Adam solver import/call 0; optimizer/Adam/SGD/moment/backward/parameter-grad
  authority 0; microstep당 aggregate `autograd.grad` 1회; final-only.
- gates: `py_compile PASS`, authoritative §17 focused `17/17 PASS`, source
  manifest PASS, no-model dry-plan `E0_PRE_CALIBRATION_PASS_GPU_HOLD`.
- identities: numerical-lock root `f0393f46...e61f`, source-manifest root
  `26ff6849...3ed`, dry-plan receipt `50098528...8d1`.
- `T_z=null`, calibration approval required, model load/GPU/Slurm/heldout
  `0/0/0/0`.
- proposed Stage 1: common fixed `h={0.0625,0.25,1,4}` ×
  `M={1,3,5,10}`; `T_z=Mh` strength/pseudo-time curve, refinement claim 0.
- Stage 1 rule: 양 model/arm finite/W-freeze/Adam0, clamp fraction `<0.5`,
  displacement coherence를 만족하는 largest safe `h`와 다음 unsafe `h`의
  closed bracket; `T_z=5*h_selected`. No bracket이면 HOLD.
- proposed Stage 2: fixed `T_z`, M5 vs M10; request endpoint relative
  discrepancy median≤0.10, p90≤0.25, max≤0.50, clamp fraction<0.50. 초과 시
  HOLD.
- calibration sample proposal: sealed `B1_CASE01` train-only,
  permanent calibration label, confirmatory denominator 제외(B2–B10). 별도
  calibration-only sealed unit 제공 시만 B1 복귀. heldout/Native/final
  performance/A±−A+ selection influence 0.
- approval requested: 위 `h` grid/selection thresholds와 B1 partition.
- detail:
  `plans/updates/server4/2026-08-22-p4-euler-tz-calibration-plan.md`,
  `audits/servers/server4/2026-08-22-p4-euler-e0.md`.
- verdict: `E0_PRE_CALIBRATION_PASS / T_Z_APPROVAL_PENDING / SCIENTIFIC_SUBMIT_HOLD`.
