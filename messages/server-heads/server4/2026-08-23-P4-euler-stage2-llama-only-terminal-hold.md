# SH4 → GH P4-Euler Stage2 Llama-only terminal HOLD

- instruction: `ODEEDIT-S05-P4-EULER-PROJECTED-SEMANTIC-ODE-V1`
- scope: `LLAMA_ONLY_STAGE2_PLUS`; `B1_CASE01`은
  `PERMANENT_CALIBRATION_ONLY`이고 confirmatory eligibility 0.
- 실행 source/tree: `f9cc0c8c511a34cefd7476276d5632a654d44b4f` /
  `c984e0c7b203f91cbfdc5779fc41b1d11d3faf0f`.
- Stage1 `h=.25` strength curve 뒤 Stage2는 고정 `T_z=1.25`에서
  `M5,h=.25` 대 `M10,h=.125`를 비교했다.
- Llama `Z+` d_i mean/median/p90/max:
  `.599766/.599819/.819999/.837496`; `Z±`:
  `.722633/.737896/.865267/1.022678`. 사전 임계
  `.10/.25/.50`를 두 arm 모두 초과했다.
- clamp는 `Z+` M5/M10 `0/50`, `0/100`; `Z±` `6/50`, `0/100`으로
  `<.50`. finite/W pointer·version·bytes/FULL-FP32 PASS;
  optimizer/Adam/backward/parameter-grad 0, autograd.grad 30, duplicate 0.
- verdict: `SCIENTIFIC_HOLD`; reason:
  `Euler discretization refinement gate failure at locked T_z=1.25`.
  이는 수치 refinement 판정이며 편집 성능 결론이 아니다.
- ZA/ZB submission 0; replay/retry/regrid/T_z/tolerance 변경 0;
  scientific promotion false.
- Qwen Stage2 raw root는 보존하되 payload/metric을 report·lock에 사용하지
  않았다. 별도 archival exclusion receipt만 있으며 scientific input 0.
- canonical Korean report:
  `experiment-reports/servers/server4/p4-euler-stage2-llama-only-2026-08-22/report-ko.md`
  (SHA `f2a11bf2f6ba3d26b03fd8557d046b6d73edf05dced75aed17eb8c9feaac38f1`).
- analysis manifest root:
  `fa433d4022517c0935f94ea6a4f0cb1ee0b9d443599e425525ff885079b68670`;
  rooted receipt identity:
  `92d0eb6692fdd2819b39d9134990453e5d72aaafa61c862e8b1f1e12d0ddfb9a`.
- final numerical lock root:
  `2d9034eb4ceb59f12469388679663710c48ed0fab394613f8cd1769eebe9a577`.
- telemetry boundary: endpoint request NLL/margin, exact endpoint tensor distance,
  d_i distribution과 compute ledger는 기록됨. request별 relative d_i,
  origin displacement vector 및 microstep objective/gradient/movement 값은
  runtime이 rooted aggregate identity만 저장해
  `NOT_RECORDED_CALIBRATION_RUNTIME_TELEMETRY_OMISSION`; HOLD 이후 GPU replay 0.
