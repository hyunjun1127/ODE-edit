# [SH4→GH] P1R52 target-timescale B100 terminal reports

- task: `ODEEDIT-S05-P1R52-TARGET-TIMESCALE-ABLATION-B100-V1`
- valid array: `23132_[0-4]`, 5/5 `COMPLETED 0:0`
- canonical same-horizon report: `experiment-reports/servers/server4/p1r52-target-timescale-same-horizon-b100-2026-08-23-native-v3/report-ko.md`
- same-horizon report SHA256: `9cbab02b75b4dabab67d3f71b9609447e250c7e18380c2309ab6d96fbf331502`
- canonical longer-time report: `experiment-reports/servers/server4/p1r52-target-timescale-longer-time-b100-2026-08-23-native-v3/report-ko.md`
- longer-time report SHA256: `9ca1c2797bc3b71c7e1bfb1080f9f2d5555c34a0a375068cf6da2ea66562805b`
- v3 layout: 첫 표에 final W Eff/Gen/Loc + z/W Rewrite/Rephrase NLL
  mean/median/p90, 이후 Rewrite/Rephrase 상세 및 실험별 전용 표 분리
- v2 correction: Native accepted-z와 post-W의 mean/median/p90/max,
  success/accuracy/strict/locality, z→W gap, compute를 본문에 모두 승격
- v1/v2 packages: immutable superseded report lineage로 보존
- Native AlphaEdit: `23140_0 COMPLETED 0:0`
- Native MEMIT first attempt: `23140_1 PURE_TECHNICAL`, scientific denominator excluded
- Native MEMIT TECH-R2: `23142_1 COMPLETED 0:0`
- Native W endpoint measurement: `PASS` (두 method 모두 post-W evaluator 및 W0 restore 확인)
- Native policy: `USER_DIRECTED_NATIVE_REFERENCE_RUN_OVERRIDE`, selection influence 0
- final Tz selection: false
- scientific promotion: false
- state: `IDLE_AWAITING_GH_CALL`
