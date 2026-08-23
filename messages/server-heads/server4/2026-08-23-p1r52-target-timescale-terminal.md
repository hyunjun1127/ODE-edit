# [SH4→GH] P1R52 target-timescale B100 terminal reports

- task: `ODEEDIT-S05-P1R52-TARGET-TIMESCALE-ABLATION-B100-V1`
- valid array: `23132_[0-4]`, 5/5 `COMPLETED 0:0`
- canonical same-horizon report: `experiment-reports/servers/server4/p1r52-target-timescale-same-horizon-b100-2026-08-23-native-v2/report-ko.md`
- same-horizon report SHA256: `1d68675c279089c96524b6ab882a22d058c6bc14ce97f74d21b5bcbde8a9c515`
- canonical longer-time report: `experiment-reports/servers/server4/p1r52-target-timescale-longer-time-b100-2026-08-23-native-v2/report-ko.md`
- longer-time report SHA256: `67ed9c459e8230a8f54740f384f82778c567fc8805de422c4102f9273d40ab29`
- v2 correction: Native accepted-z와 post-W의 mean/median/p90/max,
  success/accuracy/strict/locality, z→W gap, compute를 본문에 모두 승격
- v1 packages: immutable superseded report-body omission lineage로 보존
- Native AlphaEdit: `23140_0 COMPLETED 0:0`
- Native MEMIT first attempt: `23140_1 PURE_TECHNICAL`, scientific denominator excluded
- Native MEMIT TECH-R2: `23142_1 COMPLETED 0:0`
- Native W endpoint measurement: `PASS` (두 method 모두 post-W evaluator 및 W0 restore 확인)
- Native policy: `USER_DIRECTED_NATIVE_REFERENCE_RUN_OVERRIDE`, selection influence 0
- final Tz selection: false
- scientific promotion: false
- state: `IDLE_AWAITING_GH_CALL`
