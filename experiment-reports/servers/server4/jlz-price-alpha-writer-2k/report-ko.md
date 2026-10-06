# PRICE AlphaEdit writer 6-cell 2k — 등록 준비

- task: `jlz-price-alpha-writer-2k`.
- nonce: `USER-GH-SH4-JLZ-PRICE-ALPHA-WRITER-2K-20261007-R1`.
- 상태: `IMPLEMENTED_STATIC_CHECKED_NOT_SUBMITTED`. 실제 job IDs는 아직 없다. CPU source/import/config 검사와 actual GPU/완료는 구분한다.
- Llama/Qwen × AE_CAP075/AE_CAP100/AE_FREE100, 각각 독립 cold W0/H0, fixed first2000 BS100×20/seed20261002. 총12,000 applications, cohort2,000. B21/새 baseline/pilot/추가 fit/noCP 모두 없음.
- 기존 MEMIT 실행 source87a5a736 기반, 원 frozen source/job/raw는 보존했다. 새 source는 전용 `project/run_scripts/jlz_price_alpha_writer/`에만 추가했다.
- 동일 native planner에서 Alpha lambda1/.02 projector/LU/thin operator를 BUILD/가격/pullback/commit에 일관되게 사용한다. C0/H 비용은 별도 진단이며 C0 scale1이다.
- 계획 queue: 양 model 첫 arm은 afterany59933:59936, 각 model CAP100→FREE100 직렬 afterany, CPU collector는 own6 afterany. 기존 MEMITcollector59937 의존/성능 gate 없음. 총 server4 user cap2.
- GPU job당 1GPU/8CPU/59392MiB/exportNONE/Requeue0, 요청48h 상한은 ETA가 아니다. collector0GPU/8CPU/24576MiB/4h.
- W0는 모델/runtime/token/row/cold identity가 같은 기존 원자료만 재사용한다. W5/10/15/20 및 current/cohort 관측을 보존한다. W20 분모 R2000/P4000/N20000. 아직 새 metric은 `NOT_OBSERVED`다.
- W&B `wkdguswns2256` / `layer allocation`은 각 새 job startup online 확인 후 모델 로딩한다. 과거 smoke를 새 job 온라인 PASS로 표시하지 않는다.
- 실행 준비: `/data/janghj/ODE-edit/local/jlz-price-alpha-writer-2k/`. 실제 실행 source/config/lock/job은 등록 receipt에서 별도로 결속한다.
- 검토: [owner audit](../../../../audits/servers/server4/jlz-price-alpha-writer-2k/preflight-ko.md). [정본](../../../../project/proposals/jlz-alpha-writer-review/authorized-run-contract.json).
- Source/compact receipt만 Git, 원 raw KEEP. `NO_BROADCAST_NOT_REQUIRED`, 새 recurring monitoring/heartbeat/autoretry0.
