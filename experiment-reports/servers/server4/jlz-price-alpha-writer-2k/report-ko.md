# PRICE AlphaEdit writer 6-cell 2k — 제출 및 pending 인계

- task: `jlz-price-alpha-writer-2k`.
- nonce: `USER-GH-SH4-JLZ-PRICE-ALPHA-WRITER-2K-20261007-R1`.
- 상태: `SUBMITTED_RELEASED_PENDING`. 실제 GPU 6개/CPU collector 1개를 held 검사 후 release했다. 등록 직후 단1회 snapshot은 전부 PENDING이며 Reason은 아직 `None`이었다. held receipt의 afterany dependency는 검증했다. CPU source/import/config 검사와 actual GPU/완료는 구분한다.
- 실제 accepted turn: `01a111e6-909a-7083-8059-23edee285f9d` (GH direct-acceptance receipt 결속).
- 실행 source `019922b1524efac596bdb8bf7581d44f1721ed59`, tree `11cf47bb45e21c0ac6a348a13b18e09218cc8de7`.
- Config SHA `0296b9b380d247c431c51d1269ade800f99eb7c386b773f43c7335dba0fd63df`, source archive SHA `6aca33f9c8537187484a66f0b450ccc20a0c665d43cc7712bce0b17db9de4a2c`. 이후 보고 commit은 실행 source와 구분한다.
- Llama/Qwen × AE_CAP075/AE_CAP100/AE_FREE100, 각각 독립 cold W0/H0, fixed first2000 BS100×20/seed20261002. 총12,000 applications, cohort2,000. B21/새 baseline/pilot/추가 fit/noCP 모두 없음.
- 기존 MEMIT 실행 source87a5a736 기반, 원 frozen source/job/raw는 보존했다. 새 source는 전용 `project/run_scripts/jlz_price_alpha_writer/`에만 추가했다.
- 동일 native planner에서 Alpha lambda1/.02 projector/LU/thin operator를 BUILD/가격/pullback/commit에 일관되게 사용한다. C0/H 비용은 별도 진단이며 C0 scale1이다.
- 실제 queue는 아래와 같다. 기존 MEMITcollector59937 의존/성능 gate 없음. 총 server4 user cap2.

| Model/arm | Job | afterany |
|---|---:|---|
| LLAMA_AE_CAP075 | 59949 | 59933:59936 |
| LLAMA_AE_CAP100 | 59950 | 59949 |
| LLAMA_AE_FREE100 | 59951 | 59950 |
| QWEN_AE_CAP075 | 59952 | 59933:59936 |
| QWEN_AE_CAP100 | 59953 | 59952 |
| QWEN_AE_FREE100 | 59954 | 59953 |
| CPU collector | 59955 | 59949:59950:59951:59952:59953:59954 |

- [실제 제출 원 receipt](../../../../runs/jlz-price-alpha-writer-2k/submission.json)에 fullargv/source/lock/held 검사 SHA와 1회 snapshot을 보존했다. test-only resource 검사의 출력은 실제 job ID로 집계하지 않았다.
- GPU job당 1GPU/8CPU/59392MiB/exportNONE/Requeue0, 요청48h 상한은 ETA가 아니다. collector0GPU/8CPU/24576MiB/4h.
- W0는 모델/runtime/token/row/cold identity가 같은 기존 원자료만 재사용한다. W5/10/15/20 및 current/cohort 관측을 보존한다. W20 분모 R2000/P4000/N20000. 아직 새 metric은 `NOT_OBSERVED`다.
- W&B `wkdguswns2256` / `layer allocation`은 각 새 job startup online 확인 후 모델 로딩한다. 과거 smoke를 새 job 온라인 PASS로 표시하지 않는다.
- 실행 위치: `/data/janghj/ODE-edit/local/jlz-price-alpha-writer-2k/attempt/`. 모델별 memory 계획/serializer reserve/원 asset SHA와 source/config/lock/job을 봉인했다. quota command는 설치되어 있지 않아 `NOT_AVAILABLE`이고, free bytes/inodes는 admission 및 매 batch guard를 사용한다.
- 검토: [owner audit](../../../../audits/servers/server4/jlz-price-alpha-writer-2k/preflight-ko.md). [정본](../../../../project/proposals/jlz-alpha-writer-review/authorized-run-contract.json).
- Source/compact receipt만 Git, 원 raw KEEP. `NO_BROADCAST_NOT_REQUIRED`, 새 recurring monitoring/heartbeat/autoretry0.
- 일반 Git access helper는 새 `runs/jlz-price-alpha-writer-2k/` prefix를 지원하지 않아 `NOT_PASS`였다. 사용자 envelope의 exact allowed path에 한정한 [게시 예외](../../../../audits/servers/server4/jlz-price-alpha-writer-2k/publication-exception.json)를 기록했고 shared helper는 변경하지 않았다.
