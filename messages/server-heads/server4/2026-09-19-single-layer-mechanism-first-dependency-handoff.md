# SH4 → GH: dependency 등록 완료 / 모니터링 중지

- 사용자 최신 실행기 완성·pending 등록 및 checkpoint 미저장 지시 적용.
- **50983 / odeedit_slmf_S10_s4**, `afterok:50974`, held owner/name/source/fullargv/resource/dependency 검사 PASS 후 release.
- 등록 직후 2026-09-19 22:14:35 KST: `PENDING`, `Reason=None`, dependency `afterok:50974(unfulfilled)`. 이는 등록 관측이며 GPU 자원 부족이나 actual T0/B1 PASS를 의미하지 않는다. 이후 scheduler/log/result 조회 없음.
- execution `0aea3452d04e71dd56e428208e04b9c04c52f82f`, tree `70f7417b48aa68fbc57a723c505e453f3bbc40da`; 제출 control `eff510c77f126a93fdd6ac939dd22e12a51b45b7`. 후자의 변경은 Slurm 실제 Command+SubmitLine 인자 결속뿐이며 frozen runtime은 불변이다.
- archive SHA `fc51c3894d672cf0118f7101d250a02df3671e602860056a6a7db21d4d4921e5`, lock SHA `9436037e1c632b5708b29072d37936433712a556dd2dcc4a5a873f375fcc1846`.
- local output: `/data/janghj/ODE-edit/local/single-layer-mechanism-first/20260919-v1/PROGRAM/dependency-attempt-v1/output`; 같은 attempt의 admission/submission/held-inspection/release JSON 보존.
- 최대 B10: T0→B1→원 CUM gate→S3→원 CUM gate→S10. 동일 process RAM continuation. 기술/과학/저장 실패 시 해당 단계에서 멈추고 기록하며 사용자 허용치·방법 변경 없음.
- `save_checkpoints=false`, 저장 예외 없음, `exact_resume=NOT_AVAILABLE`. 기존50974/source/CP와 다른 job 변경·삭제0. 현재 source의 native full solve update와 후보 dense basis는 RAM-only; 작은 target/key·factor/평가/ledger는 보존.
- CPU186 PASS, 이후 제출형식+RAM integration26 PASS. 별도 read-only red에서 발견한4개 BLOCK의 좁은 회귀 PASS. Actual 새모델/full512/성능/메모리 peak는 NOT_OBSERVED.
- GPU1/CPU8/60416MiB/7일/exportNONE/Requeue0; project cap2. 제출 전 다른 admitted queue0,50974와 dependency 비중첩. free57,519,808,512B이며 전체 S10 저장용량 보장은 아님. stage별24/32/112GiB 추정 검사, 원 storage waiver 상속0.
- [준비 및 한계](../../../experiment-reports/servers/server4/single-layer-mechanism-first-20260919-v1/dependency-program-preparation-ko.md), [등록 receipt](../../../audits/servers/server4/single-layer-mechanism-first-20260919-v1/dependency-registration.json).
- generic access helper는 명시 허용된 runs 경로를 지원하지 않아 해당 검사 NOT_PASS. 공유 helper/정책 변경 없이 새 compact receipt를 승인된 server4 audits 경로에 게시한다. JSON/상대링크 검사와 whitespace 검사 수행; 별도 Markdown renderer는 실행하지 않았다.

`MONITORING_PAUSED_AWAITING_USER`, `monitoring_active=false`, `automatic_resume=false`. 등록된 프로그램만 자연 진행하며 별도 worker/daemon/callback/agent polling 없음. 상세 검산은 사용자 recall 뒤 수행한다. `NO_BROADCAST_NOT_REQUIRED`.
