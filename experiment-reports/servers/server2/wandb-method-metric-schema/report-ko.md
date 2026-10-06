# SH2 scientific W&B metric 계약 수락

Nonce: `USER-GH-ALL-SH-WANDB-METHOD-METRICS-20261007-SERVER2`

상태는 **POLICY_ACCEPTED_FUTURE_CALLER_ADOPTION_REQUIRED**다. 정본 main `c5dc1cb4fc6853b6dfc3ceac5a490c14ab131ba4`의 envelope, contract, policy, 사용자 첨부를 전체 읽고 SHA를 결속했다. 정확 hash는 acceptance.json에 기록한다. 실제 hostname/server2, 공식 SH2 session `01a0493a-074c-7f91-9a13-769116326fef`, 원 CWD `/mnt/raid5/janghj/ODE-edit`, origin `hyunjun1127/ODE-edit`를 확인했다. 기존 root dirty를 보존하고 전용 non-main worktree에서 기록했다.

## 적용 범위와 완료 경계

향후 ours/baseline scientific caller에 적용한다. 이번 정책 task에서 적용할 active unfrozen caller는 없고, 기존 FE source/job은 봉인 상태를 유지한다. SH1만 공통 helper 구현을 소유한다. SH2는 helper나 frozen caller를 변경하지 않았다. 다음 새 source 봉인 시 SH1의 exact helper commit과 자체 producer를 함께 결속한다.

정책 수락은 helper 구현 완료, SH2 caller 통합 완료 또는 실제 online 검증 완료가 아니다. Helper commit adoption은 아직 없으며 actual online validation=NOT_PERFORMED다. 단순 allowlist 확장을 축·identity까지 완성한 것으로 보고하지 않는다.

## 다음 source의 필수 결속

- `current/pre`, `current/post`, `all_seen/post`를 분리한다. BS100의 W5에서도 current는 해당 100요청이며 누적 모집단은 실제 측정된 all-seen에만 기록한다. 다른 horizon에 새 평가를 강제하지 않는다.
- 각 R/P/N에 count, success_count, success_pct, token_acc_pct, prompt_acc_pct, strict_acc_pct, true_nll, new_nll, margin_true_minus_new를 분리한다. 정수 분모와 token 분모를 혼동하지 않는다. Percent는 0–100, NLL/margin은 nats, margin=true−new. Desired target은 R/P=new, N=true다.
- Success harmonic은 pct 입력에서 `3/(1/pR+1/pP+1/pN)`, fraction 입력에서 `300/(1/rR+1/rP+1/rN)`이다. 측정 성분 중 0이 있으면 0, 하나라도 미측정이면 생략한다. Strict ACC와 별개다.
- Performance payload마다 실제 edits를 포함하고 prefix 축은 `step_metric='edits', step_sync=False`로 정의한다. pre_state_edits/post_state_edits를 별도 기록하며 fit/global_candidate는 독립 단조 축으로 사용한다.
- `W0_first2000`은 cold exact first2k에만 사용한다. FE10k 등 다른 horizon은 명시적으로 검증된 W0_first<N> schema 등록 없이 first2000으로 표기하지 않는다. 누락값 보간·가짜 0·새 forward는 금지한다.
- Config model/model_family/writer/role/metric_schema와 job/source/config identity를 함께 유지한다. `price-first2k-scalar-v1` capability 및 실제 producer compatibility, 축, immutable identity까지 확인한 SH1 helper를 채택한다.
- Create-once run ID/URL/job/source/config/model/schema receipt와 가변 transport status를 분리한다. SDK_ASYNC_NOT_REMOTE_ACK는 원격 수신 PASS가 아니다. 다음 승인된 run의 기존 bounded startup/finish readback만 사용하며 uploader 종료를 과학 완료로 보지 않는다.

## 보존·검사·미실행

GPT2 초기 자산준비, setup smoke, storage audit는 scientific metric 요건 밖이며 기존 job identity/privacy 정책은 유지된다. Entity `wkdguswns2256`, project `layer allocation`과 noCP 정책도 유지한다.

첨부의 historical W&B 관측은 사용자 제공 증거로 읽었으며 이번 SH2가 원격 재검증한 사실이 아니다. B4/B5 raw mapping·fake SDK 검사를 중복 실행하지 않았고 구현 PASS를 주장하지 않는다. 이번 검사는 owner의 문서 전체읽기/hash 및 JSON/scope/diff 확인뿐이다. 독립 reviewer는 사용하지 않았다.

GPU/model/Slurm/API/online smoke/추가 평가/원 raw 읽기/모니터링은 0이다. Existing frozen/pending/running jobs 및 과거 run의 hotpatch/cancel/restart/rename/backfill/comparison snapshot upload도 0이다. 현재 scheduler 상태는 조회하지 않았다. 원자료·다른 task·과학 조건·cap은 그대로 보존한다.

NO_BROADCAST_NOT_REQUIRED. Own compact 기록 게시 후 STOP. 다음 실제 new-source caller adoption과 online 검증은 별도 남은 의무다.
