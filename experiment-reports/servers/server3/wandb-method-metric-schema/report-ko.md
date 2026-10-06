# SH3 scientific W&B 지표 계약 수락

상태: **POLICY_ACCEPTED_FUTURE_CALLER_ADOPTION_REQUIRED**. 현재 SH3에서 새로 봉인 중인 과학 caller는 없다. 공통 helper 구현은 SH1 소유이며, 다음 승인된 ours/baseline source 봉인 시 helper commit과 caller·config·schema를 함께 결속한다. 이번 작업은 정책 수락과 문서 검산이며 구현 완료 또는 실제 온라인 검증이 아니다.

Nonce: `USER-GH-ALL-SH-WANDB-METHOD-METRICS-20261007-SERVER3`. 정본 main `c5dc1cb4fc6853b6dfc3ceac5a490c14ab131ba4`의 envelope, contract, policy, 사용자 첨부를 FULL_READ하고 size/SHA를 [수락 감사](../../../../audits/servers/server3/wandb-method-metric-schema/acceptance.json)에 기록했다. envelope와 첨부 SHA는 전달값과 일치한다. 실제 host ubuntu, session 01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3, origin hyunjun1127/ODE-edit이며 별도 non-main worktree의 boundary 검사가 PASS했다. 중복 ACK/등록은 발견하지 않았다. 기존 root dirty는 보존했다.

## 다음 신규 caller 적용 계약

| 항목 | 적용 내용 |
|---|---|
| 모집단 | current/pre, current/post, all_seen/post 분리. W5도 current는 원 batch100, 측정된 all_seen만 누적500. 가변 batch는 실제 분모 사용 |
| R/P/N | 각각 count, success_count, success_pct, token_acc_pct, prompt_acc_pct, strict_acc_pct, true_nll, new_nll, margin_true_minus_new |
| 단위·정답 | pct 0..100, NLL/margin nats. R/P desired=new, N desired=true. token micro 분모는 token, preference와 TF 정확도는 별개 |
| 조화평균 | pct이면 3/(1/pR+1/pP+1/pN), fraction이면 300/(1/rR+1/rP+1/rN). 측정값 하나라도0이면0, 결측이면 키 생략 |
| 평가 축 | 같은 payload에 edits 포함, prefix wildcard step_metric=edits와 step_sync=False. pre_state_edits/post_state_edits 별도 |
| fit 축 | 단조 fit/global_candidate 별도 축, batch별 candidate reset 중첩 금지 |
| W0 | W0_first2000은 exact cold first2000, x=0. 다른 horizon은 별도 검증된 schema 등록. w0/current/N과 w0/all_seen/N도 정확 cohort만 |
| config | model/model_family/writer/role/metric_schema 및 실제 job/source/config/attempt/array/step identity 유지. writer를 임의 memit으로 표기하지 않음 |
| job ID | 실제 raw Slurm ID와 array 표시 ID 분리, array index0 유지. run.name에 job 표시, attempt UUID 유지. local에는 가짜 ID 없음 |
| 전송 | immutable run ID/URL/job/source/config/model/schema receipt와 mutable transport 상태 분리. SDK_ASYNC_NOT_REMOTE_ACK는 remote PASS가 아님 |

측정하지 않은 값을0으로 채우거나 추가 forward로 채우지 않는다. 평가 schedule/horizon은 각 승인된 실험 그대로다. 다음 실제 승인 run의 기존 bounded startup/finish readback에서 identity와 실측 key/edits를 확인하며, 이를 위한 새 smoke·GPU·Slurm·주기 조회는 만들지 않는다. Uploader finish와 과학 완료도 분리한다.

## 구현 경계와 검산 범위

정본 시점 shared schema/client/worker의 정적 symbol 확인에서 job identity 처리 함수는 존재하지만 COMPARISON_SCHEMA는 없었다. 이는 해당 commit의 제한된 소스 관찰이며 현재 또는 향후 SH1 배포 완료를 판정한 것이 아니다. 다음 신규 봉인에서는 `price-first2k-scalar-v1` capability, strict config/metric whitelist, define_metric axes, immutable receipt 및 caller 호환성을 함께 확인한다. whitelist 추가만으로 전체 구현 완료를 보고하지 않는다.

Owner 문서·SHA·경로·JSON 검산만 수행했다. 별도 reviewer, 생산 caller 회귀, SH4 B4/B5 raw 재검산, 모델/GPU 검증, 온라인 readback은 수행하지 않았다. 첨부의 과거 관측은 사용자 제공 증거로만 취급하며 SH3 독립 재확인으로 쓰지 않는다.

자산 준비·SDK setup·저장공간 감사는 scientific schema 대상에서 제외하되 job ID/privacy 정책은 유지한다. entity `wkdguswns2256`, project `layer allocation`을 유지한다. 기존 frozen/pending/running source와 job, 과거 W&B run·snapshot·view는 변경하지 않았다. 공통 helper/과학 source 수정, 새 submit, 인증 조회, raw 업로드, monitoring은0이다. 다음 source 적용 의무를 기록하고 이번 문서 작업을 종료한다.
