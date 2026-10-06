# W&B job 번호 기록 — SH4

사용자가 첨부 비교 요건을 전달한 뒤 GPT-J 여섯 arm 제출을 **다시 승인**했다. 실제 제출 job ID는 아직 없다. 기존 job과 과거 W&B run은 변경하지 않았다.

- 정책/envelope 수신 및 SHA 확인 완료. 공통 helper는 SH1 소유이며 SH4 수정 없음. `bc63425e` job 번호/name/config와 후속 step-id 수리를 채택했다. 추가 비교 metric/config whitelist·축 계약의 SH1 확장이 남았다.
- 기존 GPT-J 과학 caller는 실제 `SLURM_JOB_ID`를 전달한다. 신규 비교 caller는 정확한 submission과 원 과학 run의 job identity를 결속해 `job<display_id>`를 이름에 넣고 Config에도 보존하도록 준비했다. CPU collector의 다른 job 번호로 원 과학 run을 잘못 표시하지 않는다.
- 로컬 fake-SDK 테스트 3개 통과: 일반 job, array index 0/step, collector와 science ID 구분, 누락/충돌 차단, bounded readback의 결측/검증 구분. 네트워크·GPU·모델 실행이나 가짜 job 업로드는 없었다. 별도 독립 reviewer 없음.
- 실제 online name/config readback은 **NOT_OBSERVED**. 비교 schema helper 채택·최종 source/config 재봉인이 남아 있다. 사용자 추가 metric 지시는 수신·caller 반영했고 기존 B4/B5/W0 raw로 검산했다. 구현 초안을 실행 완료로 취급하지 않는다.

GPT-J 최초 source freeze 시 이전 evaluator SHA 결속이 검출되어 Slurm 등록 전에 차단됐다. 원 attempt는 보존했다. 수정된 후보 config도 이후 tracking 변경 때문에 재봉인이 필요하며 현재 제출 가능 상태로 표시하지 않는다.

`NO_BROADCAST_NOT_REQUIRED`: 소형 source/receipt만 대상이며 원 raw·credential·SDK spool은 전송하지 않는다. 신규 모니터링/자동 재개 없음.
