# 기본 실험 checkpoint 미저장 정책

Instruction ID: `ODEEDIT-ALL-SH-DEFAULT-NO-CHECKPOINT-20260919-V1`

사용자: “앞으로 실험은 별도의 명시가 없으면 checkpoint 저장은 하지 않는 것으로 각 SH들에게 전달해”.

## 적용

앞으로 준비·제출하는 실험은 `save_checkpoints=false`를 기본값으로 한다. 자동 periodic/best/last/final model checkpoint, selected endpoint 전체 weight, W/M/RNG/optimizer resume bundle, 복원용 전체 weight delta 등 **checkpoint와 동등한 영속 저장**을 기본으로 만들지 않는다. 파일 이름만 endpoint/capsule/snapshot으로 바꾸어 우회하지 않는다.

저장 예외는 해당 task의 사용자 지시 또는 승인된 구체적 실험 사양에 checkpoint 저장이 **명시**된 경우뿐이다. 예외는 어떤 state를 어느 시점에 왜 저장하는지와 권한 출처를 execution lock/report에 기록한다. 일반 템플릿의 관행, 디버깅 편의, “혹시 재개”는 예외가 아니다. 이후 GH envelope 작성 시에도 미저장을 기본으로 하며 불필요한 저장 의무를 관성적으로 추가하지 않는다.

기존에 승인된 task의 구체적인 checkpoint 필수 요건은 자동 취소하지 않는다. 현재 task에 그러한 예외가 있으면 ACK에 해당 항목·출처를 간단히 명시한다. 명시적 저장 요구가 없는 아직 미제출 구현/launcher는 새 기본값을 적용한다.

## 보존과 실행 경계

- 기존 checkpoint/weight/raw는 삭제·이동·덮어쓰지 않는다. 이 정책은 정리/영구삭제 권한이 아니다.
- 이미 제출·실행 중인 job/source/config는 이 통지만으로 cancel/requeue/재제출/hot-patch하지 않는다. 새 제출부터 적용하며 기존 lock은 역사 그대로 둔다.
- Pretrained model 원본/cache, 이미 존재하는 checkpoint의 read-only 재사용은 삭제/금지 대상이 아니다.
- 현재 모델과 sequential history의 RAM/GPU state, transaction/rollback용 일시 메모리는 유지할 수 있다. 단계를 같은 process에서 이어가는 것은 가능하다.
- 필요한 metric, per-case NLL/평가, 로그, source/config/sample identity, timer, selection/history ledger, 작은 상태 hash와 provenance는 계속 기록한다. 설계가 요구하는 native target/key 또는 소규모 factor는 checkpoint와 구분하되, 전체 편집 weight나 완전 재개 state를 몰래 포함하지 않는다.
- Checkpoint가 없으면 exact crash-resume/GPU continuation이 보장된다고 보고하지 않는다. `checkpoint_saved=false`, `exact_resume=NOT_AVAILABLE` 등 실제 경계를 명시한다.
- 새 단계의 필수 disk-resume와 미저장 조건이 충돌하고 명시 예외도 없으면 임의 저장/재실행 대신 정확한 의존성을 보고한다. 실험 방법·평가·GPU cap·monitoring·기존 pause 정책은 이 통지로 변경되지 않는다.

## 전달 및 종료

대상은 현재 등록된 SH1/SH2/SH4. SH3는 미등록 future target이므로 live 전달 성공으로 쓰지 않으며 추후 onboarding에도 이 기본값을 적용한다.
정책 FULL_READ/적용 ACK만 반환하고, 별도 paused task를 재개하거나 scheduler/결과를 조회하지 않는다. 현재 수행 중인 task가 있으면 그 준비·저장 설정에 직접 관련된 정책 steer로만 처리한다.
