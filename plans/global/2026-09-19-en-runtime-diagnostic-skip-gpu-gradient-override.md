# EN R512-G256: 진단 검증 생략·GPU gradient 누적 사용자 override

## 권한·범위

2026-09-19 사용자: “en 실험 리뷰인데, 검증 부분 파이프라인에서 삭제시켜”. 후속: “문서별 gradient의 GPU→CPU 이동을 없애는 것 이것도 진행하고 검증에 관련된거 어짜피 필요가 없을 것 같은데 전부 삭제시켜”.

첨부 `101ab394-2e8f-4bfb-87d1-e8500f48ae41/붙여넣은 텍스트.txt`의 비용 분석을 읽고 적용했다. 이 지시는 기존 R512-G256 설계 중 **실행 경로의 별도 진단 검증과 CPU FP64 gradient 누적** 요구보다 우선한다. 기존 frozen source/실험 결과/teacher/체크포인트를 소급 변경하지 않는다.

이번은 코드 변경이며 **새 GPU 실험·teacher 생성·재제출·sequential 승인이나 모니터링 재개가 아니다.** `max_batches=1`, `sequential_authorized=false`를 유지한다. Project cap2/task1 정책도 바꾸지 않는다. 다음 실제 실행은 새 source/archive/lock으로 따로 승인·제출해야 한다.

## 새 MATCHED_B1 실행 경로

- GeneratedTeacherStore를 `verify_payloads=False`로 연다. 전체 teacher 파일 SHA, finite, TF argmax, FP64 logsumexp 정규화 및 문서 전후 재해시는 수행하지 않는다. 최초 setup의 동일 payload 전수 검사도 생략한다.
- Reference key/residual는 setup에서 필요한 것만 읽고, 각 objective에서는 logp만 mmap한다. 이미 CPU에 소유한 key/residual를 teacher 접근마다 다시 열지 않는다. Reference cache sweep 전후 전체 byte hash를 하지 않는다.
- EndpointSession은 `verify_bytes=False`다. 반복 source/CPU-owner/GPU-owner byte 비교를 제거한다. 새 endpoint/clone의 ID 생성용 SHA, 소유권/shape/dtype/epoch/version, 입력·source·실행 범위 결속은 남는다. 이 경량 메타데이터 검사는 NumPy/.data alias 변경 검출을 보장하지 않는다.
- 별도 reference/current physical AD·FD/parity, selected physical parity, matched exactness 인증, 실행 전후 전체 nonselected weight hash를 호출하지 않는다.
- Checkpoint는 create-once/fsync/atomic publication으로 계속 저장한다. 진단용 deserialize·finite/hash 재검사·실물 재로드 테스트는 하지 않는다. 저장물 출처 식별을 위한 생성 SHA와 inventory는 남는다.
- 결과는 `SKIPPED_USER_DIRECTED`, `numerical_validation=NOT_ESTABLISHED`, checkpoint reload `NOT_RUN`으로 기록한다. 보고서의 과거 PASS 고정 문구로 승격하지 않는다. 비교용 loss/gradient/trial/선택 ledger와 공식 평가 자체는 보존한다.

`technical.py`, `parity.py`, 명시적 CPU audit 함수 및 단위 테스트는 역사 재현/별도 분석 용도로 남지만 새 실험 runner는 위 진단을 호출하지 않는다. 기존 teacher 준비 producer와 과거 준비 결과는 변경하거나 재실행하지 않는다. 나중에 새 준비가 필요하면 그 준비 범위는 별도 명시한다.

## Gradient 이동

원 문서 순서와 FP32 문서 gradient를 유지한다. 모델 device에 FP64 accumulator를 생성하고 각 gradient를 FP64로 변환해 차례로 더한 뒤 문서 수로 한 번 나눈다. 최종 평균 tensor만 CPU로 보낸다. R512 문서 선택/가중치·full-vocab KL·실제 Ti 분모·optimizer 방향/예산은 변경하지 않는다.

L4 `[4096,14336]` FP64 tensor는 469,762,048B=448MiB다. 문서별 512회 전송의 산술량 240,518,168,576B(240.518GB/224GiB)를 최종1회 전송으로 바꾼다. 이는 **대형 gradient tensor 전송량 산술**이며 loss scalar, teacher H2D, checkpoint 저장 등의 전송은 별도다. 시간 가속률 실측이 아니며 backward 시간이 전부 사라진다는 주장도 아니다.

GPU에 accumulator 448MiB가 추가로 상주하며 문서 FP64 변환 temporary도 최대448MiB 필요하다. 실제 peak memory·GPU 처리량·과거 CPU 누적과 bitwise 동등성은 미측정이다. 새 실행 lock은 `gradient_accumulation=GPU_FP64_DOCUMENT_ORDER_FINAL_CPU`, `diagnostic_validation=SKIPPED_USER_DIRECTED`를 명시한다. 과거 CPU 누적 lock을 그대로 새 runtime 실행에 사용하지 않는다. Read-only 과거 준비 lineage 결속에만 옛 policy를 허용한다.

## 바꾸지 않는 것

EN의 수학적 후보 수용조건(KL/Armijo, Current/Past 보호, 실제 DK/leakage/반응 invariant), 원 gradient1회·최대8trial, native fit, 데이터/순서/precision, 공식 evaluator는 그대로다. Nonfinite loss/gradient 오류 처리, array shape/dtype, source/input/출력 경로·권한, partial/duplicate 방지, rollback과 history1 및 atomic 저장은 제거하지 않는다. 이들은 별도 진단 실험이 아니라 방법/상태/저장 실행에 필요한 조건이다.

큰 파일의 임의 외부 변경은 실행 중 재검증하지 않으므로 sealed artifact 불변을 가정한다. 검증 생략을 수치 정확성·방법 효능·완전한 무결성 PASS로 해석하지 않는다.
