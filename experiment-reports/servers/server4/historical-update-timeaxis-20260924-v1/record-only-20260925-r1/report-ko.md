# Historical update timeaxis — 수치 비교 기록 전용 전환·재제출 인계

권한/nonce: `ODEEDIT-GH-SH4-HISTORICAL-TIMEAXIS-RECORD-ONLY-20260925-R1`.
현재 상태는 **SUBMITTED_RELEASED / INITIAL_NOT_OBSERVED**다. 실험 완료·수치 인증 보고가 아니다.
SH1 이관은 철회 상태이며 SH4가 제출했다. 제출 후 scheduler/log/result/initial/terminal 조회는 하지 않았다.

## 실제 등록

| 범위 | 실제 job | dependency | 자원 | 확인 경계 |
| --- | --- | --- | --- | --- |
| BASE_ALPHAEDIT | 53283 | scheduler 선행 없음, 내부 양 family matching gate | 1GPU / 8CPU / 60416MiB | held 검사 후 release 명령 성공 |
| BASE_MEMIT | 53284 | scheduler 선행 없음, 내부 양 family matching gate | 1GPU / 8CPU / 60416MiB | held 검사 후 release 명령 성공 |
| T3A/T4 CPU collector | 53285 | afterany:53283:53284 | GPU0 / 8CPU / 24576MiB | held 검사 후 release 명령 성공 |

Release: `2026-09-24T22:10:22.879761Z` (KST 2026-09-25 07:10:22).
Project/task cap2, server4, exportNONE/Requeue0. GPU wall7일/collector4시간은 요청 상한이며 실측 실행시간이 아니다.
Owner/name/full argv/command/node/memory/GPU/dependency를 release 전에 검사했다.
Resource helper는 active project GPU0 + 신규2 ≤ cap2를 확인했다. 본인 server4 admitted queue도 비어 있었다.
최종 admission disk free `73,436,766,208 B`, reserve `21,474,836,480 B`.
현재 RUNNING/PENDING 여부와 GPU 가용성을 release 이후 확인하지 않았으므로 주장하지 않는다.

## source·lock·정본

- 실행 source `b856babbca101c096d72a38a3ec9c936a85a4cf3`, tree `a90cac47b0a80d2d5586243c927018033283ec67`.
- 새 immutable attempt: `/data/janghj/ODE-edit/local/historical-update-timeaxis/20260924-v1/attempt-r3-record-only/`.
- `execution.lock.json` SHA `f99939e6edda921c6320110c44a4a0b03e5c8911e2937c8955ea5e98d9ba7ebc`.
- `source.tar` SHA `78275ea84fbc2756a3f3d9ee47ac17c31159bbb8469f50f5a1cab8d385b17712` / 174,080 B. Archive는 local-only.
- 사용자 override 정본 SHA `9d2cebee7da6d70ceb69f55aca4a15ef81acfbb653b9027af0f76989acfc6240`.
- explicit reuse bridge SHA `f2a535d481d84369287490e65160e8ad1dfe30a4cecb4cf2e1d0a26862ff1d8f`.
- 설계/원 evaluator/모델 revision/fixed10k/패널/FP32·eager·MB16/FP64 whole-five-weight 제거 후 FP32 한 번 materialization은 불변이다.

기계적 제출요약: [submission.json](../../../../../audits/servers/server4/historical-update-timeaxis-20260924-v1/record-only-20260925-r1/submission.json).
사전검산: [preflight-ko.md](../../../../../audits/servers/server4/historical-update-timeaxis-20260924-v1/record-only-20260925-r1/preflight-ko.md).
원 실패 보고: [rerun RCA](../rerun-20260925-r1/report-ko.md).

## 기록 전용 정책과 유지된 차단 검사

`numerical_fidelity_policy=RECORD_ONLY_USER_DIRECTED`, `numerical_certification=NOT_ESTABLISHED`.
기준값 NLL `0.00025`, margin `0.0005`는 그대로다. 초과량을 줄이거나 원 측정값을 보정하지 않는다.
MB16↔MB1, 동일 endpoint 반복, 원 평가↔현재 평가, restore 전후, counterfactual sentinel의 오차·flip은 per-case 기록이며 실행 차단 사유가 아니다.
`COMPLETED_WITH_NUMERICAL_WARNINGS`는 경고가 기록된 상태의 계산 완결성을 뜻하며 수치 PASS가 아니다.
내부 `PASS.json`은 `STRUCTURAL_COMPLETION_NOT_NUMERICAL_CERTIFICATION` 의미 및 waiver/policy hash를 포함한다.
Peer와 collector는 동일 정책/실행 identity를 확인하고 경고 때문에 실패하지 않는다.

Source/input/token/case/order/분모/중복·손상, actual 선택weight·whole-U state, exact restore bytes,
NaN/Inf·I/O·자원·권한은 계속 차단한다. Scalar M/B/C 정의와 회계식도 바꾸지 않았다.
진단 raw/per-case 근거를 수치 판정 전에 atomic 저장한다. 비유한 값은 명시 태그로 보존하되 실패를 숨기지 않는다.

| 적용 위치 | source 함수 | 검사 경계 |
| --- | --- | --- |
| 수치 오차/flip | fidelity.py:compare_raw / historical_row | 기존 기준·실제값·초과·case 위치 기록, raise 없음 |
| identity/finite | fidelity.py:validate_rows / compare | 불일치·중복·비유한·TF row 손상은 실패 |
| raw 선저장 | worker.py:raw, backend.py:evaluate | 평가 결과 callback 저장 후 finite/수치 진단 |
| endpoint/diagonal/sentinel | worker.py:verify_endpoint / t1 / score_task | 모든 동일 의미 근접성 비교 기록 전용 |
| peer/collector | fidelity.py:check_gate, reduce.py:worker_completion | 구조적 identity/PASS 의미 검증, 경고 전달 |
| 변경하지 않은 실제 state | backend.py:state / unchanged | FP64 제거·FP32 materialization·exact restore·RNG 보호 |

## 재사용과 새 실행 범위

T0/24CP/model4/token/설계는 이전 fullSHA와 현재 불변 stat 또는 필요한 작은 입력 SHA로 결속했다.
모델/24CP 전체를 다시 읽어 hash하거나 전송하지 않았다.
기존 endpoint9개(Alpha 0/1/20/50/100, MEMIT 0/1/20/50) raw와 Alpha12 diagonal receipt를 explicit bridge로 재사용한다.
Endpoint raw는 앞 CPU2592행 독립 비교에 결속했고 새 worker에서도 CPU 진단을 기록한다. 이를 새 GPU 검사로 세지 않는다.
Alpha diagonal은 원 receipt의 state hash/restore/집계 근거 수준이며 원 per-case raw는 미저장이다.
과거 `PASS.json`을 새 실행 identity로 복사·위조하지 않는다.

MEMIT W100의 실패 상세 raw는 원래 `NOT_RECORDED`다. 새 source는 필요한 해당 평가와 미완료 MEMIT diagonal을 수행한다.
기존 scientific score task0이므로 실제 T2P/T2F/T3B 및 collector T3A/T4는 새 실행이다.
고정 main156cell + pair16cell, score task383/state173은 job3개와 구분한다.
기존 실패53182/53183/53184는 재취소하지 않았고 raw/source/cost를 보존했다. 타 job 변경0.

## 검증·비용·종료

CPU32 tests PASS: 기존 0.00052428 초과/큰 finite 초과/flip 경고 전달,
identity·NaN/Inf·restore byte 손상 차단, peer/collector warning join, 원 state routing·reducer·5그림 fixture 검증.
별도 독립 red agent는 사용하지 않았으며 owner 감사와 CPU 회귀다. 실제 GPU 수치 인증을 주장하지 않는다.
기존 parent 할당은1569 GPU초. 새 실행 allocation/programwall/peak/결과는 **NOT_OBSERVED**로 별도다.
모델/평가/new native fitting/history 변경0, `save_checkpoints=false`; score/receipt만 저장한다.

`monitoring_active=false`, `automatic_resume=false`, 추가 agent 제출0.
이 인계 후 등록된 runner/collector만 자연 진행한다. 상세 결과 회수는 사용자 recall 시 수행한다.
NO_BROADCAST_NOT_REQUIRED: 같은 host 입력 재사용, 원 raw/teacher/tensor/archive/fullstdout Git0, 대형 원격 전송0.
