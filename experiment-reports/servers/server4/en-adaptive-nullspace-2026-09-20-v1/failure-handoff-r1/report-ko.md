# EN adaptive-nullspace 51260 실패 RCA · SH3 인계

상태: **TECHNICAL_FAILURE_SERIALIZATION / SH3_REPAIR_OWNER / SH4_NO_RETRY**.
권한: `ODEEDIT-S06-EN-ADAPT-B300-FAILURE-HANDOFF-SH4-V1`, nonce
`ODEEDIT-GH-SH4-EN-ADAPT-FAILURE-TO-S3-20260920-R1`.
후속 정본 main `29123a6dbf215450e0d050f1eefcdfdfe2a6b234` 전체 읽기 완료.
이번 SH4 작업은 CPU/read-only RCA, 새 audit/report/인계뿐이다. 원 실행 수정·GPU·모델·평가·재제출·삭제 0.

## 1. 확정 원인

첫 예외는 Python **NumPy boolean JSON 직렬화 오류**다.
Frozen `controller.py`의 `arithmetic_slack=64*np.finfo(np.float64).eps*scale64`가
NumPy scalar를 만들고, 이를 사용하는 `geometry_checks.ideal_norm` 등의 비교가
`numpy.bool`을 반환한다. `runner.py:72`가 첫 EN_EXACT controller 결과를 저장하면서
`runner.py:21`의 `json.dump(..., allow_nan=False)`가 이를 처리하지 못했다.

`TypeError: Object of type bool is not JSON serializable`의 bool은 Python builtin bool이 아니다.
실제 NumPy 2.2.6과 frozen controller를 사용하는 **1×2 CPU fixture**에서 같은 예외를 재현했다.
실제 partial 파일도 `geometry_checks.ideal_norm` 값 직전에서 끝난다.
수학 판정에 쓰이는 truth value와 JSON schema의 scalar 타입 사이의 누락이다.

stderr의 최초 traceback과 `technical-failure.json` traceback이 일치한다.
앞선 transformers cache deprecation warning은 종료 원인이 아니다.
OOM/timeout/ENOSPC/비유한 수치/정상 quality fallback으로 판정할 근거는 없다.
후속 cleanup 예외는 기록되지 않았다. 실패 경로에 finally entry 복원이 없으므로
**process cleanup/entry rollback은 NOT_VERIFIED**, restore 성공이라고 주장하지 않는다.
과학 commit 전 failure receipt 자체는 정상 JSON으로 저장됐다.

## 2. 정확한 실행·비용

| 항목 | 확인값 |
| --- | --- |
| Job/owner/node | 51260 / odeedit_en_adapt_B300_s4 / janghj / server4 |
| Parent terminal | FAILED, ExitCode 1:0 |
| 시작/종료 | 2026-09-20T18:36:31 / 20:20:09, scheduler 표기 시각 |
| Parent allocation | 6,218초 × GPU 1 = **6,218 GPU-sec**, 1.727222 GPUh |
| CPU/host allocation | 8 CPU / 59 GiB |
| batch MaxRSS | 55,899,724 KiB ≈ 53.31 GiB |
| step accounting | batch FAILED / extern COMPLETED; parent에 중복 가산하지 않음 |
| 실행 commit | b6e86234640ca127546aee094f2a67bbe684a490 |
| 실행 tree | 5eacfc956214ffbd0cfccbeb56c068a888b88990 |
| archive SHA | eee7e31a072fcf54d4390f5d6679ef97b75f848208b3e9996b452f1ec81e9867 |
| lock SHA | cfc4e2150cf57f71d0fea838d2302d84626402b8ad9e8941690da3220bd671ca |
| 분석 branch 기준 | 1bb93e1d45d9144faf8af0aa9574a6d2c0dab438; 실행되지 않은 후속 분석 코드 |

지정 job accounting은 한 번 읽었다. 제출·held inspection·release receipt의 owner/name/source/fullargv가 일치한다.
51260은 terminal이므로 이 실행은 live 중복 실행이 아니다. 무관 job은 조회·변경하지 않았다.
allocation은 utilization이나 정확 program wall이 아니다. complete/cost terminal 미생성으로
일부 전체 timer는 NOT_RECORDED다. 메모리 값은 이 완료 경계까지의 실측이며 S3 후반 peak 보장이 아니다.

## 3. 마지막 완료 경계

| 단계 | 저장 근거와 상태 | 비용/관측 |
| --- | --- | --- |
| Bounded T0 | 완료; finite identity PASS, precision NOT_ESTABLISHED | 553.238393초 |
| fresh B1 shared native | 100 target / 1 solve / inner history 0 | outer 194.839890초 |
| current capture | 14336×5346 keys, 5042 logical prefixes, 304 byte variants | 12.808085초 |
| weighted TSQR/SVD | rank 4596, RESOLVED | 461.726239초 |
| R512 native gradient | 512문서 / 130235위치, J=0.0011208427271629806 | 1709.714453초 |
| spectrum | EXACT/NUM/ADAPT proposal 저장 | 22.730546초 |
| EN_EXACT controller | 함수 반환 후 첫 JSON 저장 실패; 파일 불완전 | 완전 status/selected_trial/evaluations 없음 |
| EN_NUM/EN_ADAPT controller | NOT_RUN, 실패 이후 루프 | NA |
| selection seal / 공식 평가 | 0 / 0 | R/P/N/TF 성능 NA |
| scientific history append / commit | 0 / 0 | N4조차 pending RAM 상태뿐 |
| own at-write history teacher | 0 | 새 영속 teacher 생성 없음 |
| B2/B3 / B300 complete | NOT_RUN / 없음 | NA |

타이머는 nested이므로 위 값을 합쳐 editing time을 만들지 않는다.
native receipt 내부 9.881908초는 z preload를 제외하므로 outer와 서로 대체하지 않는다.
전체 실패 비용은 이미 준비/T0/B1 비용을 포함하는 parent 6218 GPU초로 한 번 계상한다.

Partial JSON의 **이미 닫힌 leading value만** 별도로 읽었다:
첫 후보 objective J=0.003051907800332798, 그 objective timer=174.664335초;
함수 반환 objective J=0.0010912434272175137, 해당 timer=2637.622034초.
하지만 accepted/selected_trial/완전 ledger는 뒤쪽에 있어 저장되지 않았다.
이 값은 완료 성능·commit·정상 endpoint 증거가 아니며 partial 파일을 유효 JSON으로 고치지 않았다.
첫 후보 값이 나쁜 것은 기술 실패가 아니다. 종료 원인은 그 뒤 직렬화다.

T0의 cached/physical gradient는 해당4reference에서 bitwise equal, FD relative error 0.0062281390이었다.
Native/cache z 비교에는 기존 precision 진단 미충족이 있어 precision NOT_ESTABLISHED를 유지한다.
이전 FD waiver/T skip을 이번 실행에 적용하지 않았으며 S3 실제 PASS로 이월할 수 없다.

## 4. 최소 수리 및 회귀 제안 — 구현 소유자는 SH3

1. 비교 flag를 builtin `bool`로 만들거나 JSON 경계에서 `np.generic.item()`만 허용한다.
   `allow_nan=False` 유지; ndarray/tensor/알 수 없는 객체는 계속 거부한다.
   `default=str`, NaN 허용, 전체 tensor 자동 tolist, method threshold 수정은 하지 않는다.
2. 완전 compact receipt를 먼저 serialize/검사하고, 새 파일에 원자적 create-once publication한다.
   기존 partial 파일은 그대로 보존하고 새 immutable attempt를 쓴다.
3. controller 산술 테스트만이 아니라 **실제 반환 ledger 전체 JSON 저장**을 검사한다.
   accepted/fallback/no-step/exact-alias, np.bool/np.float, nonfinite 및 tensor 거부,
   중간 저장 실패 시 final 경로가 완료처럼 남지 않는 negative fixture를 포함한다.
4. 현재 CPU RCA fixture에서 scalar-only 변환 roundtrip PASS, NaN/Inf/ndarray 거부 PASS.
   이는 수리 가능성 검증이며 production 수리·GPU 검증 PASS가 아니다.

Frozen controller line 123 이후 **ideal ray slope로 quadratic curvature**를 계산하고,
actual Armijo는 실제 FP32 displacement slope를 쓰는 S4 선행 수리를 보존해야 한다.
SH3 옛 `2ff2393f` controller로 되돌리면 안 된다. 다른 S4 변경은 seed 고정,
node/Python/import/evaluator manifest binding, lazy M 및 SVD 전 current oracle 해제,
history receipts 등이다. S3 host/runtime/119GiB 설정은 새 lock으로 다시 결속한다.

## 5. 재사용·최소 restart

`save_checkpoints=false`, `exact_crash_resume=NOT_AVAILABLE`.
원 output 12개 파일은 JSON 및 실패 partial이며 WN/M/G/geometry tensor나 endpoint/resume bundle이 없다.
Native scalar/hash/spectrum receipts는 실제 weight/target/gradient를 복원하지 못한다.
따라서 **fresh W0/zeroM4의 B1 shared native 한 번부터** 동일 B300을 재시작해야 한다.
해당 native·current keys·geometry·gradient·controller 계산이 최소 필요 단위다.
EN_EXACT receipt 단계부터 이어서 실행하거나 기존 trajectory에 새 결과를 이어 붙일 수 없다.

재사용 가능: 원 model/tokenizer/Pstar/context/dataset, 이미 SH3가 검산한 동일 R512/Dev128
2560files/101519959223B, source/CPU evidence 및 이전 비용·기술 관측.
원 reference 101.5GB 역전송·재생성은 불필요하다. 새 at-write teacher는 없으므로 이관할 것도 없다.
S3 변경 경로의 필요한 bounded T0 범위는 새 authority를 따르며 과거 S4 관측을 S3 PASS로 바꾸지 않는다.
S4 runtime/source/raw/teacher는 수정·삭제하지 않았다. S4 새 제출·release·retry 0.

## 6. 인계 자료와 검증 수준

- [CPU 재현 코드](../../../../../audits/servers/server4/2026-09-20-en-adaptive-nullspace/failure-51260-v1/forensic_cpu.py)
- [CPU 재현 결과](../../../../../audits/servers/server4/2026-09-20-en-adaptive-nullspace/failure-51260-v1/cpu-reproduction.json)
- [원 파일 size/SHA inventory](../../../../../audits/servers/server4/2026-09-20-en-adaptive-nullspace/failure-51260-v1/raw-inventory.json)
- [작은 인계 allowlist](handoff-manifest.json)

새 package는 lock/config/source/첫 stderr/실패receipt/T0 compact만 exact allowlist한다.
모델/teacher/weight/raw prompt/full stdout은 포함하지 않는다. 원 frozen source는 Git object를 우선 사용한다.
SH3 sole pull; S4가 원격 파일을 덮어쓰거나 전송하지 않는다. 원 파일은 모두 보존한다.
직접 app-server 관련 turn `01a0be9b-1f0c-78c1-ab65-ec3be39cb3b0` steer 수락 확인;
실패한 dynamic wrapper 호출은 전달 성공으로 세지 않았다. GH도 첫 RCA 수신 ACK를 보냈다.

독립 agent red는 미사용. Owner source audit + 작은 CPU frozen-controller 재현이다.
GPU/model/forward/새 evaluator 0, 전체 과학 결과 리뷰 아님. NO_BROADCAST_NOT_REQUIRED.
RCA·인계 완료 후 SH4 monitoring_active=false / automatic_resume=false / STOP.
