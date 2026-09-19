# 수치 gate 구현의 bounded 독립 검토

대상 task: `ODEEDIT-S06-S2-CHECKPOINT-MECHANISM-AUDIT-20260920-V1`.
본 문서는 parent와 분리된 bounded 검토 agent가 작성했다. 소유한 신규
`validation.py`, `test_validation.py`와 이 문서 외의 source는 수정하지 않았다.
원격 접근·Slurm·모델 load·GPU 실행·commit/push는 수행하지 않았다.

## 읽은 기준과 범위

- Contract 전체: `plans/global/2026-09-20-server2-checkpoint-mechanism-audit-contract-v1.json`,
  SHA256 `9d469e90c0252228fc9a26d7d3ab9fdb2a256aba293c4dbbc58abff57c5f7238`.
- Design 전체: `plans/global/2026-09-20-server2-checkpoint-mechanism-audit-design-v1.md`,
  SHA256 `ae5d9d2b3b920125257e86d5b4bbba5afeae7b1b15114f244a927cfcf7386d00`.
- Selective input transfer 승인 전문 및 현재 `staging.py`/`common.py`.
- 최신 cap2 override는 실행 병렬화이고 threshold 변경이 아님을 인지했다.
  본 검토는 GPU admission 검증을 수행하지 않았다.

## Pure tensor API와 정확한 판정

위 함수는 `project/run_scripts/checkpoint_mechanism_audit/validation.py`에 있다.
입력은 CPU 또는 같은 device의 tensor이며 반환값은 JSON-safe scalar/list receipt다.
원 입력을 detach하여 읽고 수정하지 않는다. Shape broadcasting/빈 comparison은 거부한다.
입력 및 중간 계산의 nonfinite는 실패로 분류한다. 계산/축약 dtype은 FP64이며
원 입력 dtype, 검증 wall time, worst index, 실제 오차, 실패 수를 기록한다.

|API|고정 계약|
|---|---|
|`elementwise_gate(actual, reference)`|모든 원소의 `abs(error) <= 1e-6 + 1e-5*abs(reference)`|
|`solve_residual_gate(A, X, rhs, row_chunk=256)`|원 solve 결과의 FP64 잔차 검산. `max_j norm(A X_j-rhs_j)/norm(rhs_j) <= 1e-5`; backward-error 분모 아님|
|`update_gate(reconstructed_delta, actual_delta)`|오차 Frobenius norm / **actual** delta norm `<=1e-3`|
|`response_gate(reconstructed_delta, actual_delta, keys)`|request별 오차 response / actual response norm의 **최대** `<=1e-3`|
|`evaluation_rows_gate(actual_new, actual_true, reference_new, reference_true, metric_tags)`|각 row NLL와 safety-margin 절대오차 모두 `<=1e-4`; robust success bit 차이0|
|`repeat_spread_gate(observations, tolerance)`|반복 차원에서 원소별 max-min range가 사전 지정 tolerance의10% 이하|

모든 norm-relative 판정은 reference norm `<=1e-12`이면 상대값을 만들지 않고
절대 error norm `<=1e-7`을 요구한다. Flag/count를 남기며 상대값은 `null`이다.
Caller는 실제 delta를 반드시 `Wb.double()-Wa.double()`로 생성해야 한다.
이미 FP32로 뺀 차이를 double로 cast하는 것은 이 API 밖의 오류이며 허용되지 않는다.

RS/PS safety margin은 true−new, NS는 new−true이고 tie는 실패다.
비트가 바뀐 두 margin이 **둘 다** ±1e-4 안이면 `NUMERIC_BOUNDARY`로 기록한다.
이 경우 row별 수치 기준까지 만족한 receipt만 `passed=true`일 수 있고 changed bit를
삭제하지 않는다. 한 margin만 band 안이거나 robust bit가 바뀌면 실패한다.
Identity/order 결속은 caller의 선행조건이며 이 함수만으로 증명하지 않는다.

Repeat 함수의 tolerance는 결과를 보고 설정하는 값이 아니다. Key/block에서는
`1e-6+1e-5*abs(reference)`라는 원소별 allowance를 사용하고, 이미 정규화한 solve/update/
response 비교에는 해당 고정 relative threshold를 사용한다. 원자료 단위의 scalar
NLL 비교에는1e-4를 사용한다. 표준편차나 첫 repeat와의 차이만으로 전체 range를 대체하지 않는다.

## CPU regression

실행 명령:

```bash
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m unittest -q project.run_scripts.checkpoint_mechanism_audit.test_validation
```

Torch `2.9.1+cu128`, CPU24개 test PASS. 실제 GPU/model 호출0이다.
단일 원소/단일 작은 RHS/단일 request outlier, actual-vs-reconstructed 분모,
backward-error 오용, near-zero 경계, 서로 다른 NLL 오차의 margin 합산,
both-band/tie/NS 부호, repeat max-min, 입력·중간 overflow nonfinite,
원 입력 불변성/shape/empty/JSON finite를 검사했다.
CPU synthetic PASS를 actual Llama key/solve/evaluator parity PASS로 확대하지 않는다.
원 source/targets/torch backend 상태의 실제 검증은 owner의 단계별 receipt가 필요하다.

## 현재 staging에 대한 제한 검토

현재 source-map은2,304개 member와 plan member source집합이 정확히 같고
모든 mapped path가 regular file로 존재했다. Receiver READY는220개 파일/
450,763,288bytes 선택 수신, W/M transfer0, rsync exit0를 기록한다.

- `current.json`:100개/66,737,255bytes.
- `seen-full.json`:12개/370,456,132bytes.
- `native-targets.pt`:7개/11,733,953bytes.
- `commit.json`, `entry.json`, `native-observation.json`, `contexts.json`:각100개.

`staging.py`는 source manifest의 byte/SHA에 맞는 기존 local 파일을 우선 재사용한다.
Missing만 exact files-from으로 partial destination에 복사하고 full hash/size 검산 뒤
같은 filesystem rename으로 seal한다. `--delete`/`--remove-source-files`/W-M 복사는 없다.
SOURCE_KEEP를 유지한다. 이 검토자는 owner가 이미 한 대용량 full hash를 반복하지 않았고,
receiver receipts와 stat/presence 및 source logic의 결속만 독립 확인했다.
따라서 이 문서는 원격 실물의 독립 재해시 또는 actual model PASS 증거가 아니다.

## 잔여 실제 gate

실제 P/W/M, native mean/bare keys, h0 affine/full-forward, dense solve와 actual delta,
request별 response, row별 evaluator, 동일 경로 repeat spread는 아직 본 subtask에서
실행하지 않았다. 어느 하나 실패하면 해당 종속 claim만 막고, 유효한 저장평가/actual W
CPU 결과는 보존한다. 별도 gate skip, tolerance 완화, scientific fallback은 추가하지 않았다.
