# Alpha-key gate52527 실패 진단 — 원 native와 다른 BOS 검사 가정

진단 nonce: `ODEEDIT-GH-SH4-ALPHA-KEY-GATE-FAILURE-CHECK-20260923-R1`.
상태: `DIAGNOSIS_COMPLETE_NOT_REPAIRED / STOP_AWAITING_USER`.
이번 작업은 CPU/source 진단이다. 신규 GPU·모델 load/forward·수리·Slurm write·재제출·threshold 변경은 모두 0이다. 과거 PENDING 인계는 그대로 보존했다.

## 1. 결론과 실제 완료 경계

Gate **52527 / odeedit_alpha_key_gate_s4**는 `TechnicalFailure: WRITER_UNEXPECTED_BOS: 0`으로 실패했다. 최초 throw는 frozen `technical.py:200`의 “첫 토큰이 BOS가 아니어야 한다” 검사다. G1의 첫 model forward보다 앞이며, OOM·시간 제한·IO·serialization·수치 불일치 또는 성능 탈락이 아니다.

같은 pinned tokenizer와 원 native 구성 절차를 CPU에서 재현했다. 실제 class는 `PreTrainedTokenizerFast`; `add_bos_token=False` 대입은 이 class의 backend postprocessor를 바꾸지 않는다. Backend의 `TemplateProcessing`은 `BOS + A`를 계속 사용한다. 첫 두 실제 요청×6 context **12/12 입력에 BOS 128000이 있었고**, 원 native/current 구성으로 얻은 `input_ids`와 `attention_mask`는 exact 일치했다. Frozen `token_fixtures`를 그대로 호출했을 때 같은 첫 exception을 재현했다.

따라서 직접 확인된 원인은 **새 기술 gate가 설정 attribute False를 실제 BOS 부재로 해석한 잘못된 가정**이다. Native tokenizer를 바꾸거나 수치 허용치를 느슨하게 만드는 문제가 아니다. 원 과거 실행의 전체 token stream이 저장·독립 검증됐다고 확대하지 않으며, 재구성한 exact pinned 설정/입력의 CPU tokenization evidence다.

## 2. Exact job/accounting과 후속 영향

제출 lock/4개 job mapping을 먼저 확인한 뒤 지정 ID만 `sacct`로 한 번 조회했다. 모두 owner `janghj`, node/실행 경계는 아래와 같다. 시간은 Slurm의 server4 KST다.

| Job | 단계 | 상태/exit | Start → End (2026-09-23) | Parent GPU초 |
|---|---|---|---|---:|
| 52527 | gate | FAILED / 1:0 | 09:59:04 → 09:59:48 | 44 |
| 52528 | geometry | CANCELLED / 0:0 | Start=None → 10:00:04 | 0 |
| 52529 | writers | CANCELLED / 0:0 | Start=None → 10:00:04 | 0 |
| 52530 | CPU reducer | COMPLETED / 0:0 | 10:00:04 → 10:00:04 | 0 |

52528/52529는 `afterok:52527`와 `--kill-on-invalid-dep=yes`로 제출됐다. 시작·allocation·과학 출력이 모두 없으므로 gate 우회 실행은 발견되지 않았다. 취소 actor는 이번 accounting에 기록되지 않았으며, 이 상태는 제출된 invalid-dependency 취소 설정과 부합한다. 이번 진단 agent는 cancel/hold/release를 호출하지 않았다.

52530의 `afterany:52527:52528:52529`는 원래 실패 보고용 CPU 경로다. GPU0으로 정상 종료했고 자동 report도 `TECHNICAL_FAILED_OR_PARTIAL`, 완료 family **0/94**로 남겼다. Reducer의 COMPLETED를 과학 완료로 쓰지 않는다. 원 PENDING 사유가 accounting에 남은 `ReqNodeNotAvail`는 이번 Python 최초 오류와 구분한다.

## 3. Source-backed RCA와 CPU 반례

실행 source `a95876f8e5c4cf59df9cd9d7f824d1ac99f8bc77`, tree `37729895ac354d6d4029e41d0b6e31fbb8d9e39c`.
Archive `bf072836f3ef47395650aa994816609e9385ee25b70e22b0a109069190ff3768` / 470352289B.
Lock `4a80070051cbbcc1b5e1d01f0e94124ddc92f79bedaba3772c42dcbbc7530725`.
원 execution-source와 execution branch는 수정하지 않았다. Frozen 31개 source member의 현재 SHA/size가 lock과 모두 일치했다.

| 근거 위치 | 실제 내용 | 판정 |
|---|---|---|
| 원 native runtime.py:28–29 | AutoTokenizer 생성 후 add_bos_token=False 대입 | 읽기 전용 원 구성 |
| 신규 runtime.py:55–56 | 동일한 tokenizer 생성/attribute 대입 | 원 구성 절차 유지 |
| 신규 technical.py:193,200 | default special-token encode 후 첫 token BOS를 금지 | CPU에서 첫 오류 재현 |
| 원 rome/repr_tools.py:59–61 | default tok.encode로 prefix/subject 위치 계산 | BOS 제거를 강제하지 않음 |
| 신규 technical.py:224–228 | token fixture 뒤에 prefix forward | forward 이전 실패 |
| 신규 technical.py:279–290 | finally restore/hash/RNG 검사 뒤 원 exception 재throw | secondary 오류 기록 없음 |

CPU fixture case IDs는 18246/6208, ordinal[5000,5002)다. Context는 보존된 B050 contexts.json을 원 commit의 context hash에 대조했다. 12개 sequence의 right-padding/subject lookup 범위는 정상이고, BOS만 새 가정과 충돌했다. Synthetic `Alpha example`도 default encode의 첫 token128000, `add_special_tokens=False`의 첫 token19947로 확인했다. 후자는 반례 설명용이며 production tokenizer에 적용하지 않았다.

CPU 진단에서 모델/forward/평가/CP tensor load는 0이다. 진단 helper 작성 중 native import cwd(`globals.yml`) 및 Sequence postprocessor 출력 parser만 바로잡았으며, 이는 원 job의 오류나 runtime 수리가 아니다. 과거 CPU132 PASS는 실제 tokenizer를 사용하는 이 G1 경로의 보장이 아니었다.

## 4. Gate·복원·재사용 가능한 범위

| 항목 | 확인된 범위 |
|---|---|
| G0 | frozen source/input 검증, model load, W0 hash, W50 state load 경로 도달. 완전한 actual gate seal 없음 |
| G1 | token fixture 첫 BOS 검사에서 FAIL; parity comparisons0, forward timing0 |
| Timestamp K bank | 미생성 |
| G2 native100 / SHAM | 모두 NOT_RUN; target fit0, solve0, 새 history append0 |
| G3 | 사전등록 graph는 작동했고 afterok가 science를 막음. 실제 initial gate PASS 아님 |
| E1–E4 | 0/94 완료; R/P/N/N512/H512 새 과학 metric 없음 |
| 복원 | finally의 restore와 hash/RNG 검사를 지난 뒤 line290에서 원 오류 재throw. 별도 cleanup failure 없음. 독립 성공 receipt/사후 tensor 검증은 NOT_RECORDED |

12개 입력 CP는 기존 fullSHA/content 검증 receipt와 현재 device/inode/mtime/size가 모두 일치했다. 이번에 63.4GB를 재해시하거나 재전송하지 않았다. 원 model/P/data/context/native source, 입력 CP12, 기존 transfer/CPU 검증은 재사용 가능한 입력이다. 실패 receipt·stderr·자동 reducer 자료도 보존한다.

새 target/K/R/Δ/timestamp bank/native100 결과는 없으므로 계산 재사용할 과학 산출물은 없다. 새 full-state CP 역시 없다. 수리가 별도 승인되면 immutable 새 source/attempt에서 입력 W50을 다시 로드하고 미완료 G1/timestamp 검증부터 수행해야 한다. 이는 저장된 실행 중간 상태의 exact crash-resume이 아니다.

## 5. 실제 비용과 미측정

- Parent allocation: 1GPU×44초 = **44 GPU초 / 0.012222 GPUh**. batch/extern의 44초를 다시 더하지 않았다.
- Failure receipt의 program elapsed: **42.302591074초**.
- G1 failure receipt: **2.816167830초**, catch 시각까지이며 finally restore는 포함하지 않는다. 상위 elapsed와 중복 합산하지 않는다.
- Slurm batch MaxRSS: **43628732K**(약41.61GiB). Host 요청은59GiB. GPU peak와 model/source/IO별 분리는 `NOT_RECORDED/NOT_SEPARATED`다.
- CPU reducer TotalCPU: 0.416초, parent elapsed는 초 단위0. Allocation을 GPU utilization으로 표현하지 않는다.
- 신규 진단 GPU 비용0. 과거 transfer/CPU 준비 비용을 이번44GPU초에 중복 청구하지 않는다.

## 6. 최소 수리 제안 — 이번 turn 미구현

1. 원 native tokenizer/호출/특수 token을 그대로 두고, `technical.token_fixtures`의 blanket no-BOS 가정을 **원 native token/backend/attention mask/subject lookup exact binding**으로 교체한다.
2. Attribute `add_bos_token`와 실제 encoded BOS 여부를 별도 기록한다. 같은 pinned `PreTrainedTokenizerFast`와 actual2×6 context CPU 회귀를 추가한다. mask/lookup/input identity 검사는 유지한다.
3. 과학식·threshold·SHAM/numeric gate를 바꾸지 않는다. 단순 `add_special_tokens=False` 강제, BOS 삭제 또는 gate 전체 skip은 원 native와 다른 입력을 만들므로 이 진단의 수리안이 아니다.

이는 제안만이다. 이번 turn에서 원 runtime/config/threshold 수정, GPU 재검증, 새 제출은 하지 않았다. 현재 실패는 해결됐다고 기록하지 않는다.

## 7. 재현·증거·종료

[CPU 반례 source](../../../../../audits/servers/server4/alpha-key-causal-20260923-r1/failure-diagnosis-r1/reproduce_bos_cpu.py), [CPU 반례 receipt](../../../../../audits/servers/server4/alpha-key-causal-20260923-r1/failure-diagnosis-r1/bos-cpu-reproduction.json), [진단/비용/inventory](../../../../../audits/servers/server4/alpha-key-causal-20260923-r1/failure-diagnosis-r1/diagnosis.json).

진단 branch root에서 다음 명령으로 새 diagnostic receipt를 만든다. 이미 존재하는 output을 덮어쓰지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /data/janghj/EasyEdit/.venv/bin/python audits/servers/server4/alpha-key-causal-20260923-r1/failure-diagnosis-r1/reproduce_bos_cpu.py --root /data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1 --output /data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/failure-diagnosis-r1/bos-cpu-reproduction-new.json
```

진단 local root: `/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/failure-diagnosis-r1/`.
원 실패: `execution/attempt-r1/gate-failure.json`, `gate/prefix-g1/G1-first-failure.json`, `logs/gate-52527.err`.
Raw/log/weights는 Git에 넣지 않았다. 별도 독립 agent red는 이번 진단에서 사용하지 않았으며 owner source audit+CPU 최소 반례로 기록한다. NO_BROADCAST_NOT_REQUIRED.

전용 진단 branch의 compact package만 게시한다. 원 execution branch/main은 변경하지 않는다. `STOP_AWAITING_USER`, `monitoring_active=false`, `automatic_resume=false`; 주기 조회·자동 수리·재제출0.
