지시문을 자세히 파악하고 SH2에게 task 전달하라. server2의 GPU CAP은 2이다.

=============

# GH 지시 — Single-Layer Write-Coupled z-Flow 구현 및 검증

Instruction ID: `GH-SL-ZFLOW-IMPLEMENT-SEQ1000-20260916-V1`

다음 상세 지시문을 읽고 실제 Llama 구현, 기술 검증, sequential 실행과 결과 정리까지 진행하라.

상세 지시문

## 1. 실행 범위

Single-Layer Write-Coupled z-Flow를 실제 runtime에 연결한 뒤, **main 설정 하나를 편집 전 W0/M0부터 동일 fixed-order 1,000개 요청의 B100×10으로 실행하라.**

가장 가까운 대조는 기존 BLUE-style L4-only(N4)다. 비교 가능한 기존 결과를 재사용하고, 재사용할 수 없는 경우에만 N4 1,000개 chain을 추가하라. 기본 신규 범위는 10 batches, 조건부 N4를 포함하면 최대 20 batches다.

단일 batch는 구현 correctness 확인에 사용하라. 첫 batch의 성능을 이유로 후속 실행을 선별하거나 motivation 재증명을 새 선행조건으로 만들지 말라.

## 2. 반드시 유지할 방법

- 물리적 편집 대상은 L4 down-projection 하나다.
- 매 batch에서 `X=0`, `W(X)=W_entry+XB`로 시작한다.
- Native writer map B와 비용 Gram S는 batch당 한 번 계산한다.
- Actual-write edit loss, 고정 entry essence KL, quadratic write cost를 함께 최적화한다.
- 완성된 native z fitting을 먼저 수행하거나 `j_native`로 정규화하지 않는다.
- 전체 token의 affine cache를 사용하고, 후보마다 nonlinear suffix의 fresh loss/gradient를 계산한다.
- Accepted gradient는 다음 step에 carry한다. Rejected 후보의 계산도 비용에 포함한다.
- Inner trajectory에서는 실제 weight/history를 변경하지 않는다. 마지막 accepted 후보만 실제 weight로 구성하고 검증한다.

Main 설정은 `lambda_write=1`, `lambda_flow=1`, `beta=.0625`, `barrier=off`, `initial/max step=1`, `max_oracle_calls=25`다. Lambda_flow는 미튜닝 초기값임을 명시하라. 나머지 수치 설정은 reference contract를 상속하라.

Fixed/exponential barrier와 동일 actual-write objective의 Adam 대조는 후속 비교로 구분하라. 이번 main에 추가 chain이나 parameter sweep를 자동으로 넣지 말라.

## 3. 실제 구현해야 할 부분

현재 코드의 완료 범위는 CPU 참조 구현이다. 다음을 실제 runtime에 연결하라.

1. Native context·key·tokenizer binding.
2. Llama prefix cache와 suffix oracle.
3. 필요한 prediction/KL 위치만 사용하는 full-vocabulary head.
4. FP32 materialization과 full actual-write parity.
5. Weight/history를 함께 확정하는 durable checkpoint·resume.
6. Batch runner와 계산량·평가 기록.

새 코드가 main이나 실행 서버에 이미 있다고 가정하지 말라. 상세 지시문의 파일과 source manifest를 확보하고, 실제 실행 source/config/import identity를 별도로 봉인하라.

## 4. 기술 검증과 종료 규칙

Actual full-write와 cached suffix의 logits/NLL/X-gradient, 전체 token 반영, microbatch 가중치, 고정 teacher, history exactly-once와 resume을 확인하라.

작은 budget의 KKT 수정과 step 회복·roundoff guard를 유지하라.

`FIRST_ORDER_STATIONARY`, `RESOURCE_STOP`, `NUMERICAL_STOP`을 구분하고 계산 한도 종료를 최적 endpoint 인증으로 쓰지 말라. Accepted 후보가 없으면 no-update로 기록하며 요청은 평가 분모에 유지하라.

Parity 또는 실제 비용 검사 실패 시 weight/history를 entry로 복원하고 기술 오류를 해결하라. Native fallback이나 요청 제외로 대체하지 말라.

## 5. 필수 보고

- Node별 edit loss/KL/C/F, step, residual, accept/reject와 종료 이유.
- 실제 materialized weight의 비용·norm·parity와 history append.
- 매 batch Current R/P/N, strict, true/new NLL.
- W5 first500과 W10 전체1000 및 같은 first500의 유지 변화.
- N4와 같은 분모의 최종 비교.
- Prefix/teacher 준비, accepted·rejected suffix F/B, token work, terminal parity, commit/I/O와 peak memory.
- Source/config/sample identity, checkpoint와 raw artifact inventory.

`N_oracle=1+accepted+rejected`를 검산하라. 새 25 whole-batch sweep를 native 25 forward/24 backward와 동일 비용으로 취급하지 말라.

SH는 사실과 수치를 보고하고 GH는 별도 global review에서 품질·보존·비용을 해석하라. CPU 테스트 30개 통과를 실제 Llama 편집 성능 검증으로 확대하지 말라.

첫 회신에 source 확보, 미구현 부분, N4 재사용 판단, 실행 범위와 자원 계획을 남긴 뒤 해당 범위의 작업을 끝까지 진행하라.
