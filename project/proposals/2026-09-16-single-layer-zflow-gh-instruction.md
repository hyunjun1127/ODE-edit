# GH 전달 지시문 — Single-Layer Write-Coupled z-Flow 구현 및 SEQ1000

Instruction ID: `GH-SL-ZFLOW-IMPLEMENT-SEQ1000-20260916-V1`

수신: Global Head(GH). 실행 담당은 GH가 실제 자원과 자산 위치를 확인해 배정하되 기존 BLUE-L4 실행 자산이 있는 Server4 SH를 우선 검토한다.

상태: 사용자 전달용 지시문이다. 파일 작성 자체로 GH/SH 전송, 원격 코드 변경 또는 GPU 제출을 수행하지 않았다. 아래는 사용자가 GH에게 본 지시문을 전달했을 때 수행할 작업이다.

## 1. 수행 목표와 기본 범위

**Single-Layer Write-Coupled z-Flow를 실제 Llama runtime에 연결하고, 필요한 기술 검증을 마친 뒤 main 설정 하나를 W0/M0부터 동일 fixed-order 1,000개 요청의 B100×10 sequential chain으로 실행·보고하라.** 기존 BLUE-style L4-only(N4)를 가장 가까운 대조로 사용하라. 기술 점검이나 첫 batch 결과만 남기고 끝내지 말고, 유효한 main chain의 1,000개 처리와 산출물 정리까지 진행하라.

이번 지시문의 기본 과학 실행 범위는 **신규 SL-ZFlow main 1 chain/10 batches**다. 기존 N4가 시작 상태·요청 순서·runtime·평가에서 비교 가능하면 재사용하라. 필요한 N4 비교 자료를 재사용할 수 없는 경우에만 N4 1 chain/10 batches를 같은 조건에서 추가하라. 따라서 기본 신규 범위는 10 batches, 조건부 N4 포함 최대 20 batches이며, 기술 검증 실행은 별도 계수하라.

Main은 **양의 write 비용 + barrier off**다. Fixed/exponential barrier와 동일 actual-write objective의 Adam 비교는 구현 가능한 설정과 후속 실험 계획으로 남기되 이번 main 실행에 새 과학 chain으로 자동 추가하지 말라. Lambda/b sweep, 새로운 reference dataset, 다층 편집, REFIT4 재실행, full10k·추가 order도 이번 기본 범위에 포함하지 않는다.

MPES/L4-only/REFIT4 및 기존 target/write mismatch 결과는 방법 개발 근거로 재사용하라. Motivation 재증명, 광범위한 보호 데이터 구축, 별도 downstream panel 완료를 구현의 선행조건으로 추가하지 말라. 단일 batch의 RS/PS/NS가 좋아야 다음 batch를 실행하는 성능 gate를 만들지 말라.

## 2. 기준 산출물을 확보하고 source를 봉인하라

- [전체 파이프라인](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-16-single-layer-zflow-pipeline-v1.md)
- [Reference 설정](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-16-single-layer-zflow-contract-v1.json)
- [설계 리뷰](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-16-single-layer-zflow-design-review-ko.md)
- [CPU 구현 안내](/mnt/raid5/janghj/ODE-edit/project/run_scripts/single_layer_zflow/README.md)
- [CPU 검증·source manifest](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-16-single-layer-zflow-pipeline-checks.json)
- [CPU end-to-end 예제 결과](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-16-single-layer-zflow-pipeline-cpu-demo.json)

현재 완료 범위는 CPU reference다. 30개 테스트가 통과했고 비선형 toy에서 25 oracle/18 accept/6 reject 후 마지막 후보 materialization과 history 1회 확정을 확인했다. 해당 solver 상태는 RESOURCE_STOP이다. 실제 Llama adapter, GPU 성능, durable checkpoint transaction, 지식 편집 성능까지 검증됐다고 쓰지 말라.

새 파일들은 검토 workspace에 있으나 아직 main commit이나 실행 서버에 존재한다고 보장되지 않는다. Git pull만으로 확보했다고 가정하지 말고 위 파일과 source manifest를 실제로 대조하여 package/import하라. 원본 파일을 확보하지 못하면 다른 설계를 추측하여 대체하지 말고 정확한 누락 경로를 보고하라. 새 GPU 실행에는 별도의 source commit/hash, import path, config, run lock을 남겨라. CPU 검증 SHA를 수정된 runtime의 SHA로 사용하지 말라.

GH는 구현 책임과 변경 경로를 명시한 dispatch를 발행하라. 주 구현 namespace는 `project/run_scripts/single_layer_zflow/`이며 필요한 native binding, prefix/suffix adapter, runner와 checkpoint 코드를 여기에 추가하라. 기존 native fitter와 다른 실험은 reference로 유지하라. 격리된 `codex/` branch 또는 작업 공간을 사용하고 다른 작업자의 변경을 되돌리지 말라.

## 3. 방법의 불변 조건

물리적 편집 weight는 `model.layers.4.mlp.down_proj.weight` 하나다. 각 B100의 entry를 W_e/M_e로 두고,

\[
X_0=0,\quad W(X)=W_e+XB
\]

로 시작하라. W_e, P, M_e, B, teacher와 token/context/position 상태는 inner trajectory 동안 고정한다. Accepted 후보를 다음 node의 좌표로 사용하되 실제 parameter와 history는 terminal까지 변경하지 않는다.

Native writer map과 cost는 다음과 같다.

\[
N=P(KK^\top+M_e)+\lambda_{write}I,\quad
B=E\operatorname{solve}(N,PK)^\top,
\]

\[
S=(BM_eB^\top+\lambda_{write}BB^\top)/m,\quad
C(X)=\tfrac12\operatorname{tr}(XSX^\top).
\]

비대칭 N을 임의로 대칭화하지 말라. P/M의 물리 layer binding과 native key aggregation을 확인하라. Pinned BLUE는 context group 내부 평균 후 group 간 평균으로 request당 key 하나를 만든다. 기본 clean 1개/generated 5개 구성에서 key 가중치는 clean 1/2, generated 각각 1/10이며, edit loss는 여섯 context 각각 1/6이다. 이 두 가중치를 같게 바꾸지 말라.

목적함수는

\[
F=L_{edit}+\beta L_{essence}+\lambda_{flow}C
\]

다. Edit token weight는 1/(m c_i T_i), essence는 기존 request-derived prompt/lookup과 KL(p_current||p_entry), beta=.0625를 사용하라. Teacher는 해당 branch의 batch-entry에서 한 번 생성하고 고정한다. 별도 neighborhood/protection dataset을 controller 입력으로 추가하지 않는다.

**완성된 native z를 먼저 계산하지 말라.** Native25 warm start, j_native/C_native normalization, native endpoint를 먼저 만들고 그 주변만 보정하는 방식은 이 method가 아니다. X는 writer residual 좌표이며 realized canonical hidden은 H_c,e+XBK_c다. X를 실제 hidden displacement와 동일시하지 말라.

## 4. 실제 Llama oracle를 구현하라

고정 training sequence의 L4 down-projection 입력 K_p와 entry block output H_p,e를 한 번 계산하고 A_p=BK_p를 cache하라. 후보는

\[
H_p(X)=H_{p,e}+XA_p
\]

로 구성하여 L5–31 suffix에 전달하라. 이는 모든 token에 대한 actual-write 경로다. Subject-only intervention, request별 독립 z fitting, 성공한 column의 임의 freeze로 바꾸지 말라.

Tokenizer revision, target IDs 처리·decode/join·prediction shift, padding, attention mask, position_ids/cache_position/RoPE를 native source와 연결하라. Canonical key prompt와 teacher-forced training prompt의 token-ID prefix가 같은지 확인 없이 cache를 공유하지 말라. Frozen prefix에 실제로 영향을 주는 parameter나 state가 바뀌면 해당 cache를 무효화하라. Downstream KV cache는 재사용하지 않는다.

Oracle는 전체 request/context의 논리적인 L/g sweep이며 microbatch는 메모리 분할이다. 전역 request/context/token weight로 full X gradient를 누적하라. Teacher/prefix는 detach하되 suffix의 X-input gradient는 유지하라. 학습 대상이 아닌 모델 parameter의 dense gradient를 만들지 말라.

실제 adapter는 필요한 target prediction/KL 위치의 hidden을 모은 후 full-vocabulary head를 적용하라. Vocabulary 자체를 줄이지 말라. CPU oracle의 full-logits callback을 production head 최적화 완료로 취급하지 말라. Raw K_p는 A_p 준비 뒤 해제할 수 있다.

## 5. Main numerical settings와 fresh-compute 정책

다음은 미튜닝 working setting이며 과학 실행 전에 run lock에 고정하라.

| 항목 | 값 |
|---|---:|
| physical layer | 4 |
| lambda_write | 1 |
| lambda_flow | 1 |
| beta_essence | .0625 |
| metric relative damping | 1e-3 |
| initial/max eta | 1 / 1 |
| reject shrink | ×.5 |
| clean accepts 후 회복 | 연속 2회 뒤 ×1.5, max eta 이하 |
| max_oracle_calls | 25 |
| barrier_mode / budget | off / null |

나머지 Armijo·stationarity·feasibility 설정은 reference JSON을 명시적으로 상속하라. Lambda_flow=1은 현재 cost 단위에서의 초기값이지 최적 trade-off가 검증된 값이 아니다. Lambda_write와 값이 같다는 이유로 정당화하지 말라. 성능을 보고 batch별 lambda·step cap·NFE를 바꾸지 말라. 수치 경로 변경이 필요하면 변경 전후를 기록하고 과학 chain의 정책을 고정하라.

H=S+epsilon I를 사용하고,

\[
Y=(XH-\eta g)(H+\eta\lambda_{flow}S)^{-1}
\]

로 candidate를 만든다. Candidate에서 L/g를 한 번 fresh 계산하고 충분 감소로 수용 여부를 결정한다. Accepted L/g를 다음 step에 carry하며, reject 시에는 이전 L/g를 유지한다. N_oracle=1+accepted+rejected를 검산하라. Rejected backward도 비용이다. Scalar budget enforcement나 step 산술에는 model 호출이 없다.

Reference의 roundoff guard도 보존하라. F 감소가 rounding 허용폭보다 큰 수용만 clean accept로 센다. Rounding 범위에서는 이미 계산한 candidate residual이 max(이전 residual, 종료 threshold)를 초과하면 거절한다. 이때 추가 model pass를 만들지 않는다.

Barrier API의 의미는 off, fixed:C(Y)<=b, exponential:C(Y)<=C(X)+(1-exp(-kappa eta))(b-C(X))로 구분하라. On mode는 positive b를 요구하고 exponential의 kappa는 양수다. 누락된 budget을 조용히 off로 해석하지 말라. Kappa=0을 fixed mode로 대체하지 말라. 이번 main scientific run에서는 barrier를 사용하지 않는다.

## 6. 종료·materialization·history·재개

작은 budget의 boundary 판정은 (b-C)/b를 사용하고, raw와 normalized KKT 항목을 함께 저장하라. Exponential의 임시 cap, 작은 step, target top-1 최초 통과를 final-set stationarity로 부르지 말라. Reference의 multiplier score는 선언한 단위의 기준값을 사용하며 모든 목적함수 scaling에 불변인 인증으로 확대하지 않는다.

`FIRST_ORDER_STATIONARY`, `RESOURCE_STOP`, `NUMERICAL_STOP`을 구분한다. 정상조건은 reduced-space 1차 조건이지 global optimum이나 높은 edit 성공률의 인증이 아니다. Resource/numerical stop에서도 유효한 accepted 후보가 있으면 마지막 후보를 terminal 검사에 넘긴다. Accepted가 0이면 no-update attempt receipt를 남기고 W/M를 바꾸지 않은 채 다음 batch로 이어갈 수 있다. 요청은 평가의 원분모에 유지한다.

Terminal에서 실제 FP32 W_c를 만들고, 저장될 두 weight의 차이 `W_c.double()-W_e.double()`로 actual cost를 측정하라. Cached suffix와 full actual-write의 logits/NLL을 비교하고, 실제 저장될 weight가 선택된 후보와 같은지 확인하라. Candidate FP64→FP32 전환이나 다른 GEMM 경로의 차이를 이 검증에 포함하라.

성공하면 native CPU FP32 순서로 M_next=M_e+K K^T를 한 번 갱신한다. Inner candidate/reject에서는 append 0회다. 기존 NativeSingletonFitter.finalize만 호출한다고 중복 호출이 자동 방지되는 것은 아니므로 batch_id/parent receipt와 결속하라.

실제 checkpoint는 W/M/config/context/RNG/ledger와 필요한 resume state를 일관되게 확정하라. 임시 bundle 완성→hash manifest→같은 filesystem의 atomic commit 순서를 사용하고 완성 manifest만 resume 대상으로 인정하라. 동일 committed batch 재시도는 no-op이고, 다른 payload는 conflict다. Weight만 복구한 상태를 full resume이라고 부르지 말라.

Parity/cost 실패는 `COMMIT_PARITY_FAIL`/`COMMIT_COST_FAIL`로 구분하고 owned W/M를 entry로 복원하여 그 batch에서 중단하라. 실제 오류를 수정한 뒤 동일 정책의 완전한 checkpoint에서 재개하라. 이를 no-update 과학 결과로 조용히 넘기거나 native fallback으로 대체하지 말라. CPU in-memory transaction을 디스크 crash recovery 구현으로 재사용했다고 주장하지 않는다.

## 7. 필요한 기술 검증 후 SEQ1000을 진행하라

검증은 다음 correctness에 집중하라.

1. Native key/group/E 및 direct RHS solve와 factored XB의 수치 연결.
2. 실제 Llama full-write 대 cached suffix의 logits/NLL/X-gradient.
3. All-token injection, teacher/position/label shift, microbatch partition과 전역 weight.
4. Candidate/reject 무변경, FP32 materialization, 실제 비용, history exactly-once, checkpoint resume.
5. Tiny-budget KKT와 step recovery를 포함한 기존 CPU 회귀 검사 유지.

실제 runtime dtype/backend의 수치 오차를 측정해 허용오차를 정하고, method 성능을 평가하기 전에 봉인하라. 실패 후보마다 오차를 넓혀 통과시키지 말라. CPU toy의 logit tolerance를 Llama 기준으로 복사하지 말라. 기존 fixture와 작은 실제 모델 점검을 이용하되 기술 검증 범위를 과학적 성능 gate로 바꾸지 말라.

Main은 편집 전 W0/M0에서 fixed10k의 첫 1,000개, B001–B010을 처리한다. 비교 가능한 기존 BLUE-L4의 model/tokenizer revision·context/sample order·runtime identity를 사용하라. Toy seed31을 과학 seed로 가져오지 말고 실제 baseline의 sample/context/RNG lock을 확인하여 고정하라. B002 이후에는 자기 committed state에서 B/S/H_entry/teacher를 새로 준비하라.

N4 재사용 여부는 GH가 실제 source/state/evaluation identity로 판단하고 manifest에 남겨라. W50 REFIT4 suffix를 W0 comparator로 사용하지 말라. 기본 10 batches 또는 조건부 N4 포함 20 batches 범위 안에서 진행하고 첫 batch 성능만으로 중단하지 말라. Source 버그를 수정해 알고리즘 의미가 바뀌면 변경된 설정의 chain을 별도 attempt로 구분하라.

GPU-hour를 기존 native나 toy 시간에서 단정하지 말라. 기술 점검에서 실제 batch work와 peak memory를 계측해 자원을 산정하라. 사용자가 지정하지 않은 고정 speedup이나 online ratio 탈락선을 추가하지 말라. 다른 실행을 중단하거나 기존 output을 덮어쓰지 말라.

## 8. 보고·저장해야 할 결과

Official paraphrase/neighborhood 평가는 observer다. Lambda/b, 종료, rollback, 후보 선택에 넣지 않는다. 새 보호 데이터나 미측정 일반능력 panel을 main 실행의 조건으로 추가하지 않는다.

| 구분 | 필수 기록 |
|---|---|
| Method trace | node별 L_edit, KL, C, F, gradient/residual, eta, accept/reject, barrier mode·cap·dual, 종료 이유 |
| Commit trace | entry/terminal W/M identity, X/B/S 및 설정, actual delta cost/norm, parity 수치, append count, no-update/실패 구분 |
| Current | 매 batch R/P/N 성공 수·실제 분모, R/P TF-strict, true/new NLL·margin |
| Sequential | W5의 first500, W10에서 같은 first500과 전체1000, at-write→terminal lost/gained, active/superseded 구분 |
| Final comparison | N4와 W10 전체 요청 비교, paired 차이 및 NLL tail, 원분모 유지 |
| Compute | prefix/teacher, B/S/eigen, 전체 oracle 및 accepted/rejected F/B, 유효·padded token, head work, terminal parity, commit/I/O, peak memory |
| Reproducibility | source/import/config/sample/context/model identity, checkpoint inventory, 실제 저장 경로·hash, resume 가능한 범위 |

N_oracle와 별개로 teacher/prefix/terminal parity 비용을 더하라. 기존 native max25는 최대 25 forward/24 backward이고 요청별 early stop이 있으므로, 새 25 whole-batch sweep와 같은 비용이라고 쓰지 말라. Matrix solve 재사용만으로 speedup을 설명하지 말라. 중첩 timer와 순수 compute/평가/I/O도 구분하라.

큰 model/tensor/raw는 run별 local artifact root에 저장하고 Git에는 코드·compact manifest·집계·보고를 남겨라. 매 batch의 commit 연결과 실제로 재개 가능한 checkpoint를 보존하되 저장 비용도 계측하라. 최종 W10은 동일 base-model identity와 함께 후속 inference 평가가 가능해야 한다.

SH는 사실·수치·분모·source·실측 비용·오류·미측정만 보고하고, GH는 별도 global review에서 품질–보존–비용과 claim을 해석하라. 높은 RS/PS와 NS, strict/NLL tail, no-update 비율을 함께 보고한다. C 감소를 output locality 보장으로, RESOURCE_STOP을 optimal endpoint로, 성공한 main 하나를 ODE/IMEX 고유 기여의 증명으로 확대하지 말라.

## 9. GH의 첫 회신과 최종 전달물

첫 회신에는 **source package 확보 여부, 실제 미구현 adapter/transaction 목록, N4 재사용 판단, main 10/조건부20 batch 범위, 필요한 자산·자원 및 실행 source/output 계획**을 간결히 남겨라. 이후 이 범위의 구현·기술 검증·sequential 실행·보고를 이어가라. 새 motivation gate를 제안하는 것으로 실제 구현을 대체하지 말라.

최종 전달물은 다음이다.

1. 실제 Llama adapter와 실행 runner 및 source/config/import manifest.
2. CPU 회귀·Llama parity·commit/resume 검증 receipt.
3. SL-ZFlow SEQ1000의 batch/node/terminal 결과 및 raw artifact inventory.
4. N4 재사용 또는 조건부 재실행 근거와 같은 분모의 비교표.
5. SH 사실 보고와 분리한 GH 해석.
6. 동일 actual-write objective의 Adam 대조와 barrier ablation을 어떻게 구분할지에 대한 후속 계획. 아직 수행하지 않은 비교는 미측정으로 남긴다.

완료 범위와 미완료 범위를 명확히 보고하라. Main이 좋지 않거나 resource stop이 많아도 그 결과를 그대로 보존하고, 다른 방법으로 조용히 바꾸어 성공한 것처럼 제출하지 말라.
