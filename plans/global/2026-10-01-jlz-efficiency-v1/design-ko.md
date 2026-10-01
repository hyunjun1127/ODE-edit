# JLZ와 native baseline의 계산 효율화 설계

2026-10-01. 작성 대상은 GH와 SH1 구현 담당자다. 사용자 요청에 따라 과학적 목적과 실험 범위를 유지하면서 실행 비용을 줄이는 계산 그래프, 함수 변경, 검증 및 측정 순서를 정리한다. **우선 적용 후보는 오른쪽 padding 제거, 수정층의 중복 선형 연산 제거, entry 및 evaluator의 selected-position head다. Direct R gradient는 FP64 속도와 수치 검증을 통과할 때만 채택한다.**

본 문서는 설계 및 CPU 입력 크기·대수 검토다. 생산 코드 변경, GPU benchmark, job 변경, 새 Slurm 제출은 수행하지 않았다. 실행 중인 56684의 소스와 contract를 교체하지 않는다. 새 locality reference 목적, solver 가속법, BF16/TF32 전환은 범위 밖이다.

## 근거와 보존할 조건

검토 source는 `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-jlz-sequential-bs100x10-20261001-v1`, HEAD `7b4de31d4361caffbbb383fb0cadb8c5258e9cbe`다. 함수별 SHA, 실제 tokenizer의 입력 크기, 제공된 로그 중 B1 oracle10/20 두 행은 [evidence.json](evidence.json)에 고정했다. 이 문서의 worktree는 원 실행과 분리되어 있다.

| 구분 | 유지할 의미 |
|---|---|
| 과학 입력 | 첫1000의 순서, BS100×10 누적 W/H, 6 rewrite context와 native KL 입력, 대상 token·subject lookup·context group 가중치 |
| 목적 | NLL의 token/context 평균과 active 요청 합, KL(current‖batch-entry) .0625, norm decay .5, clamp .75, entry NLL<.05 정책 |
| 변수와 write | L4–L8 공동 R, FP32 W/activation/R, FP64 adj, batch-entry key/teacher/adj 고정 |
| materialization | `W_entry + (R.double() @ adj.T).float()`의 곱→cast→FP32 add 순서 |
| solver | prox·BB·line search·plateau·tol·120 oracle cap·최종 재평가 및 현재 후보 채택 규칙 |
| state | write 뒤 post-key H append, 원래 transaction/rollback/ledger, 새 checkpoint 없음 |
| 평가 | 동일 endpoint와 R/P/N·TF 항목, row 순서·identity·분모·active/superseded 규약, optimizer feedback 없음 |

같은 수학식과 같은 부동소수점 궤적은 다르다. padding 크기·GEMM shape·gradient 결합 순서를 바꾸면 rounding이 바뀔 수 있다. 따라서 “수학적으로 같다”만으로 경로를 승격하지 않는다. 기존 1e-3 기술 허용치를 느슨하게 하지 않고, 아래의 추가 판정 검사를 통과하지 못한 최적화는 채택하지 않는다.

## 현재 비용과 수정된 시간 기준

제공된 `gpu-56684.out`의 B1 선택 경로 기록만 읽었다. 전체 로그를 polling하거나 후속 batch 상태를 조사하지 않았다.

| 기록 | 누적 선택 경로 시간 |
|---|---:|
| 전체 oracle 번호10 | 375.199569초 |
| 전체 oracle 번호20 | 844.619627초 |
| 10→20 사이 선택 경로10회 | 469.420057초 |
| 구간 평균 | **46.942006초/oracle** |

120회/batch이면 최적화93.884분, 같은 속도로1200회이면 **15.647시간**이다. 초기 번호에는 preflight가 섞여 있지만 이 구간 차이는 선택 경로의 연속10회다. 이것은 B1 초반 속도의 상수 외삽이고 entry/geometry/evaluation/IO 및 이후 batch의 길이·성능 차이는 제외한다.

`actual_preflight()`의 `elapsed/3*1200`은 full/suffix/shared-selected 세 경로 평균이라 선택 경로 ETA가 아니다. 이 필드는 앞으로 `comparison_total_seconds`로 보고하고 생산 ETA 계산에서 제외한다. 과거35~37시간 추정은 이 한계를 명시해 대체한다. H200 baseline 시간과 A6000 JLZ 시간을 같은 하드웨어의 배율로 해석하지 않는다.

이미 적용된 L0–L3 prefix, 후보당 W_eff 공유, ours의 selected-position full-vocabulary head, 같은 endpoint 관측 재사용은 새로운 개선율에 다시 넣지 않는다.

## 계산 그래프와 함수 변경

### 현재 경로

```text
고정 R
 └─ FP64 R·adjᵀ → FP32 cast → FP32 entry addition → 5 W_eff
      └─ 각 microbatch의 L4–L31 suffix
           ├─ 원 down_proj(X) 계산
           └─ post-forward hook에서 F.linear(X, W_eff) 재계산
                └─ 선택 위치 full-vocabulary head → NLL/KL
                     └─ 5 dense dW32 → chunk 순서대로 FP32 누적
                          └─ 누적 dW.double() @ adj → FP32 dR
```

### 단계별 변경 목록

기존 reference 구현은 봉인하고 새 task-local `jlz_efficiency/` namespace에서 먼저 비교한다. 아래 경로는 source root 기준이며 함수명은 실제 구현과 새 제안 함수를 구분한다.

| 단계 | 실제 변경 대상 | 제안 변경 | 보존 및 fallback |
|---|---|---|---|
| E0 | `jlz_sequential/oracle.py:Oracle.__call__, actual_preflight`, `run.py:batch` | stage timer와 padded/valid token ledger, reference와 candidate route 분리 | 계산식 변경0. 선택 경로 timing만 ETA에 사용 |
| E1 | `jlz_pilot/run.py:functional_weights` 호출부 | task-local `effective_linear_context`로 down_proj.forward를 scoped override | 같은 F.linear를 한 번 수행. entry module/Parameter identity·bias·hook 검증, finally 복원 |
| E2 | `jlz_pilot/run.py:chunks, token_subset, prefix_cache`; `jlz_sequential/oracle.py:selected_loss` | `ChunkSpec`을 만들어 현재 row 순서에서 오른쪽 pad만 제거 | 원 tokenization/spec hash 유지. prefix는 crop된 tokens로 재생성; 선택 target 좌표를 함께 반영 |
| E3 | `jlz_pilot/run.py:capture_entry, capture_keys`; `jlz_sequential/oracle.py:committed_losses` override | entry/commit에서 필요한 위치 head만 계산; key는 마지막 L8 key 캡처 후 종료 | teacher/anchor/key 동등성 검사. 특히 L4 key exact gate 유지 |
| E4 | `alphaedit_strength_neutral_barrier/evaluator.py:evaluate_pairs`의 task-local adapter | backbone 전체 실행→target 위치 gather→full vocabulary head | left padding과 모델 원래 position route 유지. 미지원 runtime은 원본 경로 |
| E5 | `jlz_sequential/run.py:batch`, `Oracle.__init__` | 검증된 MB2/4/8 중 시간·메모리로 선택, 첫 science 전 동결 | 목적 정규화 불변. grouping에 따른 합산 순서 회귀 검사 |
| E6 | `jlz_sequential/oracle.py:Oracle.__call__`의 별도 route | FP64 proxy R leaf와 custom linear VJP | forward materialization 불변. 수치·finite·GPU성능 모두 통과해야 채택 |
| B1 | `single_layer_mechanism_first/z_hook.py` adapter 및 `memit_hj/writer.py:Adapter.z/run` | native singleton prefix/head 재사용 후 독립 요청 batching | 원본 native 최초 teacher·Adam 초기 상태, 요청별25평가/24update 및 stop 유지 |

`jlz_sequential/policy.py:admit`의 과학 규칙을 바꾸지 않는다. 수치 검증·route 선택은 wrapper의 책임이다. solver 내부 branch audit가 필요하면 원 `jlz_pilot/solver.py`를 바꾸지 않고 계측 사본을 사용한다. 동적 reference 재확인은 후술한 shared BudgetAccountant를 이 계측 사본에 연결한 뒤에만 허용한다. 첫 구현에서는 qualification된 단일 route를 고정하고, 재확인을 자주 요구하는 route를 제외하는 것이 기본이다.

## Padding과 microbatch

원 spec은 700행 전체를 최대 길이에 맞춘다. 먼저 token IDs를 다시 만들지 않고 기존 행 순서를 유지한 채 chunk별 유효 길이 최댓값까지 잘라낸다. 0/1 mask가 유효 token 뒤 오른쪽0인 것을 확인하고, 제거 구간의 target이 전부 -100이며 lookup·target 위치가 남는지 검사한다.

`ChunkSpec`에는 global row IDs, crop width, cropped IDs/mask, 원래 target/lookup 좌표, 출력 scatter index를 둔다. 전역 `spec.identity`는 그대로 보존한다. global target mask로 짧아진 hidden을 바로 indexing하지 않는다. crop된 tokens를 기존 prefix capture에 넣어 causal mask, cache_position, position_embeddings를 원 runtime이 일관되게 생성하도록 한다. 이미 생성된 cache의 tensor들을 임의의 공통 slice로 자르는 구현은 피한다.

실제 첫1000 입력으로 CPU에서 계산한 크기는 다음과 같다. 10 batch의 각 entry에서 한 oracle가 처리할 크기를 합산했다. tokenization만 수행했으며 모델·GPU는 로드하지 않았다.

| layout | padding 포함 token 수 | 기존 대비 선형 위치 수 | 기존 대비 attention L² 크기 |
|---|---:|---:|---:|
| 현재 global padding | 179,900 | 100% | 100% |
| 원 순서 MB2와 chunk crop | 121,044 | **67.28%** | **47.13%** |
| 원 순서 MB4와 chunk crop | 131,460 | 73.07% | 53.64% |
| 원 순서 MB8와 chunk crop | 136,136 | 75.67% | 57.43% |
| stable length MB2 | 108,168 | 60.13% | 39.87% |
| stable length MB8 | 108,788 | 60.47% | 40.28% |

B1만 보면22,400→11,942 tokens로46.69% 감소한다. 전체10batch의 감소는32.72%다. attention L² 수는 attention score 연산·저장 크기의 proxy이며, 전체 모델 FLOPs/시간 감소율은 아니다.

길이별 grouping은 두 번째 단계다. row별 loss·metric은 원 순서로 복원할 수 있지만 gradient reduction tree는 바뀐다. 자동으로 “동일”이라고 하지 않는다. 같은 순서 MB2를 첫 후보로 검증하고 MB4/8 또는 stable-length는 별도 route로 수치·시간을 비교한다. MB 증가는 padding을 늘려도 GPU 활용률 때문에 빨라질 수 있으므로 FLOPs 절감과 구분한다.

## 중복 선형 연산과 key 종료

`functional_weights()`의 post-forward hook는 원 down_proj 계산이 끝난 뒤 다시 F.linear를 수행한다. 새 context manager는 선택한 다섯 모듈의 원 forward binding을 저장한 뒤 candidate W_eff를 사용하는 forward로 교체한다. Parameter 데이터나 module 구조를 바꾸지 않는다. pre-hook의 입력 관측과 post-hook의 최종 출력 관측은 유지한다.

L4–L8에서 forward 예외, loss 예외, backward 예외를 각각 주입하여 원 binding과 hook가 복원되는지 검사한다. candidate scope 밖에서는 원 model이 즉시 복구되어야 한다. forward를 low-rank 두 항의 합으로 바꾸는 방법은 cast/add rounding이 달라 이번 경로에서 제외한다.

Key는 L4–L8의 down_proj 입력만 필요하다. 마지막 L8 입력 pre-hook가 key를 복사한 뒤 고유한 종료 예외를 발생시켜 남은 L8 down_proj와 L9–L31 실행을 생략한다. 원 context group 평균(.5+.1×5), 요청 순서, subject 위치는 그대로다. 필요한 key가 모두 수집됐는지 검사하고 finally에서 hook를 제거한다. 일반 exception을 성공적인 조기 종료로 삼지 않는다.

Key는 목표 token prefix가 없는 별도 입력이다. rewrite/KL cache를 key cache로 무조건 재사용하지 않는다. write 후 post-key는 바뀐 모델에서 새로 캡처한다.

추가 IO 후보는 불변 C0의 host cache다. 파일 SHA/count를 확인하고 FP32 raw/count를 최초1회 계산해 CPU에 보관하면 다음 batch의 npz 읽기·압축 해제·정규화를 생략할 수 있다. 약4GiB host 비용을 계측하고 필요할 때만 FP64 GPU로 올린다. H와 K는 바뀌므로 A/adj factorization을 다른 batch에 그대로 재사용하지 않는다. 이 cache 절감은 backbone FLOPs 절감과 분리한다.

## 출력 head와 불필요한 동기화

Ours 선택 경로는 이미 위치별 full-vocabulary head를 쓴다. 추가 대상은 `capture_entry`, inherited `committed_losses`, evaluator다. 모든 token은 backbone을 통과하며 head만 필요한 위치로 제한한다. 입력 sequence 자체를 target 위치만 남겨 줄이지 않는다.

Entry에서는 canonical anchor의 L4–L8 block output, rewrite prediction hidden, KL subject hidden을 함께 얻는다. teacher는 batch entry에서 한 번 고정한다. head GEMM shape 변경으로 teacher가 달라지면 원 head 경로로 teacher만 캡처하는 보수적 경로를 쓴다. final norm이 이미 적용된 hidden에는 norm을 다시 적용하지 않는다.

Evaluator는 **left padding**이다. 현재 모델의 기본 position 계산과 padding 폭을 그대로 두고 head만 선택한다. 오른쪽 crop 방식을 일괄 적용하거나 cumsum position_ids를 추가하지 않는다. 원 target-token prediction 좌표, full-vocabulary argmax/log-softmax, .float() 위치와 row 순서를 유지한다. pair/TF/strict 및 near-tie 판정은 아래 gate로 검증한다. 경계 fallback은 원 microbatch의 행 구성·padding 폭·입력 shape를 그대로 복원한 full-head 경로를 사용한다. 경계 row만 singleton으로 다시 호출하면 left-padding position 의미가 달라질 수 있으므로 금지한다. true/new 비교는 양쪽 각각의 원 그룹에서 얻은 reference 값을 함께 사용한다.

선택 경로의 chunk마다 `float(loss)`, `bool(isfinite(...))`가 GPU를 동기화한다. 선택적 후속 개선은 loss scalar를 FP64 device scalar로 **원 chunk 순서대로** 더하고 finite 상태를 sticky flag로 누적한 뒤 oracle 끝에서 한 번 회수하는 것이다. FP32 loss를 FP64로 바꾸는 위치는 원 Python float 변환과 같게 한다. balanced sum으로 바꾸지 않는다. 어떤 chunk의 nonfinite도 마지막 finite 값으로 덮지 않는다. 이 개선은 먼저 profiler에서 sync 비용을 확인한 뒤 독립 비교한다.

## 공동 변수에 대한 직접 역전파

이 절에서 Q는 `adj`이며, covariance system 행렬 A와 구분한다. 한 층에서 X는 T×14336, G는 T×4096, Q는14336×B, R는4096×B다.

현재 graph는 chunk별 FP32 `G.T @ X`를 FP32로 누적하고, 후보 마지막에 FP64로 Q를 곱한다. 제안 graph는 FP64 `Z=X.double() @ Q`와 `G.double().T @ Z`로 R gradient를 직접 만든다.

```python
# candidate마다 기존 effective(x)를 그대로 호출한다.
W_eff = W_entry + (R.double() @ Q.T).float()

# task-local custom autograd, R64는 같은 x에서 만든 독립 leaf
def forward(X32, R64_proxy, W_eff32, Q64, frozen_bias):
    Y = F.linear(X32, W_eff32, frozen_bias)
    Z64 = X32.reshape(-1, I).double() @ Q64
    save(W_eff32, Z64, X32.shape, X32.requires_grad)
    return Y

def backward(G32):
    dX32 = G32 @ W_eff32 if needs_input_grad else None
    dR64 = G32.reshape(-1, O).double().T @ Z64
    return dX32, dR64, None, None, None
```

R64 proxy gradient를 chunk 순서대로 FP64 누적한 뒤 전체 oracle 반환 시 한 번만 FP32로 바꾼다. 입력 dX는 기존 FP32 선형 backward 의미를 유지해야 아래층 R까지 cross effect가 전달된다. L4 입력처럼 실제로 gradient가 불필요할 때만 생략한다. 이 custom VJP는 candidate R와 W_eff identity를 assert하고 first-order oracle 전용으로 두며 double backward를 지원한다고 주장하지 않는다. Q·teacher·base W는 미분하지 않는다.

| 요소 수 또는 연산 | 값 및 의미 |
|---|---|
| 다섯 dense dW FP32 | 293,601,280개, 1.09375GiB |
| dense accum와 동시 반환 gs | 두 묶음 원시 크기 약2.1875GiB 제거 가능. 전체 peak 절감량은 미측정 |
| 다섯 R | 2,048,000개, dense의1/143.36 |
| FP64 dR 한 묶음 | 15.625MiB |
| 원 chunk dW | 층당2TOI FP32 FLOPs |
| 원 마지막 pullback | 층당2OIB FP64 FLOPs |
| 제안 direct R | 층당2TB(I+O) FP64 FLOPs |

B100에서 direct contraction의 FLOP 수는 dense dW보다 약31.86배 작지만 FP64다. FP64 실효 throughput이 FP32의1/31.86보다 낮으면 그 부분은 더 느릴 수 있다. 전체 W_eff 생성, dX, 나머지28층, head 비용도 남는다. 따라서 **143배나31.86배를 전체 속도 개선으로 발표하지 않는다.** Q를 FP32로 내려 연산하는 옵션은 이번 FP64 경로와 분리한 후속 정밀도 실험이다.

이 결합 순서는 부동소수점에서 원본과 같지 않다. CPU 소형 검사에서 FP64 재결합 상대오차7.71e-16, 기존 FP32 dW 누적 대비1.14e-7였지만, 거의 상쇄되는 두 chunk에서는 상대1.48e-2와 절대3.61e-6가 함께 나왔다. 두 층+비선형+residual custom VJP의 forward/loss는 같은 W_eff에서 bitwise 동일했고 층별 gradient 상대오차는4.37e-8/9.21e-8였다. 이는 실제 Llama/GPU 검증이 아니다. 재현 코드는 [vjp_cpu_audit.py](vjp_cpu_audit.py), 원 출력은 [vjp_cpu_results.jsonl](vjp_cpu_results.jsonl)에 보존한다. FP32 projection 비교는 감사용 반례이며 채택 경로가 아니다.

추가로 X=[1e20,-1e20], G=[1e20], Q=[1,1]ᵀ에서 기존 dense dW는 overflow→NaN이지만 direct 결과는0이 될 수 있다. 최종 gR finite만 검사해서 기존 NONFINITE를 숨기지 않는다. 층별 FP64 상한 `U=sum_{chunk,t} max_o|G|*max_i|X|`와 X/G/W/Q/Z/dR 및 실제 계산한 dX의 상태를 기록한다. U가 FP32_MAX/2보다 작아 dense 곱·누적 overflow를 배제할 수 있을 때만 빠른 finite 판정을 사용한다. 불확실하거나 direct 중간값만 nonfinite인 경우 원본 경로에서 같은 후보를 확인하고 그 결과가 최종 판정이다. 검증 wrapper가 이 처리를 끝내기 전에 policy에 NONFINITE를 전달하지 않는다. 원본도 실패하면 기존 any-nonfinite 중단을 적용한다. 추가 계산은 반드시 비용에 포함한다. 동적 재확인 장부가 구현되지 않은 경로는 이 상황을 CERTIFICATION_FAILED로 기록하고 transaction을 중단하며, finite 또는 NONFINITE 과학 판정을 추정하지 않는다.

## Native baseline의 독립 요청 실행

MEMIT-H/HJ native는 같은 batch entry에서 모든 요청의 z를 만든 뒤 writer를 실행한다. 이 구간에서만 요청을 GPU에 묶을 수 있다. BLUE의 layer-local refit처럼 앞 write가 뒤 z의 모델 상태를 바꾸는 경로는 해당 단계 안에서만 묶는다.

`z_hook.capture_prefix/suffix_hidden/native_losses/compute_z_batch`가 이미 존재하지만, native writer는 현재 원본 singleton 경로를 쓴다. 과거 helper의 검증 또는 진단 waiver를 이번 가속에 상속하지 않는다.

첫 native forward는 **요청별 원 singleton 입력 shape와 padding 폭**으로 수행해 원본 full-head teacher, anchor, z층 출력 prefix와 loss0를 한 번 얻는다. 여러 요청을 global padded 첫 호출로 묶어 얻은 teacher를 bitwise 동일하다고 가정하지 않는다. 초기화 호출은 기존25평가 중 첫1회이며 별도 teacher capture 뒤 loss0를 반복하지 않는다. 가능하면 첫 native backward/Adam update도 원본으로 수행해 delta1과 m/v/step을 정확히 넘긴다. 그 뒤 먼저 singleton cached suffix/head를 검증하고, qualified MB2/4/8로 확장한다.

요청별 Adam m/v/step, stop iteration, clamp와 final delta는 독립이다. 활성 요청들의 loss는 **합**으로 묶으며 요청 수로 나누지 않는다. 종료한 row의 momentum을 갱신하거나 packing 때문에 step을 초기화하지 않는다. 최대25 optimizer candidate iteration·24 update, total_loss<.05의 엄격 부등호, 마지막 iteration의 backward 생략을 유지한다. 별도 reference 확인이 있으면 물리적 forward/backward는25/24보다 많을 수 있으므로 추가횟수를 기록하되 Adam step을 추가하지 않는다. 이것은 whole-batch oracle120회로 제한한 JLZ 장부와 구분한다. 데이터별 조기 종료 횟수를 줄여 속도를 얻었다고 해석하지 않는다.

NLL은 max(v_loss_layer,z_layer)의 지정 hidden, KL은 final model hidden이라는 구분을 지킨다. context 생성/RNG는 원래 순서로 한 번 완료한다. 캐시 키는 W entry identity, layer, token IDs/mask/position, context, dtype, runtime과 route를 포함하고 write 뒤 무효화한다. 캐시는 provenance 없이 다른 batch나 모델 상태로 넘기지 않는다.

## 수치 검증과 판정 경계

[validation.json](validation.json)의 값은 **구현 전에 고정할 제안 gate**다. 실제 모델에서 통과했다는 뜻이 아니며, 실패 후 완화하지 않는다.

| 검사 | 제안 기준 |
|---|---|
| 입력·과학 규약 | 원 token IDs/target/lookup/행 identity·정규화·context·case 순서 일치 |
| 같은 R의 W_eff | 다섯 weight bitwise 동일 |
| 같은 후보의 loss | 요청별 NLL/KL abs≤1e-4, 전체 smooth abs≤1e-3 |
| gradient | 전체 및 layer/request block L2 차이/max(1,ref L2)≤1e-5, component maxabs≤1e-5 |
| proximal residual | 같은 reference initial scale에서 normalized 차이≤5e-6 (=0.05×tol) |
| feasibility | 원 clamp slack1e-6와 inactive exact0 유지 |
| 짧은 solver | 같은 후보 고정 검사와 독립 궤적 검사를 분리. accept/reject·stop·최종status·cap accounting 동일 |
| 짧은 solver 반환 | \\|x_new-x_ref\\|/max(1,\\|x_ref\\|)≤1e-4와 원 commit gate 통과; bitwise 궤적 주장은 하지 않음 |
| evaluator | NLL abs≤1e-4, selected logits maxabs≤1e-3/RMS≤1e-4, token/strict/pair success·분모 동일 |
| baseline | 요청별 Adam step·stop iteration·clamp 분기 동일, 고정 delta gradient gate 및 반환 delta 차이≤1e-4 |
| state | history append 수·ledger·rollback·cache invalidation 규약, 비선택 parameter guard 유지 |

소형 fixture는2~4요청, 길이 차이, single/multi-token target, 초기 zero-step, norm 경계, nearzero gradient와 chunk 상쇄, loss layer≠KL layer를 포함한다. R=0, 작은 비영점, radius의0.5, clamp 근접 후보를 같은 W/H에서 비교한다. 모든 결과에 reference route·source SHA를 넣는다.

경계 재확인은 비용·의미를 숨기지 않는다.

- Armijo margin이 loss 오차 band 안이면 같은 후보와 판정에 필요한 과거 history 항목을 reference로 확인한다. 역사값도 다른 rounding이면 candidate 하나만 재평가해 완전한 판정 일치를 주장할 수 없다.
- residual이 tol±5e-6 안이면 reference gradient로 확인한다. 최초 scale과 최종 재평가는 reference를 권장하며 각각 원래120회 안의1회다.
- baseline loss가.05±1e-4 또는 clamp 경계면 reference 확인; evaluator top2 logit margin≤2e-3 또는 true/new NLL margin≤2e-4면 원 full 경로 값으로 판정한다.
- reference 확인은 현재 후보의 판정을 보호하며 과거의 궤적 차이를 없애지 않는다. short solver에서 branch가 달라지면 해당 최적화 조합을 qualification 실패로 처리하고 reference route를 유지한다.
- **현 contract의120은 whole-batch oracle 상한이다. 같은 후보를 재확인하는 전체 oracle도1회로 과금한다.** 논리적 candidate ID, 전체 gradient oracle 횟수, 실제 chunk F/B, reference 재실행을 별도 기록한다. 내부 호출을 wrapper 아래 숨겨 “무료 검사”로 만들지 않는다. 최종1회 예약을 보장할 수 없으면 처음부터 그 후보를 reference로 평가하거나 가속 gate에서 중단한다.
- 현재 solver는 wrapper마다 calls를1 증가시키므로 별도 로그 장부만 더하면 부족하다. 동적 모드의 BudgetAccountant를 계측 solver의 evaluate, while/line-search remaining 검사, final 예약에 공동 연결한다. reference 재평가 전에도 이를 차감한다. 최종1회만 남으면 최종점을 원본 경로로 바로 평가한다. accounting adapter 자체를 cap2/3/120, line-search 재확인, history 재확인, NONFINITE, final 실패 fixture로 검증하고 반환 calls와 실제 oracle 횟수가 정확히 같아야 한다.
- budget 안의 재확인이 많으면 서로 다른 trial 수가 줄 수 있다. 이를 동일 탐색 궤적이라고 부르지 않는다. qualification에서 경계 충돌이 반복되는 route는 생산에 채택하지 않는 것이 기본이다. 논리적120회에 reference 추가 호출을 별도 허용하는 것은 현재 계약과 다른 budget 정의이므로 자동 채택하지 않는다.

## 측정 순서와 적용 조건

1. **CPU 단계:** canonical token/위치 보존, custom VJP의 입력gradient, finite 반례, transaction 예외 검사를 수행한다. 이 단계는 모델 성능 인증이 아니다.
2. **실제 모델 소형 단계:** 같은 W0와2~4요청에서 고정 후보 및 짧은 solver/native25-loop를 paired 비교한다. branch/metric mismatch를 먼저 해결한다.
3. **kernel 단계:** 실제 I/O/B/길이에서 dense와 direct FP64 contraction, 중복선형 제거, head 선택을 CUDA events로 측정한다. FP64가 느리면 E6는 제외한다.
4. **실제 BS100 단계:** 참조 경로와 최종 통합 후보 각각 같은 비영점 후보에서 warmup1회 뒤 측정3회. 여기서 참조는 현재 생산의 shared-selected dense-gradient 경로이며, full/suffix/selected 평균이 아니다. 총8 whole-batch oracle를 별도 기술 benchmark budget으로 기록한다. 기존 실행의 preflight6회에 슬쩍 추가하지 않는다. MB2/4/8 탐색은 소형/kernel 단계에서 후보를 좁히고, 실제 BS100에서 필요하면 추가횟수를 사전에 별도 기록한다.
5. **endpoint 평가 단계:** 같은 W에서 reference와 가속 evaluator를 비교하여 전체 metric·분모를 고정한다. 모델/평가질문을 subset으로 줄여 얻은 시간은 전체 observer 속도로 주장하지 않는다.
6. **생산 채택:** source/config/route/MB를 새 실행 전에 봉인한다. 현재 56684에는 적용하지 않는다. 진행 중인 무checkpoint chain을 중간에 이식하거나 재시작하지 않는다.

측정3회는 throughput calibration이며 통계적으로 안정된 전체 run 배율이 아니다. median/min/max, CUDA peak allocated/reserved, process RSS, valid/padded tokens, attention L² proxy, forward/backward/head/mapping/kernel 수를 남긴다. CUDA events의 component time과 synchronized wall time을 구분하고 profiling 자체 오버헤드는 별도다.

가속 route의 채택 조건은 수치·분기·state gate 전체 PASS, 같은 장치에서 반복 측정 median이 reference보다 빠르고 범위가 크게 겹치지 않는 것이다. direct VJP가 메모리만 줄이면 MEMORY_ONLY로 기록하고 그것 때문에 MB를 키웠을 때의 통합 효과를 다시 측정한다. 실패한 최적화는 제외하고 통과한 조합만 사용한다.

## 실측 기반 시간 예측 방식

새 route의 GPU 시간은 아직 없다. 현재 유일한 실제 선택 경로 기준은46.942초/oracle다. 다음 식으로 stage별로 갱신한다.

```text
T_run =
  T_load + T_W0_observer + T_technical
  + Σ_batch [T_entry + T_geometry
             + N_oracle × median(selected_route_oracle_seconds)
             + T_commit_history + T_observer + T_IO]
```

각 batch의 padding 길이·target 길이 및 reference fallback 비율을 함께 기록한다. 한 B1 측정을 다른9batch에 그대로 적용한 값은 상수속도 가정으로 표시한다. 조기 종료에 따른 N_oracle 감소는 구현 가속과 분리한다. full/suffix/reference 비교 시간, failed route 비용, warmup을 버리지 않고 기술 총비용에 포함한다.

조건부 예로 **실측된 통합 speedup이 나중에2배면** 최적화 부분15.65→7.82시간, **3배면**5.22시간이다. 이는 목표 시나리오이며 이번에 확인한 가속률이 아니다. 전체 ETA에는 나머지 stage가 더해진다. CPU에서 확인한32.7% token 감소를 전체 시간32.7% 감소로 치환하지 않는다.

## 구현 인계 산출물

이 설계 패키지는 `design-ko.md`, `validation.json`, `evidence.json`, `inspect_costs.py`, CPU VJP 감사 스크립트와 결과로 구성한다. 실제 구현 시에는 다음을 추가해야 한다.

- 함수별 변경 diff와 frozen reference route, source/config/input hash.
- same-candidate loss/gradient/prox·짧은 trajectory·baseline stop·observer pair 비교표.
- layout/route/MB별 stage 시간·메모리·연산 크기·논리 및 실제 호출 ledger.
- benchmark3회 결과와 stage별 갱신 ETA, 채택/제외 route 및 이유.
- scientific chain에 사용한 route를 밝힌 새 실행 계약. 기존 결과를 가속 경로 결과로 바꾸지 않는다.
