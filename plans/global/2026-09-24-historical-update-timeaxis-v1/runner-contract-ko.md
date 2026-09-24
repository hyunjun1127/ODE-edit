**실행 adapter·수치 계약 — historical update timeaxis v1**

이 문서는 구현자가 남은 선택을 임의로 바꾸지 않고 runner를 만들 수 있도록 고정한 명세다. 현재 실행 가능한 GPU runner가 있다는 의미는 아니다. `build_plan.py`는 stdlib만 사용해 셀·metadata를 만드는 설계 도구이고, 모델을 import하거나 forward하지 않는다.

**1. 입력 결속 및 기존 코드 재사용**

`asset-bindings.json`에 두 실행의 runtime·source·config·model revision·W0 선택 weight hash를 담았다. helper commit은 `1075540b45c29269e690ac63aae44758d8d63174`, native editor source는 `311b076a92e4ed0f14f5c8b4909732da781bc5f7`이다. 이번 추론에서 native editor 자체를 실행하지 않는다.

`source-evidence/`의 evaluator·contracts·evaluation·integrity 네 파일은 helper commit에서 추출하고 실제 runtime lock의 SHA256과 일치함을 확인했다. 이 코드의 수학과 tokenization을 그대로 사용하되, 무관한 package `__init__`가 writer를 import하지 않도록 별도 observation adapter를 둔다.

native token 계약은 다음과 같다.

- prompt는 `add_special_tokens=True`, target은 선행 공백을 정규화한 뒤 `add_special_tokens=False`로 따로 encode한다. target 선두 BOS/UNK 제거도 원본과 같다.
- prompt IDs와 target IDs를 직접 이어 붙인다. 전체 문자열을 다시 tokenize하거나 chat template을 추가하지 않는다.
- tokenizer 설정의 padding_side는 right지만 **실제 evaluator가 tensor를 수동으로 left-pad**한다. 이 구현상의 차이를 임의 수정하지 않는다.
- category별 new/true를 별도 score하며 microbatch=16, `use_cache=False`, eager attention, eval mode, FP32, autocast off, TF32 matmul off를 기본으로 한다.
- 기준 환경은 torch 2.9.1+cu128, transformers 4.44.2다. 다른 GPU나 의존성으로 옮기면 그 차이를 기록하고 T1 fidelity를 통과해야 한다.
- 모든 target token을 평균한 NLL과 각 token argmax를 저장한다. EOS·길이 정규화·position IDs·truncation을 조용히 바꾸지 않는다. 길이가 모델 한도를 넘으면 결측 사유로 남기고 자르지 않는다.

T0에서 실제 token 배열, evaluator kernel hash, prompt inventory hash, padding/order policy를 봉인한다. 기존 `seen-full.json`의 identity와 같은 방식으로 digest를 계산한다. 모델 snapshot은 weight index·shard의 실제 경로/크기/hash를 실행 환경에서 확인한다. 설계 단계에서는 수십 GB shard 전체를 새로 hash하지 않았다.

**2. 모델 재구성과 U materialization**

full W0는 한 worker에서 한 번 load한다. 수정된 parameter는 checkpoint의 저장 weight로 `copy_`하여 endpoint를 만든다. AlphaEdit history, projector, covariance, writer context, z cache는 새로운 update를 생성하지 않는 이 추론에 필요하지 않다. checkpoint archive에 동봉돼 있어도 GPU에 올리지 않는다.

각 U의 다섯 tensor는 `float64(W_b)−float64(W_a)`로 얻는다. 이는 저장된 FP32 parameter 상태 사이의 유한 변화량이며, native solver를 FP64로 다시 돌린다는 뜻이 아니다. counterfactual은 다음처럼 만들고 마지막에 한 번 FP32로 cast한다.

```
single removal: W_cf = float32(float64(W_t) − (float64(W_b)−float64(W_a)))
pair removal:   W_cf = float32(float64(W_t) − U64 − V64)
```

긴 구간을 수십 개의 반올림된 delta 합으로 재조립하지 않는다. nonselected parameter는 W0 그대로다. 제거 전의 실제 W_t tensor를 CPU에서 보관하고, 매번 그 snapshot으로 복원한다. `W.sub_(U); W.add_(U)`는 복원 방식으로 사용하지 않는다. 예외 발생 시에도 `finally`에서 복원하고 hash를 확인한다.

대각선 b에서 U를 제거하면 θ_a가 되어야 한다. 이 대조는 반드시 구현된 arithmetic으로 한 번 만들고 저장 θ_a와 비교한다. 단순히 θ_a를 불러오는 shortcut만 시험하면 subtraction 경로를 검증하지 못한다. 통과 뒤 production score는 기존 θ_a를 재사용할 수 있다.

`state-bank.csv`의 retained mask는 대수적 상태 식별자다. 서로 다른 계산 순서에서 float rounding이 완전히 같다는 보장은 아니다. cache의 실제 key는 **materialized weight hash + prompt/target token hash + evaluator signature**로 한다. 같은 mask라는 이유만으로 GPU score를 자동 재사용하지 않는다.

**3. T1과 셀별 검증의 구체적 기준**

검증 문항은 각 구간에서 hash 순위 16개씩 고른다(모든 결과를 보기 전에 T0에서 저장). 최소 W0와 두 arm의 t=1/20/50/100을 T1에서 재구성한다. 나머지 endpoint는 T2에서 최초 사용 직전에 같은 검증을 수행한다. 각 endpoint의 과거 저장 score와 비교할 문항은 그 endpoint까지 편집된 요청 중에서 선택한다. 파일의 기존 identity와 정확히 join한다.

| 검증 | 기준 |
|---|---|
| checkpoint 파일·선택 tensor | recorded SHA256 및 shape/dtype와 일치, nonfinite=0 |
| 실제 endpoint 재구성 | 원본 runtime의 해당 선택 weight hash와 정확히 일치 |
| 과거 평가와 fidelity | 각 답변 NLL 최대 절대 차이 ≤2.5e−4; margin 최대 차이 ≤5e−4 |
| 동일 상태 반복·restore 후 출력 | 위와 같은 오차 한도 |
| microbatch 1/16 packing 점검 | 선택 문항에서 동일 오차 한도; 초과하면 설정 차이를 해결 |
| 대각선 subtraction | 저장 θ_a의 weight 및 score와 일치; materialization 오차 별도 기록 |
| 최종 복원 | 선택 weight byte hash가 원래 endpoint와 정확히 일치 |
| score 분해 bookkeeping | float64 scalar 계산에서 `abs(ΔM−ΔB−ΔC)≤1e−10` |
| raw 행 | finite score, 중복 key=0, 기대 prompt inventory 누락=0 또는 명시적 실패 |

T1 오차 기준은 실행 전에 정한 값이다. 실패하면 효과를 보고 허용 오차를 올리지 않는다. 같은 환경·tokenization·batching을 재현하거나 원인을 기록한 새 numerical contract로 별도 version을 만든다. 역사적 fidelity를 확보하지 못한 자료를 원본 BASE와 동일한 평가라고 합치지 않는다.

관측 최대 margin 오차를 e_m이라고 기록한다. C 오차는 최대 2e_m, ΔC 오차는 최대 4e_m으로 보수적으로 표시한다. 위 기준이면 ΔC 오차 상한은 0.002이고 가장 작은 과학적 ε=0.025보다 충분히 작다. 실제 반복 측정은 모든 counterfactual의 엄밀한 오차 증명은 아니므로, **각 단일·이중 제거 상태마다 고정 sentinel 8개를 재평가**한다. sentinel은 그 상태의 score panel에서 hash로 선정한다.

기존 bit와 다른 경우 margin이 ±5e−4 밖인데 뒤집혔으면 실패다. 그 안의 boundary flip은 별도 표로 남기며 원래 strict `>0` 지표와 boundary 제외 민감도를 둘 다 낸다. 유의하지 않음을 안정으로 판정하는 절차는 없다.

**4. 상태별 score 작업과 정확한 cache 재사용**

`main-cells.csv`는 연구 비교 단위다. 실제 추론 스케줄은 `score-tasks.csv`의 state×cohort×phase 요청을 따라 수행한다. 이 파일에는 미래 구간의 기준 B_b를 계산하기 위해 **구간 이전 모델 θ_a에서 아직 편집되지 않은 그 cohort 문항을 평가하는 작업**도 들어 있다. 과거 `seen-full`에는 이 문항이 없을 수 있으므로 빠뜨리지 않는다.

actual M_t는 해당 endpoint의 all-seen 패널을 한 번 평가하여 분배할 수 있다. counterfactual θ_t−U는 우선 해당 U의 패널만 평가한다. pair θ_t−V에서 과거 U의 문항은 기존에 θ_t−V를 V 자신의 문항에서 평가한 것과 다르므로 새 score가 필요하다.

cache row의 키는 `(state_weight_hash, case_id, panel, prompt_index, target_version, prompt_token_hash, target_token_hash, evaluator_signature)`다. receipt에는 입력 순서와 실제 microbatch layout hash도 남긴다. packing 차이가 있을 때는 T1 수치 계약을 만족한 경우에만 동일 score로 취급한다. pilot와 full이 같은 key면 재사용하며 다른 key면 다시 계산한다. 중복 측정을 독립 관측으로 세지 않는다.

**5. 제안하는 worker 구조와 resource 상한**

GPU worker는 arm별 하나씩, 동시에 최대 두 개를 기본으로 한다. 한 worker가 W0를 load한 채 필요한 endpoint들을 순서대로 처리하고 상태/문항 단위 결과를 commit한다. worker의 재시작은 완료된 receipt/hash를 확인하여 미완료 row만 이어간다. 156셀을 각각 full-model load하는 156개 job으로 만들지 않는다.

시작 resource는 GPU 1장(80GB 이상 권장; 원래 GPU는 RTX PRO 6000 Blackwell), CPU 8개, host RAM 96GB/worker다. 정확한 시간은 아직 측정하지 않았다. microbatch=16에서 메모리 문제가 있으면 자동 dtype·kernel 변경 대신 typed failure를 내고 검증 가능한 작은 microbatch로 별도 numeric signature를 만든다.

선택 weight 한 벌은 약 1.094 GiB, FP64 구간 delta 한 벌은 약 2.188 GiB다. 모든 raw checkpoint 합은 약 72.19 GiB이며 AlphaEdit history가 큰 비중을 차지한다. main/pair counterfactual을 full model 파일로 저장하지 않는다. endpoint·delta를 제한된 CPU cache로 관리하고 GPU에는 현재 상태만 올린다. delta 전체를 동시에 host RAM에 물릴 필요도 없다.

pilot에서 다음 항목을 기록한다: 실제/제거 forward 시간, nonpadding/padded token 수, model load 시간, checkpoint read/materialization 시간, GPU peak memory, hash·복원 시간. 전체 시간 예측은 `측정 초/padded-token × 예정 padded-token + load/I/O/restore`로 계산하고 범위를 제시한다. 이전 editing job의 10–12시간을 이 추론의 예측치로 쓰지 않는다.

**6. 구현할 interface와 저장 파일**

GPU runner의 다음 interface는 **구현 요구사항**이다. 존재하는 CLI로 오해하지 않도록 현재 명령어 실행 예시는 제공하지 않는다.

```
bind_inputs(contract, checkpoint_path_map) -> runtime_binding, token_manifest
verify_endpoint(family, checkpoint_t, sentinel_panel) -> fidelity_receipt
materialize_state(state_recipe) -> weight_hash, restore_transaction
score_panel(state_weight_hash, panel_rows) -> score_rows, execution_receipt
join_cell(M_t, B_t, M_b, B_b, fact_metadata) -> contribution_rows
analyze_trajectories(rows, analysis_contract) -> tables
analyze_pair_interventions(pair_rows) -> D_M, D_B, I
```

출력은 `runtime-binding.json`, `token-manifest.jsonl`, `state-receipts.jsonl`, `scores.jsonl`(또는 동일 schema의 parquet), `contributions.jsonl`, `pairs.jsonl`, `missingness.csv`, `cost.json`, 그림/표다. score와 contribution을 분리해 재계산·재검토가 가능하게 한다. nullable score는 invalid 상태와 missing_reason을 반드시 동반하고 0으로 채우지 않는다.

파일은 작업용 임시 파일에 작성 후 성공 receipt와 함께 완료 이름으로 바꾼다. 중단된 tensor 상태에서 다음 셀을 계속 계산하지 않는다. 기존 checkpoint/raw 결과에 덮어쓰지 않으며 output root는 새 실행 ID 아래 둔다. 실행 metadata와 실험 결과의 scientific interpretation을 분리한다.
