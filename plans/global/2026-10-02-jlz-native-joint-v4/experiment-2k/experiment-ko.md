# JLZ native joint v4: 2,000-edit 실험 실행 설계

2026-10-02 KST. 사용자 요청: “실험 진행하도록 실험 설계 진행하라. 2000edit을 기준으로 일단 해보자.”

최신 사용자 범위 수정: **“v4 두 arm만 실험 돌리는 걸로 해”**. Baseline의 신규 main/pilot/fallback 실행은 모두 제외한다.

상태: **설계와 CPU 입력/일정 검증 완료 대상; 생산 구현·GPU pilot·제출은 미완료**. 이 문서는 `../contract.json`의 방법을 실험으로 구체화한다. 기존 two-arm v2/SPG 및 shared-subject v3를 실행하는 명령으로 대체하지 않는다. 20 step은 Adam step이 아니라 **BS100의 순차 편집 20회**다.

이 문서는 **실험 설계문**이다. 일반 method와 연산 최적화의 정본은 [method-ko.md](../method-ko.md), model/benchmark/가변 batch interface는 [portability-contract.json](../portability-contract.json), GH 전달 진입점은 [GH-HANDOFF.md](../GH-HANDOFF.md)다. 이번100/20/2k/L4–L8/6+1 설정을 공통 runner에 하드코딩하지 않는다. 다른 batch/model/benchmark는 새 profile로 연결할 수 있게 구현하되 이번 GPU 실행 범위는 기존 A/B 두 arm·동일2k로 유지한다.

## 1. 확인할 질문과 비교 행렬

첫 목적은 2k에서 signal을 확보하는 것이다. 한 개의 고정 stream/order/seed를 사용하며 반복 seed·순서 sweep·새 논문 baseline 전체 재현은 이번 범위에 넣지 않는다.

| 행 | 실행 | 답할 질문 |
|---|---|---|
| JLZ-A, eta=0 | 신규 W0부터 2k | native 공동 local-z가 실제 weight 편집에서도 strength를 확보하는가 |
| JLZ-B, eta=1 | 신규 W0부터 2k | 추가 보존·실현 비용이 A 대비 PS/NS·retention 균형을 개선하는가 |

**GPU 실험은 A/B 두 chain뿐이다.** 각 chain은 동일한 2,000 requests를 처리하며 서로 독립이다. 입력의 고유 요청 수는 2k, 두 방법의 처리 occurrence 합계는 4k다. MEMIT-H, HJ, AlphaEdit, MEMIT의 신규 main/pilot/fallback은 실행하지 않는다. 기존 결과를 읽기 전용으로 확인하여 historical 품질 참고를 추가할 수는 있지만, 자료가 없거나 조건이 맞지 않아도 A/B 실행이나 완료를 막지 않는다. Runtime이 다른 historical 결과에서 속도 배율을 계산하지 않는다. HJ의 BS10 continuation 또는 W10을 BS100 W20으로 바꾸어 부르지 않는다. 기존 ours v2/v3의 성적·W/H는 v4 결과로 재사용하지 않는다.

통제된 단일 요인 비교는 A 대 B다. Historical HJ와의 참고 비교에는 target 위치, 최적화 변수, solver, 배분 방식, 실행 조건의 차이가 있다. 이번 실험 하나로 misalignment가 locality 손실의 유일한 원인이라는 인과 결론을 내리지 않는다.

## 2. 고정 입력·방법

- 데이터: 기존 fixed10k의 ordinal `[0:2000]`, 원 순서, BS100×20, drop/dedup/shuffle 없음. `case-schedule.csv`가 case ID와 batch 경계를 고정한다.
- 모델: 기존 Meta-Llama-3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`. 실제 SH4 shard/tokenizer/config SHA는 제출 lock에 결속한다.
- 학습: baseline과 같은 rewrite6 + KL1, native target/leading-space/BOS/UNK 처리 및 subject-last. NLL은 token mean/context mean/request sum; key는 .5 + .1×5.
- 모든 편집 후보층 L4–L8 유지. 최소 활성층·균등화·top-k·사전 층 제외 없음. 한 층 전담과 0 변위를 허용한다.
- A/B: native KL(current||own batch entry), lambda_KL=.0625, native delta norm=.5, clamp=.75, 직접 delta FP32 joint Adam lr=.1, betas=.9/.999, eps=1e-8, optimizer weight_decay=0.
- A/B: 각 batch candidate loss 평가25, 업데이트24, 마지막 후보 평가 후 추가 update 없음. norm/V는 microbatch 수와 관계없이 후보당 한 번. 초기 delta=0에서도 gradient 경로를 유지한다.
- A/B: 최적화 hook 제거 후 current-key L4→L8 writer, residual=동일 층의 학습 delta. L8 residual 재분배나 inverse-Q 증폭 없음.
- A/B: 각자 H0=0, C0 scale=15000, history refresh 없음, 모든 층 write 후 post-all-layer pooled keys를 occurrence별 한 번 append. C0/H의 원 native 저장 및 append dtype FP32, geometry solve/cost FP64. H는 arm끼리 공유하지 않는다.
- FP32 모델, eval mode, no autocast, matmul/cuDNN TF32 off. 같은 런타임·attention implementation·tokenizer를 두 신규 chain에 사용한다. 실제 패키지 버전과 source closure를 lock한다.
- AlphaEdit 재사용은 native projector/L2=10/threshold=.02/blue=false 및 native solve dtype을 보존한다. MEMIT-H geometry로 바꾸지 않는다.
- 별도 G/E/replay 문장 loss 없음. 역사 key H는 유지한다. 공식 P/N은 평가 전용이며 eta·후보·예산 선택에 사용하지 않는다.

기존 baseline과 비교할 때 공동 Adam의 고정 예산과 native 요청별 early-stop의 차이를 명시한다. 기존 baseline을 이번 예산으로 재실행하거나 수정하지 않는다. `seed=20261002`; deterministic/TF32/attention 설정과 실제 비결정성 여부를 기록한다.

## 3. 짧은 pilot → 본 실험

### P0: CPU 준비

현재 method artifact의 SHA, source input hash, 첫2k ordinal·case/target·배치 경계·평가 분모를 확인한다. 기존 CPU 수식 검산을 재사용한다. SH4에서는 실제 tokenizer로 target IDs, teacher-forced input IDs, mask/position/lookup 위치를 baseline과 대조한다. 문자열 확인을 token parity로 보고하지 않는다.

추가 compute-r1 근거는 `../compute/native-work-counts.json`과 `../math/compute-reuse-check.json`이다. 단순 B1/B<d_in/B=d_in/B>d_in geometry와 마지막 partial batch를 CPU에서 확인한다. 다른 실제 모델을 GPU로 추가 실행하는 요구가 아니다. Model adapter의 native site→write 공간, readout, parameter alias와 causal capability를 점검하고 profile에 결속한다.

### P1: 작은 실제 모델 정합 pilot

본실험 밖의 fixed stream `[2000:2004]`를 사용한다. 첫2requests를 BS2 한 batch로 A/B 각각 독립 W0/H0에서 **정규 25후보·24update** 실행하고 실제 write·평가까지 수행한다. 이어 다음2requests의 B2 entry를 같은 RAM state에서 준비하여 teacher·key·H 전달과 hook 해제를 확인한다. B2의 추가 최적화/write는 하지 않는다. A/B 합계50 joint 후보다. 작은 B여도 큰 geometry solve가 필요하므로 불필요한 BS2×여러 batch 반복을 피한다.

비교 검증은 다음으로 한정한다.

1. 동일 입력의 zero-delta logits/loss와 native entry, 각 층 단독의 native 목적/gradient.
2. 여러 층 동시 개입 시 모든 delta의 gradient와 microbatch gradient 합산; 추가 비용의 analytic gradient.
3. entry 모델 불변, learn hook 제거, 요청 payload와 writer residual identity, 의도한 FP32 weight와 실제 commit bitwise identity, 다른 parameter 불변.
4. 모든 층 write 후 H 한 번 append, 다음 batch가 이전 commit을 entry로 사용, rollback이 W/H 전체를 복원.
5. 실제 committed 모델 평가 행·분모·target identity, evaluator no-mutation.

Compute-r1에서 활성화할 경로는 같은 후보의 full reference와 비교한다: subject를 포함한 suffix의 모든 δ gradient, batched head·original position/row mapping, zero-candidate 준비 결합의25회 계수, direct/Cholesky P·E/V gradient, current key와 materialized writer, saved prewrite key와 final key/H다. 원문 KL input은 보존하며 미래 위치 계산 생략만 검증한다. Prefix cache를 writer에 그대로 적용하지 않는다. 실제 commit은 선택한 writer가 만든 FP32 tensor와 bitwise 같아야 하나 서로 다른 solve/kernel 경로끼리의 bitwise 동일성을 요구하지 않는다.

실현량 UK가 delta와 같은지, virtual NLL과 actual NLL이 같은지는 PASS 조건이 아니다. 이 차이는 관찰할 과학적 결과다. 성능·집중도·수렴률 기준으로 main을 막지 않는다. 파일 `experiment.json`의 수치 허용오차는 미검증 초기 기준이며 pilot 측정으로 정확히 보고한다. 허용오차를 넘으면 같은 shape의 reference 경로로 한 번 확인하고, reference 경로가 맞으면 그 경로를 쓴다. confirmed wrong gradient/input/state/commit은 구현 수정 대상이다. 단순한 rounding 차이를 새로운 과학적 gate로 확장하지 않는다.

### P2: BS100 실행량 확인

별도의 cold W0/H0에서 main B1의 동일700행을 사용한다. A/B 각각 첫 warm-up 후보1회와 측정 후보3회, 합계8 joint 평가로 peak memory와 시간·실 token 수를 기록한다. 이 후보는 main에 이어 쓰지 않는다. A의 모델/토큰 경로가 같으므로 native task F/B 측정은 공유 비교가 가능하되 B의 geometry와 policy 비용은 별도로 계측한다. Timing receipt는 실제 실행 수를 기록한다.

MB=4를 시작값으로 하며 OOM이면 2→1로 줄여 parity를 확인한다. 편집층·문장·정밀도·예산을 줄여 메모리에 맞추지 않는다. 다른 kernel/cache 경로 선택은 구현 정합의 문제이며 P/N 성적으로 선택하지 않는다. MB1도 자원에 맞지 않으면 해당 자원 제약을 보고한다. 구현 대상 기본 경로는 compute-r1이며, 아직 검증되지 않은 최적화는 동일 목적의 reference로 대체하고 사유를 기록한다. A/B 공통 row grouping/합산 정책을 main 전에 고정한다. 추가 microbatch 확대를 시험하면8회 timing과 구분하여 별도 물리 호출 수를 기록하며 장기 sweep으로 늘리지 않는다.

### M: 본 실험

각 chain을 다시 cold W0/H0에서 시작해 한 process의 RAM state로20batch를 이어 간다. P1/P2의 W/H/Adam 상태는 가져오지 않는다. 준비·pilot이 기술적으로 성립하면 성능 확인을 위한 추가 사용자 승인 없이 main으로 연결하는 실행 명세다. Pilot PS가 낮다는 이유로 eta/학습문장/예산을 바꿔 본실험을 시작하지 않는다.

## 4. 평가 일정과 분모

| 시점 | 범위 | R / P / N |
|---|---|---|
| W0 | 첫2k 전체 | 2000 / 4000 / 20000 |
| 매 B1–B20 commit 직후 | current100 | 100 / 200 / 1000 |
| W5 | 누적500 | 500 / 1000 / 5000 |
| W10 | 누적1000 | 1000 / 2000 / 10000 |
| W20 | 최종2000 | 2000 / 4000 / 20000 |

Milestone current는 같은 all-seen raw의 부분집합으로 재사용한다. W0도 동일 모델·tokenizer·입력·평가·수치 환경을 검증한 신규 chain 사이에서 한 번만 측정한다. 비관측 all-seen 시점은 `NOT_MEASURED`; 직전 값을 복사하지 않는다. 이 일정은 계산 구현 변경과 구별되는 **새 실험의 평가 cadence**이며 평가하는 panel 내부의 표본/분모를 줄이는 것은 아니다.

기존 two-arm 일정은 RP all-seen 매batch, N current와 milestone을 사용했다. 새 일정의 편집 후 prompt-pair 수는 arm당67,600(5,200 request evaluations), 기존115,000 대비41.22% 감소한다. A/B와 공통 W0 합계는161,200 pair다. 두 target의 점수를 계산해야 하므로 pair를 실제 forward 행 수나 FLOPs로 부르지 않는다. Transformer runtime 절감률로 환산하지 않는다.

기본 성적은 실제 committed weight 모델의 strict-inequality preference다. R/P는 new_nll < true_nll, N은 true_nll < new_nll, tie는 실패. 별도로 원하는 전체 target의 teacher-forced argmax strict를 보고한다(R/P=new, N=true). 자유 생성 정확도와 구분한다.

필수 보고:

- W5/W10/W20 전체 R/P/N과 strict, current at-write 성적.
- 모든100-request cohort의 at-write→W20 retained/lost/gained와 birth-success 조건부 유지율. birth가0이면 조건부 값은 NA.
- 첫1k의 W10→W20 변화와 전체 W0→W20 변화.
- NS true/new NLL 변화와 new-target strict, preference lost/gained. NS 손실이 기존 정답 악화인지 새 target 침투인지 구분한다.
- Active/superseded 분리. 첫2k에는1,983 unique claims, conflicting-target claim13개, endpoint superseded request16개가 있다. Primary는 모든2k occurrence를 그대로 포함하고 active1,984/superseded16을 보조 표로 보고한다. History에서도 이들을 몰래 제외하지 않는다.

paired row key는 case ID/kind/prompt index 및 원 prompt/target identity다. 누락 raw를 intersection하여 분모를 조용히 바꾸지 않는다. 필요하면 CPU claim-cluster paired bootstrap을1,000회(seed고정) 사용하되 한 trajectory/order 조건부 불확실성이지 여러 독립 학습 실행의 CI가 아님을 명시한다. RS/PS/NS를 임의의 하나의 종합점수로 합쳐 승자를 정하지 않는다.

## 5. 작은 telemetry로 필요한 해석 확보

후보마다 NLL/KL/native norm/V 및 gradient norm, 실제 후보/Adam/F/B 횟수, time을 저장한다. V의 energy와 unrealized 성분을 분리한다. B의 cost/task gradient norm 비율과 cosine을 기록하여 추가 규제가 strength를 압도하는지 확인한다. 이를 optimizer에 feedback하지 않는다.

Batch/layer마다 요청 delta norm·clamp fraction·norm share, actual pooled action norm·share, residual ratio/cosine, entry/current Q trace·spectrum 요약 및 drift, solve residual, history append count를 기록한다. Norm share는 기능적/인과적 기여율이 아니다. Q≈0에서 V가 일반 quadratic norm에 가까워지는 포화를 보고한다.

A에 B의 entry geometry solve를 강제로 추가하지 않는다. A의 current writer geometry는 writer에서 이미 계산한 것으로 기록하고, entry-Q는 `NOT_MEASURED`로 둘 수 있다. V를 위한 B의 준비 비용은 실제 방법 비용으로 포함한다. 매후보 weight materialization·dense weight gradient·추가 full-weight loss를 복원하지 않는다. 큰 counterfactual cross-effect GPU 진단이나 inverse-Q repair는 이번2k의 필수 작업이 아니다.

가장 먼저 볼 해석은 A의 actual PS가 회복되는지, B가 PS를 잃어 NS만 올리는지, at-write 약함과 이후 망각 중 무엇이 큰지다. A/B 차이가 없고 V의 geometry 구별도 작으면 “배분이 효과적”이라고 보고하지 않는다. 특정 층 집중 자체는 유효한 결과다.

## 6. 실행 비용·자원·저장

A/B main 합계는40batch, joint 후보1,000회, Adam update최대960회, native input-row evaluations700,000개다. 실제 microbatch F/B와 유효/패딩 token 수를 따로 센다. Baseline의 요청별 native iteration 수와 이 숫자를 직접 나누어 speedup을 만들지 않는다.

SH4 기존 project cap2, job당1GPU, host memory≤60416MiB를 사용하고 제출 당시의 최신 cap/점유를 다시 확인한다. 기본 실행은 lane1 A, lane2 B. 다른 admitted job이 차지하면 cap 안에서 대기한다. 다른 실험·server3 job을 취소/steer/hotpatch하지 않는다. Baseline job은 등록하지 않는다.

ETA는 P2의 측정 candidate 시간에 entry 준비·실제 writer·H append·평가를 더한다. 각 arm에 대해 `20*(25*t_candidate+t_entry+t_write+t_history)+t_eval_schedule+t_load`를 초기 추정으로 쓰고 B1/B5의 실제 누적 시간으로 갱신한다. geometry solve·평가를 P2에서 재지 않았다면 해당 항은 UNKNOWN으로 남겨 전체 ETA를 확정하지 않는다. 빠른 ETA를 얻으려고 추가 큰 pilot을 요구하지 않는다. Historical baseline 시간은 실행 환경이 일치하지 않으면 속도 비교에서 제외한다.

첫 candidate가 cache/teacher 준비를 겸하면 정확한 계산은 `sum_batch(t_candidate0_preparation + 23*t_warm_forward_backward + t_final_forward + t_writer + t_history + t_evaluation)`이며 공통 준비비용을 두 번 더하지 않는다. 위25배 식은 초기에 사용하는 거친 평균식이다. Candidate25의 backward 생략, factor 재사용/재분해, cache fill/hit, cropped/pruned/유효 token, head 위치, dtype·route별 forward/backward·geometry·전송 시간을 남긴다. Source manifest에는 capability별 `qualified_optimized/reference_fallback/not_applicable`와 근거를 기록한다.

기존 사용자 checkpoint 미저장 정책을 적용한다. W/H/optimizer/RNG resume bundle, 전체 update/factor처럼 복원 가능한 payload를 영속 저장하지 않는다. Batch transaction rollback은 RAM에서만 유지한다. Raw metrics/log/source/config/input identities/scalar geometry와 상태 hash는 저장한다. `checkpoint_saved=false`, `exact_resume=NOT_AVAILABLE`. Crash 시 같은 경로에 덮어써 재개한 척하지 않고 partial 결과와 실패 지점을 남긴다. 코드/환경 수정 후 새 attempt는 cold W0에서 실행하며 실패한 구간도 비용 장부에 남긴다.

## 7. 구현·인계와 완료 조건

실행 담당은 기존 사용자 지정 흐름인 GH→SH4를 따른다. 이번 문서 자체는 GH/SH4 전달 수락이나 Slurm 제출 영수증이 아니다.

구현 책임은 새 `project/run_scripts/jlz_native_joint/`와 evaluator wrapper다. Native input/loss 정의와 Tensor/tuple 호환 adapter는 구현 정합 검증에 재사용할 수 있다. Native 목적 대조는 A/B의 기술 검증이며 별도 baseline 편집 chain이나 native z fit 실험을 추가하지 않는다. 과거 two-arm120/SPG runner는 v4 실행기로 사용하지 않는다.

필수 모듈: inputs/entry/allocation/oracle/optimize/writer/observe/run/collect. Observer와 collector는 같은 `evaluation-schedule.json`을 읽는다. 기존 collector의 hardcoded 평가 일정을 그대로 가져오지 않는다. 입력·방법·성능 측정·no-checkpoint·작업별 resource lock을 구현한다.

Model/benchmark profile을 해석하는 adapter interface는 `../portability-contract.json`을 따른다. B_t와 layer별 shape/context count는 runtime 데이터에서 얻고 마지막 partial batch를 보존한다. 이번 실행 schedule의100/20은 profile 검증 값이다. Prefix/future-prune/history key 재사용은 실제 parameter dependency가 성립하는 capability만 켠다. 큰 B의 exact operator 경로에서 요청을 독립 batch로 쪼개 Q coupling을 없애면 안 된다.

아직 없는 v4 CLI를 실행 가능한 것으로 적지 않는다. SH4가 `--help`와 dry preparation을 검증한 실제 argv/source closure를 launch manifest에 기입한 뒤 제출한다. 제출 lock의 model/stat/tokenizer/runtime/source fingerprint 미결속은 기술적 준비 미완료다.

최종 산출물은 source/config/launch manifest, A/B pilot technical+timing receipts, 각 arm의20batch commit/history ledger, immutable per-case evaluation raw, W5/W10/W20 비교표, cohort/active/NS leakage 표, 층별 배분·실현 및 비용표다. Complete는 A/B 각각20commit과 필수평가를 완료했을 때다. 기존 baseline의 참고 비교는 선택 사항이며 자료 유무가 완료 조건이 아니다. Ours 한 arm만 완료하면 partial이다. 중간 성적이 약해도 유한·정합한2k trajectory는 끝까지 진행한다.

## 근거 파일

- 방법: `../contract.json`, `../method-ko.md`, `../artifact-manifest.json`.
- 입력: `../inputs/exact-native-inputs.json`.
- Native baseline 재사용: `project/run_scripts/jlz_two_arm/prepare.py`, `baseline_pilot.py`, `collect.py`.
- 평가: `project/run_scripts/alphaedit_strength_neutral_barrier/evaluator.py`; `project/run_scripts/jlz_two_arm/observation.py`.
- HJ: `project/run_scripts/memit_hj/writer.py`, `history.py`; `plans/global/2026-09-30-memit-hj-experiment-design-v1/cells.csv`.
- 자원/저장: `control/gpu-concurrency-policy.tsv`, `servers/slurm-memory-policy.tsv`, `plans/global/2026-09-19-default-no-experiment-checkpoints.md`.
