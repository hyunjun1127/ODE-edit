# JLZ native 목적 확정 및 4요청 pilot v2

2026-10-01. 사용자 지시 “native 기준으로 바꾸자. 다시 설계해보고 pilot test 간단하게 진행”에 따른 실행 명세다. 앞선 v1의 CPU 검증을 바탕으로 **native KL을 기본 방법으로 확정하고 실제 사전학습 Llama-3-8B-Instruct에서 최대 4요청만** 실행한다. server3의 현재 실험과 상태·코드·queue를 공유하지 않는다.

## 방법 확정

목적은 활성 요청에 대해 `NLL + .0625 KL(current || entry)`를 합하고, 층·요청별 `.5 ||R_lr|| / ||h_lr||²`를 더한다. clamp는 `.75 ||h_lr||`, 진입 NLL<.05는 고정 0이다. native의 KL 인자 순서 `kl_div(entry_logp, current_logp, log_target=True)`를 그대로 사용한다. KL teacher는 해당 batch의 진입 W에서 한 번 캡처해 고정한다.

L4–L8 전체의 local-z 변위 R을 공동 최적화한다. 층별 `adj = solve(15000 C0 + H + K Kᵀ, K)`는 batch 진입에 만들고 고정한다. 반복마다 `W_eff = W_entry + (R.double() @ adj.T).float()`를 만들고, 실제 W_eff를 적용한 모델의 손실을 R에 대해 미분한다. FP32 block proximal solver가 norm decay와 clamp를 함께 처리한다. 가중치 자체는 최종 승인된 반환점 전까지 변경하지 않는다.

native 세부 규약도 맞춘다.

- rewrite NLL: 6 context를 같은 비중으로 평균하고 target token 평균 후 요청 간 합산한다.
- key: target prefix 없는 native prompt에서 subject_last를 추출한다. **context group 내부 평균 후 group 평균**이므로 `[1,5]` 구성에서는 canonical .5, 나머지는 각각 .1이다. 여섯 key를 단순 평균하지 않는다.
- C0: 기존 npz의 `mom2.mom2`는 합계다. native `SecondMoment.moment()`와 같이 **FP32 raw / count 후 FP64 변환**한다. count는 66,019,200이다.
- prefix: L4 입력, 즉 L3 출력을 재사용한다. KL은 전체 다섯 층의 후보 update를 적용한 최종 모델 분포에서 한 번 계산한다.
- 반환 평가에 사용한 FP32 W_eff를 그대로 commit한다. `CONVERGED` 또는 전 요청 `POLICY_ZERO_STEP`만 commit한다. plateau·상한·nonfinite는 반영하지 않는다.
- 각 batch의 다섯 층 commit 후 post-write key로 H를 한 번 append하고 다음 batch로 넘어간다. all-zero도 H append는 수행한다.

첨부 `jlz_ref`와 v1 결과는 원본 감사 자료로 보존한다. 새 실행 경로 `project/run_scripts/jlz_pilot/`에는 KL 방향 선택 옵션을 두지 않으며 native 방향만 제공한다.

## 최소 실행표

|단계|실행|판정|
|---|---|---|
|CPU 입력·solver|닫힌형 prox 문제 및 native tokenizer/target/lookup 비교|기술 검사 통과 후 GPU 진입|
|batch0|고정 첫 2요청, case_id 16186·3743, BS2, W0/H0|비영점 full/suffix loss·gradient 확인 후 joint solve|
|batch1|다음 2요청, case_id 1481·18140, BS2|batch0이 수렴·commit 정합을 통과했을 때만 실행|
|관측|W0에서 4요청, 각 commit 뒤 all-seen 2/4요청|native 6-context NLL 및 canonical teacher-forced strict만 기록|

본 pilot은 v1의 G0–G2 전체를 축소한 **구현 동작 시험**이다. MEMIT-H 대조군, 8요청 교정, R/P/N 전체 평가, 장기 성능 비교는 이번에 수행하지 않는다. 첫 4개 실제 target은 모두 single token이다. multi-token 입력은 별도 합성 text fixture로 native tokenizer parity를 확인한다.

## 고정 자산과 환경

- 모델: Meta-Llama-3-8B-Instruct, revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`.
- 데이터: counterfact-fixed-10k-v1 앞 4개, SHA `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`, shuffle 없음.
- context: 기존 6개, SHA `33cec0eef9ec130f26c2f0e17f7c8be39e93c717ebb47eaa5e9f1c88b263524e`.
- C0: 기존 L4–L8 Wikipedia mom2 100k cache, 재계산 없음.
- weight/activation/R: FP32, solve: FP64, TF32/autocast off, eager attention, frozen base parameters.
- runtime: 로컬 PyTorch 2.9.1+cu128 / Transformers 4.57.1. 기존 server3의 Transformers 4.44.2와 runtime byte parity를 주장하지 않는다.
- 자원: **server1/devbox A6000 1개**, CPU8, host RAM64000M, Slurm wall60분. server3 작업을 제출·수정하지 않는다.

## 예산과 종료

각 batch의 optimizer 호출은 **최대 120회**이며 마지막 반환 평가를 포함한다. full/suffix 비영점 probe 2회는 별도이므로 **최대 122회/batch**다. key/entry/post forward와 관측은 별도 계수한다. prompt microbatch는 2다.

solver tol은 **1e-4의 임시 동작 기준**이다. FP64 전체 oracle 교정을 완료한 production tolerance라고 부르지 않는다. 수렴을 주장할 때 이 tol과 정규화 residual을 함께 보고한다. 최적화 시간은 batch당 최대1200초, process는 모델 로딩 포함3300초 soft budget과 Slurm3600초 hard limit이다. 시작한 kernel·마지막 반환 평가 때문에 soft budget을 소폭 넘을 수 있다.

종료 상태는 CONVERGED, POLICY_ZERO_STEP, STALLED_AT_PRECISION, BUDGET_STOP, LINESEARCH_FAILED, NONFINITE, TECHNICAL_FAILURE로 구분한다. 첫 batch에서 수렴하지 않으면 다음 batch로 진행하지 않는다. 미수렴 결과를 반영하거나 tol·decay·λ를 결과에 맞춰 완화하지 않는다. OOM에 자동 BF16 전환이나 layer 축소도 하지 않는다.

## 필수 검증과 기록

비영점 R에서 full/suffix loss 절대오차와 gradient 정규화오차를 각각≤1e-3으로 확인한다. 이 값은 짧은 FP32 기술 허용치이며 정밀도 교정 결과는 아니다. 조합 이전 CPU 검사에서 KL 방향의 값·기울기, microbatch 축약, 실제 native token 준비의 동일성을 별도로 검증한다.

commit 시 평가된 W_eff와 byte 일치, native full forward NLL·KL 오차≤1e-3, L4 key byte 불변, 모든 block clamp 준수를 요구한다. 수치 gate가 실패하면 진입 W를 복원하고 H를 append하지 않는다. commit 이후 연산 자체가 예외를 던지면 TECHNICAL_FAILURE로 process를 종료하여 RAM 상태를 폐기한다. 미수렴 solver는 W/H를 유지한다. 비편집 parameter는 requires_grad=False이며 optimizer 대상이 아니다.

source/config/input identity, case IDs, 진입·fit·commit NLL/KL/decay, solver status/calls/backtracks/residual/signed KKT, 초기 gradient/decay 가격 비율, 층별 R·ΔK norm, 활성 block 수, H norm, 단계 시간 및 GPU peak를 기록한다. 최종 z·weight checkpoint는 저장하지 않는다. source 스냅샷과 compact receipt만 남긴다.

원본 모델 파일은 읽기 전용으로 로드하고 수정된 W/H는 pilot process RAM에만 존재한다. source와 결과는 독립 `jlz_pilot` 및 `local/jlz-native-pilot-v2/` 경로로 분리한다. 이 규모에서 확인할 것은 native 목적을 사용하는 공동 최적화의 수치 동작이며 편집 성능의 일반적 우월성은 아니다.
