# PRICE ridge M1–M3: server4 실행 지시

- Instruction: `USER-SH4-PRICE-RIDGE-M1-M3-20261008-R1`
- Task: `price-ridge-m1-m3-2k-20261008`
- 담당: server4 / `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`
- 작업 위치: `/data/janghj/ODE-edit`, repository `hyunjun1127/ODE-edit`
- 사용자 제공 원문: [user-spec.txt](user-spec.txt). 원문 예상값은 검증된 관측값과 구분한다.
- 사용자는 이 명세를 server4에 맡기라고 명시했다. 아래 범위의 구현과 단계별 실행은 이미 위임되었다.

## 최신 사용자 지시 — 원문보다 우선

> w0 실험은 하지말라고 해
> server4의 gpu cap은 2이다

**별도 W0 실험, W0 재평가, W0 generation 평가, W0를 위한 신규 GPU 준비·calibration job을 실행하지 않는다.** 기존 W0 관측/통계/저장 key만 identity를 확인하고 재사용한다. M3는 저장된 B1 mean key와 C0로만 보정한다. 필요한 값이 없으면 `NOT_RECORDED`와 정확한 누락 항목을 보고하며 새 W0 실행으로 메우지 않는다. 각 2K 편집 trajectory가 기존 원모델에서 시작하는 정상 초기화는 별도 W0 실험을 뜻하지 않는다. launcher의 자동 W0 평가도 비활성화하고 이미 있는 기준값과 연결한다. 원문 표의 반올림 W0 숫자만으로 raw 비교의 일치가 검증되었다고 주장하지 않는다.

server4 전체 `janghj` 프로젝트 할당은 **동시 GPU 2개**다. 기존 실행, 준비, 평가를 모두 합산한다. 신규 job은 각 1 GPU, 초과분은 dependency/throttle로 pending 처리한다. 기존 hold baseline `60917–60923`의 hold와 dependency를 유지하고, 취소된 `61121/61122`를 재시작하지 않는다. 기존 held job과 정확히 같은 과학 설정을 복제 제출하여 hold를 우회하지 않는다.

## 책임과 불변 사항

server4가 세 모델의 구현, 0단계 확인, 실행 계획, source freeze, 제출, 진단과 수치 보고를 맡는다. 전용 branch `codex/server4-price-ridge-m1-m3-20261008`과 task namespace를 사용한다. 다른 작업자가 있으므로 그 변경을 되돌리지 않고 통합하며, 기존 frozen source와 결과는 보존한다.

Writer는 모든 ours run에서 ridge(MEMIT)로 고정한다. 가격, 예산/정확 사영, 목적함수, EfficiencyAdam, 마지막 평가 후보 commit, commit 뒤 mean key로 모든 대상 층 H 갱신은 현 구현과 동일하다. AlphaEdit writer 전환, editing window 변경, 조기 종료, beta_max 고정은 이 task에 포함하지 않는다.

주 arm은 CAP075, B=100, 20 batches/2,000 requests다. beta_base=0.75, layer cap=0.75, 확장 4단계, F<0.05, KL=0.0625, norm=0.5를 유지한다. CAP100/FREE100을 기본 matrix에 추가하지 않는다. 최초 한 번 활성화된 요청은 끝까지 update한다. 같은 request/context 순서, seed, native hparams, evaluator와 runtime identity를 모델별로 봉인한다.

## 0단계 — 저장된 raw만 확인

1. GPT-J의 canonical subject 마지막 token 위치와 slot mapping을 확인한다. B3/47, B5/36, B7/26, B10/25에서 층별 canonical anchor, batch median, 그 비율 및 가능하면 prefix norm 평균을 표로 낸다. slot의 0/1-based indexing을 명시한다. Spain(P463)은 실제 prompt/token 위치가 확인될 때만 확정한다. 원문에 없는 outlier cutoff를 사후 발명하지 않는다. anchor 이상치가 없으면 M1 근거 실패로 보고하고 관련 본 실행을 보류한다. 저장 anchor가 없으면 `NOT_RECORDED`; 별도 W0 복원 실행은 금지다.
2. 세 모델의 기존 ridge CAP075 B1 `entry-price.json`에서 최하층의 `median(1-denominator)`를 읽는다. Llama3 L4, GPT-J L3, GPT2-XL L13. artifact path/hash, arm, lambda, request count, denominator 정의/수치 floor를 함께 기록한다. Llama3가 0.5 미만이면 불변 전제 실패이므로 원인을 검토하기 전 그 모델의 본 실행을 진행하지 않는다.
3. M3가 필요한 모델은 저장된 원모델 B1 K와 native C0만으로 H=0 ridge solve를 반복하며 lambda를 내린다. 0.5±0.01에 들어오면 종료하고 bracket/각 lambda/중앙값/허용오차/최종값을 보존한다. lambda는 native 이하의 양수, 실제 실행 전체에서 고정한다. root가 없거나 numerical failure면 몰래 damping/다른 solver를 추가하지 않는다. key/stat이 없어도 새 W0 GPU job을 만들지 않는다.

0단계 보고서는 `experiment-reports/servers/server4/price-ridge-m1-m3-2k-20261008/phase0-ko.md`에 작성한다. 수치가 예상과 같다고 가정하지 않는다. 기술적으로 독립인 모델의 CPU 구현은 병행할 수 있다.

## 구현

**M1 anchor guard:** `jlz_price_gptj/entry.py`, `jlz_price_gpt2xl/entry.py`의 `prepare_entry`에서 canonical lookup이 정확히 0인 요청에만 적용한다. 이미 잡힌 같은 forward의 prefix rewrite 5개에 대해 subject 마지막 위치 hidden의 **norm을 각각 구한 평균**을 `entry['anchors']`에 넣는다. canonical/KL 행을 prefix 평균에 넣지 않는다. 평균 hidden의 norm으로 대체하지 않는다. anchor_star도 guarded 최상위 anchor와 일치시켜 가격, cap, budget weight, norm penalty에 같은 값이 전달되게 한다. canonical hidden/vector 및 key 정의는 바꾸지 않는다. p>0인 요청, 특히 Llama3의 수치 경로는 그대로 둔다.

**M2 grace:** `K_grace=(hp.v_num_grad_steps-1)//2`. 평가 후보 수가 아니라 실제 update 수의 절반이다. Llama3/GPT-J 25 evaluations/24 updates -> 12 유지, GPT2-XL 20/19 -> 9. GPT2-XL M3-only arm은 대조를 위해 grace=12를 명시적으로 유지한다. M2-only arm은 native lambda=20000을 유지한다.

**M3 realization floor:** 기존 B1 최하층 median이 0.5 이상이면 native lambda를 정확히 유지한다. 미만이면 위 저장 tensor 보정값을 profile `lambda_C`에 기록한다. GPT-J/Llama3 native=15000, GPT2-XL native=20000. calibration key는 요청별 native mean key이며 새 anchor 규칙 때문에 mean key를 바꾸지 않는다. 배치별 lambda 재보정, test/N prompt 이용, 성능 수치에 맞춘 lambda 탐색은 없다.

의미 있는 최소 검증은 M1 조건/5-prefix reduction/anchor_star 연결, M2 update 경계 및 ablation config, M3 native no-op/threshold/고정값 적용, 기존 last-candidate/H contract다. Llama3 B1 payload 및 W 해시는 기존 run과 비교한다. generation/tracking의 RNG 소비 때문에 method trajectory가 달라지지 않게 기존 rng/seed를 보존한다. 기존 공유 `experiment_tracking/**`는 SH1 소유이므로 재작성하지 않는다.

## 1단계 — ours 2K, 총 5개 구성

| 모델 | 구성 | grace | lambda |
| --- | --- | --- | --- |
| Llama3 | CAP075 재현 | 12 | 15000, M3 비발동 확인 |
| GPT-J | CAP075 + M1, 필요 시 M3 | 12 | native 또는 저장 tensor calibration |
| GPT2-XL | CAP075 + M1 + M2 | 9 | 20000 |
| GPT2-XL | CAP075 + M1 + M3 | 12 | calibration |
| GPT2-XL | CAP075 + M1 + M2 + M3 | 9 | 같은 calibration |

모든 run은 2K까지 가며 1K는 중간 점검이다. 낮은 성능만으로 1K에서 중단하지 않는다. raw/identity/nonfinite 오류와 0단계 실패는 typed failure로 보존한다. Fluency/Consistency는 편집 후 evaluator에 포함하되 새 W0 generation은 실행하지 않는다. 실제 M3 비발동으로 완전히 같은 arm이 생기면 동치와 실행 여부를 명시하고 임의 lambda 변경으로 차이를 만들지 않는다.

- Llama3: B1 payload/W hash 일치 및 2K RS 99.80 / PS 91.18 / NS 84.94 일치. 미일치 시 method 변화인지 runtime/generation/RNG 차이인지 증거로 분리하며 재현 PASS를 선언하지 않는다.
- GPT-J: 위치 0 batch의 직후 neighborhood true NLL 변화가 동일 batch BASE_ALPHAEDIT보다 +0.1 이내인지 각 batch별 보고한다.
- GPT2-XL: base budget B1 충족 >=70%; neighbor true NLL W0 대비 1K<=+0.6, 2K<=+0.9; all-seen R new NLL 1K<=0.3, 2K<=0.4; 2K RS>=99, PS>=92; NS 1K>=67, 2K>=64. 원문 기준별 PASS/FAIL 표와 충족 개수를 낸다. 동률은 임의 tie-break를 발명하지 않고 보고한다. 어떤 구성도 NS와 true NLL 조건을 함께 충족하지 못하면 원문대로 native lambda 현행 결과를 fallback으로 보고한다. 저장 W0 기준값이 없거나 identity 불일치면 delta 기준은 평가 불가로 표시하고 새 W0 평가를 하지 않는다.

## 2단계 — matched baseline 2K

선정 profile이 정해진 뒤 세 모델의 MEMIT-BLUE, AlphaEdit-BLUE를 현 runtime/evaluator/같은 2,000건 순서로 비교한다. MEMIT와 MEMIT-BLUE는 ours와 같은 lambda로도 비교한다. native lambda 동일 모델의 겹치는 MEMIT-BLUE는 한 번만 실행한다. M3가 켜졌다면 native BLUE와 matched-lambda MEMIT-BLUE를 구분한다.

현재 hold된 기존 baseline을 release하지 않는다. 해당 기존 run과 동일한 설정이 필요한 경우 `WAITING_EXISTING_BASELINE_HOLD`로 연결하고 중복 제출하지 않는다. 새 최종 lambda/runtime 등으로 과학 설정이 달라지는 명세의 matched run만 별도 identity로 등록한다. 아직 profile이 미정이면 명세만 준비하고 성급히 baseline을 제출하지 않는다. W0 평가 금지와 GPU 합산 cap2는 이 단계에도 적용한다.

## 3단계 — ablation 2K

세 모델 각각 FLAT(price=1)과 최하층 단독+history를 실행한다. 최하층은 Llama3 L4, GPT-J L3, GPT2-XL L13. 선정 ours의 나머지 설정과 data/evaluation schedule을 유지한다. 최하층만 쓴다는 이유로 anchor/readout/loss 정의까지 바꾸지 않는다. 가격 배분 효과와 history 효과를 분리할 수 있도록 원문 baseline과 함께 표로 정리한다.

## 공통 산출물 및 완료 기준

매 batch: 직전/직후 R/P/N, neighborhood true/new NLL의 post-minus-pre, base budget 충족 비율, 확장 비율, beta_max 평균, 층별 price median, 최저가 층별 요청 수, 층별 delta-W norm. 0.5K마다 all-seen neighbor new NLL leakage와 true NLL erosion(기존 동일 W0 기준), R/P new NLL retention, RS/PS/NS 및 generation 결과. metric마다 prompt/request 수, 단위, 시점, 유효 개수를 기록한다. beta_max와 층간 가격 차이의 증가 여부를 수치로 기록한다.

실행 전 source/config/asset/evaluator/seed/order identity와 run matrix를 freeze한다. W&B는 기존 정책의 job ID/model/arm/source identity와 scalar를 사용하고, 과학 계산을 tracking 때문에 재실행하지 않는다. 등록 job ID, 의존 관계, source commit, run URL, 출력 경로와 실제 `NOT_RUN/PENDING/RUNNING/DONE/FAILED`를 보고한다. task 수락을 GPU 실행 완료로 표현하지 않는다.

기계적 preflight/post-run audit를 남기고 `block`이면 해당 실행을 멈춘다. SH 보고는 사실·수치·명세 gate 결과를 기록한다. 구현과 소형 report는 전용 branch에 게시할 수 있으며 main merge는 기존 GH 통합 절차를 따른다. 정확한 write scope, 산출물 경로와 artifact broadcast 정책은 함께 있는 instruction JSON을 따른다. 원문과 최신 사용자 지시를 넘는 추가 sweep/새 model/stat 생성은 하지 않는다.
