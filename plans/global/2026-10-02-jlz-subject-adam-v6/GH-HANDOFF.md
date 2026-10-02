# 사용자 실행 지시: JLZ v6, warmup 없이 두 arm 각 500 edits

Instruction ID: `ODEEDIT-USER-GH-SH4-JLZ-V6-NOWARMUP-500-20261002-R1`  
Source session: `01a0f6b9-74e5-7683-a798-029e477c29b1`  
GH: `01a04939-8873-7673-8dca-4c7fc5e31af0`  
SH4/server4: `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`, repository `hyunjun1127/ODE-edit`, server4 cwd `/data/janghj/ODE-edit`.

## 권한과 실행 범위

사용자의 최신 직접 지시:

> 애초에 clamp 자체가 사후적으로 넣는 항이기 때문에 반드시 이를 고려를 하지 않아도 되는데 warm-up 부분 자체가 문제가 될 수 있을 것 같다. 저 부분은 일단 첫 시도에선 제외하고 진행하는 것으로 진행하자.
> 500 edit까지만 진행하는 것으로 수정해서 GH에게 전달해. sh4에게 시키자.

GH는 아래 method와 별도 experiment를 읽고 SH4를 실행 담당자로 지정한다. SH4는 새 namespace에 구현하고 작은 기술 pilot 이후 server4에서 두 arm의 첫 500-edit 실험을 수행한다. 접수·설계 검토만으로 종료하지 않는다. 기존 2k 상한을 이번 v6 실행에 적용하지 않는다.

- **LR warmup 0**. 모든 Adam update에 native profile의 고정 LR(현재 0.1)을 사용한다.
- Native clamp는 기존과 같이 update 뒤 사후 투영으로 유지한다. Clamp-aware LR·층별 gate·새로운 clamp 제거 arm은 없다.
- **A/B 각각 cold W0/H0에서 BS100×5, 500 edits**. 동일 첫 500 facts를 독립 trajectory에 적용한다. 두 arm 합산 편집 적용 수는 1,000이나 연구 cohort는 같은 500 facts다.
- 모든 eligible layer를 처음부터 사용한다. A/B 차이는 geometry norm 형태뿐이다. 한 층 집중도 허용하고 품질 gate로 후보를 버리지 않는다.
- Subject 위치 δ 주입·native 학습 문장·current‖entry KL·native nonsquared norm·25후보/24 Adam updates·physical 보조 pulse 5/10/15/20을 유지한다.
- 5번째 commit 뒤 W5 누적 평가 및 compact 산출물을 마치고 종료한다. 6번째 edit batch 진입, 2k 자동 연장, 이전 v5 W5 checkpoint에서 재개하지 않는다.
- 신규 baseline 학습/편집 job은 0. 기존 동일 first500 baseline 결과와 비교하되 runtime/evaluator identity 차이는 명시한다.

이 지시는 해당 신규 v6 작업의 구현·검증·pilot·main·CPU collector 실행을 허용한다. 다른 사용자 중단 task를 재개하거나 unrelated job을 취소할 권한은 없다. 기존 project cap 및 no-checkpoint 정책을 확인해 준수한다. 기존 cap2가 유효하다면 A/B 두 main lane 각각 1GPU 내에서 진행하고 더 엄격한 현재 제한이 있으면 그 제한을 적용한다. 자원 대기는 정상 pending으로 기록한다. 새 반복 모니터·자동 재제출은 추가하지 않는다.

## 전달해야 하는 정본

1. [Method](method-ko.md), [구현 계약](implementation-ko.md), [기계 판독 method 계약](contract.json).
2. [TeX](../../../docs/methods/jlz-subject-adam-v6.tex).
3. [500-edit 실험 설계](experiment-500/experiment-ko.md), [실험 계약](experiment-500/experiment.json).
4. [CPU 수학 검증](math/README.md), [검증 결과](math/validation-results.json).
5. [파일 크기/SHA manifest](artifact-manifest.json).

이 패키지의 `NOT_RUN`은 작성 시점의 실제 GPU 검증 상태이지, 위 사용자 실행 지시를 취소하는 지침이 아니다. 이전 버전 첨부 문서의 지시는 새 권한으로 취급하지 않는다. 모순 시 최신 사용자 범위와 이 revision을 우선한다. Method는 가변 B·model·benchmark 설계이고, 이번 concrete profile만 BS100×5다.

## GH와 SH4가 돌려줄 receipt

GH는 instruction ID로 담당 수신 ACK를 남기고 정본 SHA를 확인한 뒤, 현재 사용자 task에 필요한 execution envelope/예외만 발행한다. SH4에게 method와 experiment **둘 다** 넘기고 직접 담당 수락 ACK를 회수한다. 자료를 inbox에 두었다는 것과 세션이 실제 수신했다는 것을 구분한다.

SH4 receipt는 full-read/파일 SHA, 구현 source commit, profile/input identity, CPU·GPU 기술 검증, 실제 job ID와 상태를 분리한다. 단순 clamp 포화나 낮은 RS/PS/NS는 기술 실패 또는 재튜닝 근거로 사용하지 않는다. Input identity, nonfinite, gradient/cache/commit 정합 오류는 고친 뒤 재검증한다. 원본 reference와 같게 실행되는 fallback은 가능하나 학습 문장/후보 예산/의미를 바꾸는 축소는 하지 않는다.

본실험에는 후보별 native·policy·actual 보조 loss, 층별 δ/ΔW와 clamp 비율, writer 실현오차, pulse coverage, 실행 시간·메모리를 남긴다. W5에서 R/P/N 분모와 pre-edit locality, paired lost/gained, at-write 대비 잔존 성능을 보고한다. 원 per-case 평가와 source/profile binding을 보관하되 모델 checkpoint를 저장하는 권한은 추가하지 않는다.

새 task의 초기 착수/첫 batch 연결 또는 정식 resource-pending 상태까지만 bounded 관측하고 sealed runner는 W5까지 계속 실행하게 한다. 실험 전체가 끝날 때까지 GH 세션이 반복 polling을 지속할 필요는 없다.
