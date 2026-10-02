# JLZ v6 두 arm 각 500 edit 실행 설계

2026-10-02 KST. 사용자의 최신 실행 조건은 **warmup 없이 일정한 native Adam learning rate**, **v6 A/B 각각 500개 순차 편집**, **GH가 SH4에 배정하여 server4에서 실행**하는 것이다. [Method](../method-ko.md), [구현 계약](../implementation-ko.md), [기계 판독 계약](../contract.json)이 알고리즘 정본이며, [experiment.json](experiment.json)은 이번 실행 범위를 고정한다. 이전 초안의 5-update warmup은 이번 지시에 적용하지 않는다.

## 1 실행 범위와 시작 상태

- Main은 **v6 A 자유 배분 / v6 B 분산 선호 두 arm만** 실행한다. 각각 BS100×5 sequential commit이며 첫 500개 요청과 순서가 같다. 총 main 편집 occurrence는 두 arm 합계 1,000개다.
- 각 arm은 원본 pretrained W0, H0=0, empty memory, 새 optimizer/RNG 상태에서 독립적으로 시작한다. 한 arm의 종료 상태를 다른 arm으로 전달하지 않는다.
- 기존 v4/v5, 특히 500개 편집 후 중단된 v5 checkpoint·수정 weight·history·memory에서 이어서 실행하지 않는다. Pilot/timing의 상태도 main에 이어 쓰지 않는다.
- **Batch 5 commit과 W5 누적평가 후 정상 종료한다. Batch 6 입력 준비·fit·commit을 시작하지 않는다.** Collector는 500개 완료 또는 실제 partial/failure 상태를 보고하며 2k 완료로 표시하지 않는다.
- 기존 baseline은 저장된 산출물로만 비교한다. 새 baseline main/pilot/fallback을 실행하지 않는다. 무관한 다른 server 실험·job은 변경하지 않는다.

## 2 고정 profile과 두 arm

이번 profile은 기존 server4의 Meta-Llama-3-8B-Instruct / CounterFact fixed10k다. 전체 eligible layer **L4–L8**을 모두 포함한다. 이는 이 모델의 실행 profile이며 method를 BS100, CounterFact 또는 이 아키텍처에 상수 결속하지 않는다. 다른 B·partial batch·context/target 길이·layer 차원은 실제 입력에서 계산하고, 새 model adapter는 별도 정합 검증을 거친다.

두 arm에 공통인 항목은 다음과 같다.

- Native subject 위치에 절대 δ를 주입하고, 모든 eligible layer의 δ를 Adam으로 공동 학습한다. 동일 요청의 native rewrite 6개와 KL 1개에 같은 층별 δ를 사용한다.
- Native NLL readout은 이 profile의 layer31, KL은 current‖own entry full vocabulary이다. Native 문장·lookup·target·context/token reduction을 바꾸지 않는다.
- Adam betas=(.9,.999), eps=1e-8, weight_decay=0. **모든 24 update에서 lr=.1**이며 warmup·scheduler·line search를 사용하지 않는다. 이 값은 bound native profile의 learning rate다.
- Batch당 native 후보 평가 25회 / Adam update 24회. 전체 logical batch의 gradient를 합한 후 step을 한 번 수행한다. 매 update 직후 native norm ball ‖δ_lr‖≤.75a_lr로 projection하고 moment를 임의 초기화하지 않는다. Post-update clamp를 제거하거나 사전 gradient clipping으로 대체하지 않는다.
- Native norm=.5, KL=.0625, C0 coefficient15000, FP32 model/FP64 geometry, eager attention, autocast/TF32 off를 유지한다. λ_W=λ_E=.1, β_subject=β_distillation=.1, past KL=.0625, past NLL degradation=1이다.
- Writer P는 batch entry의 전체 native rewrite context에서 한 번 계산하고 fit·probe·commit에서 고정한다. 최종 actual materialized weight를 그대로 commit한다.
- Physical probe는 **native 후보 5/10/15/20의 update 전**, 즉 완료한 update가4/9/14/19회일 때 실행한다. Current/past의 disjoint partition은 네 probe를 합쳐 각 요청을 한 번 포함한다. 선택한 요청을 관측하더라도 공유 write에는 전체 current B개 열을 사용한다.
- 과거 보존은 두 arm 모두 사용한다. 이미 사용한 native 문장만 담은 memory128, reference cap16, batch별 고정 표본 min(16, 실제 B, eligible resident facts)를 유지한다.

**유일한 arm 차이는 배분 geometry norm 모양**이다. A는 층별 g_l/e_l의 합, B는 각각 sqrt(sum of squares)를 사용한다. Native norm, clamp, Adam, replay, physical feedback, 계수, 입력·표본 seed는 같다. 이전 v5 A/B의 η=0/1 정의를 재사용하지 않는다. 두 arm은 첫 batch부터 geometry norm이 다르므로 B1 loss·trajectory 동등성을 요구하지 않는다. 한 층 집중은 허용하고 최소 사용 층 수·top-k·균등 분배 gate를 넣지 않는다.

## 3 입력과 재현 identity

원본 schedule은 `plans/global/2026-10-02-jlz-native-joint-v4/experiment-2k/case-schedule.csv`이며 SHA256은 `dccb4da4896aa7c3f653f7192bd01ab5d8799a0b439101417d02ee287d3fd2e2`다. 전체 파일을 검증한 다음 **stream_index0 0–499만** 선택한다. 그 500개 case ID 정수 목록을 `json.dumps(ids,separators=(',',':'))`로 직렬화한 UTF-8 bytes, trailing newline 없음의 SHA256은 `0be7d88c759e7f65a690514035f40a18c5c19d8591ddd33f97b2fa6187b64a95`다. 처음3개는16186/3743/1481, 마지막3개는3419/8549/5628이다. `active_at_W20` 열은 W5 상태가 아니므로 그대로 평가 분모에 사용하지 않는다. Active/superseded는 해당 endpoint까지의 fact/version으로 산출한다.

기존 입력의 실측 경로·SHA·runtime/model revision·5개 batch의 packed input identity·native reference source SHA는 `experiment.json`에 기록했다. 기준 증거는 server4의 `/data/janghj/ODE-edit/local/jlz-writer-coupled-v5/20261002-v1/attempt-r1/config.json`, `execution.lock.json`, `W0-reuse.json`이다. 이는 **입력·환경·기준 소스의 identity만 재사용**하는 것이며 v5의 optimizer·task graph·arm 의미를 상속하지 않는다. 실제 새 실행에서 현재 파일과 모델 상태를 다시 결속하고 차이를 기록한다.

## 4 최소 pilot과 제출

1. 기존 CPU 수학 검증, native input/lookup/readout parity, 고정 geometry·direct D backward·microbatch parity·commit/history/memory 정합을 검증한다. 실제 production 구현과 GPU 검증이 끝나기 전에는 PASS를 미리 표기하지 않는다.
2. Main 밖 개발 slice `[2000:2004]`, case IDs **541,6693,16935,17306**으로 각 arm **BS2×2 sequential** pilot을 독립적인 fresh state에서 한 번 수행한다. 정상 후보 예산 내에서 constant LR24회, 사후 clamp, B2 과거 native replay, probe partition/empty group, atomic commit을 확인한다. Pilot의 P/N 성능으로 계수·arm·층을 선택하지 않는다.
3. B100 메모리·구현 경로가 추가 확인을 필요로 할지는 **GH가 기술 근거로 판단**한다. 필요하면 별도의 fresh state에서 한 번만, native 후보 최대5개/Adam update 최대4회와 후보5의 physical probe까지를 짧게 확인한다. **후보5 probe 후 5번째 Adam update는 하지 않으며**, 이 상태는 main에 반영하지 않는다. 실제 B100 physical branch·메모리·timing 검증이 이미 충분하면 생략할 수 있다. 다수 profile sweep, 반복 timing, baseline 실행으로 확장하지 않는다.
4. 기술 정합과 가용 자원이 확인되면 SH4가 server4의 두 main lane을 제출한다. 일정한 낮은 RS/PS/NS, 특정 층 집중, clamp 포화, 고정 예산 미수렴은 후보 거절·추가 예산·중단 gate가 아니다. Nonfinite, input identity 오류, 잘못된 shape, commit 불일치는 기술 실패로 구분한다.

Main 도중 관측된 품질에 맞춰 LR·계수·probe·문장·예산을 조정하지 않는다. 자원 부족은 microbatch·정합이 검증된 구현 경로로 해결하되 objective와 logical update를 보존한다. 해결되지 않으면 변경 내용을 GH에 명시하고 결과와 혼합하지 않는다.

## 5 평가와 baseline 비교

W1–W4는 해당 current100 전량을 평가하여 R/P/N 분모100/200/1000을 저장한다. W5는 first500 전체 누적평가를 한 번 수행하여 **R500/P1000/N5000**을 저장한다. W5 current100 지표는 같은 raw에서 부분집합으로 계산하고 중복 GPU 평가하지 않는다.

주 지표는 기존 strict NLL preference rate이며 teacher-forced strict, true/new NLL·margin, token accuracy도 함께 저장한다. Cohort별 at-write→W5 lost/gained로 초기 획득과 후속 망각을 분리한다. N은 W0→W5 net 변화뿐 아니라 문항별 lost/gained도 산출한다. Official P/N 입력·결과는 학습, memory, teacher, 후보 선택, stopping, arm 선택, 계수 변경에 사용하지 않는다.

W0 평가값은 기존 raw에서 first500의 identity를 검증해 재사용한다. 사용자가 제시한 pre-edit N **87.84%=4392/5000**는 이 동일 cohort·token·metric·원본 W0 결속 확인 후 비교 기준으로 사용한다. 기존 W0 파일은 2k 전체도 포함하므로 그 전체 N88.555%를 500개 기준과 혼용하지 않는다. Historical evaluator의 실행 layout 차이는 별도로 밝히며 bitwise 동등성을 주장하지 않는다. W0 결과 재사용과 각 arm의 모델을 원본 W0에서 새로 시작하는 것은 별개다.

비교 표에는 기존 v5 A95.2/73.2/87.46, v4 A99.6/98.7/62.24, MEMIT-H99.0/87.3/86.28, AlphaEdit98.6/89.1/82.2, BLUE100/96.2/83.8, CAKE97.8/80.4/85.02를 참고 수치로 두되, 새 결과와 병합하기 전 endpoint·case·target·metric·분모를 확인한다. 다른 runtime과 BLUE의 L4+L8 구성은 표에 명시한다. 이 비교를 성능 통과선이나 재실행 기준으로 사용하지 않는다.

## 6 비용·산출물·종료

한 arm의 main 후보 평가는5×25=125회, Adam update는5×24=120회이다. 두 arm 합계250회/240회이며 physical branch 호출은 별도다. Batch의 past 표본 수를 M이라 하면 최적화 요청 bundle은 forward27B+M, backward25B+M이다. 이는 GPU 호출 수·토큰 수·runtime이 아니며 setup, geometry, history, 공식 평가 비용을 따로 기록한다.

SH4는 실제 implementation commit/source archive SHA, input/model/runtime manifest, pilot receipt, 필요한 경우 bounded B100 receipt, 제출 job IDs, batch별 entry→commit 상태 hash,25후보/24update와 실제 LR, probe teacher/current/past identity, norm/clamp 사용률, native/actual gap, 층별 배분·ΔW·history/memory telemetry, 원문항 raw와 W5 집계를 저장한다. Gradient 없는 terminal 후보는 `gradient_measured=false`, gradient값 `null`로 기록한다.

GH는 SH4 최신 registry/session과 server4 자원 cap를 확인하고 이 task 범위만 배정한다. Owner는 SH4, repository ID `hyunjun1127/ODE-edit`, remote repository `/data/janghj/ODE-edit`이다. 기존 일반 cap·무관 job을 침범하지 않는다. GPU main은 W5 산출물을 확보한 후 종료하고 collector는 A/B 각500 완료 여부를 검증한다. 반복 재시도·추가 arm·batch6·장기 polling은 이 설계에 포함하지 않는다. Pilot 완료·제출 완료·실제500완료를 구분해서 보고한다.
