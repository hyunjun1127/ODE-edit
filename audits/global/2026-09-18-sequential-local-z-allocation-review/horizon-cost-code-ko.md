# 순차 local-z allocation: 긴 horizon과 계산 구조 추가 감사

2026-09-18. 실험과 원자료는 변경하지 않았다. 봉인 실행 코드와 저장된 JSON/CSV/PT를 읽고 CPU 집계만 수행했다. 신규 GPU 실행, job 제출, target 재계산은 없다.

## 결론

B100과 현재 탐색 상한을 유지하면 **온라인 controller의 배치당 계산 차원과 상태 tensor shape는 고정**된다. 따라서 배치 수 증가에 따른 전체 controller 비용의 1차 근사는 선형이다. 그러나 native Adam 사용량, cache hit, 품질조건 통과율, pruning 완료율이 모델 상태에 따라 바뀌므로, 현재 1,000개의 시간에 5 또는 10을 곱한 값은 예산 시나리오일 뿐 5k/10k의 실행시간 예측이 아니다. 과거 전체를 주기적으로 재평가하는 비용은 별도로 더 빠르게 증가한다.

**1k에서 관측한 allocation/gate/efficacy/generalization/locality 순위를 5k/10k 성능으로 외삽할 근거는 없다.** 전체 history M은 누적되지만 온라인 Past 품질조건은 안정적인 priority 표본64개만 본다. 미사용 layer도 M이 누적되며, 과거 key를 새 모델로 refresh하지 않는다. 작은 write norm이나 낮은 W0 KL을 남은 edit capacity로 해석할 수 없다.

## 근거 파일

원격 호스트는 `codex-server4`이다. 아래 상대경로의 기준은 다음과 같다.

- `SRC` = `/data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2/source-v1`
- `RUN` = `/data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2`
- `BLUE` = `/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/blue-source`
- `REPORT` = 이 문서와 같은 디렉터리의 `report-snapshot`

실행 핵심 파일 `controller.py`, `runtime.py`, `native.py`, `metrics.py`, `runner.py`, `technical.py`는 completed-review worktree와 source-v1 사이에서 동일함을 확인했다.

## 1. 현재 runner는 5k/10k continuation을 지원하지 않는다

`SRC/project/run_scripts/sequential_local_z_allocation/runtime.py:43`은 입력을 1,000개만 읽는다. `runner.py:36`의 loop는 B1–B10, `67`의 full-seen/Dev 관측은 B5/B10, `79`의 후보별 P/N 관측은 B1/B5/B10으로 고정되어 있다. 완료 표기와 1,000개 scope도 `runner.py:35,107–112`에 고정되어 있다.

따라서 loop 상한만 바꾸면 5k/10k 실험이 되지 않는다. 입력/manifest/lock, batch loop, observer 일정, retention window, 완료 scope를 함께 새 버전으로 정해야 한다. 현재 결과에는 실제 W/M checkpoint가 없고 프로세스 내부 상태만 이어졌으므로, 종료된 1k chain을 파일에서 정확히 이어 실행할 수 없다(`runner.py:97–112`). 새로운 cold 5k/10k 실행 또는 처음부터 계획된 동일 프로세스 장기 실행이 필요하다. 본 감사에서는 실행을 제안·제출하지 않았다.

## 2. History M: 메모리는 고정되지만 보호 통계는 달라진다

physical L4–L8 각각의 M/P는 `(1, 14336, 14336)` FP32이다. M0는0이고 P는 고정한다(`runtime.py:68–75`). selected endpoint에서 현재 B100 key를 다시 추출해 다섯 층 모두에 정확히 한 번 다음을 적용한다.

`M_l(t) = M_l(t−1) + K_l(W_selected,t) K_l(W_selected,t)^T`

근거: `runtime.py:252–264`, 원래 native `BLUE/AlphaEdit/AlphaEdit_main.py:237–239`. gate0/zeroAdam/미선택 layer도 history에서 빠지지 않으며 gate로 가중하지 않는다.

| 고정 tensor | 한 층 | 다섯 층 |
|---|---:|---:|
| M 또는 P | 822,083,584 bytes = 784 MiB | 3.828125 GiB |
| P와 M 합계 | 1.53125 GiB | 7.65625 GiB |
| 선택 가능한 down_proj W의 CPU FP32 사본 | 224 MiB | 1.09375 GiB |

이는 모델 전체, W0 사본, 후보 weight snapshot, old/new M clone의 일시적 동시 보유, teacher mmap/page cache, GPU solve 임시 tensor를 제외한 구조상 크기다. 전체 peak RAM/GPU 메모리 예측으로 사용하면 안 된다.

M의 행/열은 요청 수와 함께 늘어나지 않는다. 각 batch는 같은 크기의 Gram matrix를 더한다. 이 데이터에서 history key는 층당100개이므로 정확한 산술에서 rank 증가의 상한은 batch당100이다. 이는 용량 추정치가 아니며, 실제 rank/conditioning/FP32 오차는 별도 측정이 필요하다.

오래 실행하면 M 값과 스펙트럼은 누적된다. native solve가 `P(KK^T + M) + I`를 사용하므로(`AlphaEdit_main.py:131–133`) 과거 key 방향에 대한 write 제약이 달라진다. 같은 RHS와 geometry를 고정한 직관으로는 과거 방향의 write가 더 억제될 수 있지만, 실제 K/target/gate는 매번 달라지므로 efficacy·norm이 단조 감소한다고 결론낼 수 없다. 또한 과거 key는 당시 selected 모델에서 계산한 것이며 이후 representation drift에 맞춰 재인코딩하지 않는다. **L8을 사용하지 않았다는 이유로 L8이 history 없는 새 용량으로 남는 것도 아니다.**

## 3. Cache와 메모리의 생명주기

매 batch 새 Controller가 `fit_cache`, `score_cache`, `gate_cache`, `reduction_cache`를 생성한다(`controller.py:163–168`; `runner.py:50–55`). cache key에는 namespace, physical layer, 실제 prefix state, history가 들어간다(`controller.py:267`). 다음 batch는 M/입력/entry가 바뀌므로 이전 fit을 재사용하는 전역 cache가 아니다.

runner는 batch 끝에 controller/backend/selected를 해제한다(`runner.py:104–105`). Runtime의 CPU tensor interning pool은 `WeakValueDictionary`이며 동일 tensor 내용은 RAM에서 공유한다(`runtime.py:92–111`). 따라서 후보 snapshot 수는 배치 예산에 주로 지배되고, horizon에 비례해 모든 과거 W/M을 쌓는 구조가 아니다. 반면 received 목록, receipt/ledger, target/key/loss 진단 파일과 평가 결과는 계속 누적된다. 일부 Python loop local이 이전 observer 후보를 보유할 수 있어 순간 보유량을 최소 tensor합과 동일시해서는 안 된다.

## 4. Past64가 실제로 보호하는 범위

`SRC/project/run_scripts/local_z_adaptive_allocation/policy.py:23–32`는 이미 받은 fact의 latest-active ordinal만 남기고 현재 overwrite fact를 제외한 뒤, 고정 SHA priority가 가장 작은64개를 택한다. 성공률 기반 선택은 없지만, 매번 독립적인 새64개 표본을 뽑는 방식도 아니다. 낮은 priority 항목은 여러 batch에서 반복될 수 있다.

여섯 arm의 B1–B10 `past64.json`을 모두 읽어 비교했다. **60개 파일의 arm별 표본 일정은 모두 동일**하다.

| 관측 | 실제 값 |
|---|---:|
| B2–B10 sample slots | 576 |
| Past guard에 한 번 이상 들어온 서로 다른 요청 | 185 |
| 어느 시점에든 eligible이었던 요청 union | 900 |
| 그중 한 번도 Past guard에 들어오지 않은 요청 | 715 |
| union coverage | 20.56% |
| B10 시점 eligible | 899 |
| B10 보호 표본 비중 | 64/899 = 7.119% |

B2–B10 새로 관측된 unique 수는 `64, 29, 25, 18, 10, 10, 11, 10, 8`이다. 다섯 요청은9회 모두 포함됐다. 마지막 B10에 들어온100개는 그 뒤 batch가 없어 Past로 검증될 기회 자체가 없다.

중복/overwrite가 적다고 가정하면 마지막 batch의 표본 비중은 5k에서 대략 `64/4900 = 1.31%`, 10k에서 `64/9900 = 0.65%`이다. 이것은 산술상 비중이며 실제 future 표본이나 성능을 계산한 것이 아니다. History M의 전체 key 누적과 Past64의 명시적 성공 ID/NLL 제약은 서로 다른 보호 수단이다.

Past의 품질 기준은 entry 자체가 아니라 own-entry N4 reference다(`controller.py:349–366`). N4가 이미 잃은 과거 성공은 후보가 반드시 복구해야 할 대상이 아니다. 이러한 상대 제약이 여러 batch 누적된다고 전체 과거 성공의 절대 보존 보장이 생기지 않는다.

## 5. Zero-Adam도 비용이 없는 fit이 아니다

각 native fit은 B100 target에 대해 최소100번의 initial loss forward를 한다. Adam0이면 backward/update를 생략할 뿐, target capture, 전체 key 추출, canonical layer readout, 고정 차원 dense solve, FP32 write/hash/copy, target/key/loss 파일 저장은 남는다. native 구조는 `AlphaEdit_main.py:64–140`, loss loop는 `compute_z.py:117–196`, 진단 저장은 `runtime.py:214–230`이다.

이 실행의 loss evaluation 수는 `native target 수 + Adam updates`와 일치한다: `55,200 + 215,614 = 270,814`. 계산을 Adam만으로 비교하면 C45678의 후반 zero-step 층 비용을 놓친다. compute_ks 한 호출에는 여러 model forward가 포함될 수 있으므로 key call1을 model forward1로 읽어서도 안 된다.

온라인 score1회는 현재 토큰 수/old target 가용성이 같은 본 데이터에서 B1에는178회, Past64가 있는 B2 이후에는186회 forward다. 구성은 native E의 요청별100회, canonical current new/old의14회, Past new/old의8회, S64의64회다. 근거: `metrics.py:35–71,74–120,126–138`, canonical microbatch16 및 `model_adapter.py:408–451`.

S64는 score마다 64문서 ×128 positions ×128256 vocabulary FP32 teacher를 사용한다. `TeacherStore.document`는 mmap 배열을 CPU copy한 뒤 GPU로 옮긴다(`model_adapter.py:140–146`). 논리적인 teacher tensor 처리량은 score당4,202,692,608 bytes, 약3.914 GiB다. 실제 물리 disk I/O는 OS page cache에 따라 달라지므로 이 값을 disk-read 실측으로 쓰면 안 된다. teacher storage 자체는 고정이고 재사용된다.

## 6. L4 cap 도달의 정확한 해석

`REPORT/native-target-summary.csv` 재집계 결과 L4 target6,000개 중 **5,996개가24Adam/25loss cap**, 4개는 zeroAdam이다. N4/C4는 각각1000개 전부 cap이고 F48/G48/C48/C45678은 각각999 cap +1zero다. 네 예외는 모두 B3 case8848이며 저장된 L4 native evidence에서 직접 확인했다.

각 arm L4의 평균 최종 NLL은 약0.00112–0.00115, 평균 decay는 약0.12098–0.12273이다. N4/C4의 최소 decay도0.07719/0.07742다. zero가 들어 있는 다른 네 arm B3 파일에서 zero를 제외한 최소 decay는0.09709–0.09835다. 나머지 batch의 요약 최소값을 함께 확인하면 cap에 도달한5,996개 모두 최종 decay만으로0.05를 초과한다.

native loss는 `NLL + KL + 0.5 ||delta|| / ||anchor||²`이며 stop은 총 loss<0.05다. clamp radius는0.75||anchor||다(`compute_z.py:159–189`). 최종 delta가 clamp 경계라면 decay는 `0.375 / ||anchor||`가 되어 anchor norm<7.5일 때 decay만으로 stop threshold를 넘는다. 작은 NLL인데도 cap을 채우는 관측과 부합한다.

**이는 target 예측이 부족해서24회로도 못 배웠다는 증거도,24회에 수렴했다는 증거도 아니다.** projected gradient/KKT 또는 반복 연장 비교가 없으므로 solver 반복을 늘리면 allocation/PS/locality가 어떻게 바뀔지도 이 실행으로 판정할 수 없다. 현재 allocation 비교에서 solver를 고정한 것은 효과 분리를 위한 정책이다.

## 7. 5k/10k 비용에서 외삽할 수 있는 부분

연속 arm의 batch당 공통 선언 상한은 native N4를 포함하여 target4,100, Adam12,000, loss16,100, solve41, full online score29다. history append는 모든 arm에서 층별1회, 합5회다. C4는 suffixfit0이고, 실제 search/pruning 구조 때문에 선언된 보수적인 score 상한보다 더 낮은 유효 상한을 가질 수 있다.

같은 여섯 정책/상한을 그대로 유지한다는 가정의 **호출 수 상한**은 다음처럼 배치 수에 비례한다. GPU시간/FLOPs/성능 보장은 아니다.

| 여섯 arm 합계 | 현재1k/arm,10batch | 가정5k/arm,50batch | 가정10k/arm,100batch |
|---|---:|---:|---:|
| Native targets 상한 | 89,000 | 445,000 | 890,000 |
| Adam 상한 | 408,000 | 2,040,000 | 4,080,000 |
| Loss evaluations 상한 | 497,000 | 2,485,000 | 4,970,000 |
| Native solves 상한 | 890 | 4,450 | 8,900 |
| Online score 선언 상한 | 960 | 4,800 | 9,600 |
| History appends | 300 | 1,500 | 3,000 |

실측 시간의 단순5배/10배는 동일한 request 길이, suffix 활성률, Adam 분포, cache hit, storage/GPU 자원 상황을 가정한다. 실제 장기 run에서는 M과 W가 바뀌면서 특히 suffix Adam 및 예산 소진 지점이 바뀔 수 있다. 초기에 거의 쉬던 L6–L8이 나중에 더 자주 활성화될 가능성은 있지만 현재 자료가 이를 입증하지 않는다. 그렇게 되면 같은9,600 extra Adam 아래 완료 후보 수가 줄어드는 방향의 압력은 코드상 설명할 수 있다.

`REPORT/actual-cost.csv`의 native inclusive 시간에는 target/key/solve 시간이 이미 포함된다. online scoring inclusive에도 teacher read가 포함된다. 이 내부 시간들을 다시 합산하면 이중계산이다. pure writer, state I/O, official observer forward/token 작업은 완전히 분리 계측되지 않았으며 allocated GPU hours는 utilization 실측이 아니다.

## 8. Full-seen observer는 일정에 따라 제곱형으로 커진다

현재 full-seen은 B5/B10에서 각각500/1000개를 관측하여 arm당1,500 record-evaluations이다(`runner.py:67–74`). 이 관측을 장기 run에서도5batch마다 계속한다면, Tbatch의 full-seen record-evaluations는 `500 × m(m+1)/2`, `m=T/5`이다.

| Horizon | Full-seen record-evaluations/arm | 현재 대비 |
|---|---:|---:|
| 1k, B10 | 1,500 | 1× |
| 5k, B50 | 27,500 | 18.33× |
| 10k, B100 | 105,000 | 70× |

이는 full-seen 부분만의 배수이며 전체 runtime 배수가 아니다. rewrite/paraphrase/neighborhood 및 두 target 평가의 내부 forward 수는 각 record의 prompt/token 구성에 따라 달라진다. 반대로 최종 horizon에서만 full-seen을 평가한다면 그 부분은 요청 수에 선형이다. terminal final-table/retention의 비교 목적을 유지하면서 observer 일정을 먼저 명시해야 한다. fixed64/128 teacher score와 current/candidate B100 observer는 유지되는 호출 수가 고정이면 배치 수에 선형이다.

## 9. Allocation–편집 강도–efficacy/generalization/locality의 장기 해석

gate는 서로 합1인 layer share가 아니라 **해당 prefix에서 계산한 native endpoint의 보간계수**다. a4 변경은 L5–L8 방향을 모두 바꾸며, 작은 a4를 다른 층이 어떻게 보완하는지가 정책의 핵심이다. 낮은 gate나 낮은 update norm 자체가 efficacy 손실/보존 이득을 결정하지 않는다.

현재 feasible 조건은 native rewrite E, canonical strict/pair 및 Past64를 own N4 수준으로 유지하고 S64 W0-KL을 최소화한다. paraphrase P는 observer이므로 rewrite efficacy 조건을 지켰다는 사실만으로 generalization을 보장하지 않는다. S64 locality 이득이 official N/Dev128이나 훨씬 긴 horizon에서도 유지된다는 보장도 없다. 같은 S64를 더 오래 반복 선택에 사용하면 적응적 과적합 가능성도 별도 검증 대상이다.

M의 누적, 고정 Past64의 비중 감소, 과거 key drift, native solver의 clamp/decay, 다층 후보가 더 많은 suffix 비용을 요구하는 구조가 함께 장기 결과를 좌우한다. 따라서 후속 판단은 terminal R/P/N, W0-success-conditioned retention, at-write→later forgetting, old-edit 구간, support/실질 target activation, pruning 완료율과 실측 비용을 함께 봐야 한다. 현재1k 관측은 이러한 장기 결과의 대체물이 아니다.
