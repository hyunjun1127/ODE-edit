# FE Qwen/GPT-J 붕괴 조사와 수치 수리

최신 사용자 요청: FE의 Qwen/GPT-J 붕괴 원인을 찾고 해결한다. CAKE는 이 변경의 대상이 아니다.
**사용자 정정(2026-10-11 10:25 KST): 현재 FE는 수정 EasyEdit 구현 + author hparams + 누적 history이며,
현재 배치 진입 모델에서 그 배치의 z만 계산한다. 미래 edit의 W0 target 전체 선계산은 사용하지 않는다.**
아래 author W0-fixed 결과와 CPU 대조는 별도 경로의 원인 후보 조사이며 현재 baseline의 회복 증거가 아니다.
초기 조사에서 이 경로를 현재 기준으로 바꾼 것은 잘못이었다. Qwen/GPT-J의 현재 기준 재검증이 남아 있다.

## 현재 sequential 기준에서 확인한 것

GPT-J61927은 현재 기준과 같은 수정 EasyEdit FE, author hparams(clamp.75,25steps,lr.5), FP32, 배치 진입 z, 누적 history다.
CF W5 Score/Eff/Gen/Loc=87.89/98.40/92.90/75.72였으나 W10 Score50.76, W20 Score50.18로 붕괴했다.
L3 C0 SHA는 위 W0-fixed 실험과 같은 `baee4037…`다. 실제 sequential 로그에서도 다음 역산 폭증을 확인했다.

| 배치 | L3 z 잔차 평균 norm | L3 update norm |
| --- | ---: | ---: |
| B2 | 35.2695 | 65.9980 |
| B3 | 37.3585 | 8,187.7012 |
| B7 | 38.5402 | 32,351.3276 |
| B10 | 39.1335 | 3,170,809.0134 |

따라서 공분산 결함은 참고 W0-fixed 경로에만 해당하는 발견이 아니다. 현재 경로에서 수치 보정의 성능 회복을 검증한다.
원 로그: `local/official-baselines/server1/memit-fe-history-three-model-2k/registration-r1/logs/cf-gptj-memit-fe-history-61927.out`.
배치별 추출: `audits/global/fe-collapse-repair-20261011/gptj-sequential-61927-norms.json`.

Qwen62531의 L4 weight norm은128.61에서517.08로 누적 증가했다. L5 잔차는B12부터47.39→265.94→1750.72로 증가했다.
Qwen L4 C0 최소 고유값은3.35836e-8, Frobenius norm1057.3269이며, L5 최소 고유값은6.34088e-5다.
L4는 양의 정부호여도 작은 방향의 역산에 민감할 수 있다. 이것만으로 붕괴 원인이나 보정의 정당성을 확정하지 않는다.
Qwen의 원 설정 진단에서 실제 key/target/잔차, write/history key 차이, 붕괴 직전 checkpoint를 확보한다.

main `6209019e`의 현재 경로 수정은 다음과 같다.

- 원저자 hparams를 유지하고 GPT-J 출처를 profile에 추가했다(GPT-J 기존 hparams 변경0).
- C0 검증/명시적 수리/derived SHA 재개 검증을 `fe_author_history.py`에 연결했다.
- 미래 target cache 정책을 거부하고, 관측 모드에서 fit 요청이 현재100건 및 동일한 배치 진입 가중치인지 검사한다.
- 각100건 재개 checkpoint 외에500/1000/1500/2000 및 지정한 진단 경계를 독립 보존한다.
- CPU 테스트68개 통과. GPU 결과와 구분한다.

실제 등록: **63373 GPT-J CF 수치 보정**, **63374 Qwen CF 원 설정 진단**.
둘 다 source `6209019e`, current-batch-entry z, author hparams, history, FP32, 동일한 기존 context를 사용한다.
server1 cap2 lane은63296→63373→63298 및63297→63374→63299다. 등록 당시PENDING이며 GPU 성능 회복은 미확인이다.
[실제 제출·검산 영수증](../../../audits/global/fe-collapse-repair-20261011/sequential-submission.json).

## 완료 산출물과 논문 비교

| 실행 | CF Score / Eff / Gen / Loc | zsRE Eff / Gen / Loc |
| --- | --- | --- |
| Qwen, author W0-fixed, 63217/63218 | 89.93 / 99.10 / 92.78 / 80.06 | 96.43 / 86.74 / 40.32 |
| GPT-J, author W0-fixed, 63153/63154 | **49.80 / 48.95 / 51.53 / 49.01** | 98.32 / 91.70 / 28.51 |
| DOW-KE의 GPT-J FE 재현 | 80.70 / 97.40 / 88.45 / 64.10 | 98.62 / 90.81 / 27.92 |

원본 근거: `audits/servers/server1/fe-original-w0-2k-20261011/table-rows.json`의
각 run/COMPLETE/all-seen-20 SHA와 원시 평가 JSON. 이 표는 W0-fixed 참고 결과다.
Qwen W0-fixed의 정상 수치는 현재 batch-entry baseline의 붕괴를 해결하지 않는다.

[DOW-KE Table 1과 실험 절](https://arxiv.org/html/2608.16932v1)은 100건씩 20회
sequential 편집, AlphaEdit 평가를 명시하고 FE 행을 자체 재현으로 표시한다. Qwen은
이 논문의 실험 모델이 아니다. 공개된 재현 코드가 없으므로 미공개 FE 설정을 추정하여
논문의 정확한 실행이라고 주장하지 않는다.
[FE 원저자 코드](https://github.com/jugechengzi/FE), commit
`478134dfb24b43f4e18b47e8500893ce3f9cc50f`의 README/논문 기본 실험은 bulk 2K다.
코드 자체는 W0에서 만든 고정 target을 슬라이스하고 persistent `cache_c`를 사용하는
sequential batch도 지원한다. history가 누락된 실행은 아니다.

## GPT-J 공분산 결함: W0-fixed 경로에서 확인한 원인 후보

63153의 원 로그를 배치·층별로 추출했다.

| 배치 | L3 target residual 평균 norm | L3 업데이트 norm |
| --- | ---: | ---: |
| B1 | 36.0017 | 70.2333 |
| B2 | 35.1569 | 65.4848 |
| B3 | 37.3039 | **64,549.9843** |
| B5 | 85.0725 | 820,581.8394 |

B1/B2 current Eff는 100%, B3부터 성능이 무너졌다. 같은 C0를 쓰는 zsRE에서는
그 배치의 key가 달라 동일한 폭증이 관측되지 않았다. 따라서 정상 zsRE 결과가 C0의
수치적 건전성을 증명하지 않는다.

실제 L3 자산:
`gpt-j-6b/wikipedia_stats/transformer.h.3.mlp.fc_out_float32_mom2_100000.npz`,
SHA256 `baee403704e2ddcc4c2a669064d29107273f49219a221ae447f361a76208a67f`.
count=54,924,275, dimension=16,384. 실제 runner와 동일하게 FP32 sum/count 후 FP64로
승격해 검사했다. L3 Cholesky info=13797; L4–L8은 모두0이다.

L3의 음의 고유값은 약 `-2.07531579e-6`, `-8.93828623e-7`, `-1.59093511e-8`이다.
고유쌍 잔차는 약1e-15 수준이다. 비중심 2차 모멘트는 수학적으로 양의 준정부호여야
하지만 이 저장 자산은 그렇지 않다. FP64 LU는 이러한 부정부호 행렬에서도 매우 작은
선형계 잔차를 내며 큰 업데이트를 반환한다. 기존 finite 검사로는 잡히지 않았다.
캐시 생성 당시의 누산/TF32/반올림 중 무엇이 음의 값을 만들었는지는 생성 로그 없이
확정하지 않는다.

실제 pretrained GPT-J, 같은 CF B1–B3/context/W0 target/history를 사용한 CPU L3
분리 재현에서도 B3 norm이5,254로 폭증했다. GPU 원본과 절대값이 다른 것은 이
불안정한 역산이 backend 차이에도 민감하다는 관측이며, CPU/GPU bit parity 주장이 아니다.

## 수정

`official/baselines/fe_covariance.py`에서 건강한 C0는 byte 그대로 반환한다.
명시적인 `project_roundoff` 정책일 때만 잘못된 C0의 분해능 이하 방향을 보정한다.
기준은 성능으로 고른 hparam이 아니라
`max(eps_storage, dimension * eps64) * ||C||_F`다. 이 캐시에서는8.01737356e-6이며,
6개 방향이 바뀌고 상대 Frobenius 변화는2.74076207e-7이다.
저장 정밀도로 설명할 수 있는 범위를 넘는 음의 고유값은 자동 수리하지 않는다.

단순히 음의 고유값을0으로 자르는 안도 실제 CPU 입력으로 확인했으나 B1부터
업데이트가41,588이 되었다. 거의 특이한 방향을 그대로 역산하므로 채택하지 않았다.
저장 분해능을 반영한 보정의 L3 업데이트는 B1/B2/B3 각각
**17.74 / 17.51 / 84.00**이며 세 선형계 모두 Cholesky를 통과했다.

원본 NPZ는 보존한다. 보정본은 각 run 아래 따로 저장하고 SHA/정책/고유값/변경량을
기록한다. 재개 시 그 보정본을 그대로 검증·복원한다. 각 배치의 ridge 시스템도
양의 정부호 검사를 통과해야 기존 저자 LU를 호출하도록 했다. 실패하면 해당 층에
업데이트를 기록하지 않는다. 건강한 시스템의 solve 산식, FE clamp/step/lr/decay,
이 CPU 대조의 W0 target과 history 가중치는 바꾸지 않았다. 현재 sequential 기준의 성능 검증은 별도로 필요하다.

이는 기록된 수치 정규화이며 저자의 미공개 exact C0를 복원했다는 뜻이 아니다.
기존 저자 원본 코드만 사용했다는 설명에는 이 보정의 존재를 반드시 덧붙여야 한다.

## Qwen의 이전 실행

62531/62532의 author-hparam 변형은 매 배치 진입 모델에서 target을 재계산하는
FP32 경로였다. CF B12→B15에서 L5 residual은47.39→265.94→1,750.72→8,426.36으로
증가하지만 L4 residual은45–48 수준을 유지했다. GPT-J B3의 공분산 역산 폭증과
관측 양상이 다르며 Qwen C0 L4–L8은 모두 Cholesky를 통과했다.

최신 정상 실행은 원저자 W0 고정 target/BF16 경로다. 과거 변형과의 차이는 target
참조 모델·정밀도·context가 함께 포함되어 있으므로 별도의 동일조건 A/B 없이
그중 하나만을 Qwen의 단독 원인으로 단정하지 않는다. **현재 baseline은 batch-entry 경로**다.
이 경로에서 Qwen/GPT-J 붕괴를 조사·수리하며 W0-fixed 성공으로 대체하지 않는다.
Llama author-hparam + history63357은 같은 baseline 계약이며 수정하지 않았다.

## 검증 상태

- 코드/기존 FE history/author 설정/재개 관련 CPU 테스트52개 통과.
- 실제 C0 검사: GPT-J 6층, Qwen 5층. 원 자산 쓰기0.
- 실제 pretrained L3 B1–B3 비교 완료: 원본, zero-clip, storage-resolution 보정.
- W0-fixed 전 층 CPU 진단은 사용자 정정 뒤 B3 편집 도중 중단했다. B1/B2 current100만 평가했으며 현재 baseline 결과가 아니다.
- GPU 초안63361은 zero-clip 실패로 실행 전 취소했다. GPU 사용0이다.
- W0-fixed 수리 검증63368도 사용자 정정 뒤 **PENDING에서 취소, GPU 사용0**이다.
  63298의 자원 의존성은63296으로 복구했다. 기존 CAKE 설정·산출물은 바꾸지 않았고 cap2를 재검산했다.
  [최신 취소·복구 영수증](../../../audits/global/fe-collapse-repair-20261011/sequential-scope-correction.json).
- 현재 sequential FE의 Qwen/GPT-J 수리 및2K 검증은 진행 중이다.
- 기존 GPU 실험의 source/config/가중치·기존 FE 결과는 보존했다. 2K 성능 회복은 아직 미확인이다.

로컬 상세 증거: `/mnt/raid5/janghj/ODE-edit/local/fe-collapse-repair-20261011/`.

## 추가 진단: 저자 writer + 현재 배치 z

사용자 요청에 따라 저자 FE `478134df`의 `batch_edit` 본문을 그대로 호출하는
별도 진단 backend를 추가했다. 우리 `compute_z`로 매 배치 진입 모델에서 해당100건만
계산하며, 저자 writer에 넘기는 target 열과 sample index도 현재100건으로 한정한다.
이 경로는 미래2K의 W0 target을 읽거나 생성하지 않는다.

저자 writer의 RHS solve, FP64 key Gram을 FP32 history에 누적하는 방식,
가중치 단일 cast 대입을 보존한다. FE hparams·FP32·context·입력 순서·C0 정책은
대응하는63373/63374와 같다. Qwen은 strict, GPT-J는 명시적 roundoff 보정이다.
배치 실패 시 selected weights와 history를 복원하고, 배치별 z/K/current output을 기록한다.
새 경로는 비교 진단이며 기존 baseline의 정의를 바꾸지 않는다.

관련 CPU 테스트71개 통과. 두 모델의 저자 모듈과 YAML/hparam 일치를 실제 import로 확인했다.
사용자는 이 진단에 한해 cap2 예외와 **server1 사용**을 승인했다. 기존 cap2 lane과
CAKE 실행은 유지하며 진단용 한 개 추가 lane을 준비한다. server3에는 신규 job을 제출하지 않았다.

실제 제출: **63375 Qwen RUNNING**, **63376 GPT-J PENDING afterany:63375**.
source `6d063860`, 기존 정상2개 lane 변경0, 진단용 한 개 lane 추가로 최대3 GPU.
저자 모듈의 실제 `batch_edit` 작은 CPU 입력에서도 업데이트·history가 예상 산식과
bit-identical임을 확인했다. GPU2K 완료나 붕괴 해소는 아직 주장하지 않는다.
[진단 제출 영수증](../../../audits/global/fe-collapse-repair-20261011/author-sequential-submission.json).
