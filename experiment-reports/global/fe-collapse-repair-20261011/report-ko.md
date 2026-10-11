# FE Qwen/GPT-J 붕괴 조사와 수치 수리

최신 사용자 요청: FE의 Qwen/GPT-J 붕괴 원인을 찾고 해결한다. CAKE는 이 변경의 대상이 아니다.
아래 완료 결과는 기존 실행의 원시 산출물을 확인한 값이다. 새 수리본의 2K 검증은 진행 중이며,
기존 결과를 새 성능으로 바꾸어 표시하지 않는다.

## 완료 산출물과 논문 비교

| 실행 | CF Score / Eff / Gen / Loc | zsRE Eff / Gen / Loc |
| --- | --- | --- |
| Qwen, author W0-fixed, 63217/63218 | 89.93 / 99.10 / 92.78 / 80.06 | 96.43 / 86.74 / 40.32 |
| GPT-J, author W0-fixed, 63153/63154 | **49.80 / 48.95 / 51.53 / 49.01** | 98.32 / 91.70 / 28.51 |
| DOW-KE의 GPT-J FE 재현 | 80.70 / 97.40 / 88.45 / 64.10 | 98.62 / 90.81 / 27.92 |

원본 근거: `audits/servers/server1/fe-original-w0-2k-20261011/table-rows.json`의
각 run/COMPLETE/all-seen-20 SHA와 원시 평가 JSON. 최신 Qwen 두 조건과 GPT-J zsRE는
과거의 붕괴 상태가 아니다. 현재 확실한 잔여 붕괴는 GPT-J CF다.

[DOW-KE Table 1과 실험 절](https://arxiv.org/html/2608.16932v1)은 100건씩 20회
sequential 편집, AlphaEdit 평가를 명시하고 FE 행을 자체 재현으로 표시한다. Qwen은
이 논문의 실험 모델이 아니다. 공개된 재현 코드가 없으므로 미공개 FE 설정을 추정하여
논문의 정확한 실행이라고 주장하지 않는다.
[FE 원저자 코드](https://github.com/jugechengzi/FE), commit
`478134dfb24b43f4e18b47e8500893ce3f9cc50f`의 README/논문 기본 실험은 bulk 2K다.
코드 자체는 W0에서 만든 고정 target을 슬라이스하고 persistent `cache_c`를 사용하는
sequential batch도 지원한다. history가 누락된 실행은 아니다.

## GPT-J: target 크기가 아니라 ridge 역산에서 폭증

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
W0 target 정책과 history 가중치는 바꾸지 않았다.

이는 기록된 수치 정규화이며 저자의 미공개 exact C0를 복원했다는 뜻이 아니다.
기존 저자 원본 코드만 사용했다는 설명에는 이 보정의 존재를 반드시 덧붙여야 한다.

## Qwen의 이전 실행

62531/62532의 author-hparam 변형은 매 배치 진입 모델에서 target을 재계산하는
FP32 경로였다. CF B12→B15에서 L5 residual은47.39→265.94→1,750.72→8,426.36으로
증가하지만 L4 residual은45–48 수준을 유지했다. GPT-J B3의 공분산 역산 폭증과
관측 양상이 다르며 Qwen C0 L4–L8은 모두 Cholesky를 통과했다.

최신 정상 실행은 원저자 W0 고정 target/BF16 경로다. 과거 변형과의 차이는 target
참조 모델·정밀도·context가 함께 포함되어 있으므로 별도의 동일조건 A/B 없이
그중 하나만을 Qwen의 단독 원인으로 단정하지 않는다. 현재의 baseline 경로는
정상 완료된 W0-fixed runner로 유지하며, batch-entry-target 변형을 같은 FE baseline으로
혼합하지 않는다. 진행 중인 Llama의 별도 author-hparam 변형은 수정하지 않았다.

## 검증 상태

- 코드/기존 FE history/author 설정/재개 관련 CPU 테스트52개 통과.
- 실제 C0 검사: GPT-J 6층, Qwen 5층. 원 자산 쓰기0.
- 실제 pretrained L3 B1–B3 비교 완료: 원본, zero-clip, storage-resolution 보정.
- 전 층을 함께 편집하는 GPT-J CF 2K CPU 검증 진행 중. W5/10/15/20 평가 및 checkpoint 저장.
- GPU 검증용 초안 job63361은 실행 전 PENDING 상태에서 취소했다. zero-clip이 실패하는
  것을 CPU에서 확인했기 때문이며 GPU 사용0이다. 수리본의 GPU 확인은 별도 새 실행으로 등록한다.
- 기존 GPU 실험의 source/config/가중치·기존 FE 결과는 보존했다. 2K 성능 회복은 아직 미확인이다.

로컬 상세 증거: `/mnt/raid5/janghj/ODE-edit/local/fe-collapse-repair-20261011/`.
