# FE Qwen/GPT-J 붕괴 조사와 수치 수리

최신 사용자 요청: FE의 Qwen/GPT-J 붕괴 원인을 찾고 해결한다. CAKE는 이 변경의 대상이 아니다.
**사용자 정정(2026-10-11 10:25 KST): 현재 FE는 수정 EasyEdit 구현 + author hparams + 누적 history이며,
현재 배치 진입 모델에서 그 배치의 z만 계산한다. 미래 edit의 W0 target 전체 선계산은 사용하지 않는다.**
아래 author W0-fixed 결과와 CPU 대조는 별도 경로의 원인 후보 조사이며 현재 baseline의 회복 증거가 아니다.
초기 조사에서 이 경로를 현재 기준으로 바꾼 것은 잘못이었다. Qwen/GPT-J의 현재 기준 재검증이 남아 있다.

## 현재 진행 상태 (2026-10-11 15:08 KST)

공식 `main.py` + 현재 배치 z 진단을 server1에서 진행 중이다. 기존 정상 cap2에
허용된 진단용 한 개 lane을 추가했다. 후속 원인 분리 실험은 기존 CAKE 작업이
종료하면서 비는 슬롯을 사용한다. 전체 최대 동시3 GPU는 유지한다.

| 작업 | job | 의존성/관측 상태 |
| --- | --- | --- |
| Qwen CF native sequential 2K | 63385 | COMPLETED 0:0, 저자 2K 평가에서 붕괴 확인 |
| Qwen 저장 checkpoint 공통 평가 | 63387 | COMPLETED 0:0, W0 및 500/1000/1500/2000 모두 평가 |
| GPT-J CF native sequential 2K | 63386 | RUNNING, 원본 C0 대조 |
| GPT-J 저장 checkpoint 공통 평가 | 63388 | PENDING afterany:63386 |
| GPT-J L3 C0만 보정한 native 2K → 평가 | 63389 → 63390 | 63389 RUNNING, 63390 후속 PENDING |
| Qwen 현재100건의 W0-reference z → 평가 | 63391 → 63392 | 63391 RUNNING, 63392 후속 PENDING |
| GPT-J W0-reference + 보정 C0, 동일 protocol 검증 | 63425 → 63426 | PENDING, 63390·63392 평가 종료 후 시작 |

63389/63391의 자원 의존성만 조정했다. 두 작업을 hold한 상태에서 실제 Slurm 설정과
전체 사용자/server1 GPU DAG의 최대 폭3을 확인하고 hold를 모두 해제했다. 새 제출이나
source/config/output 변경은 없으며 CAKE 작업 자체도 변경하지 않았다.
[대기 의존성 전후 검산](../../../audits/global/fe-collapse-repair-20261011/native-causal-parallel-after-cake.json).

14:04 KST에는 CAKE zsRE63299의 실제 `COMPLETED 0:0`을 확인해 그 슬롯에서
Qwen63391을 먼저 시작했다. 63391을 hold한 상태에서 남아 있던63298 대기를 제거하고,
전체 최대 폭3을 검산한 뒤 release했다. source/config/output 변경은 없다.
GPT-J 보정63389도 CAKE CF63298 정상 종료 뒤14:20 KST에 시작했다.
[실제 빈 슬롯 사용 검산](../../../audits/global/fe-collapse-repair-20261011/native-qwen-released-zsre-slot.json).

후속 평가는 저장된 W0/500/1000/1500/2000 가중치를 별도 프로세스에서 읽는다.
편집은 계속 저자 `main.py`가 수행하며 실행 중인 source/config/output 변경은0이다.
학습이 실패하면 존재하는 checkpoint만 평가하고 W20 부재를 명시한다. CF Eff/Gen/Loc
공통 평가이며 FLU/CON은 포함하지 않는다. 저자 자체 W0/W20 평가도 유지한다.

Qwen W0 파일을 재로딩하여 SHA/identity, 다섯 층의 pretrained BF16 가중치 일치,
0으로 초기화된 history를 확인했다. B1의 층별 residual/weight/update 로그 수치는
이전 author W0-fixed63217과 출력 정밀도에서 모두 같다. 전체 모델의 bit parity나
2K 성능 회복을 의미하지 않는다. B2 이후는 현재 배치 모델의 z를 사용하므로 별도 검증한다.

[평가 등록 영수증](../../../audits/global/fe-collapse-repair-20261011/native-evaluation-submission.json) ·
[W0 재로딩](../../../audits/global/fe-collapse-repair-20261011/native-qwen-w0-checkpoint-validation.json) ·
[B1 출력 수치 대조](../../../audits/global/fe-collapse-repair-20261011/native-qwen-b1-displayed-parity.json).

아래는 조사·제출 이력이며 이전에 취소된 job의 상태도 원 기록으로 남겨 둔다.

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

## 추가 CPU 원인 분리

- GPT-J 원본 moment sum을 **FP64로 올린 뒤 count로 나눠도** 최소 고유값은
  -2.06795e-6이며 Cholesky13797로 실패한다. 마지막 FP32 나눗셈만의 문제가 아니다.
  [원본 자산과 FP64 normalization 검사](../../../audits/global/fe-collapse-repair-20261011/gptj-normalization-precision.json).
- 실제 pretrained Qwen의 첫 편집층 L4 이전 부분을 CPU FP32로 실행해20배치 key를
  수집했다. L4 입력은 L4 이상 편집에 의존하지 않는다. 같은 C0와 FP32 history로
  만든20개 시스템은 모두 Cholesky를 통과했다. 이는 CPU 경로 검사이며 GPU 성능이나
  중간층의 전체 동역학을 검증한 것은 아니다.
  [20배치 검사](../../../audits/global/fe-collapse-repair-20261011/qwen-l4-history-cpu.json).
- 같은 L4 key로 저자 FP64 Gram→FP32 누적과 우리 FP32 Gram 누적을 비교했다.
  B20에서 history Frobenius 차이는0.18476이지만, C0의 가장 작은16개 방향에서
  system 최소 고유값은 각각0.0007068068과0.0007068008이다. 이 관측으로는
  첫 층의 history 정밀도 차이가 Qwen 붕괴를 유발했다는 주장을 뒷받침하지 못한다.
  [제한된 부분공간 비교](../../../audits/global/fe-collapse-repair-20261011/qwen-history-precision-cpu.json).

- 실제 Qwen pretrained 하위5개 층, 동일8건/context에서 저자 helper와 우리 helper의
  L4 key 및 현재 block output은 **bit-identical**이다. CPU FP32 확인이며 누적 편집
  전체나 GPU writer 결과가 같다는 뜻은 아니다.
  [실제 helper 대조](../../../audits/global/fe-collapse-repair-20261011/qwen-native-key-parity-cpu.json).
- 저자 clone의 원본 대비 수정은 이전 W0 runner용 `precompute_z.py`,
  `z_methods/__init__.py` 두 파일뿐이다. 이번 adapter는 둘 다 import/call하지 않는다.
  `batch_edit`, key/representation helper는 원본 그대로이며 전체 실제 bytes도 봉인했다.
  [upstream 차이](../../../audits/global/fe-collapse-repair-20261011/author-sequential-source-diff.json).
- `official.tools.verify` source integrity/외부 task import 검사 PASS.

## 최신 사용자 정정: 공식 runner 자체를 사용

63375/63376은 우리 runner가 저자 batch_edit를 호출하는 조합이며, 공식 main.py에서 z 부분만 변경한 실행이 아니다.
이 구분을 명확히 하지 못한 것은 구현 범위 오류다. 63375는18분27초 뒤 W0 저장에서
INCOMPLETE_CHECKPOINT_IDENTITY로 실패했고 편집0건이다. 63376은같은오류예방을 위해2분30초에중단했다.
공통 identity 수정은 다른 작업자가 main4841f720에 반영했다. 여기에 실제 FE10필드 identity로
W0/B1 저장·재개 및 covariance/backend 불일치 거부 회귀검사와 평가 전 사전검사를 추가했다.
기존 공식 runner 자체를 실행하는 진단으로 변경한다. 저자 hparams/BF16/원 covariance/저자 context 생성/
writer/history를 유지하고, 현재 batch-entry 모델의 해당100건만 firstforward z를 계산한다.
따라서 GPT-J의 수치 보정은 이 비교에 섞지 않는다. 각500건 저장은 관측용 I/O로 구분한다.
기존 실패 stream을 그대로 사용한다. 표본 실험이 필요하면 고정된 다른seed/order를 사전선언하고
원래 실패 결과와 함께 보고한다. 성공 표본만 본표에 대체하지 않는다.
63373/63374는취소, CAKE63298/63299 의존성복구와hold해제 완료.
[실제 상태 영수증](../../../audits/global/fe-collapse-repair-20261011/native-runner-scope-correction.json).

## 공식 main.py 실제 진단 등록

- **63385 Qwen CF 2K RUNNING**, **63386 GPT-J CF 2K PENDING afterany:63385**, server1.
- 실행은 pinned 저자 checkout의 **python main.py**이며 이후 같은 main.py의 test_only 평가를 한다.
  우리 runner나 writer adapter를 호출하지 않는다. ODE patch source60eea6e2, 저자478134df.
- 원본 main.py SHA는ea83923f8cadf39895940cad332076647c87ca8cf0e1e8ffe4ef0e515e10ae16이고 실행 파일과 같다.
  batch_edit/load_cov/get_context_templates/upd_matrix_match_shape 본문도 원본과 같다.
- 저자 안내대로 FE-memit_main.py를memit_main.py로 설치한 뒤, 전체W0 cache 공급부만
  현재100건 firstforward 계산으로 교체했다. 실제 최적화 함수는 저자 algs/memit/compute_z.py다.
  그 함수와저자z_methods/compute_z.py의 차이는출력용print2줄뿐이다.
  새로운 z helper는저자bare-prompt forward replay를현재batch-entry모델에서호출한다.
- 별도 변경은관측용checkpoint 저장이다. W0/500/1000/1500/2000의가중치/H/RNG/context/case ID를보존한다.
  자동native resume는아직구현하지않았다. 이I/O추가를“z코드한곳만변경”이라고숨기지않는다.
- author YAML hparams와기본BF16을유지하고, context도저자generate_fast로생성한다.
  기존EasyEdit공분산NPZ는원본bytes그대로저자get_cov경로에연결한다. GPT-J음의고유값보정도이진단에는없다.
  모델파일전체SHA와C0전체SHA를재검증했다. nativeget_cov가기존FP32sum/count와bit-identical인것을
  QwenL4/GPT-JL3에서CPU로확인했다. 원본통계파일을새로계산하거나덮어쓰지않았다.
- native환경은torch2.9.1+cu128/transformers4.51.3/hydra1.3.2/numpy1.26.4이다.
  저자requirements의모든dependency버전이같다는주장은하지않는다. Python/Hydra는로컬호환환경이다.
- 기존CF2Kcase ID와순서를유지한다. 새seed/표본은아직실행하지않았다.
- CPU6개검사통과: 실제저자apply+writer의history/배치시점, 작은입력에서저자firstforward재생과일치,
  미래요청거부, 모델변경거부, 저장전후상태/RNG보존. 공통checkpoint관련23개검사도통과했다.
- native stdout와저자평가파일을남긴다. 우리W&B tracker를training경로에삽입하지않았다.
- 기존CAKE63298/63299를변경하지않고진단용한개추가lane만사용한다. 최대동시3GPU를held상태에서검산했다.
  Qwen은실제main.py에서모델/C0/context 로딩까지확인했다. **W20성능회복은미확인**이다.

[실행·원본대조](../../../audits/global/fe-collapse-repair-20261011/native-execution-proof.json) ·
[z공급부diff](../../../audits/global/fe-collapse-repair-20261011/native-target-supply.patch) ·
[CPU자산검증](../../../audits/global/fe-collapse-repair-20261011/native-cpu-asset-preflight.json) ·
[등록](../../../audits/global/fe-collapse-repair-20261011/native-submission.json).

## 500건 관측과 다음 원인 분리 준비 (12:02 KST)

Qwen63385 B5 checkpoint의 SHA/identity와 현재500개 case 순서를 재검증했다.
다섯 층의 가중치/H는 모두 finite이고 H는 대칭이다. 저장된 각 층의
`15000*C0 + H`는 Cholesky를 통과했다. 다음 배치의 실제 solve에는 그 배치의
key Gram이 추가되므로 이 검사는 모든 미래 solve를 보증하지 않는다.
B5 L5 residual은250으로 증가했다. 아직 공통 CF 성능은 미측정이다.
[실제 checkpoint CPU 검사](../../../audits/global/fe-collapse-repair-20261011/native-qwen-b05-prior-audit.json).

GPT-J 결함 C0의 별도 native NPZ 보정본을 준비했다. count/나머지 metadata는
보존하고,6개 분해능 이하 방향만 보정한 FP64 moment sum을 저장했다.
파일 재로딩/정규화 후에도 Cholesky를 통과하며 보정 전후 상대 변화는2.74e-7이다.
원본 NPZ나 현재63386 설정은 변경하지 않았다. 이 자산은 새 통계 수집이 아닌
명시적 수치 정규화이며 별도 GPU 대조가 필요하다.
[원본/보정본 SHA·검산](../../../audits/global/fe-collapse-repair-20261011/native-gptj-derived-prior.json).

Qwen의 원인 분리용 `initial_model_on_arrival` target 공급 옵션을 추가했다.
현재 도착한100건만 W0 가중치에서 최적화하고, 편집 중인 가중치를 복원한 뒤
원래 저자 writer/history로 전달한다. 미래 target cache는 읽거나 만들지 않는다.
이는 **현재 batch-entry 모델 기준과 다른 대조 프로토콜**이며 baseline의 자동 대체가 아니다.
실행 중인63385/63386 frozen source에는 이 변경을 넣지 않았다.
CPU12개 검사에서 현재 요청 범위, 성공/예외 시 editing state 복원,
서로 다른 reference의 checkpoint 오표기 거부를 확인했다. GPU 실행은 아직 하지 않았다.

실제 저자 `get_cov`로도 보정 NPZ를 읽어 FP64 normalization 일치와 Cholesky 통과를 확인했다.
[저자 loader 검산](../../../audits/global/fe-collapse-repair-20261011/native-gptj-derived-prior-loader.json).

## 두 원인 분리 대조 등록 (12:06 KST)

| native main.py 대조 | 편집 job | 공통 checkpoint 평가 |
| --- | --- | --- |
| GPT-J, 현재 배치 모델 z, 결함 L3 C0만 수치 보정 | 63389 PENDING | 63390 PENDING |
| Qwen, 현재100건만 W0에서 z 계산, 원 C0 | 63391 PENDING | 63392 PENDING |

source49ab6b20, 동일 저자478134df, 동일2K 순서/author hparams/BF16/history다.
Qwen 대조는 미래 target을 계산하지 않지만 **현재 모델 기준 z와는 다른 reference**다.
이 두 효과를 구분하기 위한 진단이며 baseline 결과를 덮어쓰지 않았다.
등록 당시 추가 lane은63385→63387→63386→63388→63389→63390→63391→63392였다.
13:20 KST에 후속 두 대조의 자원 의존성을 위 현재 진행 상태에 기재한 구조로 조정했다.
기존 job 변경0, held 실제 검사 후 전부 release했으며 최대 동시3 GPU다.
GPU 성능 회복은 아직 미확인이다. Qwen63385는B6에서 L5 잔차892/update1951까지 증가했다.

[동일 설정·차이의 범위](../../../audits/global/fe-collapse-repair-20261011/native-causal-preparation.json) ·
[실제 제출·cap 검산](../../../audits/global/fe-collapse-repair-20261011/native-causal-submission.json).

## Qwen L5 증폭과 target drift: CPU checkpoint 분리 검사

동일B5 checkpoint를 읽은 CPU BF16 검사다. GPU 정확도/전체모델 bit parity 결과가 아니다.
다음B6의100개 bare prompt 중 case13617(`Vinson Plateau`, subject lookup3)의 L5 norm이
W0 38.01→B5 4696.59로123.55배 증가했다. 모델 사본의 L5 가중치만 W0로 복원하면66.42,
L4만 편집 상태로 두어도66.42다. 중간층에 누적된 가중치 변경이 이 이상치 증폭에 관여한다.
다른 층을 실제 학습에서 되돌리거나 입력 sample을 제외하지 않았다.
[100건 활성값·층 분리](../../../audits/global/fe-collapse-repair-20261011/native-qwen-b05-activation-cpu.json).

이 이상치 요청과B6 첫 요청을 고정해 **저자 compute_z와 동일한 firstforward**를 CPU에서 재계산했다.
B5 모델 기준 case13617의 L4 target norm은60.68(W0 55.64)인 반면, L5 target은
**2065.19(W0 53.10)**로 커졌다. 현재 모델의 큰 활성값이 다음 target에도 반영되는 현상을 직접 관측했다.
이 요청의 현재 모델 최적화는 저자35step/clamp1 그대로이며 최종 target 확률은 약1.14e-6이다.
현재 모델 기준 target의 누적 변화가 재현 실패에 관여한다는 근거이며,
W0-reference online 대조63391의2K 회복 여부는 아직 확인하지 않았다.
[동일 요청의 저자 z 대조](../../../audits/global/fe-collapse-repair-20261011/native-qwen-target-drift-cpu.json).

## 1K checkpoint와 GPT-J 20배치 시스템 검산

Qwen63385의1K checkpoint를 다시 읽었다. 가중치/H/identity/case 순서/SHA가 정상이며,
다섯 층 모두 `15000*C0+H` Cholesky를 통과했다. L5 weight norm은220,397.67이다.
저장 손상이나 history 누락으로 인한 현상이 아니라, 큰 가중치가 실제로 누적되어 저장된 상태다.
[1K 재로딩 검산](../../../audits/global/fe-collapse-repair-20261011/native-qwen-b10-prior-audit.json).

동일한 CF2K stream SHA의 이전 W0-fixed63217과 현재63385의 마지막 z-fit 확률을 비교했다.
B1의100건 평균/중앙값은 동일하다. 현재 B6의 평균 목표 확률은0.6052, B7은0.01097,
B8은0.00001958로 감소했고 B8의100건 모두1% 미만이다. 이전W0-fixed에서는 각 배치 평균이
약0.986~0.990이며2K 중0.5 미만인 요청은 없다. 이는 **z 최적화 중 목표 확률**이며
CF Eff/Gen/Loc으로 보고하지 않는다.
[동일 입력·fit 상태 비교](../../../audits/global/fe-collapse-repair-20261011/native-qwen-fit-quality-snapshot.json).

GPT-J의 native BF16 첫 편집층 input key를 CPU에서20배치 계산하고 저자식으로 history를 누적했다.
첫 편집층 입력은 자신 및 뒤쪽 층 편집에 의존하지 않는다. 원본 `15000*C0+H+KKᵀ`는
20회 모두 Cholesky13797로 실패했고, 보정 C0는20회 모두 통과했다.
원래 저자 context와 고정 CF2K를 사용한 CPU 검사이며 CUDA 수치/2K 성능 회복 증거와는 구분한다.
[20배치 원본/보정 대조](../../../audits/global/fe-collapse-repair-20261011/native-gptj-l3-prior-stream-cpu.json).

## 1.5K checkpoint와 저장 공간 확인

Qwen B15에서도 가중치/H가 finite이고 history는 대칭이며 다섯 층의
`15000*C0+H`가 Cholesky를 통과했다. L5 weight norm은 2,064,563.40으로
증가했다. 검사 전후 checkpoint SHA는 동일하다. 아직 CF 성능 평가가 끝난 것은 아니다.
[1.5K 재로딩 검산](../../../audits/global/fe-collapse-repair-20261011/native-qwen-b15-prior-audit.json).

server1 진단 노드는 RTX A6000 48GB 장비다. W0 snapshot의 0으로 초기화된
history 영역만 sparse hole로 변환하여 약 6.68 GiB를 확보했다. 논리적인 파일 크기,
모든 byte와 SHA, 가중치/H는 그대로다. checkpoint나 다른 실험의 산출물은 삭제하지 않았다.
[저장 전후 SHA와 할당량](../../../audits/global/fe-collapse-repair-20261011/native-qwen-w0-storage-sparse.json) ·
[실제 장비와 Slurm 할당](../../../audits/global/fe-collapse-repair-20261011/native-server1-hardware.json).

## Qwen native 2K 편집 완료, 성능 평가 진행 중

63385는 같은 2,000건의 편집을 모두 마쳤다. W0/500/1000/1500/2000 checkpoint가
저장됐고 B20의 다섯 편집층 가중치는 저자 `save_model`의 최종 export와 비트 단위로
일치한다. 원래 2K case의 순서와 파일 SHA도 검사했다. B20의 가중치/H는 finite,
H는 대칭이며 다섯 층의 `15000*C0+H` 모두 Cholesky를 통과했다. 검사 중 파일 내용은
바뀌지 않았다. L5 weight norm은 246,640,023.22다.

[2K 가중치와 history 검사](../../../audits/global/fe-collapse-repair-20261011/native-qwen-b20-prior-audit.json) ·
[저자 최종 export와의 일치](../../../audits/global/fe-collapse-repair-20261011/native-qwen-b20-native-final-parity.json).

최종 배치 100건의 z-fit 목표 확률 평균은 약 2.55e-6이며 100건 모두 1% 미만이다.
이는 z 최적화 상태를 보여 주는 수치로, CF 성능 지표가 아니다. 저자 W0/W20 평가와
후속 공통 checkpoint 평가를 진행한다. 수정 대조군의 2K 회복은 아직 확인되지 않았다.
[20배치 fit 상태](../../../audits/global/fe-collapse-repair-20261011/native-qwen-fit-quality-2k.json).

## Qwen native 대조군의 공통 평가 완료

63385와63387은 모두 `COMPLETED 0:0`이다. 공통 평가의 모든 checkpoint가 있고,
각 raw JSON의 SHA를 검증했다. 아래 편집 이후 수치는 해당 시점까지의 모든 요청을
평가한 결과이며 W0만 전체 2,000건이다. 이 결과는 현재 모델에서 현재 배치 z를 계산하는
대조군이며 수정 대조군의 결과가 아니다.

| 편집 수 | CF Score | Eff (%) | Gen (%) | Loc (%) |
| --- | ---: | ---: | ---: | ---: |
| W0 (평가 2K) | 20.68 | 13.90 | 16.30 | 85.055 |
| 500 | 87.14 | 96.00 | 87.80 | 79.220 |
| 1,000 | 49.91 | 52.60 | 49.85 | 47.530 |
| 1,500 | 50.82 | 49.80 | 51.73 | 50.973 |
| 2,000 | 50.18 | 50.90 | 49.95 | 49.715 |

저자 자체 2K 평가의 Eff/Gen/Loc은49.8/48.8/48.4이며, 그 반올림 성분의 조화평균은48.993이다.
저자 평가는 BF16 logits를 그대로 log-softmax에 넣고 한 요청의 prompt들을 묶는다.
공통 평가는 선택한 logits를 FP32로 승격하고 query batch16을 사용한다. 이 구현 차이를
확인했지만 각 차이의 기여량을 독립적으로 측정한 것은 아니다. 두 평가 모두 붕괴를 보이며,
원 수치를 각각 보존하고 모든 대조군의 비교에는 동일한 공통 평가 구성을 사용한다.

따라서 공식 runner와 저자 hparams를 그대로 사용해도, 현재 모델 기준으로 z를 매 배치
재계산하는 이 변형에서는 문제가 남는다. 초기500건은 작동하고 이후 z/L5 증폭과 함께
성능이 떨어진다. 현재100건의 W0-reference 대조63391의 결과를 계속 확인한다.

[공통 평가·raw JSON SHA](../../../audits/global/fe-collapse-repair-20261011/native-qwen-common-evaluation.json) ·
[저자 원 결과·평가 차이](../../../audits/global/fe-collapse-repair-20261011/native-qwen-author-evaluation.json).

## GPT-J 시작점과 완료된 대조군의 저장 위치

63386의 W0 여섯 편집층 가중치를 지정된 pretrained GPT-J 파일과 직접 비교했다.
BF16으로 변환한 원 가중치와 모두 비트 단위로 일치하고 history는0이다. checkpoint
identity/원 자산 SHA/검사 전후 checkpoint SHA도 확인했다.
[GPT-J W0 검사](../../../audits/global/fe-collapse-repair-20261011/native-gptj-w0-checkpoint-validation.json).

후속 checkpoint 공간을 확보하기 위해, 학습63385와 평가63387이 모두 정상 종료된 뒤
Qwen 대조군의 다섯 checkpoint를 서버1 SSD의
`/var/tmp/odeedit-fe-collapse-20261011/qwen-current-checkpoints/`에 보존했다.
각 파일을 복사·fsync·SHA 검증한 후 기존 경로를 같은 byte의 파일로 연결했다.
기존 receipt의 경로와 SHA로 계속 읽을 수 있으며 가중치/H/평가 산출물은 보존돼 있다.
RAID에서 약29.90 GiB를 확보했고, GPT-J W0의0 영역도 내용 변경 없이 sparse 처리해6 GiB를 확보했다.
[저장 이동 전체 검산](../../../audits/global/fe-collapse-repair-20261011/native-qwen-control-storage-migration.json) ·
[GPT-J W0 저장 할당량](../../../audits/global/fe-collapse-repair-20261011/native-gptj-w0-storage-sparse.json).

## GPT-J 원본 C0의 실제 native 3번째 배치 폭증

63386의 첫 세 배치에서 L3 잔차는36.0017/35.2775/37.3515였으나 업데이트 norm은
70.2333/66.3780/**60,005.8906**으로 증가했다. 현재 배치 모델의 z와 저자 runner에서도
원본 공분산의 불안정성이 재현됐다. B3의 잔차 크기가 약6% 증가하는 동안 업데이트는
약900배 커졌다. 기존 CPU 공분산/20배치 선형계 검사와 함께 수치 보정 대조의 근거로 삼는다.
63389는 L3 C0만 보정한 같은 설정이며 아직 대기 중이다. 이 기록은 수정 후 회복 증거가 아니다.
[native B1–B3 원 로그 수치](../../../audits/global/fe-collapse-repair-20261011/native-gptj-b3-instability.json).

## Qwen 두 대조군의 시작 상태 일치

63391은14:04 KST에 시작했다. 기존63385와 새63391의 W0 checkpoint를 직접 읽어
다섯 편집층 가중치의 비트 일치, 모두0인 history, 같은 native hparams·context·seed·
Torch/CUDA RNG·모델 자산·2K stream을 확인했다. 차이는 명시된 target reference다.
현재100건만 W0에서 계산하는 대조이며 미래 target cache를 쓰는 실험이 아니다.
W0의0 영역은 SHA를 유지하며 sparse 처리했다. 2K 회복 여부는 실행·평가로 확인한다.
[두 시작 상태 비교](../../../audits/global/fe-collapse-repair-20261011/native-qwen-paired-w0-validation.json) ·
[W0 저장 검산](../../../audits/global/fe-collapse-repair-20261011/native-qwen-online-w0-storage-sparse.json).

## GPT-J 500건 저장 history를 포함한 선형계

63386의 B5 checkpoint를 다시 읽었다. 여섯 층 모두 가중치/H는 finite이고 H는 대칭이다.
L3 history trace는201,056.62로 실제 누적되어 있지만 `15000*C0+H`의 Cholesky info는
여전히13797이다. L4–L8은 모두0이다. L3 weight norm은350,207.89다. checkpoint의
identity, 현재500개 case 순서, 검사 전후 SHA도 검증했다. 실제 다음 배치 solve에는
그 배치의 key Gram이 추가되므로 이 검사의 범위는 저장된 prior와 history다.

검사 도구는 실제 native cache symlink와 실행 자산 SHA를 확인한다. 원본 설정에는
기존 NPZ를, 보정 설정에는 별도 derived NPZ를 사용해 검사하도록 했다.
[저장 상태의 층별 검사](../../../audits/global/fe-collapse-repair-20261011/native-gptj-b05-prior-audit.json).

## GPT-J 보정본 시작 및 같은 초기 상태 검증

63389는 기존 CAKE 두 작업이 모두 정상 종료된 뒤 시작했다. 실제 server1의 사용자
GPU 할당3/의존성 DAG 최대 폭3을 재확인했다. GPT-J63386/63389의 W0도 직접 비교해
여섯 편집층 가중치의 비트 일치, 모두0인 history, 동일 native hparams·context·seed·
Torch/CUDA RNG·모델 자산·2K stream을 확인했다. 둘 다 현재 batch-entry 모델의 z다.
보정본 첫 배치 L3 잔차는 원본과 같은36.0017이며, 업데이트 norm은70.2333에서17.7359로 줄었다.
2K 성능 회복은 계속 실행·평가로 확인한다.

[실제 세 진단의 GPU 할당](../../../audits/global/fe-collapse-repair-20261011/native-three-active.json) ·
[GPT-J 시작 상태 비교](../../../audits/global/fe-collapse-repair-20261011/native-gptj-paired-w0-validation.json) ·
[보정본 W0 저장 검산](../../../audits/global/fe-collapse-repair-20261011/native-gptj-repaired-w0-storage-sparse.json).

## 보정 효과의 초기 GPU 관측 (2K 결과와 구분)

GPT-J의 문제 지점인 B3에서 원본/보정본 L3 잔차는37.3515/37.3312로 비슷하지만,
업데이트 norm은60,005.8906/**58.9876**으로 달라졌다. 두 실행 모두 현재 배치 모델의 z,
같은 초기 상태·hparams·stream·native writer이며 C0만 다르다. 이 폭증을 C0 보정으로
제거하는 것을 실제 GPU에서 확인했다. 최종 성능 회복은2K까지 확인한다.
[B1–B3 두 실행 비교와 원 로그 prefix SHA](../../../audits/global/fe-collapse-repair-20261011/native-gptj-b3-c0-repair-comparison.json).

Qwen W0-reference 대조의 B5 checkpoint에서도 실제 가중치/H/순서/SHA와 다섯 층의
양의 정부호 prior를 확인했다. L5 weight norm은139.08로 현재 모델 기준 대조의281.47보다 작다.
B6 L5 잔차는14.1875(원 대조892), 업데이트 norm은16.279(원 대조1951.1764)다.
이 수치들은 가중치와 z의 관측이며 CF 성능으로 보고하지 않는다.
[Qwen W0-reference B5 저장 상태](../../../audits/global/fe-collapse-repair-20261011/native-qwen-online-b05-prior-audit.json).

## GPT-J의 동일 history와 보정된 500건 checkpoint

63386/63389의 B5를 직접 비교했다. 원본/보정본 L3 history는 **비트 단위로 동일**하며,
trace는 양쪽 모두201,056.61914980412다. 같은500건·순서·context·hparams도 확인했다.
L4–L8 history는 앞 층의 편집 가중치가 달라진 결과로 서로 다르지만 모두 finite·대칭·비영이다.
따라서 L3 업데이트 감소는 history를 생략하거나 약하게 적용해서 얻은 결과가 아니다.

보정본 B5의 여섯 층 `15000*C0+H`가 모두 Cholesky를 통과했다. L3 weight norm은
**140.96**이며 원본은350,207.89다. 원본/보정 checkpoint 파일의 검사 전후 SHA도 같다.
2K 성능은 별도 공통 평가가 끝난 뒤 판단한다.
[history 직접 비교](../../../audits/global/fe-collapse-repair-20261011/native-gptj-paired-b05-history.json) ·
[보정본 B5 전체 층 검사](../../../audits/global/fe-collapse-repair-20261011/native-gptj-repaired-b05-prior-audit.json).

Qwen 원본 대조의 공통 평가도 요청별 raw NLL에서 독립 재집계했다. 각 요청 안에서
prompt별 선호 비율을 평균하고 전체 요청 평균 및 조화평균 Score를 계산했으며,
W0/500/1000/1500/2000의 모든 Eff/Gen/Loc/Score가 저장된 summary와 일치한다.
W20에서는 rewrite/paraphrase/neighborhood의 목표 토큰 정답 수가 모두0임을 raw에서
확인했다. 약50의 선호 점수를 정상적인 편집 성능으로 해석하면 안 된다.
[raw 독립 재집계](../../../audits/global/fe-collapse-repair-20261011/native-qwen-independent-metrics.json).

## Qwen: 미래 요청 선계산을 제거한 1K 가중치 일치

63391의 B10 checkpoint 다섯 편집층 가중치 byte SHA가, 기존 저자 W0 선계산 참조
63217의 `batch-10.json`에 저장된 가중치 SHA와 **모두 일치**한다. 같은1,000개 case의
순서, 참조 raw SHA, 새 checkpoint identity와 검사 전후 SHA도 확인했다. 과거 실행은
매 배치 checkpoint를 덮어썼으므로 여기서는 당시 commit에 기록된 가중치 hash와
비교했으며, 과거 B10 tensor를 직접 재로딩했다는 뜻은 아니다. B20은 과거 tensor가
남아 있어 최종 가중치와 history까지 직접 비교할 예정이다.

이 비교의 의미는 target의 **기준 모델**과 **계산 시점**이 별개라는 것이다. 저자 FE의
W0-reference를 유지하되 도착한100건만 계산할 수 있다. 각 요청의 target fit은 독립이며,
writer는 현재 배치 target과 누적 history만 사용한다. 현재 모델을 참조하는 변형과는
과학적 설정이 다르므로 같은 이름으로 몰래 바꾸지 않는다. 원저자 hparams나 샘플을
변경하지 않고 선계산을 제거하는 대조로 보고하며, 최종 CF 성능은 아직 평가 전이다.
[1K 실제 가중치 SHA 대조](../../../audits/global/fe-collapse-repair-20261011/native-qwen-online-b10-precompute-parity.json).

## 두 모델에 같은 target protocol을 쓰는 후속 검증

63389는 C0만 바꿔 원인을 분리한 대조다. Qwen과 GPT-J의 최종 FE 정의를 모델마다
다르게 정하지 않도록, GPT-J에도 현재100건의 W0-reference target과 같은 보정 C0를
함께 적용하는2K 검증을 등록했다. **63425 → 63426**, source `b8144315`, identity
`3029accd3526041723d149ff3affa3f43e1578e3185a2ce1c5effbdd75ebf077`이다.
63390과63392가 모두 끝난 뒤 시작한다. 신규2개를 held 상태에서 실제 source/config/script,
GPU/CPU/RAM/노드/QoS/의존성을 확인하고 전체 server1 사용자 DAG 최대 폭3을 검사한 뒤
release했다. 기존 작업과 source/config/output 변경0, 남은 hold0, 제출 시 GPU 결과는 미관측이다.

기존 RAID checkpoint를 유지하며 새63425의 다섯 snapshot만 server1 SSD
`/var/tmp/odeedit-fe-collapse-20261011/gptj-online-w0-repaired-checkpoints/`에 저장한다.
논리 경로는 새 run의 `checkpoints/`로 연결했다. 해당 파일시스템에16 GiB를 남기는
저장 guard와 전체 예상 용량을 검산했다. 이 차이는 관측 저장 위치이며 편집 산식이나
hparams 변경이 아니다. 기존 실행의64 GiB guard는 그대로다.

원본/수정 평가 패키지의 `official/evaluation/` Python30개 파일은 byte가 동일하다.
저자 `main.py` 및 `batch_edit`, `load_cov`, context 생성, shape 변환,
`current_batch_targets` 본문도 동일하다. 명시된 C0와 target reference 외에 평가식이나
writer 변경을 섞지 않았다. 이 후속 검증도 기본 baseline 정의를 자동으로 바꾸지 않는다.
[준비 근거](../../../audits/global/fe-collapse-repair-20261011/native-combined-preparation.json) ·
[실제 제출·held 검산](../../../audits/global/fe-collapse-repair-20261011/native-combined-submission.json) ·
[평가·native 함수 동일성](../../../audits/global/fe-collapse-repair-20261011/native-causal-source-equivalence.json).
