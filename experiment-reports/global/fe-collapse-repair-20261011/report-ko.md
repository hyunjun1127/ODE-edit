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
