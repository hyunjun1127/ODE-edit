# V14 B1 완료 결과 및 저장 산출물 리뷰

상태: **B1_COMPLETE / CPU collector COMPLETED / 저장 raw 재집계 검산 완료**.
사용자 recall: `odeedit_jlz_v14_s4_B1 까지 리뷰해서 산출물과 코드 모두 main에 push해`.
실행 승인 nonce: `ODEEDIT-USER-GH-SH4-JLZ-V14-NATIVE-WRITER-B1-20261005-R1`.
리뷰 시점: 2026-10-05 KST. 이번 recall에서 새 GPU·model forward·평가·fit·제출·취소는 0회다.

## 실행 범위와 provenance

- GPU job 58391 `odeedit_jlz_v14_s4_B1`: COMPLETED, ExitCode 0:0.
- CPU collector 58392: COMPLETED, ExitCode 0:0. 아래 과거 등록 당시 PENDING 기록과 구분한다.
- cold W0/H0, fixed10k first100, Llama3-8B-Instruct revision
  `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, L4–L8.
- V14_RD fit 1회·commit 1회, 후보 25개·update 24개.
  request evaluation 2500, request update participation 2400. 마지막 후보 ID=24/ordinal=25.
- history는 final rewrite-only native mean key로 5개 층 각각 1회, 층별 100 occurrence를 누적했다.
- B2/추가 full-B fit/순차 BS2 pilot/baseline/sweep/checkpoint/자동 retry는 없다.
- 실행 commit `2ab04d0b4d339573ee54fd85c7240d090a01e2b1`, lock SHA256
  `4891b4e21686628e71463d3c07b1c4697e2cc1270f9f44724f866a8e50d92874`.
- source closure 175파일의 size/SHA를 검산했다. main의 기존 V14 code 15파일은 실행 source와
  byte-exact이다. 이번에 추가한 CPU 요약기·테스트는 실행된 생산 알고리즘 변경이 아니다.
- ordered first100 SHA256 `6ce11aa6e5118a6209d2d3c0c193f32675e7223fbbf1de5d925d2cdf38db672a`.
  native 700행, observer 1300행. model/C0 대형 자산은 기존 full SHA + 현재 size/inode/mtime로
  결속했으며 이번 리뷰에서 다시 다운로드하거나 대형 재해시하지 않았다.

## W0와 B1 평가

Preference는 R/P `new_nll < true_nll`, N `true_nll < new_nll`; 동률은 실패다.
TF strict는 teacher-forced 모든 desired token 정답이며 free generation이 아니다.
R/P desired=NEW, N desired=TRUE. 저장 NLL/token 행을 별도 stdlib CPU reducer로 재집계했다.

| endpoint | RS | PS | NS | R TF strict | P TF strict | N TF strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W0 | 5/100 (5.0%) | 20/200 (10.0%) | 886/1000 (88.6%) | 0/100 | 0/200 | 171/1000 (17.1%) |
| V14_RD B1 | 100/100 (100.0%) | 181/200 (90.5%) | 882/1000 (88.2%) | 100/100 | 127/200 (63.5%) | 166/1000 (16.6%) |

V14의 desired token micro는 R 101/101=100%, P 129/202=63.8614%, N 195/1030=18.9320%.
prompt macro는 각각 100%, 63.5%, 18.05%다. true/new NLL과 분모는 `metrics.json`에 있다.
`margin_true_minus_new=true_nll-new_nll` 부호를 모든 원행에서 검산했다.

W0→B1 N preference: retained 878, lost 8, gained 4 (분모 1000, 총 변화 −0.4pp).
N TF strict: retained 164, lost 7, gained 2 (분모 1000, 총 변화 −0.5pp).
R/P preference lost는 0/0, gained는 95/161이다. 원 paired ID와 token identity를 검산했다.

## 실현 진단

`directional_ratio=<action,R>/||R||²`, `norm_ratio=||action||/||R||`,
`relative_error=||action-R||/||R||`. 아래 값은 owner-layer 또는 context-layer의 비율 중앙값이며
모든 벡터를 합친 단일 실현률이 아니다. planned R=0이면 비율은 null로 유지한다.

| actual action 범위 | 유효/전체 | directional median | norm-ratio median | cosine median | relative-error median |
| --- | ---: | ---: | ---: | ---: | ---: |
| native mean key | 410/500 | 0.685054 | 0.709166 | 0.974065 | 0.449725 |
| canonical | 410/500 | 0.659954 | 0.701111 | 0.980804 | 0.453488 |
| rewrite context | 2460/3000 | 0.681744 | 0.726630 | 0.936759 | 0.486206 |
| KL context | 410/500 | 0.655806 | 0.700921 | 0.977925 | 0.471509 |

native mean의 directional mean=1.394970, max=49.267981; canonical min=−15.897814.
전체 mean/max/min/RMS와 층별·ideal/effective/actual 구분은 `realization-summary.json/.csv`에
보존했다. norm ratio만으로 exact 실현을 주장하지 않는다.

zero planned owner-layer는 90/500. native mean의 zero-target leakage/entry anchor 중앙값
0.00430711, 최대 0.02137605이고, canonical은 중앙값 0.00426008, 최대 0.02595624다.
비율 null을 0이나 실패점수로 대체하지 않았다.

canonical planned/direct share 평균은 L4 0.588275/0.574149, L5 0.054803/0.058900,
L6 0.136025/0.139791, L7 0.184207/0.185560, L8 0.036690/0.041600.
share L1 중앙값=0.067265, 최대=0.230011 (100요청).
ideal Q 합=123.927130944880, effective FP32 Q 합=123.927130937568.

terminal native masked NLL 평균=0.001411469, actual full-native NLL 평균=0.001321310.
같은 600 context의 actual−masked 차이 평균=−0.000090159, RMS=0.000803098.
masked/actual current‖entry KL 평균=0.126124487/0.126774703.
층별 masked/actual prestate gap RMS는 `writer-layers.csv`에 기록했다.
전체 entry net-displacement 및 inherited/direct/final-gap 벡터 분해는 원자료에 없으므로
`NOT_RECORDED`; local direct action을 이 값들로 재명명하지 않았다.

## 후보와 기술 검산

- final J mean=0.071377561, 범위 0.055941431–0.125562753; J<.05인 요청은 0/100.
  common stop 대신 승인된 25후보/24update 예산을 모두 사용했다. 미수렴을 품질 gate로 처리하지 않았다.
- shared projection은 request-update 2400개 모두 적용. 기록된 최대 FP32 budget 초과는
  5.96046448e-8로 봉인 tolerance 1e-6 이내다. terminal 추가 backward/update 없음,
  gradient=null/gradient_measured=false/`NO_BACKWARD_TERMINAL`을 검산했다.
- actual pretrained B1의 일부 행 B=1/2/3 fixed candidate 검산을 읽었다.
  디렉터리 B2/B3는 입력 shape이며 순차 batch2/3 또는 추가 fit이 아니다.
- dense total-R gradient RMS 최대 6.67659758e-8, singleton microbatch 비교 최대
  2.04307235e-7. native full-hook gradient 오차 0, loss 최대 오차 0.
  각 reference RMS에 봉인 `1e-6+1e-3*reference_RMS`를 적용했고 모든 positive check가 통과했다.
- same-A native dense solve 비교 relative error 최대 4.86714732e-14.
  stop-solve negative control은 B=2의 L4/L5에서 tolerance를 초과했고,
  B=1 zero candidate 및 B=3에서는 초과하지 않았다. negative control 전체 FAIL로 과장하지 않았다.
- terminal native key 최대 오차=0; ideal→effective cast action 최대 오차=3.07277218e-8,
  effective→actual local action 최대 오차=1.85374314e-6. 원 elementwise tolerance 검사도 통과했다.
- 마지막 평가 weight SHA=commit weight SHA; 요청 RHS SHA, cold→commit→observer W/H hash,
  final-key history 1회 누적, fault-injection RAM rollback probe, observer 비변이를 대조했다.
- code owner audit: `R=a*u`, requested-u norm analytic gradient 1회, SUM F coupled adjoint,
  whole-B K/P solve와 upper key refresh, 임시 v adjoint bridge 1회, 동기 stop/projection,
  마지막 evaluated payload exact copy, rewrite-only H를 확인했다.
  forward의 no_grad/terminal solve metadata와 별개로 reverse의 solve/P/key adjoint를 검산했다.
- 기존 CPU reference8·tiny-model production8 PASS는 scope-limited 과거 증거다.
  이번 CPU publication 테스트 6개 PASS. 새 GPU validation은 수행하지 않았다.
  이번 리뷰는 owner audit이며 별도 독립 reviewer를 사용하지 않았다.
  저장된 검산 범위에서 기술 불일치는 발견되지 않았으나 모든 입력/상태의 보편적 무결성 주장도 아니다.

## 기존 V13과의 수치 비교

first100/native packing/profile/model assets/evaluator row+token identity/runtime source/W0 raw가
일치함을 local 저장 자료로 검산했다. **V13은 virtual planner, V14는 writer-aware planner이며
fit 자체가 다르다. 동일 fitted D나 단독 writer 변경 비교가 아니다.** V13 job 재조회·재실행은 없다.

| endpoint | RS | PS | NS |
| --- | ---: | ---: | ---: |
| V13 RT | 99.0% | 88.5% | 87.8% |
| V13 RD | 89.0% | 73.5% | 88.6% |
| V13 MT | 100.0% | 94.0% | 87.9% |
| V13 MD | 100.0% | 94.0% | 87.9% |
| V13 CD | 100.0% | 95.0% | 87.5% |
| V14 RD | 100.0% | 90.5% | 88.2% |

V14−V13 RD 산술 차이는 RS +11.0pp, PS +17.0pp, NS −0.4pp.
V13 RD native-mean directional 중앙값 0.635557, V14 0.685054 (서로 다른 fitted plan).
기존 AlphaEdit 등과 matched V14 B1 raw 비교는 `NOT_AVAILABLE`; 새 baseline은 실행하지 않았다.

## 비용과 게시 범위

GPU parent allocation=2439초=0.6775 GPUh (1 GPU, parent 1회 계상), CPU collector GPU0.
fit=2038.636초, commit/H/실현 측정=39.531초, 전체 runner=2433.562초.
최대 allocated VRAM=48,950,077,440 bytes=45.588GiB, 최대 RSS=34,717,548KiB=33.109GiB.
GPU는 NVIDIA RTX PRO 6000 Blackwell Server Edition; 요청 wall 24h는 실측 ETA가 아니다.
builder/masked/reverse-replay/optimizer 부분 timer 합은 `fit-statistics.json`에 있으며
모든 wall 구간을 완전히 분리한 비용표로 해석하지 않는다.
원 collector `cost.json`의 CPU RUNNING snapshot은 보존하고 최종 두 job 상태는
`results-review/scheduler-terminal.json`에 따로 기록했다.

게시: 전체 기존 V14 code, CPU 요약기/테스트, 완료 보고서, collector 소형 출력 6개 exact copy,
실현/배분/후보/component gradient/writer CSV와 JSON, source/input/raw inventory 및 review receipt.
`artifact-manifest.json`은 게시 파일 SHA/size를 기록한다.
local KEEP: `/data/janghj/ODE-edit/local/jlz-v14-native-writer-aware-b1/20261005-v1/attempt-r1/`.
원 prompt/tensor/fullstdout/대형 raw/model/C0/복원등가 payload는 Git에 올리지 않았다.
NO_BROADCAST_NOT_REQUIRED: 같은 서버 소형 source/report/manifest Git 게시이며 대형 원자료 전송 없음.
monitoring_active=false / automatic_resume=false / 새 실험 권한 생성 없음.

## 과거 등록 시점 기록 (아래 PENDING은 현재 상태가 아님)

# V14 B1 등록 사실 보고

상태: SUBMITTED_RELEASED_RESOURCE_PENDING. 실제 GPU 검증·B1 결과는 NOT_OBSERVED/NOT_MEASURED.

- GPU: 58391 (`odeedit_jlz_v14_s4_B1`), PENDING(Resources).
- CPU collector: 58392, PENDING(Dependency), afterany:58391.
- GPU 제출 의존성: afterany:58381. V13 GPU의 terminal로 충족됐으며 V13 source/job/result는 변경하지 않았다.
- 실행 source: `2ab04d0b4d339573ee54fd85c7240d090a01e2b1`.
- lock SHA256: `4891b4e21686628e71463d3c07b1c4697e2cc1270f9f44724f866a8e50d92874`.
- GPU1 / CPU8 / 59392MiB / 24h; collector GPU0 / CPU8 / 24576MiB / 4h.
  export NONE, Requeue0. requested wall은 ETA가 아니다.
- 모든 job의 owner/name/fullargv/script/source/lock/resources/dependency를 held 상태에서
  검증한 뒤 release했다. 중복 제출·기존 job 변경 없음.

승인 nonce: ODEEDIT-USER-GH-SH4-JLZ-V14-NATIVE-WRITER-B1-20261005-R1.
정본: plans/global/2026-10-05-jlz-v14-native-writer-aware/.

cold W0/H0 first100, L4–L8 V14_RD fit 1회·commit 1회, W0/B1 R100/P200/N1000 평가만
실행한다. B2, 추가 full-B fit, 별도 BS2 순차 pilot, baseline, sweep, checkpoint는 없다.

정본 11파일 SHA/정독, native token/lookup 700행, observer identity 1300행 검산 완료.
CPU reference 8개 및 tiny-model 생산 회귀 8개 PASS. 실제 pretrained Llama GPU PASS로
간주하지 않는다. owner audit이며 독립 reviewer는 사용하지 않았다.

구현은 requested R=a*u, context별 actual-v native loss, whole-B causal solve VJP,
공통 stop, EfficiencyAdam, requested norm analytic gradient 1회, 마지막 evaluated
materialized weight exact commit과 final native H 누적을 분리한다.

local 자료: /data/janghj/ODE-edit/local/jlz-v14-native-writer-aware-b1/20261005-v1/.
sealed 실행: 위 경로의 attempt-r1/source 및 config.json/execution.lock.json.
CPU 최종 보고 예정: attempt-r1/cpu-report/report-ko.md (아직 결과 없음).
NO_BROADCAST_NOT_REQUIRED: source/compact receipt만 공유하고 raw/tensor/prompt는 local 보존.
V13 결과 비교는 PENDING_COMPARISON이며 기존 paused scientific monitoring을 재개하지 않는다.

등록 후 한정 initial resource snapshot에서 agent monitoring을 종료했다.
monitoring_active=false / automatic_resume=false. 봉인 B1/collector는 자동 retry 없이 자연 진행한다.
