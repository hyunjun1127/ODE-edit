# Qwen PRICE hparam / Tier 1–2 — 등록 및 초기 온라인 연결

## 최신 실행 상태 (2026-10-09 03:21 KST)

사용자 `gpu 2개 써`를 반영해 각 cold arm GPU1, 두 arm 동시 실행으로 등록했다.
실행 source는 `f50a5c90f3f396fa4c7ce7acb6fac6686e2488b1`이며 이후 보고 commit과 구분한다.

| 실제 job | 단계 | GPU / CPU / RAM MiB | dependency | 관측 상태 |
|---|---|---|---|---|
| 61673 | 기본값 B1 W/H 재현 → held-out W0 500 | 1 / 8 / 59392 | 없음 | RUNNING |
| 61674 | Tier1 Q0–Q7/조건부 → 선정 → Tier2 | 2 / 16 / 118784 | afterok:61673,afterany:61618 | PENDING (Dependency) |

두 job 모두 exact held owner/Command/argv/source/resources/dependency 검사 후 release했다.
현재 own allocation은 기존 61618 GPU1과 warmup GPU1로 합계2다. sweep는 둘 종료 후
GPU2 allocation에서 독립 arm 프로세스를 한 GPU씩 실행한다. canonical2/local3 중
더 엄격한2를 적용했다. 기존 61618/타 작업 변경0, 중단한 61598 재개0, Tier3 제출0.
처음 resource helper의 `*` 이름 검사는 다른 사용자까지 포함해6으로 과대계상했다.
own queue 전체를 별도 검산하고 실제 own 이름 범위로 helper를 재검사해 cap2 PASS했다.
첫 helper 거절 때 sbatch 호출0; 실제 deliberate registration은 위 두 ID 한 번뿐이다.

공유 SH1 transport 파일을 수정/복제하지 않고 task-local strict schema adapter를 연결했다.
held-out metadata/resolved config/validation/W0_first500을 정확히 검증하며, 원 shared
sidecar·privacy·job identity·bounded readback 경로를 재사용한다. 공통 helper 자체에
기능이 배포됐다는 주장은 아니다. 새로운 fake-SDK/selection CPU 5 tests PASS;
기존 official63/source158 검산과 구분한다. 실행 archive와 11,112 source members,
입력/런처/resolved config/선정 규칙을 봉인했다. frozen archive hotpatch0.

smoke 실제 W&B startup 원격 identity readback 확인:
[server4-QWEN_smoke_Q0-execution-r1-job61673](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/5965697611e94a32).
job/source/config/model/schema/validation 역할과 immutable identity receipt가 결속됐다.
이는 **online 시작 확인**이며 B1 W/H 재현 PASS나 Tier1/2 완료가 아니다.
smoke는 validation이므로 아래 tuning view에서 제외된다. 실제 tuning arm은 각자
새 run ID로 전용 view에 기록되며 아직 그 결과/remote metric delivery는 NOT_OBSERVED다.

Tier2는 state continuation 증명 대신 허용된 cold B1–B5 refit을 사용한다. Q0와 선정
후보 비용을 포함하고 noCP를 유지한다. 미래 archive 정책은 새 CP가 없어
NOT_APPLICABLE_NO_CHECKPOINT이며 기존 CP 이동/삭제0다. smoke 불일치면 sweep 시작0.
12GiB storage reserve와 per-batch guard를 적용했고 등록 당시 free 약53.8GiB였다.
원 raw/세부 receipt는 ignored `local/qwen-price-hparam-tier2-20261009/execution-r1/`에 보존한다.
NO_BROADCAST_NOT_REQUIRED: same-host 자산 및 compact Git source/report만 전달한다.

아래는 준비 단계의 역사 기록이다. 과거 미제출/schema blocker는 위 구현·등록으로 갱신됐다.

## 최신 사용자 요청: 별도 W&B tuning page

`Qwen Tuning — Held-out 500` saved view를 실제 생성하고 원격 readback으로
이름·12 sections·task/cohort 필터를 확인했다.
[전용 view](https://wandb.ai/wkdguswns2256/layer%20allocation?nw=oeqlkjvq8nc).
프로젝트는 사용자 지정 `wkdguswns2256/layer allocation`을 유지한다.
`task_id=qwen-price-hparam-tier2-20261009`와 `cohort_role=heldout_tuning`을 동시에
요구하므로 기존 first2K/baseline/validation run을 가져오지 않는다.
Current pre/post, W5 all-seen500, matched W0_first500 및 별도 fit 축을 준비했다.

새 run·업로드 metric 0, 기존 run/다른 saved view 수정0, GPU/Slurm0.
아직 실제 tuning 결과가 없어 빈 그래프이며 0점/성공으로 대체하지 않는다.
전용 page 생성은 공통 logger의 held-out schema 허용과 별개다. SH1 소유 helper를
복제하거나 `W0_first500`을 first2000으로 바꾸지 않았으므로 기존 schema 미지원은
여전히 남는다. 실제 GPU smoke/Tier1/Tier2 및 실시간 업로드 완료가 아니다.
기존 SDK/UI 환경을 재사용했고 과학 환경이나 credential은 변경하지 않았다.
ignored receipt: `local/qwen-price-hparam-tier2-20261009/tuning-view-receipt.json`.
운영 source: `project/run_scripts/qwen_price_hparam_tier2/tuning_view.py`.

수신 nonce: USER-SH4-QWEN-PRICE-HPARAM-TIER2-20261009-R1 및 동일 R1-SCOPE.
branch: codex/server4-qwen-price-hparam-tier2-20261009.
main 직접 수정/게시 없음. 이전 실행과 신규 sweep을 분리한다.

## 기존 실행 중단

사용자 직접 중단에 따라 owner/name/Command/node를 재확인한 **61598만 취소**했다.
Slurm accounting은 CANCELLED by 1025, 종료 시각 2026-10-09 02:19:07이다.
즉시 queue는 자원 정리 중 COMPLETING이었다. 61618 historical Current replay와
61418 Llama OURS는 RUNNING 그대로 보존했다. 원 source/raw 재개·덮어쓰기·삭제 없음.

## 신규 구현과 검증

ours 전용 writer/price 및 Q0–Q7 JSON, immutable 단일 resolver/canonical SHA,
native preset/정의역 검증을 만들었다. controller·optimizer·subject·price와
GPT-J 경로에 연결했다. M1 existing-prefix anchor와 active-mask guard를 유지했다.
해석 설정을 entry-price/fit receipt에 기록한다. baseline hparams/profiles/locks는
수정하지 않았고 기존 PRICE JSON은 bytes를 보존한 채 deprecated 문서만 추가했다.

CPU 새 7 tests PASS. frozen source와 세 모델 기본값의 controller, projection,
Adam, norm, subject loss/adjoint가 bit-exact다. 전체 official/tests는 **63 PASS**.
official verifier는 source158 / Python228 / 외부 task import0 PASS.
이는 owner CPU 검산이며 독립 review나 실제 GPU/61598 W/H 재현 PASS가 아니다.
official/SOURCES.json은 변경 ours 12개와 새 resolver 1개만 SHA/provenance 갱신했다.
공유 tracking helper는 수정하지 않았다.

held-out CF [2000:2500] 500건의 native pack 5개를 CPU에서 준비했다.
eval first2K와 case/request identity overlap0, 기존/official native token/target/
lookup/pack identity 동일. lookup0는 B3 owner10·84 두 건이다.
원문 및 case-level token/ID 목록은 ignored local에만 보존한다.
기존 61598 실제 B1 writer receipt SHA
428c9a11c1a0ce2ed35e37667ad8042666a50d36f499638c104935477ed0ecdc를 smoke 기준으로 결속했다.
아직 새 W/H 재현·held-out W0·Tier1·Tier2 GPU 계산은 하지 않았다.

## 사전 선택 규칙

Q7 extreme 참고 제외는 matched W0 대비 B1 NS 손실 **5pp 초과**로 결과 관측 전 고정했다.
일반 진출은 원 지시대로 NS 손실≤1.5pp와 held-out Q0 대비 PS 개선이다.
Q4 shared_active<80%이면 lambda_N=0만 변경하는 조건부 arm을 적용한다.
Tier3 추천은 W5 RS≥99%, NS 손실≤1.2pp, numeric/projection 실패0 중 PS 최대이며
이번 task는 Tier3를 제출하지 않는다. continuation 증명 전에는 cold B1–B5 재실행
방식을 사용할 수 있으나 아직 어느 Tier2 run도 시작하지 않았다.

## 당시 blocker와 남은 일 (등록 전 역사 기록)

공통 W&B strict schema는 필수 cohort_role/tier/slice/resolved config/hash,
validation 역할, W0_first500을 거부한다. 500건 W0를 first2000으로 재명명하거나
tuning을 eval first2K 비교 행에 섞지 않는다. 전용 view도 아직 생성하지 않았다.

사용자 지시에 따라 repository app-server direct 정책과 기존 peer client로 GH
session 01a04939-8873-7673-8dca-4c7fc5e31af0에 전달을 시도했다.
등록 server1 경로의 SSH 인증이 거부되어 WebSocket/initialize 이전에 종료:
**COMMUNICATION_HOLD, 미전달**. Git inbox/message fallback은 하지 않았다.
SH1 공통 schema 입력 또는 승인된 연결의 인증 복구가 필요하다.

남은 작업은 공통 schema 결속, 자체 runner/collector/Slurm source freeze 및 검산,
기본 B1 GPU W/H 재현, held-out W0, Tier1, paired 선정, Tier2 및 비교 보고다.
GPU smoke가 불일치하면 sweep은 시작하지 않는다. 새 job IDs는 **[]**이며
이는 Slurm PENDING도, Tier2 완료도 아니다. 현 canonical cap2와 localcap3 중
더 엄격한2를 적용해 retained allocations를 포함한 새 dependency를 검산해야 한다.

새 checkpoint/전송/삭제0. 미래 job archive 정책은 기존 CP에 소급하지 않는다.
NO_BROADCAST_NOT_REQUIRED: same-host 자산, compact source/receipt만 branch 게시.
