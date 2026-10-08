# Qwen PRICE hparam / Tier 1–2 — CPU 준비, 미제출

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

## 실제 blocker와 남은 일

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
