# JLZ v7 착수 보고

후속 USER 요청: [2026-10-03 main A 500-edit 완료 간단 리뷰](main-a-review-20261003-r1/report-ko.md). 아래 착수·초기 관측 기록은 당시 상태 그대로 보존한다. 후속 리뷰는 A만 확인했고 B 전체 완료 판정은 하지 않았다.

상태: INITIAL_CONFIRMED_AGENT_PAUSED. A의 실제 main B1 commit→B2 own-entry를 검산했다. A/B 모두 실행에 진입했지만 각500 완료는 관측하지 않았으며, 이후 능동 모니터링은 중단했다.

- Nonce: `ODEEDIT-USER-GH-SH4-JLZ-V7-CAUSAL-500-20261002-R1`
- 정본15개 SHA/size 및 전체 문서·CSV 결속 완료.
- CPU 설계107+4 및 생산/adapter/pipeline/memory27검사 PASS. 실제 모델 PASS 아님.
- A/B 각각 cold BS100×5, W5 평가 후 종료. noB6/noCP/신규baseline0.
- v6 중단 유지, 기존 자료 보존. 13:55:49Z owned queue0, 취소0.
- 입력·W0재사용 및 상세 한계: [preflight](../../../../audits/servers/server4/jlz-causal-writer-v7-bs100x5-20261002-v1/preflight-r1.md).

## 실제 등록

| 구간 | Job | dependency |
|---|---:|---|
| prep A: BS2 pilot/qualification | 57511 | 없음 |
| prep B: BS2 pilot/B100 5후보 | 57512 | 없음 |
| cold main A, 500 | 57513 | afterok:57511:57512 |
| cold main B, 500 | 57514 | afterok:57511:57512 |
| CPU collector | 57515 | afterany:57511:57512:57513:57514 |

실행 source `fe6758512eba37595e99244d150d115268553f01`, lock `97a03ae24674f0e7fbe115e96f0b0d930166aaa4fa80a5af29c30206b1e9f09e`.
5개 모두 held 검산 후 정상 release. 최대 GPU 동시2, 각1GPU/8CPU/60416MiB, collector GPU0/8CPU/24576MiB. 다른 job 변경0.
소형 [제출 receipt](../../../../audits/servers/server4/jlz-causal-writer-v7-bs100x5-20261002-v1/submission-r1.json)에 archive/config/launcher 결속.
원 raw와 로그는 `/data/janghj/ODE-edit/local/jlz-causal-writer-v7/20261002-v1/attempt-r1/`에 보존. Git에는 원 raw/텐서/prompt/전체 로그를 넣지 않았다.

초기 관측은 아래 별도 receipt로 결속했다. NO_BROADCAST_NOT_REQUIRED.

## 완료된 준비와 증거 한계

| 준비 | native 후보 | Adam | 결과 |
|---|---:|---:|---|
| A BS2×2 | 50 | 48 | 2 commit/5층 H/자체 memory 연결 |
| B BS2×2 | 50 | 48 | 2 commit/5층 H/자체 memory 연결 |
| B cold B100 | 5 | 4 | 후보5 backward, 다섯 번째 step 없음 |

두 prep scheduler COMPLETED/exit0. A261+B560=821 할당 GPU초; utilization이나 순수 fit 시간과 다르다. B100 후 상태는 main으로 이전하지 않았다.
실측 B100 GPU peak 48,426,445,824B (약45.1GiB), prep host peak 34,718,064KiB (약33.1GiB); wall 요청은 ETA가 아니다.
Actual BS2 native full/cache 차이0; causal direct/checkpoint 대 dense/no-checkpoint loss 차이0, 최대 gradient RMS 차이1.81e-11. 문맥 역순/MB1 최대 gradient RMS 차이6.15e-9. 이는 고정 BS2 후보 범위이며 모든 B/상태의 보편적 수치 parity 주장이 아니다.
Stop-P negative control은 의도대로 gradient가 달랐고 제3 과학 arm이 아니다. B1 objective A/B equality gate는 두지 않았다.
소형 [actual audit](../../../../audits/servers/server4/jlz-causal-writer-v7-bs100x5-20261002-v1/actual-BS2-qualification-r1.json), [준비 원자료 index](../../../../audits/servers/server4/jlz-causal-writer-v7-bs100x5-20261002-v1/completed-preparation-r1.json).
자체 source/CPU 검산이며 별도 독립 red는 미사용. Markdown 상대링크 존재 검산, 실제 renderer 미설치로 렌더 검증은 NOT_PERFORMED.

## 대표 실제 main 초기 인계

2026-10-02T14:39:33Z에 이미 완료된 불변 receipt를 검산했다. A B1 actual25후보/24Adam/100edit commit의 W/H SHA = observer 전후 SHA = B2 entry SHA. 5층 history append, native memory100 resident와 immutable anchor/teacher 보존, B2 고정 reference16 sample을 확인했다. 후보25 gradient는 null/미측정이다. B의 초기 gate는 NOT_OBSERVED로 남긴다.

| A W1 현재100 | strict NLL preference | TF token-micro | TF prompt-macro | TF strict |
|---|---:|---:|---:|---:|
| R, new 목표 | 100/100 (100%) | 101/101 (100%) | 100% | 100/100 |
| P, new 목표 | 199/200 (99.5%) | 159/202 (78.7129%) | 78.5% | 157/200 |
| N, true 목표 | 852/1000 (85.2%) | 177/1030 (17.1845%) | 16.25% | 148/1000 |

R/P는 NLLnew<NLLtrue, N은 NLLtrue<NLLnew이며 tie는 실패다. TF는 teacher forcing이며 free generation 정확도가 아니다. 모든 수치는 저장된1300 paired-prompt row를 독립 CPU 재집계했다. 현재100 결과를 최종500/전체2k나 다른 arm의 결과로 확대하지 않는다.

| A W1 현재100 | mean true NLL | mean new NLL |
|---|---:|---:|
| R | 15.41767194 | 0.00409088 |
| P | 11.61220069 | 1.00296251 |
| N | 5.60344508 | 10.99355079 |

A B1 entry/fit/commit 포함1696.735초, observer 별도53.515초. Parent 최종 할당 GPU초는 아직 NOT_MEASURED이며 이 stage timer를 utilization이나 최종비용으로 바꾸지 않는다. B100/두 pilot 신규비용821 allocated GPU초와 W0 입력 재사용을 분리한다.

[초기 인계 및35개 immutable output SHA/size](../../../../audits/servers/server4/jlz-causal-writer-v7-bs100x5-20261002-v1/initial-handoff-r1.json). 실제 initial receipt SHA `54f1b0b6e25330c4dbdf399c44d17eac89f86969a98d7166f8a9d535a23e4f4e`.
현재 main 실행 source는 여전히 `fe6758512eba37595e99244d150d115268553f01`이며 보고 publication commit과 구분한다. 이후 scheduler/log/result polling0, heartbeat/callback/자동retry/자동recall0; `monitoring_active=false`, `automatic_resume=false`. 이미 봉인된 두 runner와 collector만 W5까지 자연 진행한다. 상세 완료 리뷰는 사용자 recall 때 수행한다.
