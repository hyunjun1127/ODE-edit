# JLZ v5 A/B 제출 및 초기 기술 검산

본 문서는 초기 인계이며 **2k 실험 완료·방법 우월성 보고가 아니다**.

## 실행 결속

- 사용자 nonce: `ODEEDIT-USER-GH-SH4-JLZ-V5-CANCEL-V4-IMPLEMENT-2K-20261002-R1`.
- 정본 main: `ba2deac2474087f8bb8b7b2b0d8848dd4097b16c`.
- 실제 execution source: `fd2082e4720aa971dc64b00a133fce1d8e5e7d93` / tree `0756344e7515fcaca74900fc2b4c0197b3f2e28c`.
- source 게시 main: `31a8e45f`. 실행 archive와 현재 보고 source는 별도로 결속한다.
- lock SHA256: `7a41c239761ee0d03c79aa6070230edaf83f3032a048cce1dea7a94a8a972904`.
- config SHA256: `0e5016e6751bce0802838fc668d2d53b6d356fa878f0a54a74331b2477981403`.
- archive SHA256: `a8767ce23ada7a4c370b9cd8f654c19eac9afe2e16a8d83ee3791fec15076496`.
- Local attempt: `/data/janghj/ODE-edit/local/jlz-writer-coupled-v5/20261002-v1/attempt-r1/`.

| 역할 | 실제 job | 연결 |
|---|---:|---|
| 공통 실제 모델 qualification | 57373 | 별도 고정 후보 검산 |
| qualification CPU 수집기 | 57374 | afterany:57373 |
| A: η=0 | 57378 | afterok:57373; pilot→timing→cold main |
| B: η=1 | 57379 | afterok:57373; A와 독립 |
| 최종 CPU 수집기 | 57380 | afterany:57373:57378:57379 |

각 GPU job 1GPU/8CPU/60416MiB, project/task cap2, exportNONE/Requeue0.
두 lane을 모두 등록하고 실제 held owner/argv/source/resource/dependency 검사 후 release했다.
타 job 변경0. A/B 각각 pilot BS2×2, timing B100 후보3회(첫1 warmup),
main coldW0/zeroH/빈memory BS100×20. main 총40commit, ≤1000후보/960backward.
2026-10-02 09:12:38 UTC에 대표 **A main B1→B2 실제 연결을 확인**했다.
그 직후 SH4의 능동 scheduler/log/result 조회와 대기를 중단했다.
`monitoring_active=false`, `automatic_resume=false`이며 등록된 프로그램은 자연 진행한다.
B의 초기 연결 및 양 arm의 20batch 완료는 **NOT_OBSERVED**다.

## 대표 A main 초기 연결과 첫100 관측

B1 후보25/backward24, prox 수락22/거절2; 마지막 accepted weight를 정확히 commit했다.
다섯 history append와 native memory 0→100 admission을 확인했다. B1은 reference0이다.
`B1.after == initial.state == W01.observer.state == B2.entry.state`가 일치하며,
observer의 W/H/memory 비변이 및 자신의 committed state에서 B2 geometry/teacher
entry 준비를 확인했다. 이 확인은 A의 대표 초기 경로에만 해당한다.

| A W01 현재100 기반 panel | Preference | true NLL | new NLL | TF token-micro | TF prompt-macro | TF strict |
|---|---:|---:|---:|---:|---:|---:|
| R | 98/100 | 9.908715 | 0.385732 | 97/101 | 0.960 | 96/100 |
| P | 166/200 | 7.388737 | 2.920359 | 94/202 | 0.465 | 92/200 |
| N | 884/1000 | 5.375049 | 11.292731 | 196/1030 | 0.181 | 167/1000 |

R/P desired target는 new, N은 true다. Preference는 target 평균 NLL 비교이며
tie는 실패로 센다. TF token-micro/prompt-macro/strict는 teacher-forced token 지표이고
자유생성 정확도가 아니다. 이 표는 저장된 W01 summary의 초기 수치 인계이며,
전체 raw 독립 완료리뷰나 최종 R2000/P4000/N20000 결과가 아니다.
미확인 B 결과·후속 endpoint·paired 최종 효과는 NOT_MEASURED로 둔다.

마지막 scheduler snapshot은 초기 연결 확인보다 이른 **09:09:40 UTC**다:
A57378 RUNNING, B57379 PENDING(Resources), collector57380 PENDING(Dependency).
두 arm은 A 완료에 대한 직렬 의존성 없이 등록되어 있다. 이 과거 snapshot을
모니터링 중단 후의 현재 상태라고 주장하지 않으며 이후 재조회하지 않았다.

## 실제 기술 검산 (성능 판정 아님)

| 고정 비교 | loss 절대차 | 최대 layer×request gradient 상대차 |
|---|---:|---:|
| zero: direct/cache ↔ dense/full | 0 | 3.10049e-8 |
| nonzero: direct/cache ↔ dense/full | 0 | 2.78234e-8 |
| nonzero: dense/full MB1 ↔ MB4 | 9.11969e-8 | 3.26630e-6 |

direct route가 qualification되었다. ε는 검증된 동일후보 dense MB 차이에서
`1e-7/request`로 사전 봉인했고 실제 B를 곱해 SUM 단위로 사용한다.
다섯 층 geometry scaled residual은 약1.46e-14–1.64e-14, jitter0이다.
이 작은 BS2 검산을 모든 후속 상태의 수치 동등성으로 확대하지 않는다.

A pilot B1의 accepted-weight exactcopy 후 실제 forward 검산:
key 최대절대차0, native NLL 최대절대차8.04663e-7.
B1: 후보25/backward24/수락23/거절1, B2:25/24/20/4.
이 수락·거절은 prox 후보 판정이며 요청 획득 성공률이 아니다.
B의 pilot 및 main 최종 값은 아직 NOT_OBSERVED/NOT_MEASURED이다.

CPU 회귀54개 PASS, 독립 bounded source audit 수행.
검토에서 찾은 FP32 H 비대칭 누적, graph 수명, 프로필계수/adapter guard,
후보·commit 비교, source/qualification 해시, dependency parser 문제를
실제 제출 전에 수정했다. CPU toy를 실제 GPU PASS로 부풀리지 않았다.

## 재사용·비용·보존 및 한계

정본14개 SHA/size와 CSV2000행 전체필드/평가21시점을 검산했다.
모델·입력·context·C0 17자산은 prior fullSHA+현재 stat로 재사용했다.
W0 기존26,000 paired rows를 token/state identity와 결속했으며 신규 W0 전체평가0.
역사적 actual package-file SHA와 layout bitwise equivalence는 NOT_ESTABLISHED다.
과거 version pins와 현 runtime source hash는 별도로 남겼다.

Qualification 자체 측정52.134초, peak GPU37,924,750,336B /
host RSS34,721,280KiB. Parent allocation과 stage timer는 같은 값으로 간주하거나
중복 합산하지 않는다. 실행 중 arm의 최종 GPU 비용/2k 결과는 미확정이다.
7일 wall은 요청 상한이지 ETA가 아니다.

09:09:40 UTC parent accounting snapshot: qualification57373 58 GPU초,
A57378 당시1072 GPU초(진행 중), B57379 0, CPU collectors GPU0이다.
합계1130 GPU초는 그 시점까지만의 값이며 final cost가 아니다. batch/extern step을
재합산하지 않았다. A B1 writer timer792.924초는 부분 stage timer이고 allocation에
더하지 않는다. Timing은 B100 후보3회 종료 후 cold main으로 전환했으나 본 인계는
관측한 첫 warmup29.280초/첫 측정29.843초만 기록한다. 세번째 수치와 B 비용은
인계용 신규 조회로 채우지 않았다. Peak 수치도 qualification 범위이지 main 상한의
실측 인증이 아니다.

v4 jobs57292/57293/57294는 GH가 취소했고 중복취소하지 않았다.
v4 source/raw/완료batch/평가/log는 KEEP, SIGTERM rollback은 NOT_VERIFIED.
v5는 v4 edited W/H/memory에서 이어 쓰지 않는다. 다른 STOP task는 그대로다.

새 baseline0, checkpoint/full delta/factor/resume bundle 저장0.
exact_resume=NOT_AVAILABLE. native 문장만 memory에 쓰며 official P/N 및
observer 결과는 writer·memory 선정으로 되돌리지 않는다.
원 raw/tensor/prompt/fullstdout은 local 보존/Git0.
NO_BROADCAST_NOT_REQUIRED.

[검산 수치 CSV](qualification.csv), [A 첫100 수치 CSV](A-W01-current.csv),
[제출·qualification 결속](../../../../audits/servers/server4/jlz-writer-coupled-v5-bs100x20-20261002-v1/submission-qualification.json),
[초기 연결·중단 receipt](../../../../audits/servers/server4/jlz-writer-coupled-v5-bs100x20-20261002-v1/initial-handoff.json).

ACK의 IMPLEMENTING_NOT_SUBMITTED와 이전 preflight/제출 snapshot은 당시 사실로
보존하고 이번 초기 인계로 연결한다. 실행 source/archive는 수정하지 않았다.
문서 링크/JSON/CSV/Markdown 표 구조는 정적 검산한다. Markdown renderer가 없어
실제 렌더 검사는 NOT_VERIFIED다. 완료 후 상세 CPU 리뷰는 다음 사용자 recall에서만 수행한다.
