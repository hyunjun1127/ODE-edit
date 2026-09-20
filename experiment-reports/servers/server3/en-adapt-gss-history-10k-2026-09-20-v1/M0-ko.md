# SH3 M0 — RES / GSS_REC independent fixed10k

ACK nonce=ODEEDIT-GH-SH3-EN-ADAPT-GSS-HISTORY-R2-20260920-R1.
ACK nonce=ODEEDIT-GH-SH3-GSS-TWOARMS-CAP2-20260920-R1.

최신 정본 main `7216cc96c0436fcbd1d3b7cba2c69c09d613bd93`, 최초 six-authority main `b9a52005df2c1d06d035d6bb6404a55b8c763eeb`를 확보했다. 정본 6개와 최신 envelope/derived contract/cells를 FULL_READ했으며 SHA/size 일치 receipt를 audit에 남겼다. 원 정본과 CRLF cells는 변경하지 않았다. 부모 정본은 원 실행 준비의 FULL_READ 이력을 재사용한다.

등록 receipt 기준 이전 all-active lifelong 신규 job 0, 본 GSS task 신규 job 0, 제외 uniform GSS job 0이다. 취소 0, source/raw 삭제 0. 기존 B300 51290 제출/완료보고는 독립 역사이며 결과를 재조회하지 않았다. 새 실행은 RES/GSS_REC 각 fresh W0/zero M4, B100×100 독립 job 두 개다. 200 native/reference-gradient batch state, candidate 최대400이 계획값이다. 각 own B2 state에서 같은 process RAM으로 B100까지 진행하며 interjob W/M/teacher 공유나 prefix restart는 없다.

부모 actual execution `5d452221288f3b924e1737578f11aaa654594422`의 native/optimized z hook/controller/geometry/strict JSON 수리 경로를 재사용한다. 완료 분석 `abd08ee1dde24ca975aace49c925a4ba00211004`는 실행 source와 구분한다. 새 구현은 `project/run_scripts/en_adapt_gss_history/`에 격리한다. 신규 version ledger/cold teacher store, signed NLL factor sketch/GSS_REC, observer/reducer 및 독립100-batch runner/submitter가 구현 대상이다. 세 독립 bounded worker는 ledger, factor/sketch, observer/report만 담당하며 SH3가 통합·제출을 소유한다.

S3 기존 Python3.12.3/torch2.9.1+cu128/transformers4.44.2, Llama revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, FP32/eager/TF32-off, L4/nativeL2=1/context/P/C0를 재사용한다. fixed10k SHA `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`, ordered root `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`, 전체 records[0:10000] 순서 및 canonical token identity를 CPU로 검산했다. R512/Dev128 2560 files/101519959223B의 기존 full SHA/size/shape 검산과 현재 stable identity를 결속했다. 신규 reference/model 전송·복제·생성 0이다.

RES는 전체 active ledger bottomhash512/uniform KL, GSS_REC는 current WN에서 signed NLL-gradient sketch backward pruning512 + h5.12 recency KL을 쓴다. feature와 at-write KL objective를 구분하고, pool≤512에서는 NLL backward/sketch를 건너뛴다. overflow≤612에서 microbatch1 graph당 NLL/KL VJP 각각1회, selected≤512만 후보 평가한다. R512 전 문서/valid position과 원 controller epsilon.05/2candidate 수식을 유지한다. 이 두 arm 비교는 선택과 recency 가중치가 동시에 달라 인과 기여를 각각 분리하지 못한다.

RS/PS/NS와 TF token-micro/prompt-macro/strict, true/new/desired NLL·margin·lost/gained를 같은 evaluator forward에서 산출한다. every-batch current/first100/fixedhash128 panel, 정해진 full-history 시점, entry→native→selected 및 at-write→현재 분석을 계획했다. official P/N은 선택 seal 이후 observer 전용이다. request-cluster bootstrap10000/seed20260920은 terminal CPU reducer로 준비한다.

준비 시점 합산 resource 계획: 각1GPU/8CPU/121856MiB, project/taskcap2, 합산16CPU/243712MiB. node96CPU/512000MiB, MemAvailable 약349GiB(비독점 관측). 새 경로 추정 peak100GiB/job, 요청상한119GiB/job이며 실제측정이 아니다. 기존 B300 MaxRSS69.375GiB와 고정 reference/cache/SVD/M/G·bounded hot history 구성으로 산정했다.

두 cold teacher/cache store를 모두 포함한 보수적 payload/metadata는 23857506288B, scalar output8GiB, atomic temporary2GiB, safety8GiB로 총43184859120B를 요구한다. 당시 free49155330048B로 headroom5970470928B; 공간은 독점 예약되지 않았다. 입력별 input/target token 총88213/10163, target 최대2token을 그대로 보존하며 truncation0이다. teacher는 FP32 무손실 원값, 관측 JSON은 lossless gzip이다.

과거 B300 core653–673초/state를 바탕으로 core 약18.1–18.7h/arm, history/observer/I/O/두 job contention을 포함한 예상28–72h/arm(합산56–144GPUh)으로 계획한다. walltime은7일/job, partition MaxTime30일이다. 추정과 실측을 구분하며 GPUh hardcap을 임의 상속하지 않는다.

save_checkpoints=false: edited W/M/delta/optimizer/resume disk0, exact crash-resume NOT_AVAILABLE. cold teacher/key만 입력 자산으로 저장한다. 최종 source/input/resource freeze와 CPU 통합 회귀가 남아 있으며 아직 actual new GPU path PASS 또는 제출을 주장하지 않는다. 부모 precision NOT_ESTABLISHED와 신규 sketch 진단 미측정은 유지한다. 이후 제출 전 admission→두 held exact 검사→release까지만 수행하고 scheduler/log/result/initial/B2/terminal 조회 없이 MONITORING_PAUSED_AWAITING_USER로 인계한다.
