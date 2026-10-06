# GH → server4: PRICE repair 및 Llama·Qwen 각 세 arm 2k 실행 요청

Instruction / ACK nonce: `USER-GH-SH4-JLZ-PRICE-CAP-BASE-REPAIR-2K-20261006-R1`

제안 task ID: `jlz-price-cap-base-repair-2k`

사용자 최신 원문:

> GH에게 전달해서 해당 부분 repair와 FREE075  를 제외한 run들 2k 실험 돌리는 것으로 하자. server4에 전달하라고 해. gpu cap은 2이다.

추가 사용자 원문:

> qwen도 제출하자. 동일한 3개 arm으로

발신은 연구 설계 chat `01a10dc1-4cd1-74b0-a951-3e043882782b`이며 GH를 사칭하지 않는다. 사용자가 repair와 Llama·Qwen 각각 세 run의 실행 및 GH→server4 전달을 명시적으로 승인했다. 앞선 문서의 DESIGN_ONLY / NOT_IMPLEMENTED / NOT_DISPATCHED는 당시의 수행 상태이며, 이 사용자 지시 이후의 실행을 보류하는 조건이 아니다. 이전 네 arm 제안보다 최신 두 모델 × 세 arm 실행 계약이 우선한다.

## 대상과 소유

- GH: `01a04939-8873-7673-8dca-4c7fc5e31af0`, CWD `/mnt/raid5/janghj/ODE-edit`.
- SH4: `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`, CWD `/data/janghj/ODE-edit`.
- repository: `hyunjun1127/ODE-edit`.
- GH는 검토 자료의 정확한 사본, 정본 계획, 실행 가능한 task/envelope, source/config 잠금과 SH4 직접 전달을 소유한다. SH4가 production repair, 구현 검산, Slurm 제출과 실제 method 실험을 소유한다. 정확한 session/CWD와 중복 nonce를 확인한다.
- GH/SH4는 기존 사용자 model/effort 설정을 유지한다. root dirty checkout이나 기존 실험 archive를 덮어쓰지 않는다.

## 필수 정독과 우선순위

이 디렉터리의 `authorized-run-contract.json`이 최신 실행 범위 정본이다. `numerical-repair-contract.json`은 공통 repair 계약, `followup-design.json`은 수식·알고리즘 세부 참조, `review.md`는 근거와 해석이다. `followup-design.json`의 FREE075는 이번 실행에서 제외한다. 기존 FLAT/REVERSE 취소도 유지한다.

`evidence-summary.json`, 네 CSV, `recover_diagnostics.py`, `verification.json`은 이전 PRICE source `0415aba3c160170d306be8196792f198dad4d122`의 B1–B15를 읽기 전용으로 검토한 자료다. 게시 기준은 `677dad8977e1650b177f37ae7d11e83b6a50dcd3`이다. 이 자료를 새 repair의 GPU 검증이나 W20 결과로 쓰지 않는다. 이전 `verification.json`은 그 manifest에 나열된 당시 파일만 봉인하며, 새 실행 권한은 이 문서와 authorized contract에 있다.

## 실행 범위

| Arm | beta_base | local cap |
|---|---:|---|
| CAP075 | 0.75 | 0.75 a |
| CAP100 | 1.00 | 0.75 a |
| FREE100 | 1.00 | 없음 |

위 세 arm을 **Llama와 Qwen 양쪽에 적용해 총 여섯 개의 독립 2k trajectory**를 제출한다. 같은 repair source, 각 model×arm의 독립 cold W0/H0, 고정 CounterFact first2000, BS100×20이다. 총 12,000 edit applications이며 서로 다른 benchmark 요청 12,000개라는 뜻은 아니다. 기존 실행을 이어받거나 기존 PRICE를 corrected CAP075로 대신하지 않는다. 각 batch는 자신의 model×arm에서 commit한 W/H를 사용한다.

Llama는 기존 Meta-Llama-3-8B-Instruct revision·순서·seed·L4–L8/anchor8·loss·writer·평가 정의를 유지한다. Qwen은 저장소의 기존 canonical Qwen 모델·tokenizer revision, model-specific eligible layers/anchor/module lookup/native covariance와 cache를 확인해 정본에 결속한다. Llama의 L4–L8·module 이름·tensor dimension을 Qwen에 그대로 이식하지 않는다. 가격·예산·cap의 세 arm 정의와 공통 repair는 동일하며, model별 구현 차이는 명시한다. 다른 모델의 edited state·평가 결과·가격을 재사용하지 않는다.

확인된 Qwen 정본은 `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`, alias `qwen2.5-7b-inst`다. 기존 Qwen 기록의 eligible layers는 L4–L8, anchor8로 같지만 **final readout은 27**, hidden 3584, intermediate 18944, vocab 152064다. Llama의 readout31/intermediate14336 하드코드를 분리한다. 근거는 `audits/servers/server4/S4-M0-storage-execution-readiness.md`와 `P4-hf-consumed-closure-readiness.md`이다. 이 cache readiness는 과거 관측이므로 SH4가 현재 실제 config/revision/cache/stats·메모리를 재확인한다. 기존 PRICE의 완성된 Qwen 실행은 확인하지 못했으므로 model-specific adapter/pack/observer/resource binding은 이번 구현 범위에 포함한다. 기존 Qwen baseline의 clamp4/lr.5나 AlphaEdit projector를 이 soft ridge PRICE method에 가져오지 않는다. method의 c.75/KL.0625/norm.5/lambdaC15000/lr.1은 공통으로 유지한다.

- `native_c=0.75`, `beta_max_native_scale=0.75`, `beta_base`와 `cap_mode`를 분리한다.
- `beta_max,r=max(beta_base,0.75*max_l pi_lr)`; 기존 4단계 지수 schedule, grace 12 own updates, F threshold .05, 25 evaluations/24 updates를 유지한다.
- base=1.0에서 pi_max<4/3이면 ceiling도 1.0으로 오르는 요청 수를 보고한다. CAP075의 ceiling을 1.0으로 바꾸지 않는다.
- cap-free는 `max(n-tau*w,0)`의 실제 uncapped projector 분기다. infinity cap이나 숨은 clip으로 대체하지 않는다. nullable cap과 mode를 producer/collector에서 일치시킨다.
- PRICE 가격·floor·own-entry 고정·다음 batch 갱신, fresh BUILD, 실제 effective response injection, full off-owner same-layer pullback, EfficiencyAdamAbs, soft native ridge writer, terminal payload commit와 history once를 유지한다. exact writer / dK·dP / builder reverse를 도입하지 않는다.

## 공통 repair

1. projector breakpoint의 owner 및 ZERO/CAP 종류를 보존한다. 선택된 endpoint를 정확한 0/cap으로 저장하고 tied endpoint를 일관되게 처리한다. 단순 norm cutoff로 작은 양수 block을 삭제하지 않는다. 기존 FP64 KKT 및 FP32 feasibility 허용기준을 유지한다.
2. 극소 양수는 `R/||R||` norm gradient를 가지므로 교정이 optimizer trajectory에 영향을 줄 수 있다. 두 모델의 세 arm 모두 동일 교정을 적용한다. moment reset 없이 task-gradient 재진입을 보존한다.
3. exact-zero / endpoint correction / tiny diagnostic을 구분하고 target/action norm, ratio defined/reason, 분모와 집계 단위를 남긴다. zero target의 cross-owner action을 0으로 만들지 않고 leakage로 보존한다. 임계값 초과 조건부 요약을 전체 실현률로 쓰지 않는다.
4. 가격 분포·최저가 층·tie/floor·종료 상태·확장 단계·own updates·base/max/current beta·실제 spend/slack·cap 접촉·support/reentry·rewrite NLL/가중 KL을 게시한다. 기존 collector가 버리던 controller 정보까지 producer→collector→보고서로 연결한다.
5. profile→prepare/preflight→price receipt→controller→fit→collect→source/config lock 전체에 cap/base knob를 전달한다. 기존 .75 hardcode를 임의로 모두 1.0으로 치환하지 않는다.

## 검산·평가·자원

사용자의 no-toy 지시를 유지한다. 별도 synthetic/toy, standalone key study, small-B fit, B1 pilot 재실행, 성능 승급 gate를 추가하지 않는다. 실제 첫 B1의 이미 계산한 geometry/proposal/response로 endpoint·KKT·feasibility·counter·terminal payload/history 계약을 확인하고 동일 BS100×20 trajectory를 계속한다. 낮은 성능은 관측 결과이며 조용한 설정 변경·재실행의 근거가 아니다.

기존 current pre/post, W0/W5/W10/W15/W20, 같은 cohort/birth cohort의 R/P/N preference·teacher-forced strict·NLL·lost/gained·획득/유지·실현·가격·비용을 보고한다. Paraphrase를 학습·가격·controller에 넣지 않는다. 평가 중복 forward를 추가하지 않고 기존 관측과 정확한 identity가 맞는 W0만 재사용한다.

**server4 GPU cap은 두 모델 합계 2다.** 각 model×arm은 1GPU이며, 기존 해당 사용자의 server4 GPU 작업과 GPU를 사용하는 준비/검산까지 합산해 동시 총량 2를 넘지 않는다. 여섯 run을 최대 두 lane의 dependency/queue로 등록한다. 기존 점유가 있으면 신규 작업을 기다리게 하며, 이 요청은 기존 job 취소·재시작 권한이 아니다. GH/SH4가 실제 node/QoS/메모리/현재 점유를 결속한다. 각 모델의 corrected CAP075를 포함해 전량 등록하고, 가능하면 두 모델이 모두 시작할 수 있도록 lane을 구성하되 실제 예약 구조는 cap2를 보장하도록 선택한다. 다른 모델/arm의 성능 통과를 submission dependency로 삼지 않는다.

NoCP 및 현재 승인된 공통 tracking 정책을 적용한다. durable 모델·M/P/K/H·activation tensor dump를 새로 추가하지 않는다. raw는 local에 유지하고 Git에는 compact source/manifest/집계/보고만 게시한다. artifact 보존·용량·전송·보고 경로는 기존 protocol과 최신 사용자 지시 안에서 GH envelope에 결속한다. 별도의 장기 모니터나 주기 알림을 만들지 않는다.

## GH 이번 turn의 완료 기준

1. 최신 사용자 범위를 반영한 정본·task/envelope를 자신의 전용 worktree에서 게시한다. 허용 source/write paths, Slurm allowed, GPU cap2, 실행 source/config lock, audit와 보고 경로를 명시한다.
2. 정확한 SH4에게 app-server direct로 전달하고 담당 수락 ACK와 accepted turn ID를 회수한다. active이면 관련 작업인지 확인하고 exact expectedTurnId로 steer하며, 관련 없는 turn에는 침범하지 않는다.
3. nonce 접수 ACK를 먼저 회신하고, 최종에는 게시 commit·task/계약 경로·SH4 수락·현재 구현/제출 상태를 구분해 보고한다. 실제 repair와 두 모델 × 세 2k run은 추가 사용자 승인 없이 SH4가 계속하도록 한다. 이 전달 turn은 GPU 실험 완료까지 기다리지 않는다.
