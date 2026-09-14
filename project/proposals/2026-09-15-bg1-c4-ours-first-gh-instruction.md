# GH 전달 지시문 — C4 reference 구축·BG-1 우선 제출·초기 gate 후 사용자 호출 대기

Instruction ID: GH-BG1-C4-OURS-FIRST-20260915-V1

수신: Global Head. 실행 담당: Server4 SH.

상태: 사용자 전달용 지시문. 이 파일 작성 자체로 GH 전송, SH 배정, GPU 제출을 수행한 것은 아니다.

## 1. 이번 사용자 지시와 실행 범위

**C4 reference를 구축하고 최소 ours인 BG-1을 구현·검증하여 W0 B100×10으로 제출하라. Baseline editing rerun은 제출하지 말라. 초기 gate가 확정되면 능동 모니터링을 중단하고 사용자가 다시 호출할 때 작업을 이어가라.**

이 지시는 아래 기존 설계의 실행 순서·신규 제출 수·agent 종료 규칙을 수정한다. 기존 문서의 7-policy/70-batch 표는 비교 대상 목록이지 이번 신규 제출 목록이 아니다. 이전 GH 지시문의 “완료까지 계속 모니터링”, “재사용이 안 되면 baseline 재실행”, “자동 후속 확장”도 이번 task에 적용하지 말라.

| 대상 | 이번 처리 |
| --- | --- |
| AlphaEdit / MEMIT / AlphaEdit-BLUE / MEMIT-BLUE / AlphaEdit-L4_only | 기존 결과·checkpoint 재사용. 신규 editing chain 0개 |
| REFIT4 | 기존 보조 근거만 연결. 신규 chain 제출하지 않음 |
| **BG-1** | **신규 scientific chain 1개, W0부터 B100×10, 동일 1000요청** |
| Reference builder / W0 teacher / BG 기술 검사 | ours 준비 작업으로 수행·필요한 작업 제출 가능 |
| 저장된 N4 checkpoint의 C4 평가 | calibration에 필요한 범위에서 허용. Editing rerun과 별도 기록 |
| S128 / Pile / BG-1R / BG-2R / ScalarGuard / FixedPenalty / endpoint 대조 / full10k | 이번 신규 제출 범위 밖. 사용자 재호출 뒤 검토 |

Scientific chain 1개라는 것은 준비·평가를 포함한 scheduler job ID가 반드시 1개라는 뜻은 아니다. 제출 receipt에서 준비 job과 BG-1 chain을 구분하라.

기준 문서:

- [Reference survey 및 구축 순서](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-reference-set-survey-ko.md)
- [C4-WebRef-v2 데이터 계약](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-bg-tw-reference-data-contract.json)
- [BG-1 단계적 method 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-bg-tw-from-base-staged-design-v2.md)
- [Method 계약](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-bg-tw-from-base-contract.json)
- [이번 제출·중단 규칙의 기계 판독 계약](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-bg1-c4-ours-first-dispatch-contract.json)

위 데이터·method 계약의 과학적 정의는 유지한다. 신규 제출을 생성하는 runner는 반드시 이번 dispatch 계약의 `new_scientific_policies=[BG-1]`를 읽어라. 기존 `first_policies` 전체를 순회하는 launcher를 사용하지 말라.

## 2. 구현을 먼저 시작하고 baseline 대조표 완성을 기다리지 말라

기존 결과의 위치, W0 revision, 요청 ID/order, layer/config, 평가 version과 분모를 reuse manifest에 기록하라. 사용자가 기존 결과의 존재를 확인했으므로 다섯 baseline을 재현 실행하는 것으로 시작하지 말라. 일부 결과의 provenance나 문항별 자료가 부족하면 `REUSE_SCOPE_LIMITED` 또는 `NOT_AVAILABLE`로 표시하라. 비교표 전체가 완성되지 않았다는 이유만으로 독립적으로 진행 가능한 reference builder·ours 구현을 멈추지 말라.

단, **BG-1의 고정 ceiling을 정하는 데 필요한 N4 calibration 자료는 실제 의존성**이다.

1. 동일 W0/order의 저장된 N4 B1–B10 checkpoint 또는 당시의 실제 저장 delta로 복원 가능한 endpoint를 찾는다.
2. 새 C4 S64를 고정한 뒤 그 endpoint에 forward 평가만 수행한다. 저장 delta의 단순 materialization은 허용하지만, target 재최적화·writer 재실행으로 N4 경로를 재생성하지 않는다.
3. 기존 계약대로 `b=max(b_num, .9*max_t D64_C4(Wt_N4))`를 고정한다. Numeric floor와 허용 오차는 기술 receipt에 선언한다. Reference source가 다른 KL나 warm W50→W60 수치를 대신 쓰지 않는다.
4. B1–B10 중 일부만 있으면 존재하는 부분의 최대값을 전체 최대값으로 쓰지 않는다. 모든 필요한 endpoint 또는 동일 reference의 검증된 기록이 없으면 `CALIBRATION_MISSING`으로 남긴다.
5. Calibration이 없더라도 ours source·데이터·teacher·기술 검사를 가능한 범위까지 완성한다. 과학적 실행은 막힌 상태로 기록하고 사용자 호출 대기로 전환하라. 임의 b, baseline rerun, silent recipe 변경으로 우회하지 말라.

Baseline 전체의 C4 재평가와 새로운 downstream 평가를 이번 ours 제출의 조건으로 추가하지 말라. 새 forward 평가 비용과 기존 editing 비용을 분리하라. 서로 다른 tokenizer/order/평가 규약의 수치를 동일한 paired 비교라고 쓰지 말라.

## 3. C4 reference를 실제 자산으로 구축하라

지금까지 확인된 것은 source prefix와 schema다. 정식 reference768과 teacher192가 이미 만들어졌다고 가정하지 말고, 기존 자산이 새로 생겼다면 manifest를 확인하여 재사용하라.

- Source: `allenai/c4`, cleaned English `en`, revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`.
- 고정 train 첫 shard와 validation 첫 shard를 전체 획득하고 full-file hash/bytes/gzip CRC/row 수를 기록한다. 두 compressed 파일의 알려진 합계는359,779,975 bytes다.
- 실제 schema는 `text/url/timestamp`. Source row ID, seed20260915, 전체 shard hash sampling, 후보 충원·중복 검사는 C4-WebRef-v2 계약을 따른다.
- Train512 = S64 + Dev128 + Reserve320, validation Report256. 총768개 identity·window·역할을 모델 loss를 보기 전에 고정한다.
- Wikipedia 일부와 domain 겹침을 허용한다. 문서·URL·명시한 near-duplicate와 알려진 평가 입력 중복을 검사하며 미확인 범위를 기록한다.
- 기존 native Wikipedia mom2/projector를 C4로 바꾸지 않는다.

기존 W0 capsule의 정확한 tokenizer로 자연 text256 tokens + BOS1을 만든다. Target input indices `[129,257)`, logits indices `[128,256)`의128개 위치를 사용한다. 문서를 연결하거나 chat template를 추가하지 않는다.

Teacher는 pre-edit W0 고정 full-vocabulary log probabilities, FP32, vocabulary128256이다. 첫 생성은 **S64+Dev128=192문서**, tensor payload 약11.74 GiB다. CPU memory-mapped cache와 GPU microbatch streaming을 사용한다. Reserve/Report teacher 생성은 이번 준비에서 연기한다. 문서·tokenizer·W0·kernel/cache hash와 실제 메모리·setup 비용을 남겨라.

필수 준비 산출물은 source/filter manifest, reference documents/tokens/splits, composition/overlap audit, teacher manifest/shards, build status다. Raw text·teacher·모델은 local artifact root에 두고 Git에는 compact manifest·코드·보고만 남겨라.

## 4. 최소 ours BG-1을 구현하라

BG-1은 단일 L4 native proposal에 executable residual correction을 한 번 추가하는 버전이다. 처음부터 두 stage, old replay, dynamic core, medoid, ODE solver를 함께 넣지 말라.

1. Pre-edit W0에서 시작하고 current B100의 native target을 계산한다. 요청당 최대24 Adam updates/25 loss evaluations, 원 early stop·anchor·clamp를 유지한다.
2. Inner parent와 native K/P/history를 고정하고 실제 writer map `S(R)=R A`를 사용한다. `W_parent+eta*R A`가 모든 token에 적용되는 provisional model에서 current canonical desired loss와 C4 D64를 계산한다.
3. Objective는 기존 BG-1의 `E_cur + mu*b_tau((b-D64)/b)`다. C2 relaxed log barrier를 사용하고 현재 target ball 및 executable write-norm trust bound를 유지한다. 시작값 zeta=.25, tau=.1, mu=.01은 개발 설정으로 기록한다.
4. Native post-write preview에서 residual gradient 최대1회/B100을 계산한다. Current와 S64 microbatch의 가중치를 정확히 누적한다. W0 자체의 KL gradient가0인 것은 수학적으로 예상되는 현상이다.
5. 후보는 raw eta1, corrected eta1/.5/.25의 최대4개다. 같은 parent에서 actual forward screen을 적용하고 feasible 후보 중 current desired loss로 선택한다. 후보 수를 늘려 숨겨진 search를 하지 않는다.
6. 모두 screen에 실패하면 parent를 유지한다. Zero-write/partial/failed 요청도 all-request 원분모에 남긴다. 별도 native fallback을 적용하지 않는다.
7. B100의100개 target을 모은 뒤 batch write한다. B1 immediate editing으로 바꾸지 않는다. Native history는 processed B100당1회, inner append0이다. Zero-write history와 accepted-label ledger를 구분한다.

Accepted canonical desired label 및 acceptance-time loss를 저장하되 BG-1에서는 old loss를 gradient에 넣지 않는다. 공식 P/N·Historical 정답·FutureN·Report·Audit/MMLU를 controller에 주지 않는다. Dev128은 W5/W10 개발 관찰만 수행한다.

## 5. 기술 검사와 제출

Scientific chain 전에 입력 scoring shift, W0 self-KL, cache checksum/finite 값, 실제 materialization, residual 방향미분, microbatch weighting, rollback, history/finalization, 재개용 상태 저장을 확인하라. 기존 fixture와 좁은 기술 검사를 사용하며 이 과정에서 독립 N4/REFIT4 순차 chain을 제출하지 말라.

GH는 Server4 SH에게 전용 `codex/` branch와 다음 구현 범위를 명시하여 위임하라.

- 신규 모듈·builder·teacher·runner·검증: `project/run_scripts/bg_tw_reference/`.
- 기존 `project/run_scripts/low_cost_write_donor_pilot/`는 가능한 한 읽기 전용으로 재사용한다. 변경이 필요하면 GH가 정확한 파일과 이유를 구현 envelope에 추가하고 기존 실험 동작을 보존한다.
- Native vendor source, 관계없는 실험, 사용자 변경을 덮어쓰지 않는다. GH/SH의 source review·submission 권한과 사실 보고 규칙은 PROTOCOL을 따른다.

필요한 기술 확인과 calibration lock이 끝나면 **BG-1 하나를 W0부터 B1–B10, 총1000요청으로 제출**하라. 계획서만 작성한 상태로 끝내지 말고 실제 job ID·execution source/config/reference hash·output path를 남겨라. 이는 이 지시문을 수령한 GH의 실행 의무이며, 지시문 작성 시점에 제출이 완료됐다는 뜻은 아니다.

실제 자원으로 wall/memory/disk를 확인한다. 이전 warm N4 시간은 참고값이며 BG 실측으로 대체하지 않는다. Batch별 checkpoint/delta와 history·ledger·RNG·next ordinal을 보존하여 접속 종료에도 chain이 진행되게 하라. 수동 B2 제출이나 GH의 다음 메시지를 기다리는 runner로 만들지 말라.

## 6. 초기 gate G0를 명시적으로 확정하라

**G0는 실행 유효성과 착수를 확인하는 기술 gate다. 1000요청 method 성능 판정이 아니다.**

`G0_PASS`에는 다음 증거가 모두 필요하다.

1. C4 reference768 identity/token split 및 initial teacher192의 실제 준비·checksum 확인.
2. W0/config/order, BG-1 구현 기술 검사와 고정 C4 ceiling의 근거 확인.
3. BG-1 전체 B1–B10을 실행하는 persistent job의 실제 제출 및 실행 확인.
4. **첫 B100 처리 완료:** finite metric, candidate 선택 또는 계약상 정상 zero-write, finalization1회, 원분모100, 다음 ordinal100, 재개 가능 상태 저장 확인.
5. 이후 B2–B10은 추가 agent 응답 없이 실행되며 terminal 저장·평가 또는 typed failure가 파일에 기록되는 구조 확인.

첫 batch의 RS/PS/NS 상승, all-request 성공, rejection0, NS 비악화, online1.5배 이내는 G0 조건이 아니다. 정상적인 zero-write가 있었다는 이유로 기술 실패라고 기록하지 않는다. G0_PASS 이후에도 효능·전체 chain 완료는 미확정이다.

다른 상태는 구분하라.

- `G0_FAIL`: 재현된 데이터/구현/상태 무결성 오류 또는 치명적 실행 실패. 증거와 영향 범위를 기록한다.
- `G0_BLOCKED`: `CALIBRATION_MISSING` 등 실제 필수 자산·자원 의존성을 해소하지 못함. 완료된 구현과 준비 자산을 보존한다.
- `G0_PENDING`: scheduler 대기 또는 초기 단계가 아직 끝나지 않음. Job ID만 받았다고 PASS로 쓰지 않는다.

G0 확정 전에는 승인 범위 안의 재현 가능한 기술 오류를 수정할 수 있다. 시도별 source/run receipt를 분리하고 중복 scientific chain이 동시에 돌지 않게 하라. 효과가 작다는 이유로 hyperparameter를 바꿔 재시도하지 말라. Terminal FAIL/BLOCKED를 확정하여 보고한 뒤에는 자동 수정을 이어가지 않는다.

Scheduler 대기가 길면 긴 blocking 대기 대신 pending 상태와 실제 대기 사유를 간결히 알린다. Pending을 PASS나 완료로 가장하지 말라. 사용자 중단 지시가 도착하면 G0 이전에도 그 지시를 따른다.

## 7. Gate 확정 뒤 모니터링을 중단하라

G0_PASS 또는 terminal FAIL/BLOCKED를 확정하면 한 번의 인계 보고를 남기고 **`WAITING_USER_RESUME`** 상태로 종료하라. GH뿐 아니라 이 task의 SH·worker·감사용 agent에도 같은 종료 경계를 전달하라.

- Agent의 Slurm polling, log tail, 주기적 상태 질의, heartbeat 기반 관찰, timer/automation wakeup, 완료 callback에 의한 새 agent turn을 중단한다.
- “실험이 끝날 때까지 대기”, “완료를 확인한 뒤 자동 분석”, “성능이 나쁘면 재제출” 루프를 남기지 않는다.
- **G0_PASS인 경우 제출된 BG-1 job은 취소·pause하지 않는다.** B2–B10과 사전에 제출한 평가·저장은 계속 실행한다. Job 내부 로그/checkpoint/typed failure 기록은 유지한다.
- 이 task를 다시 깨우는 자동 completion notification은 설정하지 않는다. 다른 task의 공용 서비스·sync 설정을 일괄 중단하지 않는다.
- Run 안의 명시된 치명적 무결성 오류·NaN·OOM 등에는 미리 구현한 fail-stop을 적용할 수 있다. Agent의 자율 restart/requeue·method 변경은 허용하지 않는다.
- 사용자 재호출 전에는 ablation, Pile/S128, BG-1R/BG-2R, baseline rerun, full10k를 자동 제출하지 않는다.

대기 중 상태는 **agent 대기 상태**와 **scheduler job 상태**를 나누어 저장한다. 예: `agent_state=WAITING_USER_RESUME`, `job_state_at_handoff=RUNNING`, `last_observed_batch=1`. 이후 job 상태가 바뀌었더라도 마지막 관찰값을 최신 상태로 표기하지 말라.

## 8. 인계 파일과 사용자 재호출 뒤 재개

GH는 Git의 compact 보고와 local raw 자산을 연결하는 resume manifest를 작성하라.

- Instruction/task/run ID, host, GH/SH 소유자, execution source/branch/config hashes.
- Reference·teacher·calibration manifest path/hash; baseline reuse 범위·미확인 항목.
- 준비/평가/scientific job IDs와 dependency, 마지막 관찰 시각·scheduler 상태.
- G0 판정과 각 항목의 실제 증거, 마지막 완료 batch, next ordinal, checkpoint/history/ledger/RNG path.
- 로그·terminal summary·evaluation 결과의 예상 위치와 현재 존재 여부.
- `monitoring_active=false`, `automatic_resume=false`, `resume_trigger=explicit_user_call`.
- 다음 읽기 순서: resume manifest → scheduler 한 번 조회 → terminal receipt/log → checkpoint/metric → GH의 결과 해석.

G0 인계의 표준 경로는 다음과 같이 두고 실제 run ID를 결속하라.

- SH 사실·gate: `experiment-reports/servers/server4/bg1-c4-ours-first-2026-09-15-v1/`.
- GH 상태·사용자 인계: `plans/global/2026-09-15-bg1-c4-ours-first-handoff.md`.
- Local resume manifest: 해당 run root의 `resume-manifest.json`.

사용자가 다시 호출하면 처음부터 재제출하지 말고 위 manifest에서 이어가라. 이미 완료됐으면 저장 결과를 수집하고 전체1000요청을 기준으로 해석한다. 아직 실행 중이면 당시 상태를 보고하고 호출 범위에서 진행한다. 실패했으면 마지막 유효 checkpoint와 원인을 확인한 뒤 재개 범위를 정한다. 원래의 baseline rerun 금지는 사용자가 명시적으로 변경하지 않는 한 유지된다.

SH는 실행 사실·수치·분모·비용·오류·typed gate만 보고한다. Method 우열·preservation surrogate의 유효성·ODE/barrier claim 및 후속 권고는 사용자 재호출 후 GH가 별도 global 분석으로 다룬다.
