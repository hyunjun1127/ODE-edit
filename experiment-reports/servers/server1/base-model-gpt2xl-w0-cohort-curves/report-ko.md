# GPT2-XL W0 cohort curve 신규 rerun 제출 인계

상태: **SUBMITTED / RELEASED / MONITORING_PAUSED**. 새 scientific 평가·W&B 온라인 시작·remote readback·완료 결과는 아직 관측하지 않았다. 실제 등록/release 증거와 CPU 검산만 아래에 구분한다.

Nonce: `USER-GH-SH1-SH2-W0-COHORT-CURVES-RERUN-20261007-R1-SERVER1`.
SH1 session `01a04939-f93a-7b50-bca0-65438eab2062`, app root `/mnt/raid5/janghj/ODE-edit`를 실제 경계로 사용했다. 역사29e4 경로로 이동하거나 공유 config를 변경하지 않았다. 전용 non-main WT/branch에서 root dirty와 타인 변경을 보존했다.

## 실제 등록

| 역할 | 실제 job | dependency | 요청 | held 검사 / release | 등록 직후 snapshot |
|---|---:|---|---|---|---|
| cold W0 GPU | 60654 | 없음 | GPU1 / CPU8 / 65536MiB / 4h | PASS / 완료 | PENDING, reason None |
| own CPU collector | 60655 | afterany:60654 | GPU0 / CPU8 / 24576MiB / 2h | PASS / 완료 | PENDING, reason None |

owner/name/source/fullargv/script bytes/GPU/CPU/memory/wall/export NONE/Requeue0/node/partition/QoS/dependency를 held에서 검산하고 collector→GPU 순으로 release했다. 기존 job 취소·수리·hotpatch는0이다. snapshot은 등록 직후 단1회이며 이후 상태라고 주장하지 않는다.

USER W0-only scoped cap exception을 적용했다. 기존 method cap2와 local/global cap 파일은 변경하지 않았다. devbox/gpu/lab_gpu_s1, GPU a6000 8개 및 당시 allocated8, host memory policy 65536≤183296MiB를 확인했다. pre-submit resource-only owner inventory는 해당 devbox GPU job0이었다. 물리 점유를 강탈하거나 GPU 공유하지 않고 scheduler PENDING을 허용했다. 4h는 예약상한이지 ETA·실측 GPU시간이 아니다.

## 기존 W0 보존 및 새 관측 범위

exact old source `de31540487643b3c4187ffad5c09554d18c69e94`와 submission/held seal에 결속된 `60156/60157`만 scheduler accounting을 단1회 확인했다. 둘 다 COMPLETED/0:0; parent elapsed는 GPU815초, CPU3초였다. 이 값은 과거 비용이며 새 rerun 비용으로 합산하지 않는다. Scientific metric/raw/log를 다시 조회하거나 성공을 재판정하지 않았다. old W0 source/job/raw와 ours/native baseline은 KEEP이다.

새 GPT2-XL revision `15ea56dee5df4983c59b2538573817e1667135e2`를 cold FP32/eager/TF32off/autocastoff로 실제 first2000 1회 평가하도록 봉인했다. 준비 시 model은 load하지 않았고 tokenizer/token/target/panel/order 26000 pair의 identity만 검산했다. Safetensors payload의 기존 exact seal+불변 stat을 재사용하며 bin 중복 load/복제와 전체 모델 재해시는 하지 않았다.

26,000 prompt-pair / 52,000 candidate row를 원 adjacent new/true MB2 grouping과 left padding/native GPT2 learned positions/final LN1회로 평가한다. model pointer/version·hook 및 선택5층 byte SHA guard를 사용한다. nonselected full-byte certification은 NOT_ESTABLISHED다. edit/targetfit/solve/optimizer/H/C0/P load/새 checkpoint는 모두0이다.

## fresh raw에서의 CPU 곡선

| payload | 비교 x | R / P / N 분모 | 실제 model/applied/pre/post edits |
|---|---|---|---|
| W0_first2000 | 0, 먼저 기록 | 2000 / 4000 / 20000 | 모두0 |
| current/post | 100..2000, 20점 | 매점100 / 200 / 1000 | 모두0 |
| all_seen/post | 500 / 1000 / 1500 / 2000 | b별100b / 200b / 1000b | 모두0 |

당해 attempt가 만든 fresh per-case row의 occurrence ordinal로 exact slice/prefix를 선택한다. 숫자 case-ID threshold/dedup, 전체2k값을 current20개로 복사, 별도 cohort forward는0이다. w0/current/N 및 w0/all_seen/N은 같은 raw의 정확 N subset alias다. CPU collector가 원 row에서 독립 산술 재집계하고 identity·count·coverage·저장 payload 일치를 검산한다. partial/미관측은0점으로 채우지 않는다.

`edits=reference_cohort_edits`는 비교 cohort 진행축이다. `reference_only=true`, `evaluation_model_state=W0`, `edits_axis_semantics=reference_cohort_progress`를 immutable config로 결속하고 actual/applied/pre/post edits는 항상0이다. 일반 shared method validator는 변경하지 않았다. task-local W0 schema/sidecar만 이 정직한 예외를 검사한다. W1..W20처럼 실제 편집된 상태로 표기하지 않는다.

R/P desired=new, N desired=true; new/true NLL 비교 tie=failure, pct0..100/NLL nats/margin true−new, token numerator/denominator 기반 micro, prompt macro, strict를 구분한다. 9개 scalar field와 harmonic(missing omit/실측 zero0)을 유지했다.

## W&B 단계와 검산

새 UUID는 `5f7146a380134da0`; 실제 run URL은 NOT_YET_OBSERVED다. 기존 run 재사용/rename/backfill0이다. actual job60654와 run.name의 job번호, model=gpt2xl/family=gpt2/writer=none/role=scientific/schema 및 W0 reference revision을 새 runtime에서 결속한다.

cheap online SDK/auth/project/immutable identity readback은 GPU model load 전에 수행한다. 접수된 scalar가 곧 remote PASS라는 주장은 하지 않는다. finish는 새run 마지막21 curve transport행만 bounded readback하여 x0+20current+4prefix의 identity/value/coverage를 검산한다. `tracker.log=False`는 명시 거절 receipt를 남기며 silent drop하지 않는다. network/flush 실패는 LOGGING_DEGRADED로 기록하고 과학 관측을 재실행하지 않는다. console/code/watch/artifact/prompt/model/tensor 업로드는 금지다.

owner CPU25개, 독립 bounded reviewer CPU25개, exact frozen archive CPU25개를 실제 실행해 PASS했다. 앞선 owner fixture의 예외 타입 assertion2개 오류는 fixture만 수리했으며 old receipt를 보존했다. 독립 source review의 partial report 과대문구 WARN도 완료/미완료로 분리했고 CPU partial fixture로 확인했다. CPU/fake SDK PASS는 actual model/GPU/online PASS가 아니다.

## source / 봉인 / 자료 / 비용

- execution source: `d7284b9466b1913ba3e6a510aca912bbcbf58073`, tree `4e7ac979e0b2b1812ea3a3788effa8cb5b769738`
- config SHA: `a675d70fbae850d90e040684c467b66cb12fceddcc0dd168ea3925abd8a7142a`
- execution lock SHA: `f7a63d4fa605f600345acfe9e735abbd7608ca07092e9b0c238cfe0b3c2ffefb`
- source archive 및46 source/input/runtime identity: [compact manifest](../../../../audits/servers/server1/base-model-gpt2xl-w0-cohort-curves/source-input-manifest.json)
- held/release/old binding/full local receipt hashes: [submission](../../../../audits/servers/server1/base-model-gpt2xl-w0-cohort-curves/submission.json), [preflight](../../../../audits/servers/server1/base-model-gpt2xl-w0-cohort-curves/preflight.json)
- 재현 CLI: [profile README](../../../../project/run_scripts/base_model_eval/gpt2xl_server1_cohort/README.md)

Local create-once attempt: `/mnt/raid5/janghj/ODE-edit/local/base-model-gpt2xl-w0-cohort-curves/20261007-v1/attempt-v1/`. GPU raw/identity/metrics와 collector report/CSV/manifest는 이 경로에 자연 생성한다. old raw를 이번 actual 측정으로 대체하지 않는다. NoCP/exact_resume=NOT_AVAILABLE. free disk 약1.19TB·free inode3.35억·2GiB 비독점 reserve를 metadata로 확인했다. 신규 allocation/program/forward/peak 비용은 미관측이며 원 worker가 기록한다.

Git에는 source/tests와 compact metadata/제출 보고만 게시한다. raw/prompt/model/tensor/fullstdout/secret0. 동일 host에서 GH 접근가능 local raw를 보존하므로 대형 raw 복제/전송0, NO_BROADCAST_NOT_REQUIRED다. 실제 Markdown renderer는 설치되어 있지 않아 NOT_RUN; GFM table/상대링크/JSON/manifest/hash/ownscope 정적 검산은 게시 전에 별도 수행한다.

sealed GPU runner와 afterany collector는 자연 진행한다. 등록 이후 scheduler/log/result polling·heartbeat·automatic retry0이며 결과 완료를 이 제출 인계에서 기다리지 않는다.
