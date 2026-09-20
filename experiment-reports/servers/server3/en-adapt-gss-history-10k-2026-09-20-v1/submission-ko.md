# SH3 RES / GSS_REC fixed10k 제출 인계

Nonce: `ODEEDIT-GH-SH3-GSS-TWOARMS-CAP2-20260920-R1`.
상태: **MONITORING_PAUSED_AWAITING_USER**. 두 job의 held owner/source/전체 args/resource 검사 PASS 후 release command가 정상 반환했다. Release 뒤 scheduler/log/result/initial/B2/terminal 조회는 0회다. 실제 초기 실행·B2·overflow·종료 및 생산 모델 sketch 정확성은 **NOT_OBSERVED**다.

| arm | job ID | execution lock SHA256 |
|---|---|---|
| EN_ADAPT_H_RES | `51352` | `694b00469fa1c95583b12107b1d34909913bfd7a59b44ed7e6c7e204363f1f74` |
| EN_ADAPT_H_GSS_REC | `51355` | `000351ef74f2323b97138ceb3cf6631d5cbe66bedc20a2875f5219f3158c2757` |

과학 실행 source: `02a85e3014c22526d192e9a5fffa0b7f72f75f9d` / tree `b102227f56371176ffd53749b7d37ce70162de81`.
Archive: `46b150bba6fd658b8400483ee1f2b03b33302664012e7077f71bc573c5a232f6` (6398522B).
실제 frozen import root: `/data/janghj/ODE-edit/local/en-adapt-gss-history/20260920-v1/frozen-source-v1/source/`.
제출 검사기만 보완한 별도 commit: `ef3f98aec6e53b0fdc42bcdde24d7c7ce793613f`. 이 보완이 frozen 과학 프로그램에서 실행되었다고 주장하지 않는다.

각 job은 1GPU/8CPU/121856MiB(119GiB), ubuntu/gpu, exportNONE, Requeue0, wall7일이다. 합산 cap2/16CPU/238GiB이며 cold W0/zero M4에서 각각 B100×100을 수행한다. B1/B2는 각 chain의 일부로 한 번만 계산되고 같은 RAM 상태로 B100까지 연속 진행한다. 별도 pilot/성능 gate/교차-job state 또는 teacher 공유는 없다. 계획 총200 native/R-gradient batch state·candidate 최대400, RES selection NLL VJP0, GSS_REC no-overwrite 예상 상한94 overflow sweep/57516 fact NLL VJP이다. 모두 계획치이며 실측 비용이 아니다.

기존 all-active lifelong과 제외 uniform GSS는 제출0/취소0이다. RES 첫 held 검사에서 Slurm `AllocNode:Sid` delimiter를 처리하지 못한 partition false-negative가 있었으나 실제 partition=gpu였고 release0/RunTime0이었다. 원 receipt를 보존하고 CPU fixture와 검사기를 최소 수정했으며 동일51352를 재사용했다. GSS_REC를 신규51355로 등록하고 두 held 검사 모두 통과한 뒤 release했다. 과학 source·수식·threshold 변경0, 중복 제출0, 기존 raw 삭제0이다.

실행 봉인 시 CPU47 tests PASS, 제출 검사기 보완 후4 tests PASS(고유48 tests). 두 arm100-batch RAM continuation, exactly-once ledger/native history, current exclusion/repeat/overwrite, immutable teacher/eviction, signed pruning, full-prefix factor/tiny Llama physical autograd, fixed-map seal, LR+LH 및 GR+GH 교차항, 고정 bank의 두 후보, expanded TF/NLL metrics, 오류·strict JSON·rollback을 검산했다. tiny CPU 검증은 생산 모델 precision PASS가 아니다. 부모 precision NOT_ESTABLISHED와 sketch 실제 검증 미관측을 유지한다.

새 `project/run_scripts/en_adapt_gss_history/`는 부모 실제실행5d452221의 native/z-hook/current/geometry/controller/metrics 원본10개 SHA와 결속했다. 완료분석abd08ee1은 source 실행과 분리했다. R512/Dev128 2560 files/101519959223B는 기존 full SHA/size/CPUshape 및 current stat identity로 재사용했고 전송·재생성0이다. Pstar FP64[14336,14326] SHA도 재확인했다. 고정 Gaussian maps의6 logical array shape/dtype/SHA를 CPU에서 사전봉인했으며 GSS overflow에서 일치 검증한다. Model/hidden/head FP32/eager/TF32off, map/sketch algebra FP64, native L4/L2=1/원 target·clamp·Adam 정책을 유지한다.

공통 입력 절대경로:

- Python `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python`
- Native `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/easyedit/`
- Model `/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`
- Dataset `/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json`; SHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1, ordered root5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729
- R512/Dev128 `/data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/inputs/generated-v1/`
- Pstar `/data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/inputs/pstar-derived-v1/basis.npy`
- Context `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/contexts/cold-capsule.json`; P/C0의 실제 consumed path는 각 `execution-inputs/manifest.json`에 결속
- RES `/data/janghj/ODE-edit/local/en-adapt-gss-history/20260920-v1/res/attempt-v1/`
- GSS_REC `/data/janghj/ODE-edit/local/en-adapt-gss-history/20260920-v1/gss_rec/attempt-v1/`

각 attempt의 `execution-inputs/config.json`, `manifest.json`, `evaluator-map.json`이 실행 경로다. `cold-history/`는 자체 final-at-write canonical supplied-target 전체 FP32/full-vocab logp·입력 key/residual이며 edited checkpoint가 아니다. `output/`에는 선택/ledger/metrics/geometry/비용과 B100 후 프로그램이 만드는 CPU 보고서만 저장한다. `save_checkpoints=false`: W/M/optimizer/delta/resume bundle disk0, exact crash-resume NOT_AVAILABLE. B100에는 후속 소비가 없는 teacher 추가 forward가 없다.

마지막 제출 전 free는 48322195456B, 두 cold store·출력·atomic temp·8GiB safety 요구량 43184859120B였다. 요구량 이후 headroom은 5137336336B였다. 비독점 관측이며 release 뒤 재조회하지 않았다. 각 새 host peak100GiB 추정/119GiB 요청, aggregate200GiB 추정/238GiB 요청을 이전 B300 MaxRSS69.375GiB와 구분했다. 각 예상 wall28–72h/합산56–144GPUh, wall limit168h/job이다. 실제 allocation/throughput은 미관측이다.

Official observer는 selection 후 current/first100/hash128 panel과 지정 full latest-valid/누적 N 시점에서 같은 forward로 RS/PS/NS, TF token-micro/prompt-macro/full-target strict, true/new/desired NLL·margin·paired lost/gained를 산출한다. entry→WN과 WN→selected 손상, at-write→현재, bank/age/relation/active·superseded 그룹을 구분한다. Full-history 시점은2,5,10,20,...,100이고 bootstrap10000/seed20260920은 request cluster이다. RES와 GSS_REC는 선택과 recency 가중치가 함께 달라 recency-only/selection-only 원인을 분리할 수 없다.

완료 후 사용자 recall 때의 CPU 비교 예시(지금 실행하지 않음):

```python
from project.run_scripts.en_adapt_gss_history.report import compare_attempts
from project.run_scripts.en_adapt_gss_history.io import save
value = compare_attempts(
    '/data/janghj/ODE-edit/local/en-adapt-gss-history/20260920-v1/res/attempt-v1/output',
    '/data/janghj/ODE-edit/local/en-adapt-gss-history/20260920-v1/gss_rec/attempt-v1/output')
save('/data/janghj/ODE-edit/local/en-adapt-gss-history/20260920-v1/explicit-user-review-v1/comparison.json', value)
```

원본 `M0-ko.md`, authority/parent/map/input/resource seals와 `audits/servers/server3/2026-09-20-en-adapt-gss-history/submission-handoff.json`에 상세 근거가 있다. 이번 범위는 구현·제출·인계까지이며 과학 완료를 주장하지 않는다. 프로그램 자연 진행을 유지하고 사용자 recall을 기다린다.
