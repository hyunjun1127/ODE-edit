# ODE-Edit / PRICE

**2026-10-08: 공식 배포·실험 기준은 저장소 최상위 [official/](official/README.md)입니다.**
CAKE 형식을 참고해 ours·baseline 구현, hparams, 공통 평가기와 서버별 runner를 한 폴더에서 관리합니다.
새 baseline 실험은 `main`에 게시한 `official/` 코드와 설정을 사용하고 commit/tree SHA를 기록합니다.

현재 준비 범위는 **FT·MEMIT·AlphaEdit·AlphaEdit-BLUE·MEMIT-FE·AlphaEdit+SPHERE ×
Llama3-8B-Instruct·Qwen2.5-7B-Instruct·GPT-J-6B × CF·zsRE = 36개 본 실험 행**입니다.
각각 2,000건을 100건씩 20 batch 순차 편집합니다. Qwen BLUE 격자·강도 대조 5개 행을 추가하며,
선택된 격자 실행의 본문 재사용을 포함하면 전체 편집 chain은 40개입니다.
ours 구현은 배포 폴더에 포함하되 이번 실행 목록에서는 제외합니다.

- [전체 체크리스트·설정·평가·checkpoint·실행 순서](official/README.md)
- [방법별 hparams와 공통 실험 계약](official/hparams/contract.json), [원본 설정 SHA](official/hparams/sources.lock.json)
- [서버별 EasyEdit 자산 연결과 runner](official/runners/README.md)
- [배포 source provenance](official/SOURCES.json), [CPU 준비 결과](official/PREPARATION.md)

**FLU/CON 참조 자산은 기존 SHA 그대로 유지합니다.** 코드는 현재 server1 baseline
job 61519·61520·61521의 실행본을 기준으로 CAKE·BLUE의 연산 방식을 참고합니다.
case별 prompt batching과 KV cache를 사용하고, 한 번 생성한 원문을 FLU/CON이 함께 평가합니다.
자세한 source pin과 생성 규칙은 [공통 FLU/CON 설명](official/README.md#flucon-기존-자산-유지-현재-server1-코드-기준)에 기록합니다.

```bash
python3 -m official.tools.verify
python3 -m official.experiments.prepare matrix --output local/official-baselines/configuration
```

## Main results

모델별로 CF와 zsRE의 최종 성능을 기록합니다. 각 실험은 2,000건을 100건씩 순차 편집한
최종 checkpoint(W20)를 기준으로 하며, 지표 정의는 [공통 실험 계약](official/hparams/contract.json)을 따릅니다.
**이번 표는 새 cold-start 재실험 전용입니다.** 과거 완료 수치나 과거 checkpoint 재개 결과를
새 실험으로 소급 입력하지 않습니다. checkpoint가 있어도 sample 구성·순서가 다르면 다시 실행합니다.
미제출은 빈칸, 실제 제출 후 대기는 `PENDING: <실제 job ID>`, 실행 중은
`ING: <실제 job ID>`로 해당 dataset의 칸에만 표시합니다(CF 6칸 / zsRE 3칸).
qualification·W0 준비·collector·tuning job은 본실험 job으로 표시하지 않습니다.

GH가 서버별 제출·상태·완료 보고를 받아 이 표를 통합합니다. 각 dataset의 상태/수치에는
server·job ID·관측 시각·실행 commit·config SHA·ordered sample identity를 담은 보고서를 연결합니다.
새 W20 실측이 완료된 지표만 수치로 교체하며 실패·취소·결측을 0이나 완료 성능으로 쓰지 않습니다.
사용자가 FLU/CON을 후속 checkpoint 평가로 미룬 실행은 factual W20 결과와 별도로
CF Flu/Con을 `DEFERRED`로 표시하고 전체 평가 완료로 주장하지 않습니다.
Ours 행은 결과 기록용이며 새 arm/실험을 자동 승인하지 않습니다.
세부 [표 관리 정책](control/main-results-policy.json)과 [사용자 지시](messages/head/2026-10-09-main-table-fresh-rerun.json)를 따릅니다.

### Llama3-8B-Instruct

**2026-10-09 SH1 W20 원자료 검산: CF 2종과 zsRE 6종 완료.**
CF FT **61771**, SPHERE **61770** 및 zsRE FT/MEMIT/AlphaEdit/BLUE/MEMIT-FE/SPHERE
**61716/61717/61718/61719/61720/61721**의 실제 2,000건·20 commit·순서·raw NLL·분모를 검산했다.
CF source `34e4d52d`, zsRE source `94304dc9`; CF Flu/Con은 DEFERRED다.
CF MEMIT-FE **61773**는 검토 시 RUNNING/14 commit이며 최종 수치는 아직 없다.
AlphaEdit **61769**·MEMIT **61772**는 사용자 지시로 CANCELLED; 아래 CF 두 수치는
그 신규 job 결과가 아닌 과거 **42657/42658 B020** 결과다(‡).
zsRE Loc은 **W0 prediction agreement**이며 별도 `Specificity_loc_ans`가 아니다.
[W20 검산 보고](experiment-reports/servers/server1/official-baselines-20261008/results-review-20261009/report-ko.md) ·
[job/source/config/raw SHA·정확한 수치](audits/servers/server1/official-baselines-20261008/results-review-20261009/results.json).
별도 GPU qualification은 `NOT_RUN_USER_DISABLED`로 유지하며, CPU 결과 검산을 GPU 검증이나 온라인 전송 검증으로 표시하지 않는다.

| Method | CF Score | CF Eff | CF Gen | CF Loc | CF Flu | CF Con | zsRE Eff | zsRE Gen | zsRE Loc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FT | 56.59 | 89.25 | 73.30 | 35.50 | DEFERRED | DEFERRED | 14.84 | 11.99 | 1.42 |
| MEMIT | 58.88‡ | 64.75‡ | 61.70‡ | 51.83‡ | DEFERRED | DEFERRED | 46.71 | 42.07 | 37.25 |
| AlphaEdit | 84.80‡ | 99.30‡ | 93.23‡ | 68.59‡ | DEFERRED | DEFERRED | 98.74 | 94.60 | 51.22 |
| AlphaEdit-BLUE | 89.84† | 99.60† | 97.15† | 76.59† |  |  | 99.41 | 95.85 | 66.69 |
| MEMIT-FE | ING: 61773 | ING: 61773 | ING: 61773 | ING: 61773 | DEFERRED | DEFERRED | 14.95 | 13.71 | 0.97 |
| AlphaEdit+SPHERE | 86.87 | 99.50 | 94.95 | 71.68 | DEFERRED | DEFERRED | 98.52 | 94.72 | 50.22 |
| PRICE (Ours) |  |  |  |  |  |  |  |  |  |

† 사용자 2026-10-09 지시에 따라 표본·순서를 대조한 기존 Llama BLUE job **39283_1**의
**B20/2,000 edits** 결과를 반영했다(새 official 재실행 결과 아님).
동일 fixed10k 파일의 첫2,000 순서가 현재 official lock과 일치한다.
Eff/Gen/Loc은 기존 strict NLL preference 집계, Score는 기존 R/P/N 성공률의 조화평균이다.
Loc은 N success 15,317/20,000 = 76.585%를 소수 둘째 자리로 반올림했다.
Flu·Con은 미관측으로 기존 빈칸을 유지한다.
native source/runtime·평가기 차이와 분모 및 대조 근거는
[BLUE 2K 확인 기록](experiment-reports/servers/server1/official-baselines-20261008/llama-blue-2k-table-check.md)에 구분했다.

‡ 사용자 `USER-SIDE-GH-LLAMA-CF-MEMIT-ALPHA-2K-TABLE-20261009-R1`의 명시 예외로
기존 native MEMIT **42658**, native AlphaEdit **42657**의 **B020/2,000 edits**를 반영했다.
새 official rerun 완료 또는 최종 10K 결과가 아니다. R/P/N 분모는 2,000/4,000/20,000,
Score는 반올림 전 성공률의 조화평균이며 표시는 소수 둘째 자리 decimal half-up이다.
Flu/Con은 미평가·DEFERRED이며 zsRE는 별도 새 W20 결과다.
동일 표본·순서는 runtime·hparams·평가기 완전동등을 뜻하지 않는다.
[원 raw SHA·분자/분모·사용자 예외 및 취소 처리 근거](experiment-reports/global/llama-native-cf-historical-b020-20261009.md).

### Qwen2.5-7B-Instruct

**Server4 → Server2 이전 준비 중: 새 SH2 job은 아직 미제출이며 아래 칸은 비워 둔다.**
Server4 첫 CF FT **61783**은 2026-10-09 12:31:29 KST에 디스크 여유 32GiB 검사
`RESOURCE_BLOCKED_STORAGE_KEEP_SOURCE`로 FAILED(1초, 모델/편집 시작 전).
archive **61784**와 후속 `afterok` 연결이 막혔다. SH4는 정확히 남은 미시작 GPU/archive
**23개**를 downstream-first 취소했고 실패 job·source·raw·checkpoint는 보존했다.
기존 PRICE/OURS/tuning은 변경하지 않았다. SH2가 기존 6방법 × CF/zsRE의 12개 cold chain을
이어 맡으며 자산·host binding 중이다. 전달 수락을 Slurm PENDING으로 표시하지 않는다.
원 CF 일정 **W0_AND_W20_FIRST2000**과 zsRE 생성 없음, 별도 qualification
`NOT_RUN_USER_DISABLED`는 유지한다. GPT-J의 DEFERRED 일정을 이식하지 않는다.
[실패·23개 취소·전달 보고](experiment-reports/servers/server4/qwen-migration-20261009/report-ko.md) ·
[SH2 준비 상태](experiment-reports/servers/server2/qwen-migration-results-20261009/report-ko.md).

별도 SH3 **61813 Q3-beta250** cold 최종 W20은 E98.05/G89.95/S68.985/Score83.770638%로
[CPU 검산 완료](experiment-reports/servers/server3/official-baselines-20261008/qwen-61813-review/report-ko.md).
이는 Q3 선택 설정의 final 평가이며 튜닝 실행 자체와 구분한다. 기본 PRICE 본표 승격 승인은 없어
아래 PRICE 행에 자동 합치지 않았으며 Flu/Con은 DEFERRED다.

| Method | CF Score | CF Eff | CF Gen | CF Loc | CF Flu | CF Con | zsRE Eff | zsRE Gen | zsRE Loc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FT |  |  |  |  |  |  |  |  |  |
| MEMIT |  |  |  |  |  |  |  |  |  |
| AlphaEdit |  |  |  |  |  |  |  |  |  |
| AlphaEdit-BLUE |  |  |  |  |  |  |  |  |  |
| MEMIT-FE |  |  |  |  |  |  |  |  |  |
| AlphaEdit+SPHERE |  |  |  |  |  |  |  |  |  |
| PRICE (Ours) |  |  |  |  |  |  |  |  |  |

### GPT-J-6B

**2026-10-09 SH2 검산: CF 6종·zsRE 6종 모두 W20/2,000건 factual 완료.**
방법 순서 FT/MEMIT/AlphaEdit/BLUE/MEMIT-FE/SPHERE의 CF job은
**61650/61725/61778/61779/61780/61781**, zsRE는 **61726/61728/61730/61732/61734/61735**다.
각 원 raw SHA·20 commit·ordered cohort·source/config를 결속하고 CPU 재집계를 대조했다.
CF 분모 R2,000/P4,000/N20,000, zsRE는 request-macro이며 Loc은 W0 prediction agreement다.
CF Score는 반올림 전 E/G/S의 조화평균이다. 소수 둘째 자리 표시는 decimal half-up이며,
CF 비율의 이진 부동소수점 꼬리는 고정 분모의 정수 분자로 정합화한다.
[완료 결과 보고](experiment-reports/servers/server2/qwen-migration-results-20261009/report-ko.md) ·
[각 job/source/config/raw SHA·정확한 수치](audits/servers/server2/qwen-migration-results-20261009/gptj-results.json).
이 검산은 GPU 추가 평가나 W&B 온라인 재검증이 아니다. 이전 W0 전송 오류·취소 이력은 보존한다.

W&B: [zsRE 전용 페이지](https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=zsreindex) ·
[Llama3](https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=zsrellama3) ·
[Qwen2.5](https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=zsreqwen25) ·
[GPT-J](https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=zsregptj).
`zsre/*`는 teacher-forced E/G, W0 agreement Specificity, 별도 loc_ans 정확도를 구분한다.
페이지 설정 원격 검증은 actual run/GPU/성능 완료 검증과 별개다.

**이 CF 실행의 FLU/CON은 W0·W20 모두 `DEFERRED_CHECKPOINT_EVALUATION`이다.**
generation 점수·count·progress를 0으로 채우지 않는다. factual 평가는 유지하고,
매 batch 완료 후 최신 checkpoint 1개와 최종 W20 checkpoint를 보존한다.
FLU/CON은 이후 별도 2k checkpoint 평가에서 측정하며, 그 consumer 완료 전에는
최종 checkpoint를 삭제하지 않는다. 아래 성능표는 미관측 수치를 채우지 않는다.

기존 source/config/dependency와 비용은
[CF 기록](experiment-reports/servers/server2/official-baselines-20261008/deferred-flucon-ready-20261009-r1.md) 및
[zsRE 기록](experiment-reports/servers/server2/zsre-wandb-20261009/report-ko.md)에 보존합니다.
새 실행의 qualification은 `NOT_RUN_USER_DISABLED`이며 GPU 검증 PASS를 뜻하지 않습니다.

| Method | CF Score | CF Eff | CF Gen | CF Loc | CF Flu | CF Con | zsRE Eff | zsRE Gen | zsRE Loc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FT | 58.97 | 88.75 | 66.75 | 40.61 | DEFERRED | DEFERRED | 23.15 | 17.95 | 0.52 |
| MEMIT | 80.63 | 97.90 | 95.38 | 60.58 | DEFERRED | DEFERRED | 93.61 | 88.95 | 55.92 |
| AlphaEdit | 88.22 | 99.70 | 96.33 | 73.56 | DEFERRED | DEFERRED | 99.79 | 96.55 | 84.76 |
| AlphaEdit-BLUE | 89.23 | 99.55 | 97.53 | 75.06 | DEFERRED | DEFERRED | 99.85 | 95.81 | 86.49 |
| MEMIT-FE | 73.85 | 87.15 | 85.70 | 57.22 | DEFERRED | DEFERRED | 29.23 | 27.67 | 29.46 |
| AlphaEdit+SPHERE | 88.39 | 99.70 | 95.73 | 74.27 | DEFERRED | DEFERRED | 99.77 | 96.39 | 86.23 |
| PRICE (Ours) |  |  |  |  |  |  |  |  |  |

기존 연구 문서와 실험 이력은 아래에 보존합니다. 새 실험의 설정 근거는 `official/`입니다.

<details>
<summary>이전 연구 방향과 저장소 기록</summary>

**2026-09-18 최신 설계:** [Base-choice constrained L4 write](plans/global/2026-09-18-base-choice-constrained-write-v2.md).
전체 reference512의 W0 답변 선택을 제약으로 사용하고 현재 L4 편집 response를 보존하는 최소 보정을 설계한다.
[GH 실행 지시문](project/proposals/2026-09-18-base-choice-constrained-write-gh-instruction-v2.md)은 cold B100 비교 뒤 조건부 B100×10 확장을 정의한다.
설계·CPU 검증·전달문 단계이며 실제 BPCW 모델 runner 및 GPU 성능 검증은 미완료다.

**2026-09-12 연구 배경:** [Baseline 메커니즘 분석에서 출발하는 lifelong 실험 방향](/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-12-baseline-mechanism-first-lifelong-editing-design.md).
기존 결과·계측 코드를 재사용해 baseline의 원인을 분석하고, 최소 개입의 결과에 따라
방법을 선택한다. Barrier/ODE는 이 진단의 실험군에 포함하지 않는다.

`ODE-Edit`는 sequential knowledge editing의 baseline 메커니즘,
edit retention과 locality를 분석하고, 그 근거에 따라 편집 방법을 개발하는 연구 저장소다.

이하 본문은 **FzCB-Edit: Fixed-z Conditional-Completion Barrier Editing**의
2026-08-31 설계와 그 이전 연구 기록이다. 당시 method pivot에 따라 output-KL/reference-fact controller를
core에서 제거하고, fixed target progress를 hard equality로 유지한 상태에서 남은 edit의
conditional completion action만 단일 barrier로 제어한다. 2026-08-30 이전 결과와 FCW
branch는 역사적 evidence로 보존하며 새 method의 성능 근거로 자동 승계하지 않는다.

## 이전 FzCB 설계 한눈에 보기

```text
stock z*/shared delta* 1회 계산
        ↓
canonical 또는 context-correct target map 구성
        ↓
MEMIT/AlphaEdit native writer basis T와 action metric H
        ↓
full-model control map C = J_Phi T
        ↓
fixed-z homotopy equality C c = b
        ↓
suffix KKT와 V_suf, spent action E
        ↓
single barrier A0 - E - V_suf >= 0
        ↓
whitened equality-null scalar rectification
        ↓
Euler predictor + nonlinear corrector + actual budget verification
        ↓
s = 1 exact target closure 뒤 atomic terminal commit
```

핵심 설계 원칙은 다음과 같다.

- 기존 editor가 산출한 direct-z는 W-write 비교에서 고정한다.
- 여러 rewrite context에 동일 absolute z를 반복하지 않고 original activation에
  shared delta를 더한 context-correct target을 사용한다.
- 단순 Euler subdivision은 negative control이며 contribution으로 보지 않는다.
- Fixed-z progress는 full-model activation equality가 담당한다.
- Barrier는 spent action과 frozen-geometry suffix action의 총 budget 하나만 사용한다.
- Output KL, locality, prior-edit prompts와 general capability는 controller가 보지 않는다.
- Closed form은 endpoint가 아니라 native low-rank writer geometry \(T,H\)로 사용한다.
- Equality KKT와 suffix KKT 뒤 whitened null-space에서 scalar rectification을 계산한다.
- Corrector의 final coefficient action까지 completion budget에 포함한다.
- Utility first-hit이 아니라 \(s=1\)의 exact target closure를 terminal event로 사용한다.
- static constrained solver가 같거나 더 좋으면 ODE claim을 폐기한다.

## Pre-reset evidence

아래 결과는 구현 자산과 historical negative evidence로만 보존한다. FzCB method를
지지하도록 설계된 scientific evidence가 아니며, 새 방법 성능으로 재해석하지 않는다.

### 확립된 기술 기반

- Llama/Qwen genuine B10에서 W64 reduced solve certificate를 검증했다.
- 동일 W64 virtual endpoint와 committed endpoint의 parameter bytes, logits와
  event identity를 검증했다.
- injected rollback과 최종 W0 restore가 exact였다.
- Native dense와 W64는 BF16 byte-exact하지 않으므로 W64를 Native의 exact
  replacement라고 부르지 않는다.

### 가장 강한 완료 결과

Warm target initialization을 사용한 no-budget `FR-A8-NEWNLL-ALLOFF`의 matched
B10에서 두 모델 모두 `Eff 10/10`, `Gen 20/20`, `Loc 80/100`을 기록했다. 이
결과는 hard H/P veto가 under-edit를 만들 수 있음을 보여주는 강한 causal reference지만,
cold fixed-E8 최종 method의 성능 증거는 아니다.

최신 완료 cold fixed-E8 R8에서 Llama Neutral은 `Eff 8/10`, `Gen 12/20`,
`Loc 89/100`이었다. Qwen Neutral trajectory는 tau=1까지 완료했지만 Soft routing의
수치 certificate failure가 post-freeze panel을 막아 paired endpoint 지표가 남지 않았다.
Common cold-coordinate 수정 후 same-seal 재실행은 당시의 다음 gate였지만,
2026-08-30 research reset 이후 scientific execution priority에서는 내려갔다.

Historical-H benefit도 당시 검증되지 않았다. 관련 B10-1 실험의 active history가
비어 있었던 한계는 pre-reset limitation으로 남기며, FzCB sequential extension을 별도로
preregister하기 전에는 해당 claim을 재개하지 않는다.

## Follow-up 읽기 순서

1. [현재 BPCW-v2 method와 GH 실행 지시문](project/proposals/2026-09-18-base-choice-constrained-write-gh-instruction-v2.md)
2. [Proposal index와 historical 문서 상태](project/proposals/README.md)
3. [2026-08-30 직전 FCW research reset proposal](project/proposals/2026-08-30-fixed-z-functional-safe-write-proposal.md)
4. [2026-08-30 F1/F2 fast falsification plan](plans/global/2026-08-30-fixed-z-fast-falsification-plan.md)
5. [이전 ODE-BF proposal](project/proposals/ODE_BF_Dynamic_Layer_Proposal.md)
6. [전체 pre-reset 실험 파이프라인](experiment-reports/global/2026-08-08-ode-bf-experiment-pipeline.md)
7. [SH1/SH2 pre-reset 실험 종합 리뷰](experiment-reports/global/2026-08-08-ode-bf-sh-experiment-review.md)
8. [이전 coefficient-space ODE-Alloc](project/proposals/00.ODE_Alloc_Proposal_Report.md)
9. [관련 연구와 novelty boundary](project/proposals/sections/02-related-work-and-novelty-boundary.md)
10. [운영 규칙](PROTOCOL.md)

## Repository map

- `project/proposals/`: 현재 proposal, 역사적 원문, method/related-work sections
- `project/run_scripts/ode_edit_method/`: low-rank factor, hook, transaction,
  controller/runtime와 tests
- `project/run_scripts/ode_edit_motivation/`: editor bridge, diagnostic math와
  pre-reset motivation/evaluator 자산
- `experiment-reports/global/`: raw-free 실험 결과와 causal interpretation
- `plans/global/`: preregistered experiment contract
- `scripts/`: session/resource/static gate utilities
- ignored `local/`: raw result, full log, dataset, checkpoint와 private runtime state

## Git과 실행 경계

Git은 proposal, source, lock, compact receipt와 report를 위한 control plane이다.
Model/GPU/Slurm 실행과 raw artifact는 ignored execution plane에 둔다. Active experiment
worktree의 미완성 source는 main에 섞지 않으며, 완료 checkpoint도 source/history가
정리되고 필수 gate를 통과한 뒤에만 main으로 승격한다.

현재 실험 설계의 진입점은 위 BPCW-v2와 GH 지시문이며, 2026-09-12 baseline mechanism 설계는 배경 기록이다.
2026-08-31 FzCB method pivot proposal은 이전 방법 기록으로 보존한다.
2026-08-30 FCW proposal과 pre-reset R8/R10 source/evidence는 별도 역사 계보로 보존하며,
FzCB의 hypothesis support로 자동 승계하지 않는다.

원격 서버 접속 정보, raw IP, username, port, key, token, password와 private dataset
secret은 저장소에 기록하지 않는다.

</details>
