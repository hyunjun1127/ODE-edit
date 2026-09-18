# Project Proposals

이 디렉터리는 ODE-Edit의 연구 방향, method proposal, 대안 mechanism, 역사적 원문과
research rationale를 보관한다. Proposal은 검증할 연구 계약이지 완료된 paper claim이
아니다.

## 현재 문서

- [2026-09-18 BPCW512 GH 실행 지시문](/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-18-base-choice-constrained-write-gh-instruction-v2.md):
  Base 답변 선택을 전체512 reference의 제약으로 사용하고, 현재 L4 편집 response를 보존하는 최소 보정을 구현한다.
  Server1의 matched cold B100 N4/BPCW512 비교 뒤 사전 gate 통과 시 각자 B100×10으로 연장한다.
  GSS는 전체 local QP의 처리 순서에만 사용한다. 전달용 문서이며 아직 dispatch·GPU 실행하지 않았다.
- [2026-09-15 사전 보존 한도 없는 파이프라인 재검토](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-bg-tw-pipeline-reset-review-ko.md)와
  [EP-TW-1 단계적 설계 v3](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-edit-quality-preserving-tw-design-v3.md):
  당시 권고안. PDF 식 (13)–(14)의 endpoint 목적을 바탕으로 자기 native proposal의 편집 품질을 유지하면서
  fixed-W0 KL을 줄인다. N4 사전 calibration·TV 한도·log barrier를 제거하고 correction-only 후보와 native 복귀를 정의한다.
  새 방법은 W0 B100×10 한 경로부터 시험한다. 원격 지시 변경·GPU 실행 결과가 아니다.
- [2026-09-15 C4 BG-1 우선 제출 GH 지시문](/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-15-bg1-c4-ours-first-gh-instruction.md):
  이전 V1 지시 원문. 후속 Server4 보고에서는 reference768 구축, teacher job47592 마지막 PENDING,
  N4 endpoint 부족에 따른 CALIBRATION_MISSING과 BG 과학 실험0건이 확인됐다.
  새 EP-TW-1은 별도 versioned admission 대상이며 이 역사적 지시의 내용을 조용히 바꾸지 않는다.
- [2026-09-15 C4·Pile reference 재조사와 구축·실험 계약 v2](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-reference-set-survey-ko.md):
  CL의 task replay와 LLM generic calibration을 구분하고, GPTQ·SparseGPT·Wanda의 반복 사용을 근거로 C4-en을
  첫 corpus, Pile를 후속 대조로 선택한다. 인용 색인 표시값의 버전·cache 한계를 기록했다.
  C4 pinned train/validation source를 확인하고 Wikipedia 일부·domain 겹침을 허용하는 S64/Dev128/Reserve320/Report256,
  고정128개 full-vocab 분포를 정의한다. 문서·token은 Server4 구축 완료 보고, teacher 완료는 미확인이다.
  당시 method 연결은 EP-TW-1이었다. 최신 BPCW의 입력 구성은 BPCW-v2 계약을 따른다.
- [2026-09-15 BG-TW PDF 상세 검토](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-bg-tw-pdf-method-review-ko.md):
  ICLR_2027.pdf의 13쪽과 완료 I2/I4/FROZEN 수치를 검토한다. 고정 writer의 endpoint 등가,
  relaxed barrier의 범위, 한 번의 gradient와 slack-weighted penalty의 등가,
  ODE/CBF 및 MetaKE와의 주장 경계를 정리한다.
- [2026-09-15 BG-TW W0 단계적 설계 v2](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-bg-tw-from-base-staged-design-v2.md):
  이전 fixed-budget BG-1 설계 기록. 일곱 경로는 역사적 비교 목록이며 현재 자동 제출 목록이 아니다.
  첫 method와 N4 기반 한도는 v3로 대체했다. 기존 동기와 native 수학 검토는 보존한다.
- [2026-09-14 REFIT4 write-refresh 1,000요청 순차 GH 지시문](/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-14-refit4-write-refresh-seq1000-gh-instruction.md):
  여섯 정책의 B100×10 구현·재사용·실행·평가·claim 판정을 GH가 관리하기 위한 전달용 지시문.
  단일 batch 성능 선별과 audit 선행 gate를 배제하고, SH 위임 범위와 완료 산출물을 명시한다.
- [2026-09-14 REFIT4 write-refresh 1,000요청 순차 확정 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-14-refit4-write-refresh-seq1000-final-design.md):
  최신 사용자 요구에 따라 단일 batch 선별을 N4·REFIT4·FROZEN2·I2·FROZEN4·I4의 B100×10 비교로 대체한다.
  두 refresh 수에 각각 frozen control을 맞추고 Adam 상태·entry teacher/clamp 유지, 평가·예산·재사용 조건을 고정한다.
  이전 G1/G1-R의 실행 순서를 대체하는 설계이며 신규 GPU 실행 결과는 아니다.
- [2026-09-14 six-arm 완료 리뷰와 REFIT4 다음 기전 대조](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-14-refit4-mechanism-next-gate-design.md):
  완료 B51–B60을 바탕으로 동일 entry의 norm-matched scalar와 frozen-first-z 재피팅을 비교한다.
  최신 사용자 의견에 따라 Audit128/MMLU68은 다음 기전 실험의 선행조건에서 제외하고 미측정으로 남긴다.
  두 entry·다섯 endpoint, 조건부 Late 순차 검증과 상태/updater 대조를 구체화한 설계이며 신규 GPU 결과는 아니다.
- [2026-09-13 저비용 write·donor pilot GH 지시문](/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-13-low-cost-write-donor-pilot-gh-instruction.md):
  위임용 지시문 초안. 여섯 endpoint와 한 후보의 5-batch suffix를 구체화하고,
  수치 참고선의 일괄 통과 대신 claim에 맞는 허용 판정과 냉정한 사실·해석 보고를 요구한다.
- [2026-09-13 저비용 write·donor pilot 구체 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-13-low-cost-write-donor-pilot-design.md):
  기존 ABC·E01 결과를 재사용하는 companion 실행 비교안. Middle의 native·두 축소·L8 보완·같은 L4 재피팅
  여섯 endpoint, 개발 품질 기준, 후보 선택 후 audit, 반복 정책의 5-batch suffix와 비용을 정의한다.
  GPU 실험 결과가 아닌 설계이며 cell 목록과 판정 계약을 함께 제공한다.
- [2026-09-12 Baseline 메커니즘에서 출발하는 lifelong 실험 방향](/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-12-baseline-mechanism-first-lifelong-editing-design.md):
  연구 배경. 기존 근거·코드를 재사용해 baseline의 target/write/history/출력 반응을
  분석하고 최소 개입으로 원인을 구분한 뒤 방법을 선택한다. E0–E2가 본 실험이며,
  누적 목적함수와 추가 layer는 조건부 후속 분기다. 신규 GPU 결과가 아닌 설계 제안이다.
- [2026-09-03 Ordered Response-Barrier ODE-Edit](2026-09-03-ordered-response-barrier-ode-edit-proposal.md):
  이전 method proposal. Official-form full-residual actuator, current terminal-response
  JVP로 만든 \(h\)-independent velocity multiplier, shallow-to-deep ordered state rebuild,
  multi-sweep revisit와 semantic first-hit을 정의한다. Realization potential은 local/soft
  barrier이며 finite-step safety를 보장하지 않는다. Residual excursion과 baseline-native
  parameter action은 controller constraint가 아닌 mechanism telemetry로만 사용한다.
- [2026-08-31 FzCB-Edit method pivot](2026-08-31-fzcb-edit-method-pivot-proposal.md):
  historical predecessor. Stock direct target의 정확한 context semantics,
  MEMIT/AlphaEdit native writer basis, full-model fixed-\(z\) homotopy equality,
  spent-plus-suffix completion budget, whitened equality-null rectification과
  predictor--corrector pipeline을 정의한다. Output KL과 reference fact는 controller에서
  제거하고 held-out 평가로만 사용한다.
- [`2026-08-30-fixed-z-functional-safe-write-proposal.md`](2026-08-30-fixed-z-functional-safe-write-proposal.md):
  2026-08-30 research-reset proposal. Functional anchor/KL contract와 FCW branch를
  정의했던 직전 proposal이며, 2026-08-31 method pivot 이후 historical 문서로 보존한다.
- [`ODE_BF_Dynamic_Layer_Proposal.md`](ODE_BF_Dynamic_Layer_Proposal.md):
  cold target, fixed-E8, full-residual dynamic routing, soft H/P audit와
  W64/BF16 transaction을 결합한 pre-reset proposal. 현재 primary method로 해석하지 않는다.
- [`00.ODE_Alloc_Proposal_Report.md`](00.ODE_Alloc_Proposal_Report.md):
  model trajectory 대신 layer coefficient를 탐색한 pre-reset alternative track. 새 proposal의
  coefficient-space MVP와 동일한 scientific contract로 간주하지 않는다.
- [`00.proposal.md`](00.proposal.md):
  최초 수령한 BF-ODE-Edit proposal 원문. 현재 method와 동일한 문서로 해석하지 않는다.

## 현재 proposal의 상태

최신 설계는 **BPCW-v2: 현재 L4 편집 response를 유지하면서 전체 reference512의 W0 답변 선택을 보호하는 최소 보정**이다.
GSS는 reference를 줄이지 않고 전체 local QP의 제약 처리 순서에 적용한다.
Server1에서 N4/BPCW512를 동일 W0의 cold B100으로 비교한 뒤 사전 gate 통과 시 각자의 B1에서 B100×10으로 연장한다.
설계·CPU 수학 검증·GH 전달문은 준비됐으며 실제 BPCW runner 구현·GPU 실행은 아직 하지 않았다.
2026-09-15 EP-TW-1의 KL 보정과 이전 baseline 분석·W50 suffix 결과는 역사적 연구 기록으로 보존한다.

## 2026-09-03 method proposal의 기록

아래는 Ordered Response-Barrier ODE-Edit의 기존 설계다. 현재 baseline 진단의
objective·방향 선택·stopping rule로 자동 승계하지 않는다.

- Entry state에서 baseline이 계산한 \(z^\star\)와 baseline-native artifacts를 고정한다.
- Prescribed remaining-layer \(R/n\) quota를 제거하고 full residual로 official-form writer
  direction을 구성한다.
- Current terminal-response JVP가 target-realization potential을 줄이는 범위에서
  \(h\)-independent velocity multiplier를 계산한다.
- L4→L8 architectural order로 actual virtual state를 전달하고 downstream field를 재구성한다.
- \(T_{\max}=Nh=1\), primary \(N=4,h=0.25\) 아래 multi-sweep를 수행하며 numerical
  convergence와 first-hit을 분리한다.
- Target-bearing compute-\(z\) contexts의 모든 target-new token이 strict teacher-forced top-1일
  때만 first-hit으로 종료하며, horizon miss와 zero-flow stall도 finite endpoint를 commit한다.
- Residual excursion, path/net action과 locality/retention은 controller 밖 telemetry/evaluation이다.
- Inner reject/retry/rollback 없이 advance-and-observe하고 terminal에서 한 번만 commit한다.
- Barrier는 continuous/local shaping이며 finite Euler invariance를 보장하지 않는다.
- PS/NS, locality와 prior-edit outcomes는 controller가 보지 않는다.
- Barrier/ODE contribution은 no-quota, frozen/Jacobi와 fixed-horizon ablation을 통과한 뒤에만
  주장한다.

FzCB, FCW와 이전 ODE-BF 결과/구현은 historical evidence와 transaction 자산으로 보존하지만,
새 proposal의 hypothesis support로 자동 승계하지 않는다.

## Historical/companion fast falsification plan

- [`2026-08-30-fixed-z-fast-falsification-plan.md`](../../plans/global/2026-08-30-fixed-z-fast-falsification-plan.md):
  직전 FCW branch의 F1/F2 execution-direction lock. 결과와 historical decision을 보존하는
  companion plan이며 2026-08-31 FzCB method 수식을 정의하지 않는다. 새 execution 순서와
  promotion 기준은 별도 date-stamped plan으로 관리한다.

## Rationale sections

- [`sections/01-motivation-validation.md`](sections/01-motivation-validation.md):
  초기 BF-share/magnitude motivation chain
- [`sections/02-related-work-and-novelty-boundary.md`](sections/02-related-work-and-novelty-boundary.md):
  MetaKE, CAKE, EvoEdit, ODE prior와 novelty boundary
- [`sections/03-direct-z-review-and-motivation-closure-design.md`](sections/03-direct-z-review-and-motivation-closure-design.md):
  direct-z de-bundling과 역사적 설계
- [`sections/04-method-design.md`](sections/04-method-design.md):
  이전 constrained controller 설계. 해당 section과 FzCB 수식은 historical이며,
  Ordered Response-Barrier 방법의 계약은 그 방법의 2026-09-03 proposal에 보존한다.
- [`sections/05-bgode-r2-rho-free-prefix-event-fp64-two-equality.md`](sections/05-bgode-r2-rho-free-prefix-event-fp64-two-equality.md):
  FP64 reduction, singular-aware boundary와 ordered-dictionary fidelity의 engineering precedent.
  Prefix-event와 two-equality controller 수식은 current proposal로 승계하지 않는다.

## Evidence navigation

아래 보고서는 모두 pre-reset evidence다. Current proposal 또는 다른 successor method의
hypothesis support로 자동 재사용하지 않는다.

- [전체 ODE-BF 실험 파이프라인](../../experiment-reports/global/2026-08-08-ode-bf-experiment-pipeline.md)
- [SH1/SH2 실험 리뷰](../../experiment-reports/global/2026-08-08-ode-bf-sh-experiment-review.md)
- [Session 01 closure](../../experiment-reports/global/2026-08-03-session01-motivation-final-closure-and-method-handoff.md)

새 experiment는 `PROTOCOL.md`의 firewall, common-policy, immutable receipt와 resource
contract를 따라 preregister한다. Model-specific rescue, post-outcome case selection,
held-out controller access와 raw artifact의 Git 반입은 허용하지 않는다.
