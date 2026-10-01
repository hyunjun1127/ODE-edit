# JLZ와 native 구현 효율화 실행 계획

이 문서는 사용자 승인 효율화 후속을 SH1의 구현 및 제한된 실제 모델 benchmark로 구체화한다. 기존 job56684의 B100×10 과학 실행은 그대로 보존한다. 새 효율화 결과를 그 job에 중간 적용하거나 같은1000/10000 chain을 다시 돌리는 권한은 없다.

Instruction 및 ACK는 `ODEEDIT-GH-SH1-JLZ-EFFICIENCY-20261001-R1`이며, 사용자 research handoff는 `ODEEDIT-JLZ-EFFICIENCY-TO-GH-20261001-R1`이다. [실행 계약](contract.json)이 이번 운영 범위와 예산의 정본이다. [원 설계](../2026-10-01-jlz-efficiency-v1/design-ko.md)와 [검증안](../2026-10-01-jlz-efficiency-v1/validation.json)은 exact bytes로 보존한다. 원 검증안의 PROPOSED 표시는 작성 당시 상태이고, 새 계약이 그 수치를 구현 전 고정한다.

## 비교 대상과 변경 순서

참조는 commit7b4de31d의 실제 `shared_selected` dense-gradient oracle다. full/suffix/selected 평균을 참조 timing으로 쓰지 않는다. 새 namespace에서 E0 계측, E1 중복 down projection 제거, E2 원 순서 MB2 오른쪽 pad crop, E3 entry와 commit head 및 key 조기 종료, E4 evaluator head, E5 MB와 cache 및 sync, 조건부 E6 direct R 순서로 구현한다.

L0–L3 prefix, candidate당 W_eff 공유, ours selected head와 동일 endpoint 관측 재사용은 이미 구현되어 있다. 이를 새 절감으로 다시 계산하지 않는다. 각 변경의 source diff와 원 source SHA를 기록한다. 일반 exception을 key 조기 종료로 처리하지 않고, scoped forward 교체 및 hook는 finally에서 반드시 복구한다.

FP32 model/activation/R와 FP64 geometry, 곱→cast→FP32 add 순서, native KL 방향과 가중치, loss 정규화, clamp/zero-step, solver와 history 의미는 바꾸지 않는다. teacher와 adj는 같은 entry에서 고정한다. teacher shape 변경이 원 teacher와 일치하지 않으면 그 teacher 준비 부분만 full-head 경로를 유지한다. 목적함수 및 입력을 바꿔 속도를 얻지 않는다.

## 사전 고정한 최소 실험

| 단계 | 입력과 비교 | 상한 |
| --- | --- | --- |
| CPU | 원 token 좌표, padding, overflow, cross-layer dX, exception rollback, 호출 장부 | GPU0 |
| 소형 JLZ | 고정 첫4요청, 4개 동일 R 후보, 최대8 route | 고정후보32 + 짧은 궤적96 + reference 재확인32 = 최대160 whole-batch oracle |
| 소형 native | 같은 첫4요청, 원 singleton/cached singleton/독립 batching | 최대300 request-candidate,288 Adam update; 추가 확인 F/B 각각100 request-equivalent 별도 |
| Kernel | 실제 Llama shape의 projection/head, FP64 경로 포함 | 최대24 case,각 warmup1+측정5 |
| BS100 JLZ | 첫100, 같은 W0/H0/R, REF와 최종 qualified route | 각 warmup1+측정3, 총8 whole-batch oracle |
| Observer | 같은 기술 endpoint에서 소형 R4/P8/N40 및 BS100 R100/P200/N1000 | 작은 비교 각1회; B100 각 warmup1+측정3 |

각 상한은 채워야 할 목표가 아니다. 동일 reference 결과를 identity가 맞으면 재사용하며, unused budget을 다른 단계나 전체 science chain으로 이동하지 않는다. Native B100 전체 fitting 또는16요청 확대는 이번에 하지 않는다. Native 소형 실험은 고정25 candidate/24 Adam 규칙을 그대로 끝까지 비교한다. JLZ의 짧은12-call 궤적은 기술 fixture이며 과학 cap120 축소 실험이 아니다.

짧은 각 궤적의12회는 initial·내부 reference/history 재확인·최종 예약1회를 모두 포함한 hard cap이다. 별도32회 reserve는 standalone qualification/finite-domain 비교만 허용하며, solver의12회를 늘리거나 추가 trial을 만드는 데 쓰지 않는다.

실모델 소형 입력은 고정 첫4개다. zero-step/nearzero/cancellation/서로 다른 loss/KL layer 같은 경계를 이 표본이 실제 포함하지 않으면 CPU fixture PASS와 actual NOT_COVERED를 구분한다. 유리한 요청으로 바꾸지 않는다. B100 R는 seed20261001의 공통 방향, 반경0.02이며 비활성 block은0이다. R은 timing이나 성능 결과를 보고 선택하지 않는다.

소형/kernel 단계에서 MB2/4/8 중 하나를 동결한 뒤 B1008회를 수행한다. B100에서 추가 MB sweep, 실패 후 다른 R 반복, reference 재확인 숨김을 하지 않는다. warmup pair의 고정 후보 parity가 실패하면 해당 candidate를 제외하고 이미 쓴 호출·비용을 남긴다. 남은 독립 검사와 보고는 계속하되 빠른 경로를 PASS로 바꾸지 않는다.

## 수치와 상태 판정

원 validation.json의 요청별 NLL/KL1e-4, smooth1e-3, global/block gradient(max1분모)1e-5, component1e-5, prox residual5e-6와 discrete branch 일치를 그대로 적용한다. 과거56684의1e-3 preflight를 이번 route의 동등성 증명으로 재사용하지 않는다. Qualification 실패는 해당 구현을 제외하는 근거이지 기존56684를 취소할 이유가 아니다.

Direct R의 forward W_eff와 dX를 보존하고, FP64 contraction의 rounding 차이 및 dense overflow를 검사한다. 원 dense가 nonfinite인데 direct가 finite인 fixture를 반드시 유지한다. 불확실하면 원 same-candidate 경로가 최종 판정이며 추가 oracle는 장부에 과금한다. 그 장부가 없다면 CERTIFICATION_FAILED로 중단하고 finite/NONFINITE를 추정하지 않는다.

계측 solver의 evaluate/remaining/final reserve가 같은 BudgetAccountant를 사용해야 한다. CPU cap2/3/120, line-search/history 재확인, final 실패를 점검한다. 논리적120회에 원본 확인을 무료로 더하지 않는다. 이번 benchmark는 단일 qualified route를 우선하며 동적 production fallback의 배포는 후속 범위다.

Evaluator는 원 left padding/group/position을 유지한다. near-tie는 원 microbatch 전체 full-head로 확인하고 true/new 양쪽을 원 그룹 값으로 비교한다. row 하나를 singleton으로 재평가하지 않는다. 모든 평가 질문은 최적화나 route의 과학적 성능 선택에 쓰지 않는다.

Native의 첫 teacher/anchor/loss0/prefix 및 첫 Adam은 원 singleton input shape에서 얻는다. 초기 forward는 기존25회 중 첫 회이며 반복하지 않는다. 요청별 m/v/step/stop/clamp와 마지막 backward 생략, total_loss<.05를 그대로 유지한다. 모델 weight write를 가로지르는 z 또는 prefix 재사용은 금지한다.

## 측정과 보고의 경계

동일GPU에서 reference/candidate의 모든 warmup·실패·fallback 비용을 기록한다. Synchronized wall과 CUDA events, nested component와 전체 allocation을 구분한다. median/min/max, peak allocated/reserved/RSS, valid/padded token, attention L² proxy, head rows, mapping GEMM 및 실제 F/B를 남긴다.

수치·분기·상태가 모두 통과하고 median이 더 빠르며 max(candidate)<min(reference)일 때만 TIMING_SEPARATED로 기록한다. 구간이 겹치면 SPEED_UNRESOLVED이며 반복을 마음대로 늘리지 않는다. 메모리만 줄면 MEMORY_ONLY다. 이는 최소3회 측정의 운영 기준이지 통계적 전체 chain 속도 인증이 아니다.

제공된 B1 call10→20의46.942초/oracle는 상수1200회 기준 최적화15.647시간의 근거일 뿐이다. entry/geometry/observer/IO와 이후batch 차이를 포함하지 않는다. 기존 mixed-preflight 평균 ETA는 새 보고에서 명시적으로 정정하되, 원 로그·보고를 덮어쓰지 않는다. 32.72% paddedtoken 감소나 FP64 FLOPs31.86배 차이를 실제 speedup으로 쓰지 않는다.

## 실행과 인계

Project cap2, 새 task GPU1, 원56684의 admission을 함께 계산한다. 자리가 없으면 기존 job을 건드리지 않고 정상 pending 또는 exact afterany dependency로 대기한다. 단일 GPU runner와 CPU afterany collector에 승인된 범위를 사전 등록하고, 소형 actual qualification/첫 benchmark 결과까지 관찰한 뒤 남아 있으면 pause한다. 초기경계 전에 짧은 benchmark가 끝나면 CPU 상세 검산·게시까지 끝내고 STOP할 수 있다.

Checkpoint 및 동등한 W/H/R/delta resume bundle은 저장하지 않는다. RAM 기술 commit/history/rollback만 검증 후 복구한다. 새 sequential이나 기존 production 중간이식은 별도 사용자 승인 없이는 실행하지 않는다. SH1은 사실/수치/route 판정과 source·비용을 보고하고 GH는 별도 해석한다.

효율화 task의 pause는 이 새 task에만 적용된다. 원56684의 B1→B2 초기확인 의무는 원 권한에 따라 유지하며, 효율화 pause를 원task 관찰 중단으로 확대하지 않는다.
