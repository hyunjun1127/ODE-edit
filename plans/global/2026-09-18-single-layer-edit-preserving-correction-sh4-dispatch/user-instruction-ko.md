# 사용자 지시 원문 및 후속 정정

## 최초 지시

지시문 자세히 확인하고 SH4 에게 task 전달하라. gpu cap은 2로 진행하자

다음 설계문을 기준으로 single-layer edit-preserving correction 실험의 구현·검증·조건부 실행을 진행하라.

/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-18-single-layer-edit-preserving-correction-design-v1.md

연결된 contract JSON, cells CSV, 방법 포지셔닝 및 BLUE/L4 audit도 함께 확인하라. 아래 M 단계 재사용 지침은 설계문의 일괄 재실행 지침보다 우선한다.

1. 연구 목표와 방법을 유지하라.

핵심 질문은 “동일한 L4 native 편집이 실현한 반응을 유지하면서, 같은 layer 안에서 locality 손상을 줄일 수 있는가?”이다.

L4 down-projection 하나만 수정할 때 고정 입력 sequence의 key가 변하지 않는 성질을 활용한다. 현재 편집 sequence의 전체 token에 대해 D K_E=0을 만족시키고, 그 공간에서 W0 기준 preservation loss를 낮추는 EN-F를 주 방법으로 유지하라.

추가 layer, 추가 z 최적화, 새 paraphrase 학습·guard set을 도입하지 말라. 이번 실행에 새로운 method나 ablation을 임의로 추가하지 말라.

2. M 단계는 기존 결과를 먼저 조사하고, 중복 실험을 건너뛰어라.

기존 보고서·raw metrics·실행 로그·checkpoint·native target 및 update artifact를 조사하여, M의 batch×arm×진단별 재사용 판정표를 먼저 작성하라.

각 항목을 다음과 같이 분류하라.

- REUSE: 비교 조건과 필요한 증거가 충족됨. 해당 실험은 재실행하지 않는다.
- EVAL_ONLY: 기존 endpoint는 유효하지만 평가·진단 일부가 부족함. 저장된 artifact에서 부족한 항목만 계산한다.
- RUN_MISSING: 해당 결과가 없거나 현재 비교를 충족하지 못함. 부족한 실행 단위만 수행한다.
- REFERENCE_ONLY: 관련 결과이지만 M의 직접 비교를 대체할 수 없음. 참고 근거로 분리한다.

재사용 여부는 성능이 아니라 조건과 증거의 적합성으로 판단하라. 불리한 결과, 실패, native fallback도 동일한 기준으로 재사용하며 좋은 결과만 골라 쓰지 말라.

판정 시 다음을 확인하라.

- 모델·tokenizer·편집 module·native hparams·수치 설정
- 요청 ID, batch 구성, context/tokenization, W0 시작과 초기 history
- 비교 arm이 공유해야 하는 native z·Δ_N·W_N·A·K·P·M
- preservation 데이터·teacher·loss 정의
- correction 공간, optimizer 예산, guard 및 candidate selection 규칙
- 평가 정의, selection과 observer의 분리, raw 결과의 추적 가능성

실험 이름이나 RS·PS·NS 평균이 같다는 이유만으로 동일 실험으로 취급하지 말라. 반대로 저장 경로나 실행 시각이 다르다는 이유만으로 유효한 결과를 재실행하지 말라.

M은 매번 W0에서 시작하는 독립 cold batch다. Sequential 실험의 B2 이후 결과를 독립 cold batch 결과로 대체하지 말라.

기존 native endpoint를 재사용할 수 있다면 해당 native fit을 다시 수행하지 말고, 같은 endpoint에서 누락된 arm만 실행하라. 기존 산출물로 평가·진단을 보완할 수 있으면 z fitting이나 correction optimization을 반복하지 말라.

M 전체가 충족되면 M 신규 실험은 전부 건너뛰고 기존 결과로 확대 gate를 판단하라. 일부만 충족되면 부족한 부분만 실행하라. 재사용할 수 없는 항목은 구체적인 불일치와 재실행 필요성을 기록하라.

3. 기술 검증과 과학 실험을 구분하라.

새 EN-F 구현에 필요한 실제 모델의 key stationarity, projector, gradient, FP32 materialization, protected response 검증은 설계문대로 수행하라.

기존 M 결과가 있다는 이유로 검증되지 않은 새 구현을 통과시키지 말라. 동시에 기술 검증을 명분으로 M 전체를 반복하지 말라. CPU toy PASS는 실제 모델 검증의 대체 증거가 아니다.

4. 기존 결과와 신규 결과를 합쳐 단계별 gate를 적용하라.

T 검증과 M 증거가 충족되면 S→R→L 순서로 진행하되, 각 단계의 확대·중단 기준을 그대로 적용하라. 재사용된 M도 동일한 gate를 적용한다.

모든 신규 독립 batch와 sequential chain은 W0에서 시작한다. Warm 5k checkpoint는 사용하지 않는다.

S 이후에는 각 arm이 자기 직전 상태에서 native z와 update를 계산한다. 서로 다른 arm의 이후 batch target이나 update를 공유하거나, M의 좋은 endpoint를 연결하여 가상의 lifelong 결과를 만들지 말라.

Official P/N은 online 최적화·후보 선택에 사용하지 말고, Report256 공개 시점도 설계문을 유지하라. 결과를 보고 threshold·rank cutoff·PS 허용폭을 바꾸지 말라.

5. 결과 해석은 다음을 구분하라.

- Correction 공간이 존재하는가?
- 그 공간에 preservation loss의 감소 방향이 있는가?
- 실제 보호 반응이 유지되는가?
- 독립 locality와 W0-correct neighborhood retention이 개선되는가?
- Paraphrase와 과거 편집 성능이 관측상 유지되는가?
- 계산 비용을 포함해 장기 적용이 유효한가?

EN-F 대 CA는 고정 writer 내부의 보정 한계를, EN-F 대 KL-P는 response-preserving 제약의 실용적 효과를 평가하는 비교로 해석하라.

현재 sequence의 반응 보존을 전체 paraphrase·과거 편집의 보장으로 확대하지 말라. Calibration KL 감소만으로 locality 개선을 선언하지 말라.

6. 산출물을 명확히 남겨라.

- 기존 결과의 경로·조건·재사용 범위·판정 근거를 담은 M 재사용 판정표
- 건너뛴 실험, 평가만 보완한 항목, 신규 실행 항목을 구분한 실행표
- 실제 모델 기술 검증 결과와 구현·데이터·checkpoint provenance
- 재사용·신규 결과를 중복 집계하지 않은 통합 분석
- 단계별 gate 판정과 진행·중단 근거
- 최종적으로 지지되는 claim, 지지되지 않는 claim, 계산 비용

기존 산출물을 덮어쓰지 말고 신규 결과와 연결하여 추적 가능하게 보존하라. 현재 실행 중인 다른 작업과 충돌하지 않도록 자원과 작업 상태를 확인하라.

우선 기존 결과 감사와 누락 범위 확정을 수행한 뒤, 필요한 구현·기술 검증·누락 실험으로 이어가라. 재사용 가능한 M 실험을 관성적으로 다시 실행하지 말라.

## 후속 지시 — 아래 최신 지시로 대체된 운영 범위

> 설계의 단계별 확대·중단 gate에 따라 최종 보고까지 계속 진행하도록 명령해

## 최신 지시 — 현재 유효한 실행·모니터링 범위

> 우선 M만 제출하고 본실험 초기 gate 확인 후 모니터링 중지 하는 것으로 수정하자

따라서 이번 제출 권한은 필요한 기술 검증과 M 누락 실행/평가뿐이다. 앞선 자동 S/R/L 확대 및 최종 보고까지 지속은 대체되었다. 원 설계의 미래 단계 사양/과학 gate는 변경하지 않으며, 후속 제출은 새 사용자 호출과 범위 지시를 기다린다.
