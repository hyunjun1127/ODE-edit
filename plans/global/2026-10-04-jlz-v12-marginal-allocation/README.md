# JLZ v12 방법 명세

첨부 v12를 수식과 구현 계약으로 구체화했다. 모든 후보 층의 subject local 목표를 공동 계획하고 하나의 상대 변화 예산을 공유한다. 이후 실제 하층 write를 반영해 상층 key와 hidden을 갱신하면서 절대 local 목표를 추적한다.

이번 산출물은 method와 CPU 수치 참조다. 모델 실험 실행, 기존 작업 변경 또는 GH 전달의 명령서가 아니다. 후속 실험의 요청 범위는 2000 edits이며, 현재 profile의 BS100×20을 method 자체의 제한으로 두지 않는다.

## 읽는 순서

1. [Method 본문](method-ko.md): 목적, 변수, optimizer, 사영, 종료, writer, claim과 한계.
2. [기계 판독 계약](contract.json): profile·수식·실패 처리·평가 정의.
3. [구현 계약](implementation-ko.md): 함수 경계, 계산 그래프, 요청별 상태, microbatch, cache와 qualification.
4. [Telemetry schema](telemetry-schema.json): 계획과 실제 write, 후보별 gradient, 종료, ACC·선호율의 기록 형식.
5. [CPU reference](reference/README-ko.md): 실행 방법, 검증 결과와 optimizer 제한의 재현.
6. [TeX](../../../docs/methods/jlz-v12-shared-budget.tex): 같은 방법의 수식 문서.

## 첨부 원안에서 구체화한 결정

| 항목 | 이번 명세 |
|---|---|
| 기본 구조 | 공동 local z, 공유 group norm 예산, 순차 target tracking 유지 |
| optimizer | EfficiencyAdam 유지, 유한 횟수 근사이며 KKT 수렴 보장 없음 |
| native 대응 | native anchor 단일변수의 loss·좌표변환·clamp로 한정, epsilon도 좌표변환 |
| gradient | model의 task backward 1회에 analytic norm gradient를 더함 |
| 종료 | 요청별 loss 기준, 최대 25평가·24갱신, 마지막 평가 후보 채택 |
| terminal 진단 | 추가 backward 없이 gradient/KKT 누락 이유를 명시 |
| KKT | task gradient의 방향·크기·비활성 조건·예산 조건을 모두 기록, 사영 임계값과 분리 |
| 예산 의미 | virtual 계획량만 제한, 실제 residual·weight·semantic damage 상한이 아님 |
| write | 실제 key/hidden 갱신, 절대 target residual, 계획과 상속 경로 차이 분리 |
| 평가 | ACC와 RS·PS·NS 동급, 편집 직후와 누적 cohort 보존 함께 보고 |

첨부 원본의 hash와 결정 근거는 [source-and-decisions.json](source-and-decisions.json)에 남겼다. 기존 성능 수치는 runtime과 원 데이터를 재결속하지 않은 상태에서 방법의 입증된 전제로 재사용하지 않았다.
