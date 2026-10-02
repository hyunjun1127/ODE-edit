# 첨부 리뷰의 수학과 근거 점검

검토 대상은 사용자 첨부 `66519567-87ee-49c1-a9f2-c0a9d8b7fd32/붙여넣은 텍스트.txt`와 후속 exact writer 제안이다. 첨부의 실행 제안은 사용자 명령으로 취급하지 않았다. 사용자는 이번 대화에서 **ridge 본선 유지와 exact pilot 비교**를 선택했다.

## 채택한 문제 제기

층별 local target은 다른 층에서 구한 residual을 옮겨 쓰는 의미상의 불일치를 줄인다. 그러나 ridge의 mean-key fitting, context별 key 차이, actual lower writes가 만드는 상태 차이는 남는다. 따라서 local target을 계산했다는 사실만으로 writer가 거의 완전히 실현한다고 해석하면 안 된다. Ridge 수식에서 실현량은 D가 아니라 DM이다.

Writer 비용을 주입 목표의 배분에 연결하는 방향은 유지한다. G와 E를 별도로 root한 두 항 대신, ridge 최적값에 대응하는 하나의 root 비용을 사용한다. Exact는 별도 constrained writer이므로 작은 paired comparison에서만 시험한다.

## 교정한 주장

| 리뷰의 주장 | 점검 결과와 v9 처리 |
|---|---|
| v7의 g/e로 L4 실현률.58–.61 등을 역산 | Coupled batch에서는 식별할 수 없다. scalar/단일 고유방향 한정이다. 해당 범위를 실측 실현률로 채택하지 않고 실제 M, DM을 기록한다. |
| δ/a를 역산한 값이 clamp와 가까우므로 실현률도 검산됨 | 같은 scalar 가정에서 나온 두 추정이 가까운 것은 그 가정을 검증하지 못한다. 직접 δ/a와 M을 대조해야 한다. |
| MEMIT-H.51–.62이므로 v8도 그 정도일 것 | 리뷰에 보고된 수치다. 현재 v8 동일 D/K/A의 실측이나 raw 재집계가 아니다. 정확한 source와 추정 방식을 확인하기 전 예측 범위로 사용하지 않는다. |
| History가 쌓이면 실현률이 감소 | Fixed K, SPD A에서 H의 PSD 증가가 X를 줄이는 방향은 맞다. 실제 순차 편집에서는 K와 D도 바뀌므로 단계별 실현률의 단조 감소는 보장되지 않는다. |
| 완벽한 실현이면 E=0 | 모든 D에 대한 M=I이면 맞다. 특정 D의 실현만 완벽하다는 것은 D(M−I)=0이며 E 행렬 전체0을 요구하지 않는다. |
| E가0이 아니면 각 요청의 실현률은 낮음 | Target residual이 존재할 수 있다는 신호다. Off-diagonal과 방향 때문에 요청별 비율은 단일 scalar shrinkage와 다르다. |
| G/E를 각각 더하면 이중 계산 | 서로 다른 비용의 L1/L2 조합이다. 중복 계산이라기보다 다른 regularizer이며, 통합 비용의 capacity 해석이 더 명확하다. |
| E/pulse는 δ를 줄이기만/키우기만 함 | Fixed metric의 전체 radial 성분과 total gradient를 혼동한다. Request cross-term, κ의 하층 의존성, alignment 방향에 따라 회전·증대·감소가 가능하다. |
| Canonical absolute tracking이면 부족이 닫힘 | Mean-key writer와 canonical key가 다르다. Exact mean fit이어도 U(k_can−κ) 오차가 남는다. Tracking 자체도 다른 층에 책임을 이전할 수 있다. |
| Exact mean fit이면 writer가 실행만 함 | Mean key의 국소 linear 증분에 한정한다. Context·상층 기저·subject 외 token·최종 분포 차이는 남는다. |
| Writer metric으로 native decay를 대체하면 새 계수가 없으므로 native 보존 | 계수 재사용과 목적함수 보존은 다르다. Native norm과 full coupled geometry를 유지한다. |
| 작은 anchor 요청이 G/E에서 더 비쌈 | 무엇을 고정하느냐에 따라 다르다. 동일 상대 변위, identity metric이면 오히려 큰 anchor가 batch RMS 비용에 더 기여한다. |
| v7 포화가 미실현 부족을 가려 PS를 높이고 NS를 낮춤 | 가능한 인과 가설이다. Frozen-D writer 비교와 이후 최적화 trace로 시험하며 확인된 원인으로 적지 않는다. |

## g와 e만으로 실현률을 알 수 없는 반례

D=I_2, M=R_θ diag(.2,.8) R_θᵀ로 둔다. θ=π/12와 π/4에서 모두 unnormalized g²=tr(M−M²)=.32, e²=tr((I−M)²)=.68이다. 그러나 diag(M)은 각각 약(.2402,.7598)과(.5,.5)다. 두 경우 모두 요청 간 혼합이 존재한다. 동일 g/e에서 다른 자기 요청 실현 계수가 가능하므로 scalar 역산은 유효하지 않다. [CPU 검증](math/README.md)에 이 반례를 포함한다.

## 선행 연구가 뒷받침하는 범위

[BLUE](https://arxiv.org/html/2502.03748v3)는 target 구성과 편집 층의 정렬 문제를 다룬다. 그 결과가 ridge fitting이나 모든 token의 출력 일치를 보장하지는 않는다.

[FE](https://arxiv.org/html/2605.00358v2)의 backward-spreading cosine과 forward-replay 잔차는 해당 설정의 수송 실험이다. 그 cosine을 이미 joint local target을 만드는 v8/v9의 실현률로 대입하지 않는다.

[DOW-KE](https://arxiv.org/html/2608.16932v1)의 near.5는 특정 모델·데이터·방법에서 목표 방향으로 실현된 local displacement의 투영 비율이다. 성공률·cosine 또는 JLZ의 실측값이 아니다. V9은 같은 개념의 직접 투영값과 orthogonal 성분을 분리해 기록한다.

[EMMET §5](https://arxiv.org/html/2403.14236v5#S5)는 equality-constrained batch writer의 선행이다. 이상식의 invertibility 조건과 실제 구현의 regularization을 구분해야 한다. 논문의 X+.1I 안정화는 일반적으로 exact equality를 완화한다. Equality가 locality를 자동 개선한다는 근거로 사용하지 않는다.

첨부의 v7 A W5 R100/P99.2/N71.96은 **리뷰가 보고한 수치**다. 이번 작업은 해당 500개 raw의 독립 재집계가 아니다. 실험 보고서에 넣을 때 동일 cohort·metric·분모와 W0를 결속한다. 문헌과 math toy는 실제 JLZ 성능 검증을 대신하지 않는다.
